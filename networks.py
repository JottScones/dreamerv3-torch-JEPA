import math
import numpy as np
import re
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

import tools
from ijepa.src.models.vision_transformer import VisionTransformer


class RSSM(nn.Module):
    """
    RSSM holds the logic regarding updating the model state: the recurrent state and the stochastic latent variable.

    It does this via two main paths:
        1. Observation path: where we use the observation network to update model state given the previous state, action and sensory input. 
        2. Imagination path: where we use the imagination network to predict the next model state of the model given the previous state and action.
    
    The main difference being where the stochastic latent variable comes from, either from input or from the new recurrent state. 
    """
    def __init__(
        self,
        stoch=30, # Number of stochastic latent variables
        deter=200, # The recurrent state
        hidden=200, # Number of hidden units in the GRU cell
        rec_depth=1,
        discrete=False, # If stochastic distribution is categorical or normal
        act="SiLU",
        norm=True,
        mean_act="none",
        std_act="softplus",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,
        embed=None,
        device=None,
    ):
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._rec_depth = rec_depth
        self._discrete = discrete
        act = getattr(torch.nn, act)
        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._num_actions = num_actions
        self._embed = embed
        self._device = device

        # If the stochastic representation is discrete, we one-hot encode.
        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            inp_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        inp_layers.append(act())

        # ----- Imagination network -----
        # Layers to process the input (stoch + action) and produce the input for the GRU cell.
        self._img_in_layers = nn.Sequential(*inp_layers)
        self._img_in_layers.apply(tools.weight_init)

        # Sequence model to update the recurrent state.
        self._cell = GRUCell(self._hidden, self._deter, norm=norm)
        self._cell.apply(tools.weight_init)

        # Layer to process the recurrent state and to produce input to imgs_stat_layer
        # to predict the next distribution of latent variable stoch.
        img_out_layers = []
        inp_dim = self._deter
        img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            img_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        img_out_layers.append(act())
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(tools.weight_init)

        # ------ Observation network -----
        obs_out_layers = []
        inp_dim = self._deter + self._embed
        obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            obs_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        obs_out_layers.append(act())
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)

        # Layers to produce the sufficient statistics of the distribution of the latent variable.
        if self._discrete:
            # If the stochastic representation is discrete, we produce logits for each class of the categorical distribution.
            # So output is length of stoch * number of classes.

            self._imgs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))

            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))

        else:
            # If the stochastic representation is continuous, we produce mean and std for each class of the normal distribution.
            # So output is two vectors of length stoch one for mean one for std.

            self._imgs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))

            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))

        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

    def initial(self, batch_size):
        """
        If the embedding distribution is discrete, we represent it as unnormalised logits, parametrising a categorical distribution.
        Stoch is a one-hot encoded vector sample from this categorical distribution: z_t.

        If the embedding distribution is continuous, we represent it as a normal distribution with mean and std.
        Stoch is a sample from this normal distribution: z_t.

        Deter is the recurrent state of the GRU cell: h_t.
        """
        deter = torch.zeros(batch_size, self._deter, device=self._device)
        if self._discrete:
            state = dict(
                logit=torch.zeros(
                    [batch_size, self._stoch, self._discrete], device=self._device
                ),
                stoch=torch.zeros(
                    [batch_size, self._stoch, self._discrete], device=self._device
                ),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch], device=self._device),
                std=torch.zeros([batch_size, self._stoch], device=self._device),
                stoch=torch.zeros([batch_size, self._stoch], device=self._device),
                deter=deter,
            )

        if self._initial == "zeros":
            return state

        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        """
        We observe the environment over multiple time steps, iterate over time progressing the recurrent state
        and the stochastic latent variable.

        Even though we are calling obs_step which calls img_step, we are not using the imagination network to produce
        stochastic latent variable. We simply use it to update the recurrent state and in obs_step we use the 
        sensory input to produce stoch.  
        """

        # Swap first two dimensions
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ), # fn
            (action, embed, is_first), # inputs
            (state, state), # start (initial prev_state[0] = None)
        )

        # (batch, time, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine_with_action(self, action, state):
        # Swap first two dimensions
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        assert isinstance(state, dict), state
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        # produce the model state by concatenating the stochastic and recurrent state.
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state, dtype=None):
        # If discrete representation we use a categorical distribution, otherwise a normal distribution.
        if self._discrete:
            logit = state["logit"]
            dist = torchd.independent.Independent(
                tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )

        else:
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
            )
        return dist

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        """
        Observation step of the RSSM, to calculate the new recurrent state and stochastic latent variable from observation (embed) rather than imagination.
        """

        # initialize all prev_state
        # If there is no previous state or every element in is_first is True, we initialize the entire previous state.
        if prev_state == None or torch.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first))
            prev_action = torch.zeros(
                (len(is_first), self._num_actions), device=self._device
            )

        # Otherwise only overwrite the prev_state only where is_first=True
        elif torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first # Set the action to zero where is_first=True
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )

        # Calculate the recurrent state h_t and use that to predict z_t (prior).
        prior = self.img_step(prev_state, prev_action)

        # Concatenate the recurrent state with the embedding of the visual and vector inputs (sensory input x_t).
        x = torch.cat([prior["deter"], embed], -1)

        # Pass the concatenated input through a linear layer to produce the input for the sufficient statistics layer.
        # (batch_size, prior_deter + embed) -> (batch_size, hidden)
        x = self._obs_out_layers(x)

        # Map sensory input to the sufficient statistics of the distribution of the latent variable.
        # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("obs", x)

        # Use the statistics to sample from the distribution of the latent variable or just use the mode.
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        """
            Imagination step of the RSSM, to calculate the new recurrent state and stochastic latent variable.
            We concatenate the previous latent state and action and pass it through a linear layer to 
            produce input of size _hidden for the GRU. 

            The GRU cell updates the recurrent state: h_t.

            The img_out and img_stat layers produce the new distribution of the latent variable: z_t.
        """ 
        # (batch, stoch, discrete_num)
        # Stoch is the sample from the distribution of the previous latent variable.
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)
        # (batch, stoch * discrete_num) -> (batch, stoch * discrete_num + action)
        x = torch.cat([prev_stoch, prev_action], -1)

        # Pass the concatenated input through a linear layer to produce the input for the GRU cell.
        # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
        x = self._img_in_layers(x)

        # Perform (one) step of the GRU cell to update the recurrent state.
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            deter = prev_state["deter"]
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            x, deter = self._cell(x, [deter])
            # Weird but both x and deter are the same value after the forward pass.
            deter = deter[0]  # Keras wraps the state in a list.

        # Pass recurrent state through a linear layer to produce the output of the GRU cell.
        # (batch, deter) -> (batch, hidden)
        x = self._img_out_layers(x)

        # Calculate the statistics (mean and std) that describe the new distribution of the latent variable.
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("ims", x)

        # Use the statistics to sample from the distribution of the latent variable or just use the mode.
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def get_stoch(self, deter):
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()

    def _suff_stats_layer(self, name, x):
        """
        Produces the sufficient statistic for the distribution of the latent variable. 
        If the distribution is discrete, then it is simply described by the logits.

        If the distribution is continuous, then it is described by the mean and std.
        """

        if self._discrete:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        # Stop gradient 
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        # Calculate the KL divergence between the posterior (encoder) and prior distributions (dynamics predictor).
        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        # this is implemented using maximum at the original repo as the gradients are not backpropagated for the out of limits.
        # we clip the loss if they are minimised well enough already - to avoid degenerate solutions.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


class MultiEncoder(nn.Module):
    """
    MultiEncoder is a module that combines CNN and MLP encoders for processing image and vector data, respectively.
    """
    def __init__(
        self,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        symlog_inputs,
    ):
        super(MultiEncoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")

        # Shapes are expected to be a dictionary with keys as the names of the inputs/observations and values as their shapes.
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }

        # In the config file we define regex patterns for the keys of the CNN and MLP inputs.
        # Here we split the shapes into CNN and MLP shapes based on these patterns.
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        if self.cnn_shapes:
            # Expect all images to have same height and width and concatenate along the channel axis.
            # The height and width of the first image in the shapes dictionary.
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            # Create CNN encoder.
            self._cnn = ConvEncoder(
                input_shape, cnn_depth, act, norm, kernel_size, minres
            )
            self.outdim += self._cnn.outdim

        if self.mlp_shapes:
            # We sum over the number of elements in each input since we will flatten and concatenate them.
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            # Create MLP encoder.
            self._mlp = MLP(
                input_size,
                None,
                mlp_layers,
                mlp_units,
                act,
                norm,
                symlog_inputs=symlog_inputs,
                name="Encoder",
            )
            self.outdim += mlp_units

    def forward(self, obs):
        # torch.cat(dim=-1) concatenates the tensors along the last dimension - which is the channel for images 
        # and a single vector for MLP.
        outputs = []
        if self.cnn_shapes:
            inputs = torch.cat([obs[k] for k in self.cnn_shapes], -1)
            outputs.append(self._cnn(inputs))

        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))
        
        # Both encoders produce vectors so we can concatenate them.
        outputs = torch.cat(outputs, -1)
        return outputs


class MultiDecoder(nn.Module):
    """
    MultiDecoder, complements the MultiEncoder, is a module that combines CNN and MLP decoders for processing image and vector data, respectively.
    """
    def __init__(
        self,
        feat_size,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        cnn_sigmoid,
        image_dist,
        vector_dist,
        outscale,
    ):
        super(MultiDecoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal")

        # Shapes are expected to be a dictionary with keys as the names of the inputs/observations and values as their shapes.
        shapes = {k: v for k, v in shapes.items() if k not in excluded}

        # Separate the shapes into CNN and MLP shapes based on the regex patterns defined in the config file.
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)

        if self.cnn_shapes:
            # Takes height and width of the first image in the shapes dictionary and sums over all input channels.
            # Shape is (channels, height, width) which is expected by the ConvDecoder.
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = ConvDecoder(
                feat_size,
                shape,
                cnn_depth,
                act,
                norm,
                kernel_size,
                minres,
                outscale=outscale,
                cnn_sigmoid=cnn_sigmoid,
            )

        if self.mlp_shapes:
            self._mlp = MLP(
                feat_size,
                self.mlp_shapes,
                mlp_layers,
                mlp_units,
                act,
                norm,
                vector_dist,
                outscale=outscale,
                name="Decoder",
            )
        self._image_dist = image_dist

    def forward(self, features):
        dists = {}
        if self.cnn_shapes:
            # Split channels into separate tensors for each input.
            feat = features
            outputs = self._cnn(feat)
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update(
                {
                    key: self._make_image_dist(output)
                    for key, output in zip(self.cnn_shapes.keys(), outputs)
                }
            )

        if self.mlp_shapes:
            dists.update(self._mlp(features))

        return dists

    def _make_image_dist(self, mean):
        # We interpret output of cnn decoder as a mean of a distribution either a Normal or MSE (deterministic output).
        if self._image_dist == "normal":
            return tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, 1), 3)
            )
        if self._image_dist == "mse":
            return tools.MSEDist(mean)
        raise NotImplementedError(self._image_dist)


class ConvEncoder(nn.Module):
    """
    CNN encoder for processing image data.

    TODO: UPDATE ENCODER HERE
    """
    def __init__(
        self,
        input_shape,
        depth=32,
        act="SiLU",
        norm=True,
        kernel_size=4,
        minres=4, # Miniumum spatial resolution (height and width) of the input image.
    ):
        super(ConvEncoder, self).__init__()
        act = getattr(torch.nn, act)
        h, w, input_ch = input_shape

        # Calculate how many stages of convolutions we need to apply to the input image to reach the minimum resolution.
        # we effectively halve the height and width of the image at each stage.
        stages = int(np.log2(h) - np.log2(minres))
        in_dim = input_ch

        # Number of output channels for each convolution layer, doubles at each stage.
        out_dim = depth

        layers = []
        for i in range(stages):
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            layers.append(act())
            in_dim = out_dim
            out_dim *= 2
            h, w = h // 2, w // 2

        self.outdim = out_dim // 2 * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        obs -= 0.5
        # Combine the batch and time dimensions into a single dimension so we have a batch of images.
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))

        # Reorder the dimensions to match the expected input shape of the Conv2d layer.
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)

        # (batch * time, ...) -> (batch * time, -1)
        # Flatten the output of the last convolution layer to be a vecor of size (batch * time, -1) where -1 is the product of all remaining dimensions.
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])

        # Separate the batch and time dimensions again.
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])
    
class JEPAEncoder(VisionTransformer):
    def __init__(self, vit_size="small", img_size=224, patch_size=14, checkpoint_path=None, **kwargs):
        base_params = {
            "norm_layer": partial(nn.LayerNorm, eps=1e-6),
            "patch_size": patch_size,
            "qkv_bias": True
        }
        custom_params= {
            "tiny": { "embed_dim": 192, "depth": 12, "num_heads": 3, "mlp_ratio": 4 },
            "small": { "embed_dim": 384, "depth": 12, "num_heads": 6, "mlp_ratio": 4 },
            "base": { "embed_dim": 768, "depth": 12, "num_heads": 12, "mlp_ratio": 4 },
            "large": { "embed_dim": 1024, "depth": 24, "num_heads": 16, "mlp_ratio": 4 },
            "huge": { "embed_dim": 1280, "depth": 32, "num_heads": 16, "mlp_ratio": 4 },
            "giant": { "embed_dim": 1408, "depth": 40, "num_heads": 16, "mlp_ratio": 48/11 },
        }
        super().__init__(img_size=[img_size], **base_params, **custom_params[vit_size], **kwargs)


        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

            epoch = checkpoint['epoch']
            
            pretrained_dict = checkpoint['encoder']

            # Remove "module." prefix if it exists which occurs when trained with DataParallel
            new_pretrained_dict = {}
            for k, v in pretrained_dict.items():
                new_key = k.replace("module.", "")
                new_pretrained_dict[new_key] = v

            msg = self.load_state_dict(pretrained_dict)
            print(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')
        else:
            print(f'NOT loaded pretrained encoder {checkpoint_path}')

        self.vit_size = vit_size
        self.img_size = img_size
        self.outdim = self.patch_embed.num_patches * self.embed_dim

    def forward(self, obs, masks=None):
        obs -= 0.5
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))

        # Reorder the dimensions to match the expected input shape of the Conv2d layer.
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)

        # Resize to 224x224
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        x = super().forward(x, masks)

        # combine last two dimensions
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])

        # Separate the batch and time dimensions again.
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])

class JEPAMultiEncoder(nn.Module):
    """
    MultiEncoder is a module that combines JEPA and MLP encoders for processing image and vector data, respectively.
    """
    def __init__(
        self,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        vit_size,
        mlp_layers,
        mlp_units,
        symlog_inputs,
        checkpoint_path,
        **kwargs
    ):
        image_keys = cnn_keys
        super(JEPAMultiEncoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")

        # Shapes are expected to be a dictionary with keys as the names of the inputs/observations and values as their shapes.
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }

        # In the config file we define regex patterns for the keys of the JEPA and MLP inputs.
        # Here we split the shapes into JEPA and MLP shapes based on these patterns.
        self.image_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(image_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Encoder JEPA shapes:", self.image_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        if self.image_shapes:
            # Create JEPA encoder.
            self._jepa = JEPAEncoder(vit_size=vit_size, checkpoint_path=checkpoint_path)
            self.outdim += self._jepa.outdim

            # Freeze weights
            self._jepa.requires_grad_(False)

        if self.mlp_shapes:
            # We sum over the number of elements in each input since we will flatten and concatenate them.
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            # Create MLP encoder.
            self._mlp = MLP(
                input_size,
                None,
                mlp_layers,
                mlp_units,
                act,
                norm,
                symlog_inputs=symlog_inputs,
                name="Encoder",
            )
            self.outdim += mlp_units

    def forward(self, obs):
        # torch.cat(dim=-1) concatenates the tensors along the last dimension - which is the channel for images 
        # and a single vector for MLP.
        outputs = []
        if self.image_shapes:
            inputs = torch.cat([obs[k] for k in self.image_shapes], -1)
            outputs.append(self._jepa(inputs))

        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))
        
        # Both encoders produce vectors so we can concatenate them.
        outputs = torch.cat(outputs, -1)
        return outputs

class ConvDecoder(nn.Module):
    """
    CNN decoder for processing image data.
    
    TODO: UPDATE DECODER HERE
    """
    def __init__(
        self,
        feat_size, # Dimension of the embedding vector from the encoder.
        shape=(3, 64, 64),
        depth=32,
        act=nn.ELU,
        norm=True,
        kernel_size=4,
        minres=4,
        outscale=1.0,
        cnn_sigmoid=False,
    ):
        super(ConvDecoder, self).__init__()
        act = getattr(torch.nn, act)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid

        # Number of layers to reach original resolution.
        layer_num = int(np.log2(shape[1]) - np.log2(minres))
        self._minres = minres

        # Number of input channels for the first layer.
        out_ch = minres**2 * depth * 2 ** (layer_num - 1)
        self._embed_size = out_ch

        # Linear layer helps to reshape the input vector to the expected shape of the first convolution layer.
        self._linear_layer = nn.Linear(feat_size, out_ch)
        self._linear_layer.apply(tools.uniform_weight_init(outscale))

        # Number of output channels halves at each layer.
        in_dim = out_ch // (minres**2)
        out_dim = in_dim // 2

        layers = []
        h, w = minres, minres
        for i in range(layer_num):
            bias = False

            if i == layer_num - 1:
                # We want raw output for the last layer no normalisatin or activation.
                out_dim = self._shape[0]
                act = False
                bias = True
                norm = False

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            if act:
                layers.append(act())
            in_dim = out_dim
            out_dim //= 2
            h, w = h * 2, w * 2
        [m.apply(tools.weight_init) for m in layers[:-1]]
        layers[-1].apply(tools.uniform_weight_init(outscale))
        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k, s, d):
        # Calculate the padding needed wrt the kernel size, stride and dilation.
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features, dtype=None):
        x = self._linear_layer(features)
        # (batch, time, -1) -> (batch * time, h, w, ch)
        x = x.reshape(
            [-1, self._minres, self._minres, self._embed_size // self._minres**2]
        )
        # (batch, time, -1) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch, time, -1) -> (batch, time, ch, h, w)
        mean = x.reshape(features.shape[:-1] + self._shape)
        # (batch, time, ch, h, w) -> (batch, time, h, w, ch)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean)
        else:
            mean += 0.5
        return mean


class MLP(nn.Module):
    """
    MLP definition used everywhere in the code for encoding, decoding, actor and critic networks.
    """
    def __init__(
        self,
        inp_dim, # Input dimension of the MLP
        shape, # Shape of the output tensor, None for encoder output (not a distribution)
        layers, # Number of hidden layers
        units, # Number of units in each hidden layer
        act="SiLU",
        norm=True,
        dist="normal", # String that specifies the type of distribution to use for the output
        std=1.0,
        min_std=0.1,
        max_std=1.0,
        absmax=None,
        temp=0.1,
        unimix_ratio=0.01,
        outscale=1.0,
        symlog_inputs=False,
        device="cuda",
        name="NoName",
    ):
        super(MLP, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        act = getattr(torch.nn, act)

        # Parameters for the distribution
        self._dist = dist
        self._std = std if isinstance(std, str) else torch.tensor((std,), device=device)
        self._min_std = min_std
        self._max_std = max_std
        self._absmax = absmax
        self._temp = temp # Used for Gumbel softmax (Normal softmax with nice gradients)
        self._unimix_ratio = unimix_ratio # Used for onehot distributions

        self._symlog_inputs = symlog_inputs
        self._device = device

        self.layers = nn.Sequential()
        # Initialise the first and hidden layers of the MLP
        for i in range(layers):
            self.layers.add_module(
                f"{name}_linear{i}", nn.Linear(inp_dim, units, bias=False)
            )
            if norm:
                self.layers.add_module(
                    f"{name}_norm{i}", nn.LayerNorm(units, eps=1e-03)
                )
            self.layers.add_module(f"{name}_act{i}", act())
            if i == 0:
                inp_dim = units
        self.layers.apply(tools.weight_init)


        if isinstance(self._shape, dict):
            # If shape is a dictionary, we create a separate linear layer for each key in the dictionary.
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))

            # If std is learned, we create a separate linear layer for each key in the dictionary. 
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))

        elif self._shape is not None:
            # If shape is not a dictionary, we create a single linear layer for the mean.
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))

            # If std is learned, we create a single linear layer for the std. 
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        if self._symlog_inputs:
            # Symlog acts as normalisation for the inputs
            x = tools.symlog(x)

        out = self.layers(x)
        # Used for encoder output which will be a vector of size "units"
        if self._shape is None:
            return out

        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)

                # Either std is learned or fixed
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)

            # Either std is learned or fixed
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        # Create a distribution based on mean and std and distribution parameters.

        if dist == "tanh_normal":
            mean = torch.tanh(mean)
            std = F.softplus(std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif dist == "normal":
            std = (self._max_std - self._min_std) * torch.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif dist == "normal_std_fixed":
            dist = torchd.normal.Normal(mean, self._std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif dist == "trunc_normal":
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif dist == "onehot":
            dist = tools.OneHotDist(mean, unimix_ratio=self._unimix_ratio)
        elif dist == "onehot_gumble":
            dist = tools.ContDist(
                torchd.gumbel.Gumbel(mean, 1 / self._temp), absmax=self._absmax
            )
        elif dist == "huber":
            dist = tools.ContDist(
                torchd.independent.Independent(
                    tools.UnnormalizedHuber(mean, std, 1.0),
                    len(shape),
                    absmax=self._absmax,
                )
            )
        elif dist == "binary":
            dist = tools.Bernoulli(
                torchd.independent.Independent(
                    torchd.bernoulli.Bernoulli(logits=mean), len(shape)
                )
            )
        elif dist == "symlog_disc":
            dist = tools.DiscDist(logits=mean, device=self._device)
        elif dist == "symlog_mse":
            dist = tools.SymlogDist(mean)
        else:
            raise NotImplementedError(dist)
        return dist


class GRUCell(nn.Module):
    """
    Gated Recurrent Unit (GRU) cell is the sequence model that takes in the previous recurrent state, the stochastic representation and an action
    and returns the next recurrent state.

    This is used alongside the dynamic model to predict the next state of the environment.
    """
    def __init__(self, inp_size, size, norm=True, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size # Size of the input to the GRU cell.
        self._size = size # Size of the recurrent state.
        self._act = act
        self._update_bias = update_bias
        self.layers = nn.Sequential()
        self.layers.add_module(
            "GRU_linear", nn.Linear(inp_size + size, 3 * size, bias=False)
        )
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * size, eps=1e-03))

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        # The state is the concatenation of the previous recurrent state and the stochastic representation.
        state = state[0]  # Keras wraps the state in a list get first and only element.
        parts = self.layers(torch.cat([inputs, state], -1))

        # GRU produces reset gate, candidate state and update gates.
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)

        # Gated update using previous state and current state.
        output = update * cand + (1 - update) * state
        return output, [output]


class Conv2dSamePad(torch.nn.Conv2d):
    """
    Specialized Conv2d layer that calculates the padding needed to keep the input and output size the same.
    """
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class ImgChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ImgChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

"""
Logdir logdir/JEPA/debug
Create envs.
/home/swj24/.conda/envs/dreamer_jepa/lib/python3.9/site-packages/gym/envs/atari/environment.py:68: UserWarning: WARN: obs_type "image" should be replaced with the image type, one of: rgb, grayscale
  logger.warn(
A.L.E: Arcade Learning Environment (version 0.7.5+db37282)
[Powered by Stella]
Action Space Box(0.0, 1.0, (6,), float32)
Prefill dataset (57 steps).
Logger: (10000 steps).
Simulate agent.
Encoder JEPA shapes: {'image': (64, 64, 3)}
Encoder MLP shapes: {}
/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/networks.py:667: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
Traceback (most recent call last):
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/dreamer.py", line 458, in <module>
    main(parser.parse_args(remaining))
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/dreamer.py", line 367, in main
    agent = Dreamer(
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/dreamer.py", line 46, in __init__
    self._wm = models.WorldModel(obs_space, act_space, self._step, config)
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/models.py", line 59, in __init__
    self.encoder = networks.JEPAMultiEncoder(shapes, **config.encoder) if config.use_jepa else networks.MultiEncoder(shapes, **config.encoder)
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/networks.py", line 746, in __init__
    self._jepa = JEPAEncoder(vit_size=vit_size, checkpoint_path=checkpoint_path)
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/networks.py", line 672, in __init__
    msg = self.load_state_dict(pretrained_dict)
  File "/home/swj24/.conda/envs/dreamer_jepa/lib/python3.9/site-packages/torch/nn/modules/module.py", line 2215, in load_state_dict
    raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
RuntimeError: Error(s) in loading state_dict for JEPAEncoder:
        Missing key(s) in state_dict: "pos_embed", "patch_embed.proj.weight", "patch_embed.proj.bias", "blocks.0.norm1.weight", "blocks.0.norm1.bias", "blocks.0.attn.qkv.weight", "blocks.0.attn.qkv.bias", "blocks.0.attn.proj.weight", "blocks.0.attn.proj.bias", "blocks.0.norm2.weight", "blocks.0.norm2.bias", "blocks.0.mlp.fc1.weight", "blocks.0.mlp.fc1.bias", "blocks.0.mlp.fc2.weight", "blocks.0.mlp.fc2.bias", "blocks.1.norm1.weight", "blocks.1.norm1.bias", "blocks.1.attn.qkv.weight", "blocks.1.attn.qkv.bias", "blocks.1.attn.proj.weight", "blocks.1.attn.proj.bias", "blocks.1.norm2.weight", "blocks.1.norm2.bias", "blocks.1.mlp.fc1.weight", "blocks.1.mlp.fc1.bias", "blocks.1.mlp.fc2.weight", "blocks.1.mlp.fc2.bias", "blocks.2.norm1.weight", "blocks.2.norm1.bias", "blocks.2.attn.qkv.weight", "blocks.2.attn.qkv.bias", "blocks.2.attn.proj.weight", "blocks.2.attn.proj.bias", "blocks.2.norm2.weight", "blocks.2.norm2.bias", "blocks.2.mlp.fc1.weight", "blocks.2.mlp.fc1.bias", "blocks.2.mlp.fc2.weight", "blocks.2.mlp.fc2.bias", "blocks.3.norm1.weight", "blocks.3.norm1.bias", "blocks.3.attn.qkv.weight", "blocks.3.attn.qkv.bias", "blocks.3.attn.proj.weight", "blocks.3.attn.proj.bias", "blocks.3.norm2.weight", "blocks.3.norm2.bias", "blocks.3.mlp.fc1.weight", "blocks.3.mlp.fc1.bias", "blocks.3.mlp.fc2.weight", "blocks.3.mlp.fc2.bias", "blocks.4.norm1.weight", "blocks.4.norm1.bias", "blocks.4.attn.qkv.weight", "blocks.4.attn.qkv.bias", "blocks.4.attn.proj.weight", "blocks.4.attn.proj.bias", "blocks.4.norm2.weight", "blocks.4.norm2.bias", "blocks.4.mlp.fc1.weight", "blocks.4.mlp.fc1.bias", "blocks.4.mlp.fc2.weight", "blocks.4.mlp.fc2.bias", "blocks.5.norm1.weight", "blocks.5.norm1.bias", "blocks.5.attn.qkv.weight", "blocks.5.attn.qkv.bias", "blocks.5.attn.proj.weight", "blocks.5.attn.proj.bias", "blocks.5.norm2.weight", "blocks.5.norm2.bias", "blocks.5.mlp.fc1.weight", "blocks.5.mlp.fc1.bias", "blocks.5.mlp.fc2.weight", "blocks.5.mlp.fc2.bias", "blocks.6.norm1.weight", "blocks.6.norm1.bias", "blocks.6.attn.qkv.weight", "blocks.6.attn.qkv.bias", "blocks.6.attn.proj.weight", "blocks.6.attn.proj.bias", "blocks.6.norm2.weight", "blocks.6.norm2.bias", "blocks.6.mlp.fc1.weight", "blocks.6.mlp.fc1.bias", "blocks.6.mlp.fc2.weight", "blocks.6.mlp.fc2.bias", "blocks.7.norm1.weight", "blocks.7.norm1.bias", "blocks.7.attn.qkv.weight", "blocks.7.attn.qkv.bias", "blocks.7.attn.proj.weight", "blocks.7.attn.proj.bias", "blocks.7.norm2.weight", "blocks.7.norm2.bias", "blocks.7.mlp.fc1.weight", "blocks.7.mlp.fc1.bias", "blocks.7.mlp.fc2.weight", "blocks.7.mlp.fc2.bias", "blocks.8.norm1.weight", "blocks.8.norm1.bias", "blocks.8.attn.qkv.weight", "blocks.8.attn.qkv.bias", "blocks.8.attn.proj.weight", "blocks.8.attn.proj.bias", "blocks.8.norm2.weight", "blocks.8.norm2.bias", "blocks.8.mlp.fc1.weight", "blocks.8.mlp.fc1.bias", "blocks.8.mlp.fc2.weight", "blocks.8.mlp.fc2.bias", "blocks.9.norm1.weight", "blocks.9.norm1.bias", "blocks.9.attn.qkv.weight", "blocks.9.attn.qkv.bias", "blocks.9.attn.proj.weight", "blocks.9.attn.proj.bias", "blocks.9.norm2.weight", "blocks.9.norm2.bias", "blocks.9.mlp.fc1.weight", "blocks.9.mlp.fc1.bias", "blocks.9.mlp.fc2.weight", "blocks.9.mlp.fc2.bias", "blocks.10.norm1.weight", "blocks.10.norm1.bias", "blocks.10.attn.qkv.weight", "blocks.10.attn.qkv.bias", "blocks.10.attn.proj.weight", "blocks.10.attn.proj.bias", "blocks.10.norm2.weight", "blocks.10.norm2.bias", "blocks.10.mlp.fc1.weight", "blocks.10.mlp.fc1.bias", "blocks.10.mlp.fc2.weight", "blocks.10.mlp.fc2.bias", "blocks.11.norm1.weight", "blocks.11.norm1.bias", "blocks.11.attn.qkv.weight", "blocks.11.attn.qkv.bias", "blocks.11.attn.proj.weight", "blocks.11.attn.proj.bias", "blocks.11.norm2.weight", "blocks.11.norm2.bias", "blocks.11.mlp.fc1.weight", "blocks.11.mlp.fc1.bias", "blocks.11.mlp.fc2.weight", "blocks.11.mlp.fc2.bias", "blocks.12.norm1.weight", "blocks.12.norm1.bias", "blocks.12.attn.qkv.weight", "blocks.12.attn.qkv.bias", "blocks.12.attn.proj.weight", "blocks.12.attn.proj.bias", "blocks.12.norm2.weight", "blocks.12.norm2.bias", "blocks.12.mlp.fc1.weight", "blocks.12.mlp.fc1.bias", "blocks.12.mlp.fc2.weight", "blocks.12.mlp.fc2.bias", "blocks.13.norm1.weight", "blocks.13.norm1.bias", "blocks.13.attn.qkv.weight", "blocks.13.attn.qkv.bias", "blocks.13.attn.proj.weight", "blocks.13.attn.proj.bias", "blocks.13.norm2.weight", "blocks.13.norm2.bias", "blocks.13.mlp.fc1.weight", "blocks.13.mlp.fc1.bias", "blocks.13.mlp.fc2.weight", "blocks.13.mlp.fc2.bias", "blocks.14.norm1.weight", "blocks.14.norm1.bias", "blocks.14.attn.qkv.weight", "blocks.14.attn.qkv.bias", "blocks.14.attn.proj.weight", "blocks.14.attn.proj.bias", "blocks.14.norm2.weight", "blocks.14.norm2.bias", "blocks.14.mlp.fc1.weight", "blocks.14.mlp.fc1.bias", "blocks.14.mlp.fc2.weight", "blocks.14.mlp.fc2.bias", "blocks.15.norm1.weight", "blocks.15.norm1.bias", "blocks.15.attn.qkv.weight", "blocks.15.attn.qkv.bias", "blocks.15.attn.proj.weight", "blocks.15.attn.proj.bias", "blocks.15.norm2.weight", "blocks.15.norm2.bias", "blocks.15.mlp.fc1.weight", "blocks.15.mlp.fc1.bias", "blocks.15.mlp.fc2.weight", "blocks.15.mlp.fc2.bias", "blocks.16.norm1.weight", "blocks.16.norm1.bias", "blocks.16.attn.qkv.weight", "blocks.16.attn.qkv.bias", "blocks.16.attn.proj.weight", "blocks.16.attn.proj.bias", "blocks.16.norm2.weight", "blocks.16.norm2.bias", "blocks.16.mlp.fc1.weight", "blocks.16.mlp.fc1.bias", "blocks.16.mlp.fc2.weight", "blocks.16.mlp.fc2.bias", "blocks.17.norm1.weight", "blocks.17.norm1.bias", "blocks.17.attn.qkv.weight", "blocks.17.attn.qkv.bias", "blocks.17.attn.proj.weight", "blocks.17.attn.proj.bias", "blocks.17.norm2.weight", "blocks.17.norm2.bias", "blocks.17.mlp.fc1.weight", "blocks.17.mlp.fc1.bias", "blocks.17.mlp.fc2.weight", "blocks.17.mlp.fc2.bias", "blocks.18.norm1.weight", "blocks.18.norm1.bias", "blocks.18.attn.qkv.weight", "blocks.18.attn.qkv.bias", "blocks.18.attn.proj.weight", "blocks.18.attn.proj.bias", "blocks.18.norm2.weight", "blocks.18.norm2.bias", "blocks.18.mlp.fc1.weight", "blocks.18.mlp.fc1.bias", "blocks.18.mlp.fc2.weight", "blocks.18.mlp.fc2.bias", "blocks.19.norm1.weight", "blocks.19.norm1.bias", "blocks.19.attn.qkv.weight", "blocks.19.attn.qkv.bias", "blocks.19.attn.proj.weight", "blocks.19.attn.proj.bias", "blocks.19.norm2.weight", "blocks.19.norm2.bias", "blocks.19.mlp.fc1.weight", "blocks.19.mlp.fc1.bias", "blocks.19.mlp.fc2.weight", "blocks.19.mlp.fc2.bias", "blocks.20.norm1.weight", "blocks.20.norm1.bias", "blocks.20.attn.qkv.weight", "blocks.20.attn.qkv.bias", "blocks.20.attn.proj.weight", "blocks.20.attn.proj.bias", "blocks.20.norm2.weight", "blocks.20.norm2.bias", "blocks.20.mlp.fc1.weight", "blocks.20.mlp.fc1.bias", "blocks.20.mlp.fc2.weight", "blocks.20.mlp.fc2.bias", "blocks.21.norm1.weight", "blocks.21.norm1.bias", "blocks.21.attn.qkv.weight", "blocks.21.attn.qkv.bias", "blocks.21.attn.proj.weight", "blocks.21.attn.proj.bias", "blocks.21.norm2.weight", "blocks.21.norm2.bias", "blocks.21.mlp.fc1.weight", "blocks.21.mlp.fc1.bias", "blocks.21.mlp.fc2.weight", "blocks.21.mlp.fc2.bias", "blocks.22.norm1.weight", "blocks.22.norm1.bias", "blocks.22.attn.qkv.weight", "blocks.22.attn.qkv.bias", "blocks.22.attn.proj.weight", "blocks.22.attn.proj.bias", "blocks.22.norm2.weight", "blocks.22.norm2.bias", "blocks.22.mlp.fc1.weight", "blocks.22.mlp.fc1.bias", "blocks.22.mlp.fc2.weight", "blocks.22.mlp.fc2.bias", "blocks.23.norm1.weight", "blocks.23.norm1.bias", "blocks.23.attn.qkv.weight", "blocks.23.attn.qkv.bias", "blocks.23.attn.proj.weight", "blocks.23.attn.proj.bias", "blocks.23.norm2.weight", "blocks.23.norm2.bias", "blocks.23.mlp.fc1.weight", "blocks.23.mlp.fc1.bias", "blocks.23.mlp.fc2.weight", "blocks.23.mlp.fc2.bias", "blocks.24.norm1.weight", "blocks.24.norm1.bias", "blocks.24.attn.qkv.weight", "blocks.24.attn.qkv.bias", "blocks.24.attn.proj.weight", "blocks.24.attn.proj.bias", "blocks.24.norm2.weight", "blocks.24.norm2.bias", "blocks.24.mlp.fc1.weight", "blocks.24.mlp.fc1.bias", "blocks.24.mlp.fc2.weight", "blocks.24.mlp.fc2.bias", "blocks.25.norm1.weight", "blocks.25.norm1.bias", "blocks.25.attn.qkv.weight", "blocks.25.attn.qkv.bias", "blocks.25.attn.proj.weight", "blocks.25.attn.proj.bias", "blocks.25.norm2.weight", "blocks.25.norm2.bias", "blocks.25.mlp.fc1.weight", "blocks.25.mlp.fc1.bias", "blocks.25.mlp.fc2.weight", "blocks.25.mlp.fc2.bias", "blocks.26.norm1.weight", "blocks.26.norm1.bias", "blocks.26.attn.qkv.weight", "blocks.26.attn.qkv.bias", "blocks.26.attn.proj.weight", "blocks.26.attn.proj.bias", "blocks.26.norm2.weight", "blocks.26.norm2.bias", "blocks.26.mlp.fc1.weight", "blocks.26.mlp.fc1.bias", "blocks.26.mlp.fc2.weight", "blocks.26.mlp.fc2.bias", "blocks.27.norm1.weight", "blocks.27.norm1.bias", "blocks.27.attn.qkv.weight", "blocks.27.attn.qkv.bias", "blocks.27.attn.proj.weight", "blocks.27.attn.proj.bias", "blocks.27.norm2.weight", "blocks.27.norm2.bias", "blocks.27.mlp.fc1.weight", "blocks.27.mlp.fc1.bias", "blocks.27.mlp.fc2.weight", "blocks.27.mlp.fc2.bias", "blocks.28.norm1.weight", "blocks.28.norm1.bias", "blocks.28.attn.qkv.weight", "blocks.28.attn.qkv.bias", "blocks.28.attn.proj.weight", "blocks.28.attn.proj.bias", "blocks.28.norm2.weight", "blocks.28.norm2.bias", "blocks.28.mlp.fc1.weight", "blocks.28.mlp.fc1.bias", "blocks.28.mlp.fc2.weight", "blocks.28.mlp.fc2.bias", "blocks.29.norm1.weight", "blocks.29.norm1.bias", "blocks.29.attn.qkv.weight", "blocks.29.attn.qkv.bias", "blocks.29.attn.proj.weight", "blocks.29.attn.proj.bias", "blocks.29.norm2.weight", "blocks.29.norm2.bias", "blocks.29.mlp.fc1.weight", "blocks.29.mlp.fc1.bias", "blocks.29.mlp.fc2.weight", "blocks.29.mlp.fc2.bias", "blocks.30.norm1.weight", "blocks.30.norm1.bias", "blocks.30.attn.qkv.weight", "blocks.30.attn.qkv.bias", "blocks.30.attn.proj.weight", "blocks.30.attn.proj.bias", "blocks.30.norm2.weight", "blocks.30.norm2.bias", "blocks.30.mlp.fc1.weight", "blocks.30.mlp.fc1.bias", "blocks.30.mlp.fc2.weight", "blocks.30.mlp.fc2.bias", "blocks.31.norm1.weight", "blocks.31.norm1.bias", "blocks.31.attn.qkv.weight", "blocks.31.attn.qkv.bias", "blocks.31.attn.proj.weight", "blocks.31.attn.proj.bias", "blocks.31.norm2.weight", "blocks.31.norm2.bias", "blocks.31.mlp.fc1.weight", "blocks.31.mlp.fc1.bias", "blocks.31.mlp.fc2.weight", "blocks.31.mlp.fc2.bias", "norm.weight", "norm.bias".
        Unexpected key(s) in state_dict: "module.pos_embed", "module.patch_embed.proj.weight", "module.patch_embed.proj.bias", "module.blocks.0.norm1.weight", "module.blocks.0.norm1.bias", "module.blocks.0.attn.qkv.weight", "module.blocks.0.attn.qkv.bias", "module.blocks.0.attn.proj.weight", "module.blocks.0.attn.proj.bias", "module.blocks.0.norm2.weight", "module.blocks.0.norm2.bias", "module.blocks.0.mlp.fc1.weight", "module.blocks.0.mlp.fc1.bias", "module.blocks.0.mlp.fc2.weight", "module.blocks.0.mlp.fc2.bias", "module.blocks.1.norm1.weight", "module.blocks.1.norm1.bias", "module.blocks.1.attn.qkv.weight", "module.blocks.1.attn.qkv.bias", "module.blocks.1.attn.proj.weight", "module.blocks.1.attn.proj.bias", "module.blocks.1.norm2.weight", "module.blocks.1.norm2.bias", "module.blocks.1.mlp.fc1.weight", "module.blocks.1.mlp.fc1.bias", "module.blocks.1.mlp.fc2.weight", "module.blocks.1.mlp.fc2.bias", "module.blocks.2.norm1.weight", "module.blocks.2.norm1.bias", "module.blocks.2.attn.qkv.weight", "module.blocks.2.attn.qkv.bias", "module.blocks.2.attn.proj.weight", "module.blocks.2.attn.proj.bias", "module.blocks.2.norm2.weight", "module.blocks.2.norm2.bias", "module.blocks.2.mlp.fc1.weight", "module.blocks.2.mlp.fc1.bias", "module.blocks.2.mlp.fc2.weight", "module.blocks.2.mlp.fc2.bias", "module.blocks.3.norm1.weight", "module.blocks.3.norm1.bias", "module.blocks.3.attn.qkv.weight", "module.blocks.3.attn.qkv.bias", "module.blocks.3.attn.proj.weight", "module.blocks.3.attn.proj.bias", "module.blocks.3.norm2.weight", "module.blocks.3.norm2.bias", "module.blocks.3.mlp.fc1.weight", "module.blocks.3.mlp.fc1.bias", "module.blocks.3.mlp.fc2.weight", "module.blocks.3.mlp.fc2.bias", "module.blocks.4.norm1.weight", "module.blocks.4.norm1.bias", "module.blocks.4.attn.qkv.weight", "module.blocks.4.attn.qkv.bias", "module.blocks.4.attn.proj.weight", "module.blocks.4.attn.proj.bias", "module.blocks.4.norm2.weight", "module.blocks.4.norm2.bias", "module.blocks.4.mlp.fc1.weight", "module.blocks.4.mlp.fc1.bias", "module.blocks.4.mlp.fc2.weight", "module.blocks.4.mlp.fc2.bias", "module.blocks.5.norm1.weight", "module.blocks.5.norm1.bias", "module.blocks.5.attn.qkv.weight", "module.blocks.5.attn.qkv.bias", "module.blocks.5.attn.proj.weight", "module.blocks.5.attn.proj.bias", "module.blocks.5.norm2.weight", "module.blocks.5.norm2.bias", "module.blocks.5.mlp.fc1.weight", "module.blocks.5.mlp.fc1.bias", "module.blocks.5.mlp.fc2.weight", "module.blocks.5.mlp.fc2.bias", "module.blocks.6.norm1.weight", "module.blocks.6.norm1.bias", "module.blocks.6.attn.qkv.weight", "module.blocks.6.attn.qkv.bias", "module.blocks.6.attn.proj.weight", "module.blocks.6.attn.proj.bias", "module.blocks.6.norm2.weight", "module.blocks.6.norm2.bias", "module.blocks.6.mlp.fc1.weight", "module.blocks.6.mlp.fc1.bias", "module.blocks.6.mlp.fc2.weight", "module.blocks.6.mlp.fc2.bias", "module.blocks.7.norm1.weight", "module.blocks.7.norm1.bias", "module.blocks.7.attn.qkv.weight", "module.blocks.7.attn.qkv.bias", "module.blocks.7.attn.proj.weight", "module.blocks.7.attn.proj.bias", "module.blocks.7.norm2.weight", "module.blocks.7.norm2.bias", "module.blocks.7.mlp.fc1.weight", "module.blocks.7.mlp.fc1.bias", "module.blocks.7.mlp.fc2.weight", "module.blocks.7.mlp.fc2.bias", "module.blocks.8.norm1.weight", "module.blocks.8.norm1.bias", "module.blocks.8.attn.qkv.weight", "module.blocks.8.attn.qkv.bias", "module.blocks.8.attn.proj.weight", "module.blocks.8.attn.proj.bias", "module.blocks.8.norm2.weight", "module.blocks.8.norm2.bias", "module.blocks.8.mlp.fc1.weight", "module.blocks.8.mlp.fc1.bias", "module.blocks.8.mlp.fc2.weight", "module.blocks.8.mlp.fc2.bias", "module.blocks.9.norm1.weight", "module.blocks.9.norm1.bias", "module.blocks.9.attn.qkv.weight", "module.blocks.9.attn.qkv.bias", "module.blocks.9.attn.proj.weight", "module.blocks.9.attn.proj.bias", "module.blocks.9.norm2.weight", "module.blocks.9.norm2.bias", "module.blocks.9.mlp.fc1.weight", "module.blocks.9.mlp.fc1.bias", "module.blocks.9.mlp.fc2.weight", "module.blocks.9.mlp.fc2.bias", "module.blocks.10.norm1.weight", "module.blocks.10.norm1.bias", "module.blocks.10.attn.qkv.weight", "module.blocks.10.attn.qkv.bias", "module.blocks.10.attn.proj.weight", "module.blocks.10.attn.proj.bias", "module.blocks.10.norm2.weight", "module.blocks.10.norm2.bias", "module.blocks.10.mlp.fc1.weight", "module.blocks.10.mlp.fc1.bias", "module.blocks.10.mlp.fc2.weight", "module.blocks.10.mlp.fc2.bias", "module.blocks.11.norm1.weight", "module.blocks.11.norm1.bias", "module.blocks.11.attn.qkv.weight", "module.blocks.11.attn.qkv.bias", "module.blocks.11.attn.proj.weight", "module.blocks.11.attn.proj.bias", "module.blocks.11.norm2.weight", "module.blocks.11.norm2.bias", "module.blocks.11.mlp.fc1.weight", "module.blocks.11.mlp.fc1.bias", "module.blocks.11.mlp.fc2.weight", "module.blocks.11.mlp.fc2.bias", "module.blocks.12.norm1.weight", "module.blocks.12.norm1.bias", "module.blocks.12.attn.qkv.weight", "module.blocks.12.attn.qkv.bias", "module.blocks.12.attn.proj.weight", "module.blocks.12.attn.proj.bias", "module.blocks.12.norm2.weight", "module.blocks.12.norm2.bias", "module.blocks.12.mlp.fc1.weight", "module.blocks.12.mlp.fc1.bias", "module.blocks.12.mlp.fc2.weight", "module.blocks.12.mlp.fc2.bias", "module.blocks.13.norm1.weight", "module.blocks.13.norm1.bias", "module.blocks.13.attn.qkv.weight", "module.blocks.13.attn.qkv.bias", "module.blocks.13.attn.proj.weight", "module.blocks.13.attn.proj.bias", "module.blocks.13.norm2.weight", "module.blocks.13.norm2.bias", "module.blocks.13.mlp.fc1.weight", "module.blocks.13.mlp.fc1.bias", "module.blocks.13.mlp.fc2.weight", "module.blocks.13.mlp.fc2.bias", "module.blocks.14.norm1.weight", "module.blocks.14.norm1.bias", "module.blocks.14.attn.qkv.weight", "module.blocks.14.attn.qkv.bias", "module.blocks.14.attn.proj.weight", "module.blocks.14.attn.proj.bias", "module.blocks.14.norm2.weight", "module.blocks.14.norm2.bias", "module.blocks.14.mlp.fc1.weight", "module.blocks.14.mlp.fc1.bias", "module.blocks.14.mlp.fc2.weight", "module.blocks.14.mlp.fc2.bias", "module.blocks.15.norm1.weight", "module.blocks.15.norm1.bias", "module.blocks.15.attn.qkv.weight", "module.blocks.15.attn.qkv.bias", "module.blocks.15.attn.proj.weight", "module.blocks.15.attn.proj.bias", "module.blocks.15.norm2.weight", "module.blocks.15.norm2.bias", "module.blocks.15.mlp.fc1.weight", "module.blocks.15.mlp.fc1.bias", "module.blocks.15.mlp.fc2.weight", "module.blocks.15.mlp.fc2.bias", "module.blocks.16.norm1.weight", "module.blocks.16.norm1.bias", "module.blocks.16.attn.qkv.weight", "module.blocks.16.attn.qkv.bias", "module.blocks.16.attn.proj.weight", "module.blocks.16.attn.proj.bias", "module.blocks.16.norm2.weight", "module.blocks.16.norm2.bias", "module.blocks.16.mlp.fc1.weight", "module.blocks.16.mlp.fc1.bias", "module.blocks.16.mlp.fc2.weight", "module.blocks.16.mlp.fc2.bias", "module.blocks.17.norm1.weight", "module.blocks.17.norm1.bias", "module.blocks.17.attn.qkv.weight", "module.blocks.17.attn.qkv.bias", "module.blocks.17.attn.proj.weight", "module.blocks.17.attn.proj.bias", "module.blocks.17.norm2.weight", "module.blocks.17.norm2.bias", "module.blocks.17.mlp.fc1.weight", "module.blocks.17.mlp.fc1.bias", "module.blocks.17.mlp.fc2.weight", "module.blocks.17.mlp.fc2.bias", "module.blocks.18.norm1.weight", "module.blocks.18.norm1.bias", "module.blocks.18.attn.qkv.weight", "module.blocks.18.attn.qkv.bias", "module.blocks.18.attn.proj.weight", "module.blocks.18.attn.proj.bias", "module.blocks.18.norm2.weight", "module.blocks.18.norm2.bias", "module.blocks.18.mlp.fc1.weight", "module.blocks.18.mlp.fc1.bias", "module.blocks.18.mlp.fc2.weight", "module.blocks.18.mlp.fc2.bias", "module.blocks.19.norm1.weight", "module.blocks.19.norm1.bias", "module.blocks.19.attn.qkv.weight", "module.blocks.19.attn.qkv.bias", "module.blocks.19.attn.proj.weight", "module.blocks.19.attn.proj.bias", "module.blocks.19.norm2.weight", "module.blocks.19.norm2.bias", "module.blocks.19.mlp.fc1.weight", "module.blocks.19.mlp.fc1.bias", "module.blocks.19.mlp.fc2.weight", "module.blocks.19.mlp.fc2.bias", "module.blocks.20.norm1.weight", "module.blocks.20.norm1.bias", "module.blocks.20.attn.qkv.weight", "module.blocks.20.attn.qkv.bias", "module.blocks.20.attn.proj.weight", "module.blocks.20.attn.proj.bias", "module.blocks.20.norm2.weight", "module.blocks.20.norm2.bias", "module.blocks.20.mlp.fc1.weight", "module.blocks.20.mlp.fc1.bias", "module.blocks.20.mlp.fc2.weight", "module.blocks.20.mlp.fc2.bias", "module.blocks.21.norm1.weight", "module.blocks.21.norm1.bias", "module.blocks.21.attn.qkv.weight", "module.blocks.21.attn.qkv.bias", "module.blocks.21.attn.proj.weight", "module.blocks.21.attn.proj.bias", "module.blocks.21.norm2.weight", "module.blocks.21.norm2.bias", "module.blocks.21.mlp.fc1.weight", "module.blocks.21.mlp.fc1.bias", "module.blocks.21.mlp.fc2.weight", "module.blocks.21.mlp.fc2.bias", "module.blocks.22.norm1.weight", "module.blocks.22.norm1.bias", "module.blocks.22.attn.qkv.weight", "module.blocks.22.attn.qkv.bias", "module.blocks.22.attn.proj.weight", "module.blocks.22.attn.proj.bias", "module.blocks.22.norm2.weight", "module.blocks.22.norm2.bias", "module.blocks.22.mlp.fc1.weight", "module.blocks.22.mlp.fc1.bias", "module.blocks.22.mlp.fc2.weight", "module.blocks.22.mlp.fc2.bias", "module.blocks.23.norm1.weight", "module.blocks.23.norm1.bias", "module.blocks.23.attn.qkv.weight", "module.blocks.23.attn.qkv.bias", "module.blocks.23.attn.proj.weight", "module.blocks.23.attn.proj.bias", "module.blocks.23.norm2.weight", "module.blocks.23.norm2.bias", "module.blocks.23.mlp.fc1.weight", "module.blocks.23.mlp.fc1.bias", "module.blocks.23.mlp.fc2.weight", "module.blocks.23.mlp.fc2.bias", "module.blocks.24.norm1.weight", "module.blocks.24.norm1.bias", "module.blocks.24.attn.qkv.weight", "module.blocks.24.attn.qkv.bias", "module.blocks.24.attn.proj.weight", "module.blocks.24.attn.proj.bias", "module.blocks.24.norm2.weight", "module.blocks.24.norm2.bias", "module.blocks.24.mlp.fc1.weight", "module.blocks.24.mlp.fc1.bias", "module.blocks.24.mlp.fc2.weight", "module.blocks.24.mlp.fc2.bias", "module.blocks.25.norm1.weight", "module.blocks.25.norm1.bias", "module.blocks.25.attn.qkv.weight", "module.blocks.25.attn.qkv.bias", "module.blocks.25.attn.proj.weight", "module.blocks.25.attn.proj.bias", "module.blocks.25.norm2.weight", "module.blocks.25.norm2.bias", "module.blocks.25.mlp.fc1.weight", "module.blocks.25.mlp.fc1.bias", "module.blocks.25.mlp.fc2.weight", "module.blocks.25.mlp.fc2.bias", "module.blocks.26.norm1.weight", "module.blocks.26.norm1.bias", "module.blocks.26.attn.qkv.weight", "module.blocks.26.attn.qkv.bias", "module.blocks.26.attn.proj.weight", "module.blocks.26.attn.proj.bias", "module.blocks.26.norm2.weight", "module.blocks.26.norm2.bias", "module.blocks.26.mlp.fc1.weight", "module.blocks.26.mlp.fc1.bias", "module.blocks.26.mlp.fc2.weight", "module.blocks.26.mlp.fc2.bias", "module.blocks.27.norm1.weight", "module.blocks.27.norm1.bias", "module.blocks.27.attn.qkv.weight", "module.blocks.27.attn.qkv.bias", "module.blocks.27.attn.proj.weight", "module.blocks.27.attn.proj.bias", "module.blocks.27.norm2.weight", "module.blocks.27.norm2.bias", "module.blocks.27.mlp.fc1.weight", "module.blocks.27.mlp.fc1.bias", "module.blocks.27.mlp.fc2.weight", "module.blocks.27.mlp.fc2.bias", "module.blocks.28.norm1.weight", "module.blocks.28.norm1.bias", "module.blocks.28.attn.qkv.weight", "module.blocks.28.attn.qkv.bias", "module.blocks.28.attn.proj.weight", "module.blocks.28.attn.proj.bias", "module.blocks.28.norm2.weight", "module.blocks.28.norm2.bias", "module.blocks.28.mlp.fc1.weight", "module.blocks.28.mlp.fc1.bias", "module.blocks.28.mlp.fc2.weight", "module.blocks.28.mlp.fc2.bias", "module.blocks.29.norm1.weight", "module.blocks.29.norm1.bias", "module.blocks.29.attn.qkv.weight", "module.blocks.29.attn.qkv.bias", "module.blocks.29.attn.proj.weight", "module.blocks.29.attn.proj.bias", "module.blocks.29.norm2.weight", "module.blocks.29.norm2.bias", "module.blocks.29.mlp.fc1.weight", "module.blocks.29.mlp.fc1.bias", "module.blocks.29.mlp.fc2.weight", "module.blocks.29.mlp.fc2.bias", "module.blocks.30.norm1.weight", "module.blocks.30.norm1.bias", "module.blocks.30.attn.qkv.weight", "module.blocks.30.attn.qkv.bias", "module.blocks.30.attn.proj.weight", "module.blocks.30.attn.proj.bias", "module.blocks.30.norm2.weight", "module.blocks.30.norm2.bias", "module.blocks.30.mlp.fc1.weight", "module.blocks.30.mlp.fc1.bias", "module.blocks.30.mlp.fc2.weight", "module.blocks.30.mlp.fc2.bias", "module.blocks.31.norm1.weight", "module.blocks.31.norm1.bias", "module.blocks.31.attn.qkv.weight", "module.blocks.31.attn.qkv.bias", "module.blocks.31.attn.proj.weight", "module.blocks.31.attn.proj.bias", "module.blocks.31.norm2.weight", "module.blocks.31.norm2.bias", "module.blocks.31.mlp.fc1.weight", "module.blocks.31.mlp.fc1.bias", "module.blocks.31.mlp.fc2.weight", "module.blocks.31.mlp.fc2.bias", "module.norm.weight", "module.norm.bias".
"""