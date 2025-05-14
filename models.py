import copy
import torch
from torch import nn

import networks
import tools

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:
    """
    running mean and std

    We normalise returns since different environments have different reward scales.
    
    The return distribution can be muliti-modal and include outliers. Normalising by smallest and largest observed returns
    would scale returns down too much. For robustness, against outliers and multi-modality, we use the 5th and 95th percentiles.
    EMA is used to smooth the estimate.

    Why use EMA? 
    We expect the distribution of returns to change over time as the agent learns. For example, larger positive rewards should
    be observed more frequently. 
    """

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):

        flat_x = torch.flatten(x.detach()) # Return distribution
        x_quantile = torch.quantile(input=flat_x, q=self.range) 

        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0) # Do not divide by anything smaller than 1.0
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    """
    Module for the World model and its training.
     
    Includes the recurrent state space model (RSSM), the encoder, the decoder, the reward predictor, and the continue predictor.
    """
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step

        # Automatic Mixed Precision - improves efficiency deciding which operations to use FP16 or FP32
        self._use_amp = True if config.precision == 16 else False

        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}

        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim

        self.dynamics = networks.RSSM(
            config.dyn_stoch, # stochastic latent size
            config.dyn_deter, # recurrent state size
            config.dyn_hidden, # various network hidden sizes
            config.dyn_rec_depth, # number of recurrent layers to update the recurrent state
            config.dyn_discrete, # is the latent discrete or continuous (if true then equal to num of classes)
            config.act, # activation function
            config.norm, # layer norm
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size, # size of the encoder output
            config.device,
        )

        # heads are the networks that take the model state and predict something (e.g. the next observation, reward, and whether the episode is finished)
        self.heads = nn.ModuleDict()

        # model state is the concatenation of the stochastic and recurrent states
        if config.dyn_discrete:
            # if discrete stochastic latent is used it is one-hot encoded by the number of classes
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter

        else:
            # if continuous then size of stochastic latent is the same
            feat_size = config.dyn_stoch + config.dyn_deter

        # Decoder takes the model state and tries to recover the inputs
        self.heads["decoder"] = networks.MultiDecoder(feat_size, shapes, **config.decoder)

        # Reward predictor takes the model state and tries to predict the instantaneous reward
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )

        # Continue predictor takes the model state and tries to predict whether the episode is finished
        # During imagination it is effectively a discount factor, after a certain t the reward estimates are masked out.
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )

        # grad_heads are the heads that are used to calculate the gradients for the model
        # for the most part all heads are used to calculate the gradients so grad_heads == heads
        for name in config.grad_heads:
            assert name in self.heads, name

        # initialise optimiser (opt=adam default)
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        # track gradients for the model
        with tools.RequiresGrad(self):
            # use automatic mixed precision
            with torch.cuda.amp.autocast(self._use_amp):
                # Predict the next latent distribution (prior) and then calculate it using observed data (post) 
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )

                # Calculate the two kl losses between the prior and posterior distributions using stop-gradients
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale

                # kl_loss is the weighted sum of the two kl losses
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )

                assert kl_loss.shape == embed.shape[:2], kl_loss.shape

                preds = {}
                # heads are decoder, reward, and continue
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads

                    # get the model state and detach it if the head is not used for gradients
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach() 

                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred

                    
                losses = {}
                for name, pred in preds.items():
                    # Negative log likelihood loss
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss

                # use custom loss scaling if available
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
            
            # Calculate the gradients and update the model parameters
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))

        # use automatic mixed precision
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        # convert every observation to torch tensor of float32
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }

        # normalise images to [0, 1]
        obs["image"] = obs["image"] / 255.0

        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = obs["discount"].unsqueeze(-1)

        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs

        # continue is the inverse of is_terminal
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        return obs

    def video_pred(self, data):
        """
        Mostly for debugging purposes.

        The first 5 observations are reconstructed using the model.
        Then there model performs open-loop prediction using the reamining actions from 6 onwards.
        For each action the model imagines a new image.
        """
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]

        # posterior from observations is used to initialise the open-loop predictions
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps

        # Concat the reconstructed images and the open-loop predictions
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model

        # Only calculate the error for the reconstructed images
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    """
    Implements and trains the actor and critic networks.
    """
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()

        # Automatic Mixed Precision - improves efficiency deciding which operations to use FP16 or FP32
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model

        # model state is the concatenation of the stochastic and recurrent states
        if config.dyn_discrete:
            # if discrete stochastic latent is used it is one-hot encoded by the number of classes
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter

        else:
            # if continuous then size of stochastic latent is the same
            feat_size = config.dyn_stoch + config.dyn_deter

        # Actor given the model state predict the next action
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )

        # Critic network, predicts the cumulative / lambda-returns given the model state
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        # In the lambda returns we bootstrap meaning we regress using the instantaneous rewards from the trajectory and the predicted
        # cumulative reward from the critic. We use a slow moving version of the critic network to bootstrap off, rather than the online
        # critic which might lead to unstable updates because of moving target issues. 
        # slow target is often the EMA of the online model
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0

        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )

        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer(
                "ema_vals", torch.zeros((2,), device=self._config.device)
            )
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start, # posterior
        objective, # instantaneous reward function
    ):
        self._update_slow_target()
        metrics = {}

        # track gradients for the model
        with tools.RequiresGrad(self.actor):
            # use automatic mixed precision
            with torch.cuda.amp.autocast(self._use_amp):

                # Sample an imagined trajectory and calculate instantaneous rewards
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )
                reward = objective(imag_feat, imag_state, imag_action)
                
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()

                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )
                # compute actor loss using REINFORCE
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                )
                # we also subtract an entropy regularisation term, to encourage the policy to remain stochastic and explore
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        # track gradients for the model
        with tools.RequiresGrad(self.value):
            # use automatic mixed precision
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())

                # use the slow_target to stabilise learning
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())

                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))

        # update actor and critic networks
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))

        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon):
        # Imagine a single roll-out from randomly sampled policy actions.

        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            # Sample an action
            action = policy(inp).sample()
            # Imagine model state (prior) after taking action
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        # We roll-out the imagined tracjectory horizon steps into the future
        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            # Use continue model output to adjust discount factor
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean

        else:
            # Otherwise use a constant discount factor
            discount = self._config.discount * torch.ones_like(reward)

        # For each imagined model state caculate the mode of the distribution of returns
        value = self.value(imag_feat).mode()

        # Calculate the lambda returns 
        target = tools.lambda_return(
            reward[1:], # instantaneous rewards 
            value[:-1], # baseline predicted cumulative rewards
            discount[1:], # discount factor
            bootstrap=value[-1], # last value prediction
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        # Compute cumulative product of discount factors, to express the contribution of future time steps to the loss
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat, # imagined world states
        imag_action, # imagined actions
        target, # lambda returns target
        weights, 
        base, # mode of predicted returns from critic
    ):
        # The actor loss can be calculated usting straight through gradients from the dynamics
        # or from REINFORCE. Although most of the time REINFORCE is used.
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            # offset: 5th percentile | scale: 95th - 5th percentile
            offset, scale = self.reward_ema(target, self.ema_vals)
            # min-max normalise
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale

            # Calculate the advantage, how much a specific action is to the average reward
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv

        elif self._config.imag_gradient == "reinforce":
            # Target is the lambda-returns and value is the critic prediction
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )

        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)

        # Actor loss is the negative weighted actor target where weights are the discount factors over time
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        # slowly mix new critic parameters into slow critic parameters
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
