import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset

        # Setup world model, actor and critic models
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)

        # reward function is just the instantaneous reward network from the world model
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()

        # exploratory behaviour depends on configuration (seems like greedy is default)
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior, # greedy comes from actor network
            random=lambda: expl.Random(config, act_space), # uniform random action sampling 
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward), # TODO: explore
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            # number of training steps depending on whether we are pretraining or not
            # seems that pretraining is a larger number which is probably to warm up the weights
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count

            if self._should_log(step):
                # At logging intervals log scalar metrics by averaging
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []

                # Also produce video predictions if desired
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))

                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step

        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state

        # Encode sensory input
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)

        # Produce posterior of stoch and recurrent state from observations
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])

        if self._config.eval_state_mean:
            # Overwrite stochastic sample to be the mean of the latent distribution
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)

        if not training:
            # if we are not training use actor policy greedily (no sampling)
            actor = self._task_behavior.actor(feat)
            action = actor.mode()

        elif self._should_expl(self._step):
            # if we are under a exploration schedule we use the exploration behaviour actor  
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()

        else:
            # otherwise sample action from policy
            actor = self._task_behavior.actor(feat)
            action = actor.sample()

        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )

        # output action, and its log probability
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        # train world model on observed data
        post, context, mets = self._wm._train(data)
        metrics.update(mets)

        # posterior holds the model state from the last observation
        start = post

        # reward funciton is the instantaneous prediction from the world model reward head
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()

        # train actor and critic networks from the following imagined roll out
        metrics.update(self._task_behavior._train(start, reward)[-1])

        # TODO: explore
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})

        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def count_steps(folder):
    # strange but counts steps by counting over the recorded steps in the NPZ files
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    # create dataset generator from the episodes
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id):
    # handles the setup of different environments
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)

    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)

    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)

    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)

    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)

    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)

    elif suite == "maniskill":
        import envs.maniskill as maniskill
        env = maniskill.ManiSkill(task=task, seed=config.seed + id, size=config.size, control_mode=config.control_mode)
        env = wrappers.OneHotAction(env)

    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)

    return env


def main(config):
    """
    Training of dreamer happens with a small initial dataset of episodes (gathered potentially with a random policy).
    We create environments, initialise the dreamer agent and simulate interactions in those environemnts. 
    
    In the simulation, everytime we call the agent to produce the next action given observations, it first trains itself (world model, actor and critic)
    from batches sampled from the training dataset of episodes.

    New episodes are generated from the simulation and saved, so, over time, the training dataset grows and grows.
    HOWEVER: it does not appear that this training dataset is accessible to the agent unless we reinitilise it.
    """
    # setup config parameters and logging
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    # ----- Create environments + load episodes -----
    print("Create envs.")
    # if we have a dataset of episodes from disk load them 
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode, id: make_env(config, mode, id)

    # we might have multiple environments so we can collect data in parallel
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]

    # if enabled run the environements in parallel
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]

    else:
        # otherwise we run the environments on the same process sequentially
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]

    acts = train_envs[0].action_space

    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    # ------ create episodes ----- 
    # if we do not have a dataset of episodes then we create a dataset using a completely random
    # policy
    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(config.envs, 1),
                    torch.tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        # agent samples action from uniform distibution
        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)

    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)

    # if agent checkpoint exists load model and optimiser state
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            # use dreamer in eval mode 
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))

        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )

        # save agent state and optimiser state
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")

    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parse evironment config
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    # basically update default config parameters with the one passed in
    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    parser = argparse.ArgumentParser()

    # every parameter in the config is a potential argument in the command line
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    main(parser.parse_args(remaining))
