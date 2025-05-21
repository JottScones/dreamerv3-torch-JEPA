import mani_skill.envs
import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper, FlattenActionSpaceWrapper

import numpy as np

to_np = lambda x: x.detach().cpu().numpy()
class ManiSkill:
    def __init__(self, task="PushT-v1", obs_key="image", act_key="action", size=(64, 64), num_envs=1, seed=0, control_mode="pd_joint_delta_pos"):
        # 9x9, 11x11, 13x13 and 15x15 are available
        env = gym.make(
            task, # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
            sensor_configs=dict(width=size[0], height=size[1]),
            num_envs=num_envs,
            obs_mode="rgb", # there is also "state_dict", "rgbd", ...
            control_mode=control_mode, # there is also "pd_joint_delta_pos", ...
            sim_backend="physx_cuda"
        )
        env = ManiSkillVectorEnv(env, auto_reset=False, ignore_terminations=False)
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=False)

        if isinstance(env.action_space, gym.spaces.Dict):
            env = FlattenActionSpaceWrapper(env)

        self._env = env
        self._seed = seed
        self._obs_key = obs_key
        self._act_key = act_key
        self._size = size
        self.num_envs = num_envs

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def observation_space(self):
        spaces = self._env.single_observation_space.spaces.copy()
        views = spaces["rgb"]
        if views.shape[0] > 1:
            print(f"multiple views are not yet supported")

        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
                "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            }
        )

    @property
    def action_space(self):
        space = self._env.single_action_space
        space.discrete = True
        return space

    def step(self, action):
        res = self._env.step(action)
        raw_obs, reward, done, truncated, info = res

        reward = reward.cpu()
        done = done.cpu()

        obs = {}
        obs["image"] = to_np(raw_obs["rgb"])
        obs["is_first"] = np.zeros(self.num_envs, dtype=bool)
        obs["is_last"] = done
        if "success" in info:
            obs["is_terminal"] = to_np(info["success"].cpu())
        else:
            obs["is_terminal"] = np.zeros(self.num_envs, dtype=bool)

        return obs, reward, done, info

    def reset(self, *args, **kwargs):
        raw_obs, info = self._env.reset(*args, seed=self._seed, **kwargs)

        obs = {}
        obs["image"] = to_np(raw_obs["rgb"])
        obs["is_first"] = np.ones(self.num_envs, dtype=bool)
        obs["is_last"] = np.zeros(self.num_envs, dtype=bool)
        obs["is_terminal"] = np.zeros(self.num_envs, dtype=bool)
        return obs
"""

"""