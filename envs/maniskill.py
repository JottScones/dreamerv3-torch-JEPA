import mani_skill.envs
import gymnasium as gym

to_np = lambda x: x.detach().cpu().numpy()
class ManiSkill:
    def __init__(self, task="PushT-v1", obs_key="image", act_key="action", size=(64, 64), seed=0, control_mode="pd_joint_delta_pos"):
        # 9x9, 11x11, 13x13 and 15x15 are available
        self._env = gym.make(
            task, # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
            sensor_configs=dict(width=size[0], height=size[1]),
            num_envs=1,
            obs_mode="rgb", # there is also "state_dict", "rgbd", ...
            control_mode=control_mode, # there is also "pd_joint_delta_pos", ...
        )
        self._seed = seed
        self._obs_key = obs_key
        self._act_key = act_key
        self._size = size
        self._gray = False

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def observation_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {self._obs_key: self._env.observation_space}
        return gym.spaces.Dict(
            {
                **spaces,
                "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            }
        )

    @property
    def action_space(self):
        space = self._env.action_space
        space.discrete = True
        return space

    def step(self, action):
        res = self._env.step(action)
        raw_obs, reward, done, truncated, info = res

        reward = reward.squeeze()
        done = done.squeeze()

        obs = dict()
        obs["image"] = to_np(raw_obs["sensor_data"]["base_camera"]["rgb"])
        obs["is_first"] = False
        obs["is_last"] = done
        obs["is_terminal"] = info.get("success", False)
        return obs, reward, done, info

    def reset(self):
        raw_obs, info = self._env.reset(seed=self._seed)

        obs = dict()
        obs["image"] = to_np(raw_obs["sensor_data"]["base_camera"]["rgb"])
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        return obs

"""
Logdir logdir/test_maniskill
Create envs.
/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/dreamer/lib/python3.9/site-packages/sapien/_vulkan_tricks.py:37: UserWarning: Failed to find Vulkan ICD file. This is probably due to an incorrect or partial installation of the NVIDIA driver. SAPIEN will attempt to provide an ICD file anyway but it may not work.
  warn(
Action Space Box(-1.0, 1.0, (7,), float32)
Prefill dataset (2500 steps).
Traceback (most recent call last):
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/dreamer.py", line 458, in <module>
    main(parser.parse_args(remaining))
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/dreamer.py", line 351, in main
    state = tools.simulate(
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/tools.py", line 201, in simulate
    length *= 1 - done
ValueError: non-broadcastable output operand with shape (1,) doesn't match the broadcast shape (1,1)
"""

