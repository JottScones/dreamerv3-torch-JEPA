import mani_skill.envs
import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
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
        self._env = ManiSkillVectorEnv(env, auto_reset=False, ignore_terminations=False)
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
        views = spaces["sensor_data"]["base_camera"]["rgb"]
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
        obs["image"] = to_np(raw_obs["sensor_data"]["base_camera"]["rgb"])
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
        obs["image"] = to_np(raw_obs["sensor_data"]["base_camera"]["rgb"])
        obs["is_first"] = np.ones(self.num_envs, dtype=bool)
        obs["is_last"] = np.zeros(self.num_envs, dtype=bool)
        obs["is_terminal"] = np.zeros(self.num_envs, dtype=bool)
        return obs
"""
Action Space Box(-1.0, 1.0, (7,), float32)
Prefill dataset (2500 steps).
Traceback (most recent call last):
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/dreamer.py", line 460, in <module>
    main(parser.parse_args(remaining))
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/dreamer.py", line 353, in main
    state = simulate_fn(
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/tools.py", line 325, in simulate_vector
    next_obs, reward, terminated, truncated, info = env.step(act_np)
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/dreamer/lib/python3.9/site-packages/gym/core.py", line 280, in step
    return self.env.step(action)
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/envs/wrappers.py", line 105, in step
    return self.env.step(action[self._key])
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/envs/wrappers.py", line 15, in step
    obs, reward, done, info = self.env.step(action)
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/envs/wrappers.py", line 44, in step
    return self.env.step(original)
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/envs/maniskill.py", line 56, in step
    res = self._env.step(action)
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/dreamer/lib/python3.9/site-packages/mani_skill/vector/wrappers/gymnasium.py", line 112, in step
    obs, rew, terminations, truncations, infos = self._env.step(actions)
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/dreamer/lib/python3.9/site-packages/mani_skill/utils/registration.py", line 161, in step
    observation, reward, terminated, truncated, info = self.env.step(action)
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/dreamer/lib/python3.9/site-packages/gymnasium/wrappers/order_enforcing.py", line 56, in step
    return self.env.step(action)
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/dreamer/lib/python3.9/site-packages/mani_skill/envs/sapien_env.py", line 948, in step
    action = self._step_action(action)
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/dreamer/lib/python3.9/site-packages/mani_skill/envs/sapien_env.py", line 1010, in _step_action
    self.agent.set_action(action)
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/dreamer/lib/python3.9/site-packages/mani_skill/agents/base_agent.py", line 324, in set_action
    self.controller.set_action(action)
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/dreamer/lib/python3.9/site-packages/mani_skill/agents/controllers/base_controller.py", line 294, in set_action
    assert action.shape == (
AssertionError: Received action of shape torch.Size([1, 7]) but expected shape (8, 7)
Segmentation fault
"""

