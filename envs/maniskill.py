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
        self._num_envs = num_envs

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

        print(raw_obs.shape)
        print(raw_obs)
        
        reward = reward.cpu()
        done = done.cpu()

        obs = []
        for env in range(self._num_envs):
            obs.append({
                "image": to_np(raw_obs["sensor_data"]["base_camera"]["rgb"].squeeze(0)),
                "is_first": False,
                "is_last": done[env],
                "is_terminal": info[env]["success"].cpu() if "success" in info[env] else False 
            })

        return obs, reward, done, info

    def reset(self):
        raw_obs, info = self._env.reset(seed=self._seed)
        print(raw_obs.shape)
        print(raw_obs)

        obs = []
        for env in range(self._num_envs):
            obs.append({
                "image": to_np(raw_obs["sensor_data"]["base_camera"]["rgb"].squeeze(0)),
                "is_first": True,
                "is_last": False,
                "is_terminal": False,
            })

        return obs

"""
Traceback (most recent call last):
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/dreamer.py", line 458, in <module>
    main(parser.parse_args(remaining))
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/dreamer.py", line 390, in main
    tools.simulate(
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/tools.py", line 178, in simulate
    action, agent_state = agent(obs, done, agent_state)
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/dreamer.py", line 92, in __call__
    policy_output, state = self._policy(obs, state, training)
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/dreamer.py", line 111, in _policy
    latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
  File "/rds/user/swj24/hpc-work/dissertation/dreamerv3-torch-JEPA/networks.py", line 256, in obs_step
    x = torch.cat([prior["deter"], embed], -1)
RuntimeError: Tensors must have same number of dimensions: got 2 and 3
"""

