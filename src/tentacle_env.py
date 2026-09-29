"""
Custom Gymnasium environment: a 24-link tendon-driven tentacle that must
learn to crawl by peristaltic (inchworm-style) undulation.

Unlike the earlier torque-motor arm, this model is driven by two
antagonistic MuJoCo `muscle` actuators pulling tendons routed through the
whole body (ctrlrange 0-1, not signed torque). The action space here is
built directly from the model's own `actuator_ctrlrange`, so this class
works unmodified for either actuator type.

qpos / qvel layout (26 DOF total):
    index 0:      base_x      (world x position of crawl_base)
    index 1:      base_z      (offset from crawl_base's resting height)
    index 2:      base_pitch  (crawl_base tilt, radians)
    index 3-25:   j1 .. j23   (the 23 inter-link hinges)
"""

import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register

_DEFAULT_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tentacle_crawl.xml")


class TentacleCrawlEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        xml_path=_DEFAULT_XML,
        frame_skip=20,
        forward_reward_weight=5.0,
        ctrl_cost_weight=0.01,
        healthy_reward=0.5,
        healthy_z_range=(0.0, 0.5),
        reset_noise_scale=0.005,
        render_mode=None,
    ):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "crawl_base")

        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * frame_skip

        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.healthy_reward = healthy_reward
        self.healthy_z_range = healthy_z_range
        self.reset_noise_scale = reset_noise_scale

        self.render_mode = render_mode
        self._viewer = None

        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        # Build the action space from the model's actual ctrlrange instead
        # of assuming +/-1: muscles here are ctrlrange="0 1" (activation),
        # not signed torque like the earlier motor-driven arm.
        ctrl_range = self.model.actuator_ctrlrange.copy()
        self.action_space = spaces.Box(
            low=ctrl_range[:, 0].astype(np.float32),
            high=ctrl_range[:, 1].astype(np.float32),
            dtype=np.float32,
        )

        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

    def _get_base_x(self):
        return float(self.data.qpos[0])

    def _get_base_height(self):
        return float(self.data.xpos[self._base_body_id, 2])

    def _is_healthy(self):
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        if not finite:
            return False
        lo, hi = self.healthy_z_range
        return lo <= self._get_base_height() <= hi

    def _get_obs(self):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        # Tendon lengths are cheap, physically meaningful feedback for a
        # tendon-driven system (roughly analogous to muscle spindle
        # feedback) -- include them alongside joint state.
        ten_length = self.data.ten_length.copy()
        return np.concatenate([qpos[1:], qvel, ten_length]).astype(np.float32)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action

        x_before = self._get_base_x()
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        x_after = self._get_base_x()

        forward_vel = (x_after - x_before) / self.dt
        forward_reward = self.forward_reward_weight * forward_vel
        ctrl_cost = self.ctrl_cost_weight * float(np.sum(np.square(action)))
        healthy = self._is_healthy()
        alive_bonus = self.healthy_reward if healthy else 0.0

        reward = forward_reward + alive_bonus - ctrl_cost
        terminated = not healthy
        truncated = False

        obs = self._get_obs()
        info = {
            "x_position": x_after,
            "forward_vel": forward_vel,
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_alive": alive_bonus,
        }
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        noise = self.reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(-noise, noise, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(-noise, noise, size=self.model.nv)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def render(self):
        if self.render_mode != "human":
            return None
        if self._viewer is None:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


register(
    id="TentacleCrawl-v0",
    entry_point="tentacle_env:TentacleCrawlEnv",
    max_episode_steps=1000,
)