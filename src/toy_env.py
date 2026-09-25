import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

class ToyArmEnv(gym.Env):
    def __init__(self, xml_path, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_site")
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.mocap_id = self.model.body_mocapid[self.target_body_id]

        n_act = self.model.nu
        self.action_space = spaces.Box(-1, 1, (n_act,), dtype=np.float32)
        obs_dim = 3*self.model.nq  # placeholder, tune to your obs
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)

        self.render_mode = render_mode
        self.viewer = None

    def _sample_target(self):
        # Arm moves in a single vertical plane (X-Z) since all joints
        # rotate about the Y axis. Sample in polar coords within that plane.
        r_min, r_max = 0.15, 0.8  # inner/outer reachable radius, tune to your link lengths
        r = np.random.uniform(r_min, r_max)
        theta = np.random.uniform(-np.pi, np.pi)  # full circle in-plane; restrict if you only want a front arc, e.g. (-np.pi/2, np.pi/2)

        x = r * np.cos(theta)
        z = 0.1 + r * np.sin(theta)  # offset by base height so targets don't clip through the ground
        y = 0.0  # locked — no out-of-plane motion

        return np.array([x, y, z])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.mocap_pos[self.mocap_id] = self._sample_target()
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        torque_range = self.model.actuator_ctrlrange
        scaled = action * torque_range[:, 1]  # crude scaling, refine
        self.data.ctrl[:] = scaled
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        ee_pos = self.data.site_xpos[self.ee_site_id]
        target_pos = self.data.mocap_pos[self.mocap_id]
        dist = np.linalg.norm(ee_pos - target_pos)

        reward = -dist - 0.001*np.sum(np.square(action)) - 1e-4*np.sum(np.square(self.data.qvel))
        terminated = False
        truncated = False
        info = {"dist": dist}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        ee_pos = self.data.site_xpos[self.ee_site_id]
        target_pos = self.data.mocap_pos[self.mocap_id]
        return np.concatenate([self.data.qpos, self.data.qvel, ee_pos, target_pos - ee_pos]).astype(np.float32)