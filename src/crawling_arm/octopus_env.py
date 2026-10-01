"""Gymnasium environment for the MuJoCo octopus-arm model (spiral_robot.xml).

Model facts this env relies on (read from the XML):
  * root body "link1" has a free joint; its quat (0.5,0.5,0.5,0.5) maps local z -> world +x,
    local y -> world +z. So the arm lies along world +x and its 23 hinges (local x axis = world y)
    bend it in the vertical x-z plane. Tendon 1 runs along the top, tendon 2 along the bottom.
  * 25 actuators: act1/act2 (muscles, ctrl in [0,1]) and adj1..adj23 (joint position targets, +-30 deg).

Action modes
  "reduced": [2 muscle activations, K spatial-mode coefficients]  (default, much easier to learn)
  "full"   : [2 muscle activations, 23 joint targets]
A phase clock (sin/cos) is added to the observation to make rhythmic gaits easy to discover.
"""
import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

N_JOINTS = 23
JOINT_LIM = 0.523598776  # rad, from the XML


class OctopusArmEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    DEFAULT_WEIGHTS = dict(
        vel=10.0,          # reward per m/s of forward COM speed (clipped to +-0.5 m/s)
        lateral=5.0,       # penalty per m/s of sideways COM speed
        roll=0.5,          # penalty for rolling the base link off the vertical bending plane
        yaw=0.5,           # penalty for the base heading drifting away from the crawl direction
        action_rate=0.02,  # penalty on action changes (smoothness)
        energy=0.02,       # penalty on muscle activation / joint effort
        fall=5.0,          # one-off penalty when the episode is terminated for rolling over
    )

    def __init__(
        self,
        xml_path="spiral_robot.xml",
        frame_skip=20,            # 20 * 1 ms = 20 ms control period
        episode_seconds=8.0,
        action_mode="reduced",
        n_modes=4,
        smooth=0.5,               # first-order low-pass on commands, 1.0 = no filtering
        phase_period=1.0,         # seconds, period of the clock fed to the policy
        direction=(1.0, 0.0),     # crawl direction in the world xy plane
        randomize=True,
        settle_steps=300,
        weights=None,
        render_mode=None,
    ):
        super().__init__()
        assert action_mode in ("reduced", "full")
        self.model = mujoco.MjModel.from_xml_path(os.path.abspath(xml_path))
        self.data = mujoco.MjData(self.model)
        m = self.model

        self.frame_skip = frame_skip
        self.dt = m.opt.timestep * frame_skip
        self.max_steps = int(episode_seconds / self.dt)
        self.action_mode = action_mode
        self.n_modes = n_modes
        self.smooth = smooth
        self.phase_period = phase_period
        self.randomize = randomize
        self.settle_steps = settle_steps
        self.render_mode = render_mode
        self.w = dict(self.DEFAULT_WEIGHTS, **(weights or {}))
        d = np.asarray(direction, dtype=float)
        self.dir = d / np.linalg.norm(d)

        def _id(kind, name):
            i = mujoco.mj_name2id(m, kind, name)
            if i < 0:
                raise ValueError(f"'{name}' not found in model")
            return i

        self.root = _id(mujoco.mjtObj.mjOBJ_BODY, "link1")
        jids = [_id(mujoco.mjtObj.mjOBJ_JOINT, f"j{i}") for i in range(1, N_JOINTS + 1)]
        self.qadr = m.jnt_qposadr[jids]
        self.dadr = m.jnt_dofadr[jids]
        self.mu_ids = np.array([_id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ("act1", "act2")])
        self.adj_ids = np.array(
            [_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"adj{i}") for i in range(1, N_JOINTS + 1)]
        )

        s = np.linspace(0.0, 1.0, N_JOINTS)  # position along the arm, base -> tip
        self.basis = np.stack([np.cos(k * np.pi * s) for k in range(n_modes)])  # (K, 23)

        self.act_dim = 2 + (n_modes if action_mode == "reduced" else N_JOINTS)
        self.action_space = spaces.Box(-1.0, 1.0, (self.act_dim,), dtype=np.float32)

        self._ctrl = np.zeros(m.nu)
        self._prev_action = np.zeros(self.act_dim)
        self._com_vel = np.zeros(3)
        self.step_count = 0
        self._renderer = None
        self._cam = None

        obs_dim = len(self._obs())
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)

    # ------------------------------------------------------------------ helpers
    def _target_ctrl(self, a):
        tgt = np.zeros(self.model.nu)
        tgt[self.mu_ids] = 0.5 * (a[:2] + 1.0)
        if self.action_mode == "full":
            joints = a[2:] * JOINT_LIM
        else:
            joints = np.clip(a[2:] @ self.basis, -1.0, 1.0) * JOINT_LIM
        tgt[self.adj_ids] = joints
        return tgt

    def _obs(self):
        d, m = self.data, self.model
        R = d.xmat[self.root].reshape(3, 3)
        tr = m.tendon_range
        ten_n = 2.0 * (d.ten_length - tr[:, 0]) / (tr[:, 1] - tr[:, 0]) - 1.0
        phase = 2.0 * np.pi * self.step_count * self.dt / self.phase_period
        obs = np.concatenate([
            d.qpos[self.qadr] / JOINT_LIM,
            d.qvel[self.dadr] * 0.1,
            -R[2, :],                   # gravity direction in the base frame
            d.qvel[3:6] * 0.5,          # base angular velocity (local frame)
            self._com_vel * 5.0,
            ten_n,
            d.act,
            self._prev_action,
            [np.sin(phase), np.cos(phase)],
        ])
        return np.clip(obs, -10, 10).astype(np.float32)

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if self.randomize:
            d.qpos[self.qadr] += self.np_random.uniform(-0.05, 0.05, N_JOINTS)
        d.ctrl[:] = 0.0
        for _ in range(self.settle_steps):  # let the arm drop onto the ground
            mujoco.mj_step(m, d)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        self._ctrl[:] = 0.0
        self._prev_action[:] = 0.0
        self._com_vel[:] = 0.0
        self._com_start = d.subtree_com[self.root].copy()
        self.step_count = 0
        return self._obs(), {}

    def step(self, action):
        m, d = self.model, self.data
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._ctrl += self.smooth * (self._target_ctrl(a) - self._ctrl)
        d.ctrl[:] = self._ctrl

        com0 = d.subtree_com[self.root].copy()
        for _ in range(self.frame_skip):
            mujoco.mj_step(m, d)
        self.step_count += 1

        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            return np.zeros(self.observation_space.shape, np.float32), -10.0, True, False, {"unstable": True}

        com1 = d.subtree_com[self.root].copy()
        self._com_vel = (com1 - com0) / self.dt
        v_fwd = self._com_vel[0] * self.dir[0] + self._com_vel[1] * self.dir[1]
        v_lat = -self._com_vel[0] * self.dir[1] + self._com_vel[1] * self.dir[0]

        R = d.xmat[self.root].reshape(3, 3)
        up = R[2, 1]                                              # 1 = bending plane still vertical
        yaw = -R[0, 2] * self.dir[1] + R[1, 2] * self.dir[0]       # base axis sideways component

        w = self.w
        reward = (
            w["vel"] * np.clip(v_fwd, -0.5, 0.5)
            - w["lateral"] * abs(v_lat)
            - w["roll"] * (1.0 - up)
            - w["yaw"] * abs(yaw)
            - w["action_rate"] * np.mean((a - self._prev_action) ** 2)
            - w["energy"] * (np.mean(d.act ** 2) + 0.1 * np.mean(np.abs(d.actuator_force[self.adj_ids])))
        )
        self._prev_action = a

        terminated = bool(up < 0.3 or com1[2] > 0.5)
        if terminated:
            reward -= w["fall"]
        truncated = self.step_count >= self.max_steps

        info = dict(
            v_fwd=float(v_fwd),
            dist=float(np.dot(com1[:2] - self._com_start[:2], self.dir)),
            up=float(up),
        )
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._cam = mujoco.MjvCamera()
            self._cam.distance, self._cam.azimuth, self._cam.elevation = 1.0, 90.0, -15.0
        self._cam.lookat[:] = self.data.subtree_com[self.root]
        self._cam.lookat[2] = 0.06
        self._renderer.update_scene(self.data, camera=self._cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
