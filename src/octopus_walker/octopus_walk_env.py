"""Gymnasium environment: train the 4-arm, 3-tendon octopus (octopus.xml) to walk.

Built for the tendon-only model made by make_octopus_xml.py (and your rescaled version of it):
  * 12 actions in [-1, 1] -> 12 muscle commands in [0, 1] (order = actuator order a1_m0, a1_m1, a1_m2, a2_m0, ...)
  * commands are low-pass filtered and (optionally) rate-limited to what the real servos can do
Observations use only what the real robot can sense:
  gravity direction in the body frame and gyro (torso IMU), the 12 tendon lengths (servo encoders),
  the 12 commanded values, and a gait clock.   No joint angles, no body velocity, no heading.
"Forward" is the torso's own +x axis, so the policy never needs to know the world heading.

Robustness (set noise=0, randomize=False to switch off): sensor noise and gyro bias, one control step of
action latency, random ground friction and torso mass, random initial heading.
"""
import os
import re

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class OctopusWalkEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    DEFAULT_WEIGHTS = dict(
        vel=30.0,          # reward per m/s of speed along the torso's +x axis (clipped to +-0.3 m/s)
        lateral=5.0,       # penalty per m/s of sideways speed
        tilt=1.0,          # penalty for the torso tipping away from upright
        height=0.0,        # bonus per metre the torso is lifted above its resting height (max 5 cm); try 20 to
                           #   encourage walking on the arm tips instead of dragging the body
        action_rate=0.05,  # penalty on action changes
        energy=0.02,       # penalty on muscle command magnitude
        fall=5.0,          # one-off penalty when the episode ends because the torso tipped over
    )

    def __init__(
        self,
        xml_path="octopus.xml",
        frame_skip=20,                 # 20 x 1 ms = 20 ms control period
        episode_seconds=20.0,
        smooth=0.3,                    # first-order low-pass on commands (1.0 = none)
        phase_period=2.0,              # seconds per cycle of the gait clock fed to the policy
        servo_rpm=None,                # real motor speed: with spool_radius it limits how fast commands may change
        spool_radius=None,             # tendon spool radius (m)
        full_stroke=0.08,              # tendon travel for command 0 -> 1 (m); about 0.08 m for the 19 cm arm
        noise=1.0,                     # scales sensor noise / bias (0 = perfect sensors)
        latency=1,                     # control steps between the policy output and the motors
        randomize=True,                # random friction, mass, heading and joint offsets each episode
        settle_steps=500,
        weights=None,
        render_mode=None,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(os.path.abspath(xml_path))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.frame_skip = frame_skip
        self.dt = m.opt.timestep * frame_skip
        self.max_steps = int(episode_seconds / self.dt)
        self.smooth, self.phase_period = smooth, phase_period
        self.noise, self.latency = float(noise), int(latency)
        self.randomize, self.settle_steps = randomize, settle_steps
        self.render_mode = render_mode
        self.w = dict(self.DEFAULT_WEIGHTS, **(weights or {}))

        def _id(kind, name):
            i = mujoco.mj_name2id(m, kind, name)
            if i < 0:
                raise ValueError(f"'{name}' not found in {xml_path}")
            return i

        self.torso = _id(mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.ground = _id(mujoco.mjtObj.mjOBJ_GEOM, "ground")
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "" for i in range(m.nu)]
        if not names or not all(re.match(r"a\d+_m\d+$", n) for n in names):
            raise ValueError("this environment expects only tendon-muscle actuators named a<i>_m<j> "
                             f"(found {names[:4]}...). Generate the XML without --joint-actuators / --yaw.")
        self.n_act = m.nu
        self.act_tendon = m.actuator_trnid[:, 0].copy()
        self.ten_range = m.tendon_range[self.act_tendon].copy()
        self.gyro_adr = m.sensor_adr[_id(mujoco.mjtObj.mjOBJ_SENSOR, "body_gyro")]

        self.muscle_rate = None        # max change of a command (0..1) per control step
        if servo_rpm is not None:
            assert spool_radius, "servo_rpm needs spool_radius (m)"
            tendon_speed = servo_rpm * 2.0 * np.pi / 60.0 * spool_radius
            self.muscle_rate = self.dt * tendon_speed / full_stroke

        self._mu0 = float(m.geom_friction[self.ground, 0])
        self._mass0 = float(m.body_mass[self.torso])
        self._inertia0 = m.body_inertia[self.torso].copy()

        self._ctrl = np.zeros(self.n_act)
        self._prev_action = np.zeros(self.n_act)
        self._queue = []
        self._com = np.zeros(3)
        self._gyro_bias = np.zeros(3)
        self.step_count = 0
        self._renderer = None
        self._cam = None

        self.action_space = spaces.Box(-1.0, 1.0, (self.n_act,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (len(self._obs()),), dtype=np.float32)

    # ------------------------------------------------------------------ observation
    def _obs(self):
        d = self.data
        R = d.xmat[self.torso].reshape(3, 3)
        n = self.noise
        rng = self.np_random
        gravity = -R[2, :] + n * 0.02 * rng.standard_normal(3)
        gyro = d.sensordata[self.gyro_adr:self.gyro_adr + 3] + self._gyro_bias + n * 0.02 * rng.standard_normal(3)
        ten = d.ten_length[self.act_tendon]
        ten_n = 2.0 * (ten - self.ten_range[:, 0]) / (self.ten_range[:, 1] - self.ten_range[:, 0]) - 1.0
        ten_n = ten_n + n * 0.01 * rng.standard_normal(self.n_act)
        phase = 2.0 * np.pi * self.step_count * self.dt / self.phase_period
        obs = np.concatenate([gravity, 0.5 * gyro, ten_n, self._ctrl, [np.sin(phase), np.cos(phase)]])
        return np.clip(obs, -10, 10).astype(np.float32)

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if self.randomize:
            m.geom_friction[self.ground, 0] = self._mu0 * self.np_random.uniform(0.6, 1.4)
            s = self.np_random.uniform(0.8, 1.2)
            m.body_mass[self.torso] = self._mass0 * s
            m.body_inertia[self.torso] = self._inertia0 * s
            yaw = self.np_random.uniform(0.0, 2.0 * np.pi)
            d.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        d.ctrl[:] = 0.0
        for _ in range(self.settle_steps):          # arms drop onto the ground
            mujoco.mj_step(m, d)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

        self._ctrl[:] = 0.0
        self._prev_action[:] = 0.0
        self._queue = [np.zeros(self.n_act) for _ in range(self.latency)]
        self._com = d.subtree_com[self.torso].copy()
        self._com0 = self._com.copy()
        R = d.xmat[self.torso].reshape(3, 3)
        self._heading0 = R[:2, 0] / (np.linalg.norm(R[:2, 0]) + 1e-9)
        self._z_rest = float(d.xpos[self.torso][2])
        self._gyro_bias = self.noise * 0.01 * self.np_random.standard_normal(3)
        self.step_count = 0
        return self._obs(), {}

    def step(self, action):
        m, d = self.model, self.data
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if self.latency > 0:
            self._queue.append(a)
            applied = self._queue.pop(0)
        else:
            applied = a
        target = 0.5 * (applied + 1.0)
        new_ctrl = self._ctrl + self.smooth * (target - self._ctrl)
        if self.muscle_rate is not None:           # servo speed limit
            new_ctrl = self._ctrl + np.clip(new_ctrl - self._ctrl, -self.muscle_rate, self.muscle_rate)
        self._ctrl = new_ctrl
        d.ctrl[:] = self._ctrl

        for _ in range(self.frame_skip):
            mujoco.mj_step(m, d)
        self.step_count += 1

        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            return np.zeros(self.observation_space.shape, np.float32), -10.0, True, False, {"unstable": True}

        com1 = d.subtree_com[self.torso].copy()
        vel = (com1 - self._com) / self.dt
        self._com = com1
        R = d.xmat[self.torso].reshape(3, 3)
        fwd = R[:2, 0] / (np.linalg.norm(R[:2, 0]) + 1e-9)
        side = np.array([-fwd[1], fwd[0]])
        v_fwd, v_lat = float(vel[:2] @ fwd), float(vel[:2] @ side)
        up = float(R[2, 2])
        rise = max(float(d.xpos[self.torso][2]) - self._z_rest, 0.0)

        w = self.w
        reward = (
            w["vel"] * np.clip(v_fwd, -0.3, 0.3)
            - w["lateral"] * abs(v_lat)
            - w["tilt"] * (1.0 - up)
            + w["height"] * min(rise, 0.05)
            - w["action_rate"] * float(np.mean((a - self._prev_action) ** 2))
            - w["energy"] * float(np.mean(self._ctrl ** 2))
        )
        self._prev_action = a
        terminated = bool(up < 0.3)
        if terminated:
            reward -= w["fall"]
        truncated = self.step_count >= self.max_steps
        info = dict(v_fwd=v_fwd, up=up, rise=rise,
                    dist=float((com1[:2] - self._com0[:2]) @ self._heading0))
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._cam = mujoco.MjvCamera()
            self._cam.distance, self._cam.azimuth, self._cam.elevation = 0.8, 90.0, -25.0
        self._cam.lookat[:] = self.data.subtree_com[self.torso]
        self._renderer.update_scene(self.data, camera=self._cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None