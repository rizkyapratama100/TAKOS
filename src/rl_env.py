import os
import mujoco
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TentacleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(script_dir, "tentacle_locked.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.render_mode = render_mode
        self.viewer = None
        
        # Action space: 2 Tendon Muscles [act1, act2], ctrlrange 0 to 1
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation space
        obs_size = self.model.nq + self.model.nv + 3 + 3 + 2 + 2
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        
        self.tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link24")
        
        self.target = np.zeros(3, dtype=np.float32)
        self.prev_action = np.zeros(2, dtype=np.float32)
        
        self.frame_skip = 5 # Number of MuJoCo steps per RL step
        
        self.max_steps = 1000
        self.current_step = 0
        
    def _get_tip_state(self):
        # tip position y, z
        xpos = self.data.xpos[self.tip_id]
        y, z = xpos[1], xpos[2]
        
        # tip orientation (theta in YZ plane)
        # 3rd column of rotation matrix is local Z axis
        xmat = self.data.xmat[self.tip_id].reshape(3, 3)
        vec_z = xmat[:, 2] # World vector representing local Z
        theta = np.arctan2(vec_z[1], vec_z[2])
        
        return np.array([y, z, theta], dtype=np.float32)
        
    def _get_obs(self):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        tip_state = self._get_tip_state()
        tendon_lengths = self.data.actuator_length[0:2].copy()
        
        obs = np.concatenate([
            qpos,
            qvel,
            tip_state,
            self.target,
            tendon_lengths,
            self.prev_action
        ]).astype(np.float32)
        
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # Randomize initial state slightly to prevent overfitting
        self.data.qpos = self.model.qpos0 + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        self.data.qvel = self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        
        mujoco.mj_forward(self.model, self.data)
        
        target_y = self.np_random.uniform(-0.3, 0.3)
        target_z = self.np_random.uniform(0.1, 0.5)
        target_theta = self.np_random.uniform(-1.57, 1.57)
        self.target = np.array([target_y, target_z, target_theta], dtype=np.float32)
        
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.current_step = 0
        
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # We only have 2 actions from the RL policy (tendons)
        self.data.ctrl[0] = action[0]
        self.data.ctrl[1] = action[1]
        
        # Step the physics simulator
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            
        self.prev_action = action
        self.current_step += 1
        
        obs = self._get_obs()
        
        tip_state = self._get_tip_state()
        pos_dist = np.linalg.norm(tip_state[0:2] - self.target[0:2])
        
        theta_diff = tip_state[2] - self.target[2]
        theta_err = np.abs((theta_diff + np.pi) % (2 * np.pi) - np.pi)
        
        reward = -1.0 * pos_dist - 0.1 * theta_err
        
        terminated = False
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
            
        if pos_dist < 0.02 and theta_err < 0.1:
            reward += 10.0
            terminated = True
            
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, {}
        
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
