from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

from toy_env import ToyArmEnv

def make_env():
    def _init():
        env = ToyArmEnv("toy_arm.xml")
        env = TimeLimit(env, max_episode_steps=300)
        return Monitor(env)
    return _init

if __name__ == "__main__":
    n_envs = 8
    vec_env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./tb_logs")
    model.learn(total_timesteps=2_000_000)
    model.save("toy_arm_ppo")