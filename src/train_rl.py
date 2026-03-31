import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from rl_env import TentacleEnv
from stable_baselines3.common.callbacks import CheckpointCallback

def main():
    print("Initializing TentacleEnv...")
    # Wrap the environment in a DummyVecEnv for SB3
    env = make_vec_env(TentacleEnv, n_envs=4) # Use 4 parallel environments for faster gathering
    
    print("Setting up PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        tensorboard_log="./ppo_tentacle_tensorboard/"
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./logs/",
        name_prefix="ppo_tentacle"
    )

    print("Starting training (1,000,000 timesteps for learning)...")
    model.learn(total_timesteps=1000000, progress_bar=True, callback=checkpoint_callback)
    
    print("Saving model to tentacle_ppo_model.zip...")
    model.save("tentacle_ppo_model")
    print("Training complete!")

if __name__ == "__main__":
    main()
