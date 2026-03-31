import time
import numpy as np
from stable_baselines3 import PPO
from rl_env import TentacleEnv

def main():
    print("Loading environment in human render mode...")
    env = TentacleEnv(render_mode="human")
    
    print("Loading trained PPO model...")
    try:
        model = PPO.load("tentacle_ppo_model")
    except FileNotFoundError:
        print("Error: Model file 'tentacle_ppo_model.zip' not found. Please run train_rl.py first.")
        return
        
    obs, info = env.reset()
    print("Starting evaluation... (Press Ctrl+C to exit)")
    
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Calculate real-time sleep to approximate simulation speed
            dt = env.model.opt.timestep * env.frame_skip
            time.sleep(dt)
            
            if terminated or truncated:
                print("Episode ended. Resetting...")
                obs, info = env.reset()
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")

    env.close()

if __name__ == "__main__":
    main()
