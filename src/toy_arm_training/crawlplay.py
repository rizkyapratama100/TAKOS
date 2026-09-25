"""
Load a trained crawl policy and watch it in MuJoCo's interactive viewer.

Usage:
    python crawl_play.py --model arm_crawl_ppo.zip --xml toy_arm_crawl.xml
"""

import argparse
import time

import gymnasium as gym
from stable_baselines3 import PPO

import crawl_env  # noqa: F401  (registers "ArmCrawl-v0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="arm_crawl_ppo.zip")
    parser.add_argument("--xml", default="toy_arm_crawl.xml")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    # render_mode="human" is what opens the MuJoCo viewer window -- it's
    # only used here, never during training, which is why train.py never
    # sets it.
    env = gym.make("ArmCrawl-v0", xml_path=args.xml, render_mode="human")
    model = PPO.load(args.model)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            env.render()  # pushes the current sim state to the open viewer window
            done = terminated or truncated
            # Real-time pacing: without this the sim runs as fast as the
            # CPU allows, which for this env's frame_skip is much faster
            # than real time and looks like a blur.
            time.sleep(env.unwrapped.dt)
        print(f"Episode {ep}: reward={ep_reward:.2f}, final x={info.get('x_position', 0):.3f}")

    env.close()


if __name__ == "__main__":
    main()