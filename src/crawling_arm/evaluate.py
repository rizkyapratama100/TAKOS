"""Evaluate a trained crawl policy; optionally record a video or open the interactive viewer.

    python evaluate.py --run runs/crawl_v1 --episodes 5
    python evaluate.py --run runs/crawl_v1 --video crawl.mp4        # headless: MUJOCO_GL=egl (or osmesa)
    python evaluate.py --run runs/crawl_v1 --viewer                 # needs a display (mjpython on macOS)
"""
import argparse
import json
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from octopus_env import OctopusArmEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--model", default="final_model.zip", help="file inside the run dir (or checkpoints/...)")
    p.add_argument("--stats", default="vecnormalize.pkl")
    p.add_argument("--xml", default=None, help="override xml path stored in config.json")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--video", default=None)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--stochastic", action="store_true")
    args = p.parse_args()

    with open(os.path.join(args.run, "config.json")) as f:
        env_kwargs = json.load(f)["env"]
    if args.xml:
        env_kwargs["xml_path"] = args.xml

    env = OctopusArmEnv(render_mode="rgb_array" if args.video else None, **env_kwargs)
    vn = VecNormalize.load(os.path.join(args.run, args.stats),
                           DummyVecEnv([lambda: OctopusArmEnv(**env_kwargs)]))
    vn.training = False
    model = PPO.load(os.path.join(args.run, args.model), device="cpu")

    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    frames, dists = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done, ret = False, 0.0
        while not done:
            a, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=not args.stochastic)
            obs, r, term, trunc, info = env.step(a[0])
            ret += r
            done = term or trunc
            if args.video:
                frames.append(env.render())
            if viewer is not None:
                viewer.sync()
                time.sleep(env.dt)
                if not viewer.is_running():
                    return
        dists.append(info["dist"])
        print(f"episode {ep}: return {ret:8.2f}  distance {info['dist']:+.3f} m  "
              f"mean speed {info['dist'] / (env.step_count * env.dt):+.3f} m/s  "
              f"{'(rolled over)' if term else ''}")
    print(f"mean distance: {np.mean(dists):+.3f} m over {args.episodes} episodes")

    if args.video:
        import imageio
        imageio.mimsave(args.video, frames, fps=int(round(1.0 / env.dt)))
        print("wrote", args.video)
    env.close()


if __name__ == "__main__":
    main()
