"""Evaluate a trained walking policy.

    python evaluate_walk.py --run runs/walk_v1 --episodes 5
    python evaluate_walk.py --run runs/walk_v1 --video walk.mp4
    mjpython evaluate_walk.py --run runs/walk_v1 --viewer        # macOS live viewer
"""
import argparse
import json
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from octopus_walk_env import OctopusWalkEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--model", default="final_model.zip")
    p.add_argument("--stats", default="vecnormalize.pkl")
    p.add_argument("--xml", default=None)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--video", default=None)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--no-noise", action="store_true", help="evaluate with perfect sensors and no randomization")
    args = p.parse_args()

    with open(os.path.join(args.run, "config.json")) as f:
        kw = json.load(f)["env"]
    if args.xml:
        kw["xml_path"] = args.xml
    if args.no_noise:
        kw.update(noise=0.0, randomize=False)

    env = OctopusWalkEnv(render_mode="rgb_array" if args.video else None, **kw)
    vn = VecNormalize.load(os.path.join(args.run, args.stats), DummyVecEnv([lambda: OctopusWalkEnv(**kw)]))
    vn.training = False
    model = PPO.load(os.path.join(args.run, args.model), device="cpu")

    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    frames, dists = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done, ret, rises, ups = False, 0.0, [], []
        while not done:
            a, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=not args.stochastic)
            obs, r, term, trunc, info = env.step(a[0])
            ret += r
            rises.append(info.get("rise", 0.0))
            ups.append(info.get("up", 1.0))
            done = term or trunc
            if args.video:
                frames.append(env.render())
            if viewer is not None:
                viewer.sync()
                time.sleep(env.dt)
                if not viewer.is_running():
                    return
        dists.append(info.get("dist", 0.0))
        print(f"episode {ep}: return {ret:8.2f}  distance {info.get('dist', 0.0):+.3f} m  "
              f"mean speed {info.get('dist', 0.0) / (env.step_count * env.dt):+.4f} m/s  "
              f"mean body lift {np.mean(rises) * 100:.1f} cm  min upright {np.min(ups):.2f}  "
              f"{'(tipped over)' if term else ''}")
    print(f"mean distance: {np.mean(dists):+.3f} m over {args.episodes} episodes")

    if args.video:
        import imageio
        imageio.mimsave(args.video, frames, fps=int(round(1.0 / env.dt)))
        print("wrote", args.video)
    env.close()


if __name__ == "__main__":
    main()