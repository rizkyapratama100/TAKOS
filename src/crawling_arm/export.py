"""Export a trained policy to a single .npz that runs on a Raspberry Pi with only numpy
(no PyTorch, stable-baselines3 or MuJoCo on the Pi).  Run this on your Mac/desktop:

    python export_policy.py --run runs/crawl_v1

Writes (into the run folder):
  policy.npz   network weights + observation normalisation + the settings needed to rebuild commands
  obs_log.npz  recorded raw observations and the original policy's actions, for checking the export
               (copy it to the Pi too and use pi_runner.py --dry-run)
"""
import argparse
import json
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from octopus_env import OctopusArmEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="final_model.zip")
    ap.add_argument("--stats", default="vecnormalize.pkl")
    ap.add_argument("--xml", default=None)
    ap.add_argument("--log-steps", type=int, default=300)
    args = ap.parse_args()

    with open(os.path.join(args.run, "config.json")) as f:
        kw = json.load(f)["env"]
    if args.xml:
        kw["xml_path"] = args.xml

    env = OctopusArmEnv(**kw)
    vn = VecNormalize.load(os.path.join(args.run, args.stats), DummyVecEnv([lambda: OctopusArmEnv(**kw)]))
    vn.training = False
    model = PPO.load(os.path.join(args.run, args.model), device="cpu")

    net = model.policy.mlp_extractor.policy_net
    linears = [m for m in net if isinstance(m, torch.nn.Linear)]
    others = [type(m).__name__ for m in net if not isinstance(m, torch.nn.Linear)]
    assert all(o == "Tanh" for o in others), f"unexpected activation(s) {others}; the Pi runner assumes tanh"

    out = {}
    for i, lin in enumerate(linears):
        out[f"W{i}"] = lin.weight.detach().numpy().astype(np.float32)
        out[f"b{i}"] = lin.bias.detach().numpy().astype(np.float32)
    out["n_hidden"] = np.array(len(linears))
    out["Wo"] = model.policy.action_net.weight.detach().numpy().astype(np.float32)
    out["bo"] = model.policy.action_net.bias.detach().numpy().astype(np.float32)
    out["obs_mean"] = vn.obs_rms.mean.astype(np.float32)
    out["obs_var"] = vn.obs_rms.var.astype(np.float32)
    out["obs_eps"] = np.array(vn.epsilon)
    out["clip_obs"] = np.array(vn.clip_obs)
    cfg = dict(control_dt=env.dt, action_mode=env.action_mode, n_modes=env.n_modes, smooth=env.smooth,
               obs_dim=int(env.observation_space.shape[0]), act_dim=int(env.action_space.shape[0]),
               phase_period=env.phase_period)
    out["config"] = np.array(json.dumps(cfg))
    path = os.path.join(args.run, "policy.npz")
    np.savez(path, **out)

    # ---- record observations and the reference actions, then check the numpy network against them
    obs_list, act_list = [], []
    obs, _ = env.reset(seed=2000)
    for _ in range(args.log_steps):
        a, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)
        obs_list.append(obs.copy())
        act_list.append(a[0].copy())
        obs, _, term, trunc, _ = env.step(a[0])
        if term or trunc:
            obs, _ = env.reset()
    obs_arr, act_arr = np.array(obs_list), np.array(act_list)
    np.savez(os.path.join(args.run, "obs_log.npz"), obs=obs_arr, actions=act_arr)

    z = np.load(path)
    err = 0.0
    for o, ref in zip(obs_arr, act_arr):
        x = np.clip((o - z["obs_mean"]) / np.sqrt(z["obs_var"] + float(z["obs_eps"])),
                    -float(z["clip_obs"]), float(z["clip_obs"]))
        for i in range(int(z["n_hidden"])):
            x = np.tanh(z[f"W{i}"] @ x + z[f"b{i}"])
        a = np.clip(z["Wo"] @ x + z["bo"], -1, 1)
        err = max(err, float(np.abs(a - ref).max()))
    print(f"wrote {path}  ({os.path.getsize(path) / 1024:.0f} kB) and obs_log.npz")
    print(f"obs_dim {cfg['obs_dim']}, act_dim {cfg['act_dim']}, control period {cfg['control_dt'] * 1000:.0f} ms")
    print(f"max |numpy action - SB3 action| over {len(obs_arr)} steps: {err:.2e}  "
          f"({'OK' if err < 1e-4 else 'TOO LARGE, do not deploy'})")


if __name__ == "__main__":
    main()