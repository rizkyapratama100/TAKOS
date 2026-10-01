"""Train the octopus arm to crawl with PPO (stable-baselines3).

    python train.py --xml spiral_robot.xml --timesteps 3000000 --n-envs 8 --run-name crawl_v1
    tensorboard --logdir runs
"""
import argparse
import json
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from octopus_env import OctopusArmEnv


def make_env(env_kwargs):
    def _init():
        return Monitor(OctopusArmEnv(**env_kwargs))
    return _init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="spiral_robot.xml")
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--action-mode", choices=["reduced", "full"], default="reduced")
    p.add_argument("--n-modes", type=int, default=4)
    p.add_argument("--smooth", type=float, default=0.5)
    p.add_argument("--episode-seconds", type=float, default=8.0)
    p.add_argument("--frame-skip", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--run-name", default="crawl")
    p.add_argument("--resume", default=None, help="path to a saved .zip to continue from")
    args = p.parse_args()

    run_dir = os.path.join("runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    env_kwargs = dict(
        xml_path=args.xml, frame_skip=args.frame_skip, episode_seconds=args.episode_seconds,
        action_mode=args.action_mode, n_modes=args.n_modes, smooth=args.smooth,
    )
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(dict(env=env_kwargs, args=vars(args)), f, indent=2)

    VecCls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    venv = VecCls([make_env(env_kwargs) for _ in range(args.n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    if args.resume:
        model = PPO.load(args.resume, env=venv, device=args.device)
    else:
        model = PPO(
            "MlpPolicy", venv,
            n_steps=512, batch_size=1024, n_epochs=10,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.001,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-1.0),
            tensorboard_log="runs", seed=args.seed, device=args.device, verbose=1,
        )

    ckpt = CheckpointCallback(
        save_freq=max(200_000 // args.n_envs, 1),
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="ppo",
        save_vecnormalize=True,
    )
    try:
        model.learn(args.timesteps, callback=ckpt, tb_log_name=args.run_name,
                    reset_num_timesteps=args.resume is None)
    finally:
        model.save(os.path.join(run_dir, "final_model"))
        venv.save(os.path.join(run_dir, "vecnormalize.pkl"))
        venv.close()
        print(f"Saved to {run_dir}/ (final_model.zip, vecnormalize.pkl)")


if __name__ == "__main__":
    main()
