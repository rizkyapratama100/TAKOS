"""Train the octopus to walk with PPO (stable-baselines3).

    python train_walk.py --xml octopus.xml --run-name walk_v1 --timesteps 10000000 --n-envs 8 \
        --servo-rpm 40 --spool-radius 0.010
    tensorboard --logdir runs

Run `python train_walk.py ... --timesteps 100000 --run-name smoke` first and read the fps column to see how long
10M steps will take on your machine.
"""
import argparse
import json
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from octopus_walk_env import OctopusWalkEnv


def make_env(env_kwargs):
    def _init():
        return Monitor(OctopusWalkEnv(**env_kwargs))
    return _init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="octopus.xml")
    p.add_argument("--timesteps", type=int, default=10_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--episode-seconds", type=float, default=20.0)
    p.add_argument("--frame-skip", type=int, default=20)
    p.add_argument("--smooth", type=float, default=0.3)
    p.add_argument("--phase-period", type=float, default=2.0)
    p.add_argument("--servo-rpm", type=float, default=None)
    p.add_argument("--spool-radius", type=float, default=None, help="metres; needed with --servo-rpm")
    p.add_argument("--full-stroke", type=float, default=0.08, help="tendon travel for command 0->1 (m)")
    p.add_argument("--noise", type=float, default=1.0)
    p.add_argument("--latency", type=int, default=1)
    p.add_argument("--no-randomize", action="store_true")
    p.add_argument("--weight", action="append", default=[], metavar="NAME=VALUE",
                   help="override a reward weight, e.g. --weight height=20 (repeatable)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--run-name", default="walk")
    p.add_argument("--resume", default=None, help="saved .zip to continue from")
    p.add_argument("--resume-stats", default=None, help="the matching vecnormalize .pkl (needed with --resume)")
    args = p.parse_args()

    run_dir = os.path.join("runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    env_kwargs = dict(
        xml_path=args.xml, frame_skip=args.frame_skip, episode_seconds=args.episode_seconds, smooth=args.smooth,
        phase_period=args.phase_period, noise=args.noise, latency=args.latency, randomize=not args.no_randomize,
        full_stroke=args.full_stroke,
    )
    if args.servo_rpm is not None:
        if args.spool_radius is None:
            raise SystemExit("--servo-rpm needs --spool-radius (metres)")
        env_kwargs.update(servo_rpm=args.servo_rpm, spool_radius=args.spool_radius)
    if args.weight:
        env_kwargs["weights"] = {k: float(v) for k, v in (s.split("=") for s in args.weight)}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(dict(env=env_kwargs, args=vars(args)), f, indent=2)

    VecCls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    venv = VecCls([make_env(env_kwargs) for _ in range(args.n_envs)])
    if args.resume:
        if not args.resume_stats:
            raise SystemExit("--resume also needs --resume-stats (the matching vecnormalize .pkl)")
        venv = VecNormalize.load(args.resume_stats, venv)
        venv.training = True
        model = PPO.load(args.resume, env=venv, device=args.device)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        model = PPO(
            "MlpPolicy", venv,
            n_steps=512, batch_size=1024, n_epochs=10,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.001,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-1.0),
            tensorboard_log="runs", seed=args.seed, device=args.device, verbose=1,
        )

    ckpt = CheckpointCallback(
        save_freq=max(500_000 // args.n_envs, 1),
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