from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from gymnasium.wrappers import TimeLimit

from toy_arm_training.toy_env import ToyArmEnv


def make_env():
    def _init():
        # No render_mode here -- rendering has no business running inside
        # 8 parallel training subprocesses, and isn't used during training
        # anyway. Use a separate single-env script (e.g. play.py) to watch
        # the trained policy with render_mode="human".
        env = ToyArmEnv("toy_arm.xml", render_mode=None)
        env = TimeLimit(env, max_episode_steps=300)
        return Monitor(env)

    return _init


if __name__ == "__main__":
    n_envs = 8

    train_env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    # Normalize observations (and reward) using a running mean/var. This
    # matters a lot here: qpos/qvel/positions are all on different scales,
    # and PPO trains much more reliably when inputs are roughly unit-scale.
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Separate eval env, wrapped with its own VecNormalize in inference mode
    # (training=False) so obs are normalized consistently at eval time
    # without polluting the training env's running statistics.
    eval_env = DummyVecEnv([make_env()])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    # Note: EvalCallback automatically syncs train_env's running obs stats
    # into eval_env before each evaluation (via sync_envs_normalization),
    # so eval_env's own stats above are just placeholders until then.

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log="./tb_logs",
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./best_model",
        log_path="./eval_logs",
        eval_freq=max(20_000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path="./checkpoints",
        name_prefix="toy_arm",
    )

    model.learn(total_timesteps=2_000_000, callback=[eval_cb, ckpt_cb], progress_bar=True)

    model.save("toy_arm_ppo")
    # VecNormalize's running stats are NOT stored inside the model .zip --
    # you must save them separately and reload them at inference time,
    # otherwise the policy will see un-normalized observations it was
    # never trained on and behave badly (this matters a lot for the Pi).
    train_env.save("vecnormalize_stats.pkl")

    print("\nWatch training progress with: tensorboard --logdir ./tb_logs")
    print("Best model (by eval reward) saved to: ./best_model/best_model.zip")