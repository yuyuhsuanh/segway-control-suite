"""Train a PPO policy for TWIP emergency braking.

Upper layer: RL outputs theta_ref_offset (backward lean = negative).
Lower layer: LQR balance controller inside EmergencyStopEnv.

Usage:
    python scripts/train_ebrake_rl.py [--steps N] [--envs N] [--out PATH]
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from segway.envs.emergency_stop_env import EmergencyStopEnv


def make_env():
    return EmergencyStopEnv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--out", type=str, default="models/ppo_ebrake_v3.zip")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    vec_env = SubprocVecEnv([make_env] * args.envs)
    eval_env = make_vec_env(make_env, n_envs=1)

    eval_cb = EvalCallback(
        eval_env,
        eval_freq=50_000 // args.envs,
        n_eval_episodes=20,
        verbose=1,
    )

    try:
        import tensorboard  # noqa: F401
        tb_log = "logs/ebrake_ppo"
    except ImportError:
        tb_log = None

    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=tb_log, device="cpu")
    model.learn(total_timesteps=args.steps, callback=eval_cb)
    model.save(str(out.with_suffix("")))
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
