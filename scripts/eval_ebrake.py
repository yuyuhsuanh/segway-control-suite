"""Compare RL emergency-stop policy against the LQR baseline (theta_ref=0).

Outputs:
  ebrake_stopping_distance.png  — stopping distance vs initial speed
  ebrake_trajectory.png         — theta(t) and v(t) at v_init=1.0 m/s

Usage:
    python scripts/eval_ebrake.py [--model PATH] [--no-rl]
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from segway.envs.emergency_stop_env import EmergencyStopEnv


V_GRID = np.linspace(0.3, 1.5, 8)
DETAIL_V = 1.0  # speed for the trajectory plot


def rollout(env: EmergencyStopEnv, policy, v_init: float) -> dict:
    """Run one episode at a fixed v_init; policy(obs) -> action."""
    # Force the initial speed by overriding the RNG result.
    obs, _ = env.reset()
    # Re-reset with a forced v_init by bypassing the random draw.
    x0 = np.array([0.0, 0.0, 0.0, 0.0, v_init, 0.0, 0.0])
    env.plant.reset(x0)
    env.lqr.reset()
    env._step = 0
    env._t = 0.0
    obs = env._obs(env.plant.state)

    ts, vs, thetas, xs = [0.0], [v_init], [0.0], [0.0]
    done = False
    while not done:
        action = policy(obs)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ts.append(env._t)
        vs.append(info["v"])
        thetas.append(float(obs[1]))
        xs.append(info["x"])

    stopping_dist = abs(xs[-1] - xs[0])
    return {"t": np.array(ts), "v": np.array(vs), "theta": np.array(thetas),
            "x": np.array(xs), "stopping_dist": stopping_dist, "fell": info["fell"]}


def zero_policy(_obs):
    return np.array([0.0], dtype=np.float32)


def load_rl_policy(model_path: str):
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_ebrake.zip")
    parser.add_argument("--no-rl", action="store_true")
    args = parser.parse_args()

    env = EmergencyStopEnv()
    env.reset(seed=0)

    # --- stopping distance sweep ---
    baseline_dists, rl_dists = [], []
    for v in V_GRID:
        r = rollout(env, zero_policy, v)
        baseline_dists.append(r["stopping_dist"])

    rl_policy = None
    if not args.no_rl and Path(args.model).exists():
        rl_policy = load_rl_policy(args.model)
        for v in V_GRID:
            r = rollout(env, rl_policy, v)
            rl_dists.append(r["stopping_dist"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(V_GRID, baseline_dists, "o-", label="LQR baseline (θ_ref=0)")
    if rl_policy is not None:
        ax.plot(V_GRID, rl_dists, "s-", label="RL policy")
    ax.set_xlabel("Initial speed (m/s)")
    ax.set_ylabel("Stopping distance (m)")
    ax.set_title("Emergency stop: stopping distance vs initial speed")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("ebrake_stopping_distance.png", dpi=150)
    print("Saved ebrake_stopping_distance.png")

    # --- trajectory at DETAIL_V ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    r_base = rollout(env, zero_policy, DETAIL_V)
    ax1.plot(r_base["t"], np.degrees(r_base["theta"]), label="LQR baseline")
    ax2.plot(r_base["t"], r_base["v"], label="LQR baseline")

    if rl_policy is not None:
        r_rl = rollout(env, rl_policy, DETAIL_V)
        ax1.plot(r_rl["t"], np.degrees(r_rl["theta"]), label="RL policy")
        ax2.plot(r_rl["t"], r_rl["v"], label="RL policy")

    ax1.axhline(0, color="k", linewidth=0.5)
    ax1.set_ylabel("Pitch θ (°)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Speed (m/s)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.suptitle(f"Emergency stop trajectory at v_init = {DETAIL_V} m/s")
    fig.tight_layout()
    fig.savefig("ebrake_trajectory.png", dpi=150)
    print("Saved ebrake_trajectory.png")


if __name__ == "__main__":
    main()
