"""Compare RL emergency-stop policy against the LQR baseline (theta_ref=0).

Outputs:
  ebrake_stopping_distance.png  — stopping distance vs initial speed
  ebrake_trajectory.png         — theta(t), v(t), and friction utilization at v_init=1.0 m/s
  ebrake_v{v}_rl.gif            — MuJoCo render of RL policy  (--gif)
  ebrake_v{v}_baseline.gif      — MuJoCo render of LQR baseline (--gif)

Usage:
    python scripts/eval_ebrake.py [--model PATH] [--no-rl] [--gif]
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mujoco

from segway.envs.emergency_stop_env import EmergencyStopEnv


def _cone_utilization(env: EmergencyStopEnv) -> float:
    """Return mean friction cone utilization across all wheel contacts after a step."""
    d = env.plant.data
    m = env.plant.model
    mu = m.geom_friction[0, 0]
    ratios = []
    for i in range(d.ncon):
        force = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, force)
        fn = force[0]
        if fn > 1e-3:
            ft = np.sqrt(force[1] ** 2 + force[2] ** 2)
            ratios.append(ft / (mu * fn))
    return float(np.mean(ratios)) if ratios else 0.0


V_GRID = np.linspace(0.3, 1.5, 8)
DETAIL_V = 1.0  # speed for the trajectory plot
GIF_VS = list(V_GRID)  # GIF for every speed in the sweep


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

    ts, vs, thetas, xs, cones = [0.0], [v_init], [0.0], [0.0], [0.0]
    done = False
    while not done:
        action = policy(obs)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ts.append(env._t)
        vs.append(info["v"])
        thetas.append(float(obs[1]))
        xs.append(info["x"])
        cones.append(_cone_utilization(env))

    stopping_dist = abs(xs[-1] - xs[0])
    return {"t": np.array(ts), "v": np.array(vs), "theta": np.array(thetas),
            "x": np.array(xs), "cone": np.array(cones),
            "stopping_dist": stopping_dist, "fell": info["fell"]}


def zero_policy(_obs):
    return np.array([0.0], dtype=np.float32)


def load_rl_policy(model_path: str):
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return policy


def render_rollout(env: EmergencyStopEnv, policy, v_init: float,
                   fps: int = 20, width: int = 640, height: int = 480) -> list:
    """Run a rollout and return rendered frames as a list of uint8 arrays."""
    import mujoco
    renderer = mujoco.Renderer(env.plant.model, height=height, width=width)
    frame_every = max(1, round(1.0 / (fps * env.dt_phys)))

    env.reset(seed=0)
    x0 = np.array([0.0, 0.0, 0.0, 0.0, v_init, 0.0, 0.0])
    env.plant.reset(x0)
    env.lqr.reset()
    env._step = 0
    env._t = 0.0
    obs = env._obs(env.plant.state)

    cam = mujoco.MjvCamera()
    cam.distance = 2.5
    cam.azimuth = 90.0   # side view
    cam.elevation = -15.0

    frames, phys_count = [], 0
    done = False
    while not done:
        action = policy(obs)
        theta_ref = float(action[0])
        for _ in range(env.substeps):
            state = env.plant.state
            _, _, _, theta, v, _, theta_dot = state
            err = np.array([0.0, v, theta - theta_ref, theta_dot])
            tau_sum = env.lqr.compute(err, env._t)
            half = tau_sum / 2.0
            env.plant.apply((half, half))
            env.plant.step()
            env._t += env.dt_phys
            if phys_count % frame_every == 0:
                cam.lookat[:] = env.plant.data.xpos[env.plant._base]
                renderer.update_scene(env.plant.data, camera=cam)
                frames.append(renderer.render().copy())
            phys_count += 1
        env._step += 1
        state = env.plant.state
        x, _, _, theta, v, _, _ = state
        obs = env._obs(state)
        done = abs(theta) > env.fall_angle or env._step >= env.max_steps
        if abs(v) < 0.005 and env._step > 10:
            # hold last frame for 0.5 s so the GIF doesn't cut abruptly
            for _ in range(fps // 2):
                frames.append(frames[-1])
            break
    renderer.close()
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_ebrake.zip")
    parser.add_argument("--no-rl", action="store_true")
    parser.add_argument("--gif", action="store_true", help="render MuJoCo GIFs")
    parser.add_argument("--out-dir", type=str, default=".")
    args = parser.parse_args()
    out = Path(args.out_dir)

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
    fig.savefig(out / "ebrake_stopping_distance.png", dpi=150)
    print(f"Saved {out / 'ebrake_stopping_distance.png'}")

    # --- trajectory at DETAIL_V ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    r_base = rollout(env, zero_policy, DETAIL_V)
    ax1.plot(r_base["t"], np.degrees(r_base["theta"]), label="LQR baseline")
    ax2.plot(r_base["t"], r_base["v"], label="LQR baseline")
    ax3.plot(r_base["t"], r_base["cone"], label="LQR baseline")

    if rl_policy is not None:
        r_rl = rollout(env, rl_policy, DETAIL_V)
        ax1.plot(r_rl["t"], np.degrees(r_rl["theta"]), label="RL policy")
        ax2.plot(r_rl["t"], r_rl["v"], label="RL policy")
        ax3.plot(r_rl["t"], r_rl["cone"], label="RL policy")

    ax1.axhline(0, color="k", linewidth=0.5)
    ax1.set_ylabel("Pitch θ (°)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.set_ylabel("Velocity (m/s)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax3.axhline(1.0, color="k", linewidth=1.0, linestyle="--", label="Friction limit (ft = μ·fn)")
    ax3.set_ylabel("Cone utilization ft/(μ·fn)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylim(0, 1.3)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig.suptitle(f"Emergency stop trajectory at v_init = {DETAIL_V} m/s")
    fig.tight_layout()
    fig.savefig(out / "ebrake_trajectory.png", dpi=150)
    print(f"Saved {out / 'ebrake_trajectory.png'}")

    # --- GIFs ---
    if args.gif:
        import imageio.v2 as iio
        for v in GIF_VS:
            for label, pol in [("baseline", zero_policy),
                                *([("rl", rl_policy)] if rl_policy is not None else [])]:
                frames = render_rollout(env, pol, v)
                path = out / f"ebrake_v{v:.1f}_{label}.gif"
                iio.mimwrite(str(path), frames, fps=20, loop=0)
                print(f"Saved {path}")


if __name__ == "__main__":
    main()
