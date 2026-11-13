"""Velocity-tracking baseline benchmark (model-based controllers).

Tunes cascaded_pid, lqr, and mpc on the WF_TRON1B equivalent TWIP parameters
using the step velocity profile (0 → 2 m/s → 0).  Reports T_accel, T_decel,
RMSE velocity error, and saves tuned gains + per-controller plots.

Usage:
    conda run -n segway python benchmarks/run_velocity.py
    python benchmarks/run_velocity.py --quick      # fewer trials
    python benchmarks/run_velocity.py --no-mpc     # skip MPC (needs cvxpy)
"""

from __future__ import annotations

import sys
from pathlib import Path

_src = Path(__file__).resolve().parents[1] / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import argparse
import json

import numpy as np

from segway.config import SimConfig
from segway.controllers import build_controller
from segway.sim import simulate
from segway.tuning.velocity_objective import (
    TRON1_PARAMS,
    T_END,
    U_MAX,
    VELOCITY_BOUNDS,
    _Clipped,
    transition_times,
    velocity_flat_to_kwargs,
    velocity_step_scenario,
    velocity_tracking_cost,
)

RESULTS = Path("benchmarks/results")
CONTROLLERS = ["cascaded_pid", "lqr", "mpc"]


def _tune(name: str, n_trials: int) -> dict:
    """Run Optuna TPE on velocity_tracking_cost; return best kwargs."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    bounds = VELOCITY_BOUNDS[name]

    def objective(trial):
        flat = {k: trial.suggest_float(k, lo, hi, log=log) for k, lo, hi, log in bounds}
        kwargs = velocity_flat_to_kwargs(name, flat)
        return velocity_tracking_cost(name, kwargs)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return velocity_flat_to_kwargs(name, study.best_params)


def _run_and_score(name: str, kwargs: dict):
    """Simulate and return (traj, t_accel, t_decel, rmse)."""
    ctrl = _Clipped(build_controller(name, TRON1_PARAMS, **kwargs), U_MAX)
    scen = velocity_step_scenario()
    traj = simulate(TRON1_PARAMS, ctrl, scen, SimConfig(duration=T_END))
    v = traj.states[:, 1]
    from segway.tuning.velocity_objective import _v_des, V_MAX
    v_des = np.array([_v_des(t) for t in traj.t])
    rmse = float(np.sqrt(np.mean((v - v_des) ** 2)))
    t_accel, t_decel = transition_times(traj.t, v)
    return traj, t_accel, t_decel, rmse


def _plot_single(name: str, traj, t_accel: float, t_decel: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from segway.tuning.velocity_objective import T_DOWN, T_UP, V_MAX, _v_des

    v_des = np.array([_v_des(t) for t in traj.t])
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    axes[0].plot(traj.t, traj.states[:, 1], label="v actual")
    axes[0].plot(traj.t, v_des, "k--", label="v_des")
    axes[0].axvline(T_UP, color="gray", lw=0.8, ls=":")
    axes[0].axvline(T_DOWN, color="gray", lw=0.8, ls=":")
    if not np.isinf(t_accel):
        axes[0].axvline(T_UP + t_accel, color="C0", lw=1.0, ls="--",
                        label=f"T_accel={t_accel:.2f}s")
    if not np.isinf(t_decel):
        axes[0].axvline(T_DOWN + t_decel, color="C1", lw=1.0, ls="--",
                        label=f"T_decel={t_decel:.2f}s")
    axes[0].set_ylabel("velocity [m/s]")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"{name} — velocity tracking")
    axes[1].plot(traj.t, np.degrees(traj.states[:, 2]))
    axes[1].set_ylabel("tilt [°]")
    axes[1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(RESULTS / f"velocity_{name}.png", dpi=120)
    plt.close(fig)


def _plot_comparison(records: list[tuple]) -> None:
    """Overlay all controllers' v-t traces on one figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from segway.tuning.velocity_objective import T_DOWN, T_UP, _v_des

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    t_ref = records[0][1].t
    v_des = np.array([_v_des(t) for t in t_ref])

    axes[0].plot(t_ref, v_des, "k--", lw=1.5, label="v_des", zorder=5)
    axes[0].axvline(T_UP, color="gray", lw=0.8, ls=":")
    axes[0].axvline(T_DOWN, color="gray", lw=0.8, ls=":")
    for i, (name, traj, t_accel, t_decel, _fell) in enumerate(records):
        axes[0].plot(traj.t, traj.states[:, 1], label=name)
        axes[1].plot(traj.t, np.degrees(traj.states[:, 2]), label=name)

    axes[0].set_ylabel("velocity [m/s]")
    axes[0].legend(fontsize=9)
    axes[0].set_title("All controllers — velocity tracking comparison")
    axes[1].set_ylabel("tilt [°]")
    axes[1].set_xlabel("time [s]")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "velocity_comparison.png", dpi=120)
    plt.close(fig)


def _record_gif(name: str, kwargs: dict) -> Path | None:
    """Render a MuJoCo GIF.  Returns None if mujoco/imageio unavailable."""
    try:
        from segway.viz import render_rollout
    except ImportError:
        return None
    out = RESULTS / f"velocity_{name}.gif"
    try:
        render_rollout(
            TRON1_PARAMS,
            _Clipped(build_controller(name, TRON1_PARAMS, **kwargs), U_MAX),
            velocity_step_scenario(),
            sim=SimConfig(duration=T_END),
            path=out,
            fps=30,
        )
    except Exception as exc:
        print(f"       [gif] skipped ({exc})")
        return None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer trials (faster)")
    ap.add_argument("--no-mpc", action="store_true", help="skip MPC (needs cvxpy)")
    ap.add_argument("--trials", type=int, default=None)
    args = ap.parse_args(argv)
    trials = args.trials or (20 if args.quick else 80)

    RESULTS.mkdir(parents=True, exist_ok=True)

    names = [n for n in CONTROLLERS if not (args.no_mpc and n == "mpc")]
    rows = []       # (name, t_accel, t_decel, rmse, fell)
    records = []    # (name, traj, t_accel, t_decel, fell) — for comparison plot
    gains: dict[str, dict] = {}

    for name in names:
        if name == "mpc":
            try:
                import cvxpy  # noqa: F401
            except ImportError:
                print(f"[skip] {name} — cvxpy not installed")
                continue

        print(f"[tune] {name} ({trials} trials) ...", flush=True)
        best_kwargs = _tune(name, trials)
        gains[name] = best_kwargs

        traj, t_accel, t_decel, rmse = _run_and_score(name, best_kwargs)
        rows.append((name, t_accel, t_decel, rmse, traj.fell))
        records.append((name, traj, t_accel, t_decel, traj.fell))
        status = "FELL" if traj.fell else "ok"
        print(f"       T_accel={t_accel:.3f}s  T_decel={t_decel:.3f}s  "
              f"RMSE={rmse:.4f} m/s  [{status}]", flush=True)

        if not traj.fell:
            _plot_single(name, traj, t_accel, t_decel)
            print(f"       [gif] recording {name} ...", flush=True)
            gif = _record_gif(name, best_kwargs)
            if gif:
                print(f"       [gif] saved → {gif}", flush=True)

    if records:
        _plot_comparison(records)
        print(f"Comparison plot → {RESULTS}/velocity_comparison.png")

    (RESULTS / "velocity_gains.json").write_text(json.dumps(gains, indent=2))

    # Summary table
    print("\n| controller | T_accel [s] | T_decel [s] | RMSE [m/s] |")
    print("|---|---|---|---|")
    md_rows = ["| controller | T_accel [s] | T_decel [s] | RMSE [m/s] |",
               "|---|---|---|---|"]
    for name, ta, td, rmse, fell in rows:
        ta_s = f"{ta:.3f}" if not np.isinf(ta) else "∞ (failed)"
        td_s = f"{td:.3f}" if not np.isinf(td) else "∞ (failed)"
        flag = " ⚠️ FELL" if fell else ""
        print(f"| {name} | {ta_s} | {td_s} | {rmse:.4f}{flag} |")
        md_rows.append(f"| {name} | {ta_s} | {td_s} | {rmse:.4f}{flag} |")

    report = "\n".join([
        "# Velocity Baseline Report",
        "",
        "WF_TRON1B equivalent TWIP, step profile 0→2→0 m/s, u_max=80 Nm.",
        "",
        *md_rows,
        "",
        "## Tuned gains",
        "",
        "```json",
        json.dumps(gains, indent=2),
        "```",
    ])
    (RESULTS / "velocity_report.md").write_text(report)
    print(f"\nSaved to {RESULTS}/velocity_report.md and velocity_gains.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
