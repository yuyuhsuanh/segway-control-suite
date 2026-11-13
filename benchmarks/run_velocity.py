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
from segway.tuning.objective import flat_to_kwargs, search_bounds
from segway.tuning.velocity_objective import (
    TRON1_PARAMS,
    T_END,
    U_MAX,
    _Clipped,
    transition_times,
    velocity_step_scenario,
    velocity_tracking_cost,
)

RESULTS = Path("benchmarks/results")
CONTROLLERS = ["cascaded_pid", "lqr", "mpc"]


def _tune(name: str, n_trials: int) -> dict:
    """Run Optuna TPE on velocity_tracking_cost; return best kwargs."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    bounds = search_bounds.get(name) or search_bounds["lqr"]  # mpc shares lqr bounds

    def objective(trial):
        flat = {k: trial.suggest_float(k, lo, hi, log=log) for k, lo, hi, log in bounds}
        kwargs = flat_to_kwargs(name, flat)
        return velocity_tracking_cost(name, kwargs)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return flat_to_kwargs(name, study.best_params)


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


def _plot(name: str, traj, t_accel: float, t_decel: float) -> Path:
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
    t_a_abs = T_UP + t_accel if not np.isinf(t_accel) else None
    t_d_abs = T_DOWN + t_decel if not np.isinf(t_decel) else None
    if t_a_abs:
        axes[0].axvline(t_a_abs, color="C0", lw=1.0, ls="--",
                        label=f"T_accel={t_accel:.2f}s")
    if t_d_abs:
        axes[0].axvline(t_d_abs, color="C1", lw=1.0, ls="--",
                        label=f"T_decel={t_decel:.2f}s")
    axes[0].set_ylabel("velocity [m/s]")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"{name} — velocity tracking")

    axes[1].plot(traj.t, np.degrees(traj.states[:, 2]))
    axes[1].set_ylabel("tilt [°]")
    axes[1].set_xlabel("time [s]")

    fig.tight_layout()
    out = RESULTS / f"velocity_{name}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
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
    rows = []
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
        status = "FELL" if traj.fell else "ok"
        print(f"       T_accel={t_accel:.3f}s  T_decel={t_decel:.3f}s  "
              f"RMSE={rmse:.4f} m/s  [{status}]", flush=True)

        if not traj.fell:
            _plot(name, traj, t_accel, t_decel)

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
