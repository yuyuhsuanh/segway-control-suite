"""Velocity-tracking tuning objective for the slip study.

Step profile (from prompt.md):
  t = 0–2 s : v_des = 0  (stabilise)
  t = 2–5 s : v_des = 2 m/s  (accelerate + hold)
  t = 5–7 s : v_des = 0  (decelerate)

Cost = RMSE(v − v_des) over the full run.  Fall → FALL_PENALTY.
"""
from __future__ import annotations

import numpy as np

from ..config import RobotParams, SimConfig
from ..controllers import build_controller
from ..sim import Scenario, simulate

V_MAX: float = 2.0   # target speed [m/s]
T_UP: float = 2.0    # step-up time [s]
T_DOWN: float = 5.0  # step-down time [s]
T_END: float = 7.0   # episode length [s]
U_MAX: float = 80.0  # combined wheel torque limit [Nm]  (40 Nm/wheel × 2)
FALL_PENALTY: float = 1e6

# WF_TRON1B equivalent TWIP parameters at standing pose.
# Legs lumped into m_base; main body (9.595 kg) is the pendulum at l=0.616 m.
# ponytail: leg joints ignored; refine if balance tuning fails badly
TRON1_PARAMS = RobotParams(
    m_base=11.215,   # leg mass [kg]
    m_wheel=1.08,    # each wheel [kg]
    m_pend=9.595,    # torso/body [kg]
    l=0.616,         # body COM above wheel axle [m]
    r=0.127,         # wheel radius [m]
    I_pend=0.155,    # body rotational inertia [kg·m²]
)


class _Clipped:
    """Wraps any controller and clips its output to ±u_max."""

    def __init__(self, ctrl, u_max: float) -> None:
        self._ctrl = ctrl
        self._u_max = float(u_max)

    def compute(self, err: np.ndarray, t: float) -> float:
        return float(np.clip(self._ctrl.compute(err, t), -self._u_max, self._u_max))

    def reset(self) -> None:
        self._ctrl.reset()


def _v_des(t: float) -> float:
    return V_MAX if T_UP <= t < T_DOWN else 0.0


def velocity_step_scenario() -> Scenario:
    """Scenario encoding the step velocity command profile."""
    return Scenario(
        name="velocity_step",
        initial_tilt=0.0,
        reference_fn=lambda t: np.array([0.0, _v_des(t), 0.0, 0.0]),
    )


def velocity_tracking_cost(
    controller_name: str,
    kwargs: dict,
    params: RobotParams | None = None,
    u_max: float = U_MAX,
) -> float:
    """Scalar cost for Optuna: RMSE velocity error (lower is better)."""
    params = params or TRON1_PARAMS
    try:
        ctrl = _Clipped(build_controller(controller_name, params, **kwargs), u_max)
    except Exception:
        return FALL_PENALTY
    traj = simulate(params, ctrl, velocity_step_scenario(), SimConfig(duration=T_END))
    if traj.fell:
        return FALL_PENALTY
    v = traj.states[:, 1]
    v_des_arr = np.array([_v_des(t) for t in traj.t])
    return float(np.sqrt(np.mean((v - v_des_arr) ** 2)))


# Velocity-task-specific search bounds.
# LQR/MPC: q0 (position penalty) fixed to 0 — robot is allowed to drift forward.
# cascaded_pid: kx fixed to 0 — outer loop uses velocity error only, no position pull-back.
VELOCITY_BOUNDS: dict[str, list[tuple]] = {
    "lqr": [("q1", 0.1, 500, True), ("q2", 1.0, 2000, True),
            ("q3", 0.1, 200, True), ("R", 1e-3, 10, True)],
    "cascaded_pid": [("kp", 40, 400, False), ("kd", 5, 100, False),
                     ("kv", -3.0, -0.05, False)],
}
VELOCITY_BOUNDS["mpc"] = VELOCITY_BOUNDS["lqr"]


def velocity_flat_to_kwargs(name: str, flat: dict) -> dict:
    if name in ("lqr", "mpc"):
        return {"Q": [0.0, flat["q1"], flat["q2"], flat["q3"]], "R": flat["R"]}
    if name == "cascaded_pid":
        return {"kp": flat["kp"], "kd": flat["kd"], "kx": 0.0, "kv": flat["kv"]}
    raise ValueError(f"no velocity bounds for {name!r}")


def transition_times(traj_t: np.ndarray, traj_v: np.ndarray, tol: float = 0.1) -> tuple[float, float]:
    """Return (T_accel, T_decel): seconds to reach V_MAX and to stop, after each step command.

    Returns inf if the threshold is never crossed.
    """
    # T_accel: first t >= T_UP where v is within tol of V_MAX
    mask_up = traj_t >= T_UP
    hit_up = np.where(mask_up & (np.abs(traj_v - V_MAX) < tol))[0]
    t_accel = float(traj_t[hit_up[0]] - T_UP) if len(hit_up) else float("inf")

    # T_decel: first t >= T_DOWN where |v| < tol
    mask_dn = traj_t >= T_DOWN
    hit_dn = np.where(mask_dn & (np.abs(traj_v) < tol))[0]
    t_decel = float(traj_t[hit_dn[0]] - T_DOWN) if len(hit_dn) else float("inf")

    return t_accel, t_decel
