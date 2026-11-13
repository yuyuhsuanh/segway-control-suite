"""MuJoCo simulation backend.

A high-fidelity, articulated-body simulation of the same plant, used for visualization and
as an *independent physics cross-check* of the analytic model. The single motor torque
``tau`` is mapped to the two joints exactly as the analytic model assumes: a traction force
``tau/r`` on the base and a reaction torque ``-tau`` on the body. Physical parameters are
overwritten from :class:`RobotParams` at load time so both backends describe the same robot.

Requires the ``sim`` extra (``pip install -e ".[sim]"`` for MuJoCo).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..config import RobotParams, SimConfig
from ..controllers.base import Controller
from .runner import Trajectory
from .scenarios import Scenario

_ASSETS = Path(__file__).resolve().parent.parent / "models" / "assets"
ASSET_PATH = _ASSETS / "segway.xml"
# Elevated cart-pole geometry used for swing-up rendering (body can rotate without clipping
# the floor). Same dynamics, same body/joint names — only the visualization differs.
CARTPOLE_PATH = _ASSETS / "cartpole.xml"


class MuJoCoPlant:
    """Thin wrapper around a MuJoCo model whose physics match ``RobotParams``."""

    def __init__(self, params: RobotParams, xml_path: str | Path | None = None):
        import mujoco as mj

        self._mj = mj
        self.params = params
        self.model = mj.MjModel.from_xml_path(str(xml_path or ASSET_PATH))
        self.data = mj.MjData(self.model)
        self._apply_params(params)
        self.r = params.r

    def _apply_params(self, p: RobotParams) -> None:
        mj = self._mj
        m = self.model

        def body_id(name: str) -> int:
            return mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, name)

        def joint_dof(name: str) -> int:
            jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, name)
            return int(m.jnt_dofadr[jid]) if jid != -1 else -1

        base = body_id("base")
        if base != -1:
            m.body_mass[base] = p.M

        pend = body_id("pendulum")
        if pend != -1:
            m.body_mass[pend] = p.m_pend
            m.body_inertia[pend][:] = [p.I_pend, p.I_pend, p.I_pend / 20.0]
            m.body_ipos[pend][2] = p.l

        sd, hd = joint_dof("slide"), joint_dof("pend_hinge")
        if sd != -1:
            m.dof_damping[sd] = p.b_x
        if hd != -1:
            m.dof_damping[hd] = p.b_theta

    @property
    def state(self) -> np.ndarray:
        d = self.data
        return np.array([d.qpos[0], d.qvel[0], d.qpos[1], d.qvel[1]])

    def reset(self, state: np.ndarray) -> None:
        self.data = self._mj.MjData(self.model)
        self.data.qpos[0] = state[0]
        self.data.qvel[0] = state[1]
        self.data.qpos[1] = state[2]
        self.data.qvel[1] = state[3]
        self._mj.mj_forward(self.model, self.data)

    def apply(self, u: float) -> None:
        # tau -> traction force tau/r on the base, reaction torque -tau on the body.
        self.data.ctrl[0] = float(np.clip(u / self.r, -2000.0, 2000.0))
        self.data.ctrl[1] = float(np.clip(-u, -500.0, 500.0))

    def kick(self, impulse: float) -> None:
        self.data.qvel[1] += impulse

    def step(self) -> None:
        self._mj.mj_step(self.model, self.data)

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)


def simulate_mujoco(
    params: RobotParams,
    controller: Controller,
    scenario: Scenario | None = None,
    sim: SimConfig | None = None,
    step_callback: Callable[[MuJoCoPlant, float], None] | None = None,
    plant: MuJoCoPlant | None = None,
    xml_path: str | Path | None = None,
) -> Trajectory:
    """Closed-loop rollout in MuJoCo. Returns a :class:`Trajectory` (same type as the
    analytic backend), so metrics, plotting, and the benchmark all work unchanged.

    ``step_callback(plant, t)`` is invoked every step (used by the renderer to grab frames).
    A pre-built ``plant`` may be supplied so a renderer can share the exact same model;
    otherwise ``xml_path`` selects the model geometry (e.g. the cart-pole for swing-up).
    """
    scenario = scenario or Scenario()
    sim = sim or SimConfig()
    controller.reset()

    if plant is None:
        plant = MuJoCoPlant(params, xml_path=xml_path)
    plant.model.opt.timestep = sim.dt
    plant.reset(scenario.initial_state())

    dt = sim.dt
    n_steps = int(round(sim.duration / dt))
    control_dt = sim.control_dt

    ts: list[float] = []
    xs: list[np.ndarray] = []
    us: list[float] = []
    u = 0.0
    last_ctrl_t = -np.inf
    fired: set[int] = set()
    fell = False

    for i in range(n_steps + 1):
        t = i * dt
        state = plant.state

        for j, d in enumerate(scenario.disturbances):
            if j not in fired and abs(t - d.time) < dt / 2:
                plant.kick(d.impulse)
                fired.add(j)
                state = plant.state

        if abs(state[2]) > sim.fall_angle:
            fell = True
            break

        if control_dt is None or (t - last_ctrl_t) >= control_dt - 1e-12:
            u = controller.compute(state - scenario.reference_at(t), t)
            last_ctrl_t = t
        plant.apply(u)

        if i % sim.record_every == 0:
            ts.append(t)
            xs.append(state.copy())
            us.append(u)
        if step_callback is not None:
            step_callback(plant, t)

        if i < n_steps:
            plant.step()

    if not ts:
        ts.append(0.0)
        xs.append(scenario.initial_state())
        us.append(0.0)

    return Trajectory(
        t=np.asarray(ts),
        states=np.asarray(xs),
        controls=np.asarray(us),
        fell=fell,
        params=params,
        scenario=scenario,
        controller_name=getattr(controller, "name", "unknown"),
    )
