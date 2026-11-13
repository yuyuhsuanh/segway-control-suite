"""MuJoCo backend for the planar TWIP — the real rolling-contact wheeled robot.

Where ``mujoco_backend.py`` is a contact-free cross-check of the 1-D plant, this is the
genuine two-wheeled robot: a free-floating chassis balancing on two driven wheels that roll on
the ground through frictional contact. It is used (a) for true 3-D navigation visualization and
(b) as an independent physics check that the analytic :func:`segway.models.twip.twip_dynamics`
decoupling holds up against full contact dynamics.

The same :class:`~segway.navigation.control.TWIPController` drives both backends: it returns
``(tau_L, tau_R)`` from the planar state, and here those are applied directly as wheel-hinge
motor torques — traction and the chassis reaction then emerge from contact physics.

Requires the ``sim`` extra (MuJoCo).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..config import SimConfig, TWIPParams
from ..navigation.control import TWIPController
from ..navigation.sim import TWIPTrajectory

_ASSETS = Path(__file__).resolve().parent.parent / "models" / "assets"
TWIP_PATH = _ASSETS / "twip.xml"

CommandFn = Callable[[float, np.ndarray], "tuple[float, float]"]
StopFn = Callable[[np.ndarray], bool]


class MuJoCoTWIP:
    """Free-floating two-wheeled inverted pendulum in MuJoCo, matched to ``TWIPParams``."""

    def __init__(self, params: TWIPParams, xml_path: str | Path | None = None,
                 xml_string: str | None = None):
        import mujoco as mj

        self._mj = mj
        self.params = params
        if xml_string is not None:
            self.model = mj.MjModel.from_xml_string(xml_string)
        else:
            self.model = mj.MjModel.from_xml_path(str(xml_path or TWIP_PATH))
        self.data = mj.MjData(self.model)
        self._ids()
        self._apply_params(params)
        mj.mj_forward(self.model, self.data)

    # --- setup ----------------------------------------------------------------
    def _ids(self) -> None:
        mj, m = self._mj, self.model
        b = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        g = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
        j = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
        self._base, self._pole = b("base"), b("pole")
        self._lw, self._rw = b("left_wheel"), b("right_wheel")
        self._lh, self._rh = j("left_hinge"), j("right_hinge")
        self._lwg = g("left_wheel") if g("left_wheel") != -1 else None
        # geom ids by index on the wheel bodies (one cylinder geom each)
        self._lwgeom = int(np.where(m.geom_bodyid == self._lw)[0][0])
        self._rwgeom = int(np.where(m.geom_bodyid == self._rw)[0][0])

    def _apply_params(self, p: TWIPParams) -> None:
        m = self.model
        base, pole = p.base, p
        r = base.r
        m.body_mass[self._base] = base.m_base
        m.body_mass[self._pole] = base.m_pend
        m.body_ipos[self._pole][2] = base.l
        m.body_inertia[self._pole][:] = [base.I_pend, base.I_pend, base.I_pend / 20.0]
        m.body_mass[self._lw] = base.m_wheel
        m.body_mass[self._rw] = base.m_wheel
        # geometry: wheel radius + track + axle height
        m.geom_size[self._lwgeom][0] = r
        m.geom_size[self._rwgeom][0] = r
        # Update broad-phase bounding radius after size change (MuJoCo caches this).
        hl = float(m.geom_size[self._lwgeom][1])
        rbound = float(np.sqrt(r ** 2 + hl ** 2))
        m.geom_rbound[self._lwgeom] = rbound
        m.geom_rbound[self._rwgeom] = rbound
        m.body_pos[self._lw] = [0.0, pole.track / 2.0, 0.0]
        m.body_pos[self._rw] = [0.0, -pole.track / 2.0, 0.0]
        m.body_pos[self._base] = [0.0, 0.0, r]
        # rolling resistance ~ axle viscous damping
        m.dof_damping[m.jnt_dofadr[self._lh]] = base.b_x
        m.dof_damping[m.jnt_dofadr[self._rh]] = base.b_x

    # --- state <-> mujoco -----------------------------------------------------
    @property
    def state(self) -> np.ndarray:
        """Planar state ``[x, y, psi, theta, v, psi_dot, theta_dot]`` from the chassis."""
        mj, m, d = self._mj, self.model, self.data
        bid = self._base
        x, y = float(d.xpos[bid][0]), float(d.xpos[bid][1])
        R = d.xmat[bid].reshape(3, 3)
        psi = float(np.arctan2(R[1, 0], R[0, 0]))
        theta = float(np.arctan2(-R[2, 0], R[2, 2]))   # pitch about the wheel axle

        vel = np.zeros(6)
        mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, bid, vel, 0)  # world frame [ang; lin]
        w_world, v_world = vel[:3], vel[3:]
        v = float(v_world[0] * np.cos(psi) + v_world[1] * np.sin(psi))
        psi_dot = float(w_world[2])
        theta_dot = float(np.dot(w_world, R[:, 1]))    # pitch rate about body y-axis
        return np.array([x, y, psi, theta, v, psi_dot, theta_dot])

    def reset(self, x0: np.ndarray | None = None) -> None:
        mj, m = self._mj, self.model
        self.data = mj.MjData(m)
        d = self.data
        s = np.zeros(7) if x0 is None else np.asarray(x0, dtype=float)
        x, y, psi, theta, v, psi_dot, theta_dot = s
        d.qpos[0:3] = [x, y, self.params.base.r]
        # orientation = yaw(psi) then pitch(theta): R = Rz(psi) @ Ry(theta)
        R = _rotz(psi) @ _roty(theta)
        quat = np.zeros(4)
        mj.mju_mat2Quat(quat, R.reshape(9))
        d.qpos[3:7] = quat
        # free-joint velocity: linear in world frame, angular in local (body) frame.
        w_world = np.array([0.0, 0.0, psi_dot]) + theta_dot * R[:, 1]
        d.qvel[0:3] = [v * np.cos(psi), v * np.sin(psi), 0.0]
        d.qvel[3:6] = R.T @ w_world
        # wheels rolling at the commanded forward speed
        d.qvel[m.jnt_dofadr[self._lh]] = v / self.params.base.r
        d.qvel[m.jnt_dofadr[self._rh]] = v / self.params.base.r
        mj.mj_forward(m, d)

    def apply(self, torques: tuple[float, float]) -> None:
        lim = float(self.model.actuator_ctrlrange[0, 1])
        self.data.ctrl[0] = float(np.clip(torques[0], -lim, lim))
        self.data.ctrl[1] = float(np.clip(torques[1], -lim, lim))

    def step(self) -> None:
        self._mj.mj_step(self.model, self.data)

    @property
    def fell(self) -> bool:
        return abs(self.state[3]) > 1.2


def twip_world_xml(world=None, start=None, goal=None) -> str:
    """Return the TWIP model XML with a world's obstacles + start/goal drawn as visual geoms.

    Obstacles are rendered as semi-transparent pillars (visual only — ``contype=0`` — so the
    planar, collision-free control assumption is preserved; the planner already routes around
    them). Used for 3-D navigation rendering.
    """
    base_xml = TWIP_PATH.read_text()
    extras: list[str] = []
    if world is not None:
        for o in world.obstacles:
            extras.append(
                f'<geom type="cylinder" pos="{o.x} {o.y} 0.5" size="{o.r} 0.5" '
                f'rgba="0.85 0.3 0.3 0.55" contype="0" conaffinity="0"/>'
            )
    if start is not None:
        extras.append(
            f'<geom type="sphere" pos="{start[0]} {start[1]} 0.05" size="0.1" '
            f'rgba="0.1 0.1 0.1 0.9" contype="0" conaffinity="0"/>'
        )
    if goal is not None:
        extras.append(
            f'<geom type="cylinder" pos="{goal[0]} {goal[1]} 0.02" size="0.18 0.02" '
            f'rgba="0.95 0.55 0.1 0.8" contype="0" conaffinity="0"/>'
        )
    if not extras:
        return base_xml
    block = "\n    " + "\n    ".join(extras) + "\n  </worldbody>"
    return base_xml.replace("  </worldbody>", block, 1)


def _rotz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _roty(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def simulate_twip_mujoco(
    params: TWIPParams,
    controller: TWIPController,
    command_fn: CommandFn,
    sim: SimConfig | None = None,
    x0: np.ndarray | None = None,
    command_dt: float = 0.1,
    stop_fn: StopFn | None = None,
    plant: MuJoCoTWIP | None = None,
    step_callback: Callable[[MuJoCoTWIP, float], None] | None = None,
) -> TWIPTrajectory:
    """Closed-loop TWIP rollout in MuJoCo, returning the same :class:`TWIPTrajectory` type as
    the analytic backend (so plotting and metrics work unchanged).

    Two rates as in the analytic simulator: the inner balance + yaw controller fires at
    ``sim.control_dt`` (every step by default) and ``command_fn`` at ``command_dt``.
    """
    sim = sim or SimConfig(dt=0.002, duration=40.0, fall_angle=1.2)
    controller.reset()
    if plant is None:
        plant = MuJoCoTWIP(params)
    plant.model.opt.timestep = sim.dt
    plant.reset(x0)

    dt = sim.dt
    n_steps = int(round(sim.duration / dt))
    control_dt = sim.control_dt

    ts: list[float] = []
    xs: list[np.ndarray] = []
    cmds: list[tuple[float, float]] = []
    us: list[tuple[float, float]] = []
    u = (0.0, 0.0)
    cmd = (0.0, 0.0)
    last_ctrl_t = -np.inf
    last_cmd_t = -np.inf
    fell = False
    reached = False

    for i in range(n_steps + 1):
        t = i * dt
        state = plant.state
        if abs(state[3]) > sim.fall_angle:
            fell = True
            break
        if stop_fn is not None and stop_fn(state):
            reached = True
            break
        if (t - last_cmd_t) >= command_dt - 1e-12:
            cmd = tuple(command_fn(t, state))
            last_cmd_t = t
        if control_dt is None or (t - last_ctrl_t) >= control_dt - 1e-12:
            u = controller.compute(state, cmd[0], cmd[1], t)
            last_ctrl_t = t
        plant.apply(u)
        if i % sim.record_every == 0:
            ts.append(t)
            xs.append(state.copy())
            cmds.append(cmd)
            us.append(u)
        if step_callback is not None:
            step_callback(plant, t)
        if i < n_steps:
            plant.step()

    if not ts:
        ts.append(0.0)
        xs.append(plant.state.copy())
        cmds.append(cmd)
        us.append(u)

    return TWIPTrajectory(
        t=np.asarray(ts),
        states=np.asarray(xs),
        commands=np.asarray(cmds),
        torques=np.asarray(us),
        fell=fell,
        params=params,
        controller_name=getattr(controller, "balance_name", "unknown"),
        reached=reached,
    )
