"""Render a closed-loop rollout to an animated GIF or MP4 using MuJoCo offscreen rendering.

Requires the ``sim`` (MuJoCo) and ``viz`` (imageio) extras.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import RobotParams, SimConfig, TWIPParams
from ..controllers.base import Controller
from ..sim.mujoco_backend import MuJoCoPlant, simulate_mujoco
from ..sim.scenarios import Scenario


def render_rollout(
    params: RobotParams,
    controller: Controller,
    scenario: Scenario | None = None,
    sim: SimConfig | None = None,
    path: str | Path = "rollout.gif",
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    distance: float = 3.5,
    elevation: float = -12.0,
    azimuth: float = 90.0,
    xml_path: str | Path | None = None,
) -> str:
    """Simulate ``controller`` on ``scenario`` in MuJoCo and write an animation to ``path``.

    Returns the output path. The format (``.gif`` or ``.mp4``) is inferred from the suffix.
    ``xml_path`` selects the model geometry (e.g. the cart-pole used for swing-up).
    """
    import imageio.v2 as imageio
    import mujoco as mj

    sim = sim or SimConfig()
    scenario = scenario or Scenario()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # One plant instance, shared by the simulator and the renderer (same MjModel).
    plant = MuJoCoPlant(params, xml_path=xml_path)

    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = distance, elevation, azimuth
    cam.lookat = np.zeros(3)

    renderer = mj.Renderer(plant.model, height=height, width=width)
    frames: list[np.ndarray] = []
    frame_every = max(1, int(round(1.0 / (fps * sim.dt))))
    counter = {"i": 0}

    def grab(plant: MuJoCoPlant, t: float) -> None:
        if counter["i"] % frame_every == 0:
            cam.lookat[0] = plant.state[0]  # follow robot's x position
            renderer.update_scene(plant.data, camera=cam)
            frames.append(renderer.render().copy())
        counter["i"] += 1

    simulate_mujoco(params, controller, scenario, sim, step_callback=grab, plant=plant, xml_path=xml_path)
    renderer.close()

    if path.suffix.lower() == ".gif":
        imageio.mimsave(path, frames, duration=1.0 / fps, loop=0)
    else:
        imageio.mimsave(path, frames, fps=fps)
    return str(path)


def render_navigation(
    params: TWIPParams,
    world,
    start,
    goal,
    *,
    balance: str = "lqr",
    planner: str = "a_star",
    follower: str = "pure_pursuit",
    sim: SimConfig | None = None,
    path: str | Path = "navigation.gif",
    width: int = 720,
    height: int = 480,
    fps: int = 30,
    distance: float | None = None,
    elevation: float = -28.0,
    azimuth: float = 110.0,
    terrain=None,
) -> str:
    """Plan a route then render the balancing TWIP driving it in 3-D, to a GIF/MP4.

    Obstacles, start, and goal are drawn in the scene. With ``terrain`` given, the robot drives
    over the heightfield. Returns the output path.
    """
    import imageio.v2 as imageio
    import mujoco as mj

    from ..navigation.navigator import Navigator

    sim = sim or SimConfig(dt=0.002, duration=40.0, fall_angle=1.2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    nav = Navigator(params, balance=balance, planner=planner, follower=follower, backend="mujoco")
    route = nav.plan(world, start, goal)
    if route is None:
        raise ValueError("planner found no path from start to goal")

    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    psi0 = float(np.arctan2(route[1, 1] - route[0, 1], route[1, 0] - route[0, 0])) \
        if len(route) >= 2 else float(np.arctan2(goal[1] - start[1], goal[0] - start[0]))
    x0 = np.zeros(7)
    x0[0], x0[1], x0[2] = start[0], start[1], psi0

    if terrain is not None:
        from ..sim.mujoco_terrain import MuJoCoTWIPTerrain, simulate_twip_terrain
        plant = MuJoCoTWIPTerrain(params, terrain, world=world, start=start, goal=goal)
    else:
        from ..sim.mujoco_twip import MuJoCoTWIP, twip_world_xml
        plant = MuJoCoTWIP(params, xml_string=twip_world_xml(world, start, goal))

    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mj.mj_name2id(plant.model, mj.mjtObj.mjOBJ_BODY, "base")
    cam.distance = distance if distance is not None else 0.65 * max(world.width, world.height)
    cam.elevation, cam.azimuth = elevation, azimuth

    renderer = mj.Renderer(plant.model, height=height, width=width)
    frames: list[np.ndarray] = []
    frame_every = max(1, int(round(1.0 / (fps * sim.dt))))
    counter = {"i": 0}

    def grab(pl, t: float) -> None:
        if counter["i"] % frame_every == 0:
            renderer.update_scene(pl.data, camera=cam)
            frames.append(renderer.render().copy())
        counter["i"] += 1

    nav.follower.reset()

    def command_fn(t, state):
        v, w, _done = nav.follower.command(
            (float(state[0]), float(state[1]), float(state[2])), route, world)
        return v, w

    def stop_fn(state):
        return np.hypot(state[0] - goal[0], state[1] - goal[1]) < nav.goal_tol

    if terrain is not None:
        simulate_twip_terrain(params, nav.controller, command_fn, terrain, sim=sim, x0=x0,
                              world=world, plant=plant, step_callback=grab, stop_fn=stop_fn)
    else:
        from ..sim.mujoco_twip import simulate_twip_mujoco
        simulate_twip_mujoco(params, nav.controller, command_fn, sim=sim, x0=x0,
                             plant=plant, step_callback=grab, stop_fn=stop_fn)
    renderer.close()

    if path.suffix.lower() == ".gif":
        imageio.mimsave(path, frames, duration=1.0 / fps, loop=0)
    else:
        imageio.mimsave(path, frames, fps=fps)
    return str(path)
