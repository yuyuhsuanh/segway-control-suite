"""Emergency-stop gym environment for the TWIP.

RL upper layer: action = theta_ref_offset.  Negative = backward lean = braking.
Lower layer: LQR balance controller (unchanged from navigation stack).
Backend: MuJoCoTWIP (real contact / friction physics).

Reward: -|v|*dt_ctrl every step (minimize stopping distance) + -50 on fall.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..config import TWIPParams
from ..controllers import build_controller
from ..sim.mujoco_twip import MuJoCoTWIP


class EmergencyStopEnv(gym.Env):
    """TWIP emergency-stop task.  Agent controls the pitch lean reference; LQR handles balance."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        params: TWIPParams | None = None,
        dt_ctrl: float = 0.02,
        dt_phys: float = 0.002,
        max_seconds: float = 10.0,
        v_range: tuple[float, float] = (0.3, 1.5),
        fall_angle: float = 1.2,
        v_exit: float = 0.05,
    ):
        super().__init__()
        self.params = params or TWIPParams()
        self.dt_ctrl = float(dt_ctrl)
        self.dt_phys = float(dt_phys)
        self.substeps = round(dt_ctrl / dt_phys)
        self.max_steps = int(max_seconds / dt_ctrl)
        self.v_range = v_range
        self.fall_angle = float(fall_angle)
        self.v_exit = float(v_exit)

        self.lqr = build_controller("lqr", self.params.base)
        self.plant = MuJoCoTWIP(self.params)
        self.plant.model.opt.timestep = self.dt_phys

        high = np.array([20.0, np.pi, 20.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        # Wide bound: RL learns the useful range; fall terminates + penalizes extreme actions.
        # ±1.5 rad is far beyond any physical operating point (fall_angle = 1.2 rad).
        self.action_space = spaces.Box(-1.5, 1.5, shape=(1,), dtype=np.float32)

        self._step = 0
        self._t = 0.0

    def _obs(self, state: np.ndarray) -> np.ndarray:
        _, _, _, theta, v, _, theta_dot = state
        return np.array([v, theta, theta_dot], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        v_init = float(self.np_random.uniform(*self.v_range))
        x0 = np.array([0.0, 0.0, 0.0, 0.0, v_init, 0.0, 0.0])
        self.plant.reset(x0)
        self.lqr.reset()
        self._step = 0
        self._t = 0.0
        return self._obs(self.plant.state), {}

    def step(self, action: np.ndarray):
        theta_ref = float(action[0])

        for _ in range(self.substeps):
            state = self.plant.state
            _, _, _, theta, v, _, theta_dot = state
            # Inject backward-lean reference into the LQR error vector.
            # err[1] = v - v_des with v_des=0; err[2] = theta deviation from lean target.
            err = np.array([0.0, v, theta - theta_ref, theta_dot])
            tau_sum = self.lqr.compute(err, self._t)
            half = tau_sum / 2.0
            self.plant.apply((half, half))
            self.plant.step()
            self._t += self.dt_phys

        self._step += 1
        state = self.plant.state
        x, _, _, theta, v, _, _ = state

        fell = abs(theta) > self.fall_angle
        rolled_back = v < -self.v_exit
        reward = -abs(v) * self.dt_ctrl
        if fell or rolled_back:
            reward -= 50.0

        terminated = fell or rolled_back
        truncated = self._step >= self.max_steps
        return self._obs(state), float(reward), terminated, truncated, {"fell": fell, "v": v, "x": x}
