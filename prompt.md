## Goal

Train an end-to-end RL policy (action = motor torque) for a wheeled inverted pendulum to
actively exploit wheel slip during acceleration and deceleration.

## Mission

Given a step velocity command (0 → V_max → 0), minimize the time to reach V_max and the
time to return to 0. This is a concept study — insights will later be applied to the
WF_TRON1B wheel-foot biped separately.

---

## Settled Design Decisions

### Slip definition
True wheel slip (ω·r ≠ v_chassis). The motor (±40 Nm/wheel) far exceeds the traction
limit at μ=0.6 (~8.6 Nm/wheel), so slip occurs naturally above ~22% throttle. The
challenge is controlled exploitation, not whether to slip.

### Simulation backend
MuJoCo TWIP with rolling-contact physics (`sim/mujoco_twip.py`). The 1-D RK4 model
ignores friction entirely and cannot be used for this task.

### Robot parameters (WF_TRON1B equivalent at standing pose)
| param | value |
|---|---|
| Total mass m | 22.97 kg |
| Wheel radius r | 0.127 m |
| Pendulum length l (COM above axle) | 0.616 m |
| Friction μ | 0.6 |
| Torque limit u_max | 80 Nm combined (40 Nm/wheel) |

### Action space
Single torque (1-D, no yaw). Both wheels driven equally.

### Step command — velocity profile
Fixed profile (seconds → m/s):

| t (s) | v_des (m/s) |
|---|---|
| 0 – 2 | 0 (stationary, robot stabilizes) |
| 2 – 5 | 2.0 (step up) |
| 5 – 7 | 0 (step down) |

V_max = 2.0 m/s. Episode length = 7 s.

### Metric
Minimize:
- **T_accel**: time from v=0 to first reaching V_max
- **T_decel**: time from v=V_max to |v| < 0.1 m/s

### Model-based baseline
Controllers: cascaded PID, LQR, MPC.
Tuning: Optuna auto-tuner with RMSE velocity tracking error as the cost function.
(Existing auto-tuner cost is balance-oriented; define a new velocity-tracking cost.)

### RL training
- Framework: Stable-Baselines3 PPO, from scratch (new obs space, new env)
- Observation: `[θ, θ̇, v, v_des]`
- Reward: dense velocity tracking error + phase completion time bonus (faster = bigger bonus)
- Domain randomization: none for this concept phase
- Slip ratio in obs: **reserve as fallback** — first attempt reward shaping only

### Reward shaping strategy
1. Train baseline with dense tracking + time bonus → check if slip naturally emerges
2. If not, add explicit slip reward term (e.g. `+k·max(0, ω·r − v)` during accel phase)
3. Only if reward fails → add slip ratio to observation space

### Migration plan
This study is a proof-of-concept on the TWIP model. Applying to the actual WF_TRON1B
requires a separate training run (different kinematics, leg joints, full-body dynamics)
in MuJoCo or IsaacLab — retraining is expected.

---

## Steps

### 1. Set up equivalent TWIP model
- Update `RobotParams` (or a new `TWIPParams`) with WF_TRON1B values above
- Set MuJoCo floor friction to 0.6, wheel torque limit to 40 Nm/wheel

### 2. Build model-based baseline
- Define velocity-tracking cost (RMSE) for the auto-tuner
- Run Optuna on cascaded PID, LQR, MPC
- Sweep V_max upward until each controller fails → record ceiling velocity and
  minimum T_accel / T_decel for each

### 3. Train baseline RL policy
- Write new MuJoCo Gym env: obs `[θ, θ̇, v, v_des]`, action single torque,
  reward = dense tracking + time bonus
- Train PPO from scratch; compare T_accel / T_decel against model-based baseline

### 4. Adjust reward to encourage slip
- Analyze whether the baseline RL policy is already exploiting slip (plot ω·r vs v)
- If not, add slip reward term and iterate
- Last resort: add slip ratio to observation space
