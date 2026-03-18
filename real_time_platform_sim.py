#!/usr/bin/env python3
"""
real_time_platform_sim.py
===========================================================================
Project  : AI-Assisted Vibration Control of a Platform
Course   : ME 295A – Master's Project, Mechanical Engineering
Institute: San José State University
Author   : Khoi Le  |  May 2026
---------------------------------------------------------------------------
Provides:
  • PlatformConfig        – all physical / sensor / simulation constants
  • DisturbanceGenerator  – seismic / harmonic torque disturbances
  • PlatformPhysics       – 8-state RK4 rigid-body dynamics engine
  • PlatformEnv           – Gymnasium-compatible RL interface
  • RealTimeVisualizer    – 5-panel live matplotlib animation + 3-D view
  • PDController          – baseline PD controller for demo
  • main()                – stand-alone demo; run directly to see simulation

Coordinate conventions
-----------------------
  θx – roll  about x-axis (right-hand rule; +θx tilts the y+ side upward)
  θy – pitch about y-axis (right-hand rule; +θy tilts the x+ side downward)

Usage
-----
  python real_time_platform_sim.py            # live demo with PD controller
  python real_time_platform_sim.py --random   # random actuator commands
"""

from __future__ import annotations

import sys
import argparse
import warnings
import numpy as np
import matplotlib
# Backend priority: macOS native first (avoids Tk version requirements on macOS),
# then Qt5, then TkAgg, then whatever matplotlib defaults to.
import sys as _sys
_backends = (
    ["macosx", "Qt5Agg", "TkAgg"] if _sys.platform == "darwin"
    else ["Qt5Agg", "TkAgg", "macosx"]
)
for _b in _backends:
    try:
        matplotlib.use(_b)
        break
    except Exception:
        continue

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401  (side-effect)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
import gymnasium as gym
from gymnasium import spaces

warnings.filterwarnings("ignore")

# ============================================================================
# §1  CONFIGURATION
# ============================================================================

class PlatformConfig:
    """
    Centralised repository of all physical, sensor, and simulation constants.

    Platform geometry
    -----------------
    Square rigid plate, 50 mm × 50 mm, supported at four corners by
    independent spring-damper assemblies.  Each corner houses one electric
    actuator (0 – 10 mm shaft extension) fitted with a current sensor.
    A single IMU is mounted at the platform centroid.

    Corner numbering (viewed from above):
        1 – Front-Left  (FL) : (px, py) = (-L, +L)
        2 – Front-Right (FR) : (px, py) = (+L, +L)
        3 – Rear-Right  (RR) : (px, py) = (+L, -L)
        4 – Rear-Left   (RL) : (px, py) = (-L, -L)
    """

    # ── Geometry ─────────────────────────────────────────────────────────────
    SIDE: float = 0.050          # Platform side length [m]
    L:    float = 0.025          # Half-side = corner offset from centre [m]

    # Corner (px, py) body-frame positions [m]
    CORNERS: np.ndarray = np.array([
        [-0.025,  0.025],        # C1 – FL
        [ 0.025,  0.025],        # C2 – FR
        [ 0.025, -0.025],        # C3 – RR
        [-0.025, -0.025],        # C4 – RL
    ], dtype=float)

    # ── Inertial Properties ──────────────────────────────────────────────────
    MASS:  float = 0.500         # Platform mass [kg]
    I_MOM: float = 1.042e-4     # Rotational inertia (x or y) [kg·m²]
    #   Derived: I = m·L² / 3 = 0.500 × (0.025)² / 3 ≈ 1.042 × 10⁻⁴ kg·m²

    # ── Passive Spring-Damper (per corner) ───────────────────────────────────
    SPRING_K: float = 500.0      # Stiffness [N/m]
    DAMPER_C: float = 2.0        # Viscous damping coefficient [N·s/m]

    # ── Electric Actuator ────────────────────────────────────────────────────
    ACT_MIN: float = 0.000       # Minimum shaft extension [m]
    ACT_MAX: float = 0.010       # Maximum shaft extension [m]   (10 mm)
    ACT_BW:  float = 50.0        # Bandwidth [Hz]
    ACT_TAU: float = 3.183e-3    # First-order time constant τ=1/(2π·50) [s]

    # ── Motor / Current Sensor ────────────────────────────────────────────────
    MOTOR_KF:    float = 5.0     # Force-to-current constant [N/A]
    MOTOR_IMAX:  float = 2.0     # Saturation current [A]
    MOTOR_NOISE: float = 0.005   # Current measurement noise σ [A]

    # ── IMU ───────────────────────────────────────────────────────────────────
    IMU_ANG_NOISE:  float = 8.727e-4  # Angle noise σ ≈ 0.05° [rad]
    IMU_RATE_NOISE: float = 1.745e-3  # Rate noise σ ≈ 0.10°/s [rad/s]
    IMU_ANG_BIAS:   float = 1.745e-4  # Constant bias ≈ 0.01° [rad]

    # ── Simulation ────────────────────────────────────────────────────────────
    DT:          float = 0.001   # Integration timestep [s]   (1 ms)
    EP_DURATION: float = 10.0    # Episode length [s]
    N_STEPS:     int   = 10_000  # Steps per episode  (= EP_DURATION / DT)

    # ── Control Tolerances ────────────────────────────────────────────────────
    THETA_TOL: float = 8.727e-3  # ±0.5° angle settling band [rad]
    OMEGA_TOL: float = 8.727e-2  # ±5°/s rate band [rad/s]
    HOLD_TIME: float = 1.0       # Minimum dwell inside band to declare settled [s]
    T_REF:     float = 10.0      # Reference (worst-case) settling time [s]
    E_REF:     float = 80.0      # Max effort reference [A·s] (2A × 4 motors × 10 s)
    V_REF:     float = 0.5236    # Max angular rate reference ≈ 30°/s [rad/s]

    # ── Reward Weights (convex: wt + wa + wv = 1) ────────────────────────────
    W_TIME:   float = 0.5
    W_EFFORT: float = 0.3
    W_MOTION: float = 0.2
    BONUS_B:  float = 1.0

    # ── Observation normalisation ─────────────────────────────────────────────
    OBS_NORM: np.ndarray = np.array(
        [np.pi/4, np.pi/4, np.pi, np.pi, 2.0, 2.0, 2.0, 2.0], dtype=float
    )
    # Divides [θx, θy (rad), ωx, ωy (rad/s), I1..I4 (A)] so values ≈ [−1, 1]


CFG = PlatformConfig()    # module-level singleton used throughout


# ============================================================================
# §2  DISTURBANCE GENERATOR
# ============================================================================

class DisturbanceGenerator:
    """
    Generates realistic external torque disturbances (N·m) acting on the
    platform to emulate seismic ground motion or road-roughness excitation.

    Two modes
    ---------
    'seismic'  : AR(1) coloured noise shaped to 0.5–5 Hz ground-motion band.
    'harmonic' : Superposition of 3–6 sinusoids with random amplitudes.

    Call reset() at the start of each episode to re-randomise parameters.
    """

    def __init__(self, mode: str = "seismic",
                 rng: np.random.Generator | None = None):
        self.mode = mode
        self.rng  = rng or np.random.default_rng()
        self._reset_params()

    def reset(self) -> None:
        """Re-randomise disturbance parameters for a new episode."""
        self._reset_params()

    def torque(self, t: float) -> np.ndarray:
        """Return disturbance torque vector [τx, τy] in N·m at time t."""
        if self.mode == "harmonic":
            d = np.sum(
                self._amps * np.sin(2.0 * np.pi * self._freqs * t + self._phases),
                axis=0
            )
            return d
        else:  # AR(1) coloured noise
            noise      = self.rng.standard_normal(2)
            self._prev = self._coeff * self._prev + (1.0 - self._coeff) * noise
            return self._scale * self._prev

    # ── Private ───────────────────────────────────────────────────────────────

    def _reset_params(self) -> None:
        if self.mode == "harmonic":
            n = int(self.rng.integers(3, 7))
            self._freqs  = self.rng.uniform(0.5, 5.0, (n, 2))
            self._amps   = self.rng.uniform(0.0, 0.40, (n, 2))
            self._phases = self.rng.uniform(0.0, 2 * np.pi, (n, 2))
        else:
            self._prev  = np.zeros(2)
            self._coeff = 0.95
            self._scale = self.rng.uniform(0.15, 0.50, 2)


# ============================================================================
# §3  PHYSICS ENGINE
# ============================================================================

class PlatformPhysics:
    """
    Rigid-body rotational dynamics of the four-corner spring-damper platform,
    integrated with a 4th-order Runge-Kutta (RK4) scheme at timestep DT.

    State vector s ∈ ℝ⁸
    --------------------
    [θx, θy, ωx, ωy, s₁, s₂, s₃, s₄]
      θx, θy  – roll / pitch angles   [rad]
      ωx, ωy  – angular rates         [rad/s]
      s₁..s₄ – actuator shaft ext.   [m]

    Action vector u ∈ ℝ⁴
    ---------------------
    [u₁, u₂, u₃, u₄] – commanded shaft extensions [m],
    clipped to [ACT_MIN, ACT_MAX].

    Corner force model (upward positive)
    -------------------------------------
    F_i = K(sᵢ − zᵢ) + C(ṡᵢ − żᵢ)
    where  zᵢ  = pyᵢ·θx − pxᵢ·θy     (tilt-induced height)
           żᵢ  = pyᵢ·ωx − pxᵢ·ωy     (tilt-induced velocity)
    """

    def __init__(self, cfg: PlatformConfig = CFG,
                 disturbance: DisturbanceGenerator | None = None):
        self.cfg    = cfg
        self.dist   = disturbance or DisturbanceGenerator()
        self.state  = np.zeros(8)
        self.forces = np.zeros(4)
        self.t      = 0.0
        self.reset()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, theta_x0: float = 0.0, theta_y0: float = 0.0,
              perturb: bool = True) -> np.ndarray:
        """
        Reset to a (near-)level initial state.

        Parameters
        ----------
        theta_x0, theta_y0 : desired initial tilt [rad]
        perturb            : if True, add ±~6° random initial tilt
        """
        self.state = np.zeros(8)
        rng = np.random.default_rng()
        if perturb:
            theta_x0 += rng.uniform(-0.10, 0.10)
            theta_y0 += rng.uniform(-0.10, 0.10)
        self.state[0] = theta_x0
        self.state[1] = theta_y0
        self.state[4:8] = self.cfg.ACT_MAX / 2   # mid-range shaft position
        self.t      = 0.0
        self.forces = np.zeros(4)
        self.dist.reset()
        return self.state.copy()

    def step(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Advance simulation by one timestep using RK4 integration.

        Parameters
        ----------
        u : commanded shaft extensions [m], shape (4,)

        Returns
        -------
        state_new (8,), forces (4,)  – corner forces [N], upward positive
        """
        u_cmd = np.clip(u, self.cfg.ACT_MIN, self.cfg.ACT_MAX)
        d     = self.dist.torque(self.t)

        dt = self.cfg.DT
        k1 = self._deriv(self.state,                u_cmd, d)
        k2 = self._deriv(self.state + 0.5 * dt * k1, u_cmd, d)
        k3 = self._deriv(self.state + 0.5 * dt * k2, u_cmd, d)
        k4 = self._deriv(self.state +       dt * k3,  u_cmd, d)

        self.state = self.state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.state[4:8] = np.clip(self.state[4:8],
                                   self.cfg.ACT_MIN, self.cfg.ACT_MAX)
        self.t += dt

        self.forces = self._corner_forces(self.state, u_cmd)
        return self.state.copy(), self.forces.copy()

    # ── Sensor Models ─────────────────────────────────────────────────────────

    def imu_obs(self) -> np.ndarray:
        """Return noisy IMU measurement: [θx, θy, ωx, ωy] (rad, rad/s)."""
        c = self.cfg
        s = self.state
        return np.array([
            s[0] + c.IMU_ANG_BIAS  + np.random.normal(0, c.IMU_ANG_NOISE),
            s[1] + c.IMU_ANG_BIAS  + np.random.normal(0, c.IMU_ANG_NOISE),
            s[2]                   + np.random.normal(0, c.IMU_RATE_NOISE),
            s[3]                   + np.random.normal(0, c.IMU_RATE_NOISE),
        ])

    def motor_currents(self) -> np.ndarray:
        """
        Compute noisy current sensor readings I₁..I₄ [A].

        Current is proportional to corner force magnitude:
            Iᵢ = |Fᵢ| / MOTOR_KF  (clipped to MOTOR_IMAX, plus Gaussian noise)
        """
        c   = self.cfg
        cur = np.abs(self.forces) / c.MOTOR_KF
        cur = np.clip(cur, 0.0, c.MOTOR_IMAX)
        cur += np.abs(np.random.normal(0, c.MOTOR_NOISE, 4))
        return cur

    def corner_displacements_mm(self) -> np.ndarray:
        """
        Return vertical height of each corner relative to level [mm].

        Derived from IMU angles (as would be reconstructed from accelerometers):
            xᵢ = (pyᵢ · θx − pxᵢ · θy) × 1000
        """
        θx, θy = self.state[0], self.state[1]
        return np.array([
            (self.cfg.CORNERS[i, 1] * θx - self.cfg.CORNERS[i, 0] * θy) * 1e3
            for i in range(4)
        ])

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _corner_forces(self, s: np.ndarray, u_cmd: np.ndarray) -> np.ndarray:
        """Compute upward spring-damper force at each corner [N]."""
        θx, θy = s[0], s[1]
        ωx, ωy = s[2], s[3]
        sh     = s[4:8]
        K, C   = self.cfg.SPRING_K, self.cfg.DAMPER_C
        forces = np.empty(4)
        for i in range(4):
            px, py  = self.cfg.CORNERS[i]
            z_i     = py * θx - px * θy                       # corner height
            dz_i    = py * ωx - px * ωy                       # corner velocity
            dsh_i   = (u_cmd[i] - sh[i]) / self.cfg.ACT_TAU  # shaft velocity
            forces[i] = K * (sh[i] - z_i) + C * (dsh_i - dz_i)
        return forces

    def _deriv(self, s: np.ndarray, u_cmd: np.ndarray,
               d: np.ndarray) -> np.ndarray:
        """Evaluate ṡ = f(s, u, d) for RK4."""
        ds = np.zeros(8)
        # Angular kinematics
        ds[0] = s[2];  ds[1] = s[3]
        # Actuator dynamics (first-order lag)
        ds[4:8] = (u_cmd - s[4:8]) / self.cfg.ACT_TAU
        # Torques from corners
        F  = self._corner_forces(s, u_cmd)
        τx = τy = 0.0
        for i in range(4):
            px, py = self.cfg.CORNERS[i]
            τx += F[i] * py
            τy += F[i] * (-px)
        τx += d[0];  τy += d[1]
        ds[2] = τx / self.cfg.I_MOM
        ds[3] = τy / self.cfg.I_MOM
        return ds


# ============================================================================
# §4  GYMNASIUM ENVIRONMENT
# ============================================================================

class PlatformEnv(gym.Env):
    """
    Gymnasium environment wrapping PlatformPhysics for RL training.

    Observation space (8-D, approximately normalised to [−1, 1])
    -------------------------------------------------------------
    [θx, θy, ωx, ωy, I₁, I₂, I₃, I₄]

    Action space (4-D, continuous [−1, 1])
    ----------------------------------------
    Linearly mapped to shaft commands:
        u_cmd = (a + 1) / 2 × ACT_MAX   ∈ [0, ACT_MAX]

    Step-level reward (dense shaping)
    ----------------------------------
    r = −(wt·r_angle + wv·r_rate + wa·r_effort) + 0.1·r_bonus

    Episode-level metrics (returned in info at truncation)
    -------------------------------------------------------
    Follows Eq. (1) of the project report:
        R_ep = −(T·wt + E·wa + V·wv) + B·I(T ≤ T_ref ∧ V ≤ V_tol)
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, cfg: PlatformConfig = CFG,
                 disturbance_mode: str = "seismic",
                 render_mode: str | None = None):
        super().__init__()
        self.cfg         = cfg
        self.render_mode = render_mode
        self.physics     = PlatformPhysics(
            cfg, DisturbanceGenerator(mode=disturbance_mode))

        obs_bound = np.full(8, 5.0, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_bound, obs_bound, dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        self._reset_episode_metrics()

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None,
              options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.physics.reset(perturb=True)
        self._reset_episode_metrics()
        return self._obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        cfg    = self.cfg
        u_cmd  = (np.asarray(action) + 1.0) / 2.0 * cfg.ACT_MAX
        state, _ = self.physics.step(u_cmd)
        obs      = self._obs()

        θx, θy = state[0], state[1]
        ωx, ωy = state[2], state[3]
        currents = self.physics.motor_currents()

        # ── Update episode metrics ────────────────────────────────────────
        self._step += 1
        omega_mag   = np.hypot(ωx, ωy)
        self._peak_omega  = max(self._peak_omega, omega_mag)
        self._cum_effort += np.sum(currents) * cfg.DT

        # Settling detection (Eq. 2 of report)
        in_band = abs(θx) < cfg.THETA_TOL and abs(θy) < cfg.THETA_TOL
        t_now   = self._step * cfg.DT
        if in_band:
            if self._band_entry is None:
                self._band_entry = t_now
            elif (t_now - self._band_entry >= cfg.HOLD_TIME
                  and self._t_settle is None):
                self._t_settle = self._band_entry
        else:
            self._band_entry = None

        # ── Dense step reward ─────────────────────────────────────────────
        r_angle  = -(abs(θx) + abs(θy)) / (2.0 * cfg.THETA_TOL)
        r_rate   = -omega_mag / cfg.V_REF
        r_effort = -np.sum(currents) / (4.0 * cfg.MOTOR_IMAX)
        r_bonus  = 0.1 if in_band else 0.0
        reward   = (cfg.W_TIME   * r_angle  +
                    cfg.W_MOTION * r_rate   +
                    cfg.W_EFFORT * r_effort +
                    r_bonus)

        # ── Termination ───────────────────────────────────────────────────
        truncated  = self._step >= cfg.N_STEPS
        terminated = False

        # ── Info / episodic metrics ───────────────────────────────────────
        info: dict = {
            "theta_x_deg": float(np.degrees(θx)),
            "theta_y_deg": float(np.degrees(θy)),
            "omega_mag":   float(omega_mag),
            "currents":    currents,
            "t_sim":       t_now,
        }
        if truncated or terminated:
            T_n = (self._t_settle or cfg.T_REF) / cfg.T_REF
            E_n = min(self._cum_effort / cfg.E_REF, 1.0)
            V_n = min(self._peak_omega / cfg.V_REF, 1.0)
            settled = (self._t_settle is not None and omega_mag <= cfg.OMEGA_TOL)
            B  = cfg.BONUS_B if settled else 0.0
            # Eq. (1): R = −(T·wt + E·wa + V·wv) + B·I(T≤T_tol ∧ V≤V_tol)
            ep_r = -(T_n * cfg.W_TIME + E_n * cfg.W_EFFORT + V_n * cfg.W_MOTION) + B
            info.update({
                "ep_T_norm":  float(T_n),
                "ep_E_norm":  float(E_n),
                "ep_V_norm":  float(V_n),
                "ep_reward":  float(ep_r),
                "t_settle_s": self._t_settle,
                "settled":    settled,
            })

        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        pass   # handled by RealTimeVisualizer

    def close(self) -> None:
        pass

    # ── Private ───────────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        imu  = self.physics.imu_obs()
        curr = self.physics.motor_currents()
        raw  = np.concatenate([imu, curr])
        return (raw / self.cfg.OBS_NORM).astype(np.float32)

    def _reset_episode_metrics(self) -> None:
        self._step       = 0
        self._t_settle   = None
        self._band_entry = None
        self._cum_effort = 0.0
        self._peak_omega = 0.0


# ============================================================================
# §5  BASELINE PD CONTROLLER
# ============================================================================

class PDController:
    """
    Proportional-Derivative baseline controller for demo and benchmarking.

    Derives actuator extension commands from IMU-measured roll and pitch:

    Corner convention (C1=FL, C2=FR, C3=RR, C4=RL):
        u₁ = u₀ − Δx − Δy
        u₂ = u₀ − Δx + Δy
        u₃ = u₀ + Δx + Δy
        u₄ = u₀ + Δx − Δy

    where Δx = Kp·θx + Kd·ωx,  Δy = Kp·θy + Kd·ωy

    The signs enforce a restoring torque opposing the measured tilt.
    """

    def __init__(self, cfg: PlatformConfig = CFG,
                 Kp: float = 0.025, Kd: float = 0.004):
        self.cfg = cfg
        self.Kp  = Kp
        self.Kd  = Kd

    def __call__(self, obs_normalised: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        obs_normalised : (8,) observation from PlatformEnv (normalised)

        Returns
        -------
        action : (4,) in [−1, 1] compatible with PlatformEnv action space
        """
        raw = obs_normalised * self.cfg.OBS_NORM
        θx, θy, ωx, ωy = raw[0], raw[1], raw[2], raw[3]

        Δx  = self.Kp * θx + self.Kd * ωx
        Δy  = self.Kp * θy + self.Kd * ωy
        u0  = self.cfg.ACT_MAX / 2.0

        u = np.array([u0 - Δx - Δy,
                      u0 - Δx + Δy,
                      u0 + Δx + Δy,
                      u0 + Δx - Δy])
        u = np.clip(u, self.cfg.ACT_MIN, self.cfg.ACT_MAX)
        return (2.0 * u / self.cfg.ACT_MAX - 1.0).astype(np.float32)


# ============================================================================
# §6  REAL-TIME VISUALIZER
# ============================================================================

class RealTimeVisualizer:
    """
    Live matplotlib animation showing five panels:

    Left column (time series, rolling 5 s window)
    ──────────────────────────────────────────────
    1. Roll & Pitch angles θx, θy [°]
    2. Angular rates ωx, ωy [°/s]
    3. Motor currents I₁–I₄ [A]
    4. Corner displacements x₁–x₄ [mm]

    Right column
    ────────────
    5. 3-D perspective view of the tilting platform

    Usage
    -----
    viz = RealTimeVisualizer(env, controller=pd)
    viz.show()   # blocks until window is closed
    """

    _WIN_S     = 5.0    # seconds of history displayed
    _STEPS_F   = 30     # physics steps per animation frame
    _BG        = "#1C1C1E"
    _PANEL     = "#2C2C2E"
    _GRID      = "#3A3A3C"
    _LABEL     = "#AFAFAF"

    def __init__(self, env: PlatformEnv, controller=None):
        self.env  = env
        self.ctrl = controller
        cfg       = env.cfg

        n_buf = int(self._WIN_S / cfg.DT)
        self._n   = n_buf
        self._ptr = 0

        # Ring buffers
        self._t   = np.full(n_buf, np.nan)
        self._tx  = np.zeros(n_buf)
        self._ty  = np.zeros(n_buf)
        self._wx  = np.zeros(n_buf)
        self._wy  = np.zeros(n_buf)
        self._I   = np.zeros((n_buf, 4))
        self._X   = np.zeros((n_buf, 4))

        self._build_figure()
        obs, _ = self.env.reset()
        self._obs = obs

    # ── Figure construction ───────────────────────────────────────────────────

    def _build_figure(self) -> None:
        self.fig = plt.figure(figsize=(17, 9), facecolor=self._BG)
        self.fig.suptitle(
            "AI-Assisted Vibration Control — Real-Time Platform Simulation",
            color="white", fontsize=11, fontweight="bold", y=0.98)

        gs = gridspec.GridSpec(
            4, 2, figure=self.fig,
            left=0.06, right=0.97,
            top=0.94, bottom=0.07,
            hspace=0.55, wspace=0.28,
            width_ratios=[1.1, 1])

        kw = dict(facecolor=self._PANEL)
        self.ax_ang  = self.fig.add_subplot(gs[0, 0], **kw)
        self.ax_rate = self.fig.add_subplot(gs[1, 0], **kw)
        self.ax_cur  = self.fig.add_subplot(gs[2, 0], **kw)
        self.ax_disp = self.fig.add_subplot(gs[3, 0], **kw)
        self.ax_3d   = self.fig.add_subplot(gs[:, 1], projection="3d")
        self.ax_3d.set_facecolor(self._PANEL)

        self._style_time_axes()
        self._init_lines()
        self._setup_3d()

    def _style_time_axes(self) -> None:
        titles  = ["Roll & Pitch Angle [°]",
                   "Angular Rate [°/s]",
                   "Motor Current I₁–I₄ [A]",
                   "Corner Displacement x₁–x₄ [mm]"]
        ylabels = ["θ [°]", "ω [°/s]", "I [A]", "x [mm]"]
        tol_deg = np.degrees(self.env.cfg.THETA_TOL)

        for ax, ttl, yl in zip(
                [self.ax_ang, self.ax_rate, self.ax_cur, self.ax_disp],
                titles, ylabels):
            ax.set_title(ttl, color="white", fontsize=8.5, pad=3, loc="left")
            ax.set_ylabel(yl, color=self._LABEL, fontsize=7.5)
            ax.set_xlabel("Time [s]", color=self._LABEL, fontsize=7.5)
            ax.tick_params(colors=self._LABEL, labelsize=6.5)
            ax.spines[:].set_color("#444")
            ax.grid(True, color=self._GRID, linewidth=0.5, linestyle="--")
            ax.axhline(0, color="#555", linewidth=0.7)

        # Tolerance band on angle plot
        self.ax_ang.axhline( tol_deg, color="#FFD60A", lw=0.8, ls="--", alpha=0.8)
        self.ax_ang.axhline(-tol_deg, color="#FFD60A", lw=0.8, ls="--", alpha=0.8)

    def _init_lines(self) -> None:
        t = self._t
        lw = dict(linewidth=1.2)

        # — Angles
        self.l_tx, = self.ax_ang.plot(t, self._tx, "#FF453A", label="θx", **lw)
        self.l_ty, = self.ax_ang.plot(t, self._ty, "#30D158", label="θy", **lw)
        self.ax_ang.legend(fontsize=7, loc="upper right",
                           facecolor=self._PANEL, labelcolor="white", framealpha=0.8)

        # — Rates
        self.l_wx, = self.ax_rate.plot(t, self._wx, "#FF9F0A", label="ωx", **lw)
        self.l_wy, = self.ax_rate.plot(t, self._wy, "#64D2FF", label="ωy", **lw)
        self.ax_rate.legend(fontsize=7, loc="upper right",
                            facecolor=self._PANEL, labelcolor="white", framealpha=0.8)

        # — Currents
        c_I = ["#FF453A", "#30D158", "#FF9F0A", "#64D2FF"]
        self.l_I = [
            self.ax_cur.plot(t, self._I[:, i], c_I[i],
                             label=f"I{i+1}", **lw)[0]
            for i in range(4)]
        self.ax_cur.legend(fontsize=7, loc="upper right", ncol=2,
                           facecolor=self._PANEL, labelcolor="white", framealpha=0.8)

        # — Displacements
        c_X = ["#BF5AF2", "#FFD60A", "#FF6961", "#1AC8DB"]
        self.l_X = [
            self.ax_disp.plot(t, self._X[:, i], c_X[i],
                              label=f"x{i+1}", **lw)[0]
            for i in range(4)]
        self.ax_disp.legend(fontsize=7, loc="upper right", ncol=2,
                            facecolor=self._PANEL, labelcolor="white", framealpha=0.8)

    def _setup_3d(self) -> None:
        ax  = self.ax_3d
        L_m = self.env.cfg.L * 1e3   # mm
        ax.set_xlim(-38, 38);  ax.set_xlabel("X [mm]", color=self._LABEL, fontsize=7)
        ax.set_ylim(-38, 38);  ax.set_ylabel("Y [mm]", color=self._LABEL, fontsize=7)
        ax.set_zlim(-14, 14);  ax.set_zlabel("Z [mm]", color=self._LABEL, fontsize=7)
        ax.tick_params(colors=self._LABEL, labelsize=6)
        ax.set_title("3-D Platform View", color="white", fontsize=9, pad=6)
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False

        # Static base plate
        L = L_m
        bv = np.array([[-L,-L,-12],[L,-L,-12],[L,L,-12],[-L,L,-12]])
        ax.add_collection3d(
            Poly3DCollection([bv], alpha=0.3, facecolor="#555", edgecolor="#777"))

    # ── Animation callback ────────────────────────────────────────────────────

    def _update_frame(self, _frame: int) -> None:
        cfg = self.env.cfg

        for _ in range(self._STEPS_F):
            action = (self.ctrl(self._obs) if self.ctrl is not None
                      else self.env.action_space.sample())
            self._obs, _, terminated, truncated, _ = self.env.step(action)

            s    = self.env.physics.state
            curr = self.env.physics.motor_currents()
            disp = self.env.physics.corner_displacements_mm()
            t_now = self.env.physics.t

            i = self._ptr % self._n
            self._t[i]  = t_now
            self._tx[i] = np.degrees(s[0])
            self._ty[i] = np.degrees(s[1])
            self._wx[i] = np.degrees(s[2])
            self._wy[i] = np.degrees(s[3])
            self._I[i]  = curr
            self._X[i]  = disp
            self._ptr  += 1

            if terminated or truncated:
                self._obs, _ = self.env.reset()

        # Ordered indices for chronological display
        p   = self._ptr % self._n
        idx = np.roll(np.arange(self._n), -p)
        t_d = self._t[idx]

        # Update time-series lines
        self.l_tx.set_data(t_d, self._tx[idx])
        self.l_ty.set_data(t_d, self._ty[idx])
        self.l_wx.set_data(t_d, self._wx[idx])
        self.l_wy.set_data(t_d, self._wy[idx])
        for i, l in enumerate(self.l_I):
            l.set_data(t_d, self._I[idx, i])
        for i, l in enumerate(self.l_X):
            l.set_data(t_d, self._X[idx, i])

        valid = ~np.isnan(t_d)
        if valid.any():
            t_lo = float(np.nanmin(t_d[valid]))
            t_hi = float(np.nanmax(t_d[valid])) + 0.01
            for ax in [self.ax_ang, self.ax_rate, self.ax_cur, self.ax_disp]:
                ax.set_xlim(t_lo, t_hi)
                ax.relim()
                ax.autoscale_view(scalex=False)

        # ── 3-D platform update ───────────────────────────────────────────
        self._update_3d()

    def _update_3d(self) -> None:
        ax  = self.ax_3d
        cfg = self.env.cfg
        s   = self.env.physics.state
        θx, θy = s[0], s[1]

        # Corner 3-D positions [mm]
        c3d = np.array([
            [cfg.CORNERS[i, 0]*1e3,
             cfg.CORNERS[i, 1]*1e3,
             (cfg.CORNERS[i, 1]*θx - cfg.CORNERS[i, 0]*θy)*1e3]
            for i in range(4)
        ])

        # Remove old dynamic artists (tagged in previous frame)
        for art in list(ax.collections):
            if getattr(art, "_dynamic", False):
                art.remove()
        for art in list(ax.lines):
            if getattr(art, "_dynamic", False):
                art.remove()

        # Platform polygon
        plat = Poly3DCollection([c3d[[0, 1, 2, 3]]],
                                alpha=0.60, facecolor="#30D158",
                                edgecolor="#FFD60A", linewidth=1.5)
        plat._dynamic = True
        ax.add_collection3d(plat)

        # Actuator shafts + corner markers
        colors_c = ["#FF453A", "#30D158", "#FF9F0A", "#64D2FF"]
        labels_c = ["C1(FL)", "C2(FR)", "C3(RR)", "C4(RL)"]
        for i, (c, col, lbl) in enumerate(zip(c3d, colors_c, labels_c)):
            shaft = ax.plot([c[0], c[0]], [c[1], c[1]], [-12, c[2]],
                            color=col, linewidth=2.5, zorder=3)[0]
            shaft._dynamic = True
            dot = ax.plot([c[0]], [c[1]], [c[2]],
                          "o", color="#FFD60A", markersize=7, zorder=4)[0]
            dot._dynamic = True
            txt = ax.text(c[0], c[1], c[2]+1.5, lbl,
                          color="white", fontsize=6, ha="center")
            txt._dynamic = True

        # IMU marker at centroid
        imu = ax.plot([0], [0], [0], "s", color="#30D158",
                      markersize=10, zorder=5, label="IMU")[0]
        imu._dynamic = True

        ax.set_title(
            f"3-D Platform View   θx = {np.degrees(θx):+.2f}°   "
            f"θy = {np.degrees(θy):+.2f}°",
            color="white", fontsize=8.5, pad=6)

    # ── Public ────────────────────────────────────────────────────────────────

    def show(self) -> None:
        """Launch the animation window (blocks until closed)."""
        self._anim = animation.FuncAnimation(
            self.fig, self._update_frame,
            interval=50,             # ≈ 20 fps
            cache_frame_data=False)
        plt.show()


# ============================================================================
# §7  ENTRY POINT
# ============================================================================

def main(use_random: bool = False) -> None:
    """
    Stand-alone simulation demo.

    Launches RealTimeVisualizer with either the PD baseline controller
    (default) or random actuator commands (--random flag).
    """
    env  = PlatformEnv(cfg=CFG, disturbance_mode="seismic")
    ctrl = None if use_random else PDController(CFG)

    mode = "random actions" if use_random else "PD controller"
    print(f"[INFO] Starting real-time simulation with {mode}.")
    print("[INFO] Close the plot window to exit.\n")

    viz = RealTimeVisualizer(env, controller=ctrl)
    viz.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI-Assisted Vibration Control — Platform Simulation Demo")
    parser.add_argument("--random", action="store_true",
                        help="Use random actuator commands instead of PD controller")
    args = parser.parse_args()
    main(use_random=args.random)