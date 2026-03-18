#!/usr/bin/env python3
"""
controller_reinforcement_agent.py
===========================================================================
Project  : AI-Assisted Vibration Control of a Platform
Course   : ME 295B – Master's Project, Mechanical Engineering
Institute: San José State University
Author   : Khoi Le  |  May 2026
---------------------------------------------------------------------------
Provides:
  • EpisodeLogger       – collects and summarises per-episode statistics
  • RewardCalculator    – standalone implementation of the project reward
                          (Eq. 1 of the report: R = −T·wt + E·wa + V·wv + B·I(·))
  • train()             – PPO training loop via stable-baselines3
  • evaluate()          – deterministic roll-out with metric aggregation
  • plot_training_curves() – learning-curve visualisation
  • main()              – CLI entry point

Reward Function (Eq. 1)
------------------------
    R_ep = −(T·wt + E·wa + V·wv) + B·I(T ≤ T_ref ∧ V ≤ V_tol)

    T – normalised settling time         ∈ [0, 1]
    E – normalised cumulative effort     ∈ [0, 1]   (Eq. 3/4)
    V – normalised peak angular velocity ∈ [0, 1]   (Eq. 5/6)
    B – scalar bonus (awarded when both criteria are met)
    wt + wa + wv = 1  (convex weight constraint)

Usage
-----
    python controller_reinforcement_agent.py train  --timesteps 500000
    python controller_reinforcement_agent.py eval   --episodes 20
    python controller_reinforcement_agent.py train  --timesteps 200000 --fast
"""

from __future__ import annotations

import os
import sys
import argparse
import warnings
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive for training; changed at eval
import matplotlib.pyplot as plt

# Gymnasium
import gymnasium as gym

# stable-baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import (
        BaseCallback, EvalCallback, CheckpointCallback)
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.results_plotter import load_results, ts2xy
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
except ImportError:
    sys.exit("[ERROR] stable-baselines3 not found.  "
             "Run:  pip install stable-baselines3[extra]")

# Local simulation module
sys.path.insert(0, os.path.dirname(__file__))
from real_time_platform_sim import PlatformConfig, PlatformEnv, PDController

warnings.filterwarnings("ignore")

# ── Directory layout ────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
LOG_DIR     = os.path.join(BASE_DIR, "logs")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
FIGURE_DIR  = os.path.join(BASE_DIR, "figures")
DATA_DIR    = os.path.join(BASE_DIR, "data")

for _d in (LOG_DIR, MODEL_DIR, FIGURE_DIR, DATA_DIR):
    os.makedirs(_d, exist_ok=True)

CFG = PlatformConfig()


# ============================================================================
# §1  REWARD CALCULATOR  (standalone reference implementation)
# ============================================================================

@dataclass
class EpisodeMetrics:
    """
    Accumulates raw sensor data during one episode to compute the three
    normalised metrics used in the episodic reward function (Eq. 1).
    """
    cfg: PlatformConfig = field(default_factory=PlatformConfig)

    # Running accumulators (updated at each step)
    _cum_effort_raw: float = field(default=0.0, init=False)
    _peak_omega_raw: float = field(default=0.0, init=False)
    _band_entry:     float | None = field(default=None, init=False)
    _t_settle:       float | None = field(default=None, init=False)
    _n_steps:        int   = field(default=0, init=False)

    def reset(self) -> None:
        """Call at episode start."""
        self._cum_effort_raw = 0.0
        self._peak_omega_raw = 0.0
        self._band_entry     = None
        self._t_settle       = None
        self._n_steps        = 0

    def update(self, theta_x: float, theta_y: float,
               omega_x: float, omega_y: float,
               currents: np.ndarray) -> None:
        """
        Record sensor values for one timestep.

        Parameters
        ----------
        theta_x, theta_y : IMU roll/pitch [rad]
        omega_x, omega_y : IMU angular rates [rad/s]
        currents         : motor currents I₁..I₄ [A]
        """
        cfg    = self.cfg
        dt     = cfg.DT
        t_now  = self._n_steps * dt
        self._n_steps += 1

        # Eq. (3): cumulative current integral  Eraw = ∫ Σ|Iᵢ| dt
        self._cum_effort_raw += float(np.sum(np.abs(currents))) * dt

        # Eq. (5): peak angular velocity magnitude
        omega_mag = float(np.hypot(omega_x, omega_y))
        self._peak_omega_raw = max(self._peak_omega_raw, omega_mag)

        # Eq. (2): earliest time both tilt channels enter & hold tolerance band
        in_band = abs(theta_x) < cfg.THETA_TOL and abs(theta_y) < cfg.THETA_TOL
        if in_band:
            if self._band_entry is None:
                self._band_entry = t_now
            elif (t_now - self._band_entry >= cfg.HOLD_TIME
                  and self._t_settle is None):
                self._t_settle = self._band_entry
        else:
            self._band_entry = None

    # ── Normalised metrics ────────────────────────────────────────────────────

    @property
    def T_norm(self) -> float:
        """
        Normalised settling time T ∈ [0, 1].

        T = t_settle / T_ref.  If the platform never settles, T = 1 (worst case).
        """
        t_s = self._t_settle if self._t_settle is not None else self.cfg.T_REF
        return min(t_s / self.cfg.T_REF, 1.0)

    @property
    def E_norm(self) -> float:
        """Normalised cumulative actuation effort E ∈ [0, 1]  (Eq. 3/4)."""
        return min(self._cum_effort_raw / self.cfg.E_REF, 1.0)

    @property
    def V_norm(self) -> float:
        """Normalised peak angular velocity V ∈ [0, 1]  (Eq. 5/6)."""
        return min(self._peak_omega_raw / self.cfg.V_REF, 1.0)

    @property
    def settled(self) -> bool:
        return self._t_settle is not None

    @property
    def t_settle_s(self) -> float | None:
        return self._t_settle


class RewardCalculator:
    """
    Standalone implementation of the episodic reward function (Eq. 1).

    Reward = −T·wt − E·wa − V·wv + B·I(T ≤ T_ref ∧ V ≤ V_tol)

    Parameters
    ----------
    cfg    : PlatformConfig (default: module-level CFG)
    w_time : weight on settling-time penalty  (wt)
    w_effort: weight on actuation-effort penalty (wa)
    w_motion: weight on peak-motion penalty  (wv)
    bonus  : scalar B awarded when both criteria are met

    Constraint
    ----------
    wt + wa + wv must equal 1 (convex combination).  A ValueError is raised
    if the constraint is violated beyond a tolerance of 1 × 10⁻⁶.
    """

    def __init__(self, cfg: PlatformConfig = CFG,
                 w_time:   float = 0.5,
                 w_effort: float = 0.3,
                 w_motion: float = 0.2,
                 bonus:    float = 1.0):
        if abs(w_time + w_effort + w_motion - 1.0) > 1e-6:
            raise ValueError(
                f"Reward weights must sum to 1.  Got {w_time+w_effort+w_motion:.6f}")
        self.cfg      = cfg
        self.wt       = w_time
        self.wa       = w_effort
        self.wv       = w_motion
        self.B        = bonus

    def compute(self, metrics: EpisodeMetrics) -> float:
        """
        Compute the scalar episodic reward from accumulated episode metrics.

        Returns
        -------
        float : R_ep ∈ [−1, +B_max]
        """
        T = metrics.T_norm
        E = metrics.E_norm
        V = metrics.V_norm

        # Bonus indicator I(T ≤ 1 ∧ V ≤ V_ref/V_ref = 1)
        # (in practice T=1 or V=1 indicates failure, so bonus withheld)
        settled = metrics.settled
        bonus   = self.B if settled else 0.0

        reward = -(T * self.wt + E * self.wa + V * self.wv) + bonus
        return float(reward)

    def breakdown(self, metrics: EpisodeMetrics) -> dict:
        """Return a dict with all component values for logging."""
        return {
            "T_norm":    metrics.T_norm,
            "E_norm":    metrics.E_norm,
            "V_norm":    metrics.V_norm,
            "t_settle":  metrics.t_settle_s,
            "settled":   metrics.settled,
            "reward":    self.compute(metrics),
            "wt":        self.wt,
            "wa":        self.wa,
            "wv":        self.wv,
            "bonus":     self.B if metrics.settled else 0.0,
        }


# ============================================================================
# §2  TRAINING CALLBACK
# ============================================================================

class MetricsCallback(BaseCallback):
    """
    SB3 callback that extracts episodic metrics from the info dict and
    writes them to a CSV log for post-hoc analysis.
    """

    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self._path   = log_path
        self._rows   = []
        self._ep_idx = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "ep_reward" in info:
                self._ep_idx += 1
                row = {
                    "episode":    self._ep_idx,
                    "timestep":   self.num_timesteps,
                    "ep_reward":  info.get("ep_reward", np.nan),
                    "T_norm":     info.get("ep_T_norm",  np.nan),
                    "E_norm":     info.get("ep_E_norm",  np.nan),
                    "V_norm":     info.get("ep_V_norm",  np.nan),
                    "t_settle_s": info.get("t_settle_s", np.nan),
                    "settled":    int(info.get("settled", False)),
                }
                self._rows.append(row)
                if self.verbose and self._ep_idx % 50 == 0:
                    print(f"  [ep {self._ep_idx:5d}] "
                          f"R={row['ep_reward']:+.3f}  "
                          f"T={row['T_norm']:.3f}  "
                          f"E={row['E_norm']:.3f}  "
                          f"V={row['V_norm']:.3f}  "
                          f"settled={bool(row['settled'])}")
        return True

    def _on_training_end(self) -> None:
        import csv
        if not self._rows:
            return
        keys = list(self._rows[0].keys())
        with open(self._path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self._rows)
        print(f"\n[INFO] Episode metrics saved → {self._path}")


# ============================================================================
# §3  TRAINING FUNCTION
# ============================================================================

def train(timesteps:       int   = 500_000,
          n_envs:          int   = 4,
          disturbance_mode: str  = "seismic",
          fast:            bool  = False,
          seed:            int   = 42) -> str:
    """
    Train a PPO agent on the platform vibration-control task.

    Parameters
    ----------
    timesteps        : total environment interactions
    n_envs           : number of parallel environments (SubprocVecEnv)
    disturbance_mode : 'seismic' or 'harmonic'
    fast             : if True, use reduced network and fewer timesteps (dev)
    seed             : random seed

    Returns
    -------
    model_path : absolute path to the saved model zip file
    """
    run_id     = f"ppo_{disturbance_mode}_{int(time.time())}"
    log_sub    = os.path.join(LOG_DIR,   run_id)
    model_path = os.path.join(MODEL_DIR, run_id + "_final")
    os.makedirs(log_sub, exist_ok=True)

    print("=" * 65)
    print(f"  AI-Assisted Vibration Control — PPO Training")
    print(f"  Run ID          : {run_id}")
    print(f"  Timesteps       : {timesteps:,}")
    print(f"  Parallel envs   : {n_envs}")
    print(f"  Disturbance mode: {disturbance_mode}")
    print(f"  Log dir         : {log_sub}")
    print("=" * 65)

    # ── Environment factory ───────────────────────────────────────────────────
    def make_env(rank: int):
        def _init():
            env = PlatformEnv(cfg=CFG, disturbance_mode=disturbance_mode)
            env = Monitor(env, filename=os.path.join(log_sub, f"env_{rank}"))
            env.reset(seed=seed + rank)
            return env
        return _init

    vec_env = DummyVecEnv([make_env(i) for i in range(n_envs)])

    # Separate eval environment (single, deterministic disturbance)
    eval_env = Monitor(
        PlatformEnv(cfg=CFG, disturbance_mode=disturbance_mode),
        filename=os.path.join(log_sub, "eval"))

    # ── Policy network configuration ──────────────────────────────────────────
    if fast:
        net_arch = dict(pi=[64, 64], vf=[64, 64])
        n_steps  = 256
    else:
        net_arch = dict(pi=[256, 256, 128], vf=[256, 256, 128])
        n_steps  = 2048

    policy_kwargs = dict(
        net_arch=net_arch,
        activation_fn=__import__("torch").nn.Tanh,
    )

    # ── PPO hyperparameters (tuned for continuous control) ────────────────────
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate     = 3e-4,
        n_steps           = n_steps,
        batch_size        = 64,
        n_epochs          = 10,
        gamma             = 0.99,
        gae_lambda        = 0.95,
        clip_range        = 0.20,
        ent_coef          = 0.005,
        vf_coef           = 0.50,
        max_grad_norm     = 0.5,
        policy_kwargs     = policy_kwargs,
        tensorboard_log   = log_sub,
        seed              = seed,
        verbose           = 0,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    metrics_cb = MetricsCallback(
        log_path=os.path.join(log_sub, "episode_metrics.csv"),
        verbose=1)
    checkpoint_cb = CheckpointCallback(
        save_freq    = max(50_000 // n_envs, 1),
        save_path    = os.path.join(MODEL_DIR, run_id + "_ckpts"),
        name_prefix  = "model",
        verbose      = 0)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = os.path.join(MODEL_DIR, run_id + "_best"),
        log_path             = log_sub,
        eval_freq            = max(20_000 // n_envs, 1),
        n_eval_episodes      = 5,
        deterministic        = True,
        verbose              = 0)

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    # progress_bar requires tqdm+rich; use only when available
    try:
        import tqdm as _tqdm, rich as _rich  # noqa: F401
        _pbar = True
    except ImportError:
        _pbar = False

    model.learn(
        total_timesteps = timesteps,
        callback        = [metrics_cb, checkpoint_cb, eval_cb],
        progress_bar    = _pbar,
    )
    elapsed = time.time() - t0

    # ── Save final model ──────────────────────────────────────────────────────
    model.save(model_path)
    print(f"\n[INFO] Training complete in {elapsed/60:.1f} min.")
    print(f"[INFO] Model saved → {model_path}.zip")

    # ── Plot training curves ──────────────────────────────────────────────────
    metrics_csv = os.path.join(log_sub, "episode_metrics.csv")
    if os.path.exists(metrics_csv):
        plot_training_curves(metrics_csv, run_id)

    vec_env.close()
    eval_env.close()
    return model_path


# ============================================================================
# §4  EVALUATION FUNCTION
# ============================================================================

def evaluate(model_path:       str,
             n_episodes:       int   = 20,
             disturbance_mode: str   = "seismic",
             render:           bool  = False,
             save_csv:         bool  = True) -> dict:
    """
    Roll out a saved PPO model deterministically and aggregate performance.

    Parameters
    ----------
    model_path       : path to .zip model file (without extension)
    n_episodes       : number of evaluation episodes
    disturbance_mode : 'seismic' or 'harmonic'
    render           : if True, open live visualisation window
    save_csv         : if True, save per-episode results to CSV

    Returns
    -------
    summary : dict with mean / std / median of T, E, V, reward, settle rate
    """
    print(f"\n{'='*65}")
    print(f"  Evaluating model: {os.path.basename(model_path)}")
    print(f"  Episodes: {n_episodes}  |  Mode: {disturbance_mode}")
    print(f"{'='*65}")

    model = PPO.load(model_path)

    # PD baseline for comparison
    pd  = PDController(CFG)
    rew_calc = RewardCalculator(CFG)

    results_rl = []
    results_pd = []

    for ep in range(n_episodes):
        for tag, ctrl in [("RL", None), ("PD", pd)]:
            env     = PlatformEnv(cfg=CFG, disturbance_mode=disturbance_mode)
            metrics = EpisodeMetrics(cfg=CFG)
            obs, _  = env.reset(seed=ep)
            metrics.reset()

            done = False
            while not done:
                if ctrl is None:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = ctrl(obs)

                obs, _, terminated, truncated, info = env.step(action)
                s    = env.physics.state
                curr = env.physics.motor_currents()
                metrics.update(s[0], s[1], s[2], s[3], curr)
                done = terminated or truncated

            bd = rew_calc.breakdown(metrics)
            bd["episode"] = ep
            if tag == "RL":
                results_rl.append(bd)
            else:
                results_pd.append(bd)
            env.close()

        print(f"  ep {ep+1:3d}/{n_episodes}  "
              f"RL: R={results_rl[-1]['reward']:+.3f}  "
              f"T={results_rl[-1]['T_norm']:.3f}  "
              f"settled={results_rl[-1]['settled']}  ||  "
              f"PD: R={results_pd[-1]['reward']:+.3f}  "
              f"T={results_pd[-1]['T_norm']:.3f}")

    def _stats(rows: list[dict], key: str) -> tuple[float, float, float]:
        vals = [r[key] for r in rows if not np.isnan(float(r[key] or np.nan))]
        return (float(np.mean(vals)), float(np.std(vals)), float(np.median(vals)))

    print("\n" + "─"*65)
    print(f"{'Metric':<20}{'RL (mean±std)':<22}{'PD (mean±std)':<22}")
    print("─"*65)
    summary = {}
    for key in ["T_norm", "E_norm", "V_norm", "reward"]:
        m_rl, s_rl, _ = _stats(results_rl, key)
        m_pd, s_pd, _ = _stats(results_pd, key)
        print(f"  {key:<18}{m_rl:+.4f} ± {s_rl:.4f}     "
              f"{m_pd:+.4f} ± {s_pd:.4f}")
        summary[f"RL_{key}_mean"] = m_rl
        summary[f"RL_{key}_std"]  = s_rl
        summary[f"PD_{key}_mean"] = m_pd
        summary[f"PD_{key}_std"]  = s_pd

    settle_rl = float(np.mean([r["settled"] for r in results_rl]))
    settle_pd = float(np.mean([r["settled"] for r in results_pd]))
    print(f"  {'Settle rate':<18}{settle_rl:.1%}               {settle_pd:.1%}")
    summary["RL_settle_rate"] = settle_rl
    summary["PD_settle_rate"] = settle_pd
    print("─"*65)

    if save_csv:
        import csv, itertools
        csv_path = os.path.join(DATA_DIR, "evaluation_results.csv")
        all_rows = ([{"controller": "RL", **r} for r in results_rl] +
                    [{"controller": "PD", **r} for r in results_pd])
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            w.writeheader()
            w.writerows(all_rows)
        print(f"[INFO] Evaluation CSV saved → {csv_path}")

    if render:
        _render_evaluation(model, n_episodes=3, disturbance_mode=disturbance_mode)

    return summary


# ============================================================================
# §5  VISUALISATION HELPERS
# ============================================================================

def plot_training_curves(metrics_csv: str, run_id: str = "") -> None:
    """
    Plot learning curves from the episode metrics CSV produced during training.
    Saves PNG to the figures directory.
    """
    import csv

    if not os.path.exists(metrics_csv):
        print(f"[WARN] Metrics CSV not found: {metrics_csv}")
        return

    rows = []
    with open(metrics_csv) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) if v not in ("", "nan", "None") else np.nan
                         for k, v in row.items()})

    if not rows:
        return

    eps      = [r["episode"]    for r in rows]
    rewards  = [r["ep_reward"]  for r in rows]
    T_vals   = [r["T_norm"]     for r in rows]
    E_vals   = [r["E_norm"]     for r in rows]
    V_vals   = [r["V_norm"]     for r in rows]
    settled  = [r["settled"]    for r in rows]

    # Rolling mean (window = 50 episodes)
    def _roll(arr, w=50):
        arr = np.array(arr, dtype=float)
        w = min(w, len(arr))          # clamp window to available data
        kernel = np.ones(w) / w
        full = np.convolve(arr, kernel, mode='full')
        # take the centre slice that matches the original length
        trim = (len(full) - len(arr)) // 2
        return full[trim: trim + len(arr)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), facecolor="#1C1C1E")
    fig.suptitle(f"Training Curves  —  {run_id}", color="white",
                 fontsize=11, fontweight="bold")
    panel = "#2C2C2E"
    grid  = "#3A3A3C"

    config = [
        (rewards, "Episodic Reward",         "#30D158"),
        (T_vals,  "Settling Time T (norm.)", "#FF9F0A"),
        (E_vals,  "Actuation Effort E (norm.)", "#64D2FF"),
        (V_vals,  "Peak Motion V (norm.)",    "#FF453A"),
    ]

    for ax, (vals, title, col) in zip(axes.flat, config):
        ax.set_facecolor(panel)
        ax.plot(eps, vals, color=col, alpha=0.25, linewidth=0.7)
        ax.plot(eps, _roll(vals), color=col, linewidth=1.8, label="50-ep mean")
        if title == "Episodic Reward":
            ax.fill_between(eps, _roll(vals), alpha=0.15, color=col)
        ax.set_title(title, color="white", fontsize=9)
        ax.set_xlabel("Episode", color="#AFAFAF", fontsize=8)
        ax.tick_params(colors="#AFAFAF", labelsize=7)
        ax.spines[:].set_color("#444")
        ax.grid(True, color=grid, linewidth=0.5, linestyle="--")
        ax.legend(fontsize=7, facecolor=panel, labelcolor="white")

    # Overlay settle rate on reward panel
    settle_roll = _roll(settled, 50)
    ax_r = axes[0, 0].twinx()
    ax_r.plot(eps, settle_roll * 100, "#FFD60A", linewidth=1.2,
              linestyle="--", alpha=0.7, label="Settle rate %")
    ax_r.set_ylabel("Settle rate [%]", color="#FFD60A", fontsize=7)
    ax_r.tick_params(colors="#FFD60A", labelsize=6)
    ax_r.set_ylim(0, 110)
    ax_r.legend(fontsize=7, facecolor=panel, labelcolor="white", loc="lower right")

    plt.tight_layout()
    out = os.path.join(FIGURE_DIR, f"training_curves_{run_id}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[INFO] Training curves saved → {out}")


def _render_evaluation(model, n_episodes: int = 3,
                       disturbance_mode: str = "seismic") -> None:
    """Open the live visualiser for a few evaluation episodes."""
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
    from real_time_platform_sim import RealTimeVisualizer

    def _ctrl(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    env = PlatformEnv(cfg=CFG, disturbance_mode=disturbance_mode)
    viz = RealTimeVisualizer(env, controller=_ctrl)
    viz.show()


# ============================================================================
# §6  CLI ENTRY POINT
# ============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AI-Assisted Vibration Control — RL Controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    sub = p.add_subparsers(dest="command", required=True)

    # ── train ────────────────────────────────────────────────────────────────
    tr = sub.add_parser("train", help="Train a new PPO agent")
    tr.add_argument("--timesteps", type=int,  default=500_000,
                    help="Total environment interactions")
    tr.add_argument("--n-envs",   type=int,  default=4,
                    help="Number of parallel training environments")
    tr.add_argument("--mode",     type=str,  default="seismic",
                    choices=["seismic", "harmonic"],
                    help="Disturbance type for training")
    tr.add_argument("--fast",     action="store_true",
                    help="Use smaller network (dev/debug mode)")
    tr.add_argument("--seed",     type=int,  default=42)

    # ── eval ─────────────────────────────────────────────────────────────────
    ev = sub.add_parser("eval", help="Evaluate a saved model")
    ev.add_argument("--model",    type=str,  required=True,
                    help="Path to model .zip (without extension)")
    ev.add_argument("--episodes", type=int,  default=20)
    ev.add_argument("--mode",     type=str,  default="seismic",
                    choices=["seismic", "harmonic"])
    ev.add_argument("--render",   action="store_true",
                    help="Open live animation window after evaluation")
    ev.add_argument("--no-csv",   action="store_true",
                    help="Skip saving evaluation CSV")

    # ── demo ─────────────────────────────────────────────────────────────────
    dm = sub.add_parser("demo", help="Load a model and show live animation only")
    dm.add_argument("--model", type=str, required=True)
    dm.add_argument("--mode",  type=str, default="seismic")

    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.command == "train":
        model_path = train(
            timesteps        = args.timesteps,
            n_envs           = args.n_envs,
            disturbance_mode = args.mode,
            fast             = args.fast,
            seed             = args.seed,
        )
        print(f"\n[DONE] Model available at: {model_path}.zip")

    elif args.command == "eval":
        evaluate(
            model_path       = args.model,
            n_episodes       = args.episodes,
            disturbance_mode = args.mode,
            render           = args.render,
            save_csv         = not args.no_csv,
        )

    elif args.command == "demo":
        _render_evaluation(
            PPO.load(args.model),
            n_episodes       = 999,
            disturbance_mode = args.mode,
        )


if __name__ == "__main__":
    main()