#!/usr/bin/env python3
"""
plot_training_curve.py
----------------------
Generates a publication-ready training curve PNG from episode_metrics.csv.

Usage (run from your project root):
    python3 plot_training_curve.py

Looks for the most recent episode_metrics.csv under logs/ automatically,
or you can pass a path explicitly:
    python3 plot_training_curve.py --csv logs/ppo_seismic_1773805456/episode_metrics.csv
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — no display needed, just saves PNG
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Find CSV ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv", default=None, help="Path to episode_metrics.csv")
parser.add_argument("--out", default=None, help="Output PNG path (optional)")
args = parser.parse_args()

if args.csv:
    csv_path = args.csv
else:
    # Auto-find the most recently modified episode_metrics.csv under logs/
    matches = glob.glob("logs/**/episode_metrics.csv", recursive=True)
    if not matches:
        raise FileNotFoundError(
            "No episode_metrics.csv found under logs/. "
            "Pass --csv path/to/episode_metrics.csv explicitly."
        )
    csv_path = max(matches, key=os.path.getmtime)

print(f"[INFO] Reading: {csv_path}")
run_id = os.path.basename(os.path.dirname(csv_path))

df = pd.read_csv(csv_path)
print(f"[INFO] {len(df)} episodes loaded.")
print(f"[INFO] Columns: {list(df.columns)}")

# ── Rolling mean helper ───────────────────────────────────────────────────────
def roll(arr, w=50):
    arr = np.array(arr, dtype=float)
    w = min(w, len(arr))
    out = np.convolve(arr, np.ones(w) / w, mode="full")
    trim = (len(out) - len(arr)) // 2
    return out[trim: trim + len(arr)]

eps = np.arange(1, len(df) + 1)

# ── Map column names flexibly ─────────────────────────────────────────────────
col = lambda *names: next((c for c in names if c in df.columns), None)

reward_col  = col("reward", "ep_reward", "total_reward", "R")
T_col       = col("T_norm", "t_norm", "settling_time")
E_col       = col("E_norm", "e_norm", "energy")
V_col       = col("V_norm", "v_norm", "peak_velocity")
settle_col  = col("settled", "settle", "success")

# ── Plot ──────────────────────────────────────────────────────────────────────
DARK   = "#1C1C1E"
PANEL  = "#2C2C2E"
GRID   = "#3A3A3C"
WHITE  = "#F5F5F7"
MUTED  = "#8E8E93"

BLUE   = "#378ADD"
GREEN  = "#639922"
AMBER  = "#EF9F27"
CORAL  = "#D85A30"
TEAL   = "#1D9E75"

fig = plt.figure(figsize=(13, 8), facecolor=DARK)
fig.suptitle(f"PPO Training Curves  —  {run_id}",
             color=WHITE, fontsize=12, fontweight="bold", y=0.97)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                       left=0.07, right=0.97, top=0.91, bottom=0.08)

def setup_ax(ax, title, ylabel):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=WHITE, fontsize=9.5, fontweight="bold", pad=6)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.set_xlabel("Episode", color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

# ── Panel 1: Episode reward ───────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
setup_ax(ax1, "Episode Reward", "Reward")
if reward_col:
    vals = df[reward_col].values
    ax1.plot(eps, vals, color=BLUE, alpha=0.25, linewidth=0.6)
    ax1.plot(eps, roll(vals), color=BLUE, linewidth=2.0, label="50-ep mean")
    ax1.fill_between(eps, roll(vals), alpha=0.12, color=BLUE)
    ax1.axhline(0, color=GRID, linewidth=0.8, linestyle=":")
    ax1.legend(fontsize=7.5, labelcolor=MUTED,
               facecolor=PANEL, edgecolor=GRID, framealpha=0.8)
else:
    ax1.text(0.5, 0.5, "reward column not found", transform=ax1.transAxes,
             ha="center", color=MUTED, fontsize=9)

# ── Panel 2: Settling time ────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
setup_ax(ax2, "Normalised Settling Time  T_norm", "T_norm  (0 = fast, 1 = timeout)")
if T_col:
    vals = df[T_col].values
    ax2.plot(eps, vals, color=AMBER, alpha=0.25, linewidth=0.6)
    ax2.plot(eps, roll(vals), color=AMBER, linewidth=2.0, label="50-ep mean")
    ax2.fill_between(eps, roll(vals), alpha=0.12, color=AMBER)
    ax2.set_ylim(-0.05, 1.1)
    ax2.axhline(1.0, color=CORAL, linewidth=0.8, linestyle=":", label="timeout")
    ax2.legend(fontsize=7.5, labelcolor=MUTED,
               facecolor=PANEL, edgecolor=GRID, framealpha=0.8)

# ── Panel 3: Actuation energy ─────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
setup_ax(ax3, "Normalised Actuation Energy  E_norm", "E_norm  (lower = more efficient)")
if E_col:
    vals = df[E_col].values
    ax3.plot(eps, vals, color=TEAL, alpha=0.25, linewidth=0.6)
    ax3.plot(eps, roll(vals), color=TEAL, linewidth=2.0, label="50-ep mean")
    ax3.fill_between(eps, roll(vals), alpha=0.12, color=TEAL)
    ax3.legend(fontsize=7.5, labelcolor=MUTED,
               facecolor=PANEL, edgecolor=GRID, framealpha=0.8)

# ── Panel 4: Settle rate ──────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
setup_ax(ax4, "Rolling Settle Rate  (50-ep window)", "Settle rate  [%]")
if settle_col:
    settled = (df[settle_col].astype(float).values > 0.5).astype(float)
    ax4.plot(eps, roll(settled, 50) * 100, color=GREEN, linewidth=2.0,
             label="50-ep settle %")
    ax4.fill_between(eps, roll(settled, 50) * 100, alpha=0.12, color=GREEN)
    ax4.set_ylim(-5, 105)
    ax4.axhline(100, color=GRID, linewidth=0.8, linestyle=":")
    ax4.legend(fontsize=7.5, labelcolor=MUTED,
               facecolor=PANEL, edgecolor=GRID, framealpha=0.8)
elif reward_col:
    # Fallback: peak velocity norm
    if V_col:
        vals = df[V_col].values
        ax4.plot(eps, vals, color=CORAL, alpha=0.25, linewidth=0.6)
        ax4.plot(eps, roll(vals), color=CORAL, linewidth=2.0, label="50-ep mean")
        ax4.fill_between(eps, roll(vals), alpha=0.12, color=CORAL)
        ax4.set_title("Normalised Peak Velocity  V_norm", color=WHITE,
                      fontsize=9.5, fontweight="bold", pad=6)
        ax4.set_ylabel("V_norm", color=MUTED, fontsize=8)
        ax4.legend(fontsize=7.5, labelcolor=MUTED,
                   facecolor=PANEL, edgecolor=GRID, framealpha=0.8)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)
out_path = args.out or f"figures/training_curves_{run_id}.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=DARK)
print(f"[INFO] Saved → {out_path}")
plt.close(fig)
