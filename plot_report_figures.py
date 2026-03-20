#!/usr/bin/env python3
"""
plot_report_figures.py
----------------------
Generates Figures 8, 9, and 10 for the ME 295B project report
from data/evaluation_results.csv.

Usage:
    python3 plot_report_figures.py

Outputs (saved to figures/):
    fig08_settling_time.png        -- Figure 8: T_norm per episode, PPO vs PD
    fig09_energy_velocity.png      -- Figure 9: E_norm bar comparison
    fig10_reward_breakdown.png     -- Figure 10: Reward comparison + decomposition
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.makedirs("figures", exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("data/evaluation_results.csv")
rl = df[df["controller"] == "RL"].reset_index(drop=True)
pd_df = df[df["controller"] == "PD"].reset_index(drop=True)

n_ep = len(rl)
episodes = np.arange(1, n_ep + 1)

# ── Style ──────────────────────────────────────────────────────────────────────
BLUE   = "#2E86AB"
RED    = "#E74C3C"
GREEN  = "#27AE60"
GOLD   = "#F4A261"
DARK   = "#1C1C1E"
PANEL  = "#2C2C2E"
MUTED  = "#8E8E93"
WHITE  = "#F5F5F7"
GRID   = "#3A3A3C"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": PANEL,
    "figure.facecolor": DARK,
    "text.color": WHITE,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DPI = 200

# =============================================================================
# FIGURE 8 — Settling Time T_norm per episode
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 5), facecolor=DARK)
ax.set_facecolor(PANEL)

x = np.arange(n_ep)
w = 0.35

bars_rl = ax.bar(x - w/2, rl["T_norm"], w, color=BLUE, label="PPO Agent", alpha=0.9, zorder=3)
bars_pd = ax.bar(x + w/2, pd_df["T_norm"], w, color=RED,  label="PD Baseline", alpha=0.9, zorder=3)

# Annotate the two RL outlier episodes
for i, v in enumerate(rl["T_norm"]):
    if v > 0.05:
        ax.text(i - w/2, v + 0.01, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8, color=GOLD, fontweight="bold")

ax.set_xlabel("Episode Number", fontsize=11, color=MUTED)
ax.set_ylabel("T_norm  (lower = faster settling)", fontsize=11, color=MUTED)
ax.set_title("Figure 8: Settling Time T_norm per Episode — PPO Agent vs. PD Baseline",
             fontsize=12, color=WHITE, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels([str(i+1) for i in range(n_ep)], fontsize=9)
ax.set_ylim(0, 1.15)
ax.axhline(1.0, color=RED, linewidth=0.8, linestyle=":", alpha=0.7, label="Timeout (T_norm = 1.0)")
ax.yaxis.grid(True); ax.set_axisbelow(True)
legend = ax.legend(fontsize=10, facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE, loc="upper left")

# Stats annotation
ax.text(0.98, 0.96,
        f"PPO:  mean = {rl['T_norm'].mean():.4f} ± {rl['T_norm'].std():.4f}\n"
        f"PD:   mean = {pd_df['T_norm'].mean():.4f} ± {pd_df['T_norm'].std():.4f}\n"
        f"Settle rate:  PPO 100%  |  PD 0%",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color=MUTED,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=DARK, edgecolor=GRID, alpha=0.8))

ax.text(0.5, -0.12,
        "Source: data/evaluation_results.csv  |  20 deterministic evaluation episodes  |"
        "  Settling criterion: |θx|, |θy| < 0.5° held for 1 s",
        transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=MUTED, style="italic")

plt.tight_layout()
out8 = "figures/fig08_settling_time.png"
fig.savefig(out8, dpi=DPI, bbox_inches="tight", facecolor=DARK)
plt.close(fig)
print(f"[INFO] Saved → {out8}")

# =============================================================================
# FIGURE 9 — Actuation Energy E_norm comparison
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=DARK)

# Left: bar chart mean E_norm
ax1 = axes[0]; ax1.set_facecolor(PANEL)
controllers = ["PPO Agent", "PD Baseline"]
e_means = [rl["E_norm"].mean(), pd_df["E_norm"].mean()]
e_stds  = [rl["E_norm"].std(),  pd_df["E_norm"].std()]
colors  = [BLUE, RED]
bars = ax1.bar(controllers, e_means, color=colors, alpha=0.9, width=0.45,
               yerr=e_stds, capsize=6, error_kw={"color": WHITE, "linewidth": 1.5},
               zorder=3)
for bar, val in zip(bars, e_means):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.003,
             f"{val:.4f}", ha="center", va="bottom", fontsize=12,
             color=WHITE, fontweight="bold")
ax1.set_ylabel("E_norm  (lower = less energy)", fontsize=11, color=MUTED)
ax1.set_title("Mean Normalised Actuation Energy", fontsize=11, color=WHITE, fontweight="bold")
ax1.set_ylim(0, 0.35)
ax1.yaxis.grid(True); ax1.set_axisbelow(True)
# Improvement annotation
reduction = (1 - e_means[0]/e_means[1]) * 100
ax1.annotate(f"−{reduction:.1f}%", xy=(0.5, (e_means[0]+e_means[1])/2),
             xycoords=("axes fraction", "data"),
             ha="center", va="center", fontsize=13, color=GREEN, fontweight="bold")

# Right: episode-by-episode E_norm line plot
ax2 = axes[1]; ax2.set_facecolor(PANEL)
ax2.plot(episodes, rl["E_norm"],   color=BLUE, linewidth=1.8, marker="o", markersize=4,
         label="PPO Agent", alpha=0.9, zorder=3)
ax2.plot(episodes, pd_df["E_norm"], color=RED,  linewidth=1.8, marker="s", markersize=4,
         label="PD Baseline", alpha=0.9, zorder=3)
ax2.fill_between(episodes, rl["E_norm"], pd_df["E_norm"], alpha=0.12, color=GREEN,
                 label="PPO energy savings")
ax2.set_xlabel("Episode Number", fontsize=11, color=MUTED)
ax2.set_ylabel("E_norm", fontsize=11, color=MUTED)
ax2.set_title("E_norm per Episode", fontsize=11, color=WHITE, fontweight="bold")
ax2.set_ylim(0.18, 0.32)
ax2.yaxis.grid(True); ax2.set_axisbelow(True)
ax2.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE)

fig.suptitle("Figure 9: Normalised Actuation Energy E_norm — PPO Agent vs. PD Baseline",
             fontsize=12, color=WHITE, fontweight="bold", y=1.01)
fig.text(0.5, -0.03,
         "Source: data/evaluation_results.csv  |  E_ref = 80 A·s  |  20 evaluation episodes",
         ha="center", fontsize=7.5, color=MUTED, style="italic")

plt.tight_layout()
out9 = "figures/fig09_energy.png"
fig.savefig(out9, dpi=DPI, bbox_inches="tight", facecolor=DARK)
plt.close(fig)
print(f"[INFO] Saved → {out9}")

# =============================================================================
# FIGURE 10 — Reward breakdown and comparative analysis
# =============================================================================
fig = plt.figure(figsize=(13, 6), facecolor=DARK)
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

# Left: episode reward per episode (RL + PD)
ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor(PANEL)
ax1.plot(episodes, rl["reward"],     color=BLUE, linewidth=2.0, marker="o",
         markersize=4, label="PPO Agent", zorder=3)
ax1.plot(episodes, pd_df["reward"],  color=RED,  linewidth=2.0, marker="s",
         markersize=4, label="PD Baseline", zorder=3)
ax1.axhline(0, color=GRID, linewidth=0.8, linestyle=":")
ax1.fill_between(episodes, rl["reward"], pd_df["reward"], alpha=0.10, color=GREEN)
ax1.set_xlabel("Episode Number", fontsize=11, color=MUTED)
ax1.set_ylabel("Episode Reward R_ep", fontsize=11, color=MUTED)
ax1.set_title("Episode Reward per Evaluation Episode", fontsize=11, color=WHITE, fontweight="bold")
ax1.set_ylim(-1.05, 1.05)
ax1.yaxis.grid(True); ax1.set_axisbelow(True)
ax1.legend(fontsize=10, facecolor=PANEL, edgecolor=GRID, labelcolor=WHITE)
ax1.text(0.98, 0.05,
         f"PPO:  {rl['reward'].mean():.4f} ± {rl['reward'].std():.4f}\n"
         f"PD:  {pd_df['reward'].mean():.4f} ± {pd_df['reward'].std():.4f}",
         transform=ax1.transAxes, ha="right", va="bottom",
         fontsize=9, color=MUTED,
         bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK, edgecolor=GRID, alpha=0.8))

# Right: PPO reward decomposition (stacked horizontal bars)
ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor(PANEL)

# Approximate mean component values
T_pen  = 0.5 * rl["T_norm"].mean()   # wt × T
E_pen  = 0.3 * rl["E_norm"].mean()   # wa × E
V_pen  = 0.2 * rl["V_norm"].mean()   # wv × V
bonus  = rl["bonus"].mean()           # B × I()
net    = bonus - T_pen - E_pen - V_pen

components = [
    ("Bonus B·I()", bonus, GREEN),
    ("V penalty (−wv·V)", -V_pen, GOLD),
    ("E penalty (−wa·E)", -E_pen, BLUE),
    ("T penalty (−wt·T)", -T_pen, RED),
]

y_pos = [3, 2, 1, 0]
labels = [c[0] for c in components]
values = [c[1] for c in components]
colors_bar = [c[2] for c in components]

bars = ax2.barh(y_pos, values, color=colors_bar, alpha=0.9, height=0.55, zorder=3)
for bar, val in zip(bars, values):
    xpos = val + 0.015 if val > 0 else val - 0.015
    ha   = "left" if val > 0 else "right"
    ax2.text(xpos, bar.get_y() + bar.get_height()/2,
             f"{val:+.4f}", va="center", ha=ha, fontsize=11,
             color=WHITE, fontweight="bold")

ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels, fontsize=10, color=WHITE)
ax2.axvline(0, color=GRID, linewidth=1.0)
ax2.set_xlim(-0.35, 1.15)
ax2.set_xlabel("Contribution to R_ep", fontsize=11, color=MUTED)
ax2.set_title("PPO Reward Decomposition (mean)", fontsize=11, color=WHITE, fontweight="bold")
ax2.xaxis.grid(True); ax2.set_axisbelow(True)

# Net reward annotation
ax2.axvline(net, color=WHITE, linewidth=1.5, linestyle="--", alpha=0.6)
ax2.text(net + 0.02, -0.55, f"Net R_ep ≈ {net:.4f}",
         fontsize=10, color=WHITE, fontweight="bold")

fig.suptitle("Figure 10: Episode Reward Breakdown and Comparative Analysis — PPO Agent vs. PD Baseline",
             fontsize=12, color=WHITE, fontweight="bold", y=1.01)
fig.text(0.5, -0.04,
         "Source: data/evaluation_results.csv  |  wt = 0.5,  wa = 0.3,  wv = 0.2,  B = 1.0  |  20 evaluation episodes",
         ha="center", fontsize=7.5, color=MUTED, style="italic")

plt.tight_layout()
out10 = "figures/fig10_reward_breakdown.png"
fig.savefig(out10, dpi=DPI, bbox_inches="tight", facecolor=DARK)
plt.close(fig)
print(f"[INFO] Saved → {out10}")

print("\n[DONE] All report figures saved to figures/")
print(f"  Figure 8  → {out8}")
print(f"  Figure 9  → {out9}")
print(f"  Figure 10 → {out10}")
