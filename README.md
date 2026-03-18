# AI-Assisted Vibration Control of a Platform
**ME 295B – Master's Project | Mechanical Engineering | San José State University**
**Author:** Khoi Le | May 2026
**Committee Chair:** Dr. Freidoon Barez

---

## Project Overview

This project develops and evaluates an AI-assisted active vibration control scheme for a
four-corner actuated platform.  A Proximal Policy Optimisation (PPO) reinforcement-
learning agent issues continuous commands to four independent electric actuators to
restore and maintain the platform's horizontal orientation when external seismic or
harmonic disturbances are applied.  Platform orientation and angular velocity are
provided by a central IMU; per-motor current draw is monitored by four current sensors.

The reward function penalises long settling times, high actuation effort, and excessive
angular motion, and awards a scalar bonus when both settling time and peak angular
velocity fall below pre-defined thresholds:

```
R_ep = −(T·wt + E·wa + V·wv) + B·I(T ≤ T_tol ∧ V ≤ V_tol)
```

where `wt + wa + wv = 1`, default weights `wt = 0.5`, `wa = 0.3`, `wv = 0.2`, `B = 1.0`.

---

## Repository Layout

```
vibration_control/
├── real_time_platform_sim.py        # Physics engine + Gym env + live visualiser
├── controller_reinforcement_agent.py# PPO agent, training loop, evaluator
├── generate_csv.py                  # One-shot script to (re)generate training CSV
├── requirements.txt                 # Python package dependencies
├── README.md                        # This file
│
├── data/
│   ├── vibration_training_data.csv  # 400 k rows, 40 episodes × 10 s at 1 ms
│   └── evaluation_results.csv       # Created automatically during evaluation
│
├── models/                          # PPO checkpoints saved during training
├── logs/                            # TensorBoard event files
└── figures/                         # Training-curve PNG exports
```

---

## Platform Specification

| Parameter | Value |
|---|---|
| Platform geometry | 50 mm × 50 mm rigid square plate |
| Platform mass | 0.5 kg |
| Corner spring stiffness K | 500 N/m |
| Corner damping C | 2 N·s/m |
| Actuator travel | 0 – 10 mm |
| Actuator bandwidth | 50 Hz (first-order lag) |
| IMU noise (1σ) | 0.05° |
| Current sensor noise (1σ) | 0.02 A |
| Simulation timestep Δt | 1 ms |
| Episode length | 10 s |

**Corner numbering (viewed from above):**

```
  C1 FL (−25,+25) mm    C2 FR (+25,+25) mm
        ┌─────────────────┐
        │                 │
        │    IMU (0,0)    │
        │                 │
        └─────────────────┘
  C4 RL (−25,−25) mm    C3 RR (+25,−25) mm
```

---

## Quick-Start

### 1  Install dependencies

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2  Run the live simulation demo

```bash
# PD-controller demo with real-time 5-panel animation + 3-D platform view
python real_time_platform_sim.py

# Same demo with random actuator commands (stress test)
python real_time_platform_sim.py --random
```

The visualiser shows five live panels:

| Panel | Content |
|---|---|
| 1 | Roll θx and pitch θy [°] with ±0.5° tolerance band |
| 2 | Angular rates ωx, ωy [°/s] |
| 3 | Motor currents I1–I4 [A] |
| 4 | Corner shaft displacements x1–x4 [mm] |
| 5 | 3-D perspective view of the tilting platform |

### 3  Train the PPO agent

```bash
# Single-process training (500 k timesteps, recommended for first run)
python controller_reinforcement_agent.py train --timesteps 500000

# Parallel training across 4 environments (faster)
python controller_reinforcement_agent.py train --timesteps 1000000 --n-envs 4

# Launch TensorBoard to monitor training live
tensorboard --logdir logs/
```

Checkpoints are saved every 50 k steps under `models/`.
The best model (by mean episode reward on the evaluation environment) is saved as
`models/<run_id>_best/best_model.zip`.

### 4  Evaluate a trained model

```bash
python controller_reinforcement_agent.py eval \
    --model models/<run_id>_best/best_model \
    --episodes 20 \
    --render
```

A CSV summary is written to `data/evaluation_results.csv` and a training-curve PNG
is saved to `figures/`.

### 5  Run demo animation only

```bash
python controller_reinforcement_agent.py demo \
    --model models/<run_id>_best/best_model
```

### 6  Re-generate the training CSV

```bash
python generate_csv.py
# Produces data/vibration_training_data.csv  (400 000 rows)
```

---

## Training Data Format

`data/vibration_training_data.csv` — 400 000 rows × 12 columns:

| Column | Unit | Description |
|---|---|---|
| `x_angle` | ° | Roll angle θx (IMU, with noise) |
| `y_angle` | ° | Pitch angle θy (IMU, with noise) |
| `xdot` | °/s | Roll rate ωx |
| `ydot` | °/s | Pitch rate ωy |
| `I1` | A | Motor current – corner C1 FL |
| `I2` | A | Motor current – corner C2 FR |
| `I3` | A | Motor current – corner C3 RR |
| `I4` | A | Motor current – corner C4 RL |
| `x1` | mm | Shaft displacement – corner C1 FL |
| `x2` | mm | Shaft displacement – corner C2 FR |
| `x3` | mm | Shaft displacement – corner C3 RR |
| `x4` | mm | Shaft displacement – corner C4 RL |

Generated from 40 simulated episodes (10 s each at Δt = 1 ms) using a mix of
AR(1) seismic disturbance profiles and multi-frequency harmonic disturbances.
Episodes alternate between PD-controlled and random-action rollouts to ensure
coverage of both stable and exploratory state-action regions.

---

## Reward Function Parameters

| Symbol | Value | Description |
|---|---|---|
| `wt` | 0.5 | Weight: settling-time penalty |
| `wa` | 0.3 | Weight: actuation-effort penalty |
| `wv` | 0.2 | Weight: peak angular-velocity penalty |
| `B` | 1.0 | Bonus scalar |
| `θ_tol` | 0.5° | Angular tolerance for settling |
| `Δt_hold` | 1 s | Required dwell time inside tolerance |
| `T_ref` | 10 s | Reference (worst-case) settling time |
| `E_ref` | 80 A·s | Reference cumulative current integral |
| `V_ref` | 30 °/s | Reference peak angular velocity |

---

## PPO Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 3 × 10⁻⁴ |
| n_steps | 2 048 |
| Batch size | 64 |
| n_epochs | 10 |
| γ (discount) | 0.99 |
| λ (GAE) | 0.95 |
| Network architecture | π = [256, 256, 128], V = [256, 256, 128], Tanh |

---

## Evaluation Metrics

1. **Settling time T** — mean ± std, median, worst-case across test episodes;
   percentage reduction vs. PD baseline.
2. **Actuation energy E** — cumulative |I| integral per episode (A·s);
   ensemble statistics and percent change vs. baseline.
3. **Excessive motion V** — peak angular speed √(ωx² + ωy²) per episode (°/s);
   absolute and percentage reduction, time-above-threshold.

---

## Citation

> K. Le, "AI-Assisted Vibration Control of a Platform," M.S. Project,
> Dept. of Mechanical Engineering, San José State University, May 2026.

---

## License

For academic use only.  All rights reserved © 2026 Khoi Le.
