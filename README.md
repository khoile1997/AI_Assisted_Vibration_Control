# Active Vibration Control Platform - RL Controller

## Master's Thesis Project: Reinforcement Learning for Platform Stabilization

This repository contains an improved implementation of a reinforcement learning controller for active vibration control of a rectangular platform with 4 electromagnetic actuators.

## 📋 Table of Contents

- [System Overview](#system-overview)
- [Key Improvements](#key-improvements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [System Architecture](#system-architecture)
- [Configuration](#configuration)
- [Results & Evaluation](#results--evaluation)
- [Troubleshooting](#troubleshooting)

## 🎯 System Overview

The system consists of:

### Hardware (Simulated)
- **Platform**: Rectangular horizontal platform
- **Actuators**: 4 electromagnetic motors at corners (current-controlled)
- **Sensors**: 
  - Central IMU measuring roll (φ), pitch (θ), and angular rates
  - 4 current sensors (one per motor)

### Software Components
1. **Simulator** (`real_time_platform_sim.py`): Real-time physics simulation with TCP/IP control interface
2. **RL Agent** (`controller_reinforcement_agent_improved.py`): REINFORCE-based policy gradient controller
3. **Visualization**: Interactive system diagrams and training plots

### State & Action Spaces
- **State (Observation)**: `[φ, θ, φ̇, θ̇]` - angles and angular rates
- **Action**: `[I₁, I₂, I₃, I₄]` - motor currents in Amperes
- **Control Frequency**: 20 Hz (dt = 0.05s)

## ✨ Key Improvements

### Code Quality
1. **Type Hints & Documentation**: Comprehensive docstrings and type annotations
2. **Data Classes**: Structured data with `PlatformState` and `EpisodeMetrics`
3. **Error Handling**: Robust exception handling and connection verification
4. **Logging**: Professional logging with different levels

### Architecture
1. **Modular Design**: Separated concerns (client, policy, agent)
2. **Better Network**: Improved policy network with proper initialization
3. **Enhanced Training**: Normalized returns, gradient clipping, best model tracking
4. **Evaluation Tools**: Comprehensive metrics and statistics

### Features
1. **Auto-retry**: Automatic retry for network communication failures
2. **Checkpointing**: Save/load models with training state
3. **Progress Tracking**: Recent reward history and moving averages
4. **Flexible Configuration**: Command-line arguments for all hyperparameters

## 🚀 Installation

### Requirements
```bash
pip install numpy pandas torch
```

Or use the provided requirements file:

```bash
pip install -r requirements.txt
```

### Requirements.txt
```
numpy>=1.21.0
pandas>=1.3.0
torch>=1.10.0
```

## 🏁 Quick Start

### 1. Prepare Initial Conditions

Create `vibration_data.csv` with disturbance scenarios:

```csv
phi,theta,phi_dot,theta_dot
0.05,0.03,0.1,0.05
-0.04,0.06,-0.08,0.12
0.08,-0.02,0.15,-0.1
...
```

### 2. Start Simulator

In one terminal:
```bash
python real_time_platform_sim.py
```

### 3. Train Agent

In another terminal:
```bash
python controller_reinforcement_agent_improved.py --mode train \
    --epochs 200 \
    --eps_per_epoch 8 \
    --lr 1e-3 \
    --model_path policy.pth
```

### 4. Evaluate

```bash
python controller_reinforcement_agent_improved.py --mode eval \
    --model_path policy.pth \
    --n_eval 50
```

## 📖 Usage

### Training

Basic training:
```bash
python controller_reinforcement_agent_improved.py --mode train
```

Advanced training with custom hyperparameters:
```bash
python controller_reinforcement_agent_improved.py \
    --mode train \
    --epochs 500 \
    --eps_per_epoch 16 \
    --lr 5e-4 \
    --hidden 256 256 \
    --exploration_noise 0.1 \
    --w_time 1.0 \
    --w_effort 0.2 \
    --w_vibration 0.5 \
    --bonus 100.0 \
    --model_path models/policy_v2.pth
```

### Evaluation

Evaluate trained policy:
```bash
python controller_reinforcement_agent_improved.py \
    --mode eval \
    --model_path policy.pth \
    --n_eval 100
```

Load and continue training:
```bash
python controller_reinforcement_agent_improved.py \
    --mode train \
    --load_model \
    --model_path policy.pth \
    --epochs 100
```

### Command-Line Arguments

#### Connection
- `--host`: Simulator host (default: `127.0.0.1`)
- `--port`: Simulator port (default: `5005`)

#### Data
- `--csv`: Path to initial conditions CSV (default: `vibration_data.csv`)

#### Control
- `--dt`: Control timestep in seconds (default: `0.05`)
- `--max_time`: Max episode duration in seconds (default: `8.0`)
- `--i_min`: Minimum motor current in A (default: `-2.0`)
- `--i_max`: Maximum motor current in A (default: `2.0`)

#### Thresholds
- `--angle_thresh`: Recovery angle threshold in rad (default: `0.01`)
- `--rate_thresh`: Recovery rate threshold in rad/s (default: `0.05`)
- `--motion_rate_thresh`: Excessive motion threshold in rad/s (default: `0.2`)
- `--recovery_time_thresh`: Fast recovery time for bonus in s (default: `1.0`)

#### Reward Weights
- `--w_time`: Time penalty weight (default: `1.0`)
- `--w_effort`: Actuation effort penalty weight (default: `0.1`)
- `--w_vibration`: Vibration time penalty weight (default: `0.5`)
- `--bonus`: Fast recovery bonus reward (default: `50.0`)

#### Training
- `--epochs`: Number of training epochs (default: `200`)
- `--eps_per_epoch`: Episodes per epoch (default: `8`)
- `--lr`: Learning rate (default: `1e-3`)
- `--hidden`: Hidden layer sizes (default: `128 128`)
- `--exploration_noise`: Exploration noise std (default: `0.05`)

#### Model
- `--model_path`: Path for model checkpoints (default: `policy.pth`)
- `--load_model`: Load existing model before train/eval
- `--n_eval`: Number of evaluation episodes (default: `20`)

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PLATFORM DYNAMICS                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │   Rectangular Platform with 4 Corner Actuators      │   │
│  │                                                      │   │
│  │   M1 ●────────────● M2    ⚡ Electromagnetic        │   │
│  │      │            │          Motors                  │   │
│  │      │    📡 IMU  │       📊 State:                 │   │
│  │      │   (φ,θ)    │          [φ, θ, φ̇, θ̇]         │   │
│  │      │            │                                  │   │
│  │   M3 ●────────────● M4    ⚙️  Currents:             │   │
│  │                              [I₁, I₂, I₃, I₄]       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ TCP/IP (127.0.0.1:5005)
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    RL CONTROLLER                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Policy Network π(a|s)                   │   │
│  │  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐        │   │
│  │  │Input │──▶│ FC   │──▶│ FC   │──▶│Output│         │   │
│  │  │ 4D   │   │ 128  │   │ 128  │   │ 4D   │         │   │
│  │  └──────┘   └──────┘   └──────┘   └──────┘         │   │
│  │                                                      │   │
│  │  🎲 Gaussian Policy: a ~ N(μ(s), σ²)               │   │
│  │  📈 REINFORCE Update: ∇J = E[∇log π(a|s) × R]      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Measurement**: IMU measures platform state `[φ, θ, φ̇, θ̇]`
2. **Communication**: State sent via TCP/IP to RL agent
3. **Policy Evaluation**: Neural network computes action distribution
4. **Action Selection**: Sample currents `[I₁, I₂, I₃, I₄]` from policy
5. **Actuation**: Apply currents to electromagnetic motors
6. **Physics Update**: Platform dynamics evolve
7. **Trajectory Recording**: Store `(state, action, log_prob)` for training
8. **Episodic Learning**: Update policy with REINFORCE at episode end

## ⚙️ Configuration

### Reward Function

The episodic reward is designed to encourage fast recovery with minimal energy:

```
R = -(T × w_t + E × w_a + V × w_v) + B × I

Where:
  T = Recovery time (s) - or max_episode_time if not recovered
  E = Actuation effort = ∫ Σ(I_i²) dt
  V = Vibration time (s) - time with |φ̇| or |θ̇| > 0.2 rad/s
  I = Recovery indicator (1 if recovered within 1s, else 0)
  
Weights:
  w_t = 1.0   (time penalty)
  w_a = 0.1   (effort penalty)
  w_v = 0.5   (vibration penalty)
  B = 50.0    (recovery bonus)
```

### Recovery Criteria

Platform is considered "recovered" when:
- `|φ| ≤ 0.01 rad` AND `|θ| ≤ 0.01 rad` (angles small)
- `|φ̇| ≤ 0.05 rad/s` AND `|θ̇| ≤ 0.05 rad/s` (rates small)

Fast recovery bonus awarded if recovered within 1.0 second.

### Hyperparameter Tuning Tips

1. **Learning Rate**: Start with `1e-3`, reduce if unstable, increase if slow
2. **Network Size**: Larger networks (`[256, 256]`) for complex dynamics
3. **Exploration Noise**: Higher (`0.1-0.2`) early, lower (`0.01-0.05`) later
4. **Reward Weights**: 
   - Increase `w_t` to prioritize speed
   - Increase `w_a` to save energy
   - Increase `w_v` to reduce vibration
   - Increase `B` to emphasize fast recovery

## 📊 Results & Evaluation

### Training Metrics

During training, monitor:
- **Average Reward**: Should increase over epochs
- **Recovery Rate**: Percentage of episodes with `I=1`
- **Mean Recovery Time**: Should decrease
- **Policy Loss**: May fluctuate but trend downward

Example output:
```
[Epoch 100/200] Episode 1/8: R=35.2, T=0.8s, Recovered=True
[Epoch 100/200] Episode 2/8: R=28.5, T=1.2s, Recovered=True
...
Epoch 100 Summary: Loss=0.0234, Reward=31.4±5.2
New best model! Reward=31.4
```

### Evaluation Output

```
[Eval 1/50] R=42.3, T=0.65s, Recovered=True
[Eval 2/50] R=38.7, T=0.85s, Recovered=True
...

Evaluation Summary:
                T          E          V          I      reward
count   50.000000  50.000000  50.000000  50.000000   50.000000
mean     0.856000   2.341000   0.234000   0.940000   38.234000
std      0.234000   0.567000   0.098000   0.239000    6.123000
min      0.450000   1.234000   0.050000   0.000000   22.340000
25%      0.678000   1.890000   0.167000   1.000000   34.567000
50%      0.834000   2.234000   0.223000   1.000000   39.123000
75%      1.023000   2.678000   0.289000   1.000000   42.890000
max      1.890000   4.123000   0.567000   1.000000   48.234000

Recovery Rate: 94.0%
```

### Performance Benchmarks

Good performance indicators:
- **Recovery Rate**: >90%
- **Mean Recovery Time**: <1.0s
- **Mean Reward**: >30.0
- **Actuation Effort**: <3.0

## 🐛 Troubleshooting

### Connection Issues

**Problem**: `Cannot connect to simulator`
```
Solution:
1. Verify simulator is running: ps aux | grep real_time_platform_sim
2. Check port availability: netstat -an | grep 5005
3. Try alternative port: --port 5006
```

### Training Instability

**Problem**: Reward oscillates wildly or diverges
```
Solutions:
1. Reduce learning rate: --lr 5e-4 or --lr 1e-4
2. Reduce exploration: --exploration_noise 0.01
3. Check initial conditions in CSV for outliers
4. Verify simulator is stable independently
```

### Poor Recovery Performance

**Problem**: Low recovery rate even after training
```
Solutions:
1. Increase training duration: --epochs 500
2. Increase batch size: --eps_per_epoch 16
3. Adjust reward weights to emphasize recovery:
   --w_time 2.0 --bonus 100.0
4. Check if current limits are appropriate:
   --i_min -3.0 --i_max 3.0
5. Verify initial conditions are feasible
```

### Memory Issues

**Problem**: Out of memory during training
```
Solutions:
1. Reduce network size: --hidden 64 64
2. Reduce episodes per epoch: --eps_per_epoch 4
3. Shorten episodes: --max_time 5.0
```

## 📁 Project Structure

```
.
├── controller_reinforcement_agent_improved.py  # Main RL controller
├── platform_system_diagram.html               # Interactive visualization
├── README.md                                   # This file
├── requirements.txt                            # Python dependencies
├── vibration_data.csv                         # Initial conditions (create this)
├── policy.pth                                 # Saved model (after training)
└── evaluation_results.csv                     # Evaluation metrics (after eval)
```

## 🎓 For Thesis

### Recommended Experiments

1. **Baseline Comparison**: Compare RL controller with PID/LQR
2. **Ablation Study**: Test impact of each reward component
3. **Generalization**: Train on subset, test on different disturbances
4. **Robustness**: Add sensor noise, actuator saturation
5. **Multi-objective**: Pareto frontier of speed vs. energy

### Figures to Include

1. Platform system diagram (use provided HTML visualization)
2. Training curves (reward, recovery rate vs. epochs)
3. Evaluation histograms (recovery time distribution)
4. State trajectories (φ, θ vs. time for sample episodes)
5. Action profiles (I₁-I₄ vs. time)
6. Comparison with baselines

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{yourname2026vibration,
  title={Active Vibration Control Using Deep Reinforcement Learning},
  author={Your Name},
  year={2026},
  school={Your University}
}
```

## 📄 License

MIT License - Feel free to use and modify for research purposes.

## 🤝 Contributing

This is a thesis project, but suggestions and improvements are welcome!

## ✉️ Contact

For questions or collaboration: [your.email@university.edu]

---

**Good luck with your master's thesis! 🎓🚀**
