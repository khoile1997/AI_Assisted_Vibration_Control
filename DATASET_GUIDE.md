# 📊 Training Dataset Guide for Vibration Control

## Overview

Three comprehensive datasets have been created with **2000 initial conditions each** for progressive training of your reinforcement learning vibration control system.

---

## 📗 Dataset 1: EASY (vibration_data_easy.csv)

### **Specifications**
- **Samples:** 2000
- **Angles:** ±0.05 rad (±2.9°)
- **Rates:** ±0.08 rad/s
- **Difficulty:** ⭐ BEGINNER

### **Purpose**
Perfect for **initial learning**. The agent should master these conditions first before moving to harder scenarios.

### **Expected Training Results**
```
Training Time: 4-6 hours (100 epochs)
First Success: Epoch 20-30
Final Performance: 90-95% recovery rate
Mean Recovery Time: 0.5-0.8 seconds
```

### **When to Use**
- ✅ Starting fresh training from scratch
- ✅ Agent has never succeeded before
- ✅ Learning the basic control strategy

### **Command**
```bash
python controller_reinforcement_agent_improved.py --mode train \
    --csv vibration_data_easy.csv \
    --epochs 100 \
    --eps_per_epoch 4 \
    --max_time 5.0 \
    --dt 0.1 \
    --exploration_noise 0.1 \
    --model_path policy_easy.pth
```

---

## 📘 Dataset 2: MEDIUM (vibration_data_medium.csv)

### **Specifications**
- **Samples:** 2000
- **Angles:** ±0.08 rad (±4.6°)
- **Rates:** ±0.15 rad/s
- **Difficulty:** ⭐⭐ INTERMEDIATE

### **Purpose**
For **transfer learning** after mastering easy conditions. Tests generalization and robustness.

### **Expected Training Results**
```
Training Time: 5-8 hours (100 epochs)
First Success: Epoch 15-25 (with pretrained model)
Final Performance: 80-90% recovery rate
Mean Recovery Time: 0.6-1.0 seconds
```

### **When to Use**
- ✅ After achieving 85%+ success on EASY dataset
- ✅ Want to improve robustness
- ✅ Preparing for real-world deployment

### **Command**
```bash
# Continue training from easy model
python controller_reinforcement_agent_improved.py --mode train \
    --load_model \
    --model_path policy_easy.pth.best \
    --csv vibration_data_medium.csv \
    --epochs 100 \
    --eps_per_epoch 4 \
    --max_time 5.0 \
    --dt 0.1 \
    --exploration_noise 0.08 \
    --model_path policy_medium.pth
```

---

## 📕 Dataset 3: HARD (vibration_data_hard.csv)

### **Specifications**
- **Samples:** 2000
- **Angles:** ±0.15 rad (±8.6°)
- **Rates:** ±0.30 rad/s
- **Difficulty:** ⭐⭐⭐ ADVANCED
- **Mix:** 70% moderate + 30% extreme cases

### **Purpose**
**Final challenge** and **evaluation benchmark**. Tests maximum capability and worst-case handling.

### **Expected Training Results**
```
Training Time: 8-12 hours (150 epochs)
First Success: Epoch 20-40 (with pretrained model)
Final Performance: 70-85% recovery rate
Mean Recovery Time: 0.8-1.5 seconds
```

### **When to Use**
- ✅ After achieving 80%+ success on MEDIUM dataset
- ✅ Final model for thesis evaluation
- ✅ Demonstrating robustness to extreme disturbances
- ✅ Comparing against baseline controllers (PID, LQR)

### **Command**
```bash
# Continue training from medium model
python controller_reinforcement_agent_improved.py --mode train \
    --load_model \
    --model_path policy_medium.pth.best \
    --csv vibration_data_hard.csv \
    --epochs 150 \
    --eps_per_epoch 8 \
    --max_time 5.0 \
    --dt 0.05 \
    --exploration_noise 0.05 \
    --lr 5e-4 \
    --model_path policy_hard.pth
```

---

## 🎓 Progressive Training Strategy (Recommended)

### **Phase 1: Master Easy Conditions (Day 1-2)**

```bash
# Train on easy data
python controller_reinforcement_agent_improved.py --mode train \
    --csv vibration_data_easy.csv \
    --epochs 100 \
    --eps_per_epoch 4 \
    --model_path policy_phase1.pth

# Evaluate
python controller_reinforcement_agent_improved.py --mode eval \
    --model_path policy_phase1.pth.best \
    --csv vibration_data_easy.csv \
    --n_eval 100

# Target: >90% success rate before moving on
```

### **Phase 2: Transfer to Medium (Day 3-4)**

```bash
# Continue from Phase 1
python controller_reinforcement_agent_improved.py --mode train \
    --load_model \
    --model_path policy_phase1.pth.best \
    --csv vibration_data_medium.csv \
    --epochs 100 \
    --eps_per_epoch 4 \
    --model_path policy_phase2.pth

# Evaluate
python controller_reinforcement_agent_improved.py --mode eval \
    --model_path policy_phase2.pth.best \
    --csv vibration_data_medium.csv \
    --n_eval 100

# Target: >85% success rate before moving on
```

### **Phase 3: Final Challenge (Day 5-7)**

```bash
# Continue from Phase 2
python controller_reinforcement_agent_improved.py --mode train \
    --load_model \
    --model_path policy_phase2.pth.best \
    --csv vibration_data_hard.csv \
    --epochs 150 \
    --eps_per_epoch 8 \
    --max_time 5.0 \
    --dt 0.05 \
    --exploration_noise 0.05 \
    --lr 5e-4 \
    --model_path policy_final.pth

# Final evaluation
python controller_reinforcement_agent_improved.py --mode eval \
    --model_path policy_final.pth.best \
    --csv vibration_data_hard.csv \
    --n_eval 200

# Target: >75% success rate for excellent thesis results
```

---

## 📊 Cross-Dataset Evaluation

Test **generalization** by evaluating each model on all datasets:

```bash
# Evaluate EASY model on all datasets
python controller_reinforcement_agent_improved.py --mode eval \
    --model_path policy_phase1.pth.best \
    --csv vibration_data_easy.csv \
    --n_eval 100
    
python controller_reinforcement_agent_improved.py --mode eval \
    --model_path policy_phase1.pth.best \
    --csv vibration_data_medium.csv \
    --n_eval 100
    
python controller_reinforcement_agent_improved.py --mode eval \
    --model_path policy_phase1.pth.best \
    --csv vibration_data_hard.csv \
    --n_eval 100

# Repeat for MEDIUM and FINAL models
```

### **Expected Generalization Results**

| Model Trained On | Easy Dataset | Medium Dataset | Hard Dataset |
|------------------|--------------|----------------|--------------|
| **Easy only** | 95% | 60% | 30% |
| **Medium** | 98% | 85% | 50% |
| **Hard (final)** | 99% | 95% | 75% |

---

## 🎯 For Your Thesis

### **Recommended Experiments**

1. **Learning Curve Analysis**
   - Plot recovery rate vs. epochs for each dataset
   - Show progressive learning across difficulty levels

2. **Generalization Study**
   - Train on EASY, test on MEDIUM/HARD
   - Compare transfer learning vs. training from scratch

3. **Baseline Comparison**
   - Compare final RL model against PID/LQR on HARD dataset
   - Show RL handles extreme cases better

4. **Robustness Analysis**
   - Plot success rate vs. initial disturbance magnitude
   - Show performance degrades gracefully with difficulty

5. **Statistical Significance**
   - Run multiple training seeds (5-10 runs)
   - Report mean ± std for all metrics
   - Use t-tests to compare methods

### **Key Figures for Thesis**

1. **Figure 1:** Learning curves (all 3 datasets)
2. **Figure 2:** Recovery time histograms
3. **Figure 3:** Generalization heatmap
4. **Figure 4:** Sample trajectories (successful vs. failed)
5. **Figure 5:** Comparison with baseline controllers
6. **Table 1:** Performance summary across datasets
7. **Table 2:** Statistical significance tests

---

## 📈 Performance Benchmarks

### **Excellent Results (A+ Thesis)**
- Easy: >95% recovery, <0.6s mean time
- Medium: >85% recovery, <0.9s mean time
- Hard: >75% recovery, <1.3s mean time
- Beats PID/LQR on hard cases

### **Good Results (A Thesis)**
- Easy: >90% recovery, <0.8s mean time
- Medium: >75% recovery, <1.2s mean time
- Hard: >60% recovery, <1.8s mean time
- Competitive with baselines

### **Acceptable Results (B Thesis)**
- Easy: >85% recovery, <1.0s mean time
- Medium: >65% recovery, <1.5s mean time
- Hard: >50% recovery, <2.0s mean time
- Shows learning occurred

---

## 🛠️ Troubleshooting

### **Problem: Low success rate on EASY (<70% after 100 epochs)**

**Solutions:**
- Verify simulator is responding (test with diagnostic script)
- Check loss is positive (not negative)
- Increase training time: `--epochs 150`
- Reduce exploration: `--exploration_noise 0.05`
- Check file is actually vibration_data_easy.csv (not the hard one!)

### **Problem: Success on EASY but fails on MEDIUM**

**Solutions:**
- Train longer on EASY first (150-200 epochs)
- Use smaller learning rate for transfer: `--lr 5e-4`
- Gradually increase difficulty (create custom intermediate dataset)

### **Problem: Training is very slow**

**Solutions:**
- Use fewer episodes: `--eps_per_epoch 4` (not 8)
- Larger timestep: `--dt 0.1` (not 0.05)
- Shorter max time: `--max_time 3.0` for testing

---

## 📁 File Locations

All datasets are saved in the same directory as your controller:

```
your_project/
├── controller_reinforcement_agent_improved.py
├── real_time_platform_sim.py
├── vibration_data_easy.csv     ← 2000 samples, ⭐
├── vibration_data_medium.csv   ← 2000 samples, ⭐⭐
└── vibration_data_hard.csv     ← 2000 samples, ⭐⭐⭐
```

---

## 🎯 Quick Start

```bash
# Day 1: Start with EASY
python controller_reinforcement_agent_improved.py --mode train \
    --csv vibration_data_easy.csv \
    --epochs 100 \
    --eps_per_epoch 4 \
    --max_time 5.0 \
    --dt 0.1 \
    --model_path policy.pth

# Check if learning: should see first success by epoch 25-30
# Continue to MEDIUM only after >85% success on EASY
```

---

## ✅ Success Criteria Checklist

### Before Moving to Next Dataset:

**EASY → MEDIUM:**
- [ ] >85% recovery rate on easy
- [ ] Mean time <0.8s
- [ ] Model saved as policy_easy.pth.best

**MEDIUM → HARD:**
- [ ] >80% recovery rate on medium
- [ ] Mean time <1.0s
- [ ] Model saved as policy_medium.pth.best

**Ready for Thesis:**
- [ ] >75% recovery rate on hard
- [ ] Comprehensive evaluation on all 3 datasets
- [ ] Comparison with baseline controllers
- [ ] Statistical analysis complete

---

## 💡 Pro Tips

1. **Always evaluate before moving on** - don't rush to harder datasets
2. **Save your best models** - they take hours to train!
3. **Use transfer learning** - always use `--load_model` when increasing difficulty
4. **Monitor loss** - should always be positive
5. **Be patient** - learning takes time, especially on harder datasets
6. **Run overnight** - 100 epochs takes 3-6 hours
7. **Document everything** - save training logs for thesis analysis

---

**Good luck with your training! 🚀**
