# 🚀 GPU Training Guide & Cheat Sheet

**Mission Gym PPO Training - Complete Reference**

---

## 📋 Table of Contents
1. [Quick Start Commands](#quick-start-commands)
2. [GPU Configuration Tiers](#gpu-configuration-tiers)
3. [Command-Line Parameters](#command-line-parameters)
4. [Performance Expectations](#performance-expectations)
5. [Reward System Documentation](#reward-system-documentation)
6. [Troubleshooting](#troubleshooting)

---

## 🎯 Quick Start Commands

### Resume Training with GPU (Balanced)
```bash
python -m mission_gym.scripts.train_ppo \
  --timesteps 50000000 \
  --n-envs 32 \
  --subproc \
  --network-arch "512,512,256" \
  --n-epochs 20 \
  --parent-checkpoint runs/YOUR-RUN/checkpoints/ppo_mission_XXXXX_steps \
  --branch-name your-branch-name \
  --notes "Description of your experiment"
```

### Start New Training from Scratch
```bash
python -m mission_gym.scripts.train_ppo \
  --timesteps 10000000 \
  --n-envs 16 \
  --subproc \
  --network-arch "256,256" \
  --n-epochs 10 \
  --run-name my-experiment
```

---

## ⚡ GPU Configuration Tiers

### 🐢 Conservative (Safe & Stable)
**Best for:** First runs, testing, limited hardware

```bash
python -m mission_gym.scripts.train_ppo \
  --timesteps 50000000 \
  --n-envs 16 \
  --subproc \
  --network-arch "256,256" \
  --n-epochs 10 \
  --parent-checkpoint runs/parent-run/checkpoints/ppo_mission_XXXXX_steps \
  --branch-name conservative-run
```

**Performance:**
- GPU Utilization: 30-50%
- VRAM Usage: 200-500 MB (out of 8 GB)
- FPS: ~350-450
- Time to 50M steps: ~31-40 hours

---

### 🏃 Balanced (Recommended)
**Best for:** Most training runs, good GPU utilization

```bash
python -m mission_gym.scripts.train_ppo \
  --timesteps 50000000 \
  --n-envs 32 \
  --subproc \
  --network-arch "512,512,256" \
  --n-epochs 20 \
  --parent-checkpoint runs/parent-run/checkpoints/ppo_mission_XXXXX_steps \
  --branch-name balanced-run
```

**Performance:**
- GPU Utilization: 60-80% ⭐
- VRAM Usage: 800-1500 MB
- FPS: ~400-550
- Time to 50M steps: ~25-35 hours
- **Better learning:** Larger network + more epochs

---

### 🚀 High Performance
**Best for:** Fast iteration, good GPU

```bash
python -m mission_gym.scripts.train_ppo \
  --timesteps 50000000 \
  --n-envs 64 \
  --subproc \
  --network-arch "512,512,256" \
  --n-epochs 20 \
  --parent-checkpoint runs/parent-run/checkpoints/ppo_mission_XXXXX_steps \
  --branch-name high-perf-run
```

**Performance:**
- GPU Utilization: 70-85%
- VRAM Usage: 1000-2000 MB
- FPS: ~500-700
- Time to 50M steps: ~20-28 hours

---

### 🔥 Maximum GPU (Extreme)
**Best for:** Maxing out GPU, research experiments

```bash
python -m mission_gym.scripts.train_ppo \
  --timesteps 50000000 \
  --n-envs 64 \
  --subproc \
  --network-arch "1024,512,512,256" \
  --n-epochs 30 \
  --parent-checkpoint runs/parent-run/checkpoints/ppo_mission_XXXXX_steps \
  --branch-name extreme-run
```

**Performance:**
- GPU Utilization: 80-95% 🔥
- VRAM Usage: 2000-4000 MB
- FPS: ~350-500 (slower due to massive network)
- Time to 50M steps: ~28-40 hours
- **Highest capacity:** Best for complex behaviors

---

## 🎛️ Command-Line Parameters

### Essential Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--timesteps` | int | 100000 | Total training steps |
| `--n-envs` | int | 4 | Number of parallel environments |
| `--subproc` | flag | false | Use true parallelism (recommended for n-envs > 4) |
| `--seed` | int | 42 | Random seed for reproducibility |

### GPU Optimization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--device` | str | "auto" | Device: "auto", "cuda", "cuda:0", or "cpu" |
| `--network-arch` | str | "256,256" | Network layers (e.g., "512,512,256") |
| `--n-epochs` | int | 10 | Policy update epochs (more = more GPU work) |

### Checkpoint & Branching

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--parent-checkpoint` | str | None | Path to parent checkpoint for branching |
| `--load-checkpoint` | str | None | Alias for --parent-checkpoint |
| `--branch-name` | str | None | Branch name (creates run-name from it) |
| `--run-name` | str | None | Custom run name prefix |
| `--notes` | str | None | Notes saved in lineage.json |

### Monitoring & Callbacks

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--eval-freq` | int | 5000 | Evaluation frequency in steps |
| `--no-tensorboard` | flag | false | Disable TensorBoard logging |

---

## 📊 Performance Expectations

### GPU Utilization vs Network Size

| Network Architecture | Parameters | GPU Util | VRAM | Best For |
|---------------------|------------|----------|------|----------|
| `256,256` | ~133K | 30-50% | 200-500 MB | Fast prototyping |
| `512,512,256` | ~800K | 60-80% | 800-1500 MB | **Recommended** |
| `1024,512,512,256` | ~2.8M | 80-95% | 2-4 GB | Complex behaviors |

### Training Speed Comparison

| Configuration | FPS | 50M Steps | GPU% | Quality |
|--------------|-----|-----------|------|---------|
| CPU (16 envs) | 180 | 77 hrs | 0% | Baseline |
| GPU Conservative | 400 | 35 hrs | 40% | Good |
| GPU Balanced ⭐ | 500 | 28 hrs | 70% | **Excellent** |
| GPU High-Perf | 650 | 21 hrs | 85% | Excellent |
| GPU Extreme | 450 | 31 hrs | 95% | **Best quality** |

### Hardware Requirements

**Minimum (Conservative):**
- GPU: 4GB VRAM (RTX 3050, GTX 1650)
- CPU: 8 cores
- RAM: 16 GB

**Recommended (Balanced/High-Perf):**
- GPU: 8GB VRAM (RTX 3070, RTX 4060)
- CPU: 12+ cores
- RAM: 32 GB

**Optimal (Extreme):**
- GPU: 12+ GB VRAM (RTX 4080, A4000)
- CPU: 16+ cores
- RAM: 64 GB

---

## 🎁 Reward System Documentation

### Reward Philosophy

The reward function is **tuned for learning** with these principles:
1. **Winning should be attractive:** Objective rewards dominate penalties
2. **Shaped rewards guide:** Dense feedback toward objective
3. **Small penalties:** Don't overwhelm early exploration
4. **Mission-aligned combat:** Only reward combat near objective

---

### Objective Rewards (Primary Goals)

| Component | Weight | Description | Max Value |
|-----------|--------|-------------|-----------|
| `capture_progress` | 2.0 | Per second of capture progress | ~24/episode |
| `win_bonus` | 200.0 | One-time bonus on victory | 200 |
| `zone_entry_bonus` | 20.0 | First entry into capture zone | 20 |
| `zone_time` | 2.0 | Per second inside zone | ~24/episode |

**Total winning episode reward:** ~250-300

---

### Shaping Rewards (Guidance)

| Component | Weight | Description | Max Value |
|-----------|--------|-------------|-----------|
| `min_dist_potential` | 0.5 | Per meter closer to objective | ~50/episode |
| `ring_bonus` | 5.0 | Per milestone ring crossed | 25 (5 rings) |
| `spread_formation` | 0.005 | Staying in formation | ~6/episode |

**Ring distances:** 80m, 60m, 40m, 25m, 15m from objective center

**Purpose:** Guide agent toward objective with dense feedback

---

### Combat Rewards (Mission-Aligned Only)

| Component | Weight | Description | Notes |
|-----------|--------|-------------|-------|
| `tag_hit_bonus` | 0.2 | Per successful tag hit | Only within 60m of objective |
| `defender_disabled_bonus` | 10.0 | Per defender disabled | Only within 60m of objective |

**Key insight:** Combat is rewarded only when near the objective to prevent "hunting" behavior away from mission goal.

---

### Penalties (Kept Small)

| Component | Weight | Description | Max Penalty |
|-----------|--------|-------------|-------------|
| `time_penalty` | -0.001 | Per timestep | ~-1.2/episode |
| `collision_penalty` | -0.5 | Per collision | ~-5/episode |
| `integrity_loss_penalty` | -0.1 | Per integrity point lost | ~-10/episode |
| `unit_disabled_penalty` | -20.0 | When unit dies | -20 per unit |
| `detected_time_penalty` | -0.05 | Per step while detected | ~-6/episode |

---

### Outcome Penalties (Terminal)

| Outcome | Penalty | When |
|---------|---------|------|
| `stalled` | -50.0 | No progress for too long |
| `timeout` | -20.0 | Episode time limit reached |
| `all_disabled` | -100.0 | All attackers destroyed |

---

### Expected Total Rewards

| Episode Outcome | Reward Range | Description |
|-----------------|--------------|-------------|
| **Victory** 🎉 | +100 to +250 | Successful capture |
| **Close attempt** 😊 | 0 to +30 | Got very close, partial progress |
| **Timeout** 😐 | -20 to +10 | Time expired, some progress |
| **Stalled** 😞 | -50 to -70 | No progress made |
| **All dead** 💀 | -100 to -150 | All units destroyed |

---

### Reward Tuning History

**Key changes that improved learning:**

1. **Increased objective rewards** (200 → 200, but added zone_time)
   - Made winning more attractive than avoiding penalties

2. **Added ring milestones** (5.0 each)
   - Provides dense intermediate rewards for progress

3. **Reduced time penalty** (-0.01 → -0.001)
   - Was dominating early learning, discouraging exploration

4. **Mission-aligned combat** (60m restriction)
   - Prevents agent from "hunting" defenders away from objective

5. **Potential-based shaping** (0.5 per meter)
   - Strong guidance toward objective without terminal dependence

---

## 🛠️ Troubleshooting

### GPU Not Detected

**Symptoms:** Training runs on CPU despite having GPU

**Solutions:**
```bash
# Check GPU visibility
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If false, may need driver reload or reboot
sudo systemctl restart nvidia-persistenced

# Or reboot (most reliable)
sudo reboot
```

---

### Low GPU Utilization (<50%)

**Cause:** CPU environments bottleneck GPU

**Solutions:**
1. Increase network size: `--network-arch "512,512,256"`
2. Increase epochs: `--n-epochs 20`
3. Reduce n-envs if CPU-bound: `--n-envs 32`

---

### Out of Memory (OOM)

**Symptoms:** CUDA out of memory error

**Solutions:**
1. Reduce network size: `--network-arch "256,256"`
2. Reduce n-envs: `--n-envs 32` or `--n-envs 16`
3. Reduce n-epochs: `--n-epochs 10`

---

### TensorBoard Protobuf Errors

**Symptoms:** Spam of AttributeError during startup

**Solution:**
```bash
pip install --upgrade tensorboard protobuf
# Or pin version:
pip install "protobuf>=3.19.0,<4.0.0" --force-reinstall
```

**Note:** These errors are harmless and don't affect training!

---

### Slow Training Despite GPU

**Check:**
```bash
# Monitor GPU usage in real-time
watch -n 2 nvidia-smi

# Should show:
# - GPU Utilization: 60-90%
# - Memory Usage: 500MB - 4GB
# - Power: 20-50W
```

If GPU util is low:
- Increase `--network-arch` size
- Increase `--n-epochs`
- Check CPU isn't bottlenecked (`htop`)

---

## 📈 Monitoring Your Training

### Real-Time Monitoring

```bash
# GPU usage (refresh every 2s)
watch -n 2 nvidia-smi

# CPU usage
htop

# Training dashboard (auto-refresh every 5s)
# Open in browser: runs/YOUR-RUN/dashboard.html

# TensorBoard
tensorboard --logdir runs/YOUR-RUN/logs
# Then open: http://localhost:6006
```

---

### Key Metrics to Watch

**During Training:**
- **FPS:** Should be 400-700 for GPU training
- **GPU Utilization:** Target 60-90%
- **Mean Reward:** Should trend upward over time
- **Episode Length:** May increase as agent improves
- **Win Rate:** Track in dashboard

**In Dashboard:**
- Reward components breakdown
- Episode outcomes (win/timeout/stalled)
- Policy performance over time
- Lineage tree (if branching)

---

## 🎓 Best Practices

### 1. Start with Balanced Configuration
Use the **Balanced** tier (`512,512,256`, 20 epochs, 32 envs) as your default. It offers excellent GPU utilization without overkill.

### 2. Use Policy Branching
Always use `--parent-checkpoint` and `--branch-name` to track experiments:
```bash
--parent-checkpoint runs/parent/checkpoints/ppo_mission_XXXXX_steps
--branch-name exploring-stealth-behavior
--notes "Testing increased stealth rewards"
```

### 3. Monitor Early (<10M steps)
Check dashboard after first 5-10M steps. If reward isn't improving, stop and adjust hyperparameters.

### 4. Save Intermediate Checkpoints
Checkpoints save every 100K steps by default. Keep them! You can branch from any checkpoint.

### 5. Experiment with Network Size
Larger networks can learn more complex behaviors but train slower. Start small, scale up if needed.

### 6. Use Subproc for Parallelism
Always use `--subproc` with more than 4 environments for true CPU parallelism.

### 7. Reproducibility
Set `--seed` explicitly for reproducible experiments:
```bash
--seed 42  # Or any fixed number
```

---

## 📝 Example Workflow

### 1. Initial Training (From Scratch)
```bash
python -m mission_gym.scripts.train_ppo \
  --timesteps 10000000 \
  --n-envs 16 \
  --subproc \
  --run-name baseline-v1 \
  --seed 42
```

### 2. Continue Training (Scale Up)
```bash
python -m mission_gym.scripts.train_ppo \
  --timesteps 30000000 \
  --n-envs 32 \
  --subproc \
  --network-arch "512,512,256" \
  --n-epochs 20 \
  --parent-checkpoint runs/baseline-v1-TIMESTAMP/checkpoints/ppo_mission_10000000_steps \
  --branch-name baseline-v1-continued \
  --seed 42
```

### 3. Experiment (Branch & Modify)
```bash
python -m mission_gym.scripts.train_ppo \
  --timesteps 20000000 \
  --n-envs 32 \
  --subproc \
  --network-arch "512,512,256" \
  --n-epochs 20 \
  --parent-checkpoint runs/baseline-v1-continued-TIMESTAMP/checkpoints/ppo_mission_40000000_steps \
  --branch-name test-stealth-focus \
  --notes "Increased detection penalty to encourage stealth" \
  --seed 42
```

### 4. Final Polish (Large Network)
```bash
python -m mission_gym.scripts.train_ppo \
  --timesteps 50000000 \
  --n-envs 64 \
  --subproc \
  --network-arch "1024,512,512,256" \
  --n-epochs 30 \
  --parent-checkpoint runs/test-stealth-focus-TIMESTAMP/checkpoints/ppo_mission_XXXXX_steps \
  --branch-name final-polish \
  --notes "Final training with large network for competition"
```

---

## 🔗 Related Documentation

- **Branching Guide:** `docs/BRANCHING_GUIDE.md`
- **Workflow Guide:** `docs/WORKFLOW_GUIDE.md`
- **Observations & Actions:** `docs/OBSERVATIONS_AND_ACTIONS.md`
- **API Continuation:** `docs/API_CONTINUATION_RULES.md`

---

## 💡 Quick Reference Card

```
┌────────────────────────────────────────────────────────────────┐
│ QUICK COMMANDS                                                 │
├────────────────────────────────────────────────────────────────┤
│ Check GPU:        nvidia-smi                                   │
│ Monitor GPU:      watch -n 2 nvidia-smi                        │
│ Check CUDA:       python -c "import torch; print(..."          │
│ View Dashboard:   firefox runs/YOUR-RUN/dashboard.html        │
│ TensorBoard:      tensorboard --logdir runs/YOUR-RUN/logs     │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ BALANCED CONFIG (RECOMMENDED)                                  │
├────────────────────────────────────────────────────────────────┤
│ --n-envs 32                                                    │
│ --subproc                                                      │
│ --network-arch "512,512,256"                                   │
│ --n-epochs 20                                                  │
│                                                                │
│ Expected: 60-80% GPU, ~500 FPS, ~28 hours for 50M steps       │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ REWARD RANGES                                                  │
├────────────────────────────────────────────────────────────────┤
│ Victory:      +100 to +250                                     │
│ Close call:   0 to +30                                         │
│ Timeout:      -20 to +10                                       │
│ Failure:      -50 to -150                                      │
└────────────────────────────────────────────────────────────────┘
```

---

**Last Updated:** January 25, 2026  
**Version:** 1.0  
**GPU Tested On:** RTX 4070 Laptop (8GB VRAM)
