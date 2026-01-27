# 🚀 Mission Gym Training Cheat Sheet

## Quick Commands

### GPU Check
```bash
nvidia-smi                          # Check GPU status
watch -n 2 nvidia-smi              # Monitor GPU real-time
python -c "import torch; print(torch.cuda.is_available())"  # Check PyTorch CUDA
```

### Training - Quick Start (USE PRESETS!)
```bash
# Fast preset (testing, ~30% GPU)
python -m mission_gym.scripts.train_ppo --preset fast --timesteps 10000000 --run-name my-run

# Heavy preset (recommended, ~50% GPU) ⭐
python -m mission_gym.scripts.train_ppo --preset heavy --timesteps 50000000 --run-name my-run

# Beast preset (maximum, ~80% GPU, needs 16 CPU cores)
python -m mission_gym.scripts.train_ppo --preset beast --timesteps 50000000 --run-name my-run

# Parallel (4 jobs for GPU saturation → 70-90%)
for i in {1..4}; do
  python -m mission_gym.scripts.train_ppo --preset heavy --timesteps 50M --run-name "job-$i" &
done
```

### Training - With Branching
```bash
# Use preset for easy GPU optimization
python -m mission_gym.scripts.train_ppo --preset heavy --timesteps 50000000 \
  --parent-checkpoint runs/PARENT-RUN/checkpoints/ppo_mission_XXXXX_steps \
  --branch-name my-branch --notes "Experiment description"
```

### Monitoring
```bash
# Open dashboard (auto-refreshes every 5s)
firefox runs/YOUR-RUN/dashboard.html

# Monitor GPU properly (high frequency!)
nvidia-smi dmon -s u              # Device monitoring (best)
watch -n 0.2 nvidia-smi           # Fast refresh

# TensorBoard
tensorboard --logdir runs/YOUR-RUN/logs
```

---

## Performance Tiers (Single Job) - WITH GPU OPTIMIZATIONS

| Preset | n-envs | n-steps | Batch | Batch Size | Epochs | Network | GPU %* | FPS | Time (50M) |
|--------|--------|---------|-------|------------|--------|---------|--------|-----|------------|
| **fast** | 16 | 256 | 4K | 2048 | 15 | 512,512 | ~30% | 800 | 17h |
| **heavy** ⭐ | **32** | **256** | **8K** | **4096** | **20** | **1024,512,256** | **~50%** | **700** | **20h** |
| **beast** 🔥 | 64 | 512 | 32K | 8192 | 30 | 1024,512,512,256 | ~80% | 600 | 23h |
| **Parallel×4** 💪 | heavy×4 | - | - | - | - | - | **70-90%** | ~700 ea | ~20h |

**GPU optimizations enabled:** TF32 matmul, CPU thread limiting, large batches, SubprocVecEnv forkserver

---

## Command-Line Parameters

### Essential
- `--timesteps N` - Total training steps
- `--n-envs N` - Parallel environments (16/32/64)
- `--subproc` - True parallelism (always use!)
- `--run-name NAME` - Custom run name

### GPU Optimization (NEW!)
- `--preset heavy` - **Use this!** Auto-configures for ~50% GPU utilization
- `--preset fast` - Testing (16 envs, ~30% GPU)
- `--preset beast` - Maximum (64 envs, ~80% GPU, needs 16 CPU cores)
- `--batch-size 4096` - Manual batch size override (must divide n_envs × n_steps)
- **Tip:** Run 4 parallel `heavy` jobs for 70-90% GPU saturation!

### Branching
- `--parent-checkpoint PATH` - Load parent checkpoint
- `--branch-name NAME` - Branch name for lineage
- `--notes "..."` - Experiment notes

### Other
- `--seed 42` - Random seed (reproducibility)
- `--eval-freq 5000` - Evaluation frequency
- `--no-tensorboard` - Disable TensorBoard

---

## Reward System (Quick Reference)

### Win Episode: +100 to +250
- Capture progress: +2.0/sec
- Win bonus: +200
- Zone entry: +20
- Zone time: +2.0/sec

### Shaping Rewards
- Distance potential: +0.5/meter closer
- Ring bonuses: +5.0 each (5 rings)
- Formation: +0.005/sec

### Combat (Mission-Aligned)
- Tag hit: +0.2 (only <60m from objective)
- Defender disabled: +10.0 (only <60m)

### Penalties
- Time: -0.001/step (~-1.2/episode)
- Collision: -0.5 each
- Integrity loss: -0.1/point
- Unit disabled: -20.0
- Detected: -0.05/step

### Terminal Outcomes
- Stalled: -50
- Timeout: -20
- All disabled: -100

---

## Troubleshooting

**GPU not detected?**
```bash
sudo systemctl restart nvidia-persistenced
# Or: sudo reboot
```

**Low GPU utilization?**
- **Solution:** Use `--preset heavy` or `--preset beast` (optimized for GPU!)
- Or: Run 4 parallel jobs: `for i in {1..4}; do python -m ... --preset heavy & done`
- **Note:** Monitor with `nvidia-smi dmon -s u` (not 1-second snapshots!)

**Out of memory?**
- Reduce `--n-envs 16`
- Reduce `--network-arch "256,256"`

**TensorBoard errors?**
```bash
pip install --upgrade tensorboard protobuf
```

---

## Best Practices

1. ✅ **Use presets!** Start with `--preset heavy` (optimal for most cases)
2. ✅ Run **4 parallel jobs** for maximum GPU utilization (70-90%)
3. ✅ Monitor with `nvidia-smi dmon -s u` (not 1-second snapshots!)
4. ✅ Always use `--parent-checkpoint` + `--branch-name` for experiments
5. ✅ Check dashboard after first 5-10M steps
6. ✅ Set `--seed` for reproducibility
7. ✅ Beast preset needs ~16 CPU cores (use `heavy` if you have fewer)

---

## File Locations

```
runs/
  ├── dashboard.html                          # All runs overview
  └── YOUR-RUN-TIMESTAMP/
      ├── dashboard.html                      # This run (auto-refresh)
      ├── lineage.json                        # Parent/compatibility info
      ├── checkpoints/
      │   └── ppo_mission_XXXXX_steps.zip    # Model checkpoints
      ├── configs/                            # Config snapshot
      ├── logs/                               # TensorBoard logs
      └── summary.txt                         # Run summary
```

---

## Example Workflow (WITH PRESETS!)

```bash
# 1. Initial training (fast preset for testing)
python -m mission_gym.scripts.train_ppo --preset fast --timesteps 10000000 --run-name baseline --seed 42

# 2. Continue with heavy preset (GPU-optimized)
python -m mission_gym.scripts.train_ppo --preset heavy --timesteps 50000000 \
  --parent-checkpoint runs/baseline-TIMESTAMP/checkpoints/ppo_mission_10000000_steps \
  --branch-name baseline-heavy --seed 42

# 3. Branch for experiment
python -m mission_gym.scripts.train_ppo --preset heavy --timesteps 20000000 \
  --parent-checkpoint runs/baseline-heavy-TIMESTAMP/checkpoints/ppo_mission_XXXXX_steps \
  --branch-name test-new-reward --notes "Testing reward changes" --seed 42

# 4. Run 4 parallel heavy jobs for GPU saturation (70-90%)
for i in {1..4}; do
  python -m mission_gym.scripts.train_ppo --preset heavy --timesteps 50M --run-name "parallel-$i" --seed $((42+i)) &
done
```

---

## GPU Targets

| GPU | VRAM | Recommended Config |
|-----|------|--------------------|
| RTX 3050 | 4GB | Conservative (16 envs, 256×256) |
| RTX 3060 | 6GB | Balanced (32 envs, 512×512×256) |
| RTX 3070 | 8GB | High-Perf (64 envs, 512×512×256) |
| RTX 4070 | 8GB | High-Perf or Maximum |
| RTX 4080+ | 12GB+ | Maximum (64 envs, 1024×512×512×256) |

---

**Full Documentation:** `docs/GPU_TRAINING_GUIDE.md`  
**Last Updated:** January 25, 2026
