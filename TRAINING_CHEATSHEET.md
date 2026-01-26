# 🚀 Mission Gym Training Cheat Sheet

## Quick Commands

### GPU Check
```bash
nvidia-smi                          # Check GPU status
watch -n 2 nvidia-smi              # Monitor GPU real-time
python -c "import torch; print(torch.cuda.is_available())"  # Check PyTorch CUDA
```

### Training - Quick Start
```bash
# Conservative (safe, 16 envs, small batch)
python -m mission_gym.scripts.train_ppo --timesteps 10000000 --n-envs 16 --n-steps-per-env 128 --subproc --run-name my-run

# Balanced (recommended, 32 envs, 8192 batch)
python -m mission_gym.scripts.train_ppo --timesteps 50000000 --n-envs 32 --n-steps-per-env 256 --subproc \
  --network-arch "512,512,256" --n-epochs 20 --run-name my-run

# Maximum (64 envs, 16384 batch, large network)
python -m mission_gym.scripts.train_ppo --timesteps 50000000 --n-envs 64 --n-steps-per-env 256 --subproc \
  --network-arch "1024,512,512,256" --n-epochs 30 --run-name my-run

# Parallel (4 jobs for GPU saturation)
./add_parallel_jobs.sh  # Or: ./parallel_train.sh for fresh start
```

### Training - With Branching
```bash
python -m mission_gym.scripts.train_ppo --timesteps 50000000 --n-envs 32 --subproc \
  --network-arch "512,512,256" --n-epochs 20 \
  --parent-checkpoint runs/PARENT-RUN/checkpoints/ppo_mission_XXXXX_steps \
  --branch-name my-branch --notes "Experiment description"
```

### Monitoring
```bash
# Open dashboard (auto-refreshes every 5s)
firefox runs/YOUR-RUN/dashboard.html

# TensorBoard
tensorboard --logdir runs/YOUR-RUN/logs

# Watch training live
tail -f runs/YOUR-RUN/training.log
```

---

## Performance Tiers (Single Job)

| Config | n-envs | n-steps-per-env | Buffer | network-arch | n-epochs | GPU %* | FPS | Time (50M) |
|--------|--------|-----------------|--------|--------------|----------|--------|-----|------------|
| Conservative | 16 | 128 | 2K | 256,256 | 10 | 5-10% | 450 | 31h |
| **Balanced** ⭐ | **32** | **256** | **8K** | **512,512,256** | **20** | **10-20%** | **600** | **23h** |
| High-Perf | 48 | 256 | 12K | 1024,512,256 | 25 | 15-25% | 700 | 20h |
| Maximum | 64 | 256 | 16K | 1024,512,512,256 | 30 | 20-30% | 750 | 18h |
| **Parallel×4** 🔥 | 24×4 | 256 | 6K ea | mixed | 20-30 | **50-70%** | ~600 ea | ~23h |

**Note:** With MLP policies, GPU is idle during CPU-bound rollout phase. Use parallel jobs for higher GPU utilization!

---

## Command-Line Parameters

### Essential
- `--timesteps N` - Total training steps
- `--n-envs N` - Parallel environments (16/32/64)
- `--subproc` - True parallelism (always use!)
- `--run-name NAME` - Custom run name

### GPU Optimization
- `--n-steps-per-env 256` - Steps per env per rollout (buffer = n_steps × n_envs)
- `--network-arch "512,512,256"` - Network layers (larger = more GPU compute)
- `--n-epochs 20` - Epochs per update (more = more GPU work per rollout)
- **Tip:** Run 3-4 parallel jobs to saturate GPU (single MLP job uses only ~10-25%)

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

**Low GPU utilization? (10-25% for single MLP job is NORMAL)**
- This is expected: GPU is idle during CPU-bound rollout phase
- **Solution:** Run 3-4 parallel jobs to saturate GPU (use `add_parallel_jobs.sh`)
- Or: Increase `--n-steps-per-env 512` + `--network-arch "1024,512,512,256"` + `--n-epochs 30`

**Out of memory?**
- Reduce `--n-envs 16`
- Reduce `--network-arch "256,256"`

**TensorBoard errors?**
```bash
pip install --upgrade tensorboard protobuf
```

---

## Best Practices

1. ✅ Start with **Balanced** config (32 envs × 256 steps = 8192 buffer)
2. ✅ Use `--subproc` for true parallelism
3. ✅ Run **3-4 parallel jobs** for maximum GPU utilization (50-70% vs 10-25%)
4. ✅ Always use `--parent-checkpoint` + `--branch-name` for experiments
5. ✅ Monitor GPU with `watch -n 2 nvidia-smi`
6. ✅ Check dashboard after first 5-10M steps
7. ✅ Keep checkpoints (auto-save every 10K steps)
8. ✅ Set `--seed` for reproducibility

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

## Example Workflow

```bash
# 1. Initial training
python -m mission_gym.scripts.train_ppo --timesteps 10000000 --n-envs 16 --n-steps-per-env 128 \
  --subproc --run-name baseline --seed 42

# 2. Continue with better config
python -m mission_gym.scripts.train_ppo --timesteps 30000000 --n-envs 32 --n-steps-per-env 256 \
  --subproc --network-arch "512,512,256" --n-epochs 20 \
  --parent-checkpoint runs/baseline-TIMESTAMP/checkpoints/ppo_mission_10000000_steps \
  --branch-name baseline-continued --seed 42

# 3. Branch for experiment
python -m mission_gym.scripts.train_ppo --timesteps 20000000 --n-envs 32 --n-steps-per-env 256 \
  --subproc --network-arch "512,512,256" --n-epochs 20 \
  --parent-checkpoint runs/baseline-continued-TIMESTAMP/checkpoints/ppo_mission_XXXXX_steps \
  --branch-name test-new-reward --notes "Testing increased stealth rewards" --seed 42

# 4. Run parallel experiments for GPU saturation
./add_parallel_jobs.sh  # Adds 3 more jobs to existing training (4 total)
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
