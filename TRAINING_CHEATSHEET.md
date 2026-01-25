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
# Conservative (safe, 16 envs)
python -m mission_gym.scripts.train_ppo --timesteps 10000000 --n-envs 16 --subproc --run-name my-run

# Balanced (recommended, 32 envs, larger network)
python -m mission_gym.scripts.train_ppo --timesteps 50000000 --n-envs 32 --subproc \
  --network-arch "512,512,256" --n-epochs 20 --run-name my-run

# Maximum GPU (64 envs, large network)
python -m mission_gym.scripts.train_ppo --timesteps 50000000 --n-envs 64 --subproc \
  --network-arch "1024,512,512,256" --n-epochs 30 --run-name my-run
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

## Performance Tiers

| Config | n-envs | network-arch | n-epochs | GPU % | FPS | Time (50M) |
|--------|--------|--------------|----------|-------|-----|------------|
| Conservative | 16 | 256,256 | 10 | 40% | 400 | 35h |
| **Balanced** ⭐ | **32** | **512,512,256** | **20** | **70%** | **500** | **28h** |
| High-Perf | 64 | 512,512,256 | 20 | 85% | 650 | 21h |
| Maximum | 64 | 1024,512,512,256 | 30 | 95% | 450 | 31h |

---

## Command-Line Parameters

### Essential
- `--timesteps N` - Total training steps
- `--n-envs N` - Parallel environments (16/32/64)
- `--subproc` - True parallelism (always use!)
- `--run-name NAME` - Custom run name

### GPU Optimization
- `--network-arch "512,512,256"` - Network layers
- `--n-epochs 20` - Epochs per update (more = more GPU work)
- `--device auto|cuda|cpu` - Device selection

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
- Increase `--network-arch "512,512,256"`
- Increase `--n-epochs 20`

**Out of memory?**
- Reduce `--n-envs 16`
- Reduce `--network-arch "256,256"`

**TensorBoard errors?**
```bash
pip install --upgrade tensorboard protobuf
```

---

## Best Practices

1. ✅ Start with **Balanced** config (32 envs, 512×512×256)
2. ✅ Use `--subproc` for true parallelism
3. ✅ Always use `--parent-checkpoint` + `--branch-name` for experiments
4. ✅ Monitor GPU with `watch -n 2 nvidia-smi`
5. ✅ Check dashboard after first 5-10M steps
6. ✅ Keep checkpoints (save every 100K steps)
7. ✅ Set `--seed` for reproducibility

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
python -m mission_gym.scripts.train_ppo --timesteps 10000000 --n-envs 16 --subproc --run-name baseline --seed 42

# 2. Continue with better config
python -m mission_gym.scripts.train_ppo --timesteps 30000000 --n-envs 32 --subproc \
  --network-arch "512,512,256" --n-epochs 20 \
  --parent-checkpoint runs/baseline-TIMESTAMP/checkpoints/ppo_mission_10000000_steps \
  --branch-name baseline-continued --seed 42

# 3. Branch for experiment
python -m mission_gym.scripts.train_ppo --timesteps 20000000 --n-envs 32 --subproc \
  --network-arch "512,512,256" --n-epochs 20 \
  --parent-checkpoint runs/baseline-continued-TIMESTAMP/checkpoints/ppo_mission_XXXXX_steps \
  --branch-name test-new-reward --notes "Testing increased stealth rewards" --seed 42
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
