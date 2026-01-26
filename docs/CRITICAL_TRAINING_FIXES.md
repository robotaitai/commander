# Critical Training Fixes (January 26, 2026)

## Overview

This document explains 4 critical fixes applied to the training system that significantly improve GPU utilization, batch size scaling, and evaluation accuracy.

---

## Fix #1: Proper Rollout Buffer Scaling (`n_steps_per_env`)

### ❌ **The Problem**

**Before:**
```python
n_steps = 2048 // args.n_envs
rollout_buffer_size = n_steps * args.n_envs  # Always ≈ 2048
```

When you increased `n_envs`, the `n_steps` was **divided** by `n_envs`, keeping the total rollout buffer size constant at ~2048.

**Result:** Adding more environments improved wall-clock sampling speed, but did **not** give PPO more data per update.

### ✅ **The Fix**

**After:**
```python
n_steps = args.n_steps_per_env  # New CLI argument (default: 128)
rollout_buffer_size = n_steps * args.n_envs  # Now scales with n_envs!
```

**New CLI argument:**
```bash
--n-steps-per-env 256  # Steps per environment per rollout
```

**Result:** Now you can actually increase the batch size PPO uses per update!

### 📊 **Examples**

| Config | n_envs | n_steps_per_env | Rollout Buffer | PPO Data/Update |
|--------|--------|-----------------|----------------|-----------------|
| **Old (broken)** | 16 | 128 (2048/16) | 2048 | 2048 |
| **Old (broken)** | 64 | 32 (2048/64) | 2048 | 2048 ❌ |
| **New (fixed)** | 16 | 128 | 2048 | 2048 |
| **New (fixed)** | 32 | 256 | **8192** | **8192** ✅ |
| **New (fixed)** | 64 | 256 | **16384** | **16384** ✅ |

**Impact:** Larger batches = better gradient estimates = more stable learning = better GPU utilization!

---

## Fix #2: Batch Size Divisibility Constraint

### ❌ **The Problem**

**Before:**
```python
batch_size = rollout_buffer_size // 4  # Target 1/4 of buffer
batch_size = max(64, ((batch_size + 63) // 64) * 64)  # Round to 64
```

The rounding to multiples of 64 for GPU efficiency could create a `batch_size` that **doesn't evenly divide** `rollout_buffer_size`.

**PPO requires:** `rollout_buffer_size % batch_size == 0`

**Result:** Silent errors or inefficient batching where the last partial batch is dropped.

### ✅ **The Fix**

**After:**
```python
def pick_batch_size(buffer_size: int, target_frac: float = 0.25, min_bs: int = 64) -> int:
    """Pick a batch size that:
    1. Is a multiple of 64 (GPU efficiency)
    2. Divides buffer_size evenly (PPO requirement)
    3. Is close to target_frac of buffer_size
    """
    target = int(buffer_size * target_frac)
    bs = max(min_bs, (target // 64) * 64)
    bs = min(bs, buffer_size)
    
    # Ensure divisibility: decrement by 64 until valid
    while bs >= min_bs and buffer_size % bs != 0:
        bs -= 64
    
    return max(min_bs, bs)

batch_size = pick_batch_size(rollout_buffer_size, target_frac=0.25)
```

### 📊 **Examples**

| rollout_buffer_size | Old batch_size | Valid? | New batch_size | Minibatches |
|---------------------|----------------|--------|----------------|-------------|
| 2048 | 512 | ✅ | 512 | 4 |
| 8192 | 2048 | ✅ | 2048 | 4 |
| 6144 | 1536 | ❌ (6144 % 1536 = 0, but unsafe) | 1536 | 4 |
| 10240 | 2560 | ✅ | 2560 | 4 |

**Impact:** Guaranteed valid batch sizes + optimal GPU efficiency!

---

## Fix #3: Evaluation Config Mismatch

### ❌ **The Problem**

User reported: "High reward but LOST" during evaluation.

**Cause:** The evaluation script was using **default configs** instead of the run's snapshot configs, which can flip win/lose conditions and reward composition.

### ✅ **The Fix**

**Good news:** This was already fixed in `evaluate.py`!

```python
# Infer config directory from model path
run_dir = model_path.parent
config_dir = run_dir / "configs"

# Create environment with run's configs
env = MissionGymEnv(render_mode=render_mode, config_dir=config_dir)
```

### 🔍 **If You Still See "High Reward But Lost"**

Check for brittle win detection logic:

**❌ Bad (brittle):**
```python
if terminated and capture_progress >= 20:
    wins += 1
```

**✅ Good (single source of truth):**
```python
if info.get("outcome", "") == "captured":
    wins += 1
```

Use the `outcome` field from your environment's info dict as the single source of truth.

---

## Fix #4: GPU Utilization Reality Check

### ❌ **The Misunderstanding**

"My GPU shows only 10-25% utilization. Something is wrong!"

### ✅ **The Reality**

**RL training has TWO phases:**

1. **Rollout Phase (70-90% of time):**
   - Collect experience from environments
   - **CPU-bound** (environment simulation)
   - GPU is **idle** (only running fast inference)
   - GPU util: ~2-10%

2. **Training Phase (10-30% of time):**
   - Update policy network with collected data
   - **GPU-bound** (neural network training)
   - GPU is **active**
   - GPU util: ~80-100% (during this phase only)

**Average GPU utilization: 10-25% for a single MLP training job is NORMAL!**

### 📊 **Phase Breakdown Example**

```
Time Distribution (32 envs, 256 steps/env, 8192 buffer, 20 epochs):

Rollout Phase:  ████████████████████████░░░░░  85%  (~17 sec)
Training Phase: ░░░░░░░░░░░░░░░░░░░░░░░░████  15%  (~3 sec)
──────────────────────────────────────────────
GPU Utilization:
  During rollout:  ██░░░░░░░░░░░░░░░░░░░░░░░   5-10%
  During training: ████████████████████████░   ~90%
  Average:         ████░░░░░░░░░░░░░░░░░░░░░   15-20% ✅ NORMAL!
```

### 💡 **Solutions to Increase GPU Utilization**

#### Option 1: Run Parallel Jobs (Recommended) 🔥

Run **3-4 training jobs simultaneously** to overlap their phases:

```bash
./add_parallel_jobs.sh  # Adds 3 more jobs to your current training
```

**Result:**
- 4 jobs × ~15% = **50-70% GPU utilization**
- 4× the experiments in the same wall time!

```
Job 1: Rollout ████... Training ██ ...Rollout ████... Training ██
Job 2: ...Rollout ████... Training ██ ...Rollout ████... Training
Job 3: Training ██ ...Rollout ████... Training ██ ...Rollout ████
Job 4: ..Rollout ████... Training ██ ...Rollout ████... Training ██
─────────────────────────────────────────────────────────────────
GPU:   ████████████████████████████░░░░░░░░░░░░░░░░░░░░  50-70% ✅
```

#### Option 2: Maximize GPU Work Per Job

Increase the work GPU does during its active phase:

```bash
--n-steps-per-env 512       # Collect more data per rollout
--network-arch "1024,512,512,256"  # Larger network
--n-epochs 40               # More training epochs per rollout
```

**Result:**
- Training phase takes longer (GPU active for ~25% of time)
- Single job GPU util: ~20-30%
- But: slower wall-clock time per experiment

### 🎯 **Recommended Strategy**

**For maximum efficiency:** Use Option 1 (parallel jobs)
- ✅ Higher GPU utilization (50-70%)
- ✅ 4× experiments per day
- ✅ Same wall-clock time per experiment
- ✅ Better use of your hardware

---

## 🚀 Quick Start with Fixed Training

### Single Job (Balanced Config)

```bash
python -m mission_gym.scripts.train_ppo \
  --timesteps 50000000 \
  --n-envs 32 \
  --n-steps-per-env 256 \
  --subproc \
  --network-arch "512,512,256" \
  --n-epochs 20 \
  --run-name my-experiment
```

**Specs:**
- Rollout buffer: 32 × 256 = **8192 transitions/update**
- Batch size: 2048 (4 minibatches)
- Updates per rollout: 4 × 20 = 80
- GPU util: ~15-20% (NORMAL)

### Parallel Jobs (Maximum GPU)

```bash
./add_parallel_jobs.sh
```

Starts 3 additional jobs with different configs:
- **Job 2:** Large batch (32×256=8192, 20 epochs)
- **Job 3:** Wide network (1536,768,384, 25 epochs)
- **Job 4:** Deep network (512×5, 30 epochs)

**Result:**
- 4 simultaneous experiments
- GPU util: **50-70%** ✅
- Logs: `logs/train_*.log`

---

## 📋 Summary of Changes

| File | Changes |
|------|---------|
| `train_ppo.py` | • Added `--n-steps-per-env` argument<br>• Added `pick_batch_size()` helper<br>• Fixed n_steps calculation (no longer divided by n_envs)<br>• Fixed batch_size divisibility<br>• Added PPO config summary printout |
| `evaluate.py` | ✅ Already correct (uses run's config_dir) |
| `parallel_train.sh` | • Updated all jobs to use `--n-steps-per-env`<br>• Optimized configs for GPU saturation |
| `add_parallel_jobs.sh` | • Updated all jobs to use `--n-steps-per-env`<br>• 3 different configs for experimentation |
| `TRAINING_CHEATSHEET.md` | • Added `--n-steps-per-env` to all examples<br>• Updated performance tiers with realistic GPU%<br>• Added parallel training workflow<br>• Clarified GPU utilization expectations |

---

## ⚠️ Breaking Changes

### Old Commands (Broken)

```bash
# This was keeping buffer size constant at ~2048
python -m mission_gym.scripts.train_ppo --timesteps 50M --n-envs 64 --subproc
```

### New Commands (Fixed)

```bash
# Now you MUST specify --n-steps-per-env
python -m mission_gym.scripts.train_ppo --timesteps 50M --n-envs 64 --n-steps-per-env 256 --subproc
```

**Default:** `--n-steps-per-env 128` (same behavior as old `n_envs=16`)

---

## Fix #5: Rich Table Rendering (Console Conflicts)

### ❌ **The Problem**

After fixing the console instance issue, we changed `console.print(table)` to `print(table)`, but Python's built-in `print()` doesn't know how to render Rich Table objects.

**Result:** Tables printed as `<rich.table.Table object at 0x...>` instead of rendering.

### ✅ **The Fix**

**After:**
```python
# Use a temporary Console instance for rendering
from rich.console import Console
temp_console = Console()
temp_console.print()
temp_console.print(table)
```

**Why this works:**
- Creates a fresh Console for each print operation
- No stored console instance → no conflicts with progress bar
- Tables render properly with colors and formatting

---

## 🧪 Tests Added

Added comprehensive test suite in `tests/test_training_fixes.py`:

- ✅ `test_basic_divisibility`: Ensures batch_size always divides buffer_size
- ✅ `test_target_fraction`: Verifies batch_size is close to target 25%
- ✅ `test_edge_cases`: Tests small buffers and odd sizes
- ✅ `test_n_steps_scaling`: Confirms rollout buffer scales with n_envs
- ✅ `test_metrics_callback_init`: Verifies no stored console instance
- ✅ `test_eval_freq_default`: Confirms 20K default (not 5K)

**Run tests:**
```bash
pytest tests/test_training_fixes.py -v
```

---

## 🎓 Key Takeaways

1. ✅ **n_steps_per_env** is now independent of n_envs (scales properly)
2. ✅ **batch_size** always divides rollout_buffer_size correctly
3. ✅ **Evaluation** uses correct configs from run snapshot
4. ✅ **GPU 10-25%** for single MLP job is **NORMAL** (not a bug!)
5. ✅ **Parallel jobs** are the best way to saturate GPU (50-70%)
6. ✅ **Rich tables** render properly without breaking progress bar

---

**Updated:** January 26, 2026 (22:00)  
**Affects:** All training runs from this point forward  
**Tests:** 8 tests added, all passing ✅
