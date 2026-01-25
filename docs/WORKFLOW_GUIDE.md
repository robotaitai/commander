# Mission Gym Training Workflow Guide

## 🚀 Starting a New Training Run with Changes

### Scenario: Add Defenders and Branch from Best Policy

When you want to modify the scenario (e.g., add defenders) and continue training from your best checkpoint, follow this workflow:

---

## Step 1: Clean Up Old Runs

First, identify and remove unrelated runs to keep your workspace clean:

```bash
# Preview what will be deleted (dry run)
python cleanup_runs.py --keep-active-lineage --dry-run

# Keep only the most recent lineage tree and delete the rest
python cleanup_runs.py --keep-active-lineage

# Or delete only failed runs (0 checkpoints)
python cleanup_runs.py --delete-failed
```

**Recommended:** Use `--keep-active-lineage` to keep only your best run and its lineage.

---

## Step 2: Modify the Scenario

Edit `configs/scenario.yaml` to add/remove units:

```yaml
defenders:
  - unit_type: DEF_UGV
    spawn:
      x: 100.0
      y: 85.0
      heading: 90.0
    patrol_waypoints:
      - [90.0, 90.0]
      - [110.0, 90.0]
      - [110.0, 110.0]
      - [90.0, 110.0]
  
  # Add second defender
  - unit_type: DEF_UGV
    spawn:
      x: 100.0
      y: 115.0
      heading: -90.0
    patrol_waypoints:
      - [85.0, 105.0]
      - [115.0, 105.0]
      - [115.0, 95.0]
      - [85.0, 95.0]
```

---

## Step 3: Start a New Branched Training Run

**IMPORTANT:** When you change the number of units, you CANNOT load from a checkpoint because the observation and action spaces will be incompatible. You must train from scratch.

### Option A: Train from Scratch with New Scenario

```bash
python -m mission_gym.scripts.train_ppo \
    --timesteps 50000000 \
    --n-envs 16 \
    --run-name "2-defenders-experiment" \
    --notes "Added second UGV defender to increase difficulty"
```

### Option B: Continue Training (Same Scenario)

If you're NOT changing the number of units, you can branch from your best checkpoint:

```bash
# Find your best checkpoint
ls -lht runs/branch-20260122-193222/checkpoints/*.zip | head -1

# Resume training from it
python -m mission_gym.scripts.train_ppo \
    --timesteps 50000000 \
    --n-envs 16 \
    --parent-checkpoint runs/branch-20260122-193222/checkpoints/ppo_mission_27670000_steps \
    --branch-name "continued-training" \
    --notes "Continuing from 27.67M steps with refined rewards"
```

**Note:** The `.zip` extension is added automatically, don't include it!

---

## Step 4: Monitor Training

### Watch the Unified Dashboard

Open in your browser:
```
file:///home/itai/code/commander/runs/dashboard.html
```

The dashboard will:
- Show all runs sorted by most recent activity (active run at top)
- Display lineage relationships (indented with `↳` for child runs)
- Auto-refresh every 5 seconds

### View Lineage-Filtered Dashboard

To generate a dashboard showing only runs from a specific lineage tree:

```python
from mission_gym.scripts.run_utils import generate_unified_dashboard

# Show only the active lineage tree
generate_unified_dashboard(lineage_filter="active")

# Show specific lineage tree
generate_unified_dashboard(lineage_filter="branch-20260122-193222")
```

---

## Step 5: Evaluate Your Model

### Evaluate with Run's Config Snapshot

```bash
# Automatically uses the run's config directory
python -m mission_gym.scripts.evaluate \
    --model runs/branch-20260122-193222/checkpoints/ppo_mission_27670000_steps

# Or specify a different config directory
python -m mission_gym.scripts.evaluate \
    --model runs/branch-20260122-193222/checkpoints/ppo_mission_27670000_steps \
    --config-dir configs/
```

---

## Understanding Lineage and Compatibility

### Lineage Tracking

Every run records:
- **Parent checkpoint** (if branched)
- **Observation space signature** (vector dimensions)
- **Action space signature** (MultiDiscrete dimensions)
- **Config hash** (checksums of all config files)
- **Git commit** (code version)

### Compatibility Rules

You can load a checkpoint ONLY if:

1. ✅ **Number of units is unchanged**
   - Same number of attackers
   - Same number of defenders
   
2. ✅ **Action space is unchanged**
   - Same actions available per unit
   
3. ✅ **Observation space is unchanged**
   - Same vector dimension

### What You CAN Change Without Breaking Compatibility

✅ **Reward weights** in `configs/reward.yaml`  
✅ **Unit stats** (speed, integrity, damage) in `configs/units_*.yaml`  
✅ **Defender behavior** in `configs/defender_randomization.yaml`  
✅ **Scenario randomization** in `configs/scenario_randomization.yaml`  
✅ **Engagement mechanics** in `configs/engagement.yaml`  
✅ **World obstacles** in `configs/world.yaml`  
✅ **Termination conditions** in `configs/world.yaml`

### What BREAKS Compatibility (Requires Training from Scratch)

❌ **Number of attackers** in `configs/scenario.yaml`  
❌ **Number of defenders** in `configs/scenario.yaml`  
❌ **Actions per unit** in `configs/units_*.yaml`  
❌ **Observation features** (changing vec_dim)

---

## Cleanup Strategy

### Keep Only Active Lineage

```bash
# This is the RECOMMENDED approach
python cleanup_runs.py --keep-active-lineage
```

**What it does:**
- Finds the most recently updated run (your active training)
- Traces its entire lineage tree (parents + siblings + children)
- Deletes everything else

**Example:**
```
Lineage tree (KEPT):
  branch-20260122-193222 (parent, 27.67M steps)
  ↳ continued-training-20260125-120000 (child, active)

Deleted:
  zen-comet-20260122-012517 (unrelated lineage)
  warm-panther-20260122-165345 (failed experiment)
  all other test runs
```

### Keep Specific Lineage

```bash
python cleanup_runs.py --keep-lineage branch-20260122-193222
```

Use this if you want to manually choose which lineage to preserve.

### Delete Only Failed Runs

```bash
python cleanup_runs.py --delete-failed
```

Removes runs with 0 checkpoints (crashed immediately).

---

## Best Practices

### 1. Always Use `--notes` When Branching

```bash
python -m mission_gym.scripts.train_ppo \
    --parent-checkpoint ... \
    --branch-name "reward-tuning-v2" \
    --notes "Increased zone_time_bonus from 0.1 to 0.5, reduced time_penalty"
```

This creates a searchable history of what you tried.

### 2. Clean Up Regularly

Every few experiments, run:
```bash
python cleanup_runs.py --keep-active-lineage --dry-run  # preview
python cleanup_runs.py --keep-active-lineage             # execute
```

### 3. Evaluate Before Branching

Before branching from a checkpoint, evaluate it to confirm it's worth continuing:

```bash
python -m mission_gym.scripts.evaluate \
    --model runs/YOUR-RUN/checkpoints/ppo_mission_XXXXX_steps
```

### 4. Check Compatibility Errors

If you get a compatibility error, it will tell you exactly what changed:

```
Observation space mismatch!
  Parent: {"type": "Box", "shape": [68]}
  Current: {"type": "Box", "shape": [85]}
This usually means you changed:
  - Number of units in scenario
```

Start a fresh run instead of trying to force-load the incompatible checkpoint.

---

## Quick Reference

| Task | Command |
|------|---------|
| Train from scratch | `python -m mission_gym.scripts.train_ppo --timesteps 50000000` |
| Branch from checkpoint | `python -m mission_gym.scripts.train_ppo --parent-checkpoint PATH --branch-name NAME` |
| Evaluate model | `python -m mission_gym.scripts.evaluate --model PATH` |
| Clean up old runs | `python cleanup_runs.py --keep-active-lineage` |
| View dashboard | Open `file:///.../runs/dashboard.html` |
| Find latest checkpoint | `ls -lht runs/YOUR-RUN/checkpoints/*.zip \| head -1` |

---

## Troubleshooting

### "Checkpoint not found" Error

Make sure you:
1. Don't include the `.zip` extension (it's added automatically)
2. Use the full path: `runs/RUN-NAME/checkpoints/ppo_mission_XXXXX_steps`

### "Action space mismatch!" Error

You changed the number of units or actions. Train from scratch:

```bash
python -m mission_gym.scripts.train_ppo --timesteps 50000000 --run-name "new-scenario"
```

### Evaluation Shows Wrong Number of Units

You're using the run's config snapshot, not your local `configs/`. Use `--config-dir` to override:

```bash
python -m mission_gym.scripts.evaluate \
    --model runs/OLD-RUN/checkpoints/... \
    --config-dir configs/
```

---

## GPU Crashed? Resume Training

If training crashes due to GPU errors:

```bash
# Find the latest checkpoint
ls -lht runs/YOUR-RUN/checkpoints/*.zip | head -1

# Resume from it
python -m mission_gym.scripts.train_ppo \
    --timesteps 50000000 \
    --parent-checkpoint runs/YOUR-RUN/checkpoints/ppo_mission_XXXXX_steps \
    --branch-name "resumed-XXXXX" \
    --notes "Resumed after GPU crash at XXXXX steps"
```

This creates a new run directory and continues training.
