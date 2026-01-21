# Policy Branching and Lineage Tracking Guide

## Overview

Mission Gym now supports **policy branching** - the ability to load a trained checkpoint as a "parent" and create a new training lineage with full compatibility tracking. This enables safe experimentation while maintaining a clear history of policy evolution.

## Quick Start

### Basic Training (No Parent)

```bash
python -m mission_gym.scripts.train_ppo \
  --timesteps 500000 \
  --n-envs 16 \
  --run-name baseline
```

This creates a new run with auto-generated name: `baseline-20260122-143052`

### Branch from Existing Policy

```bash
python -m mission_gym.scripts.train_ppo \
  --parent-checkpoint runs/baseline-20260122-143052/checkpoints/ppo_mission_100000_steps.zip \
  --branch-name explore-v2 \
  --timesteps 500000 \
  --notes "Testing dense rewards around objective"
```

This creates: `explore-v2-20260122-150430` with full lineage tracking.

## Features

### 1. Automatic Lineage Tracking

Every run creates `lineage.json` with:
- **Parent Info**: checkpoint path, run name, run directory
- **Git Commit**: SHA of code version used
- **Config Hash**: SHA256 of all config files
- **Space Signatures**: Observation and action space shapes/types
- **Metadata**: Branch name, notes, timestamp

Example `lineage.json`:
```json
{
  "created_at": "2026-01-22T15:04:30.123456",
  "git_commit_hash": "6287d95abc...",
  "config_hash": "a3f2e1b...",
  "parent_checkpoint_path": "runs/baseline-.../ppo_mission_100000_steps.zip",
  "parent_run_name": "baseline-20260122-143052",
  "branch_name": "explore-v2",
  "notes": "Testing dense rewards around objective",
  "obs_space_signature": {
    "type": "Box",
    "shape": [42],
    "dtype": "float32"
  },
  "action_space_signature": {
    "type": "MultiDiscrete",
    "nvec": [9, 9, 9, 9]
  }
}
```

### 2. Compatibility Checking

Before loading a checkpoint, Mission Gym automatically checks if the current environment is compatible with the parent policy.

**Compatible**:
```bash
✓ Checkpoint is compatible
✓ Parent checkpoint loaded: ppo_mission_100000_steps.zip
  Parent trained for 100,000 timesteps
```

**Incompatible** (e.g., changed number of units):
```bash
✗ Checkpoint incompatibility detected!
  Observation space mismatch!
    Parent: {"type": "Box", "shape": [42], ...}
    Current: {"type": "Box", "shape": [52], ...}
  This usually means you changed:
    - Number of units in scenario
    - Observation features (vec_dim)
    - From Dict to Box or vice versa

  To fix this, either:
    1. Revert config changes to match parent checkpoint
    2. Train a new policy from scratch (remove --load-checkpoint)
```

### 3. Run Naming

| Scenario | CLI Args | Result Run Name |
|----------|----------|----------------|
| New training | `--run-name my-exp` | `my-exp-20260122-143052` |
| Auto-generated | (none) | `swift-falcon-20260122-143052` |
| Branch with name | `--parent-checkpoint ... --branch-name explore` | `explore-20260122-143052` |
| Branch without name | `--parent-checkpoint ...` | `branch-20260122-143052` |
| Prefer run-name | `--run-name my-exp --branch-name explore` | `my-exp-20260122-143052` |

## Configuration Compatibility

### ✅ Safe Changes (Won't Break Compatibility)

These changes can be made between parent and branch without issues:

1. **Reward Function** (`configs/reward.yaml`)
   - Change any weight value
   - Enable/disable reward components
   - Add new components (if they don't change obs space)

2. **Enable/Disable Flags** (`configs/engagement.yaml`)
   ```yaml
   enable:
     tag: false    # Disable TAG without breaking action space
     scan: false   # Disable SCAN without breaking action space
   ```

3. **Physics Parameters**
   - Speed, acceleration, turn rates
   - Cooldown durations
   - Damage values
   - Integrity thresholds

4. **Sensors** (`configs/sensors.yaml`)
   - Range, FOV, update rates
   - Noise parameters
   - LOS requirements

5. **World Layout** (`configs/world.yaml`)
   - Obstacle positions
   - Obstacle types/sizes
   - Arena boundaries (if units stay in bounds)

6. **Objective** (`configs/scenario.yaml`)
   - Objective position
   - Capture requirements
   - Episode duration (if policy doesn't depend on exact timing)

### ❌ Breaking Changes (Will Fail Compatibility Check)

These changes will cause the checkpoint to be incompatible:

1. **Number of Units** (`configs/scenario.yaml`)
   ```yaml
   # Changing this breaks both obs and action spaces
   attackers:
     - type: UGV_A    # Can't add/remove units
     - type: UGV_B
   ```

2. **Action Space**
   - Adding/removing actions from unit types
   - **Solution**: Use enable flags instead

3. **Observation Features**
   - Adding new per-unit features to `_build_vector()`
   - Changing feature normalization ranges
   - Changing from Box to Dict or vice versa

4. **Number of Observation Features**
   - Changes to `vec_dim` calculation
   - Adding global features

## CLI Arguments Reference

```bash
python -m mission_gym.scripts.train_ppo [OPTIONS]

Training Options:
  --timesteps INT              Total training timesteps (default: 100000)
  --n-envs INT                 Number of parallel environments (default: 4)
  --subproc                    Use SubprocVecEnv for true parallelism
  --seed INT                   Random seed (default: 42)
  --eval-freq INT              Evaluation frequency (default: 5000)

Branching Options:
  --load-checkpoint PATH       Resume training from checkpoint
  --parent-checkpoint PATH     Alias for --load-checkpoint (clearer intent)
  --branch-name NAME           Branch name (creates '<name>-<timestamp>' run)
  --notes TEXT                 Notes about this run (saved in lineage.json)

Naming Options:
  --run-name NAME              Custom run name (timestamp auto-appended)
```

## Best Practices

### 1. Branch Naming Convention

Use descriptive branch names that explain what you're testing:

```bash
--branch-name reward-shaping-v1    # Testing new reward function
--branch-name dense-obj-bonus      # Dense bonuses near objective
--branch-name no-collision-penalty # Removing collision penalty
--branch-name explore-beta-0.05    # Testing entropy coefficient
```

### 2. Use Notes for Context

```bash
--notes "Increased ring bonuses to [10, 20, 30]. Reduced time penalty to -0.0001."
```

### 3. Version Control Sync

Always commit your config changes to git before training:

```bash
git add configs/
git commit -m "Increase ring bonuses for explore-v2 branch"
git push

# Now train with the committed config
python -m mission_gym.scripts.train_ppo --parent-checkpoint ... --branch-name explore-v2
```

This ensures the `git_commit_hash` in `lineage.json` matches your config changes.

### 4. Test Compatibility First

Before starting a long training run, test if your changes are compatible:

```bash
# This will fail fast if incompatible
python -m mission_gym.scripts.train_ppo \
  --parent-checkpoint runs/baseline/checkpoints/ppo_mission_100000_steps.zip \
  --timesteps 1000 \
  --branch-name test-compat
```

### 5. Checkpoint Selection

For branching, prefer checkpoints with:
- **Higher performance**: Branch from your best checkpoint
- **Round numbers**: `100000_steps`, `500000_steps` for cleaner history
- **Stable training**: Avoid checkpoints during instability

## Workflow Example

### Initial Training

```bash
# 1. Train baseline policy
python -m mission_gym.scripts.train_ppo \
  --timesteps 500000 \
  --n-envs 16 \
  --run-name baseline-config-v1

# Result: runs/baseline-config-v1-20260122-140000/
```

### First Branch - Reward Tuning

```bash
# 2. Modify rewards (safe change)
vim configs/reward.yaml  # Increase zone_time: 5.0

# 3. Branch from baseline
python -m mission_gym.scripts.train_ppo \
  --parent-checkpoint runs/baseline-config-v1-.../checkpoints/ppo_mission_500000_steps.zip \
  --branch-name reward-tune-v1 \
  --timesteps 500000 \
  --notes "Increased zone_time to 5.0 for stronger objective focus"

# Result: runs/reward-tune-v1-20260122-150000/
```

### Second Branch - Disable Engagement

```bash
# 4. Disable TAG/SCAN (safe via enable flags)
vim configs/engagement.yaml
# Set tag: false, scan: false

# 5. Branch from reward-tune-v1
python -m mission_gym.scripts.train_ppo \
  --parent-checkpoint runs/reward-tune-v1-.../checkpoints/ppo_mission_500000_steps.zip \
  --branch-name no-combat \
  --timesteps 300000 \
  --notes "Disabled TAG/SCAN to test pure navigation objective"

# Result: runs/no-combat-20260122-160000/
```

### Branch Tree Visualization

```
baseline-config-v1-20260122-140000 (500k steps)
  └─ reward-tune-v1-20260122-150000 (500k steps)
       └─ no-combat-20260122-160000 (300k steps)
```

Each run's `lineage.json` traces back to its parent, creating a full policy genealogy.

## Troubleshooting

### Problem: "Checkpoint incompatibility detected"

**Cause**: Observation or action space changed between parent and current env.

**Solution**:
1. Check the error message for which space changed
2. Revert changes to `configs/scenario.yaml` (unit counts)
3. Revert changes to `configs/units_attackers.yaml` (action counts)
4. Or train from scratch without `--parent-checkpoint`

### Problem: "Checkpoint not found"

**Cause**: Invalid path to checkpoint file.

**Solution**:
1. List checkpoints: `ls runs/<run-name>/checkpoints/`
2. Use full path: `runs/baseline-.../checkpoints/ppo_mission_100000_steps.zip`
3. Or relative path from project root

### Problem: Training starts from step 0 despite loading checkpoint

**Cause**: This is expected! The `--timesteps` is the *additional* timesteps to train.

**Explanation**: If you load a 100k-step checkpoint and set `--timesteps 500000`, you'll train for 500k more steps (ending at 600k total).

**Solution**: This is correct behavior. The parent's trained steps are logged:
```
✓ Parent checkpoint loaded: ppo_mission_100000_steps.zip
  Parent trained for 100,000 timesteps
```

## Advanced: Lineage Queries

You can write scripts to analyze the lineage tree:

```python
import json
from pathlib import Path

def get_lineage_tree(run_name):
    """Trace lineage back to root policy."""
    lineage_path = Path(f"runs/{run_name}/lineage.json")
    if not lineage_path.exists():
        return [run_name]
    
    with open(lineage_path) as f:
        lineage = json.load(f)
    
    tree = [run_name]
    if "parent_run_name" in lineage:
        tree.extend(get_lineage_tree(lineage["parent_run_name"]))
    
    return tree

# Usage
tree = get_lineage_tree("no-combat-20260122-160000")
print(" → ".join(reversed(tree)))
# Output: baseline-config-v1-... → reward-tune-v1-... → no-combat-...
```

## Summary

**Policy branching** enables:
- ✅ Safe experimentation with config changes
- ✅ Clear history of policy evolution
- ✅ Automatic compatibility checking
- ✅ Full reproducibility (git + config hash)
- ✅ Easy A/B testing of hypotheses

**Key takeaway**: Use `--parent-checkpoint` + `--branch-name` for experiments. Let Mission Gym ensure compatibility!
