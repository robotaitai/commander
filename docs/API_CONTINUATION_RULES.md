# Policy Continuation and Compatibility Rules

**Last Updated**: 2026-01-22

This document explains which configuration changes are safe for policy branching (continuing training from a checkpoint) and which will break compatibility.

---

## Quick Reference

| Change Type | Safe? | Notes |
|-------------|-------|-------|
| **Reward weights** | ✅ Yes | All weights in `reward.yaml` |
| **Enable/disable flags** | ✅ Yes | `tag_enabled`, `scan_enabled`, etc. |
| **Termination config** | ✅ Yes | Stagnation thresholds, early success |
| **Physics parameters** | ✅ Yes | Speeds, turn rates, accelerations |
| **Sensor configs** | ✅ Yes | Ranges, FOV, noise |
| **Obstacle layout** | ✅ Yes | Positions, sizes |
| **Number of defenders** | ✅ Yes | Defenders not in observation space! |
| **Defender types/behaviors** | ✅ Yes | AI-controlled, not policy input |
| **Number of attackers** | ❌ No | Changes obs + action space |
| **Attacker action list** | ❌ No | Don't add/remove actions |
| **Observation features** | ❌ No | Don't change vec_dim |

---

## Episode Termination and Outcomes

Episodes can end in 5 ways, tracked in `info["outcome"]`:

| Outcome | `terminated` | `truncated` | Description |
|---------|--------------|-------------|-------------|
| `"captured"` | ✅ True | False | Objective fully captured (20s cumulative in zone) |
| `"early_success"` | ✅ True | False | Reached early_success threshold (if configured) |
| `"stalled"` | False | ✅ True | No progress for stagnation_seconds (default 30s) |
| `"all_disabled"` | False | ✅ True | All attacker units disabled |
| `"timeout"` | False | ✅ True | Max episode duration reached (default 300s) |

### Stagnation Detection (Early Termination)

To speed up training, episodes end early if no progress is made:

**Progress is defined as**:
- Capture progress increases by any amount, **OR**
- Min distance to objective improves by ≥ `min_dist_epsilon` (default 1.0m)

**Configuration** (`configs/world.yaml`):
```yaml
termination:
  stagnation_seconds: 30.0              # End if no progress for 30s
  min_dist_epsilon: 1.0                 # Min improvement to count as progress
  early_success_capture_progress: null  # Optional: end early at threshold (e.g., 15.0)
```

---

## Safe Changes (Won't Break Compatibility)

These changes can be made between parent and branch without issues:

### 1. Reward Function (`configs/reward.yaml`)

**All safe to modify**:
```yaml
weights:
  capture_progress: 2.0          # ✅ Change any weight
  win_bonus: 200.0              # ✅ Change bonuses
  time_penalty: -0.001          # ✅ Change penalties
  zone_time: 5.0                # ✅ Change zone bonus
  min_dist_potential: 0.5       # ✅ Change shaping
  ring_bonus: 10.0              # ✅ Change ring values
  # ... all other weights

ring_distances: [80, 60, 40, 25, 15]  # ✅ Change thresholds

enable:
  detected_time_penalty: false  # ✅ Toggle components

gating:
  tag_hit_bonus: true           # ✅ Change gating rules
  engagement_reward_min_dist: 60.0  # ✅ Change thresholds
```

### 2. Termination Config (`configs/world.yaml`)

```yaml
termination:
  stagnation_seconds: 30.0              # ✅ Change stagnation time
  min_dist_epsilon: 1.0                 # ✅ Change progress threshold
  early_success_capture_progress: 15.0  # ✅ Add/remove/change early win
```

### 3. Engagement Config (`configs/engagement.yaml`)

**Important**: Keep actions in `units_attackers.yaml`, disable via flags!

```yaml
enable:
  tag: false    # ✅ Disable TAG action (action stays in list)
  scan: false   # ✅ Disable SCAN action (action stays in list)

tag_beam:
  range: 20.0          # ✅ Change tag mechanics
  optimal_range: 8.0   # ✅ Change damage falloff
  fov: 30.0            # ✅ Change FOV
  cooldown: 3.0        # ✅ Change cooldown
  damage: 25.0         # ✅ Change damage values
  # ... all other tag parameters

scan:
  duration: 0.5        # ✅ Change scan mechanics
  range_boost: 1.5     # ✅ Change boost
  cooldown: 5.0        # ✅ Change cooldown
```

### 4. Physics Parameters (`configs/world.yaml`, `configs/units_*.yaml`)

```yaml
# World physics
physics:
  tick_rate: 20.0      # ✅ Change (affects training speed)
  action_repeat: 5     # ✅ Change (affects command rate)

episode:
  max_duration: 300.0  # ✅ Change episode length

# Unit parameters (in units_attackers.yaml)
UGV_A:
  max_speed: 10.0      # ✅ Change speeds
  max_accel: 5.0       # ✅ Change acceleration
  max_turn_rate: 90.0  # ✅ Change turn rate
  initial_speed: 4.0   # ✅ Change initial speed
  # ... other physics params
```

### 5. Sensors (`configs/sensors.yaml`)

```yaml
sensors:
  lidar:
    max_range: 50.0         # ✅ Change range
    fov: 120.0              # ✅ Change FOV
    range_noise_std: 0.5    # ✅ Change noise
    # ... all sensor parameters
```

### 6. World Layout (`configs/world.yaml`)

```yaml
obstacles:
  - type: rect
    x: 100.0      # ✅ Change positions
    y: 40.0       # ✅ Change positions
    width: 25.0   # ✅ Change sizes
    height: 15.0  # ✅ Change sizes
    angle: 0.0    # ✅ Change angles
  # ✅ Add new obstacles
  # ✅ Remove obstacles
```

---

## Breaking Changes (Will Fail Compatibility Check)

These changes will cause the compatibility check to fail and prevent loading the checkpoint:

### 1. Number of Attackers (`configs/scenario.yaml`)

**❌ BREAKING**:
```yaml
attackers:
  - type: UGV_A
    x: 20
    y: 100
  - type: UGV_B
    x: 20
    y: 120
  - type: UAV_A
    x: 40
    y: 100
  - type: UAV_B
    x: 40
    y: 120
  # ❌ Adding a 5th attacker breaks compatibility!
  # Observation shape: (42,) → (52,)  # vec_dim = num_attackers × 10 + 2
  # Action shape: [9,9,9,9] → [9,9,9,9,9]
```

**Why**: Changes observation vector size (includes attacker features) and action space dimensions (policy controls attackers).

**✅ SAFE** - Number of defenders:
```yaml
defenders:
  - type: DEF_UGV
    x: 100
    y: 85
  # ✅ Adding more defenders is SAFE!
  # Defenders are AI-controlled and NOT in observation space
  # Observation shape: (42,) → (42,)  # unchanged!
  # Action space: [9,9,9,9] → [9,9,9,9]  # unchanged!
```

**Why safe**: Policy observes only attackers, not defenders. Defenders are part of environment dynamics.

### 2. Action Lists (`configs/units_attackers.yaml`)

**❌ BREAKING**:
```yaml
UGV_A:
  actions:
    - STOP
    - NORTH
    - NORTHEAST
    - EAST
    - SOUTHEAST
    - SOUTH
    - SOUTHWEST
    - WEST
    - NORTHWEST
    # ❌ Adding a 10th action breaks compatibility!
    # Action space: [9,9,9,9] → [10,9,9,9]
```

**Why**: Changes action space dimensions.

**✅ CORRECT WAY** to disable actions:
```yaml
# In configs/engagement.yaml
enable:
  tag: false    # Disable TAG without removing from action list
  scan: false   # Disable SCAN without removing from action list
```

### 3. Observation Features

**❌ BREAKING**: Modifying `mission_gym/env.py` to:
- Add new per-unit features to `_build_vector()`
- Remove features
- Change feature order
- Change normalization (would break learned weights)

---

## Branching Workflow

### Step 1: Train Baseline

```bash
python -m mission_gym.scripts.train_ppo \
  --timesteps 500000 \
  --n-envs 16 \
  --run-name baseline-v1
```

### Step 2: Modify Config (Safe Changes Only)

```bash
# Example: Increase zone time bonus
vim configs/reward.yaml
# Change: zone_time: 2.0 → zone_time: 5.0

# Commit changes (optional but recommended)
git add configs/reward.yaml
git commit -m "Increase zone_time bonus for exploration"
```

### Step 3: Branch from Baseline

```bash
python -m mission_gym.scripts.train_ppo \
  --parent-checkpoint runs/baseline-v1-<timestamp>/final_model.zip \
  --branch-name higher-zone-bonus \
  --timesteps 500000 \
  --notes "Increased zone_time to 5.0 for stronger objective focus"
```

**What happens**:
1. Compatibility check runs automatically
2. If compatible: Training starts from parent weights
3. If incompatible: Error with explanation of what changed
4. Lineage tracked in `lineage.json` with parent info

### Step 4: Compare Results

```bash
# Evaluate baseline
python -m mission_gym.scripts.evaluate \
  --model runs/baseline-v1-<timestamp>/final_model

# Evaluate branch
python -m mission_gym.scripts.evaluate \
  --model runs/higher-zone-bonus-<timestamp>/final_model

# Compare in unified dashboard
open runs/dashboard.html
```

---

## Compatibility Error Examples

### Example 1: Added Attacker (Breaking)

```bash
$ python -m mission_gym.scripts.train_ppo \
  --parent-checkpoint runs/baseline/final_model.zip \
  --branch-name five-attackers

✗ Checkpoint incompatibility detected!
  Observation space mismatch!
    Parent: {"type": "Box", "shape": [42], "dtype": "float32"}  # 4 attackers × 10 + 2
    Current: {"type": "Box", "shape": [52], "dtype": "float32"} # 5 attackers × 10 + 2
  This usually means you changed:
    - Number of attackers in scenario
    - Observation features (vec_dim)
    - From Dict to Box or vice versa

  To fix this, either:
    1. Revert config changes to match parent checkpoint
    2. Train a new policy from scratch (remove --load-checkpoint)

  Note: Adding defenders does NOT cause this error!
```

### Example 2: Changed Actions (Breaking)

```bash
$ python -m mission_gym.scripts.train_ppo \
  --parent-checkpoint runs/baseline/final_model.zip \
  --branch-name ten-actions

✗ Checkpoint incompatibility detected!
  Action space mismatch!
    Parent: {"type": "MultiDiscrete", "nvec": [9, 9, 9, 9]}
    Current: {"type": "MultiDiscrete", "nvec": [10, 9, 9, 9]}
  This usually means you changed:
    - Number of units
    - Number of actions per unit

  To fix this, either:
    1. Revert config changes to match parent checkpoint
    2. Train a new policy from scratch (remove --load-checkpoint)
```

---

## Best Practices

1. **Always commit config changes before branching**
   - Ensures `git_commit_hash` in `lineage.json` matches your changes
   - Helps trace which experiments used which configs

2. **Use descriptive branch names**
   - Good: `higher-zone-bonus`, `no-collision-penalty`, `fast-stagnation`
   - Bad: `test1`, `experiment`, `try-this`

3. **Add notes to explain changes**
   - Use `--notes` to document what you're testing
   - Shows up in `lineage.json` and helps with analysis

4. **Test compatibility early**
   - Run a short branch (`--timesteps 10000`) to verify compatibility
   - Catches issues before committing to long training runs

5. **Keep action lists stable**
   - Never add/remove actions from `units_attackers.yaml`
   - Always use `enable.tag` / `enable.scan` flags to disable
   - This ensures all experiments can branch from each other

6. **Document breaking changes**
   - If you must make breaking changes, document in commit message
   - Start a new baseline policy line
   - Name clearly: `baseline-v2-five-units`

---

## Testing Configuration Compatibility

Before starting a long training run, test if your config changes are compatible:

```bash
# Quick compatibility test
python -c "
from mission_gym.scripts.run_utils import check_checkpoint_compatibility
from mission_gym.env import MissionGymEnv

env = MissionGymEnv()
is_compatible, error = check_checkpoint_compatibility(
    'runs/baseline/final_model.zip',
    env.observation_space,
    env.action_space
)

if is_compatible:
    print('✓ Configuration is compatible!')
else:
    print(f'✗ Incompatible: {error}')
"
```

---

## Summary

| Want to change | Safe? | Method |
|----------------|-------|--------|
| Reward weights | ✅ | Edit `reward.yaml` directly |
| Disable TAG/SCAN | ✅ | Set `enable.tag: false` in `engagement.yaml` |
| Episode length | ✅ | Change `max_duration` in `world.yaml` |
| Stagnation time | ✅ | Change `stagnation_seconds` in `world.yaml` |
| Unit speeds | ✅ | Change `max_speed` in `units_*.yaml` |
| **Number of defenders** | ✅ | Edit `scenario.yaml` defenders section |
| **Defender behaviors** | ✅ | Edit `defender_randomization.yaml` |
| Add/remove **attackers** | ❌ | Requires new baseline |
| Add/remove actions | ❌ | Requires new baseline |
| Change obs features | ❌ | Requires new baseline |

**Rule of thumb**: 
- If it changes what the policy sees (attackers) or does → **breaking**
- If it changes environment dynamics (defenders, rewards, physics) → **safe**
- **Key insight**: Policy only observes attackers, not defenders!
