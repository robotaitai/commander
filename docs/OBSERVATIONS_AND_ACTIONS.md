# Mission Gym: Observations and Actions API

**Last Updated**: 2026-01-22  
**Version**: Vector-only observations (MlpPolicy)

This document defines the **stable API** for Mission Gym. Changes to observation or action spaces will break checkpoint compatibility and require retraining from scratch.

---

## Observation Space

The environment uses a **Box observation space** (vector-only, no images):

### Vector Features - Observation
- **Shape**: `(N,)` where `N = num_attackers × 10 + 2`
- **Type**: `Box(-inf, inf, dtype=float32)`
- **Normalization**: All features normalized to reasonable ranges for neural network training

**Per-Unit Features** (10 per attacker):
| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | `x` | [0, 1] | X position normalized to arena width |
| 1 | `y` | [0, 1] | Y position normalized to arena height |
| 2 | `heading_cos` | [-1, 1] | Cosine of heading angle |
| 3 | `heading_sin` | [-1, 1] | Sine of heading angle |
| 4 | `speed` | [0, 1] | Current speed normalized to max_speed |
| 5 | `integrity` | [0, 1] | Health normalized to initial_integrity |
| 6 | `tag_cooldown` | [0, 1] | TAG cooldown progress (0=ready, 1=cooling) |
| 7 | `scan_cooldown` | [0, 1] | SCAN cooldown progress (0=ready, 1=cooling) |
| 8 | `altitude` | [0, 1] | Altitude band (UAVs only, UGVs always 0) |
| 9 | `disabled` | [0, 1] | Unit disabled flag (0=active, 1=disabled) |

**Global Features** (2):
| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 10 | `time_remaining` | [0, 1] | Remaining episode time normalized to max_duration |
| 11 | `capture_progress` | [0, 1] | Objective capture progress (normalized to capture_time_required) |

**Example**: Default scenario with 4 attackers → observation shape = `(4 × 10 + 2) = 42`

### ⚠️ Compatibility Note
Changing the number of attackers in `scenario.yaml` will change observation shape and **break checkpoint compatibility**.

### Bird's Eye View (BEV) - Debug/Visualization Only

**Important**: BEV is **NOT part of the policy observation**. It's available via `env.get_debug_bev()` for rendering and debugging.

- **Shape**: `(128, 128, 8)` - 128×128 pixel raster with 8 channels
- **Type**: `Box(0.0, 1.0, dtype=float32)`
- **Channels**:
  0. **Obstacles** - Static obstacles (buildings, rocks, trees)
  1. **Objective Zone** - Target capture area (15m radius circle)
  2. **Attackers** - Your units (friendly)
  3. **Defenders** - Enemy units (patrolling)
  4. **Attacker IDs** - Normalized unit IDs (0-1) for identification
  5. **Defender FOV** - Detection zones around defenders (0.3 intensity)
  6. **Tag Cooldown** - Heatmap showing cooldown status (0-1)
  7. **Capture Progress** - Broadcast across entire map (0-1)

**Usage**:
```python
env = MissionGymEnv()
obs, info = env.reset()  # obs is (42,) vector
bev = env.get_debug_bev()  # Get BEV for visualization
```

---

## Action Space

**Type**: `MultiDiscrete` - One discrete action per attacker unit

**Current Setup** (from `scenario.yaml`):
- **4 attackers**: 1×UGV_A, 1×UGV_B, 1×UAV_A, 1×UAV_B
- **Action dimensions**: `[9, 9, 9, 9]` (total action space size = 9⁴ = 6,561)

### All Units (Ground and Air) - 9 High-Level Actions:

**Note**: Attackers (your units) use these 9 directional actions. Defenders also have `TAG` and `SCAN` actions.

**Combat Behavior**: When defenders execute TAG or SCAN actions, they automatically **halt** (reduce speed to 0) to provide stable aim during engagement. This creates realistic "stop and shoot" behavior, making combat more visible.

| Index | Action | Target Heading | Description |
|-------|--------|----------------|-------------|
| 0 | `STOP` | N/A | Stop moving (speed → 0) |
| 1 | `NORTH` | 90° | Move north |
| 2 | `NORTHEAST` | 45° | Move northeast |
| 3 | `EAST` | 0° | Move east |
| 4 | `SOUTHEAST` | 315° | Move southeast |
| 5 | `SOUTH` | 270° | Move south |
| 6 | `SOUTHWEST` | 225° | Move southwest |
| 7 | `WEST` | 180° | Move west |
| 8 | `NORTHWEST` | 135° | Move northwest |

### How It Works:

**High-Level Control**: Each action immediately sets a target heading and speed (80% of max speed).

**Low-Level Controller**: The dynamics engine smoothly:
- Turns toward target heading (respecting max turn rate)
- Accelerates/decelerates toward target speed (respecting max acceleration)
- Handles realistic inertia and physics

**Benefits**:
- ✅ One action causes meaningful motion immediately
- ✅ No need for 20+ timesteps of "turn right" then "throttle up"
- ✅ Dramatically faster learning (capture learns in minutes)
- ✅ Still respects physics constraints (turn rate, acceleration)

**UAV Altitude**: UAVs automatically maintain altitude band 1 (mid-altitude) to fly over obstacles.

**Note**: This high-level action space is **10x easier to learn** than incremental throttle/turn commands while maintaining realistic physics.

---

## Action Format

When stepping the environment:
```python
action = np.array([a1, a2, a3, a4], dtype=np.int32)
# Where each action ∈ [0, 8]:
# a1 = UGV_A action (0-8)
# a2 = UGV_B action (0-8)  
# a3 = UAV_A action (0-8)
# a4 = UAV_B action (0-8)

# Examples:
action = np.array([3, 3, 3, 3])  # All units move EAST
action = np.array([1, 5, 2, 7])  # N, S, NE, W
action = np.array([0, 0, 0, 0])  # All units STOP
```

---

## Episode Outcomes

Episodes can end in 5 ways, tracked in `info["outcome"]`:

| Outcome | `terminated` | `truncated` | Description |
|---------|--------------|-------------|-------------|
| `"captured"` | True | False | Objective fully captured (20s in zone) |
| `"early_success"` | True | False | Early success threshold reached |
| `"stalled"` | False | True | No progress for 30s (configurable) |
| `"all_disabled"` | False | True | All attackers disabled |
| `"timeout"` | False | True | Max duration reached (300s) |

See [API_CONTINUATION_RULES.md](API_CONTINUATION_RULES.md) for details on early termination and stagnation detection.

---

## Example Usage

```python
import numpy as np
from mission_gym.env import MissionGymEnv

# Create environment
env = MissionGymEnv()

# Reset - returns vector observation
obs, info = env.reset()
print(obs.shape)  # (42,) for 4 attackers

# Access features
unit0_x = obs[0]           # Unit 0 X position
unit0_y = obs[1]           # Unit 0 Y position
time_remaining = obs[40]   # Global: time remaining
capture_progress = obs[41] # Global: capture progress

# Take action (all units move EAST)
action = np.array([3, 3, 3, 3], dtype=np.int32)
obs, reward, terminated, truncated, info = env.step(action)

# Check episode outcome
if terminated or truncated:
    outcome = info["outcome"]
    print(f"Episode ended: {outcome}")

# Get BEV for visualization (not part of policy obs)
bev = env.get_debug_bev()  # (128, 128, 8)

env.close()
```

---

## Key Details

- **Action Repeat**: Actions are executed for 5 physics steps (4 Hz command rate)
- **Episode Length**: Up to 1200 steps = 300 seconds (may end early via stagnation)
- **World Size**: 200m × 200m
- **Objective**: Circle at (100, 100) with 15m radius
- **Capture Time**: 20 seconds cumulative presence required to win
- **Stagnation**: Episodes end early if no progress for 30s (configurable)

---

## Policy Continuation

For information on safe vs. breaking configuration changes when branching policies, see [API_CONTINUATION_RULES.md](API_CONTINUATION_RULES.md).