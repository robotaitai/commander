# Mission Gym: Observations and Actions

## Observation Space

The environment uses a **Dict observation space** with two components:

### 1. Bird's Eye View (BEV) - `bev`
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

### 2. Vector Features - `vec`
- **Shape**: `(N,)` where `N = num_attackers × 10 + 2`
- **Type**: `Box(-inf, inf, dtype=float32)`
- **Per-Unit Features** (10 per attacker):
  0. `x` - Normalized X position [0, 1]
  1. `y` - Normalized Y position [0, 1]
  2. `heading_cos` - Heading cosine component [-1, 1]
  3. `heading_sin` - Heading sine component [-1, 1]
  4. `speed` - Normalized speed [0, 1] (max ~15 m/s)
  5. `integrity` - Health normalized [0, 1] (max 100)
  6. `tag_cooldown` - Cooldown progress [0, 1]
  7. `scan_cooldown` - Scan cooldown progress [0, 1]
  8. `altitude` - Altitude band normalized [0, 1] (for UAVs)
  9. `disabled` - Binary flag [0, 1]

- **Global Features** (2):
  10. `time_remaining` - Normalized remaining time [0, 1]
  11. `capture_progress` - Objective capture progress [0, 1]

**Example**: With 4 attackers → `vec` shape = `(4 × 10 + 2) = 42`

---

## Action Space

**Type**: `MultiDiscrete` - One discrete action per attacker unit

**Current Setup** (from `scenario.yaml`):
- **4 attackers**: 1×UGV_A, 1×UGV_B, 1×UAV_A, 1×UAV_B
- **Action dimensions**: `[7, 7, 8, 8]` (total action space size = 7×7×8×8 = 3,136)

### Ground Units (UGV_A, UGV_B) - 7 Actions:
| Index | Action | Description |
|-------|--------|-------------|
| 0 | `NOOP` | No operation (do nothing) |
| 1 | `THROTTLE_UP` | Increase speed |
| 2 | `THROTTLE_DOWN` | Decrease speed |
| 3 | `TURN_LEFT` | Turn left (yaw) |
| 4 | `TURN_RIGHT` | Turn right (yaw) |
| 5 | `BRAKE` | Decelerate quickly |
| 6 | `HOLD` | Maintain current state |

### Air Units (UAV_A, UAV_B) - 8 Actions:
| Index | Action | Description |
|-------|--------|-------------|
| 0 | `NOOP` | No operation |
| 1 | `THROTTLE_UP` | Increase speed |
| 2 | `THROTTLE_DOWN` | Decrease speed |
| 3 | `YAW_LEFT` | Turn left (yaw) |
| 4 | `YAW_RIGHT` | Turn right (yaw) |
| 5 | `ALT_UP` | Increase altitude band |
| 6 | `ALT_DOWN` | Decrease altitude band |
| 7 | `HOLD` | Maintain current state |

**Note**: `TAG` and `SCAN` actions were removed to simplify the action space for learning. They can be re-added once basic capture behavior is learned.

---

## Action Format

When stepping the environment:
```python
action = np.array([a1, a2, a3, a4], dtype=np.int32)
# Where:
# a1 ∈ [0, 6] for UGV_A
# a2 ∈ [0, 6] for UGV_B  
# a3 ∈ [0, 7] for UAV_A
# a4 ∈ [0, 7] for UAV_B
```

---

## Example Observation

```python
obs = {
    'bev': np.array(shape=(128, 128, 8), dtype=float32),  # Raster map
    'vec': np.array(shape=(42,), dtype=float32)           # Vector features
}
```

---

## Key Details

- **Action Repeat**: Actions are executed for 5 physics steps (4 Hz command rate)
- **Episode Length**: 1200 steps = 300 seconds (5 minutes)
- **World Size**: 200m × 200m
- **Objective**: Circle at (100, 100) with 15m radius
- **Capture Time**: 20 seconds cumulative presence required to win
