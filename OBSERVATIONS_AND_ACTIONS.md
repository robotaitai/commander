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
- **Action dimensions**: `[9, 9, 9, 9]` (total action space size = 9⁴ = 6,561)

### All Units (Ground and Air) - 9 High-Level Actions:

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
