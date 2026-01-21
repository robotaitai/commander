# Mission Gym Documentation

Mission Gym is a game-like reinforcement learning environment for commanding a fleet to capture an objective zone while defenders patrol and attempt to disable your units.

## 📚 Documentation Index

- **[API & Continuation Rules](API_CONTINUATION_RULES.md)** - **START HERE** - Safe vs breaking config changes for policy branching
- **[Observations and Actions](OBSERVATIONS_AND_ACTIONS.md)** - Complete API reference for observation space and action space
- **[Policy Branching Guide](BRANCHING_GUIDE.md)** - Complete guide to policy branching, lineage tracking, and compatibility
- **[Instructions](Insructions.md)** - Original project specifications and requirements
- **[Scenario View](scenario_view.png)** - Visual reference for the default scenario

## 🎮 Quick Start

### Installation

```bash
cd /home/itai/code/commander
pip install -e .
```

### Run Manual Control

```bash
python -m mission_gym.scripts.play_manual
```

**Controls:**
- Arrow keys: Navigate menu / Control selected unit
- Number keys (1-4): Select unit
- Space: Pause/Resume
- ESC: Quit

### Train with PPO

```bash
# Basic training (100k timesteps, 4 parallel envs)
python -m mission_gym.scripts.train_ppo --timesteps 100000

# Advanced training (more envs, custom name)
python -m mission_gym.scripts.train_ppo \
  --timesteps 500000 \
  --n-envs 16 \
  --subproc \
  --run-name my-experiment
```

### Resume Training from Checkpoint

```bash
python -m mission_gym.scripts.train_ppo \
  --timesteps 500000 \
  --load-checkpoint runs/my-run/checkpoints/ppo_mission_100000_steps.zip
```

### Policy Branching and Lineage Tracking

Mission Gym now supports **policy branching** - loading a parent checkpoint and creating a new training lineage with full compatibility tracking:

```bash
# Create a branch from an existing checkpoint
python -m mission_gym.scripts.train_ppo \
  --parent-checkpoint runs/baseline/checkpoints/ppo_mission_100000_steps.zip \
  --branch-name explore-v2 \
  --timesteps 500000 \
  --notes "Testing new reward shaping"
```

**Features:**
- **Lineage Tracking**: Every run saves `lineage.json` with parent info, git commit, config hash, and space signatures
- **Compatibility Checking**: Automatically detects incompatible changes in observation/action spaces
- **Run Naming**: Branch runs are named `<branch-name>-<timestamp>` (e.g., `explore-v2-20260122-143052`)

**What changes are safe?**
- ✅ Reward weights in `reward.yaml`
- ✅ Enable/disable flags (e.g., `tag_enabled`, `scan_enabled`)
- ✅ Physics parameters (speeds, turn rates)
- ✅ Sensor configurations
- ✅ Obstacle layouts

**What breaks compatibility?**
- ❌ Number of units in scenario (changes action space)
- ❌ Number of observation features per unit (changes obs space)
- ❌ Switching from vector to dict observations or vice versa
- ❌ Adding/removing actions from unit types (changes action space)

**Tip**: To disable TAG/SCAN without breaking compatibility, use the enable flags in `configs/engagement.yaml` instead of removing actions from unit configs.

### Evaluate Trained Model

```bash
python -m mission_gym.scripts.evaluate \
  --model runs/my-run/final_model \
  --episodes 10
```

### Record Video

```bash
python -m mission_gym.scripts.record_video \
  --model runs/my-run/final_model \
  --episodes 3
```

## 🏗️ Architecture

### Environment

- **Type**: Gymnasium environment with `Box` observation space (vector-only)
- **Observation**: `Box(shape=(42,), dtype=float32)` - Pure vector features (no images)
  - Per-unit features (10 × 4 units = 40): position, heading (cos/sin), speed, integrity, cooldowns, altitude, disabled
  - Global features (2): time_remaining, capture_progress
  - **Note**: BEV rendering is available via `env.get_debug_bev()` for visualization only
- **Action Space**: `MultiDiscrete([9, 9, 9, 9])` - One high-level action per unit
- **Policy**: `MlpPolicy` (2-layer MLP, 256 units per layer)
- **Episode Length**: 1200 steps (5 minutes simulation time)

**Why Vector-Only?**
- ✅ **10x faster training**: No CNN processing overhead
- ✅ **Better for branching**: Smaller models, faster checkpoints
- ✅ **Still has BEV**: Available for rendering and debugging via `env.get_debug_bev()`
- ✅ **Simpler policy**: Pure MLP is easier to analyze and transfer

### Action Space (Simplified)

Each unit can execute one of 9 high-level commands:
- **STOP** (0): Stop moving
- **NORTH** (1): Move north (90°)
- **NORTHEAST** (2): Move northeast (45°)
- **EAST** (3): Move east (0°)
- **SOUTHEAST** (4): Move southeast (315°)
- **SOUTH** (5): Move south (270°)
- **SOUTHWEST** (6): Move southwest (225°)
- **WEST** (7): Move west (180°)
- **NORTHWEST** (8): Move northwest (135°)

A low-level controller smoothly turns and accelerates toward targets while respecting physics constraints.

### Reward Function

The reward function is modular and configured via `configs/reward.yaml`:

**Objective Rewards:**
- Capture progress: +2.0 per second of capture
- Win bonus: +200.0 on successful capture
- Zone entry: +20.0 one-time bonus
- Zone time: +2.0 per second in zone

**Shaping Rewards:**
- Distance potential: +0.5 per meter closer to objective
- Ring bonuses: +5.0 for crossing distance milestones
- Formation spread: +0.005 for staying spread out

**Engagement Bonuses:**
- Tag hit: +0.2 per successful tag (mission-aligned)
- Defender disabled: +10.0 per defender disabled (mission-aligned)

**Penalties:**
- Time: -0.001 per step
- Collisions: -0.5 per collision
- Integrity loss: -0.1 per point
- Unit disabled: -20.0 per unit
- Detection: -0.05 per step while detected

## 📊 Monitoring

### Dashboard

Training automatically generates HTML dashboards:
- **Unified Dashboard**: `runs/dashboard.html` - View all training runs
- **Individual Run**: `runs/<run-name>/dashboard.html` - Detailed metrics for specific run

Features:
- Real-time training progress
- Reward component breakdown
- Episode metrics and statistics
- GPU monitoring
- Evaluation snapshots
- Quick command buttons
- Action log (last 100 commands)

### TensorBoard

```bash
tensorboard --logdir runs/<run-name>/logs
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_env.py

# Run smoke test
python -m mission_gym.scripts.smoke_test
```

## 📝 Configuration

All game parameters are configurable via YAML files in `configs/`:

- `world.yaml` - Arena size, obstacles, physics settings
- `scenario.yaml` - Unit spawns, objective location
- `units_attackers.yaml` - Attacker unit types and capabilities
- `units_defenders.yaml` - Defender unit types and capabilities
- `sensors.yaml` - Sensor configurations (lidar, radar, camera)
- `engagement.yaml` - Tag/disable mechanics, damage, cooldowns
- `reward.yaml` - Reward function weights and toggles

## 🎯 Default Scenario

- **Arena**: 200m × 200m with obstacles (buildings, rocks, forest)
- **Objective**: Circle at (100, 100) with 15m radius
- **Attackers**: 4 units (2 UGVs, 2 UAVs) starting at edges
- **Defenders**: 3 UGVs patrolling around objective
- **Win Condition**: Stay in objective zone for 20 cumulative seconds

## 🚀 Performance

- **Physics Rate**: 20 Hz (internal simulation)
- **Command Rate**: 4 Hz (agent actions every 5 physics steps)
- **Training Speed**: ~90 FPS on RTX 4070 (16 parallel envs)
- **Episode Length**: 1200 steps = 300 seconds = 5 minutes

## 🔬 Metrics Tracked

**Mission Outcomes:**
- Win/loss, termination reason, episode time

**Fleet Performance:**
- Distance traveled, speed, formation metrics

**Engagement Stats:**
- Tag attempts, hits, conversion rates
- Tag opportunities (diagnostic)
- Average distance to defenders

**Detection/Stealth:**
- Detected time percentage
- First detection time
- Detection events

**Collisions:**
- Total collisions per episode
- Per-unit collision counts

## 📖 Additional Resources

- See `OBSERVATIONS_AND_ACTIONS.md` for detailed observation and action space specifications
- Check `runs/<run-name>/configs/` for snapshot of configuration used in each training run
- View `runs/<run-name>/summary.txt` for training run summaries
