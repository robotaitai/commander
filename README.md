# Mission Gym

A game-like reinforcement learning environment for commanding a fleet to capture an objective zone while defenders try to tag/disable units.

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install gymnasium numpy pyyaml stable-baselines3 torch pygame shapely tensorboard
```

## Quick Start

### Smoke Test

Verify the environment works correctly:

```bash
python -m mission_gym.scripts.smoke_test
```

### Manual Play

Play the game manually with keyboard controls:

```bash
python -m mission_gym.scripts.play_manual
```

**Controls:**
- `1-4`: Select attacker unit
- `W/S`: Throttle up/down
- `A/D`: Turn left/right
- `SPACE`: Brake
- `H`: Hold position
- `T`: Tag (attempt to disable nearby defender)
- `E`: Scan
- `R/F`: Altitude up/down (UAV only)
- `ESC`: Quit

---

## Training with Monitoring

### Start Training with HTML Dashboard

```bash
python -m mission_gym.scripts.train_ppo --timesteps 500000
```

This will create:
1. **HTML Dashboard** (`training_dashboard.html`) - Open in browser, auto-refreshes every 30s
2. **TensorBoard logs** (`./logs/`) - For detailed metrics

### Monitoring Options

#### Option 1: HTML Dashboard (Recommended for quick monitoring)

The HTML dashboard is automatically generated during training:

```bash
# Start training
python -m mission_gym.scripts.train_ppo --html-dashboard my_training.html

# Open the HTML file in your browser
# It auto-refreshes every 30 seconds
```

**Dashboard features:**
- 📊 Real-time stats: timesteps, episodes, FPS
- 📈 Reward curves with interactive charts
- 📋 Recent episode table with status
- 🎯 Evaluation results overlay

#### Option 2: TensorBoard (Detailed analysis)

```bash
# In a separate terminal, start TensorBoard
tensorboard --logdir ./logs

# Open http://localhost:6006 in your browser
```

**TensorBoard features:**
- Detailed training curves
- Hyperparameter tracking
- Policy loss, value loss, entropy
- Episode statistics

#### Option 3: Both (Full monitoring)

```bash
# Terminal 1: Training with HTML dashboard
python -m mission_gym.scripts.train_ppo --timesteps 500000

# Terminal 2: TensorBoard
tensorboard --logdir ./logs

# Browser Tab 1: training_dashboard.html (quick overview)
# Browser Tab 2: http://localhost:6006 (detailed analysis)
```

### Training Options

```bash
python -m mission_gym.scripts.train_ppo \
  --timesteps 500000 \        # Total training steps
  --n-envs 8 \                # Parallel environments (faster training)
  --eval-freq 10000 \         # Evaluation frequency
  --html-dashboard dash.html \ # HTML dashboard path
  --save-path my_model \      # Model save path
  --log-dir ./my_logs         # TensorBoard log directory
```

### Evaluate Trained Model

```bash
python -m mission_gym.scripts.evaluate --model ppo_mission_gym --episodes 10
```

---

## Visualization Options

### Option 1: Live Training Visualization (Pygame Window)

Watch the simulation during training with a live pygame window:

```bash
python -m mission_gym.scripts.live_training --timesteps 100000 --render-freq 50
```

This opens a pygame window every 50 episodes showing how the agent is performing.

### Option 2: Record Episodes as Video/GIF

Record simulation episodes as GIF or MP4:

```bash
# Record random agent
python -m mission_gym.scripts.record_video --episodes 3 --format gif

# Record trained agent
python -m mission_gym.scripts.record_video --model ppo_mission_gym --episodes 3 --format gif
```

Output saved to `recordings/` folder.

### Option 3: HTML Dashboard with Snapshots

The HTML dashboard automatically includes simulation snapshots from evaluations:

```bash
python -m mission_gym.scripts.train_ppo --timesteps 100000 --eval-freq 5000
```

Open `training_dashboard.html` in your browser - it shows:
- 📊 Training stats and reward curves
- 🎬 Simulation snapshots from each evaluation
- Auto-refreshes every 30 seconds

---

## Environment Details

### Observation Space

The environment provides a `Dict` observation:

1. **BEV (Bird's Eye View)**: `Box(0, 1, (128, 128, 8), float32)`
   - Channel 0: Obstacles
   - Channel 1: Objective zone
   - Channel 2: All attackers
   - Channel 3: All defenders
   - Channel 4: Attacker type ID map
   - Channel 5: Defender FOV/detection map
   - Channel 6: Tag cooldown heatmap
   - Channel 7: Capture progress (broadcast)

2. **Vector**: `Box(-inf, inf, (N,), float32)`
   - Per-unit features: x, y, heading, speed, integrity, cooldowns, altitude
   - Global: time remaining, capture progress

### Action Space

`MultiDiscrete` with one discrete action per attacker unit:

**UGV Actions (9):** NOOP, THROTTLE_UP, THROTTLE_DOWN, TURN_LEFT, TURN_RIGHT, BRAKE, HOLD, TAG, SCAN

**UAV Actions (10):** NOOP, THROTTLE_UP, THROTTLE_DOWN, YAW_LEFT, YAW_RIGHT, ALT_UP, ALT_DOWN, HOLD, TAG, SCAN

### Configuration

All parameters are configurable via YAML files in `configs/`:

| File | Description |
|------|-------------|
| `world.yaml` | Arena size, obstacles, physics settings |
| `scenario.yaml` | Unit spawns, objective zone location |
| `units_attackers.yaml` | Attacker unit type definitions |
| `units_defenders.yaml` | Defender unit type definitions |
| `sensors.yaml` | Lidar, radar, camera configurations |
| `engagement.yaml` | Tag beam parameters, cooldowns |
| `reward.yaml` | Reward function weights |

---

## Project Structure

```
comander/
├── pyproject.toml
├── README.md
├── configs/
│   ├── world.yaml
│   ├── scenario.yaml
│   ├── units_attackers.yaml
│   ├── units_defenders.yaml
│   ├── sensors.yaml
│   ├── engagement.yaml
│   └── reward.yaml
└── mission_gym/
    ├── __init__.py
    ├── env.py              # Main Gymnasium environment
    ├── config.py           # Configuration loading
    ├── scenario.py         # Scenario management
    ├── dynamics.py         # Unit physics/kinematics
    ├── sensors.py          # Sensor simulation
    ├── engagement.py       # Tag mechanics
    ├── reward.py           # Reward function
    ├── renderer.py         # Pygame rendering
    ├── defenders.py        # Scripted defender AI
    ├── backends/
    │   ├── base.py         # Abstract backend
    │   ├── simple2p5d.py   # Simple 2.5D physics
    │   ├── isaac_stub.py   # Isaac Sim placeholder
    │   └── mujoco_stub.py  # MuJoCo placeholder
    └── scripts/
        ├── smoke_test.py   # Verification script
        ├── play_manual.py  # Manual keyboard play
        ├── train_ppo.py    # PPO training with monitoring
        ├── evaluate.py     # Model evaluation
        └── monitoring.py   # HTML dashboard callback
```

## License

MIT
