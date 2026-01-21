You are building a Python project called mission_gym: a game-like reinforcement learning environment for commanding a fleet to capture an objective zone while defenders try to tag/disable units. This is NOT a real-world combat simulator. Use only non-lethal abstractions: "tag/disable/jam/capture". No content about harming people, weapons, or real tactics.

Goal:
- Implement a Gymnasium environment with centralized single-policy control over multiple units (attackers) via per-unit atomic commands (MultiDiscrete).
- Defenders are scripted first, but share the same atomic action interface so they can be trained later.
- Observations include: (1) global bird’s-eye-view raster image (BEV) and (2) vector features.
- Everything is configurable via multiple YAML files: world, scenario, unit types, sensors, engagement, reward.
- Provide:
  1) manual play script (keyboard + simple UI) to send commands to the fleet
  2) PPO training script using stable-baselines3 MultiInputPolicy (CNN on BEV + MLP on vec)
  3) unit tests or smoke test script to reset/step and validate shapes
  4) clean backend abstraction: Simple2.5D backend implemented now, IsaacBackend and MuJoCoBackend as placeholders

Tech choices:
- Python 3.10+
- Dependencies: gymnasium, numpy, pyyaml, stable-baselines3, torch, pygame (for human rendering), shapely (optional for geometry; if not used, implement basic geometry manually)
- Keep it fast: internal physics at 20 Hz, commands at 4 Hz via action_repeat=5.
- Episode duration configurable (default 5 minutes sim time).

Core mechanics:
- World: 2D continuous arena (e.g., 200x200 meters). Obstacles are circles and rectangles that block motion and line-of-sight.
- Units: 4 attacker types exist, but default scenario spawns 4 attackers (UGV_A, UGV_B, UAV_A, UAV_B). Each unit has kinematic dynamics: position, heading, speed, max accel, max turn rate; UAV has altitude bands (0..2).
- Sensors (configurable per unit): lidar rays (2D distances), radar detections (range/bearing with noise), camera FOV sector (LOS + angle).
- Engagement: Tag beam requires LOS and range, with cooldown. Tag reduces unit integrity; thresholds degrade mobility and sensing; integrity <= 0 disables unit.
- Objective: Capture zone. Capture progresses when any attacker is inside zone; win when progress reaches threshold (e.g., 20 seconds cumulative).
- Reward: specified in reward.yaml with weights: capture_progress, win_bonus, time_penalty, integrity_loss_penalty, collision_penalty, optional detected_time_penalty (toggle).

Observation:
- BEV raster: shape (128,128,8) float32 with channels:
  0 obstacles
  1 objective zone
  2 attackers (all)
  3 defenders (all)
  4 attacker type id m3.2 Launch the control loop in simulation

Inside the container:

python gr00t_wbc/control/main/teleop/run_g1_control_loop.py


Keyboard controls are documented (activate policy, move, rotate, etc.).

Why this matters: your “navigation head” later can just output a (vx, vy, yaw_rate) style command, and the whole-body policy handles balance/contacts. That design is exactly what the blog describes.

4) Synthetic navigation + GR00T post-training with COMPASS (the PointNav part)

This is where the blog’s “COMPASS-generated synthetic datasets” and “fine-tune GR00T for point-to-point navigation” happens.

4.1 Install COMPASS (expects Isaac Lab)

COMPASS’s README gives the core steps (venv + install deps via isaaclab.sh).

High-level flow:

Install Isaac Lab (separate environment, likely Python 3.11 if you use Isaac Sim 5.x).

Create venv, then inside COMPASS:

python3 -m venv venv
source venv/bin/activate

# install requirements via Isaac Lab runner
${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -r requirements.txt


They also provide:

install X-Mobility wheel

download pretrained X-Mobility checkpoint

download environment USDs and place them in the expected folder

4.2 Train a residual RL specialist (or start from theirs)

Train:

${ISAACLAB_PATH}/isaaclab.sh -p run.py \
  -c configs/train_config.gin \
  -o <output_dir> \
  -b <path/to/x_mobility_ckpt> \
  --enable_cameraap (optional simple encoding)
  5 defender FOV/detection map (optional)
  6 tag cooldown heatmap (optional)
  7 capture progress scalar broadcast (optional)
- Vec: concatenated per-unit features: x,y,heading,speed,integrity,cooldowns,altitude + global time remaining + capture_progress.

Actions:
- Per unit discrete action set (size may differ for UGV/UAV but implement as per-unit MultiDiscrete):
  UGV actions: NOOP, THROTTLE_UP, THROTTLE_DOWN, TURN_LEFT, TURN_RIGHT, BRAKE, HOLD, TAG, SCAN
  UAV actions: NOOP, THROTTLE_UP, THROTTLE_DOWN, YAW_LEFT, YAW_RIGHT, ALT_UP, ALT_DOWN, HOLD, TAG, SCAN
- The env maps actions to control targets and applies physics over action_repeat ticks.

Defender scripted policy:
- Simple: patrol around objective using waypoints; if any attacker visible and in range, attempt TAG; else move toward nearest attacker last-seen position.
- Keep last-seen memory for a few seconds to look realistic in-game.

Repo structure to generate:
mission_gym/
  pyproject.toml (or requirements.txt)
  README.md with how to run manual play and training
  mission_gym/
    __init__.py
    env.py
    config.py
    scenario.py
    dynamics.py
    sensors.py
    engagement.py
    reward.py
    renderer.py
    defenders.py
    backends/
      __init__.py
      base.py
      simple2p5d.py
      isaac_stub.py
      mujoco_stub.py
  configs/
    world.yaml
    scenario.yaml
    units_attackers.yaml
    units_defenders.yaml
    sensors.yaml
    engagement.yaml
    reward.yaml
  scripts/
    play_manual.py
    train_ppo.py
    smoke_test.py

Definition of Done (DoD):
- `python -m mission_gym.scripts.smoke_test` runs and prints obs/action shapes, runs 200 steps without errors.
- `python -m mission_gym.scripts.play_manual` opens a window and lets me select a unit and issue discrete commands and see movement + BEV.
- `python -m mission_gym.scripts.train_ppo` starts training (even if reward is initially weak) and logs episodic returns.

Implementation notes:
- Keep deterministic seeding.
- Separate reward into RewardFn that reads reward.yaml.
- Keep geometry robust: line-segment vs circle/rect for LOS checks.
- Ensure the env follows Gymnasium API exactly and returns (obs, reward, terminated, truncated, info).
Generate the full code and configs.
