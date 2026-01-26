#!/bin/bash
# Resume current training with fixed parameters
# (Less eval spam, proper batch scaling, clean Rich output)

LATEST_RUN="runs/warm-panther-branch-20260126-200133"
LATEST_CHECKPOINT="$LATEST_RUN/checkpoints/ppo_mission_4682496_steps"

echo "🔄 Resuming training from checkpoint..."
echo "   Run: $LATEST_RUN"
echo "   Checkpoint: $LATEST_CHECKPOINT"
echo ""
echo "✨ New fixes applied:"
echo "   • Eval every 20K steps (was 5K) - less spam"
echo "   • Proper batch scaling with --n-steps-per-env"
echo "   • Fixed Rich console conflicts"
echo "   • Mean reward in metrics table"
echo ""
echo "Press Ctrl+C in the training terminal to stop current training, then run this."
echo ""

python -m mission_gym.scripts.train_ppo \
  --timesteps 50000000 \
  --n-envs 48 \
  --n-steps-per-env 128 \
  --subproc \
  --network-arch "1024,512,512,256" \
  --n-epochs 10 \
  --load-checkpoint "$LATEST_CHECKPOINT" \
  --branch-name "continued" \
  --notes "Resumed with fixes: proper batch scaling, less eval spam, clean output"
