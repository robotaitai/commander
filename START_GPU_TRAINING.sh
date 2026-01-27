#!/bin/bash
# Start GPU-optimized training with all fixes applied
# Uses "heavy" preset for optimal GPU utilization (~50%)

set -e

LATEST_CHECKPOINT="runs/warm-panther-clean-output-20260126-221549/checkpoints/ppo_mission_9454848_steps"

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                   🚀 GPU-HEAVY TRAINING - ALL FIXES APPLIED                  ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Configuration: HEAVY PRESET"
echo "   • n_envs: 32 (parallel environments)"
echo "   • n_steps: 256 (per environment)"
echo "   • Total batch: 8192 transitions/update"
echo "   • Batch size: 4096 (2 minibatches)"
echo "   • Epochs: 20 (GPU work per rollout)"
echo "   • Network: 1024 → 512 → 256 (large MLP)"
echo ""
echo "🔧 Optimizations Enabled:"
echo "   ✅ TF32 matmul (2x faster on RTX 4070)"
echo "   ✅ CPU thread limiting (no thread storms)"
echo "   ✅ SubprocVecEnv with forkserver (true parallelism)"
echo "   ✅ Large batch sizes (4096 for GPU efficiency)"
echo "   ✅ Stagnation fixes (capture progress resets timer)"
echo "   ✅ Clean output (tables every 500 eps, no eval spam)"
echo ""
echo "📈 Expected Results:"
echo "   • GPU utilization: ~50% (bursts to 80%)"
echo "   • Training FPS: ~700 it/s"
echo "   • Win rate: Higher (no premature stalls)"
echo "   • Training stability: Much better (larger batches)"
echo ""
echo "🎯 Starting from checkpoint:"
echo "   $LATEST_CHECKPOINT"
echo "   (9.45M steps completed)"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to start..."
sleep 5

python -m mission_gym.scripts.train_ppo \
  --preset heavy \
  --timesteps 50000000 \
  --load-checkpoint "$LATEST_CHECKPOINT" \
  --branch-name "gpu-heavy" \
  --notes "GPU-optimized: TF32, large batches, stagnation fixes, clean output" \
  --seed 42

echo ""
echo "✅ Training complete!"
