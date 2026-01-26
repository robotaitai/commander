#!/bin/bash
# Add 3 more training jobs to run in parallel with current job
# This will saturate the GPU (4 total jobs x 200MB = ~800MB / 8GB)

set -e

export CUDA_VISIBLE_DEVICES=0

TIMESTEPS=50000000
BASE_CHECKPOINT="runs/2-defenders-gpu-beast-20260125-234819/checkpoints/ppo_mission_34724352_steps"

mkdir -p logs

echo "🚀 Adding 3 parallel training jobs (you already have 1 running)..."
echo "📊 Target: 4 jobs x 200MB = ~800-1000MB VRAM utilization"
echo ""

# Job 2: Larger batch size (more data per update)
echo "▶️  Job 2: Large batch (32 envs × 256 steps = 8192 buffer)..."
nohup python -m mission_gym.scripts.train_ppo \
    --timesteps $TIMESTEPS \
    --n-envs 32 \
    --n-steps-per-env 256 \
    --subproc \
    --network-arch "1024,512,256" \
    --n-epochs 20 \
    --load-checkpoint $BASE_CHECKPOINT \
    --branch-name "large-batch" \
    --notes "8192 buffer size, 20 epochs" \
    > logs/train_large_batch.log 2>&1 &

echo "   PID: $! - Log: logs/train_large_batch.log"
sleep 10

# Job 3: Wide network with medium batch
echo "▶️  Job 3: Wide network (1536,768,384) + medium batch..."
nohup python -m mission_gym.scripts.train_ppo \
    --timesteps $TIMESTEPS \
    --n-envs 24 \
    --n-steps-per-env 256 \
    --subproc \
    --network-arch "1536,768,384" \
    --n-epochs 25 \
    --load-checkpoint $BASE_CHECKPOINT \
    --branch-name "wide-net" \
    --notes "Wide 3-layer network, 6144 buffer, 25 epochs" \
    > logs/train_wide.log 2>&1 &

echo "   PID: $! - Log: logs/train_wide.log"
sleep 10

# Job 4: Deep network with high epochs
echo "▶️  Job 4: Deep network (512x5) + high epochs..."
nohup python -m mission_gym.scripts.train_ppo \
    --timesteps $TIMESTEPS \
    --n-envs 24 \
    --n-steps-per-env 256 \
    --subproc \
    --network-arch "512,512,512,512,512" \
    --n-epochs 30 \
    --load-checkpoint $BASE_CHECKPOINT \
    --branch-name "deep-net" \
    --notes "5-layer deep network, 6144 buffer, 30 epochs" \
    > logs/train_deep.log 2>&1 &

echo "   PID: $! - Log: logs/train_deep.log"

echo ""
echo "✅ 3 additional jobs started! (4 total now)"
echo ""
echo "📊 Check GPU utilization (should be 20-40% now):"
echo "   nvidia-smi"
echo ""
echo "📝 View logs:"
echo "   tail -f logs/train_large_batch.log"
echo "   tail -f logs/train_wide.log"
echo "   tail -f logs/train_deep.log"
echo ""
echo "🔍 Monitor all jobs:"
echo "   watch -n 2 './monitor_parallel.sh'"
echo ""
echo "⏹️  Stop all new jobs:"
echo "   pkill -f 'large-batch|wide-net|deep-net'"
echo ""
