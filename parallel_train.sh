#!/bin/bash
# Parallel GPU Training Script
# Runs multiple training jobs simultaneously to maximize GPU utilization

set -e

# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Base configuration
TIMESTEPS=50000000
BASE_CHECKPOINT="runs/2-defenders-gpu-beast-20260125-234819/checkpoints/ppo_mission_34724352_steps"

echo "🚀 Starting 4 parallel training runs to saturate GPU..."
echo "📊 Each run will use ~200MB VRAM (total ~800MB / 8GB)"
echo ""

# Job 1: Large batch (maximize data per update)
echo "▶️  Job 1: Large batch (32×256 = 8192)..."
python -m mission_gym.scripts.train_ppo \
    --timesteps $TIMESTEPS \
    --n-envs 32 \
    --n-steps-per-env 256 \
    --subproc \
    --network-arch "1024,512,256" \
    --n-epochs 20 \
    --load-checkpoint $BASE_CHECKPOINT \
    --branch-name "large-batch" \
    --notes "8192 rollout buffer, 20 epochs" \
    > logs/parallel_train_job1.log 2>&1 &

JOB1=$!
sleep 5

# Job 2: Wide network + medium batch
echo "▶️  Job 2: Wide network (1536,768,384)..."
python -m mission_gym.scripts.train_ppo \
    --timesteps $TIMESTEPS \
    --n-envs 24 \
    --n-steps-per-env 256 \
    --subproc \
    --network-arch "1536,768,384" \
    --n-epochs 25 \
    --load-checkpoint $BASE_CHECKPOINT \
    --branch-name "wide-net" \
    --notes "Wide 3-layer network, 6144 buffer, 25 epochs" \
    > logs/parallel_train_job2.log 2>&1 &

JOB2=$!
sleep 5

# Job 3: Deep network + high epochs
echo "▶️  Job 3: Deep network (512x5)..."
python -m mission_gym.scripts.train_ppo \
    --timesteps $TIMESTEPS \
    --n-envs 24 \
    --n-steps-per-env 256 \
    --subproc \
    --network-arch "512,512,512,512,512" \
    --n-epochs 30 \
    --load-checkpoint $BASE_CHECKPOINT \
    --branch-name "deep-net" \
    --notes "5-layer deep network, 6144 buffer, 30 epochs" \
    > logs/parallel_train_job3.log 2>&1 &

JOB3=$!
sleep 5

# Job 4: Huge batch (test scalability)
echo "▶️  Job 4: Huge batch (32×512 = 16384)..."
python -m mission_gym.scripts.train_ppo \
    --timesteps $TIMESTEPS \
    --n-envs 32 \
    --n-steps-per-env 512 \
    --subproc \
    --network-arch "1024,512,512,256" \
    --n-epochs 15 \
    --load-checkpoint $BASE_CHECKPOINT \
    --branch-name "huge-batch" \
    --notes "16384 rollout buffer for max GPU utilization, 15 epochs" \
    > logs/parallel_train_job4.log 2>&1 &

JOB4=$!

echo ""
echo "✅ All 4 jobs started!"
echo ""
echo "📊 Monitor GPU usage:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "📝 View logs:"
echo "   tail -f logs/parallel_train_job1.log"
echo "   tail -f logs/parallel_train_job2.log"
echo "   tail -f logs/parallel_train_job3.log"
echo "   tail -f logs/parallel_train_job4.log"
echo ""
echo "🔍 Check process status:"
echo "   ps aux | grep train_ppo"
echo ""
echo "⏹️  Stop all jobs:"
echo "   kill $JOB1 $JOB2 $JOB3 $JOB4"
echo ""

# Wait for all jobs to complete
echo "⏳ Waiting for all jobs to complete..."
wait $JOB1 $JOB2 $JOB3 $JOB4

echo "✨ All training jobs completed!"
