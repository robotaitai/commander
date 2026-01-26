#!/bin/bash
# Monitor parallel training jobs

echo "🎯 Parallel Training Monitor"
echo "=============================="
echo ""

# GPU stats
echo "📊 GPU Utilization:"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F',' '{printf "   GPU: %s%%  |  VRAM: %s%% (%sMB / %sMB)\n", $1, $2, $3, $4}'
echo ""

# Running processes
echo "🏃 Active Training Processes:"
ps aux | grep "[p]ython -m mission_gym.scripts.train_ppo" | \
    awk '{printf "   PID %s: CPU %s%% | RAM %s%% | Runtime %s\n", $2, $3, $4, $10}'
echo ""

# Log file status
echo "📝 Latest Log Lines:"
for log in logs/parallel_train_job*.log; do
    if [ -f "$log" ]; then
        echo "   $(basename $log):"
        tail -3 "$log" | grep -E "(it/s|Episode|FPS)" | head -1 | sed 's/^/     /'
    fi
done
echo ""

echo "💡 Tip: Run 'watch -n 2 ./monitor_parallel.sh' for live monitoring"
