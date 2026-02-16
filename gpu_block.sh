#!/bin/bash
# GPU Blocking Script for Safe Cluster Usage
# Usage: source gpu_block.sh [gpu1] [gpu2] ...

# Default to GPUs 0 and 6
GPUS=${@:-0 6}

echo "🔒 Blocking GPUs - Only allowing access to: $GPUS"

# Convert space-separated list to comma-separated
CUDA_VISIBLE_DEVICES=$(echo $GPUS | tr ' ' ',')

# Set environment variable
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

echo "✅ Set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "🔒 Only GPUs $GPUS are now visible to PyTorch"
echo ""
echo "💡 To unblock, run: unset CUDA_VISIBLE_DEVICES"
echo "💡 To check current status: echo \$CUDA_VISIBLE_DEVICES"
echo "💡 To monitor GPU usage: nvidia-smi -i $GPUS"
echo ""
echo "🚀 Now you can run your Python scripts safely!"
echo "   They will only see GPUs $GPUS and won't interfere with other users."