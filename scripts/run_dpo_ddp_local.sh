#!/bin/bash
# Local multi-GPU training script using torchrun
# This script is for running DDP on a single node with multiple GPUs

# Usage:
#   ./scripts/run_dpo_ddp_local.sh [num_gpus]
# Example:
#   ./scripts/run_dpo_ddp_local.sh 4

# Number of GPUs (default: all available)
NGPUS=12 #${1:-$(python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 1)")}

echo "=========================================="
echo "Local DDP Training"
echo "=========================================="
echo "Number of GPUs: $NGPUS"
echo "=========================================="

# Training parameters
TRAIN_DATA="data/input/test_data_real.json"
OUTPUT_DIR="outputs/dpo_ddp_local_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=1  # Per GPU batch size
NUM_EPOCHS=10
LEARNING_RATE=3e-7
BETA=0.1
TEMPERATURE=0.1

# Run with torchrun
torchrun \
    --nproc_per_node=${NGPUS} \
    --master_port=29500 \
    -m dpo_inv.run_dpo_ddp \
        --train_data ${TRAIN_DATA} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size ${BATCH_SIZE} \
        --num_epochs ${NUM_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --beta ${BETA} \
        --temperature ${TEMPERATURE}

echo "Training complete!"
echo "Results saved to: ${OUTPUT_DIR}"

