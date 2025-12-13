#!/bin/bash
#PBS -l select=2:system=aurora,place=scatter
#PBS -l walltime=02:00:00
#PBS -q workq
#PBS -A <your_project_name>
#PBS -N dpo_training
#PBS -o logs/dpo_ddp_${PBS_JOBID}.out
#PBS -e logs/dpo_ddp_${PBS_JOBID}.err

# Aurora SLURM submission script for DDP training
# This script runs DPO training across multiple nodes with Intel XPU

# Load modules
module use /soft/modulefiles
module load frameworks/2024.2.1_u1

# Set up environment
cd ${PBS_O_WORKDIR}
source scripts/setup_xpu_env.sh

# Create logs directory
mkdir -p logs

# Set distributed training environment variables
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
export MASTER_PORT=29500

# Get number of nodes and GPUs per node
NNODES=$(wc -l < $PBS_NODEFILE)
NPROC_PER_NODE=6  # Aurora has 6 XPUs per node

# Total number of processes
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

echo "=========================================="
echo "DDP Training Configuration"
echo "=========================================="
echo "Number of nodes: $NNODES"
echo "GPUs per node: $NPROC_PER_NODE"
echo "Total processes: $WORLD_SIZE"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "=========================================="

# Training parameters
TRAIN_DATA="data/input/test_data_real.json"
OUTPUT_DIR="outputs/dpo_ddp_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=1  # Per GPU batch size
NUM_EPOCHS=10
LEARNING_RATE=3e-7
BETA=0.1
TEMPERATURE=0.1

# Run DDP training with mpiexec
mpiexec -n ${WORLD_SIZE} \
    -ppn ${NPROC_PER_NODE} \
    --hostfile ${PBS_NODEFILE} \
    python -m dpo_inv.run_dpo_ddp \
        --train_data ${TRAIN_DATA} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size ${BATCH_SIZE} \
        --num_epochs ${NUM_EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --beta ${BETA} \
        --temperature ${TEMPERATURE} \
        --backend ccl

echo "Training complete!"
echo "Results saved to: ${OUTPUT_DIR}"

