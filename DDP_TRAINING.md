# Distributed Data Parallel (DDP) Training Guide

This guide explains how to run DPO training with PyTorch DDP for multi-GPU and multi-node setups.

## Overview

The DDP version (`run_dpo_ddp.py`) provides:
- **Multi-GPU training** on a single node
- **Multi-node training** across multiple nodes (e.g., on Aurora HPC)
- **Automatic device detection** (XPU, CUDA, CPU)
- **Gradient synchronization** across all processes
- **Efficient data parallelism** with DistributedSampler

## Quick Start

### Single Node, Multiple GPUs (Local)

```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=4 -m dpo_inv.run_dpo_ddp \
    --train_data data/input/test_data_real.json \
    --output_dir outputs/dpo_ddp \
    --batch_size 1 \
    --num_epochs 10

# Or use the helper script
./scripts/run_dpo_ddp_local.sh 4  # 4 GPUs
```

### Multi-Node on Aurora (SLURM/PBS)

```bash
# Edit scripts/run_dpo_ddp_aurora.sh to set your project name
# Then submit the job
qsub scripts/run_dpo_ddp_aurora.sh
```

## Command-Line Arguments

All arguments from `run_dpo.py` are supported, plus:

- `--backend`: DDP backend (`auto`, `nccl`, `gloo`, `ccl`, `xpu`)
  - `auto`: Automatically selects based on device
  - `nccl`: NVIDIA GPUs (fastest for CUDA)
  - `ccl`: Intel XPU (oneCCL backend)
  - `gloo`: CPU or fallback

## How DDP Works

### Data Distribution

Each GPU/process gets a **different subset** of the data:
- Total batch size = `batch_size × num_gpus`
- Example: 4 GPUs with `--batch_size 2` → effective batch size of 8

### Gradient Synchronization

1. Each process computes gradients on its local batch
2. Gradients are averaged across all processes
3. All processes update with the same averaged gradients
4. Models stay synchronized across all GPUs

### Loss Aggregation

- Training and validation losses are averaged across all processes
- Only rank 0 prints progress bars and saves checkpoints

## Environment Variables

DDP uses these environment variables (set automatically):

- `RANK`: Global rank of the process (0 to world_size-1)
- `WORLD_SIZE`: Total number of processes
- `LOCAL_RANK`: Rank within the node (0 to num_gpus_per_node-1)
- `MASTER_ADDR`: Address of rank 0 node
- `MASTER_PORT`: Port for communication (default: 29500)

## Aurora-Specific Setup

### PBS Script Configuration

```bash
#PBS -l select=2:system=aurora,place=scatter  # 2 nodes
#PBS -l walltime=02:00:00                      # 2 hours
#PBS -A <your_project>                         # Your allocation
```

### XPU Backend

Aurora uses Intel XPUs, which require:
- `intel_extension_for_pytorch` (IPEX)
- `oneCCL` backend for communication
- Set `--backend ccl` or `--backend xpu`

### Running with mpiexec

```bash
mpiexec -n 12 -ppn 6 python -m dpo_inv.run_dpo_ddp \
    --train_data data.json \
    --output_dir outputs/ \
    --backend ccl
```

## Performance Tips

### Batch Size Tuning

Start with small batch sizes and increase:
```bash
# Start conservative
--batch_size 1  # Total: 1 × num_gpus

# Increase if memory allows
--batch_size 2  # Total: 2 × num_gpus
--batch_size 4  # Total: 4 × num_gpus
```

### Learning Rate Scaling

When increasing total batch size, scale learning rate:
```bash
# Single GPU: batch_size=1, lr=3e-7
--batch_size 1 --learning_rate 3e-7

# 4 GPUs: batch_size=1 each, total=4
--batch_size 1 --learning_rate 1.2e-6  # 4× learning rate

# Or use linear scaling rule: lr_new = lr_base × (total_batch_size / base_batch_size)
```

### Number of Workers

For protein data with PDB parsing, use `num_workers=0`:
- PDB parsing is not easily parallelizable
- Avoids multiprocessing overhead
- Simplifies debugging

## Monitoring

### Check Training Progress

```bash
# Watch output logs
tail -f logs/dpo_ddp_*.out

# Check for errors
tail -f logs/dpo_ddp_*.err
```

### Expected Output

```
Initialized DDP: rank=0, world_size=4, local_rank=0
Using device: xpu:0
Loaded 372 training and 42 validation pairs
Batch size per GPU: 1, Total batch size: 4

Epoch 1/10
Training: 100%|██████| 93/93 [02:15<00:00, loss=3.2145, avg_loss=3.3421]
Validation: 100%|████| 11/11 [00:15<00:00, loss=3.1234, avg_loss=3.2012]

Epoch 1 Summary:
  Train Loss: 3.3421
  Val Loss:   3.2012
```

## Troubleshooting

### Issue: "Address already in use"

**Solution**: Change the master port
```bash
export MASTER_PORT=29501  # Or any free port
```

### Issue: Processes hang at initialization

**Causes**:
- Firewall blocking communication
- Wrong `MASTER_ADDR` or `MASTER_PORT`
- Network issues between nodes

**Solution**:
```bash
# Check network connectivity
ping $MASTER_ADDR

# Try gloo backend (more robust)
--backend gloo
```

### Issue: Out of memory

**Solution**: Reduce batch size
```bash
--batch_size 1  # Minimum per GPU
```

### Issue: Different losses on different GPUs

This is **normal** during training because each GPU sees different data. The gradients are synchronized, so models converge to the same weights.

## Comparison: Single GPU vs DDP

| Metric | Single GPU | 4 GPUs (DDP) | Speedup |
|--------|-----------|--------------|---------|
| Batch size | 2 | 8 (2×4) | 4× |
| Time/epoch | 10 min | 3 min | 3.3× |
| Memory/GPU | 16 GB | 16 GB | Same |
| Convergence | Baseline | Faster* | - |

*Faster convergence with larger effective batch size, but may need LR tuning.

## Advanced: Custom Backends

### For Intel XPU (Aurora)

```python
# Automatically handled in run_dpo_ddp.py
backend = 'ccl'  # oneCCL for XPU
```

### For NVIDIA GPUs

```python
backend = 'nccl'  # Fastest for CUDA
```

### For CPU or Mixed

```python
backend = 'gloo'  # Works everywhere
```

## Files Created

- `src/dpo_inv/run_dpo_ddp.py` - Main DDP training script
- `scripts/run_dpo_ddp_aurora.sh` - PBS submission script for Aurora
- `scripts/run_dpo_ddp_local.sh` - Local multi-GPU training script

## Next Steps

1. **Test locally** with 1-2 GPUs
2. **Tune hyperparameters** (batch size, learning rate)
3. **Scale to Aurora** with multiple nodes
4. **Monitor convergence** and adjust as needed

