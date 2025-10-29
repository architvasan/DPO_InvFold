# Intel XPU Troubleshooting Guide

## Segmentation Fault on XPU

If you're getting a segmentation fault like:
```
Segmentation fault from GPU at 0xff00fffffec03000, ctx_id: 1 (CCS) type: 0 (NotPresent), level: 0 (PTE), access: 0 (Read), banned: 1, aborting.
```

This is a GPU memory access error on Intel XPU. Here are solutions:

---

## Solution 1: Force CPU Mode (Recommended for Now)

The safest option is to run on CPU while we debug the XPU issue:

```bash
python -m dpo_inv.run_dpo \
    --train_data data/input/test_data_real.json \
    --output_dir outputs/my_model \
    --no_cuda \
    --batch_size 1
```

The `--no_cuda` flag forces CPU mode and disables all GPU acceleration.

**Pros:**
- ✅ Will definitely work
- ✅ No segfaults
- ✅ Stable training

**Cons:**
- ❌ Slower than GPU
- ❌ May need smaller batch size

---

## Solution 2: Reduce Batch Size

XPU memory issues often occur with large batches. Try batch size 1:

```bash
python -m dpo_inv.run_dpo \
    --train_data data/input/test_data_real.json \
    --output_dir outputs/my_model \
    --batch_size 1
```

---

## Solution 3: Set XPU Environment Variables

Intel XPU sometimes needs specific environment variables:

```bash
# Set XPU memory allocation mode
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_PI_LEVEL_ZERO_BATCH_SIZE=1

# Disable JIT cache (can cause issues)
export SYCL_CACHE_PERSISTENT=0

# Enable verbose logging
export SYCL_PI_TRACE=1

# Run training
python -m dpo_inv.run_dpo \
    --train_data data/input/test_data_real.json \
    --output_dir outputs/my_model \
    --batch_size 1
```

---

## Solution 4: Update Intel Extension for PyTorch

Make sure you have the latest IPEX:

```bash
# Check current version
python -c "import intel_extension_for_pytorch as ipex; print(ipex.__version__)"

# Update to latest
pip install --upgrade intel-extension-for-pytorch

# Or install specific version for Aurora
pip install intel-extension-for-pytorch==2.1.0+xpu
```

---

## Solution 5: Use Mixed CPU/XPU Mode

The code now includes a workaround that featurizes on CPU then moves to XPU. This should help, but if it still fails, you can modify the code to keep more operations on CPU.

---

## Solution 6: Check Aurora-Specific Settings

Since you're on Aurora (based on the path), you may need Aurora-specific settings:

```bash
# Load Aurora modules
module load frameworks/2025.0.0

# Set Aurora-specific environment
export ZE_AFFINITY_MASK=0
export SYCL_DEVICE_FILTER=level_zero

# Run with single GPU
python -m dpo_inv.run_dpo \
    --train_data data/input/test_data_real.json \
    --output_dir outputs/my_model \
    --batch_size 1
```

---

## Solution 7: Debug Mode

Run with Python debugging to see exactly where it crashes:

```bash
python -u -m dpo_inv.run_dpo \
    --train_data data/input/test_data_real.json \
    --output_dir outputs/my_model \
    --batch_size 1 2>&1 | tee training.log
```

The `-u` flag unbuffers output so you see exactly where it crashes.

---

## Recommended Workflow

### For Immediate Results (Use CPU):

```bash
python -m dpo_inv.run_dpo \
    --train_data data/input/test_data_real.json \
    --output_dir outputs/cpu_model \
    --no_cuda \
    --batch_size 2 \
    --num_epochs 10
```

### For Debugging XPU:

```bash
# Try with environment variables
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_CACHE_PERSISTENT=0

python -m dpo_inv.run_dpo \
    --train_data data/input/test_data_real.json \
    --output_dir outputs/xpu_test \
    --batch_size 1 \
    --num_epochs 1
```

### For Production (Once XPU Works):

```bash
python -m dpo_inv.run_dpo \
    --train_data data/input/test_data_real.json \
    --output_dir outputs/production \
    --batch_size 4 \
    --num_epochs 20
```

---

## Common XPU Issues and Fixes

### Issue: "NotPresent" Page Fault

**Cause**: Trying to access GPU memory that wasn't allocated

**Fix**: 
- Reduce batch size to 1
- Use `--no_cuda` to force CPU
- Set `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`

### Issue: Out of Memory

**Cause**: XPU ran out of memory

**Fix**:
- Reduce `--batch_size` to 1
- Reduce `--max_length` to filter out long proteins
- Use CPU mode

### Issue: Slow Training on CPU

**Cause**: CPU is slower than GPU

**Fix**:
- This is expected - CPU is 10-100x slower
- Reduce `--num_epochs` for testing
- Use smaller dataset for testing
- Once XPU works, switch back to GPU

---

## Performance Comparison

Approximate training times for 1 epoch with 372 samples:

| Device | Batch Size | Time per Epoch |
|--------|------------|----------------|
| CPU    | 1          | ~30-60 min     |
| CPU    | 2          | ~20-40 min     |
| XPU    | 1          | ~5-10 min      |
| XPU    | 4          | ~2-5 min       |
| CUDA   | 4          | ~1-3 min       |

*Times are approximate and depend on protein sizes*

---

## Checking What Went Wrong

After a crash, check:

1. **Last successful operation**:
   ```bash
   tail -50 training.log
   ```

2. **XPU status**:
   ```bash
   python scripts/check_device_support.py
   ```

3. **Memory usage**:
   ```bash
   # During training, in another terminal:
   watch -n 1 'xpu-smi dump -m'
   ```

---

## Contact Support

If none of these work:

1. **Save the error log**:
   ```bash
   python -m dpo_inv.run_dpo ... 2>&1 | tee error.log
   ```

2. **Check IPEX version**:
   ```bash
   python -c "import intel_extension_for_pytorch as ipex; print(ipex.__version__)"
   python -c "import torch; print(torch.__version__)"
   ```

3. **Report the issue** with:
   - Error log
   - IPEX version
   - PyTorch version
   - System info (Aurora, etc.)

---

## Quick Reference

```bash
# CPU mode (safest)
--no_cuda

# Reduce memory
--batch_size 1
--max_length 200

# XPU environment
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_CACHE_PERSISTENT=0

# Debug
python -u -m dpo_inv.run_dpo ... 2>&1 | tee log.txt
```

---

## My Recommendation

**For now, use CPU mode** to get your training working:

```bash
python -m dpo_inv.run_dpo \
    --train_data data/input/test_data_real.json \
    --output_dir outputs/my_model \
    --no_cuda \
    --batch_size 2 \
    --num_epochs 10 \
    --learning_rate 3e-7
```

This will be slower but stable. Once training works on CPU, we can debug the XPU issue separately without blocking your progress.

The XPU segfault is likely a compatibility issue between:
- Intel Extension for PyTorch version
- PyTorch version  
- Aurora's Level Zero drivers
- The specific operations in `tied_featurize()`

We can debug this in parallel while you get results on CPU.

