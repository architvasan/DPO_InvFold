# Intel XPU Support

This project now supports Intel XPU (Intel GPU) acceleration through Intel Extension for PyTorch (IPEX).

## Device Priority

The code automatically detects and uses devices in this order:
1. **Intel XPU** (if IPEX is installed and XPU is available)
2. **NVIDIA CUDA** (if available)
3. **CPU** (fallback)

## Setup for Intel XPU

### 1. Install Intel Extension for PyTorch

```bash
# For Intel Data Center GPU Max Series or Intel Arc GPUs
pip install intel-extension-for-pytorch
```

Or follow the official installation guide:
https://intel.github.io/intel-extension-for-pytorch/

### 2. Verify XPU Availability

```python
import torch
try:
    import intel_extension_for_pytorch as ipex
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print(f"XPU available: {torch.xpu.device_count()} device(s)")
        print(f"XPU device name: {torch.xpu.get_device_name(0)}")
    else:
        print("XPU not available")
except ImportError:
    print("Intel Extension for PyTorch not installed")
```

### 3. Run Training on XPU

No changes needed! The code automatically detects XPU:

```bash
python -m src.dpo_inv.run_dpo \
    --train_data data/training_data.json \
    --output_dir outputs/my_model \
    --num_epochs 10
```

You should see:
```
Using device: xpu (Intel XPU)
```

### 4. Run Inference on XPU

```bash
python -m src.dpo_inv.inference \
    --model_path outputs/my_model/best_model.pt \
    --pdb_file your_protein.pdb \
    --output_dir outputs/inference
```

## Force CPU Mode

To disable GPU acceleration and force CPU mode:

```bash
# Training
python -m src.dpo_inv.run_dpo \
    --train_data data/training_data.json \
    --no_cuda

# Inference
python -m src.dpo_inv.inference \
    --model_path outputs/model.pt \
    --pdb_file protein.pdb \
    --no_cuda
```

Note: The `--no_cuda` flag now disables both CUDA and XPU.

## Supported Intel Hardware

### Intel Data Center GPUs
- Intel Data Center GPU Max Series (Ponte Vecchio)
- Intel Data Center GPU Flex Series

### Intel Client GPUs
- Intel Arc A-Series Graphics
- Intel Iris Xe Graphics (with driver support)

### Intel CPUs
- Will fall back to CPU if XPU is not available
- Still benefits from IPEX optimizations on CPU

## Performance Tips

### For XPU Training

1. **Batch Size**: XPU memory may differ from CUDA GPUs
   ```bash
   --batch_size 2  # Adjust based on your XPU memory
   ```

2. **Mixed Precision**: IPEX supports automatic mixed precision
   ```python
   # Future enhancement - not yet implemented
   # model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16)
   ```

3. **Monitor Memory**: Check XPU memory usage
   ```python
   import torch
   if torch.xpu.is_available():
       print(f"XPU Memory allocated: {torch.xpu.memory_allocated(0) / 1e9:.2f} GB")
       print(f"XPU Memory reserved: {torch.xpu.memory_reserved(0) / 1e9:.2f} GB")
   ```

## Troubleshooting

### "Intel Extension for PyTorch not installed"

Install IPEX:
```bash
pip install intel-extension-for-pytorch
```

### "XPU not available" despite having Intel GPU

1. Check driver installation:
   ```bash
   # Linux
   clinfo  # Should show Intel GPU
   
   # Windows
   # Check Device Manager for Intel GPU
   ```

2. Verify IPEX installation:
   ```python
   import intel_extension_for_pytorch as ipex
   print(ipex.__version__)
   ```

3. Check PyTorch version compatibility:
   - IPEX version must match PyTorch version
   - See: https://intel.github.io/intel-extension-for-pytorch/

### Code falls back to CPU unexpectedly

Check the output message:
- `Using device: xpu (Intel XPU)` - XPU is working ✓
- `Using device: cuda` - Using NVIDIA GPU ✓
- `Using device: cpu` - No GPU detected or `--no_cuda` flag used

### Performance is slower than expected

1. Ensure you're using the latest IPEX version
2. Check that XPU drivers are up to date
3. Monitor XPU utilization:
   ```bash
   # Linux
   xpu-smi dump -m 1  # Monitor XPU usage
   ```

## Compatibility Notes

### What Works on XPU
- ✅ All PyTorch tensor operations
- ✅ Model training (forward/backward passes)
- ✅ Inference/sampling
- ✅ Gradient computation
- ✅ Optimizer updates

### Known Limitations
- Some advanced PyTorch features may have limited XPU support
- Check IPEX documentation for specific operation support
- Performance may vary compared to CUDA on different workloads

## Environment Variables

Useful environment variables for XPU:

```bash
# Enable verbose logging
export IPEX_VERBOSE=1

# Set XPU device
export ZE_AFFINITY_MASK=0  # Use first XPU device

# Enable profiling
export IPEX_ENABLE_PROFILING=1
```

## Benchmarking

To compare performance across devices:

```bash
# XPU
python -m src.dpo_inv.run_dpo --train_data data.json --num_epochs 1

# CUDA (if available)
python -m src.dpo_inv.run_dpo --train_data data.json --num_epochs 1

# CPU
python -m src.dpo_inv.run_dpo --train_data data.json --num_epochs 1 --no_cuda
```

Monitor training time and memory usage for each.

## Additional Resources

- **Intel Extension for PyTorch**: https://intel.github.io/intel-extension-for-pytorch/
- **Installation Guide**: https://intel.github.io/intel-extension-for-pytorch/index.html#installation
- **API Documentation**: https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/api_doc.html
- **Performance Tuning**: https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/performance_tuning/tuning_guide.html

## Future Enhancements

Potential optimizations for XPU (not yet implemented):
- Automatic mixed precision (AMP) with bfloat16
- XPU-specific kernel optimizations
- Multi-XPU training support
- XPU memory profiling integration

## Support

If you encounter XPU-specific issues:
1. Check IPEX GitHub issues: https://github.com/intel/intel-extension-for-pytorch/issues
2. Verify your hardware is supported
3. Ensure drivers and IPEX versions are compatible

