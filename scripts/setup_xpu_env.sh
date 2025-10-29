#!/bin/bash
# Setup environment variables for Intel XPU training
# Source this file before running training: source scripts/setup_xpu_env.sh

echo "Setting up Intel XPU environment variables..."

# Use immediate command lists for better stability
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
echo "✓ Set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1"

# Disable persistent cache to avoid corruption issues
export SYCL_CACHE_PERSISTENT=0
echo "✓ Set SYCL_CACHE_PERSISTENT=0"

# Use single device to avoid multi-GPU issues
export ZE_AFFINITY_MASK=0
echo "✓ Set ZE_AFFINITY_MASK=0 (using first XPU only)"

# Filter to use Level Zero backend
export SYCL_DEVICE_FILTER=level_zero
echo "✓ Set SYCL_DEVICE_FILTER=level_zero"

# Disable JIT cache directory (can cause issues)
export SYCL_CACHE_DIR=/tmp/sycl_cache_$$
mkdir -p $SYCL_CACHE_DIR
echo "✓ Set SYCL_CACHE_DIR=$SYCL_CACHE_DIR"

# Enable verbose error messages (optional, comment out if too noisy)
# export SYCL_PI_TRACE=1
# echo "✓ Set SYCL_PI_TRACE=1 (verbose logging)"

# Set memory allocation strategy
export SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=1
echo "✓ Set SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=1"

# Optimize for large allocations
export SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=1
echo "✓ Set SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=1"

echo ""
echo "XPU environment configured!"
echo ""
echo "Now run your training command:"
echo "  python -m dpo_inv.run_dpo --train_data data.json --output_dir outputs/ --batch_size 1"
echo ""
echo "If you still get segfaults, try:"
echo "  1. Reduce batch size to 1"
echo "  2. Check IPEX version: python -c 'import intel_extension_for_pytorch as ipex; print(ipex.__version__)'"
echo "  3. Update IPEX: pip install --upgrade intel-extension-for-pytorch"
echo "  4. See XPU_TROUBLESHOOTING.md for more options"
echo ""

