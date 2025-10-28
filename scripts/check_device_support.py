#!/usr/bin/env python
"""
Check available compute devices (CPU, CUDA, XPU) for DPO training.
"""

import sys

def check_pytorch():
    """Check PyTorch installation."""
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
        return torch
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Install: pip install torch")
        return None


def check_cuda(torch):
    """Check CUDA availability."""
    if torch is None:
        return False
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("✗ CUDA not available")
        return False


def check_xpu(torch):
    """Check Intel XPU availability."""
    if torch is None:
        return False
    
    try:
        import intel_extension_for_pytorch as ipex
        print(f"✓ Intel Extension for PyTorch installed: {ipex.__version__}")
        
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            print(f"✓ Intel XPU available")
            print(f"  Number of XPU devices: {torch.xpu.device_count()}")
            for i in range(torch.xpu.device_count()):
                print(f"  XPU {i}: {torch.xpu.get_device_name(i)}")
            return True
        else:
            print("✗ Intel XPU not available (no compatible hardware detected)")
            return False
    except ImportError:
        print("✗ Intel Extension for PyTorch not installed")
        print("  Install: pip install intel-extension-for-pytorch")
        print("  See: https://intel.github.io/intel-extension-for-pytorch/")
        return False


def test_device_selection():
    """Test the device selection logic used in the code."""
    import torch
    
    print("\n" + "="*60)
    print("Testing Device Selection Logic")
    print("="*60)
    
    # Simulate the device selection from run_dpo.py
    no_cuda = False  # Simulate --no_cuda flag
    
    if not no_cuda:
        try:
            import intel_extension_for_pytorch as ipex
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                device = torch.device('xpu')
                print(f"Selected device: {device} (Intel XPU)")
            elif torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"Selected device: {device} (NVIDIA CUDA)")
            else:
                device = torch.device('cpu')
                print(f"Selected device: {device} (CPU)")
        except ImportError:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"Selected device: {device} (NVIDIA CUDA)")
            else:
                device = torch.device('cpu')
                print(f"Selected device: {device} (CPU)")
    else:
        device = torch.device('cpu')
        print(f"Selected device: {device} (CPU - forced)")
    
    # Test tensor creation on selected device
    try:
        test_tensor = torch.randn(10, 10, device=device)
        print(f"✓ Successfully created tensor on {device}")
        print(f"  Tensor shape: {test_tensor.shape}")
        print(f"  Tensor device: {test_tensor.device}")
    except Exception as e:
        print(f"✗ Failed to create tensor on {device}")
        print(f"  Error: {e}")


def check_dependencies():
    """Check other required dependencies."""
    print("\n" + "="*60)
    print("Checking Other Dependencies")
    print("="*60)
    
    deps = {
        'numpy': 'numpy',
        'biopython': 'Bio',
        'pyyaml': 'yaml',
        'tqdm': 'tqdm',
    }
    
    for name, module in deps.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name} not installed")
            print(f"  Install: pip install {name}")


def print_recommendations():
    """Print recommendations based on available hardware."""
    print("\n" + "="*60)
    print("Recommendations")
    print("="*60)
    
    import torch
    
    has_xpu = False
    has_cuda = False
    
    try:
        import intel_extension_for_pytorch as ipex
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            has_xpu = True
    except ImportError:
        pass
    
    if torch.cuda.is_available():
        has_cuda = True
    
    if has_xpu:
        print("✓ Intel XPU detected - optimal for training!")
        print("  Your system will automatically use XPU acceleration.")
        print("  No additional flags needed.")
    elif has_cuda:
        print("✓ NVIDIA CUDA detected - optimal for training!")
        print("  Your system will automatically use CUDA acceleration.")
        print("  No additional flags needed.")
    else:
        print("⚠ No GPU detected - will use CPU")
        print("  Training will be slower on CPU.")
        print("  Consider:")
        print("  - Using a smaller batch size (--batch_size 1)")
        print("  - Using smaller proteins (--max_length 200)")
        print("  - Training for fewer epochs")
        print("  - Using a machine with GPU acceleration")


def main():
    """Main function."""
    print("="*60)
    print("DPO InvFold - Device Support Check")
    print("="*60)
    print()
    
    # Check PyTorch
    torch = check_pytorch()
    if torch is None:
        print("\n✗ PyTorch is required. Please install it first.")
        sys.exit(1)
    
    print()
    
    # Check CUDA
    has_cuda = check_cuda(torch)
    print()
    
    # Check XPU
    has_xpu = check_xpu(torch)
    print()
    
    # Test device selection
    test_device_selection()
    
    # Check other dependencies
    check_dependencies()
    
    # Print recommendations
    print_recommendations()
    
    print("\n" + "="*60)
    print("Check Complete!")
    print("="*60)
    
    if has_xpu or has_cuda:
        print("\n✓ Your system is ready for GPU-accelerated training!")
    else:
        print("\n⚠ Your system will use CPU for training.")


if __name__ == "__main__":
    main()

