#!/usr/bin/env python
"""
Test script to verify DPO InvFold installation.
"""

import sys


def test_python_version():
    """Check Python version."""
    print("="*60)
    print("Checking Python Version")
    print("="*60)
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version is compatible (3.8+)")
        return True
    else:
        print("✗ Python version is too old. Need 3.8+")
        return False


def test_imports():
    """Test that all required packages can be imported."""
    print("\n" + "="*60)
    print("Checking Package Imports")
    print("="*60)
    
    packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'Bio': 'Biopython',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
    }
    
    all_ok = True
    for module, name in packages.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name} not installed")
            print(f"  Install: pip install {module if module != 'Bio' else 'biopython'}")
            all_ok = False
    
    return all_ok


def test_dpo_inv_import():
    """Test that dpo_inv package can be imported."""
    print("\n" + "="*60)
    print("Checking DPO InvFold Package")
    print("="*60)
    
    try:
        import dpo_inv
        print("✓ dpo_inv package found")
        
        # Try importing submodules
        try:
            from dpo_inv.model import BioMPNN
            print("✓ dpo_inv.model.BioMPNN imported")
        except Exception as e:
            print(f"⚠ Could not import BioMPNN (may need MPNN libs): {e}")
        
        try:
            from dpo_inv.data import DD
            print("✓ dpo_inv.data.DD imported")
        except Exception as e:
            print(f"✗ Could not import dpo_inv.data.DD: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"✗ dpo_inv package not found: {e}")
        print("\nTo fix:")
        print("  1. Make sure you're in the DPO_InvFold directory")
        print("  2. Run: pip install -e .")
        print("  3. OR add to PYTHONPATH: export PYTHONPATH=\"${PYTHONPATH}:$(pwd)/src\"")
        return False


def test_torch_device():
    """Check available PyTorch devices."""
    print("\n" + "="*60)
    print("Checking PyTorch Devices")
    print("="*60)
    
    try:
        import torch
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("✗ CUDA not available")
        
        # Check XPU
        try:
            import intel_extension_for_pytorch as ipex
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                print(f"✓ Intel XPU available")
                print(f"  IPEX version: {ipex.__version__}")
                print(f"  Number of XPU devices: {torch.xpu.device_count()}")
            else:
                print("✗ Intel XPU not available")
        except ImportError:
            print("✗ Intel Extension for PyTorch not installed")
        
        # CPU always available
        print("✓ CPU available")
        
        return True
        
    except Exception as e:
        print(f"✗ Error checking devices: {e}")
        return False


def test_directory_structure():
    """Check if expected directories exist."""
    print("\n" + "="*60)
    print("Checking Directory Structure")
    print("="*60)
    
    import os
    
    expected_dirs = [
        'src/dpo_inv',
        'scripts',
        'examples',
    ]
    
    all_ok = True
    for dir_path in expected_dirs:
        if os.path.isdir(dir_path):
            print(f"✓ {dir_path}/ exists")
        else:
            print(f"✗ {dir_path}/ not found")
            all_ok = False
    
    # Check for data directory (may not exist yet)
    if os.path.isdir('data'):
        print(f"✓ data/ exists")
        if os.path.isdir('data/input/BioMPNN/soluble_model_weights'):
            print(f"✓ data/input/BioMPNN/soluble_model_weights/ exists")
        else:
            print(f"⚠ data/input/BioMPNN/soluble_model_weights/ not found")
            print(f"  Run: ./scripts/setup_base_models.sh")
    else:
        print(f"⚠ data/ not found (will be created when needed)")
    
    return all_ok


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("Next Steps")
    print("="*60)
    print()
    print("1. Download ProteinMPNN base models:")
    print("   ./scripts/setup_base_models.sh")
    print()
    print("2. Prepare your training data (see examples/):")
    print("   examples/example_training_data.json")
    print()
    print("3. Train a model:")
    print("   python -m dpo_inv.run_dpo \\")
    print("       --train_data data/training.json \\")
    print("       --output_dir outputs/my_model")
    print()
    print("4. Run inference:")
    print("   python -m dpo_inv.inference \\")
    print("       --model_path outputs/my_model/best_model.pt \\")
    print("       --pdb_file data/protein.pdb")
    print()
    print("See README.md for detailed instructions!")
    print()


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("DPO InvFold Installation Test")
    print("="*60)
    print()
    
    results = []
    
    results.append(("Python Version", test_python_version()))
    results.append(("Package Imports", test_imports()))
    results.append(("DPO InvFold Package", test_dpo_inv_import()))
    results.append(("PyTorch Devices", test_torch_device()))
    results.append(("Directory Structure", test_directory_structure()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    
    if all_passed:
        print("="*60)
        print("✓ All tests passed! Installation is complete.")
        print("="*60)
        print_next_steps()
        return 0
    else:
        print("="*60)
        print("✗ Some tests failed. Please fix the issues above.")
        print("="*60)
        print()
        print("Common fixes:")
        print("  - Install missing packages: pip install -e .")
        print("  - Install package in editable mode: pip install -e .")
        print("  - Check you're in the DPO_InvFold directory")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())

