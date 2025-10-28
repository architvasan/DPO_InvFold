# DPO InvFold

**Direct Preference Optimization for Inverse Protein Folding with BioMPNN**

Fine-tune ProteinMPNN using Direct Preference Optimization (DPO) to generate protein sequences optimized for specific preferences.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🧬 **DPO Training**: Fine-tune ProteinMPNN with preference pairs
- 🔗 **Multi-Chain Support**: Design specific chains while using others as context
- 🚀 **GPU Acceleration**: Supports CUDA and Intel XPU
- 🎯 **Flexible Inference**: Generate sequences for new protein backbones
- 📊 **Soluble Model Weights**: Optimized for soluble protein design

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Preparing ProteinMPNN Base Models](#preparing-proteinmpnn-base-models)
- [Training](#training)
- [Inference](#inference)
- [Multi-Chain Design](#multi-chain-design)
- [Hardware Support](#hardware-support)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.12 or higher
- CUDA (optional, for GPU acceleration)
- Intel Extension for PyTorch (optional, for Intel XPU)

### Step 1: Clone the Repository

```bash
git clone https://github.com/architvasan/DPO_InvFold.git
cd DPO_InvFold
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Using conda
conda create -n dpo_invfold python=3.9
conda activate dpo_invfold

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install the Package

```bash
# Install in editable mode with dependencies
pip install -e .

# OR install with development dependencies
pip install -e ".[dev]"

# OR install with Intel XPU support
pip install -e ".[xpu]"
```

### Step 4: Verify Installation

```bash
# Check available compute devices
python scripts/check_device_support.py
```

This will show:
- ✓ PyTorch installation
- ✓ CUDA availability (if applicable)
- ✓ Intel XPU availability (if applicable)
- ✓ All dependencies

---

## Quick Start

### 1. Download ProteinMPNN Base Models

```bash
# Run the automated setup script
./scripts/setup_base_models.sh
```

This will:
- Download ProteinMPNN soluble model weights
- Generate model configuration files
- Set up the directory structure

**Manual Download** (if script fails):
```bash
# Create directories
mkdir -p data/input/BioMPNN/soluble_model_weights
mkdir -p data/input/BioMPNN/base_hparams

# Download weights
cd data/input/BioMPNN/soluble_model_weights
wget https://files.ipd.uw.edu/pub/training_sets/soluble_model_weights/v_48_020.pt

cd ../../../..
```

### 2. Prepare Training Data

Create a JSON file with preference pairs:

```json
[
    {
        "pdb_file": "data/structures/protein1.pdb",
        "preferred_seq": "MKTAYIAKQRQISFVKSHFS",
        "unpreferred_seq": "MKTAYIAKQRQISFVKSHFA",
        "chain_id": "A"
    }
]
```

See `examples/example_training_data.json` for more examples.

### 3. Train the Model

```bash
python -m dpo_inv.run_dpo \
    --train_data data/training_data.json \
    --output_dir outputs/my_model \
    --num_epochs 10 \
    --batch_size 2 \
    --learning_rate 1e-4
```

### 4. Run Inference

```bash
python -m dpo_inv.inference \
    --model_path outputs/my_model/best_model.pt \
    --pdb_file data/structures/new_protein.pdb \
    --num_samples 100 \
    --temperature 0.1 \
    --output_dir outputs/inference
```

---

## Preparing ProteinMPNN Base Models

### Automated Setup (Recommended)

```bash
./scripts/setup_base_models.sh
```

**Options:**
1. **Automatic download** - Downloads soluble model weights from official source
2. **Clone ProteinMPNN repo** - Clones the full repository
3. **Use existing weights** - Point to your existing model files

### Manual Setup

#### 1. Download Model Weights

Download from: https://files.ipd.uw.edu/pub/training_sets/soluble_model_weights/

Available models:
- `v_48_002.pt` - Trained on 2020-02 PDB
- `v_48_010.pt` - Trained on 2020-10 PDB
- `v_48_020.pt` - Trained on 2021-02 PDB (recommended)
- `v_48_030.pt` - Trained on 2021-03 PDB

Place in: `data/input/BioMPNN/soluble_model_weights/`

#### 2. Generate Config Files

The training script will auto-generate YAML config files on first run, or you can create them manually:

```yaml
# data/input/BioMPNN/base_hparams/v_48_020.yaml
hidden_dim: 128
num_layers: 3
# ... other parameters
```

### Model Types

- **Soluble Models** (default): Optimized for soluble, globular proteins
- **Vanilla Models**: General-purpose models
- **Membrane Models**: Optimized for membrane proteins

To use vanilla or membrane models, change `--base_model_dir`:
```bash
--base_model_dir data/input/BioMPNN/vanilla_model_weights
```

---

## Training

### Basic Training

```bash
python -m dpo_inv.run_dpo \
    --train_data data/training_data.json \
    --output_dir outputs/my_model \
    --num_epochs 10
```

### Training Arguments

**Data Arguments:**
- `--train_data`: Path to training data JSON file (required)
- `--max_length`: Maximum protein length (default: 500)

**Model Arguments:**
- `--base_model_name`: Base model name (default: v_48_020)
- `--base_model_dir`: Directory with .pt files (default: soluble_model_weights)
- `--base_model_config_dir`: Directory with .yaml configs

**Training Arguments:**
- `--num_epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 2)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--beta`: DPO beta parameter (default: 0.1)
- `--temperature`: Sampling temperature (default: 0.1)

**Output Arguments:**
- `--output_dir`: Directory to save models and logs
- `--no_cuda`: Force CPU mode (disables GPU)

### Training Data Format

```json
[
    {
        "pdb_file": "path/to/structure.pdb",
        "preferred_seq": "ACDEFGHIKLMNPQRSTVWY",
        "unpreferred_seq": "ACDEFGHIKLMNPQRSTVWA",
        "chain_id": "A"
    }
]
```

**Fields:**
- `pdb_file`: Path to PDB structure file
- `preferred_seq`: Sequence with desired properties
- `unpreferred_seq`: Sequence with undesired properties
- `chain_id`: Chain to design (optional, defaults to first chain)

### Example Training Commands

**Small protein, quick training:**
```bash
python -m dpo_inv.run_dpo \
    --train_data data/small_proteins.json \
    --output_dir outputs/quick_test \
    --num_epochs 5 \
    --batch_size 4
```

**Large protein, careful training:**
```bash
python -m dpo_inv.run_dpo \
    --train_data data/large_proteins.json \
    --output_dir outputs/careful_training \
    --num_epochs 20 \
    --batch_size 1 \
    --learning_rate 5e-5
```

**CPU-only training:**
```bash
python -m dpo_inv.run_dpo \
    --train_data data/training.json \
    --output_dir outputs/cpu_model \
    --no_cuda \
    --batch_size 1
```

---

## Inference

### Basic Inference

```bash
python -m dpo_inv.inference \
    --model_path outputs/my_model/best_model.pt \
    --pdb_file data/structures/protein.pdb \
    --num_samples 100 \
    --output_dir outputs/inference
```

### Inference Arguments

**Input Arguments:**
- `--model_path`: Path to fine-tuned model .pt file (required)
- `--pdb_file`: Single PDB file to design
- `--pdb_list`: Text file with list of PDB files (one per line)
- `--chain_id`: Chain to design (optional)

**Multi-Chain Arguments:**
- `--design_chains`: Chains to design (e.g., `--design_chains B C`)
- `--fixed_chains`: Chains to keep fixed (e.g., `--fixed_chains A`)

**Model Arguments:**
- `--base_model_dir`: Directory with base model weights
- `--base_model_config_dir`: Directory with config files

**Sampling Arguments:**
- `--num_samples`: Number of sequences to generate (default: 10)
- `--temperature`: Sampling temperature (default: 0.1)

**Output Arguments:**
- `--output_dir`: Directory to save results (default: outputs/inference)
- `--no_cuda`: Force CPU mode

### Output Files

For each input PDB, inference generates:

1. **JSON file** (`protein_sequences.json`):
```json
{
    "pdb_file": "protein.pdb",
    "chain_id": "A",
    "sequences": ["ACDEFG...", "ACDEFH...", ...],
    "num_samples": 100,
    "temperature": 0.1
}
```

2. **FASTA file** (`protein_sequences.fasta`):
```
>protein.pdb_sample_1
ACDEFGHIKLMNPQRSTVWY
>protein.pdb_sample_2
ACDEFGHIKLMNPQRSTVWA
...
```

3. **Summary file** (`all_results.json`): Combined results for all inputs

### Example Inference Commands

**Single protein:**
```bash
python -m dpo_inv.inference \
    --model_path outputs/model.pt \
    --pdb_file data/protein.pdb \
    --num_samples 100
```

**Multiple proteins:**
```bash
# Create list file
ls data/structures/*.pdb > pdb_list.txt

# Run inference
python -m dpo_inv.inference \
    --model_path outputs/model.pt \
    --pdb_list pdb_list.txt \
    --num_samples 50
```

**High diversity sampling:**
```bash
python -m dpo_inv.inference \
    --model_path outputs/model.pt \
    --pdb_file protein.pdb \
    --num_samples 1000 \
    --temperature 0.5
```

---

## Multi-Chain Design

Design specific chains while using others as structural context.

### Use Cases

- **Antibody-Antigen**: Design antibody while keeping antigen fixed
- **Protein-Protein Interface**: Design one protein with binding partner as context
- **Multi-Domain Proteins**: Design one domain while keeping others fixed

### Training Data Format

```json
[
    {
        "pdb_file": "complex_AB.pdb",
        "preferred_seq": "MKTAYIAKQRQISFVKSHFS",
        "unpreferred_seq": "MKTAYIAKQRQISFVKSHFA",
        "design_chains": ["B"],
        "fixed_chains": ["A"]
    }
]
```

**Important**: `preferred_seq` and `unpreferred_seq` should contain **only** the sequences of the design chains, not the fixed chains.

### Training Example

```bash
python -m dpo_inv.run_dpo \
    --train_data data/multichain_training.json \
    --output_dir outputs/multichain_model \
    --num_epochs 10 \
    --batch_size 2
```

### Inference Example

```bash
python -m dpo_inv.inference \
    --model_path outputs/multichain_model/best_model.pt \
    --pdb_file data/complex_AB.pdb \
    --design_chains B \
    --fixed_chains A \
    --num_samples 100 \
    --output_dir outputs/design_B
```

This will:
1. Load chains A and B from the PDB
2. Keep chain A fixed (provides context)
3. Design chain B (generates 100 sequences)
4. Use inter-chain interactions to guide design

### Multiple Design Chains

Design chains A and B while keeping C fixed:

```bash
python -m dpo_inv.inference \
    --model_path outputs/model.pt \
    --pdb_file complex_ABC.pdb \
    --design_chains A B \
    --fixed_chains C \
    --num_samples 50
```

---

## Hardware Support

### CUDA (NVIDIA GPUs)

Automatically detected if available:
```bash
python -m dpo_inv.run_dpo --train_data data.json --output_dir outputs/
# Output: Using device: cuda
```

### Intel XPU (Intel GPUs)

Install Intel Extension for PyTorch:
```bash
pip install intel-extension-for-pytorch
```

Automatically detected:
```bash
python -m dpo_inv.run_dpo --train_data data.json --output_dir outputs/
# Output: Using device: xpu (Intel XPU)
```

See `XPU_SUPPORT.md` for detailed XPU setup and troubleshooting.

### CPU

Force CPU mode:
```bash
python -m dpo_inv.run_dpo --train_data data.json --no_cuda
# Output: Using device: cpu (forced CPU mode)
```

### Device Priority

The code automatically selects devices in this order:
1. Intel XPU (if IPEX installed and XPU available)
2. NVIDIA CUDA (if available)
3. CPU (fallback)

---

## Project Structure

```
DPO_InvFold/
├── src/dpo_inv/              # Main package
│   ├── __init__.py
│   ├── model.py              # BioMPNN model wrapper
│   ├── run_dpo.py            # DPO training script
│   ├── inference.py          # Inference script
│   ├── overwrite.py          # ProteinMPNN utilities
│   ├── mpnn_utils.py         # MPNN helper functions
│   └── data/                 # Data utilities
│       ├── __init__.py
│       ├── dd.py             # DotDict configuration
│       ├── global.py         # Global variables
│       └── utils.py          # Utility functions
├── scripts/                  # Helper scripts
│   ├── setup_base_models.sh  # Download ProteinMPNN models
│   ├── check_device_support.py  # Check hardware
│   └── test_multichain.py    # Test multi-chain support
├── examples/                 # Example files
│   ├── example_training_data.json
│   ├── example_multichain_training_data.json
│   └── run_example_training.sh
├── data/                     # Data directory (created by setup)
│   ├── input/BioMPNN/
│   │   ├── soluble_model_weights/
│   │   └── base_hparams/
│   └── structures/           # Your PDB files
├── outputs/                  # Training/inference outputs
├── pyproject.toml            # Package configuration
├── README.md                 # This file
└── LICENSE
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dpo_invfold2024,
  author = {Vasan, Archit},
  title = {DPO InvFold: Direct Preference Optimization for Inverse Protein Folding},
  year = {2024},
  url = {https://github.com/architvasan/DPO_InvFold}
}
```

Also cite the original ProteinMPNN paper:
```bibtex
@article{dauparas2022robust,
  title={Robust deep learning--based protein sequence design using ProteinMPNN},
  author={Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel and Bai, Hua and Ragotte, Robert J and Milles, Lukas F and Wicky, Basile IM and Courbet, Alexis and de Haas, Rob J and Bethel, Neville and others},
  journal={Science},
  volume={378},
  number={6615},
  pages={49--56},
  year={2022},
  publisher={American Association for the Advancement of Science}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'dpo_inv'`:
```bash
# Make sure you installed the package
pip install -e .

# OR add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### CUDA Out of Memory

Reduce batch size:
```bash
python -m dpo_inv.run_dpo --train_data data.json --batch_size 1
```

### Slow Training on CPU

- Use GPU if available
- Reduce `--max_length` to filter out long proteins
- Reduce `--batch_size`
- Reduce `--num_epochs`

### Chain Not Found in PDB

Check which chains exist:
```bash
grep "^ATOM" your_file.pdb | awk '{print $5}' | sort -u
```

### Sequence Length Mismatch

For multi-chain design, ensure `preferred_seq` length matches the total length of design chains only (not fixed chains).

---

## Additional Resources

- **Detailed Guides**: See `readmes/` directory
  - `SETUP_GUIDE.md` - Detailed setup instructions
  - `QUICKSTART.md` - Quick start guide
  - `MULTICHAIN_GUIDE.md` - Multi-chain design guide
  - `XPU_SUPPORT.md` - Intel XPU setup
  - `README_SOLUBLE_MODELS.md` - Soluble model information

- **Examples**: See `examples/` directory
  - Training data examples
  - Shell script examples

- **ProteinMPNN**: https://github.com/dauparas/ProteinMPNN
- **DPO Paper**: https://arxiv.org/abs/2305.18290

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Contact

For questions or issues, please open an issue on GitHub or contact:
- Archit Vasan: 125404521+architvasan@users.noreply.github.com
