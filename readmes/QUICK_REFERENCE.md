# DPO InvFold - Quick Reference Card

## Installation

```bash
# Clone and install
git clone https://github.com/architvasan/DPO_InvFold.git
cd DPO_InvFold
pip install -e .

# Test installation
python scripts/test_installation.py

# Check devices
python scripts/check_device_support.py
```

## Setup ProteinMPNN Models

```bash
# Automated setup
./scripts/setup_base_models.sh

# Manual download
mkdir -p data/input/BioMPNN/soluble_model_weights
cd data/input/BioMPNN/soluble_model_weights
wget https://files.ipd.uw.edu/pub/training_sets/soluble_model_weights/v_48_020.pt
```

## Training

### Basic Training
```bash
python -m dpo_inv.run_dpo \
    --train_data data/training.json \
    --output_dir outputs/my_model \
    --num_epochs 10
```

### Common Training Options
```bash
# Small batch for large proteins
--batch_size 1

# Adjust learning rate
--learning_rate 1e-4

# DPO beta parameter
--beta 0.1

# Force CPU
--no_cuda

# Longer training
--num_epochs 20
```

### Training Data Format
```json
[
    {
        "pdb_file": "protein.pdb",
        "preferred_seq": "ACDEFG...",
        "unpreferred_seq": "ACDEFH...",
        "chain_id": "A"
    }
]
```

## Inference

### Basic Inference
```bash
python -m dpo_inv.inference \
    --model_path outputs/my_model/best_model.pt \
    --pdb_file data/protein.pdb \
    --num_samples 100 \
    --output_dir outputs/inference
```

### Common Inference Options
```bash
# Multiple PDBs
--pdb_list pdb_files.txt

# Specific chain
--chain_id A

# Sampling temperature
--temperature 0.1

# More samples
--num_samples 1000
```

## Multi-Chain Design

### Training Data
```json
{
    "pdb_file": "complex.pdb",
    "preferred_seq": "SEQUENCE_OF_CHAIN_B",
    "unpreferred_seq": "MUTANT_OF_CHAIN_B",
    "design_chains": ["B"],
    "fixed_chains": ["A"]
}
```

### Inference
```bash
python -m dpo_inv.inference \
    --model_path outputs/model.pt \
    --pdb_file complex.pdb \
    --design_chains B \
    --fixed_chains A \
    --num_samples 100
```

## File Locations

```
DPO_InvFold/
├── data/
│   ├── input/BioMPNN/
│   │   ├── soluble_model_weights/  # ProteinMPNN .pt files
│   │   └── base_hparams/           # Config .yaml files
│   └── structures/                 # Your PDB files
├── outputs/                        # Training outputs
│   └── my_model/
│       ├── best_model.pt          # Best checkpoint
│       ├── final_model.pt         # Final checkpoint
│       └── training_log.txt       # Training log
└── src/dpo_inv/                   # Source code
```

## Common Issues

### Import Error
```bash
# Fix: Install package
pip install -e .

# OR add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### CUDA Out of Memory
```bash
# Fix: Reduce batch size
--batch_size 1
```

### Chain Not Found
```bash
# Check chains in PDB
grep "^ATOM" file.pdb | awk '{print $5}' | sort -u
```

### Slow on CPU
```bash
# Use GPU or reduce workload
--max_length 200
--batch_size 1
--num_epochs 5
```

## Device Selection

```bash
# Automatic (XPU > CUDA > CPU)
python -m dpo_inv.run_dpo --train_data data.json

# Force CPU
python -m dpo_inv.run_dpo --train_data data.json --no_cuda

# Check available devices
python scripts/check_device_support.py
```

## Output Files

### Training
- `best_model.pt` - Best model checkpoint
- `final_model.pt` - Final model checkpoint
- `training_log.txt` - Training metrics

### Inference
- `protein_sequences.json` - Sequences in JSON
- `protein_sequences.fasta` - Sequences in FASTA
- `all_results.json` - Combined results

## Examples

### Quick Test
```bash
# 1. Setup
./scripts/setup_base_models.sh

# 2. Train (small)
python -m dpo_inv.run_dpo \
    --train_data examples/example_training_data.json \
    --output_dir outputs/test \
    --num_epochs 2 \
    --batch_size 1

# 3. Inference
python -m dpo_inv.inference \
    --model_path outputs/test/best_model.pt \
    --pdb_file data/structures/test.pdb \
    --num_samples 10
```

### Production Run
```bash
# Train
python -m dpo_inv.run_dpo \
    --train_data data/production_training.json \
    --output_dir outputs/production_model \
    --num_epochs 20 \
    --batch_size 2 \
    --learning_rate 5e-5

# Inference (batch)
ls data/structures/*.pdb > pdb_list.txt
python -m dpo_inv.inference \
    --model_path outputs/production_model/best_model.pt \
    --pdb_list pdb_list.txt \
    --num_samples 1000 \
    --temperature 0.1
```

### Multi-Chain Design
```bash
# Train
python -m dpo_inv.run_dpo \
    --train_data data/multichain_training.json \
    --output_dir outputs/multichain_model \
    --num_epochs 10

# Inference
python -m dpo_inv.inference \
    --model_path outputs/multichain_model/best_model.pt \
    --pdb_file data/complex_AB.pdb \
    --design_chains B \
    --fixed_chains A \
    --num_samples 100
```

## Help

```bash
# Training help
python -m dpo_inv.run_dpo --help

# Inference help
python -m dpo_inv.inference --help

# Check installation
python scripts/test_installation.py

# Check devices
python scripts/check_device_support.py
```

## Documentation

- `README.md` - Full documentation
- `readmes/SETUP_GUIDE.md` - Detailed setup
- `readmes/QUICKSTART.md` - Quick start guide
- `readmes/MULTICHAIN_GUIDE.md` - Multi-chain design
- `readmes/XPU_SUPPORT.md` - Intel XPU setup
- `examples/` - Example files

## Links

- GitHub: https://github.com/architvasan/DPO_InvFold
- ProteinMPNN: https://github.com/dauparas/ProteinMPNN
- Issues: https://github.com/architvasan/DPO_InvFold/issues

