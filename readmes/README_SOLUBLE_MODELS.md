# Using Soluble Model Weights

This project is configured to use **ProteinMPNN soluble model weights** by default, which are optimized for soluble proteins.

## Quick Setup

Run the automated setup script:

```bash
./scripts/setup_base_models.sh
```

This will:
1. Create the directory structure
2. Download soluble model weights from the official source
3. Generate configuration files

## Manual Download

If you prefer to download manually:

### Download URLs

Soluble model weights are available at:
```
https://files.ipd.uw.edu/pub/training_sets/soluble_model_weights/
```

Common models:
- `v_48_002.pt` - 48 neighbors, variant 002
- `v_48_010.pt` - 48 neighbors, variant 010
- `v_48_020.pt` - 48 neighbors, variant 020 (default)
- `v_48_030.pt` - 48 neighbors, variant 030

### Installation

1. Download the `.pt` files
2. Place them in: `data/input/BioMPNN/soluble_model_weights/`
3. Run the setup script to generate configs, or create them manually

## Directory Structure

```
data/input/BioMPNN/
├── soluble_model_weights/    # Soluble model .pt files
└── base_hparams/              # Model configuration .yaml files
```

## Using Soluble Models

### Training

The default configuration uses soluble models:

```bash
python -m src.dpo_inv.run_dpo \
    --train_data data/training_data.json \
    --output_dir outputs/my_model \
    --base_model_name v_48_020
```

The `--base_model_dir` defaults to `data/input/BioMPNN/soluble_model_weights`.

### Inference

Similarly for inference:

```bash
python -m src.dpo_inv.inference \
    --model_path outputs/my_model/best_model.pt \
    --pdb_file your_protein.pdb \
    --output_dir outputs/inference
```

## Using Other Model Types

If you want to use vanilla (non-soluble) models or other variants:

1. Download the appropriate weights
2. Place them in a different directory (e.g., `data/input/BioMPNN/vanilla_model_weights/`)
3. Specify the directory when running:

```bash
python -m src.dpo_inv.run_dpo \
    --train_data data/training_data.json \
    --base_model_dir data/input/BioMPNN/vanilla_model_weights \
    --base_model_name v_48_020
```

## Model Variants

### Soluble Models (Recommended for most proteins)
- Optimized for soluble, globular proteins
- Better performance on typical protein design tasks
- **Use these by default**

### Vanilla Models
- General-purpose models
- Can be used for any protein type
- Available from the ProteinMPNN repository

### Membrane Models
- Optimized for membrane proteins
- Use if designing transmembrane proteins
- Requires different weights (not included by default)

## Troubleshooting

### "File not found" errors

Make sure you've run the setup script:
```bash
./scripts/setup_base_models.sh
```

Or manually verify files exist:
```bash
ls data/input/BioMPNN/soluble_model_weights/
ls data/input/BioMPNN/base_hparams/
```

### Download failures

If automatic download fails:
1. Download manually from: https://files.ipd.uw.edu/pub/training_sets/soluble_model_weights/
2. Place files in `data/input/BioMPNN/soluble_model_weights/`
3. Run setup script again to generate configs

### Using existing weights

If you already have ProteinMPNN weights elsewhere:

```bash
# Option 1: Create symbolic link
ln -s /path/to/your/weights data/input/BioMPNN/soluble_model_weights

# Option 2: Copy files
cp /path/to/your/weights/*.pt data/input/BioMPNN/soluble_model_weights/

# Then generate configs
./scripts/setup_base_models.sh
```

## References

- ProteinMPNN: https://github.com/dauparas/ProteinMPNN
- Paper: Dauparas et al., Science 2022
- Soluble models are trained specifically on soluble protein structures

