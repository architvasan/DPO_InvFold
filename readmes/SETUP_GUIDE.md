# DPO_InvFold Setup and Usage Guide

This guide will help you set up and run Direct Preference Optimization (DPO) training on BioMPNN (ProteinMPNN).

## Prerequisites

### 1. Install Dependencies

```bash
pip install torch numpy biopython pyyaml tqdm
```

### 2. Download ProteinMPNN Base Weights

You need to download the ProteinMPNN base model weights and configuration files. These are required before running DPO training.

#### Option A: Download from ProteinMPNN Repository

```bash
# Create directories
mkdir -p data/input/BioMPNN/vanilla_model_weights
mkdir -p data/input/BioMPNN/base_hparams

# Download from ProteinMPNN repository
# Visit: https://github.com/dauparas/ProteinMPNN
# Download the model weights (.pt files) and place them in:
#   data/input/BioMPNN/vanilla_model_weights/

# Common model names:
# - v_48_002.pt
# - v_48_010.pt
# - v_48_020.pt (default)
# - v_48_030.pt
```

#### Option B: Use Existing CAPE-MPNN Weights

If you already have CAPE-MPNN or ProteinMPNN weights:

```bash
# Create symbolic links or copy files
ln -s /path/to/your/CAPE-MPNN/vanilla_model_weights data/input/BioMPNN/vanilla_model_weights
ln -s /path/to/your/CAPE-MPNN/base_hparams data/input/BioMPNN/base_hparams
```

### 3. Create Base Model Configuration Files

You need YAML configuration files for each model. Create them in `data/input/BioMPNN/base_hparams/`.

Example `v_48_020.yaml`:

```yaml
num_letters: 21
node_features: 128
edge_features: 128
hidden_dim: 128
num_encoder_layers: 3
num_decoder_layers: 3
vocab: 21
k_neighbors: 48
dropout: 0.1
```

Adjust parameters based on your specific model variant.

## Preparing Training Data

### Data Format

Create a JSON file with preference pairs. Each entry should have:
- `pdb_file`: Path to protein backbone structure (PDB format)
- `preferred_seq`: Sequence with higher reward/quality
- `unpreferred_seq`: Sequence with lower reward/quality
- `chain_id`: (optional) Chain to design, defaults to first chain

Example (`training_data.json`):

```json
[
    {
        "pdb_file": "data/pdbs/1abc.pdb",
        "preferred_seq": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPIL",
        "unpreferred_seq": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPXX",
        "chain_id": "A"
    },
    {
        "pdb_file": "data/pdbs/2xyz.pdb",
        "preferred_seq": "ACDEFGHIKLMNPQRSTVWY",
        "unpreferred_seq": "ACDEFGHIKLMNPQRSTVWX",
        "chain_id": "A"
    }
]
```

### How to Generate Preference Pairs

You can create preference pairs based on:

1. **Experimental validation**: Sequences that work vs. don't work
2. **Computational metrics**: 
   - High vs. low predicted stability (e.g., Rosetta energy)
   - High vs. low predicted binding affinity
   - High vs. low sequence recovery from other models
3. **Functional assays**: Active vs. inactive variants
4. **Natural sequences**: Native vs. mutated sequences

See `examples/example_training_data.json` for a template.

## Running DPO Training

### Basic Training Command

```bash
python -m src.dpo_inv.run_dpo \
    --train_data data/training_data.json \
    --output_dir outputs/dpo_model \
    --base_model_name v_48_020 \
    --num_epochs 10 \
    --batch_size 1 \
    --lr 3e-7 \
    --beta 0.1 \
    --temperature 0.1
```

### Training Arguments

**Data:**
- `--train_data`: Path to training JSON file (required)
- `--val_data`: Path to validation JSON file (optional, will split train if not provided)

**Model:**
- `--base_model_name`: Name of base ProteinMPNN model (default: `v_48_020`)
- `--base_model_dir`: Directory with .pt files (default: `data/input/BioMPNN/vanilla_model_weights`)
- `--base_model_config_dir`: Directory with .yaml files (default: `data/input/BioMPNN/base_hparams`)

**Training:**
- `--num_epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 1, increase if you have GPU memory)
- `--lr`: Learning rate (default: 3e-7, very small to avoid catastrophic forgetting)
- `--weight_decay`: Weight decay (default: 0.0)
- `--beta`: DPO beta parameter - KL penalty weight (default: 0.1)
- `--temperature`: Sampling temperature (default: 0.1)
- `--max_length`: Maximum protein length (default: 500)

**Output:**
- `--output_dir`: Where to save checkpoints and logs (default: `outputs/dpo`)
- `--save_every`: Save checkpoint every N epochs (default: 1)

**Hardware:**
- `--no_cuda`: Disable CUDA even if available

### Example Training Session

```bash
# Train with custom parameters
python -m src.dpo_inv.run_dpo \
    --train_data data/my_preferences.json \
    --val_data data/my_val_preferences.json \
    --output_dir outputs/my_dpo_model \
    --base_model_name v_48_020 \
    --num_epochs 20 \
    --batch_size 2 \
    --lr 1e-7 \
    --beta 0.2 \
    --save_every 5
```

### Output Files

After training, you'll find in `--output_dir`:
- `config.json`: Training configuration
- `model_epoch_X.pt`: Model checkpoints
- `best_model.pt`: Best model based on validation loss

## Running Inference

Use the fine-tuned model to generate sequences for new protein backbones.

### Basic Inference Command

```bash
python -m src.dpo_inv.inference \
    --model_path outputs/dpo_model/best_model.pt \
    --pdb_file data/pdbs/my_protein.pdb \
    --output_dir outputs/inference \
    --num_samples 10 \
    --temperature 0.1
```

### Inference Arguments

**Input:**
- `--model_path`: Path to fine-tuned model .pt file (required)
- `--pdb_file`: Single PDB file to design
- `--pdb_list`: Text file with list of PDB files (one per line)
- `--chain_id`: Chain to design (optional, uses first chain if not specified)

**Model paths (for loading configs):**
- `--base_model_dir`: Same as training
- `--base_model_config_dir`: Same as training

**Sampling:**
- `--num_samples`: Number of sequences to generate per structure (default: 10)
- `--temperature`: Sampling temperature (default: 0.1, lower = more conservative)

**Output:**
- `--output_dir`: Where to save results (default: `outputs/inference`)

### Batch Inference

For multiple structures:

```bash
# Create a list file
echo "data/pdbs/protein1.pdb" > pdb_list.txt
echo "data/pdbs/protein2.pdb" >> pdb_list.txt
echo "data/pdbs/protein3.pdb" >> pdb_list.txt

# Run inference
python -m src.dpo_inv.inference \
    --model_path outputs/dpo_model/best_model.pt \
    --pdb_list pdb_list.txt \
    --output_dir outputs/batch_inference \
    --num_samples 20 \
    --temperature 0.1
```

### Output Files

For each input PDB, you'll get:
- `{pdb_name}_sequences.json`: Sequences in JSON format
- `{pdb_name}_sequences.fasta`: Sequences in FASTA format
- `all_results.json`: Combined results for all structures

## Tips and Best Practices

### Training Tips

1. **Start with small learning rate**: 3e-7 to 1e-6 to avoid catastrophic forgetting
2. **Adjust beta**: Higher beta (0.2-0.5) = stronger preference learning, lower beta (0.05-0.1) = more conservative
3. **Monitor validation loss**: Stop if validation loss increases (overfitting)
4. **Use quality preference pairs**: The better your preferred/unpreferred distinction, the better the results

### Inference Tips

1. **Temperature**: 
   - Lower (0.05-0.1) = more conservative, closer to training distribution
   - Higher (0.3-1.0) = more diverse sequences
2. **Generate multiple samples**: Diversity helps find better sequences
3. **Post-process**: Filter generated sequences by your quality metrics

### Troubleshooting

**Out of memory:**
- Reduce `--batch_size` to 1
- Reduce `--max_length`
- Use smaller proteins for training

**Model not learning:**
- Increase `--lr` slightly (try 1e-6)
- Increase `--beta` (try 0.2-0.3)
- Check that preference pairs are meaningful

**Poor sequence quality:**
- Lower temperature during inference
- Train for more epochs
- Use better quality preference pairs

## Directory Structure

```
DPO_InvFold/
├── data/
│   ├── input/
│   │   └── BioMPNN/
│   │       ├── vanilla_model_weights/  # Base model .pt files
│   │       └── base_hparams/           # Model config .yaml files
│   ├── pdbs/                           # Your PDB structures
│   ├── training_data.json              # Training preferences
│   └── val_data.json                   # Validation preferences
├── outputs/
│   ├── dpo_model/                      # Training outputs
│   └── inference/                      # Inference results
├── src/
│   └── dpo_inv/
│       ├── run_dpo.py                  # Training script
│       ├── inference.py                # Inference script
│       ├── model.py                    # BioMPNN model
│       └── ...
└── examples/
    └── example_training_data.json      # Example data format
```

## Next Steps

1. Download ProteinMPNN weights
2. Create your preference dataset
3. Run training
4. Evaluate on validation set
5. Run inference on new backbones
6. Validate generated sequences experimentally or computationally

