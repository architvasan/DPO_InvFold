# Quick Start Guide

Get started with DPO training for BioMPNN in 5 minutes!

## 1. Install Dependencies

```bash
pip install torch numpy biopython pyyaml tqdm
```

## 2. Download Base Model Weights

Run the setup script:

```bash
chmod +x scripts/setup_base_models.sh
./scripts/setup_base_models.sh
```

Or manually download from [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) and place in:
- Weights: `data/input/BioMPNN/vanilla_model_weights/`
- Configs: `data/input/BioMPNN/base_hparams/`

## 3. Prepare Training Data

Create a JSON file with your preference pairs:

```json
[
    {
        "pdb_file": "path/to/structure.pdb",
        "preferred_seq": "ACDEFGHIKLMNPQRSTVWY",
        "unpreferred_seq": "ACDEFGHIKLMNPQRSTVWX",
        "chain_id": "A"
    }
]
```

See `examples/example_training_data.json` for a template.

## 4. Train the Model

```bash
python -m src.dpo_inv.run_dpo \
    --train_data data/training_data.json \
    --output_dir outputs/my_model \
    --num_epochs 10 \
    --batch_size 1 \
    --lr 3e-7 \
    --beta 0.1
```

## 5. Run Inference

Generate sequences for new backbones:

```bash
python -m src.dpo_inv.inference \
    --model_path outputs/my_model/best_model.pt \
    --pdb_file data/pdbs/my_protein.pdb \
    --output_dir outputs/inference \
    --num_samples 10
```

## What You Need Before Starting

### Required Files

1. **Base Model Weights** (`.pt` files)
   - Download from ProteinMPNN repository
   - Place in `data/input/BioMPNN/vanilla_model_weights/`
   - Common models: `v_48_020.pt`, `v_48_030.pt`

2. **Model Configs** (`.yaml` files)
   - Created automatically by setup script
   - Or create manually in `data/input/BioMPNN/base_hparams/`

3. **Training Data** (`.json` file)
   - Your preference pairs
   - Each pair: PDB file + preferred sequence + unpreferred sequence

4. **PDB Structures** (`.pdb` files)
   - Protein backbone structures
   - Can be from PDB database or computational models

### Directory Structure

```
DPO_InvFold/
├── data/
│   ├── input/BioMPNN/
│   │   ├── vanilla_model_weights/  ← Put .pt files here
│   │   └── base_hparams/           ← Put .yaml files here
│   ├── pdbs/                       ← Put your PDB files here
│   └── training_data.json          ← Your training data
├── outputs/                        ← Training outputs go here
└── src/dpo_inv/                    ← Source code
```

## Common Issues

### "Could not import BioMPNN modules"
- Make sure you installed all dependencies: `pip install torch numpy biopython pyyaml tqdm`

### "BioMPNN.base_model_yaml_dir_path must be set"
- Ensure you have `.yaml` config files in `data/input/BioMPNN/base_hparams/`
- Run the setup script to create them automatically

### "File not found" errors
- Check that paths in your training JSON are correct
- Use absolute paths or paths relative to where you run the script

### Out of memory
- Reduce `--batch_size` to 1
- Use smaller proteins (reduce `--max_length`)
- Use CPU instead: `--no_cuda`

## Next Steps

- Read the full [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed instructions
- Adjust hyperparameters for your specific use case
- Validate generated sequences with your quality metrics

## Key Parameters to Tune

- `--lr`: Learning rate (default: 3e-7)
  - Too high → unstable training
  - Too low → slow learning
  
- `--beta`: DPO preference strength (default: 0.1)
  - Higher → stronger preference learning
  - Lower → more conservative
  
- `--temperature`: Sampling diversity (default: 0.1)
  - Lower → more conservative sequences
  - Higher → more diverse sequences

## Getting Help

- Check [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed documentation
- Review example files in `examples/`
- Ensure all dependencies are installed correctly

