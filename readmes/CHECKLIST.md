# Pre-Training Checklist

Use this checklist to ensure you have everything ready before running DPO training.

## ✅ Setup Checklist

### 1. Dependencies Installed
- [ ] PyTorch installed (`pip install torch`)
- [ ] NumPy installed (`pip install numpy`)
- [ ] BioPython installed (`pip install biopython`)
- [ ] PyYAML installed (`pip install pyyaml`)
- [ ] tqdm installed (`pip install tqdm`)

**Verify:** Run `python -c "import torch, numpy, Bio, yaml, tqdm; print('All dependencies OK!')"`

### 2. Base Model Files

#### Model Weights (.pt files)
- [ ] Downloaded ProteinMPNN model weights
- [ ] Placed in `data/input/BioMPNN/vanilla_model_weights/`
- [ ] At least one model file exists (e.g., `v_48_020.pt`)

**Verify:** Run `ls data/input/BioMPNN/vanilla_model_weights/*.pt`

#### Model Configs (.yaml files)
- [ ] Created YAML config files
- [ ] Placed in `data/input/BioMPNN/base_hparams/`
- [ ] Config exists for each model weight file

**Verify:** Run `ls data/input/BioMPNN/base_hparams/*.yaml`

**Quick Setup:** Run `./scripts/setup_base_models.sh` to automate this

### 3. Training Data Prepared

- [ ] Created training data JSON file
- [ ] JSON follows correct format (see below)
- [ ] All PDB files referenced in JSON exist
- [ ] Sequences match the length of structures
- [ ] Preference pairs are meaningful (preferred vs unpreferred)

**JSON Format:**
```json
[
    {
        "pdb_file": "path/to/structure.pdb",
        "preferred_seq": "ACDEFG...",
        "unpreferred_seq": "ACDEFH...",
        "chain_id": "A"
    }
]
```

**Verify:** Run `python -c "import json; data=json.load(open('data/training_data.json')); print(f'{len(data)} training examples loaded')"`

### 4. Directory Structure

- [ ] `data/input/BioMPNN/vanilla_model_weights/` exists
- [ ] `data/input/BioMPNN/base_hparams/` exists
- [ ] `data/pdbs/` exists (or wherever your PDB files are)
- [ ] `outputs/` directory will be created automatically

**Verify:** Run `ls -d data/input/BioMPNN/*/`

### 5. PDB Files

- [ ] All PDB files are valid
- [ ] PDB files contain the chains specified in training data
- [ ] Structures are reasonable quality (no major missing atoms)

**Verify:** Check one PDB file:
```python
from Bio.PDB import PDBParser
parser = PDBParser()
structure = parser.get_structure('test', 'your_file.pdb')
print(f"Chains: {[c.id for c in structure[0].get_chains()]}")
```

## 🚀 Ready to Train?

If all items above are checked, you're ready to start training!

### Minimal Training Command

```bash
python -m src.dpo_inv.run_dpo \
    --train_data data/training_data.json \
    --output_dir outputs/my_model
```

### Recommended Training Command

```bash
python -m src.dpo_inv.run_dpo \
    --train_data data/training_data.json \
    --output_dir outputs/my_model \
    --base_model_name v_48_020 \
    --num_epochs 10 \
    --batch_size 1 \
    --lr 3e-7 \
    --beta 0.1 \
    --temperature 0.1
```

## 📊 During Training

Monitor these things:

- [ ] Training loss is decreasing
- [ ] Validation loss is decreasing (or stable)
- [ ] No CUDA out of memory errors
- [ ] Checkpoints are being saved to output directory

**Warning Signs:**
- Training loss increases → Learning rate too high
- Validation loss increases while training decreases → Overfitting
- Loss is NaN → Learning rate too high or data issue

## 🔍 After Training

- [ ] Best model saved: `outputs/my_model/best_model.pt`
- [ ] Config saved: `outputs/my_model/config.json`
- [ ] Training completed without errors

## 🧪 Testing Inference

Before running on many structures, test on one:

```bash
python -m src.dpo_inv.inference \
    --model_path outputs/my_model/best_model.pt \
    --pdb_file data/pdbs/test_protein.pdb \
    --output_dir outputs/test_inference \
    --num_samples 5
```

Check:
- [ ] Sequences generated successfully
- [ ] Output files created (JSON and FASTA)
- [ ] Sequences look reasonable (no excessive X's or weird patterns)

## 🐛 Troubleshooting

### Common Issues

**"Could not import BioMPNN modules"**
- Check all dependencies are installed
- Verify you're running from the correct directory

**"BioMPNN.base_model_yaml_dir_path must be set"**
- Ensure YAML config files exist
- Check file names match model names

**"File not found" for PDB**
- Verify paths in JSON are correct
- Use absolute paths if relative paths don't work

**Out of memory**
- Reduce `--batch_size` to 1
- Use smaller proteins
- Add `--no_cuda` to use CPU

**Model not learning**
- Increase learning rate slightly (try 1e-6)
- Increase beta (try 0.2)
- Check preference pairs are meaningful

## 📚 Additional Resources

- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Detailed setup instructions
- `examples/example_training_data.json` - Example data format
- `examples/run_example_training.sh` - Example training script

## 🎯 Best Practices

1. **Start small**: Test with a few examples first
2. **Validate data**: Ensure preference pairs are high quality
3. **Monitor training**: Watch for overfitting
4. **Save checkpoints**: Use `--save_every` to save regularly
5. **Test inference**: Validate generated sequences before large-scale use

---

**Ready to go?** Run your training command and monitor the output!

