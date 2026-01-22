# Mutable Mask Feature for Diversity-Regularized DPO

## Overview

The mutable mask feature allows you to specify which positions in your sequences should be considered when computing diversity loss. This is particularly useful for **targeted mutagenesis** where you only want to mutate specific residues while keeping others fixed.

## Problem It Solves

When doing targeted mutagenesis (e.g., mutating only residues 10-15 in a 100-residue peptide):
- Without mutable mask: Diversity is computed over ALL positions, so 85% of positions are identical
- With mutable mask: Diversity is computed ONLY over the mutable positions (10-15), giving accurate diversity metrics

## Usage

### Training Data Format

Add `mutable_positions` or `mutable_ranges` to your training JSON:

```json
[
    {
        "pdb_file": "structure.pdb",
        "preferred_seq": "ACDEFGHIKLMNPQRSTVWY",
        "unpreferred_seq": "ACDEFGHIKLMNPQRSTVWX",
        "mutable_positions": [10, 11, 12, 13, 14, 15]
    }
]
```

Or use ranges:

```json
[
    {
        "pdb_file": "structure.pdb",
        "preferred_seq": "ACDEFGHIKLMNPQRSTVWY",
        "unpreferred_seq": "ACDEFGHIKLMNPQRSTVWX",
        "mutable_ranges": [[10, 15], [20, 25]]
    }
]
```

**Note:** Positions are 0-indexed. Ranges are inclusive.

### Training

```bash
python src/dpo_inv/run_dpo_divreg.py \
    --train_data data/train_with_mutable.json \
    --diversity_lambda 0.5 \
    --num_epochs 10 \
    --batch_size 4
```

The diversity loss will automatically use the mutable masks from your data.

### Inference

#### Option 1: Specify mutable positions

```bash
python src/dpo_inv/inference.py \
    --model_path output/best_model.pt \
    --pdb_file structure.pdb \
    --mutable_positions 10 11 12 13 14 15 \
    --num_samples 100 \
    --temperature 0.1
```

#### Option 2: Specify mutable ranges

```bash
python src/dpo_inv/inference.py \
    --model_path output/best_model.pt \
    --pdb_file structure.pdb \
    --mutable_ranges 10-15 20-25 \
    --num_samples 100 \
    --temperature 0.1
```

#### Option 3: No mutable mask (design all positions)

```bash
python src/dpo_inv/inference.py \
    --model_path output/best_model.pt \
    --pdb_file structure.pdb \
    --num_samples 100 \
    --temperature 0.1
```

## How It Works

### Diversity Loss Computation

The diversity loss is computed as:

```
L_div = -λ * (2 / (K(K-1))) * Σ_{i<j} d(y_i, y_j)
```

Where `d(y_i, y_j)` is the distance between sequences i and j:

```
d(y_i, y_j) = 1 - sequence_identity(y_i, y_j)
```

**With mutable mask:**
- `sequence_identity` is computed ONLY over positions where `mutable_mask = 1.0`
- Conserved regions are ignored in the diversity calculation
- This gives accurate diversity metrics for targeted mutagenesis

**Without mutable mask:**
- `sequence_identity` is computed over ALL valid positions
- Standard behavior for full sequence design

### Example

Suppose you have a 20-residue peptide and want to mutate only positions 10-15:

```python
# Sequence 1: AAAAAAAAAA[DEFGHI]AAAA
# Sequence 2: AAAAAAAAAA[KLMNPQ]AAAA
#                       ^^^^^^ mutable region

# Without mutable mask:
# sequence_identity = 14/20 = 0.70
# distance = 1 - 0.70 = 0.30

# With mutable mask on positions 10-15:
# sequence_identity = 0/6 = 0.00  (only comparing mutable positions)
# distance = 1 - 0.00 = 1.00  (maximally diverse!)
```

## Best Practices

1. **For targeted mutagenesis:** Always use `mutable_positions` or `mutable_ranges`
2. **For full sequence design:** Omit mutable mask (default behavior)
3. **For multiple regions:** Use `mutable_ranges` instead of listing all positions
4. **Diversity lambda:** Start with `diversity_lambda=0.1` and increase if needed

## Implementation Details

- Mutable mask is combined with the chain mask: `effective_mask = chain_M * mutable_mask`
- Only positions that are both valid (chain_M=1) AND mutable (mutable_mask=1) are considered
- If no mutable mask is provided, all valid positions are used (backward compatible)

## See Also

- `example_mutable_positions.json` - Example training data with mutable masks
- Paper: "Diversity-Regularized Direct Preference Optimization for Peptide Inverse Folding"

