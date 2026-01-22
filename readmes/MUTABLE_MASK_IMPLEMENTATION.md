# Mutable Mask Implementation Summary

## Overview

This document summarizes the implementation of the mutable mask feature for diversity-regularized DPO training. The feature allows users to specify which positions should be considered when computing diversity loss, enabling accurate diversity metrics for targeted mutagenesis.

## Files Modified

### 1. `src/dpo_inv/run_dpo_divreg.py`

#### New Functions

**`create_mutable_mask(item, seq_length, device='cpu')`**
- Creates a mutable position mask from dataset item
- Supports two formats:
  - `mutable_positions`: List of 0-indexed positions (e.g., `[10, 11, 12, 13, 14, 15]`)
  - `mutable_ranges`: List of [start, end] ranges (e.g., `[[10, 15], [20, 25]]`)
- Returns tensor of shape `[seq_length]` with 1.0 for mutable positions
- Default: All positions are mutable if not specified

#### Modified Functions

**`collate_preference_batch(batch_list, device='cpu')`**
- Now returns 4 values instead of 3: `(structure_batch, S_preferred, S_unpreferred, mutable_mask_batch)`
- Creates mutable masks for each item in the batch using `create_mutable_mask()`
- Stacks mutable masks into batch tensor of shape `[B, L]`

**`compute_diversity_loss(sequences, mask, mutable_mask=None)`**
- Added optional `mutable_mask` parameter
- If provided, combines with regular mask: `effective_mask = mask * mutable_mask`
- Only positions that are both valid AND mutable are considered for diversity
- Backward compatible: if `mutable_mask=None`, uses regular mask (old behavior)

**`compute_dpo_loss(..., mutable_mask=None, ...)`**
- Added `mutable_mask` parameter
- Passes mutable mask to `compute_diversity_loss()`

**`train_epoch(...)` and `validate(...)`**
- Updated to unpack 4 values from `collate_preference_batch()`
- Pass `mutable_mask` to `compute_dpo_loss()`

#### Updated Documentation

**`PreferenceDataset.__init__()` docstring**
- Added documentation for `mutable_positions` and `mutable_ranges` fields

### 2. `src/dpo_inv/inference.py`

#### Modified Functions

**`generate_sequences(..., mutable_positions=None, mutable_ranges=None, ...)`**
- Added `mutable_positions` and `mutable_ranges` parameters
- Creates mutable mask and combines with `chain_M` to restrict design to mutable positions
- Prints diagnostic information about mutable positions

**`main(args)`**
- Parses `mutable_ranges` from command-line format (e.g., "10-15" → [10, 15])
- Passes mutable positions/ranges to `generate_sequences()`

#### New Command-Line Arguments

```bash
--mutable_positions 10 11 12 13 14 15
--mutable_ranges 10-15 20-25
```

## New Files Created

### 1. `example_mutable_positions.json`
Example training data showing how to use mutable positions/ranges

### 2. `MUTABLE_MASK_USAGE.md`
Comprehensive user guide with:
- Problem description
- Usage examples for training and inference
- How it works (mathematical explanation)
- Best practices
- Implementation details

### 3. `test_mutable_mask_simple.py`
Simple test script to verify:
- Default behavior (all positions mutable)
- Specific mutable positions
- Mutable ranges

## Key Design Decisions

### 1. Backward Compatibility
- If no mutable mask is specified, all positions are mutable (default behavior)
- Existing code without mutable masks continues to work unchanged

### 2. Flexibility
- Supports both individual positions and ranges
- Can specify different mutable positions for each item in the batch

### 3. Consistency
- Same format for training data and inference arguments
- Positions are always 0-indexed
- Ranges are always inclusive

### 4. Safety
- Bounds checking: positions outside [0, seq_length) are ignored
- Combines with chain_M mask: only positions that are both valid AND mutable are designed

## Usage Examples

### Training with Mutable Positions

```json
{
    "pdb_file": "structure.pdb",
    "preferred_seq": "ACDEFGHIKLMNPQRSTVWY",
    "unpreferred_seq": "ACDEFGHIKLMNPQRSTVWX",
    "mutable_positions": [10, 11, 12, 13, 14, 15]
}
```

```bash
python src/dpo_inv/run_dpo_divreg.py \
    --train_data data/train_with_mutable.json \
    --diversity_lambda 0.5
```

### Inference with Mutable Positions

```bash
python src/dpo_inv/inference.py \
    --model_path output/best_model.pt \
    --pdb_file structure.pdb \
    --mutable_positions 10 11 12 13 14 15 \
    --num_samples 100
```

Or with ranges:

```bash
python src/dpo_inv/inference.py \
    --model_path output/best_model.pt \
    --pdb_file structure.pdb \
    --mutable_ranges 10-15 20-25 \
    --num_samples 100
```

## Testing

Run the simple test:

```bash
python test_mutable_mask_simple.py
```

Expected output:
```
Test 1: Default (all mutable)
  Sum: 20.0/20
  ✓ PASSED

Test 2: Specific positions
  Sum: 6.0/20
  Mutable positions: [10, 11, 12, 13, 14, 15]
  ✓ PASSED

Test 3: Ranges
  Sum: 11.0/20
  Mutable positions: [5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19]
  ✓ PASSED
```

## Impact on Diversity Metrics

For a 20-residue peptide with mutations only at positions 10-15:

**Without mutable mask:**
- Compares all 20 positions
- 14 positions identical, 6 different
- Sequence identity = 14/20 = 0.70
- Distance = 0.30

**With mutable mask (positions 10-15):**
- Compares only 6 mutable positions
- 0 positions identical, 6 different
- Sequence identity = 0/6 = 0.00
- Distance = 1.00 (maximally diverse!)

This gives accurate diversity metrics for targeted mutagenesis scenarios.

