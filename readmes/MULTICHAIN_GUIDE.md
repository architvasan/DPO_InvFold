# Multi-Chain Protein Design Guide

This guide explains how to use the multi-chain capabilities of BioMPNN for DPO training and inference.

## Overview

The code now supports **multi-chain protein design**, allowing you to:
- Design one or more chains while keeping other chains fixed as context
- Use inter-chain interactions to guide sequence design
- Maintain backward compatibility with single-chain workflows

## Key Concepts

### Design Chains vs Fixed Chains

- **Design Chains** (`design_chains`): Chains that will be redesigned (masked during training)
- **Fixed Chains** (`fixed_chains`): Chains that provide structural context but are not redesigned (visible during training)

### Example Use Cases

1. **Antibody Design**: Design the CDR loops (chain H) while keeping the antigen (chain A) fixed
2. **Protein-Protein Interface**: Design one protein while using the binding partner as context
3. **Multi-Domain Proteins**: Design one domain while keeping others fixed
4. **Symmetric Complexes**: Design multiple chains simultaneously

## Training Data Format

### Multi-Chain Format (New)

```json
[
    {
        "pdb_file": "path/to/structure.pdb",
        "preferred_seq": "ACDEFG...",
        "unpreferred_seq": "ACDEFH...",
        "design_chains": ["B"],
        "fixed_chains": ["A"]
    }
]
```

**Fields:**
- `pdb_file`: Path to PDB structure containing multiple chains
- `preferred_seq`: Preferred sequence for the **design chains only** (concatenated if multiple)
- `unpreferred_seq`: Unpreferred sequence for the **design chains only**
- `design_chains`: List of chain IDs to design (e.g., `["B"]` or `["A", "B"]`)
- `fixed_chains`: List of chain IDs to keep fixed (e.g., `["A"]` or `["C", "D"]`)

### Single-Chain Format (Legacy, Still Supported)

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

**Fields:**
- `chain_id`: Single chain to design (optional, defaults to first chain)

## Training Examples

### Example 1: Design Chain B with Chain A as Context

**Scenario**: You have a protein-protein complex (chains A and B). You want to design chain B to bind better to chain A.

**Training Data** (`data/train_AB.json`):
```json
[
    {
        "pdb_file": "data/structures/complex_AB.pdb",
        "preferred_seq": "MKTAYIAKQRQISFVKSHFS",
        "unpreferred_seq": "MKTAYIAKQRQISFVKSHFA",
        "design_chains": ["B"],
        "fixed_chains": ["A"],
        "description": "Design chain B to improve binding to chain A"
    }
]
```

**Training Command**:
```bash
python -m src.dpo_inv.run_dpo \
    --train_data data/train_AB.json \
    --output_dir outputs/design_B_with_A_context \
    --num_epochs 10 \
    --batch_size 2
```

### Example 2: Design Multiple Chains Simultaneously

**Scenario**: Design chains A and B together while keeping chain C fixed.

**Training Data** (`data/train_ABC.json`):
```json
[
    {
        "pdb_file": "data/structures/complex_ABC.pdb",
        "preferred_seq": "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",
        "unpreferred_seq": "ACDEFGHIKLMNPQRSTVWAACDEFGHIKLMNPQRSTVWA",
        "design_chains": ["A", "B"],
        "fixed_chains": ["C"],
        "description": "Co-design chains A and B with C as context"
    }
]
```

**Note**: The `preferred_seq` and `unpreferred_seq` should be the concatenation of sequences for chains A and B (in that order).

**Training Command**:
```bash
python -m src.dpo_inv.run_dpo \
    --train_data data/train_ABC.json \
    --output_dir outputs/codesign_AB_with_C_context \
    --num_epochs 10
```

### Example 3: Single Chain (Backward Compatible)

**Scenario**: Traditional single-chain design (no multi-chain context).

**Training Data** (`data/train_single.json`):
```json
[
    {
        "pdb_file": "data/structures/protein.pdb",
        "preferred_seq": "ACDEFGHIKLMNPQRSTVWY",
        "unpreferred_seq": "ACDEFGHIKLMNPQRSTVWA",
        "chain_id": "A"
    }
]
```

**Training Command**:
```bash
python -m src.dpo_inv.run_dpo \
    --train_data data/train_single.json \
    --output_dir outputs/single_chain_design \
    --num_epochs 10
```

## Inference Examples

### Example 1: Design Chain B with Chain A as Context

**Command**:
```bash
python -m src.dpo_inv.inference \
    --model_path outputs/my_model/best_model.pt \
    --pdb_file data/structures/complex_AB.pdb \
    --design_chains B \
    --fixed_chains A \
    --num_samples 100 \
    --temperature 0.1 \
    --output_dir outputs/inference_B
```

**Output**: 100 sequences for chain B, designed to interact with the fixed chain A.

### Example 2: Design Multiple Chains

**Command**:
```bash
python -m src.dpo_inv.inference \
    --model_path outputs/my_model/best_model.pt \
    --pdb_file data/structures/complex_ABC.pdb \
    --design_chains A B \
    --fixed_chains C \
    --num_samples 50 \
    --output_dir outputs/inference_AB
```

**Output**: 50 sequences for chains A and B (concatenated), designed with chain C as context.

### Example 3: Single Chain Inference (Legacy)

**Command**:
```bash
python -m src.dpo_inv.inference \
    --model_path outputs/my_model/best_model.pt \
    --pdb_file data/structures/protein.pdb \
    --chain_id A \
    --num_samples 100 \
    --output_dir outputs/inference_single
```

## Important Notes

### Sequence Length

When using multi-chain design:
- `preferred_seq` and `unpreferred_seq` should contain **only the sequences of the design chains**
- If designing multiple chains (e.g., `["A", "B"]`), concatenate their sequences in the same order
- Fixed chains are NOT included in the sequence strings

**Example**:
```json
{
    "design_chains": ["B", "C"],
    "fixed_chains": ["A"],
    "preferred_seq": "SEQFORCHAINB" + "SEQFORCHAINC"
}
```

### Chain Order

- Chains are processed in the order specified in `design_chains`
- Make sure your sequences match this order
- The PDB file must contain all specified chains

### Compatibility

- **Backward Compatible**: Old single-chain format (`chain_id`) still works
- **Default Behavior**: If no chains specified, uses first chain in PDB
- **Mixed Format**: You can mix single-chain and multi-chain entries in the same training file

## Advanced Usage

### Antibody-Antigen Design

Design antibody CDRs while keeping antigen fixed:

```json
{
    "pdb_file": "antibody_antigen.pdb",
    "preferred_seq": "CDRH3_SEQUENCE",
    "unpreferred_seq": "CDRH3_MUTANT",
    "design_chains": ["H"],
    "fixed_chains": ["A"],
    "description": "Design heavy chain with antigen context"
}
```

### Symmetric Complexes

Design all chains in a symmetric complex:

```json
{
    "pdb_file": "symmetric_trimer.pdb",
    "preferred_seq": "SEQ_A" + "SEQ_B" + "SEQ_C",
    "unpreferred_seq": "SEQ_A_MUT" + "SEQ_B_MUT" + "SEQ_C_MUT",
    "design_chains": ["A", "B", "C"],
    "fixed_chains": [],
    "description": "Co-design all three chains"
}
```

### Partial Interface Design

Design only the interface residues (requires custom masking - future feature):

```json
{
    "pdb_file": "complex.pdb",
    "design_chains": ["B"],
    "fixed_chains": ["A"],
    "design_positions": [10, 11, 12, 45, 46, 47],
    "description": "Design only interface residues (future feature)"
}
```

## Troubleshooting

### Error: "Chain X not found in PDB"

**Cause**: The specified chain ID doesn't exist in the PDB file.

**Solution**: 
- Check your PDB file to see which chains are present
- Use a PDB viewer or: `grep "^ATOM" your_file.pdb | awk '{print $5}' | sort -u`

### Error: "Sequence length mismatch"

**Cause**: The `preferred_seq` length doesn't match the total length of design chains.

**Solution**:
- Count residues in your design chains
- Make sure `preferred_seq` = chain1_seq + chain2_seq + ... (concatenated)
- Don't include fixed chain sequences

### Warning: "Chain X not found"

**Cause**: A chain in `design_chains` or `fixed_chains` is missing from the PDB.

**Solution**: 
- Verify all chains exist in the PDB file
- Remove missing chains from your JSON

### Sequences don't look right

**Cause**: Chain order might be wrong.

**Solution**:
- Ensure sequence concatenation matches the order in `design_chains`
- Example: `design_chains: ["B", "A"]` → `seq = seq_B + seq_A`

## Best Practices

1. **Start Simple**: Test with single-chain design first, then move to multi-chain
2. **Validate PDB Files**: Ensure all chains are present and properly formatted
3. **Check Sequence Lengths**: Always verify sequence lengths match chain lengths
4. **Use Descriptive Names**: Add `description` fields to your training data for clarity
5. **Test Inference**: Run inference on a few samples before large-scale generation
6. **Monitor Training**: Check that loss decreases and sequences look reasonable

## Performance Considerations

- **Memory Usage**: Multi-chain structures use more memory
  - Reduce `--batch_size` if you run out of memory
  - Typical: `--batch_size 1` or `2` for multi-chain
  
- **Training Time**: Multi-chain training is slower
  - More residues to process
  - More complex interactions to learn
  
- **Inference Speed**: Proportional to total number of residues
  - Design chains + fixed chains

## Examples Directory

See `examples/example_multichain_training_data.json` for complete examples.

## Questions?

If you encounter issues:
1. Check this guide
2. Verify your PDB files and chain IDs
3. Test with the provided examples first
4. Check sequence lengths carefully

