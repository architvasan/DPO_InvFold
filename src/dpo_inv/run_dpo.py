#!/usr/bin/env python
"""
Generalized Direct Preference Optimization (DPO) training script for ProteinMPNN.

This script fine-tunes ProteinMPNN using DPO with arbitrary protein backbone structures
and sequence preferences. It takes:
- Backbone structures (PDB files)
- Preferred sequences (higher reward)
- Unpreferred sequences (lower reward)

The model learns to generate sequences more similar to preferred examples.
"""

import os
import sys
import argparse
import json
import copy
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import ProteinMPNN components
try:
    from dpo_inv.model import BioMPNN
    from dpo_inv.overwrite import tied_featurize
except ImportError:
    print("Error: Could not import BioMPNN modules. Make sure the libs directory is in your PYTHONPATH.")
    sys.exit(1)

class PreferenceDataset(Dataset):
    """Dataset for DPO training with preferred/unpreferred sequence pairs."""

    def __init__(self, data_file, max_length=500):
        """
        Initialize dataset from a JSON file.

        Args:
            data_file: Path to JSON file with format:
                [
                    {
                        "pdb_file": "path/to/structure.pdb",
                        "preferred_seq": "ACDEFG...",
                        "unpreferred_seq": "ACDEFH...",
                        "chain_id": "A"  # optional, defaults to first chain

                        # Optional multi-chain support:
                        "design_chains": ["B"],  # chains to design
                        "fixed_chains": ["A"],   # chains to keep fixed (context)
                    },
                    ...
                ]
            max_length: Maximum protein length to include
        """
        self.max_length = max_length

        with open(data_file, 'r') as f:
            self.data = json.load(f)

        # Filter by length
        self.data = [d for d in self.data if len(d['preferred_seq']) <= max_length]

        print(f"Loaded {len(self.data)} preference pairs from {data_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_pdb_structure(pdb_file, chain_id=None, design_chains=None, fixed_chains=None):
    """
    Load a PDB structure and extract coordinates and sequence.

    Args:
        pdb_file: Path to PDB file
        chain_id: Single chain to extract (legacy, for backward compatibility)
        design_chains: List of chain IDs to design (will be masked)
        fixed_chains: List of chain IDs to keep fixed (context chains)

    Returns:
        dict with 'seq', 'coords_chain_X', 'name' keys compatible with tied_featurize
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    # Get first model
    model = structure[0]

    # Determine which chains to load
    if design_chains is not None or fixed_chains is not None:
        # Multi-chain mode
        chains_to_load = []
        if design_chains:
            chains_to_load.extend(design_chains)
        if fixed_chains:
            chains_to_load.extend(fixed_chains)

        # Remove duplicates while preserving order
        seen = set()
        chains_to_load = [x for x in chains_to_load if not (x in seen or seen.add(x))]
    elif chain_id is not None:
        # Single chain mode (legacy)
        chains_to_load = [chain_id]
    else:
        # Default: use first chain
        first_chain = list(model.get_chains())[0]
        chains_to_load = [first_chain.id]
    
    # Extract sequence and coordinates for all chains
    aa_3to1 = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }

    batch_entry = {
        'name': os.path.basename(pdb_file),
    }

    all_seq = []

    for chain_id in chains_to_load:
        try:
            chain = model[chain_id]
        except KeyError:
            print(f"Warning: Chain {chain_id} not found in {pdb_file}")
            continue

        seq = []
        coords_n = []
        coords_ca = []
        coords_c = []
        coords_o = []

        for residue in chain:
            if residue.id[0] != ' ':  # Skip hetero residues
                continue

            resname = residue.resname
            if resname not in aa_3to1:
                seq.append('X')
            else:
                seq.append(aa_3to1[resname])

            # Get backbone atoms
            try:
                coords_n.append(residue['N'].coord)
                coords_ca.append(residue['CA'].coord)
                coords_c.append(residue['C'].coord)
                coords_o.append(residue['O'].coord)
            except KeyError:
                # Missing backbone atoms - use CA position for all
                ca_coord = residue['CA'].coord
                coords_n.append(ca_coord)
                coords_ca.append(ca_coord)
                coords_c.append(ca_coord)
                coords_o.append(ca_coord)

        seq_str = ''.join(seq)
        all_seq.append(seq_str)

        # Add chain-specific data
        batch_entry[f'seq_chain_{chain_id}'] = seq_str
        batch_entry[f'coords_chain_{chain_id}'] = {
            f'N_chain_{chain_id}': coords_n,
            f'CA_chain_{chain_id}': coords_ca,
            f'C_chain_{chain_id}': coords_c,
            f'O_chain_{chain_id}': coords_o,
        }

    # Add full sequence (concatenation of all chains)
    batch_entry['seq'] = ''.join(all_seq)

    # Add multi-chain information for tied_featurize
    if design_chains is not None:
        batch_entry['masked_list'] = design_chains
    else:
        # Default: design the first/only chain
        batch_entry['masked_list'] = [chains_to_load[0]]

    if fixed_chains is not None:
        batch_entry['visible_list'] = fixed_chains
    else:
        batch_entry['visible_list'] = []

    batch_entry['num_of_chains'] = len(chains_to_load)

    return batch_entry


def seq_to_tensor(seq, device='cpu'):
    """Convert amino acid sequence to tensor indices.

    Always creates tensor on CPU first, then moves to device to avoid XPU issues.
    """
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    indices = [alphabet.index(aa) if aa in alphabet else alphabet.index('X') for aa in seq]
    tensor = torch.tensor(indices, dtype=torch.long, device='cpu')
    if str(device) != 'cpu':
        tensor = tensor.to(device)
    return tensor


def collate_preference_batch(batch_list, device='cpu'):
    """
    Collate a batch of preference pairs into tensors.

    Args:
        batch_list: List of dicts with 'pdb_file', 'preferred_seq', 'unpreferred_seq'
                    Optional: 'chain_id', 'design_chains', 'fixed_chains'
        device: Device to place tensors on

    Returns:
        Tuple of (structure_batch, S_preferred, S_unpreferred, mask)
    """
    # Load structures
    pdb_batch = []
    for item in batch_list:
        pdb_entry = load_pdb_structure(
            item['pdb_file'],
            chain_id=item.get('chain_id'),
            design_chains=item.get('design_chains'),
            fixed_chains=item.get('fixed_chains')
        )
        pdb_batch.append(pdb_entry)

    # ALWAYS featurize on CPU first to avoid XPU segfaults
    # tied_featurize creates tensors with .to(device) which can cause issues on XPU
    structure_batch = list(tied_featurize(pdb_batch, 'cpu', None))

    # Move all tensors to target device after featurization
    if str(device) != 'cpu':
        structure_batch_moved = []
        for item in structure_batch:
            if isinstance(item, torch.Tensor):
                structure_batch_moved.append(item.to(device))
            elif isinstance(item, list):
                # Keep lists as-is (like letter_list_list, visible_list_list, etc.)
                structure_batch_moved.append(item)
            else:
                structure_batch_moved.append(item)
        structure_batch = tuple(structure_batch_moved)
    
    # Get the actual structure length from the batch
    # The structure determines the max length, not the sequences
    # structure_batch is a tuple: (X, S, mask, lengths, ...)
    # X is the first element with shape [B, L, 4, 3] or [B, L, 1, 3]
    if isinstance(structure_batch, (list, tuple)):
        X_temp = structure_batch[0]  # X is the first element
        structure_max_len = X_temp.shape[1]  # [B, L, ...]
    else:
        # Fallback to sequence-based max length
        structure_max_len = max(len(item['preferred_seq']) for item in batch_list)

    # Convert sequences to tensors
    S_preferred_list = []
    S_unpreferred_list = []

    for item in batch_list:
        S_pref = seq_to_tensor(item['preferred_seq'], device)
        S_unpref = seq_to_tensor(item['unpreferred_seq'], device)

        # Pad or truncate to match structure length
        if len(S_pref) < structure_max_len:
            S_pref = F.pad(S_pref, (0, structure_max_len - len(S_pref)), value=0)
        elif len(S_pref) > structure_max_len:
            S_pref = S_pref[:structure_max_len]

        if len(S_unpref) < structure_max_len:
            S_unpref = F.pad(S_unpref, (0, structure_max_len - len(S_unpref)), value=0)
        elif len(S_unpref) > structure_max_len:
            S_unpref = S_unpref[:structure_max_len]

        S_preferred_list.append(S_pref)
        S_unpreferred_list.append(S_unpref)

    S_preferred = torch.stack(S_preferred_list, dim=0)
    S_unpreferred = torch.stack(S_unpreferred_list, dim=0)
    
    return structure_batch, S_preferred, S_unpreferred


def compute_dpo_loss(model, base_model, structure_batch, S_preferred, S_unpreferred, 
                     beta=0.1, temperature=0.1, device='cpu'):
    """
    Compute DPO loss for a batch.
    
    Args:
        model: Fine-tuned model (being trained)
        base_model: Reference model (frozen)
        structure_batch: Featurized structure batch from tied_featurize
        S_preferred: Preferred sequences [B, L]
        S_unpreferred: Unpreferred sequences [B, L]
        beta: DPO beta parameter (KL penalty weight)
        temperature: Sampling temperature
        device: Device
    
    Returns:
        loss: DPO loss scalar
    """
    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, \
        visible_list_list, masked_list_list, masked_chain_length_list_list, \
        chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, \
        tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, \
        bias_by_res_all, tied_beta = structure_batch
    
    B = X.shape[0]
    max_length = S_preferred.shape[1]
    decoding_order = torch.arange(max_length).repeat(B, 1).to(device)

    # Debug: Check sequence indices are valid
    if S_preferred.max() > 20 or S_unpreferred.max() > 20:
        print(f"WARNING: Invalid sequence indices detected!")
        print(f"S_preferred max: {S_preferred.max()}, min: {S_preferred.min()}")
        print(f"S_unpreferred max: {S_unpreferred.max()}, min: {S_unpreferred.min()}")
        print(f"Expected range: 0-20 (21 amino acids)")

    # Forward pass through fine-tuned model
    log_probs_pref = model(X, S_preferred, mask, chain_M, residue_idx,
                           chain_encoding_all, decoding_order,
                           omit_AA_mask=omit_AA_mask, temperature=temperature)
    
    log_probs_unpref = model(X, S_unpreferred, mask, chain_M, residue_idx,
                             chain_encoding_all, decoding_order,
                             omit_AA_mask=omit_AA_mask, temperature=temperature)
    
    # Forward pass through base model (no gradients)
    with torch.no_grad():
        base_log_probs_pref = base_model(X, S_preferred, mask, chain_M, residue_idx,
                                         chain_encoding_all, decoding_order,
                                         omit_AA_mask=omit_AA_mask, temperature=temperature)
        
        base_log_probs_unpref = base_model(X, S_unpreferred, mask, chain_M, residue_idx,
                                           chain_encoding_all, decoding_order,
                                           omit_AA_mask=omit_AA_mask, temperature=temperature)
    
    # Gather log probabilities for actual sequences
    # Check vocab size and clamp indices to valid range
    vocab_size = log_probs_pref.shape[-1]

    # Debug output
    print(f"Model output vocab size: {vocab_size}")
    print(f"log_probs_pref shape: {log_probs_pref.shape}")
    print(f"S_preferred shape: {S_preferred.shape}, max: {S_preferred.max()}, min: {S_preferred.min()}")

    # Clamp indices to valid range to avoid out-of-bounds errors
    S_preferred_clamped = torch.clamp(S_preferred, 0, vocab_size - 1)
    S_unpreferred_clamped = torch.clamp(S_unpreferred, 0, vocab_size - 1)

    log_pi_theta_pref = log_probs_pref.gather(-1, S_preferred_clamped.unsqueeze(-1)).squeeze(-1)
    log_pi_theta_unpref = log_probs_unpref.gather(-1, S_unpreferred_clamped.unsqueeze(-1)).squeeze(-1)

    log_pi_base_pref = base_log_probs_pref.gather(-1, S_preferred_clamped.unsqueeze(-1)).squeeze(-1)
    log_pi_base_unpref = base_log_probs_unpref.gather(-1, S_unpreferred_clamped.unsqueeze(-1)).squeeze(-1)

    # Mask out padded positions using chain_M (1.0 for positions to predict, 0.0 for padding)
    log_pi_theta_pref = (log_pi_theta_pref * chain_M).sum(1)
    log_pi_theta_unpref = (log_pi_theta_unpref * chain_M).sum(1)

    log_pi_base_pref = (log_pi_base_pref * chain_M).sum(1)
    log_pi_base_unpref = (log_pi_base_unpref * chain_M).sum(1)
    
    # DPO loss: -log(sigmoid(beta * (log_ratio_preferred - log_ratio_unpreferred)))
    loss = -torch.log(torch.sigmoid(
        beta * (log_pi_theta_pref - log_pi_base_pref - log_pi_theta_unpref + log_pi_base_unpref)
    ))
    
    return loss.mean()


def train_epoch(model, base_model, dataloader, optimizer, beta, temperature, device):
    """Train for one epoch."""
    model.train()
    base_model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        structure_batch, S_pref, S_unpref = collate_preference_batch(batch, device)
        
        optimizer.zero_grad()
        loss = compute_dpo_loss(model, base_model, structure_batch, S_pref, S_unpref,
                               beta, temperature, device)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(model, base_model, dataloader, beta, temperature, device):
    """Validate the model."""
    model.eval()
    base_model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            structure_batch, S_pref, S_unpref = collate_preference_batch(batch, device)
            loss = compute_dpo_loss(model, base_model, structure_batch, S_pref, S_unpref,
                                   beta, temperature, device)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main(args):
    """Main training function."""
    # Set device with XPU support
    if not args.no_cuda:
        try:
            import intel_extension_for_pytorch as ipex
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                device = torch.device('xpu')
                print(f"Using device: {device} (Intel XPU)")
                print(f"XPU device count: {torch.xpu.device_count()}")
                print(f"IPEX version: {ipex.__version__}")
                print()
                print("XPU Optimization: All tensors created on CPU first, then moved to XPU")
                print("If you encounter segfaults:")
                print("  1. Source scripts/setup_xpu_env.sh before running")
                print("  2. Reduce --batch_size to 1")
                print("  3. See XPU_TROUBLESHOOTING.md")
                print()
            elif torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"Using device: {device}")
            else:
                device = torch.device('cpu')
                print(f"Using device: {device}")
        except ImportError:
            # IPEX not available, fall back to CUDA or CPU
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"Using device: {device}")
            else:
                device = torch.device('cpu')
                print(f"Using device: {device}")
        except Exception as e:
            print(f"Warning: Error initializing GPU: {e}")
            print("Falling back to CPU mode")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print(f"Using device: {device} (forced CPU mode)")

    # Set paths for ProteinMPNN models
    BioMPNN.base_model_pt_dir_path = args.base_model_dir
    BioMPNN.base_model_yaml_dir_path = args.base_model_config_dir

    # Load base model
    print(f"Loading base model: {args.base_model_name}")
    base_model = BioMPNN.from_file(args.base_model_name)
    base_model.eval()
    base_model.to(device)

    # Create fine-tuned model (copy of base)
    print("Creating fine-tuned model...")
    model = copy.deepcopy(base_model)
    model.to(device)

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Load datasets
    print("Loading datasets...")
    train_dataset = PreferenceDataset(args.train_data, max_length=args.max_length)

    if args.val_data:
        val_dataset = PreferenceDataset(args.val_data, max_length=args.max_length)
    else:
        # Split train dataset
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

    # Custom collate function to handle our data format
    def identity_collate(batch):
        """Identity collate function - just returns the batch as-is."""
        return batch

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=0, collate_fn=identity_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0, collate_fn=identity_collate)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training config
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    best_val_loss = float('inf')

    print(f"\nStarting training for {args.num_epochs} epochs...")
    print(f"Beta: {args.beta}, Temperature: {args.temperature}, LR: {args.lr}")

    for epoch in range(args.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*60}")

        # Train
        train_loss = train_epoch(model, base_model, train_loader, optimizer,
                                args.beta, args.temperature, device)

        # Validate
        val_loss = validate(model, base_model, val_loader,
                          args.beta, args.temperature, device)

        print(f"\nEpoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save checkpoint
        checkpoint_path = output_dir / f'checkpoint_epoch_{epoch + 1}.pt'
        model.save_to_file(str(checkpoint_path))
        print(f"Saved checkpoint to {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / 'best_model.pt'
            model.save_to_file(str(best_path))
            print(f"New best model! Saved to {best_path}")

        # Save latest
        latest_path = output_dir / 'latest_model.pt'
        model.save_to_file(str(latest_path))

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Data arguments
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data JSON file')
    parser.add_argument('--val_data', type=str, default=None,
                       help='Path to validation data JSON file (if None, splits train data)')

    # Model arguments
    parser.add_argument('--base_model_name', type=str, default='v_48_020',
                       help='Name of base ProteinMPNN model')
    parser.add_argument('--base_model_dir', type=str,
                       default='data/input/BioMPNN/soluble_model_weights',
                       help='Directory containing base model .pt files (use soluble_model_weights for soluble proteins)')
    parser.add_argument('--base_model_config_dir', type=str,
                       default='data/input/BioMPNN/base_hparams',
                       help='Directory containing base model config .yaml files')

    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (number of protein structures per batch)')
    parser.add_argument('--lr', type=float, default=3e-7,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay for optimizer')
    parser.add_argument('--beta', type=float, default=0.1,
                       help='DPO beta parameter (KL penalty weight)')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Sampling temperature')
    parser.add_argument('--max_length', type=int, default=500,
                       help='Maximum protein length to include')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output/dpo_training',
                       help='Directory to save trained models and logs')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable GPU acceleration (CUDA/XPU) and force CPU mode')

    args = parser.parse_args()

    main(args)

