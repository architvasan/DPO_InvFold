#!/usr/bin/env python
"""
Distributed Data Parallel (DDP) version of DPO training for ProteinMPNN.

This script uses PyTorch DDP for multi-GPU/multi-node training on HPC systems like Aurora.

Usage:
    # Single node, multiple GPUs
    torchrun --nproc_per_node=4 -m dpo_inv.run_dpo_ddp --train_data data.json --output_dir outputs/

    # Multi-node (e.g., on Aurora with SLURM)
    srun python -m dpo_inv.run_dpo_ddp --train_data data.json --output_dir outputs/
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# Import ProteinMPNN components
try:
    from dpo_inv.model import BioMPNN
    from dpo_inv.overwrite import tied_featurize
except ImportError:
    print("Error: Could not import BioMPNN modules. Make sure the libs directory is in your PYTHONPATH.")
    sys.exit(1)


def setup_distributed():
    """Initialize distributed training environment."""
    # Check if running with torchrun or SLURM
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # torchrun sets these
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        # SLURM environment
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        
        # Set environment variables for PyTorch
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
    else:
        # Single process
        rank = 0
        world_size = 1
        local_rank = 0
    
    return rank, world_size, local_rank


def init_process_group(backend='nccl'):
    """Initialize the distributed process group."""
    rank, world_size, local_rank = setup_distributed()
    
    if world_size > 1:
        # Initialize process group
        if backend == 'xpu':
            # For Intel XPU
            try:
                import intel_extension_for_pytorch as ipex
                backend = 'ccl'  # Use oneCCL backend for XPU
            except ImportError:
                print("Warning: Intel Extension for PyTorch not found, falling back to gloo")
                backend = 'gloo'
        
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'
        
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device(local_rank, no_cuda=False):
    """Get the appropriate device for this process."""
    if no_cuda:
        return torch.device('cpu')
    
    # Try XPU first (Intel GPU)
    try:
        import intel_extension_for_pytorch as ipex
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            device = torch.device(f'xpu:{local_rank}')
            torch.xpu.set_device(device)
            return device
    except (ImportError, AttributeError):
        pass
    
    # Try CUDA
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        return device
    
    # Fallback to CPU
    return torch.device('cpu')


def print_rank0(message, rank):
    """Print only from rank 0."""
    if rank == 0:
        print(message)


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
                    },
                    ...
                ]
            max_length: Maximum sequence length
        """
        self.max_length = max_length

        with open(data_file, 'r') as f:
            self.data = json.load(f)

        # Validate data
        for i, item in enumerate(self.data):
            if 'pdb_file' not in item:
                raise ValueError(f"Item {i} missing 'pdb_file'")
            if 'preferred_seq' not in item:
                raise ValueError(f"Item {i} missing 'preferred_seq'")
            if 'unpreferred_seq' not in item:
                raise ValueError(f"Item {i} missing 'unpreferred_seq'")

            # Check sequences are same length
            if len(item['preferred_seq']) != len(item['unpreferred_seq']):
                raise ValueError(
                    f"Item {i}: preferred and unpreferred sequences must have same length. "
                    f"Got {len(item['preferred_seq'])} vs {len(item['unpreferred_seq'])}"
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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


def identity_collate(batch):
    """Identity collate function - just returns the batch as-is."""
    return batch


def collate_preference_batch(batch_list, device='cpu'):
    """
    Collate a batch of preference pairs.

    Args:
        batch_list: List of dicts with 'pdb_file', 'preferred_seq', 'unpreferred_seq'
        device: Device to move tensors to

    Returns:
        structure_batch: Featurized structures from tied_featurize
        S_preferred: Tensor of preferred sequences [B, L]
        S_unpreferred: Tensor of unpreferred sequences [B, L]
    """
    from Bio.PDB import PDBParser

    # Extract PDB files and chain IDs
    pdb_files = []
    chain_ids = []
    for item in batch_list:
        pdb_files.append(item['pdb_file'])
        chain_ids.append(item.get('chain_id', None))

    # Parse PDB files to get backbone coordinates
    parser = PDBParser(QUIET=True)
    pdb_batch = []

    for pdb_file, chain_id in zip(pdb_files, chain_ids):
        structure = parser.get_structure('protein', pdb_file)

        # Get the specified chain or first chain
        if chain_id:
            chain = structure[0][chain_id]
        else:
            chain = list(structure[0].get_chains())[0]

        # Extract backbone coordinates (N, CA, C, O)
        coords = []
        for residue in chain:
            if residue.id[0] != ' ':  # Skip hetero residues
                continue
            try:
                n = residue['N'].get_coord()
                ca = residue['CA'].get_coord()
                c = residue['C'].get_coord()
                o = residue['O'].get_coord()
                coords.append([n, ca, c, o])
            except KeyError:
                # Skip residues missing backbone atoms
                continue

        coords = np.array(coords)  # Shape: [L, 4, 3]
        pdb_batch.append(coords)

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
                     beta=0.1, temperature=0.1, device='cpu', verbose=False):
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
        verbose: If True, print diagnostic information

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
    # Clamp indices to valid range to avoid out-of-bounds errors
    vocab_size = log_probs_pref.shape[-1]
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

    # Compute log ratios
    log_ratio_pref = log_pi_theta_pref - log_pi_base_pref
    log_ratio_unpref = log_pi_theta_unpref - log_pi_base_unpref

    # DPO loss: -log(sigmoid(beta * (log_ratio_preferred - log_ratio_unpreferred)))
    # This encourages the model to increase the ratio for preferred sequences
    # and decrease it for unpreferred sequences
    logits = beta * (log_ratio_pref - log_ratio_unpref)
    loss = -torch.nn.functional.logsigmoid(logits)

    # Diagnostic output
    if verbose:
        print(f"\n=== DPO Loss Diagnostics ===")
        print(f"log_pi_theta_pref:  mean={log_pi_theta_pref.mean().item():.2f}, std={log_pi_theta_pref.std().item():.2f}")
        print(f"log_pi_theta_unpref: mean={log_pi_theta_unpref.mean().item():.2f}, std={log_pi_theta_unpref.std().item():.2f}")
        print(f"log_pi_base_pref:   mean={log_pi_base_pref.mean().item():.2f}, std={log_pi_base_pref.std().item():.2f}")
        print(f"log_pi_base_unpref:  mean={log_pi_base_unpref.mean().item():.2f}, std={log_pi_base_unpref.std().item():.2f}")
        print(f"log_ratio_pref:     mean={log_ratio_pref.mean().item():.2f}, std={log_ratio_pref.std().item():.2f}")
        print(f"log_ratio_unpref:   mean={log_ratio_unpref.mean().item():.2f}, std={log_ratio_unpref.std().item():.2f}")
        print(f"logits (beta * diff): mean={logits.mean().item():.2f}, std={logits.std().item():.2f}")
        print(f"loss:               mean={loss.mean().item():.4f}")
        print(f"beta: {beta}, temperature: {temperature}")
        print(f"===========================\n")

    return loss.mean()


def train_epoch(model, base_model, dataloader, optimizer, beta, temperature, device, rank):
    """Train for one epoch."""
    model.train()
    base_model.eval()

    total_loss = 0.0
    num_batches = 0

    # Only show progress bar on rank 0
    if rank == 0:
        pbar = tqdm(dataloader, desc="Training")
    else:
        pbar = dataloader

    for batch in pbar:
        structure_batch, S_pref, S_unpref = collate_preference_batch(batch, device)

        optimizer.zero_grad()
        # Print diagnostics on first batch only (rank 0)
        verbose = (num_batches == 0 and rank == 0)
        loss = compute_dpo_loss(model, base_model, structure_batch, S_pref, S_unpref,
                               beta, temperature, device, verbose=verbose)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Update progress bar with current loss (rank 0 only)
        if rank == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss/num_batches:.4f}'})

    # Gather losses from all ranks
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    if dist.is_initialized():
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()

    return avg_loss


def validate(model, base_model, dataloader, beta, temperature, device, rank):
    """Validate the model."""
    model.eval()
    base_model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        # Only show progress bar on rank 0
        if rank == 0:
            pbar = tqdm(dataloader, desc="Validation")
        else:
            pbar = dataloader

        for batch in pbar:
            structure_batch, S_pref, S_unpref = collate_preference_batch(batch, device)
            loss = compute_dpo_loss(model, base_model, structure_batch, S_pref, S_unpref,
                                   beta, temperature, device)
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar with current loss (rank 0 only)
            if rank == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss/num_batches:.4f}'})

    # Gather losses from all ranks
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    if dist.is_initialized():
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()

    return avg_loss


def main():
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
                       help='Directory containing base model weights')

    # Training arguments
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save checkpoints and logs')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size per GPU')
    parser.add_argument('--lr', '--learning_rate', type=float, default=3e-7, dest='lr',
                       help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.1,
                       help='DPO beta parameter (KL penalty weight)')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Sampling temperature')

    # Device arguments
    parser.add_argument('--no_cuda', action='store_true',
                       help='Force CPU mode')
    parser.add_argument('--backend', type=str, default='auto',
                       choices=['auto', 'nccl', 'gloo', 'ccl', 'xpu'],
                       help='DDP backend (auto will choose based on device)')

    args = parser.parse_args()

    # Initialize distributed training
    backend = args.backend
    if backend == 'auto':
        backend = 'nccl'  # Will be changed to ccl/gloo if needed

    rank, world_size, local_rank = init_process_group(backend=backend)

    print_rank0(f"Initialized DDP: rank={rank}, world_size={world_size}, local_rank={local_rank}", rank)

    # Get device
    device = get_device(local_rank, args.no_cuda)
    print_rank0(f"Using device: {device}", rank)

    # Create output directory (rank 0 only)
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Wait for rank 0 to create directory
    if dist.is_initialized():
        dist.barrier()

    # Load base model
    print_rank0("Loading base model...", rank)
    base_model_path = Path(args.base_model_dir) / f'{args.base_model_name}.pt'
    base_model = BioMPNN(device=device)
    base_model.load_from_file(str(base_model_path))
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False

    # Create fine-tuned model (copy of base model)
    print_rank0("Creating fine-tuned model...", rank)
    model = BioMPNN(device=device)
    model.load_from_file(str(base_model_path))
    model.train()

    # Wrap model in DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if str(device).startswith('cuda') else None)

    # Load datasets
    print_rank0("Loading datasets...", rank)
    train_dataset = PreferenceDataset(args.train_data)

    if args.val_data:
        val_dataset = PreferenceDataset(args.val_data)
        print_rank0(f"Loaded {len(train_dataset)} training and {len(val_dataset)} validation pairs", rank)
    else:
        # Split train dataset
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        print_rank0(f"Loaded {len(train_dataset) + len(val_dataset)} preference pairs", rank)
        print_rank0(f"Train size: {train_size}, Val size: {val_size}", rank)

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    ) if world_size > 1 else None

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if world_size > 1 else None

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=identity_collate,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=identity_collate,
        num_workers=0,
        pin_memory=False
    )

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_loss = float('inf')

    print_rank0(f"\nStarting training for {args.num_epochs} epochs...", rank)
    print_rank0(f"Beta: {args.beta}, Temperature: {args.temperature}, LR: {args.lr}", rank)
    print_rank0(f"Batch size per GPU: {args.batch_size}, Total batch size: {args.batch_size * world_size}", rank)

    for epoch in range(args.num_epochs):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        print_rank0(f"\n{'='*60}", rank)
        print_rank0(f"Epoch {epoch + 1}/{args.num_epochs}", rank)
        print_rank0(f"{'='*60}", rank)

        # Train
        train_loss = train_epoch(model, base_model, train_loader, optimizer,
                                args.beta, args.temperature, device, rank)

        # Validate
        val_loss = validate(model, base_model, val_loader,
                          args.beta, args.temperature, device, rank)

        print_rank0(f"\n{'='*60}", rank)
        print_rank0(f"Epoch {epoch + 1} Summary:", rank)
        print_rank0(f"  Train Loss: {train_loss:.4f}", rank)
        print_rank0(f"  Val Loss:   {val_loss:.4f}", rank)
        if val_loss < best_val_loss:
            print_rank0(f"  ✓ New best validation loss! (previous: {best_val_loss:.4f})", rank)
        print_rank0(f"{'='*60}", rank)

        # Save checkpoint (rank 0 only)
        if rank == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch + 1}.pt'
            # Unwrap DDP if needed
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_to_file(str(checkpoint_path))
            print(f"\nSaved checkpoint to {checkpoint_path}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = output_dir / 'best_model.pt'
                model_to_save.save_to_file(str(best_path))
                print(f"Saved best model to {best_path}")

            # Save latest
            latest_path = output_dir / 'latest_model.pt'
            model_to_save.save_to_file(str(latest_path))

        # Wait for rank 0 to finish saving
        if dist.is_initialized():
            dist.barrier()

    print_rank0(f"\n{'='*60}", rank)
    print_rank0("Training complete!", rank)
    print_rank0(f"Best validation loss: {best_val_loss:.4f}", rank)
    print_rank0(f"Models saved to: {output_dir}", rank)
    print_rank0(f"{'='*60}", rank)

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()

