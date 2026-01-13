#!/usr/bin/env python
"""
Inference script for fine-tuned BioMPNN models.

This script loads a fine-tuned BioMPNN model and generates sequences for input protein backbones.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from dpo_inv.model import BioMPNN
from dpo_inv.overwrite import tied_featurize
from dpo_inv.run_dpo import load_pdb_structure


def generate_sequences(model, pdb_file, chain_id=None, design_chains=None,
                       fixed_chains=None, num_samples=10, temperature=0.1, device='cpu'):
    """
    Generate sequences for a given protein backbone.

    Args:
        model: BioMPNN model
        pdb_file: Path to PDB file
        chain_id: Single chain to design (legacy, for backward compatibility)
        design_chains: List of chain IDs to design (will be masked)
        fixed_chains: List of chain IDs to keep fixed (context chains)
        num_samples: Number of sequences to generate
        temperature: Sampling temperature
        device: Device to run on

    Returns:
        List of generated sequences
    """
    # Load structure with multi-chain support
    pdb_entry = load_pdb_structure(pdb_file, chain_id=chain_id,
                                   design_chains=design_chains,
                                   fixed_chains=fixed_chains)
    
    # Featurize
    batch = [pdb_entry]
    structure_batch = tied_featurize(batch, device, None)
    
    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, \
        visible_list_list, masked_list_list, masked_chain_length_list_list, \
        chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, \
        tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, \
        bias_by_res_all, tied_beta = structure_batch
    
    # Prepare for sampling
    randn = torch.randn(chain_M.shape, device=device)
    
    # Alphabet for decoding
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    
    sequences = []

    # Prepare omit_AAs_np and bias_AAs_np as numpy arrays
    # These control which amino acids to omit/bias during sampling
    omit_AAs_np = np.zeros(21)  # No amino acids omitted by default
    bias_AAs_np = np.zeros(21)  # No bias by default

    model.eval()
    with torch.no_grad():
        for i in range(num_samples):
            # Sample sequence
            S_sample, log_probs = model.sample(
                X, randn, S, chain_M, chain_encoding_all,
                residue_idx, mask=mask, chain_M_pos=chain_M_pos, temperature=temperature,
                omit_AA_mask=omit_AA_mask, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np
            )

            # Decode sequence
            seq_str = ''.join([alphabet[aa] for aa in S_sample[0].cpu().numpy() if aa < len(alphabet)])
            sequences.append(seq_str)

            print(f"Sample {i+1}/{num_samples}: {seq_str[:50]}..." if len(seq_str) > 50 else f"Sample {i+1}/{num_samples}: {seq_str}")

    return sequences


def main(args):
    """Main inference function."""
    # Set device with XPU support
    if not args.no_cuda:
        try:
            import intel_extension_for_pytorch as ipex
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                device = torch.device('xpu')
                print(f"Using device: {device} (Intel XPU)")
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
    else:
        device = torch.device('cpu')
        print(f"Using device: {device} (forced CPU mode)")
    
    # Set paths for base models (needed for loading)
    BioMPNN.base_model_pt_dir_path = args.base_model_dir
    BioMPNN.base_model_yaml_dir_path = args.base_model_config_dir
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = BioMPNN.from_file(args.model_path)
    model.eval()
    model.to(device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process input PDB files
    if args.pdb_file:
        pdb_files = [args.pdb_file]
    elif args.pdb_list:
        with open(args.pdb_list, 'r') as f:
            pdb_files = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("Must provide either --pdb_file or --pdb_list")
    
    all_results = {}
    
    for pdb_file in pdb_files:
        print(f"\n{'='*60}")
        print(f"Processing: {pdb_file}")
        print(f"{'='*60}")
        
        if not os.path.exists(pdb_file):
            print(f"Warning: File not found: {pdb_file}")
            continue
        
        # Generate sequences
        sequences = generate_sequences(
            model, pdb_file,
            chain_id=args.chain_id,
            design_chains=args.design_chains,
            fixed_chains=args.fixed_chains,
            num_samples=args.num_samples,
            temperature=args.temperature,
            device=device
        )

        # Store results
        pdb_name = os.path.basename(pdb_file)
        all_results[pdb_name] = {
            'pdb_file': pdb_file,
            'chain_id': args.chain_id,
            'design_chains': args.design_chains,
            'fixed_chains': args.fixed_chains,
            'sequences': sequences,
            'num_samples': args.num_samples,
            'temperature': args.temperature
        }
        
        # Save individual results
        output_file = output_dir / f"{Path(pdb_file).stem}_sequences.json"
        with open(output_file, 'w') as f:
            json.dump(all_results[pdb_name], f, indent=2)
        print(f"\nSaved sequences to: {output_file}")
        
        # Also save as FASTA
        fasta_file = output_dir / f"{Path(pdb_file).stem}_sequences.fasta"
        with open(fasta_file, 'w') as f:
            for i, seq in enumerate(sequences):
                f.write(f">{pdb_name}_sample_{i+1}\n{seq}\n")
        print(f"Saved FASTA to: {fasta_file}")
    
    # Save all results
    all_results_file = output_dir / 'all_results.json'
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"All results saved to: {all_results_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to fine-tuned model checkpoint (.pt file)')
    parser.add_argument('--pdb_file', type=str, default=None,
                       help='Path to single PDB file')
    parser.add_argument('--pdb_list', type=str, default=None,
                       help='Path to text file with list of PDB files (one per line)')
    parser.add_argument('--chain_id', type=str, default=None,
                       help='Chain ID to design (legacy, for single chain)')
    parser.add_argument('--design_chains', type=str, nargs='+', default=None,
                       help='Chain IDs to design (e.g., --design_chains B C)')
    parser.add_argument('--fixed_chains', type=str, nargs='+', default=None,
                       help='Chain IDs to keep fixed as context (e.g., --fixed_chains A)')

    # Model paths (for loading base model configs)
    parser.add_argument('--base_model_dir', type=str,
                       default='data/input/BioMPNN/soluble_model_weights',
                       help='Directory containing base model .pt files (use soluble_model_weights for soluble proteins)')
    parser.add_argument('--base_model_config_dir', type=str,
                       default='data/input/BioMPNN/base_hparams',
                       help='Directory containing base model config .yaml files')
    
    # Sampling arguments
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of sequences to generate per structure')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Sampling temperature')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs/inference',
                       help='Directory to save generated sequences')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable GPU acceleration (CUDA/XPU) and force CPU mode')
    
    args = parser.parse_args()
    main(args)

