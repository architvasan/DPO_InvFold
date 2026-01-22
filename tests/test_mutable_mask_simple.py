#!/usr/bin/env python3
"""Simple test for mutable mask functionality."""

import torch

# Test create_mutable_mask function
def create_mutable_mask(item, seq_length, device='cpu'):
    """Create a mutable position mask from dataset item."""
    mutable_mask = torch.ones(seq_length, dtype=torch.float32, device='cpu')
    
    if 'mutable_positions' in item and item['mutable_positions'] is not None:
        mutable_mask = torch.zeros(seq_length, dtype=torch.float32, device='cpu')
        for pos in item['mutable_positions']:
            if 0 <= pos < seq_length:
                mutable_mask[pos] = 1.0
    
    elif 'mutable_ranges' in item and item['mutable_ranges'] is not None:
        mutable_mask = torch.zeros(seq_length, dtype=torch.float32, device='cpu')
        for start, end in item['mutable_ranges']:
            start = max(0, start)
            end = min(seq_length - 1, end)
            mutable_mask[start:end+1] = 1.0
    
    if str(device) != 'cpu':
        mutable_mask = mutable_mask.to(device)
    
    return mutable_mask


print("Test 1: Default (all mutable)")
item1 = {}
mask1 = create_mutable_mask(item1, 20)
print(f"  Sum: {mask1.sum().item()}/20")
assert mask1.sum() == 20
print("  ✓ PASSED")

print("\nTest 2: Specific positions")
item2 = {'mutable_positions': [10, 11, 12, 13, 14, 15]}
mask2 = create_mutable_mask(item2, 20)
print(f"  Sum: {mask2.sum().item()}/20")
print(f"  Mutable positions: {mask2.nonzero().squeeze().tolist()}")
assert mask2.sum() == 6
print("  ✓ PASSED")

print("\nTest 3: Ranges")
item3 = {'mutable_ranges': [[5, 10], [15, 19]]}
mask3 = create_mutable_mask(item3, 20)
print(f"  Sum: {mask3.sum().item()}/20")
print(f"  Mutable positions: {mask3.nonzero().squeeze().tolist()}")
expected = (10-5+1) + (19-15+1)  # 6 + 5 = 11
assert mask3.sum() == expected
print("  ✓ PASSED")

print("\n" + "="*50)
print("All tests PASSED!")
print("="*50)

