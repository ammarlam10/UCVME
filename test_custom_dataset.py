#!/usr/bin/env python3
"""
Quick test script to verify custom So2Sat dataset loader works correctly.
"""
import sys
import numpy as np
from datasets import get_dataset

# Test the custom dataset
print("Testing custom So2Sat dataset loader...")

dataset_class = get_dataset('so2sat_pop_custom')
print(f"Dataset class: {dataset_class}")

# Create a small test dataset
kwargs = {
    "target_type": ["POP"],
    "mean": 0.5,
    "std": 0.5,
    "normalize_mean": 1085.0,
    "normalize_std": 2800.0
}

try:
    dataset = dataset_class(
        root="/workspace/data/So2Sat_POP",
        split="train",
        **kwargs
    )
    
    print(f"\nDataset loaded successfully!")
    print(f"Total samples: {len(dataset)}")
    
    # Test loading a single sample
    print("\nTesting sample loading...")
    img, label = dataset[0]
    
    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    print(f"Image min/max: {img.min():.4f} / {img.max():.4f}")
    print(f"Label (normalized): {label:.4f}")
    print(f"Label (original): {label * 2800.0 + 1085.0:.2f}")
    
    print("\n✓ Custom dataset test passed!")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
