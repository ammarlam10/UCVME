#!/usr/bin/env python3
"""
Split TEST set into 50% VAL and 50% TEST in FileList.csv

This script modifies FileList.csv to split the TEST samples into:
- 50% VAL (for validation during training)
- 50% TEST (for final evaluation)

Usage:
    python scripts/split_test_to_val.py \
        --data_dir /work/ammar/sslrp/data/So2Sat_POP \
        --backup
"""

import os
import pandas as pd
import argparse
import shutil
from pathlib import Path


def split_test_to_val(data_dir, file_list_name="FileList.csv", backup=True, seed=0):
    """
    Split TEST samples into 50% VAL and 50% TEST.
    
    Args:
        data_dir: Directory containing FileList.csv
        file_list_name: Name of the FileList CSV file
        backup: Whether to create a backup of original file
        seed: Random seed for reproducibility
    """
    import numpy as np
    np.random.seed(seed)
    
    file_list_path = os.path.join(data_dir, file_list_name)
    
    if not os.path.exists(file_list_path):
        raise FileNotFoundError(f"FileList not found: {file_list_path}")
    
    # Create backup if requested
    if backup:
        backup_path = file_list_path + ".backup"
        if not os.path.exists(backup_path):
            shutil.copy2(file_list_path, backup_path)
            print(f"Created backup: {backup_path}")
        else:
            print(f"Backup already exists: {backup_path}")
    
    # Read the CSV
    print(f"Reading {file_list_path}...")
    df = pd.read_csv(file_list_path)
    
    # Check current splits
    if 'SPLIT' not in df.columns:
        raise ValueError("FileList.csv must have a 'SPLIT' column")
    
    print(f"\nCurrent split distribution:")
    print(df['SPLIT'].value_counts())
    
    # Get TEST samples
    test_mask = df['SPLIT'].str.upper() == 'TEST'
    test_df = df[test_mask].copy()
    
    if len(test_df) == 0:
        print("Warning: No TEST samples found. Nothing to split.")
        return
    
    print(f"\nTEST samples to split: {len(test_df)}")
    
    # Shuffle TEST samples
    test_indices = test_df.index.tolist()
    np.random.shuffle(test_indices)
    
    # Split 50/50
    split_point = len(test_indices) // 2
    val_indices = test_indices[:split_point]
    test_indices_new = test_indices[split_point:]
    
    # Update splits
    df.loc[val_indices, 'SPLIT'] = 'VAL'
    df.loc[test_indices_new, 'SPLIT'] = 'TEST'
    
    print(f"  → VAL: {len(val_indices)} samples")
    print(f"  → TEST: {len(test_indices_new)} samples")
    
    # Save updated CSV
    df.to_csv(file_list_path, index=False)
    print(f"\nUpdated {file_list_path}")
    print(f"\nNew split distribution:")
    print(df['SPLIT'].value_counts())
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split TEST set into 50% VAL and 50% TEST")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing FileList.csv")
    parser.add_argument("--file_list", type=str, default="FileList.csv",
                        help="Name of FileList CSV file (default: FileList.csv)")
    parser.add_argument("--backup", action="store_true", default=True,
                        help="Create backup of original file (default: True)")
    parser.add_argument("--no-backup", dest="backup", action="store_false",
                        help="Don't create backup")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility (default: 0)")
    
    args = parser.parse_args()
    
    split_test_to_val(args.data_dir, args.file_list, args.backup, args.seed)
