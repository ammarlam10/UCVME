#!/usr/bin/env python3
"""
Create SSL split file for semi-supervised learning.
This script creates FileList_ssl_{rd_label}_{rd_unlabel}.csv with SSL_SPLIT column.
"""

import os
import pandas as pd
import numpy as np
import argparse


def create_ssl_split(data_dir, file_list_name, rd_label, rd_unlabel, seed=0):
    """
    Create SSL split from FileList.csv.
    
    Args:
        data_dir: Directory containing FileList.csv
        file_list_name: Name of the base FileList CSV file
        rd_label: Number of labeled samples
        rd_unlabel: Number of unlabeled samples
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    file_list_path = os.path.join(data_dir, file_list_name)
    
    if not os.path.exists(file_list_path):
        raise FileNotFoundError(f"FileList not found: {file_list_path}")
    
    # Read the CSV
    print(f"Reading {file_list_path}...")
    df = pd.read_csv(file_list_path)
    
    # Check current splits
    if 'SPLIT' not in df.columns:
        raise ValueError("FileList.csv must have a 'SPLIT' column")
    
    print(f"\nOriginal split distribution:")
    print(df['SPLIT'].value_counts())
    
    # Get TRAIN samples only
    train_df = df[df['SPLIT'] == 'TRAIN'].copy()
    other_df = df[df['SPLIT'] != 'TRAIN'].copy()
    
    print(f"\nTrain samples: {len(train_df)}")
    print(f"Other samples (val/test): {len(other_df)}")
    
    # Shuffle train samples
    train_indices = train_df.index.tolist()
    np.random.shuffle(train_indices)
    
    # Split into labeled and unlabeled
    labeled_indices = train_indices[:rd_label]
    unlabeled_indices = train_indices[rd_label:rd_label + rd_unlabel]
    
    # Add SSL_SPLIT column
    df['SSL_SPLIT'] = 'EXCLUDED'  # Default for all
    df.loc[labeled_indices, 'SSL_SPLIT'] = 'LABELED'
    df.loc[unlabeled_indices, 'SSL_SPLIT'] = 'UNLABELED'
    
    # For val/test, set to empty or keep as EXCLUDED
    df.loc[other_df.index, 'SSL_SPLIT'] = ''
    
    print(f"\nSSL split:")
    print(f"  Labeled: {len(labeled_indices)}")
    print(f"  Unlabeled: {len(unlabeled_indices)}")
    print(f"  Excluded: {len(train_df) - len(labeled_indices) - len(unlabeled_indices)}")
    
    # Save to new CSV
    output_name = f"{file_list_name.replace('.csv', '')}_ssl_{rd_label}_{rd_unlabel}.csv"
    output_path = os.path.join(data_dir, output_name)
    df.to_csv(output_path, index=False)
    
    print(f"\nSaved to: {output_path}")
    print(f"\nSSL_SPLIT distribution:")
    print(df['SSL_SPLIT'].value_counts())
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create SSL split for semi-supervised learning")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing FileList.csv")
    parser.add_argument("--file_list_name", type=str, default="FileList.csv",
                        help="Name of the FileList CSV file")
    parser.add_argument("--rd_label", type=int, required=True,
                        help="Number of labeled samples")
    parser.add_argument("--rd_unlabel", type=int, required=True,
                        help="Number of unlabeled samples")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    create_ssl_split(args.data_dir, args.file_list_name, args.rd_label, args.rd_unlabel, args.seed)
