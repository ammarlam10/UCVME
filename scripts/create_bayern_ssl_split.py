#!/usr/bin/env python3
"""
Create SSL splits for Bayern Forest Height dataset.
Generates FileList_ssl_{labeled}_{unlabeled}.csv files for different percentages.
"""
import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_ssl_split(data_dir, label_percentage, seed=0):
    """
    Create SSL split for Bayern Forest Height dataset.
    
    Args:
        data_dir: Path to dataset directory
        label_percentage: Percentage of labeled data (0.05, 0.10, 0.20)
        seed: Random seed
    """
    print(f"\n{'='*60}")
    print(f"Creating SSL split: {label_percentage*100:.0f}% labeled")
    print(f"{'='*60}")
    
    # Count total samples from h5 files
    h5_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])
    print(f"Found {len(h5_files)} .h5 files")
    
    # Calculate total samples (40 files × 361 samples = 14,440)
    total_samples = 14440
    
    # Calculate train/val/test split (80/10/10)
    train_end = int(0.8 * total_samples)  # 11,552 training samples
    
    print(f"Total samples: {total_samples}")
    print(f"Training samples: {train_end}")
    
    # Calculate labeled and unlabeled counts
    num_labeled = int(train_end * label_percentage)
    num_unlabeled = train_end - num_labeled
    
    print(f"Labeled: {num_labeled} ({label_percentage*100:.0f}%)")
    print(f"Unlabeled: {num_unlabeled} ({(1-label_percentage)*100:.0f}%)")
    
    # Create split
    np.random.seed(seed)
    train_indices = np.arange(train_end)
    np.random.shuffle(train_indices)
    
    # Create DataFrame with all samples
    data = pd.DataFrame({
        'Index': np.arange(total_samples),
        'SSL_SPLIT': 'UNLABELED'
    })
    
    # Mark labeled samples
    labeled_indices = train_indices[:num_labeled]
    data.loc[labeled_indices, 'SSL_SPLIT'] = 'LABELED'
    
    # Save to file
    output_file = os.path.join(data_dir, f"FileList_ssl_{num_labeled}_{num_unlabeled}.csv")
    data.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total rows: {len(data)}")
    print(f"  Labeled: {(data['SSL_SPLIT'] == 'LABELED').sum()}")
    print(f"  Unlabeled: {(data['SSL_SPLIT'] == 'UNLABELED').sum()}")
    
    return num_labeled, num_unlabeled


def main():
    data_dir = "/home/ammar/data/Bayern_forest_height_reduced"
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    print(f"Data directory: {data_dir}")
    
    # Create splits for 5%, 10%, and 20%
    percentages = [0.05, 0.10, 0.20]
    splits = {}
    
    for pct in percentages:
        num_labeled, num_unlabeled = create_ssl_split(data_dir, pct, seed=0)
        splits[pct] = (num_labeled, num_unlabeled)
    
    print(f"\n{'='*60}")
    print("Summary of SSL splits created:")
    print(f"{'='*60}")
    for pct, (num_labeled, num_unlabeled) in splits.items():
        print(f"{pct*100:>5.0f}% labeled: {num_labeled:>5} labeled, {num_unlabeled:>5} unlabeled")
    print(f"{'='*60}")
    print("\n✓ All SSL splits created successfully!")
    print("\nUse these postfixes in your config files:")
    for pct, (num_labeled, num_unlabeled) in splits.items():
        print(f"  {pct*100:.0f}%: ssl_postfix='_ssl_{num_labeled}_{num_unlabeled}'")


if __name__ == "__main__":
    main()
