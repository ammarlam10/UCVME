#!/usr/bin/env python3
"""
Script to compare CSV labels vs filename-extracted labels for UTKFace dataset.
"""

import pandas as pd
import os
import re
import sys

def extract_age_from_filename(filename):
    """Extract age from UTKFace filename."""
    basename = os.path.basename(filename)
    match = re.match(r'^(\d+)_', basename)
    if match:
        return int(match.group(1))
    else:
        return None

def compare_labels(csv_path):
    """Compare CSV age labels with filename-extracted labels."""
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'age' not in df.columns:
        print("ERROR: 'age' column not found in CSV!")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    print(f"\nTotal samples: {len(df)}")
    print(f"CSV columns: {df.columns.tolist()}")
    
    # Extract age from filenames
    print("\nExtracting age from filenames...")
    df['age_from_filename'] = df['FileName'].apply(extract_age_from_filename)
    
    # Check for extraction failures
    failed = df[df['age_from_filename'].isna()]
    if len(failed) > 0:
        print(f"Warning: {len(failed)} filenames could not extract age")
        print("Sample failed extractions:")
        print(failed[['FileName']].head(5))
    
    # Remove rows where extraction failed for comparison
    df_compare = df.dropna(subset=['age_from_filename']).copy()
    
    # Convert to int for comparison
    df_compare['age'] = df_compare['age'].astype(int)
    df_compare['age_from_filename'] = df_compare['age_from_filename'].astype(int)
    
    # Calculate differences
    df_compare['age_diff'] = df_compare['age_from_filename'] - df_compare['age']
    df_compare['abs_diff'] = df_compare['age_diff'].abs()
    
    # Statistics
    print(f"\n{'='*60}")
    print("LABEL COMPARISON STATISTICS")
    print(f"{'='*60}")
    
    total = len(df_compare)
    identical = len(df_compare[df_compare['age_diff'] == 0])
    different = len(df_compare[df_compare['age_diff'] != 0])
    
    print(f"\nTotal samples (with valid filename extraction): {total}")
    print(f"Identical labels: {identical} ({100*identical/total:.2f}%)")
    print(f"Different labels: {different} ({100*different/total:.2f}%)")
    
    if different > 0:
        print(f"\nDifference Statistics:")
        print(f"  Mean absolute difference: {df_compare['abs_diff'].mean():.2f} years")
        print(f"  Max absolute difference: {df_compare['abs_diff'].max()} years")
        print(f"  Std of differences: {df_compare['age_diff'].std():.2f} years")
        print(f"  Mean difference (filename - CSV): {df_compare['age_diff'].mean():.2f} years")
        
        print(f"\nDistribution of absolute differences:")
        print(df_compare['abs_diff'].value_counts().sort_index().head(20))
        
        print(f"\n{'='*60}")
        print("SAMPLES WITH LARGEST DIFFERENCES")
        print(f"{'='*60}")
        largest_diff = df_compare.nlargest(20, 'abs_diff')[['FileName', 'age', 'age_from_filename', 'abs_diff', 'SPLIT']]
        print(largest_diff.to_string(index=False))
        
        print(f"\n{'='*60}")
        print("RANDOM SAMPLE OF DIFFERENCES")
        print(f"{'='*60}")
        sample_diff = df_compare[df_compare['abs_diff'] > 0].sample(min(20, different))[['FileName', 'age', 'age_from_filename', 'abs_diff', 'SPLIT']]
        print(sample_diff.to_string(index=False))
    
    # Statistics by split
    print(f"\n{'='*60}")
    print("STATISTICS BY SPLIT")
    print(f"{'='*60}")
    for split in df_compare['SPLIT'].unique():
        split_df = df_compare[df_compare['SPLIT'] == split]
        split_identical = len(split_df[split_df['age_diff'] == 0])
        split_different = len(split_df[split_df['age_diff'] != 0])
        print(f"\n{split}:")
        print(f"  Total: {len(split_df)}")
        print(f"  Identical: {split_identical} ({100*split_identical/len(split_df):.2f}%)")
        print(f"  Different: {split_different} ({100*split_different/len(split_df):.2f}%)")
        if split_different > 0:
            print(f"  Mean abs diff: {split_df['abs_diff'].mean():.2f} years")
            print(f"  Max abs diff: {split_df['abs_diff'].max()} years")
    
    # Age distribution comparison
    print(f"\n{'='*60}")
    print("AGE DISTRIBUTION COMPARISON")
    print(f"{'='*60}")
    print(f"\nCSV labels:")
    print(f"  Min: {df_compare['age'].min()}, Max: {df_compare['age'].max()}")
    print(f"  Mean: {df_compare['age'].mean():.2f}, Std: {df_compare['age'].std():.2f}")
    print(f"\nFilename labels:")
    print(f"  Min: {df_compare['age_from_filename'].min()}, Max: {df_compare['age_from_filename'].max()}")
    print(f"  Mean: {df_compare['age_from_filename'].mean():.2f}, Std: {df_compare['age_from_filename'].std():.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        csv_path = "/work/ammar/sslrp/UCVME/DATA_DIR/FileList.csv"
        print(f"No CSV path provided, using default: {csv_path}")
    else:
        csv_path = sys.argv[1]
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    compare_labels(csv_path)
