#!/usr/bin/env python3
"""
Custom script to create FileList.csv for So2Sat_POP dataset with specific requirements:
- Use sen2spring (RGB bands 2-4)
- Train: first 90% from train directory
- Valid: last 10% from train directory  
- Test: from test directory
- Reduce 80% of labels where POP <= 100 on training set

Usage:
    python scripts/create_so2sat_pop_custom.py \
        --data_dir /work/ammar/sslrp/data/So2Sat_POP \
        --output /work/ammar/sslrp/data/So2Sat_POP/FileList.csv
"""

import os
import pandas as pd
import argparse
import numpy as np
from pathlib import Path


def create_filelist_custom(data_dir, output_path, part1_dir="So2Sat_POP_Part1", seed=42):
    """
    Create FileList.csv for So2Sat_POP dataset with custom train/val split.
    
    Args:
        data_dir: Root directory containing Part1
        output_path: Path to save FileList.csv
        part1_dir: Name of Part1 directory (contains labels and Sentinel-2 images)
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    data_dir = Path(data_dir)
    part1_path = data_dir / part1_dir
    
    if not part1_path.exists():
        raise ValueError(f"Part1 directory not found: {part1_path}")
    
    records = []
    
    # Process train and test splits
    for split in ['train', 'test']:
        split_path = part1_path / split
        
        if not split_path.exists():
            print(f"Warning: {split} directory not found: {split_path}")
            continue
        
        print(f"\nProcessing {split} split...")
        
        # Get all city directories
        city_dirs = [d for d in split_path.iterdir() if d.is_dir()]
        print(f"  Found {len(city_dirs)} cities")
        
        for city_dir in sorted(city_dirs):
            city_name = city_dir.name
            
            # Look for CSV file with labels
            csv_files = list(city_dir.glob("*.csv"))
            if len(csv_files) == 0:
                print(f"  Warning: No CSV file found in {city_name}, skipping")
                continue
            
            csv_file = csv_files[0]
            
            try:
                city_data = pd.read_csv(csv_file)
            except Exception as e:
                print(f"  Warning: Could not read {csv_file}: {e}")
                continue
            
            # Check required columns
            if 'GRD_ID' not in city_data.columns or 'POP' not in city_data.columns:
                print(f"  Warning: {csv_file} missing required columns (GRD_ID, POP), skipping")
                continue
            
            # Process each row
            for _, row in city_data.iterrows():
                grd_id = str(row['GRD_ID'])
                pop = row['POP']
                class_val = row.get('Class', '')
                
                # Find corresponding Sentinel-2 image in Part1 using sen2spring
                # Images are in: Part1/split/city/sen2spring/Class_X/GRD_ID_sen2spring.tif
                class_dir_name = f"Class_{class_val}" if class_val != '' else "Class_0"
                image_path = part1_path / split / city_name / "sen2spring" / class_dir_name / f"{grd_id}_sen2spring.tif"
                
                # Alternative: check if image exists with different naming
                if not image_path.exists():
                    # Try without class directory
                    image_path_alt = part1_path / split / city_name / "sen2spring" / f"{grd_id}_sen2spring.tif"
                    if image_path_alt.exists():
                        image_path = image_path_alt
                    else:
                        # Skip if image not found
                        continue
                
                # Create relative path from data_dir
                rel_image_path = image_path.relative_to(data_dir)
                
                # Create record
                records.append({
                    'FileName': str(rel_image_path),
                    'SPLIT': split.upper(),
                    'POP': pop,
                    'GRD_ID': grd_id,
                    'Class': class_val,
                    'City': city_name
                })
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    if len(df) == 0:
        raise ValueError("No valid records found. Check data directory structure.")
    
    print(f"\nInitial records: {len(df)}")
    print(f"  Train: {len(df[df['SPLIT'] == 'TRAIN'])}")
    print(f"  Test: {len(df[df['SPLIT'] == 'TEST'])}")
    
    # Split TRAIN into 90% TRAIN and 10% VAL
    train_df = df[df['SPLIT'] == 'TRAIN'].copy()
    test_df = df[df['SPLIT'] == 'TEST'].copy()
    
    # Sort by index to ensure consistent ordering (first 90%, last 10%)
    train_df = train_df.sort_index()
    
    split_point = int(len(train_df) * 0.9)
    train_90 = train_df.iloc[:split_point].copy()
    val_10 = train_df.iloc[split_point:].copy()
    
    # Update SPLIT for validation
    val_10['SPLIT'] = 'VAL'
    
    print(f"\nAfter train/val split:")
    print(f"  Train (90%): {len(train_90)}")
    print(f"  Val (10%): {len(val_10)}")
    print(f"  Test: {len(test_df)}")
    
    # Reduce 80% of labels where POP <= 100 on training set
    train_low_pop = train_90[train_90['POP'] <= 100].copy()
    train_high_pop = train_90[train_90['POP'] > 100].copy()
    
    print(f"\nBefore reduction:")
    print(f"  Train with POP <= 100: {len(train_low_pop)}")
    print(f"  Train with POP > 100: {len(train_high_pop)}")
    
    # Keep only 20% of low population samples
    n_keep = int(len(train_low_pop) * 0.2)
    train_low_pop_sampled = train_low_pop.sample(n=n_keep, random_state=seed)
    
    print(f"\nAfter 80% reduction of POP <= 100:")
    print(f"  Train with POP <= 100 (kept 20%): {len(train_low_pop_sampled)}")
    print(f"  Train with POP > 100: {len(train_high_pop)}")
    
    # Combine back
    train_final = pd.concat([train_low_pop_sampled, train_high_pop], ignore_index=True)
    
    # Combine all splits
    df_final = pd.concat([train_final, val_10, test_df], ignore_index=True)
    
    print(f"\nFinal dataset:")
    print(f"  Train: {len(df_final[df_final['SPLIT'] == 'TRAIN'])}")
    print(f"  Val: {len(df_final[df_final['SPLIT'] == 'VAL'])}")
    print(f"  Test: {len(df_final[df_final['SPLIT'] == 'TEST'])}")
    print(f"  Total: {len(df_final)}")
    
    # Save to CSV
    df_final.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    
    # Print statistics
    print(f"\nPopulation statistics (all data):")
    print(f"  Mean: {df_final['POP'].mean():.2f}")
    print(f"  Std: {df_final['POP'].std():.2f}")
    print(f"  Min: {df_final['POP'].min():.0f}")
    print(f"  Max: {df_final['POP'].max():.0f}")
    
    print(f"\nPopulation statistics (train only):")
    train_stats = df_final[df_final['SPLIT'] == 'TRAIN']['POP']
    print(f"  Mean: {train_stats.mean():.2f}")
    print(f"  Std: {train_stats.std():.2f}")
    print(f"  Min: {train_stats.min():.0f}")
    print(f"  Max: {train_stats.max():.0f}")
    
    return df_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create custom FileList.csv for So2Sat_POP dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing So2Sat_POP_Part1")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for FileList.csv (default: data_dir/FileList.csv)")
    parser.add_argument("--part1_dir", type=str, default="So2Sat_POP_Part1",
                        help="Name of Part1 directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(args.data_dir, "FileList.csv")
    
    create_filelist_custom(args.data_dir, output_path, args.part1_dir, args.seed)
