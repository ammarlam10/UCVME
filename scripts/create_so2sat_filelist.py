#!/usr/bin/env python3
"""
Helper script to create FileList.csv for So2Sat_POP dataset.

This script scans the So2Sat_POP directory structure and creates a FileList.csv
that can be used with the UCVME training script.

Usage:
    python scripts/create_so2sat_filelist.py \
        --data_dir /work/ammar/sslrp/data/So2Sat_POP \
        --output /work/ammar/sslrp/data/So2Sat_POP/FileList.csv
"""

import os
import pandas as pd
import argparse
from pathlib import Path


def create_filelist(data_dir, output_path, part1_dir="So2Sat_POP_Part1", part2_dir="So2Sat_POP_Part2"):
    """
    Create FileList.csv for So2Sat_POP dataset.
    
    Args:
        data_dir: Root directory containing Part1 and Part2
        output_path: Path to save FileList.csv
        part1_dir: Name of Part1 directory (contains labels and Sentinel-2 images)
        part2_dir: Name of Part2 directory (not used for Sentinel-2, kept for compatibility)
    """
    data_dir = Path(data_dir)
    part1_path = data_dir / part1_dir
    
    if not part1_path.exists():
        raise ValueError(f"Part1 directory not found: {part1_path}")
    
    records = []
    
    # Process train and test splits
    for split in ['train', 'test']:
        split_path_part1 = part1_path / split
        
        if not split_path_part1.exists():
            print(f"Warning: {split} directory not found in Part1, skipping...")
            continue
        
        # Get all city directories
        city_dirs = [d for d in split_path_part1.iterdir() if d.is_dir()]
        
        print(f"Processing {split} split: {len(city_dirs)} cities")
        
        for city_dir in city_dirs:
            city_name = city_dir.name
            
            # Read city CSV file (contains GRD_ID, Class, POP)
            city_csv = city_dir / f"{city_name}.csv"
            if not city_csv.exists():
                print(f"Warning: {city_csv} not found, skipping city {city_name}")
                continue
            
            city_data = pd.read_csv(city_csv)
            
            # Check required columns
            if 'GRD_ID' not in city_data.columns or 'POP' not in city_data.columns:
                print(f"Warning: {city_csv} missing required columns (GRD_ID, POP), skipping")
                continue
            
            # Process each row
            for _, row in city_data.iterrows():
                grd_id = str(row['GRD_ID'])
                pop = row['POP']
                class_val = row.get('Class', '')
                
                # Find corresponding Sentinel-2 image in Part1
                # Images are in: Part1/split/city/sen2summer/Class_X/GRD_ID_sen2summer.tif
                class_dir_name = f"Class_{class_val}" if class_val != '' else "Class_0"
                image_path = part1_path / split / city_name / "sen2summer" / class_dir_name / f"{grd_id}_sen2summer.tif"
                
                # Alternative: check if image exists with different naming
                if not image_path.exists():
                    # Try without class directory
                    image_path_alt = part1_path / split / city_name / "sen2summer" / f"{grd_id}_sen2summer.tif"
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
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nCreated FileList.csv with {len(df)} records")
    print(f"  Train: {len(df[df['SPLIT'] == 'TRAIN'])}")
    print(f"  Test: {len(df[df['SPLIT'] == 'TEST'])}")
    print(f"  Saved to: {output_path}")
    
    # Print statistics
    print(f"\nPopulation statistics:")
    print(f"  Mean: {df['POP'].mean():.2f}")
    print(f"  Std: {df['POP'].std():.2f}")
    print(f"  Min: {df['POP'].min():.0f}")
    print(f"  Max: {df['POP'].max():.0f}")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create FileList.csv for So2Sat_POP dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing So2Sat_POP_Part1 and So2Sat_POP_Part2")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for FileList.csv (default: data_dir/FileList.csv)")
    parser.add_argument("--part1_dir", type=str, default="So2Sat_POP_Part1",
                        help="Name of Part1 directory")
    parser.add_argument("--part2_dir", type=str, default="So2Sat_POP_Part2",
                        help="Name of Part2 directory")
    
    args = parser.parse_args()
    
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(args.data_dir, "FileList.csv")
    
    create_filelist(args.data_dir, output_path, args.part1_dir, args.part2_dir)

