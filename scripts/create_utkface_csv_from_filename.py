#!/usr/bin/env python3
"""
Script to create a new FileList.csv for UTKFace dataset where age labels
are extracted from filenames instead of using the existing age column.

UTKFace filename format: age_gender_race_date.jpg.chip.jpg
Example: 35_0_0_20170117175619787.jpg.chip.jpg -> age = 35
"""

import pandas as pd
import os
import sys
import re

def extract_age_from_filename(filename):
    """
    Extract age from UTKFace filename.
    Format: age_gender_race_date.jpg.chip.jpg
    Returns the age as integer, or None if extraction fails.
    """
    # Get just the filename without path
    basename = os.path.basename(filename)
    
    # Extract first number before first underscore
    match = re.match(r'^(\d+)_', basename)
    if match:
        return int(match.group(1))
    else:
        return None

def create_csv_from_filename(input_csv, output_csv):
    """
    Read input CSV and create new CSV with age extracted from filenames.
    """
    print(f"Reading input CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"Original CSV shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Extract age from filename
    print("Extracting age from filenames...")
    df['age'] = df['FileName'].apply(extract_age_from_filename)
    
    # Check for any None values (extraction failures)
    failed_extractions = df[df['age'].isna()]
    if len(failed_extractions) > 0:
        print(f"Warning: {len(failed_extractions)} filenames could not extract age:")
        print(failed_extractions[['FileName']].head(10))
        # Remove rows where age extraction failed
        df = df.dropna(subset=['age'])
        print(f"Removed {len(failed_extractions)} rows with failed age extraction")
    
    # Convert age to int
    df['age'] = df['age'].astype(int)
    
    # Show some statistics
    print(f"\nAge statistics from filenames:")
    print(f"  Min age: {df['age'].min()}")
    print(f"  Max age: {df['age'].max()}")
    print(f"  Mean age: {df['age'].mean():.2f}")
    print(f"  Std age: {df['age'].std():.2f}")
    
    # Show sample of changes
    print(f"\nSample of extracted ages:")
    sample = df[['FileName', 'age', 'SPLIT']].head(10)
    print(sample.to_string(index=False))
    
    # Save new CSV
    print(f"\nSaving new CSV to: {output_csv}")
    df.to_csv(output_csv, index=False)
    print(f"New CSV shape: {df.shape}")
    print(f"Done! Created {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_utkface_csv_from_filename.py <input_csv> <output_csv>")
        print("Example: python create_utkface_csv_from_filename.py FileList.csv FileList_from_filename.csv")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    if not os.path.exists(input_csv):
        print(f"Error: Input file not found: {input_csv}")
        sys.exit(1)
    
    create_csv_from_filename(input_csv, output_csv)
