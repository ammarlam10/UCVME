"""
Script to generate FileList.csv for So2Sat_POP dataset.
Can be run standalone to pre-generate the file list.
"""

import os
import sys
import pandas as pd
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datasets.so2sat_pop import So2SatPOPDataset
from utils.config_loader import load_dataset_config


def main():
    parser = argparse.ArgumentParser(description="Generate FileList.csv for So2Sat_POP")
    parser.add_argument("--data_root", type=str, default="/work/ammar/sslrp/data",
                        help="Root directory containing datasets")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for FileList.csv (default: {data_root}/So2Sat_POP/FileList.csv)")
    parser.add_argument("--season", type=str, default="spring",
                        choices=["spring", "summer", "autumn", "winter"],
                        help="Season to use for images")
    
    args = parser.parse_args()
    
    # Load config
    config = load_dataset_config("so2sat_pop")
    config["data_root"] = os.path.join(args.data_root, "So2Sat_POP")
    config["season"] = args.season
    
    # Determine output path
    if args.output is None:
        output_path = os.path.join(config["data_root"], "FileList.csv")
    else:
        output_path = args.output
    
    print(f"Generating FileList.csv for So2Sat_POP")
    print(f"Data root: {config['data_root']}")
    print(f"Season: {config['season']}")
    print(f"Output: {output_path}")
    
    # Create dataset instance (this will generate the file list)
    dataset = So2SatPOPDataset(
        config=config,
        split="ALL",  # Get all splits
        ssl_type=0
    )
    
    # Save file list
    dataset._save_file_list(output_path)
    
    print(f"\nFileList.csv generated successfully!")
    print(f"Total samples: {len(dataset.file_list)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()

