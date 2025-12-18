"""
So2Sat_POP dataset implementation.
Handles TIFF images with CSV-based population targets.
"""

import os
import pandas as pd
import numpy as np
import cv2
from typing import Dict, Any, Optional
from .base_dataset import BaseDataset


class So2SatPOPDataset(BaseDataset):
    """
    So2Sat Population dataset.
    
    Structure:
    - So2Sat_POP_Part1/train/{city_id}_{pop}_{cityname}/{city_id}_{pop}_{cityname}.csv
    - Images: {city_dir}/sen2{season}/Class_{class}/{grd_id}_sen2{season}.tif
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        split: str = "train",
        ssl_type: int = 0,
        ssl_postfix: str = "",
        ssl_mult: int = 1,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        pad: Optional[int] = None,
    ):
        """Initialize So2Sat_POP dataset."""
        super().__init__(config, split, ssl_type, ssl_postfix, ssl_mult, mean, std, pad)
        
        # Extract config parameters
        self.data_root = config["data_root"]
        self.season = config.get("season", "spring")
        self.parts = config.get("parts", ["Part1", "Part2"])
        self.csv_target_column = config["csv_target_column"]
        self.csv_id_column = config["csv_id_column"]
        self.csv_class_column = config["csv_class_column"]
        self.image_path_template = config["image_path_template"]
        self.use_rgb_only = config.get("use_rgb_only", True)
        self.file_list_name = config.get("file_list", "FileList.csv")
        
        # Load file list
        self._load_file_list()
        
        # Filter by SSL split if needed
        self._filter_ssl_split()
        
        # Apply SSL multiplier for labeled data
        if self.ssl_type == 1 and self.ssl_mult > 0:
            self._apply_ssl_multiplier()
    
    def _load_file_list(self):
        """Load or generate file list."""
        file_list_path = os.path.join(self.data_root, self.file_list_name)
        
        # Check if FileList exists
        if os.path.exists(file_list_path):
            print(f"Loading FileList from: {file_list_path}")
            self._load_from_csv(file_list_path)
        else:
            print(f"FileList not found. Generating from data structure...")
            self._generate_file_list()
            # Save generated file list
            self._save_file_list(file_list_path)
    
    def _load_from_csv(self, csv_path: str):
        """Load file list from existing CSV."""
        data = pd.read_csv(csv_path)
        
        # Normalize SPLIT column
        if "SPLIT" in data.columns:
            data["SPLIT"] = data["SPLIT"].str.upper()
        
        # Filter by split
        if self.split != "ALL":
            data = data[data["SPLIT"] == self.split]
        
        # Extract file identifiers and targets
        self.file_list = []
        self.targets = []
        self.metadata = []  # Store additional info (city_dir, class, etc.)
        
        for idx, row in data.iterrows():
            # File identifier format: {city_dir}|{grd_id}|{class}
            # Try to parse from FileName if it exists, otherwise construct
            file_id = row.get("FileName", "")
            if not file_id or "|" not in str(file_id):
                # Construct file_id from components
                city_dir = row.get("CityDir", "")
                grd_id = str(row.get(self.csv_id_column, ""))
                class_val = str(row.get(self.csv_class_column, ""))
                file_id = f"{city_dir}|{grd_id}|{class_val}"
            
            # Parse file_id to extract components if needed
            if "|" in file_id:
                parts = file_id.split("|")
                city_dir = parts[0] if len(parts) > 0 else row.get("CityDir", "")
                grd_id = parts[1] if len(parts) > 1 else str(row.get(self.csv_id_column, ""))
                class_val = parts[2] if len(parts) > 2 else str(row.get(self.csv_class_column, ""))
            else:
                city_dir = row.get("CityDir", "")
                grd_id = str(row.get(self.csv_id_column, ""))
                class_val = str(row.get(self.csv_class_column, ""))
            
            self.file_list.append(file_id)
            self.targets.append(float(row[self.csv_target_column]))
            
            # Store metadata for path resolution
            metadata = {
                "city_dir": city_dir,
                "grd_id": grd_id,
                "class": class_val,
                "split": row.get("SPLIT", self.split),
            }
            if "SSL_SPLIT" in row:
                metadata["ssl_split"] = row["SSL_SPLIT"]
            self.metadata.append(metadata)
    
    def _generate_file_list(self):
        """Generate file list by scanning directory structure."""
        print("Scanning So2Sat_POP directory structure...")
        
        self.file_list = []
        self.targets = []
        self.metadata = []
        
        # Scan each part
        for part in self.parts:
            part_dir = os.path.join(self.data_root, f"So2Sat_POP_{part}", self.split.lower())
            
            if not os.path.exists(part_dir):
                print(f"Warning: Part directory not found: {part_dir}")
                continue
            
            # Scan city directories
            for city_dir_name in os.listdir(part_dir):
                city_dir = os.path.join(part_dir, city_dir_name)
                
                if not os.path.isdir(city_dir):
                    continue
                
                # Look for CSV file
                csv_file = os.path.join(city_dir, f"{city_dir_name}.csv")
                
                if not os.path.exists(csv_file):
                    continue
                
                # Read CSV
                try:
                    city_data = pd.read_csv(csv_file)
                    
                    # Check if required columns exist
                    if self.csv_target_column not in city_data.columns:
                        continue
                    if self.csv_id_column not in city_data.columns:
                        continue
                    if self.csv_class_column not in city_data.columns:
                        continue
                    
                    # Process each row
                    for _, row in city_data.iterrows():
                        grd_id = str(row[self.csv_id_column])
                        class_val = str(int(row[self.csv_class_column]))
                        pop = float(row[self.csv_target_column])
                        
                        # Check if image exists
                        image_path = self._resolve_image_path(
                            city_dir, grd_id, class_val
                        )
                        
                        if image_path:
                            if os.path.exists(image_path):
                                file_id = f"{city_dir}|{grd_id}|{class_val}"
                                self.file_list.append(file_id)
                                self.targets.append(pop)
                                self.metadata.append({
                                    "city_dir": city_dir,
                                    "grd_id": grd_id,
                                    "class": class_val,
                                    "split": self.split,
                                })
                            # Skip if image doesn't exist (silently)
                
                except Exception as e:
                    print(f"Error processing {csv_file}: {e}")
                    continue
        
        print(f"Generated file list with {len(self.file_list)} samples")
    
    def _resolve_image_path(self, city_dir: str, grd_id: str, class_val: str) -> Optional[str]:
        """Resolve image path from template."""
        template = self.image_path_template
        path = template.format(
            city_dir=city_dir,
            season=self.season,
            class=class_val,
            grd_id=grd_id
        )
        return path
    
    def _save_file_list(self, output_path: str):
        """Save generated file list to CSV."""
        data = {
            "FileName": self.file_list,
            "SPLIT": [m["split"] for m in self.metadata],
            self.csv_target_column: self.targets,
            "CityDir": [m["city_dir"] for m in self.metadata],
            self.csv_id_column: [m["grd_id"] for m in self.metadata],
            self.csv_class_column: [m["class"] for m in self.metadata],
        }
        
        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved file list to: {output_path}")
    
    def _filter_ssl_split(self):
        """Filter by SSL split if SSL postfix is provided."""
        if not self.ssl_postfix:
            return
        
        # Load SSL file list
        ssl_file_list_path = os.path.join(
            self.data_root, 
            f"FileList{self.ssl_postfix}.csv"
        )
        
        if not os.path.exists(ssl_file_list_path):
            print(f"Warning: SSL file list not found: {ssl_file_list_path}")
            return
        
        ssl_data = pd.read_csv(ssl_file_list_path)
        
        if "SSL_SPLIT" not in ssl_data.columns:
            return
        
        # Create mapping
        ssl_map = dict(zip(ssl_data["FileName"], ssl_data["SSL_SPLIT"]))
        
        # Filter based on ssl_type
        if self.ssl_type == 1:  # Labeled only
            filtered_indices = [
                i for i, file_id in enumerate(self.file_list)
                if ssl_map.get(file_id) == "LABELED"
            ]
        elif self.ssl_type == 2:  # Unlabeled only
            filtered_indices = [
                i for i, file_id in enumerate(self.file_list)
                if ssl_map.get(file_id) != "LABELED"
            ]
        else:  # All
            filtered_indices = list(range(len(self.file_list)))
        
        # Apply filter
        self.file_list = [self.file_list[i] for i in filtered_indices]
        self.targets = [self.targets[i] for i in filtered_indices]
        self.metadata = [self.metadata[i] for i in filtered_indices]
        
        print(f"After SSL filtering (type={self.ssl_type}): {len(self.file_list)} samples")
    
    def _apply_ssl_multiplier(self):
        """Apply SSL multiplier to duplicate labeled samples."""
        if self.ssl_mult <= 1:
            return
        
        original_file_list = self.file_list.copy()
        original_targets = self.targets.copy()
        original_metadata = self.metadata.copy()
        
        # Repeat samples
        self.file_list = original_file_list * self.ssl_mult
        self.targets = original_targets * self.ssl_mult
        self.metadata = original_metadata * self.ssl_mult
        
        print(f"Applied SSL multiplier {self.ssl_mult}: {len(self.file_list)} samples")
    
    def _get_image_path(self, index: int) -> str:
        """Get image path for given index."""
        metadata = self.metadata[index]
        return self._resolve_image_path(
            metadata["city_dir"],
            metadata["grd_id"],
            metadata["class"]
        )
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load and preprocess TIFF image."""
        # Read TIFF using OpenCV (supports multi-channel)
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        
        # Handle different image formats
        if len(image.shape) == 2:
            # Grayscale - convert to 3-channel
            image = np.stack([image, image, image], axis=0)
        elif len(image.shape) == 3:
            # Multi-channel image
            if image.shape[2] > 3 and self.use_rgb_only:
                # Use only first 3 channels
                image = image[:, :, :3]
            # Convert from (H, W, C) to (C, H, W)
            image = image.transpose(2, 0, 1)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        # Ensure float32
        image = image.astype(np.float32)
        
        return image
    
    def _get_target(self, index: int) -> float:
        """Get target value for given index."""
        return self.targets[index]

