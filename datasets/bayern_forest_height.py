import os
import numpy as np
import h5py
import torch
import torchvision
from typing import List, Optional, Tuple


class BayernForestHeightDataset(torchvision.datasets.VisionDataset):
    """
    Bayern Forest Height Dataset for pixel-wise regression.
    
    Dataset structure:
    - Input: RGB images (256x256x3)
    - Target: Height maps/NDSM (256x256x1)
    - Data stored in HDF5 files with keys 'rgb' and 'ndsm'
    
    Args:
        root: Path to data directory containing .h5 files
        split: One of 'train', 'val', 'test'
        target_type: List of target types (default: ['height'])
        mean: Mean for image normalization
        std: Std for image normalization
        pad: Padding for data augmentation (train only)
        ssl_postfix: Postfix for SSL file list (for compatibility)
        ssl_type: SSL type (for compatibility)
        ssl_mult: SSL multiplier (for compatibility)
        seed: Random seed for train/val/test split
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: List[str] = None,
        mean: float = 0.0,
        std: float = 1.0,
        pad: Optional[int] = None,
        ssl_postfix: str = "",
        ssl_type: int = 0,
        ssl_mult: int = -1,
        seed: int = 0,
        file_list_name: str = "FileList.csv",  # For compatibility
        **kwargs
    ):
        super(BayernForestHeightDataset, self).__init__(root)
        
        if target_type is None:
            target_type = ["height"]
        
        self.split = split.upper()
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.pad = pad
        self.ssl_postfix = ssl_postfix
        self.ssl_type = ssl_type
        self.ssl_mult = ssl_mult
        self.seed = seed
        
        # Check if SSL split file exists
        self.use_ssl_split = False
        if ssl_postfix and self.split == "TRAIN":
            ssl_split_file = os.path.join(root, f"FileList{ssl_postfix}.csv")
            if os.path.exists(ssl_split_file):
                self.use_ssl_split = True
                print(f"Using SSL split file: {ssl_split_file}")
        
        # Load all h5 files
        self.h5_files = sorted([
            os.path.join(root, f) for f in os.listdir(root) 
            if f.endswith('.h5')
        ])
        
        if len(self.h5_files) == 0:
            raise ValueError(f"No .h5 files found in {root}")
        
        print(f"Found {len(self.h5_files)} .h5 files in {root}")
        
        # Load all data into memory (small dataset)
        self.rgb_data = []
        self.ndsm_data = []
        
        for h5_file in self.h5_files:
            with h5py.File(h5_file, 'r') as f:
                # Shape: (N, H, W, C)
                rgb = f['rgb'][:]  # (N, 256, 256, 3)
                ndsm = f['ndsm'][:]  # (N, 256, 256, 1)
                
                self.rgb_data.append(rgb)
                self.ndsm_data.append(ndsm)
        
        # Concatenate all data
        self.rgb_data = np.concatenate(self.rgb_data, axis=0)  # (Total, 256, 256, 3)
        self.ndsm_data = np.concatenate(self.ndsm_data, axis=0)  # (Total, 256, 256, 1)
        
        total_samples = len(self.rgb_data)
        print(f"Total samples loaded: {total_samples}")
        
        # Create train/val/test split (80/10/10)
        np.random.seed(seed)
        indices = np.random.permutation(total_samples)
        
        train_end = int(0.8 * total_samples)
        val_end = int(0.9 * total_samples)
        
        if self.split == "TRAIN":
            train_indices = indices[:train_end]
            
            # Handle SSL split if specified
            if self.use_ssl_split:
                # Load SSL split information
                ssl_split_file = os.path.join(root, f"FileList{ssl_postfix}.csv")
                import pandas as pd
                ssl_data = pd.read_csv(ssl_split_file)
                
                # Get labeled and unlabeled indices
                if ssl_type == 1:
                    # Labeled data
                    labeled_mask = ssl_data['SSL_SPLIT'] == 'LABELED'
                    labeled_indices = ssl_data[labeled_mask].index.tolist()
                    self.indices = np.array([train_indices[i] for i in labeled_indices if i < len(train_indices)])
                    print(f"SSL Labeled split: {len(self.indices)} samples")
                    
                    # Apply ssl_mult if specified
                    if ssl_mult > 0:
                        # Repeat labeled data ssl_mult times
                        self.indices = np.tile(self.indices, ssl_mult)
                        print(f"SSL multiplier {ssl_mult}: {len(self.indices)} samples (after repetition)")
                elif ssl_type == 2:
                    # Unlabeled data
                    unlabeled_mask = ssl_data['SSL_SPLIT'] == 'UNLABELED'
                    unlabeled_indices = ssl_data[unlabeled_mask].index.tolist()
                    self.indices = np.array([train_indices[i] for i in unlabeled_indices if i < len(train_indices)])
                    print(f"SSL Unlabeled split: {len(self.indices)} samples")
                else:
                    # Use all training data
                    self.indices = train_indices
            else:
                # No SSL split, use all training data
                self.indices = train_indices
                print(f"Split {self.split}: {len(self.indices)} samples")
        elif self.split == "VAL":
            self.indices = indices[train_end:val_end]
            print(f"Split {self.split}: {len(self.indices)} samples")
        elif self.split == "TEST":
            self.indices = indices[val_end:]
            print(f"Split {self.split}: {len(self.indices)} samples")
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'")
        
        # Calculate normalization statistics from training data
        if self.split == "TRAIN":
            train_rgb = self.rgb_data[self.indices]
            train_ndsm = self.ndsm_data[self.indices]
            
            # Image statistics
            self.rgb_mean = train_rgb.mean()
            self.rgb_std = train_rgb.std()
            
            # Target statistics (for normalization)
            self.ndsm_mean = train_ndsm.mean()
            self.ndsm_std = train_ndsm.std()
            
            print(f"RGB stats: mean={self.rgb_mean:.2f}, std={self.rgb_std:.2f}")
            print(f"NDSM stats: mean={self.ndsm_mean:.2f}, std={self.ndsm_std:.2f}")
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get actual index in full dataset
        actual_idx = self.indices[index]
        
        # Load RGB and NDSM
        rgb = self.rgb_data[actual_idx].astype(np.float32)  # (256, 256, 3)
        ndsm = self.ndsm_data[actual_idx].astype(np.float32)  # (256, 256, 1)
        
        # Transpose to (C, H, W) format for PyTorch
        rgb = rgb.transpose(2, 0, 1)  # (3, 256, 256)
        ndsm = ndsm.transpose(2, 0, 1)  # (1, 256, 256)
        
        # Normalize RGB
        if isinstance(self.mean, (float, int)):
            rgb = (rgb - self.mean) / self.std
        else:
            # Reshape for broadcasting: (3,) -> (3, 1, 1)
            mean_reshaped = self.mean.reshape(3, 1, 1) if hasattr(self.mean, 'reshape') else np.array(self.mean).reshape(3, 1, 1)
            std_reshaped = self.std.reshape(3, 1, 1) if hasattr(self.std, 'reshape') else np.array(self.std).reshape(3, 1, 1)
            rgb = (rgb - mean_reshaped) / std_reshaped
        
        # Data augmentation for training
        if self.split == "TRAIN":
            # Random horizontal flip
            if np.random.rand() > 0.5:
                rgb = rgb[:, :, ::-1].copy()
                ndsm = ndsm[:, :, ::-1].copy()
            
            # Random vertical flip
            if np.random.rand() > 0.5:
                rgb = rgb[:, ::-1, :].copy()
                ndsm = ndsm[:, ::-1, :].copy()
            
            # Random padding crop (if specified)
            if self.pad is not None and self.pad > 0:
                c, h, w = rgb.shape
                
                # Add padding
                rgb_padded = np.zeros((c, h + 2 * self.pad, w + 2 * self.pad), dtype=rgb.dtype)
                rgb_padded[:, self.pad:-self.pad, self.pad:-self.pad] = rgb
                
                ndsm_padded = np.zeros((1, h + 2 * self.pad, w + 2 * self.pad), dtype=ndsm.dtype)
                ndsm_padded[:, self.pad:-self.pad, self.pad:-self.pad] = ndsm
                
                # Random crop
                i = np.random.randint(0, 2 * self.pad + 1)
                j = np.random.randint(0, 2 * self.pad + 1)
                
                rgb = rgb_padded[:, i:i+h, j:j+w].copy()
                ndsm = ndsm_padded[:, i:i+h, j:j+w].copy()
        
        # Convert to tensors
        rgb = torch.from_numpy(rgb)
        ndsm = torch.from_numpy(ndsm)
        
        # Target is the height map (not normalized - will be normalized in training loop)
        return rgb, ndsm.squeeze(0)  # Return (3, 256, 256) and (256, 256)
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def extra_repr(self) -> str:
        lines = [f"Split: {self.split}"]
        lines.append(f"Samples: {len(self.indices)}")
        return '\n'.join(lines)
