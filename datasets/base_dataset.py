"""
Base dataset class for all regression datasets.
"""

import os
import torch
import torchvision
from typing import Dict, Any, Optional, Tuple
import numpy as np


class BaseDataset(torchvision.datasets.VisionDataset):
    """
    Abstract base class for regression datasets.
    
    All dataset implementations should inherit from this class and implement
    the required methods.
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
        """
        Initialize base dataset.
        
        Args:
            config: Dataset configuration dictionary
            split: Dataset split (train, val, test)
            ssl_type: SSL split type (0=all, 1=labeled, 2=unlabeled)
            ssl_postfix: Postfix for SSL file list
            ssl_mult: Multiplier for labeled samples
            mean: Image mean for normalization
            std: Image std for normalization
            pad: Padding parameter for augmentation
        """
        root = config.get("data_root")
        if root is None:
            raise ValueError("config must contain 'data_root'")
        
        super().__init__(root)
        
        self.config = config
        self.split = split.upper()
        self.ssl_type = ssl_type
        self.ssl_postfix = ssl_postfix
        self.ssl_mult = ssl_mult
        self.pad = pad
        
        # Image normalization
        if mean is None:
            mean = np.array([0.0, 0.0, 0.0])
        if std is None:
            std = np.array([1.0, 1.0, 1.0])
        
        self.mean = mean
        self.std = std
        
        # Data storage
        self.file_list = []
        self.targets = []
        
    def _load_file_list(self):
        """
        Load file list from CSV or generate it.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _load_file_list")
    
    def _get_image_path(self, index: int) -> str:
        """
        Get image path for given index.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _get_image_path")
    
    def _load_image(self, path: str) -> np.ndarray:
        """
        Load and preprocess image from path.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _load_image")
    
    def _get_target(self, index: int) -> float:
        """
        Get target value for given index.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _get_target")
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float]:
        """
        Get item by index.
        
        Returns:
            Tuple of (image_tensor, target_value)
            - image_tensor: (C, H, W) normalized tensor
            - target_value: float target value
        """
        # Load image
        image_path = self._get_image_path(index)
        image = self._load_image(image_path)
        
        # Apply augmentation (horizontal flip)
        if self.split == "TRAIN" and self.config.get("horizontal_flip", True):
            if np.random.randint(0, 2) == 0:
                image = image[:, :, ::-1]  # Horizontal flip
        
        # Apply padding augmentation
        if self.pad is not None and self.split == "TRAIN":
            c, h, w = image.shape
            temp = np.zeros((c, h + 2 * self.pad, w + 2 * self.pad), dtype=image.dtype)
            temp[:, self.pad:-self.pad, self.pad:-self.pad] = image
            i, j = np.random.randint(0, 2 * self.pad, 2)
            image = temp[:, i:(i + h), j:(j + w)]
        
        # Normalize image
        if isinstance(self.mean, (float, int)):
            image = image - self.mean
        else:
            image = image - self.mean.reshape(-1, 1, 1)
        
        if isinstance(self.std, (float, int)):
            image = image / self.std
        else:
            image = image / self.std.reshape(-1, 1, 1)
        
        # Get target
        target = self._get_target(index)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float()
        target_value = float(target)
        
        return image_tensor, target_value
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.file_list)
    
    def get_target_stats(self) -> Tuple[float, float]:
        """
        Get target statistics (mean, std).
        
        Returns:
            Tuple of (mean, std)
        """
        if len(self.targets) == 0:
            return 0.0, 1.0
        
        targets_array = np.array(self.targets)
        mean = float(np.mean(targets_array))
        std = float(np.std(targets_array))
        
        if std == 0:
            std = 1.0
        
        return mean, std

