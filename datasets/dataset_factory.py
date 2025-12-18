"""
Dataset factory for creating dataset instances.
"""

from typing import Dict, Any, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datasets.so2sat_pop import So2SatPOPDataset
from utils.config_loader import load_dataset_config, validate_dataset_config


# Registry of available datasets
DATASET_REGISTRY = {
    "so2sat_pop": So2SatPOPDataset,
    # Add more datasets here as they are implemented
    # "bayern_forest_height": BayernForestDataset,
    # "utkface_all": UTKFaceDataset,
}


def get_dataset(
    dataset_name: str,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    **kwargs
):
    """
    Factory function to create dataset instances.
    
    Args:
        dataset_name: Name of the dataset (must be in registry)
        config: Configuration dictionary (optional, will load default if not provided)
        config_path: Path to config file (optional, overrides default)
        **kwargs: Additional arguments to pass to dataset constructor
        
    Returns:
        Dataset instance
    """
    if dataset_name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Available datasets: {available}"
        )
    
    # Load config if not provided
    if config is None:
        if config_path:
            from utils.config_loader import load_config
            config = load_config(config_path)
        else:
            config = load_dataset_config(dataset_name)
    
    # Validate config
    validate_dataset_config(config)
    
    # Get dataset class
    dataset_class = DATASET_REGISTRY[dataset_name]
    
    # Create instance
    dataset = dataset_class(config=config, **kwargs)
    
    return dataset


def list_available_datasets():
    """List all available datasets."""
    return list(DATASET_REGISTRY.keys())

