"""
Configuration loader for datasets and experiments.
"""

import os
import yaml
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary containing configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_dataset_config(dataset_name: str, config_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Load default dataset configuration.
    
    Args:
        dataset_name: Name of the dataset
        config_dir: Directory containing config files (default: configs/datasets)
        
    Returns:
        Dictionary containing dataset configuration
    """
    if config_dir is None:
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "datasets")
    
    config_path = os.path.join(config_dir, f"{dataset_name}.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found for dataset '{dataset_name}': {config_path}\n"
            f"Available configs: {os.listdir(config_dir)}"
        )
    
    return load_config(config_path)


def validate_dataset_config(config: Dict[str, Any]) -> bool:
    """
    Validate dataset configuration has required fields.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_fields = [
        "name",
        "data_root",
        "data_type",
        "csv_target_column",
        "image_path_template",
    ]
    
    missing = [field for field in required_fields if field not in config]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")
    
    return True

