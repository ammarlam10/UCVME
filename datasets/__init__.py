from .utk_dta import UTKdta
from .so2sat_pop import So2SatDataset
from .so2sat_pop_custom import So2SatDatasetCustom
from .bayern_forest_height import BayernForestHeightDataset

# Dataset registry for easy selection
DATASET_REGISTRY = {
    'utkface': UTKdta,
    'so2sat_pop': So2SatDataset,
    'so2sat_pop_custom': So2SatDatasetCustom,
    'bayern_forest_height': BayernForestHeightDataset,
}

def get_dataset(dataset_name):
    """Get dataset class by name."""
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[dataset_name_lower]