# UNET Implementation for Pixel-Wise Regression

## Summary

Successfully implemented UNET model with uncertainty estimation for pixel-wise regression tasks, specifically for the Bayern Forest Height dataset. The implementation maintains full backward compatibility with existing image-level regression models (ResNet50, EfficientNetB0).

## What Was Implemented

### 1. UNET Model (`models/unet.py`)
- **Architecture**: 3 downsampling layers + 3 upsampling layers
- **Features**: 
  - Dual output heads (mean and log-variance)
  - MC Dropout for uncertainty estimation
  - Skip connections for better gradient flow
  - Configurable base features (default: 64)
- **Parameters**: ~7.2M (features=64) or ~1.8M (features=32)

### 2. Bayern Forest Height Dataset (`datasets/bayern_forest_height.py`)
- **Data Source**: `/work/ammar/sslrp/data/Bayern_forest_height_reduced`
- **Format**: HDF5 files with 'rgb' and 'ndsm' keys
- **Splits**: 
  - Train: 80% (11,552 samples)
  - Val: 10% (1,444 samples)
  - Test: 10% (1,444 samples)
- **Total**: 14,440 samples from 40 files
- **Input**: RGB images (256×256×3)
- **Target**: Height maps/NDSM (256×256×1)
- **Data Augmentation**: Random flips, random crops with padding

### 3. Pixel-Wise Utilities (`utils_pixelwise.py`)
- **Loss Functions**:
  - `pixel_wise_loss_with_uncertainty()`: NLL loss with uncertainty weighting
  - `pixel_wise_mse_loss()`: Simple MSE for pixel-wise outputs
- **Metrics**:
  - `compute_pixelwise_metrics()`: MAE, RMSE, R² for spatial outputs
- **Helpers**:
  - `is_pixelwise_output()`: Detect if model output is spatial or scalar
  - `denormalize_pixelwise()`: Denormalize predictions

### 4. Modified Training Loop (`ucvme_age.py`)
- **Automatic Detection**: Detects pixel-wise vs image-level outputs
- **Backward Compatible**: Works with ResNet50, EfficientNetB0, and UNET
- **Handles Both**:
  - Image-level: (B, 1) outputs
  - Pixel-wise: (B, H, W) or (B, 1, H, W) outputs
- **Loss Computation**: Adapts based on output type
- **Metrics**: Flattens pixel-wise outputs for consistent metrics

### 5. Configuration (`configs/bayern_forest_height_unet.yaml`)
- Model: UNET with drp_p=0.2
- Batch size: 8 (memory-efficient for pixel-wise)
- Learning rate: 0.0001
- Epochs: 50
- Target normalization: mean=15.0, std=10.0

## Dataset Statistics

```
Total files: 40 HDF5 files
Total samples: 14,440
Train split: 11,552 samples (80%)
Val split: 1,444 samples (10%)
Test split: 1,444 samples (10%)

RGB values: min=1.0, max=254.0
  Training mean: 69.37, std: 48.07

NDSM (height) values: min=-10.23m, max=42.98m
  Training mean: 12.26m, std: 11.05m
```

## Files Created/Modified

### New Files
1. `models/unet.py` - UNET model with uncertainty
2. `datasets/bayern_forest_height.py` - Dataset loader
3. `utils_pixelwise.py` - Pixel-wise utilities
4. `configs/bayern_forest_height_unet.yaml` - Configuration
5. `test_bayern_unet.py` - Test script
6. `inspect_bayern_dataset.py` - Dataset inspection script

### Modified Files
1. `models/__init__.py` - Added unet_unc import
2. `datasets/__init__.py` - Added BayernForestHeightDataset
3. `ucvme_age.py` - Updated training loop for pixel-wise support
4. `requirements_pip.txt` - Added h5py>=3.0.0

## How to Use

### 1. Run Training
```bash
python3 ucvme_age.py --config configs/bayern_forest_height_unet.yaml
```

### 2. Test Implementation
```bash
python3 test_bayern_unet.py
```

### 3. Inspect Dataset
```bash
python3 inspect_bayern_dataset.py
```

### 4. Using Docker
```bash
# Build image (if needed)
docker build -t ucvme:latest .

# Run training
docker run --rm --gpus all \
  -v /work/ammar/sslrp/data:/workspace/data \
  -v /work/ammar/sslrp/UCVME:/workspace/ucvme \
  ucvme:latest \
  python3 /workspace/ucvme/ucvme_age.py \
  --config /workspace/ucvme/configs/bayern_forest_height_unet.yaml
```

## Backward Compatibility

The implementation is fully backward compatible:

### Image-Level Regression (ResNet50, EfficientNetB0)
```yaml
model:
  name: resnet50  # or efficientnetb0
data:
  dataset_name: utkface  # or so2sat_pop_custom
```

### Pixel-Wise Regression (UNET)
```yaml
model:
  name: unet
data:
  dataset_name: bayern_forest_height
```

The training loop automatically detects the output type and applies appropriate loss functions and metrics.

## Key Design Decisions

1. **In-Memory Loading**: Dataset loads all data into memory for faster training (14,440 samples × 256×256 is manageable)

2. **Automatic Detection**: Training loop uses `is_pixelwise_output()` to detect output type rather than requiring explicit configuration

3. **Consistent Interface**: Both pixel-wise and image-level models use the same dual-output interface (mean, variance)

4. **Flattened Metrics**: Pixel-wise predictions are flattened for metrics computation to maintain consistency with image-level metrics

5. **Separate Utilities**: Pixel-wise functions in separate module (`utils_pixelwise.py`) for clean separation of concerns

## Testing Results

All tests passed successfully:
- ✓ Model creation (7.2M parameters)
- ✓ Forward pass (256×256 spatial output)
- ✓ Pixel-wise detection (correctly identified)
- ✓ Loss computation (uncertainty-weighted)
- ✓ Metrics computation (MAE, RMSE, R²)

## Next Steps

1. **Run Full Training**: Execute training for 50 epochs
2. **Hyperparameter Tuning**: Adjust learning rate, batch size, features
3. **SSL Support**: Implement SSL for pixel-wise regression (future work)
4. **Visualization**: Add visualization tools for height map predictions
5. **Model Comparison**: Compare UNET vs other architectures

## Notes

- The UNET model uses bilinear upsampling by default (can be changed to transposed convolutions)
- Dropout is applied after each encoder/decoder block for MC Dropout uncertainty
- The dataset uses fixed train/val/test split with seed=0 for reproducibility
- Target normalization uses approximate statistics (will be recalculated from training data)
