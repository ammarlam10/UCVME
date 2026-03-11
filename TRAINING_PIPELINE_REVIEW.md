# Bayern Forest Height Training Pipeline - Complete Review

**Date**: 2026-03-11  
**Dataset**: Bayern Forest Height (RGB → Height maps)  
**Model**: UNET with uncertainty estimation  
**Task**: Pixel-wise regression (256×256)  
**Status**: ✅ **BUG FIXED - Ready for experiments**

---

## Executive Summary

Completed comprehensive review of the Bayern Forest Height training pipeline. Found and **fixed a critical bug** in the SSL consistency loss that was preventing proper semi-supervised learning for pixel-wise regression.

### Key Findings:
- ✅ Data augmentation is correct (RGB and NDSM synchronized)
- ✅ Model architecture is appropriate (UNET with skip connections)
- ✅ Labeled data loss is correct (uncertainty-weighted NLL)
- 🔧 **SSL consistency loss was broken** → **NOW FIXED**
- ✅ Validation loop is correct
- ✅ Metrics computation is correct

---

## 1. Data Pipeline ✅

### Data Loading
- **Format**: HDF5 files with 'rgb' and 'ndsm' keys
- **Input**: RGB images (256×256×3), range [0-255]
- **Target**: Height maps (256×256×1), range [-2m to 32m]
- **Splits**: 80% train, 10% val, 10% test

### Data Augmentation (Training Only)
```python
# Applied to BOTH RGB and NDSM with same random seed
1. Random horizontal flip (50% probability)
2. Random vertical flip (50% probability)  
3. Random crop with padding (pad=5)
   - Pad by 5 pixels → 266×266
   - Random crop back to 256×256
```

**Verification**: Created visualization scripts that confirm:
- ✅ Same augmentation applied to RGB and NDSM
- ✅ Pixel correspondence maintained
- ✅ No misalignment between input and target

### Normalization
- **RGB**: (x - 0.0) / 1.0 → essentially no normalization, keeps [0-255]
- **Target**: (y - 12.26) / 11.05 → normalized in loss computation
  - y_mean = 12.26m (average height)
  - y_std = 11.05m (height std dev)

---

## 2. Model Architecture ✅

### UNET Structure
```
Input: (B, 3, 256, 256) RGB

Encoder:
  - DoubleConv(3 → 64)
  - Down1: MaxPool + DoubleConv(64 → 128)
  - Down2: MaxPool + DoubleConv(128 → 256)
  - Down3: MaxPool + DoubleConv(256 → 512)

Decoder:
  - Up1: Upsample + DoubleConv(512 → 256) + skip from Down2
  - Up2: Upsample + DoubleConv(256 → 128) + skip from Down1
  - Up3: Upsample + DoubleConv(128 → 64) + skip from DoubleConv

Output Heads:
  - Mean: Conv2d(64 → 1)
  - Log-Variance: Conv2d(64 → 1)

Output: mean (B, 1, 256, 256), var (B, 1, 256, 256)
```

**Features**:
- ✅ Skip connections preserve spatial details
- ✅ MC Dropout (p=0.2) for uncertainty estimation
- ✅ Dual heads for mean and variance
- ✅ Appropriate depth (3 levels) for 256×256 images

**Parameters**: ~1.8M (features=32) or ~7.2M (features=64)

---

## 3. Loss Functions

### 3.1 Supervised Loss (Labeled Data) ✅

**Uncertainty-weighted negative log-likelihood**:

```python
# Normalize target
outcome_norm = (outcome - y_mean) / y_std  # (B, H, W)

# Compute loss for each model
loss_mse = (mean - outcome_norm) ** 2  # (B, H, W)
loss = 0.5 * (exp(-(var + var_1)/2) * loss_mse + (var + var_1)/2)
loss_reg = loss.mean()  # Average over all pixels and batch
```

**Why this is correct**:
- ✅ Normalizes target before computing loss
- ✅ Uses uncertainty weighting (high uncertainty → lower weight)
- ✅ Averages over all pixels (treats each pixel as independent sample)
- ✅ Combines predictions from both models

### 3.2 SSL Consistency Loss (Unlabeled Data) ✅ FIXED

**Purpose**: Enforce that both models predict similar heights for the same pixels.

**Process**:
1. Forward pass unlabeled data through both models
2. Generate pseudo-labels by averaging predictions over MC samples
3. Compute consistency loss between each model and pseudo-label

**Fixed implementation**:
```python
# Generate pseudo-labels (MC sampling)
with torch.no_grad():
    for samp in range(samp_ssl):
        mean_0, var_0 = model(X_unlabeled)  # (B, H, W)
        mean_1, var_1 = model_1(X_unlabeled)
        # Accumulate samples...

avg_mean = (mean_0 + mean_1) / 2  # (B, H, W) - pseudo-label
avg_var = (var_0 + var_1) / 2

# Consistency loss: each model should match pseudo-label
pred_0 = model(X_unlabeled)  # (B, H, W)
loss_mse_0 = ((pred_0 - avg_mean) ** 2)  # (B, H, W)
loss_cps = 0.5 * (exp(-avg_var) * loss_mse_0 + avg_var)
loss_cps_final = loss_cps.mean()  # Average over pixels and batch
```

**What was fixed**:
- 🔧 Changed `.view(-1)` → `.squeeze(1)` to preserve spatial dims
- 🔧 Now computes per-pixel consistency within each image
- 🔧 No longer mixes pixels across different images
- ✅ Maintains backward compatibility with image-level regression

### 3.3 Total Loss

```python
loss_total = loss_reg + w_ulb * loss_reg_cps + ((var_1 - var) ** 2).mean()
```

Where:
- `loss_reg`: Supervised loss on labeled data
- `loss_reg_cps`: SSL consistency loss on unlabeled data (NOW FIXED)
- `((var_1 - var) ** 2).mean()`: Variance consistency between models
- `w_ulb = 10.0`: Weight for SSL loss (10× supervised loss!)

---

## 4. Training Configuration

### Current Settings (20% labeled)
```yaml
training:
  num_epochs: 60
  lr: 0.0001          # Conservative learning rate
  weight_decay: 0.001  # L2 regularization
  batch_size: 16       # For 256×256 pixel-wise
  num_workers: 6       # Parallel data loading

ssl:
  ssl_mult: 3      # Repeat labeled 3× per epoch
  w_ulb: 10.0      # SSL loss weight (10× supervised)
  samp_ssl: 5      # MC samples for pseudo-labels
  samp_fq: 5       # MC samples for validation

target:
  y_mean: 12.26    # Height mean (meters)
  y_std: 11.05     # Height std dev (meters)
```

### Analysis

**Strengths**:
- ✅ Batch size appropriate for memory (16 for 256×256)
- ✅ Multiple workers for efficient data loading (6)
- ✅ MC sampling for uncertainty (5 samples)
- ✅ Target normalization based on dataset statistics

**Potential Concerns**:
- ⚠️ `w_ulb = 10.0` is very aggressive (10× supervised loss)
  - Now that SSL is fixed, this might be too high
  - Consider experimenting with 1.0, 5.0, 10.0
- ⚠️ `ssl_mult = 3` repeats labeled data 3× per epoch
  - With 20% labeled: 2,310 → 6,930 samples
  - Unlabeled: 9,242 samples
  - Ratio is reasonable but could experiment

---

## 5. Validation & Metrics ✅

### Validation Loop
- ✅ Correctly handles pixel-wise outputs
- ✅ Uses MC sampling (samp_fq=5) for uncertainty
- ✅ Averages predictions from both models
- ✅ Denormalizes before computing metrics

### Metrics
- **MAE**: Mean Absolute Error (meters)
- **RMSE**: Root Mean Squared Error (meters)
- **R²**: Coefficient of determination

**Computation**:
- ✅ Predictions flattened: all pixels treated as independent samples
- ✅ Denormalized before metrics: (pred * y_std + y_mean)
- ✅ Standard sklearn metrics used

---

## 6. Potential Issues & Recommendations

### ✅ Fixed Issues

1. **SSL Consistency Loss** - FIXED
   - Was: `.view(-1)` destroyed spatial structure
   - Now: `.squeeze(1)` preserves spatial dims
   - Impact: SSL will now properly leverage unlabeled data

### ⚠️ Considerations for Future

1. **SSL Weight Tuning**
   - Current `w_ulb = 10.0` might be too aggressive
   - Recommend experimenting: [1.0, 5.0, 10.0]
   - Monitor if SSL loss dominates supervised loss

2. **Padding Strategy**
   - Current: Zero padding for random crop
   - Alternative: Reflection padding (more natural for images)
   - Impact: Minor, current approach is acceptable

3. **RGB Normalization**
   - Current: No normalization (keeps [0-255])
   - Standard: ImageNet normalization or [0,1] scaling
   - Impact: Model learns to handle [0-255] range, works but non-standard

4. **Batch Size**
   - Current: 16 (safe for 32GB GPU)
   - Could try: 32 or 64 if memory allows
   - Impact: Larger batch → more stable gradients

5. **Learning Rate Schedule**
   - Current: Step decay every 20 epochs
   - Alternative: Cosine annealing, ReduceLROnPlateau
   - Impact: Minor, current approach is reasonable

6. **Additional Augmentations**
   - Current: Flips + random crop
   - Could add: Rotation (90°, 180°, 270°), Gaussian blur
   - Impact: More augmentation → better generalization
   - Note: Must apply same transform to RGB and NDSM!

---

## 7. Next Steps

### Immediate Actions

1. ✅ **Bug Fix Applied**: SSL consistency loss now correct
2. ✅ **Tests Passed**: All dimension handling verified
3. ✅ **Visualizations Created**: Data pipeline verified

### Recommended Experiments

1. **Re-run all Bayern experiments** with fixed SSL:
   ```bash
   # 5% labeled
   python3 ucvme_age.py --config configs/bayern_forest_height_unet_5percent.yaml
   
   # 10% labeled
   python3 ucvme_age.py --config configs/bayern_forest_height_unet_10percent.yaml
   
   # 20% labeled
   python3 ucvme_age.py --config configs/bayern_forest_height_unet_20percent.yaml
   ```

2. **Compare results** before/after fix:
   - Expect improvement in SSL scenarios
   - Especially at low labeled percentages (5%, 10%)

3. **Tune SSL weight** (optional):
   - Try `w_ulb = [1.0, 5.0, 10.0]`
   - See which gives best val performance

4. **Ablation study** (optional):
   - SSL on vs off
   - Different augmentation strategies
   - Different batch sizes

---

## 8. Files Modified

### Core Changes
- ✅ `ucvme_age.py` (lines 700-799): Fixed SSL consistency loss
- ✅ `datasets/bayern_forest_height.py` (line 190): Added alignment comment

### New Files Created
- ✅ `scripts/visualize_bayern_pipeline_fast.py`: Fast data visualization
- ✅ `scripts/test_ssl_loss_fix.py`: Unit tests for fix
- ✅ `scripts/validate_ssl_fix_with_model.py`: Integration test with UNET
- ✅ `BAYERN_TRAINING_ANALYSIS.md`: Detailed analysis
- ✅ `SSL_BUG_FIX_SUMMARY.md`: Fix documentation
- ✅ `TRAINING_PIPELINE_REVIEW.md`: This document

### Visualizations Generated
- ✅ `output/bayern_viz_1_raw.png`: Raw HDF5 data
- ✅ `output/bayern_viz_2_normalized.png`: After normalization
- ✅ `output/bayern_viz_3_augmentation_variations.png`: Augmentation examples
- ✅ `output/bayern_viz_4_pixel_correspondence.png`: Alignment verification
- ✅ `output/bayern_viz_5_augmentation_comparison.png`: Before/after comparison

---

## 9. Conclusion

The Bayern Forest Height training pipeline is now **correct and ready for experiments**:

### What Works ✅
- Data loading and augmentation (synchronized RGB + NDSM)
- Model architecture (UNET with uncertainty)
- Supervised loss (pixel-wise NLL with uncertainty weighting)
- SSL consistency loss (NOW FIXED for pixel-wise)
- Validation and metrics
- Target normalization

### What Was Fixed 🔧
- SSL consistency loss now preserves spatial dimensions
- Per-pixel consistency enforced within each image
- No pixel mixing across different images
- Backward compatible with image-level regression

### Expected Improvements 📈
With the SSL fix, you should see:
- Better performance at low labeled percentages (5%, 10%, 20%)
- Unlabeled data now properly contributes to learning
- More stable training with consistency regularization
- Better uncertainty estimates

### Confidence Level
**HIGH** - The training pipeline is theoretically sound and correctly implemented for pixel-wise height prediction. The SSL bug was the only critical issue, and it's now fixed and tested.

---

## Appendix: Technical Deep Dive

### A. Loss Function Derivation

**Uncertainty-weighted NLL** (for Gaussian likelihood):
```
p(y|x) = N(y; μ(x), σ²(x))
-log p(y|x) = 0.5 * log(2π) + 0.5 * log(σ²) + 0.5 * (y - μ)²/σ²
            = 0.5 * log(σ²) + 0.5 * (y - μ)²/σ²
            = 0.5 * (exp(-log_var) * (y - μ)² + log_var)  [using log_var = log(σ²)]
```

This is exactly what the code implements:
```python
loss = 0.5 * (exp(-var) * mse + var)
```

### B. SSL Consistency Regularization

**Idea**: Two models should agree on unlabeled data.

**Implementation**:
1. Generate pseudo-labels by averaging both models (with MC sampling)
2. Enforce each model to match the pseudo-label
3. Weight by uncertainty (high uncertainty → lower weight)

**Why it works**:
- Models learn from each other's predictions
- Averaging reduces noise
- Uncertainty weighting focuses on confident predictions
- Consistency acts as regularization

### C. Dual Model Training

**Why two models?**
- Better uncertainty estimation (epistemic + aleatoric)
- Consistency regularization between models
- Ensemble effect (average at inference)

**Variance consistency term**:
```python
((var_1 - var) ** 2).mean()
```
Encourages both models to have similar uncertainty estimates.

---

## References

- **Visualizations**: `output/bayern_viz_*.png`
- **Test scripts**: `scripts/test_ssl_loss_fix.py`, `scripts/validate_ssl_fix_with_model.py`
- **Analysis docs**: `BAYERN_TRAINING_ANALYSIS.md`, `SSL_BUG_FIX_SUMMARY.md`
