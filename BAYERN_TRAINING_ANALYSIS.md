# Bayern Forest Height Training Pipeline Analysis

## Overview

Analysis of the training pipeline for Bayern Forest Height pixel-wise regression task using UNET model with SSL (Semi-Supervised Learning).

**Date**: 2026-03-11  
**Dataset**: Bayern Forest Height (RGB → Height maps)  
**Model**: UNET with uncertainty estimation  
**Task**: Pixel-wise regression (256×256 height prediction)

---

## ✅ What's Working Correctly

### 1. Data Loading & Augmentation
- **Dataset**: `BayernForestHeightDataset` correctly loads RGB and NDSM from HDF5
- **Augmentation**: ✅ Synchronized transforms applied to both RGB and NDSM
  - Random horizontal flip (50%)
  - Random vertical flip (50%)
  - Random crop with padding (pad=5)
- **Pixel correspondence**: ✅ Verified - same transform applied to input and target
- **Normalization**: 
  - RGB: (x - 0.0) / 1.0 (essentially no normalization, keeps [0-255] range)
  - Target: (y - 12.26) / 11.05 (normalized in loss computation)

### 2. Model Architecture
- **UNET**: 3 down-sampling, 3 up-sampling layers
- **Outputs**: Two heads (mean, log-variance) for uncertainty estimation
- **Dropout**: MC Dropout (p=0.2) for uncertainty quantification
- **Output shape**: (B, 1, 256, 256) for both mean and variance

### 3. Loss Function (Labeled Data)
```python
# Pixel-wise uncertainty-weighted loss
loss_mse = (mean - outcome_norm) ** 2
loss = 0.5 * (exp(-(var + var_1)/2) * loss_mse + (var + var_1)/2)
loss_reg = loss.mean()
```
- ✅ Correct: Averages over all pixels
- ✅ Uses uncertainty weighting (NLL loss)
- ✅ Target is normalized before loss computation

### 4. Dual Model Training
- Two models (`model` and `model_1`) trained jointly
- Helps with uncertainty estimation and regularization
- Variance consistency term: `((var_1 - var) ** 2).mean()`

---

## 🚨 CRITICAL BUG: SSL Consistency Loss for Pixel-Wise Data [✅ FIXED]

### The Problem (NOW FIXED)

In the SSL consistency loss computation (lines 700-758), the code uses `.view(-1)` which **flattens the entire batch** of pixel-wise predictions:

```python
# Line 703-704: WRONG for pixel-wise!
mean1_0 = mean1_raw_0.view(-1)  # Flattens (B, 1, 256, 256) → (B*256*256,)
var1_0 = var1_raw_0.view(-1)
```

This means:
- For pixel-wise UNET output: `(B, 1, 256, 256)` becomes `(B*65536,)` 
- All spatial structure is lost
- Pixels from different images get mixed together
- Consistency loss compares pixels across different images (WRONG!)

### Why This Is Wrong

The SSL consistency loss should enforce that **two models predict similar heights for the same pixel in the same image**. But with `.view(-1)`, it's comparing:
- Pixel (0,0) from image 0 with pixel (0,1) from image 0
- Pixel (255,255) from image 0 with pixel (0,0) from image 1
- Completely random pixel correspondences!

### Expected Behavior

For pixel-wise regression, the consistency loss should:
1. Keep spatial dimensions: `(B, 1, H, W)` → `(B, H, W)`
2. Compute per-pixel consistency within each image
3. Average over pixels AND batch

---

## ✅ FIXES APPLIED

### Fix 1: SSL Consistency Loss (Lines 700-740) - COMPLETED

**Was (WRONG)**:
```python
mean1_0 = mean1_raw_0.view(-1)  # Flattens everything
var1_0 = var1_raw_0.view(-1)
```

**Now (FIXED)**:
```python
# Handle pixel-wise vs image-level outputs
if utils_pixelwise.is_pixelwise_output(mean1_raw_0):
    # Pixel-wise: squeeze channel dim but keep spatial (B, H, W)
    if mean1_raw_0.dim() == 4 and mean1_raw_0.size(1) == 1:
        mean1_0 = mean1_raw_0.squeeze(1)
        var1_0 = var1_raw_0.squeeze(1)
    else:
        mean1_0 = mean1_raw_0
        var1_0 = var1_raw_0
else:
    # Image-level: flatten to 1D (backward compatible)
    mean1_0 = mean1_raw_0.view(-1)
    var1_0 = var1_raw_0.view(-1)
```

Applied to:
- ✅ Lines 705-716 (model 0 sampling)
- ✅ Lines 724-736 (model 1 sampling)
- ✅ Lines 769-786 (consistency loss computation)
- ✅ Lines 798-799 (variance consistency)

### Fix 2: Consistency Loss Computation - COMPLETED

**Was (WRONG)**:
```python
loss_mse_cps_0 = ((all_output_unlb_0_pred_0.view(-1) - avg_mean01)**2)
```

**Now (FIXED)**:
```python
# Handle pixel-wise vs image-level
if utils_pixelwise.is_pixelwise_output(all_output_unlb_0_pred_0):
    if all_output_unlb_0_pred_0.dim() == 4 and all_output_unlb_0_pred_0.size(1) == 1:
        pred_0 = all_output_unlb_0_pred_0.squeeze(1)  # (B, H, W)
    else:
        pred_0 = all_output_unlb_0_pred_0
else:
    pred_0 = all_output_unlb_0_pred_0.view(-1)

loss_mse_cps_0 = ((pred_0 - avg_mean01) ** 2)
```

### Testing - COMPLETED

Created and ran test scripts:
- ✅ `scripts/test_ssl_loss_fix.py` - Unit tests for dimension handling
- ✅ `scripts/validate_ssl_fix_with_model.py` - Validation with actual UNET

All tests passed!

---

## 📊 Impact Assessment

### Current Behavior (with bug)
- SSL consistency loss is computed on **randomly shuffled pixels** from different images
- No meaningful spatial consistency is enforced
- The model might still learn (from labeled data), but SSL component is broken
- `w_ulb = 10.0` means this broken loss has 10× weight of supervised loss!

### After Fix
- SSL consistency will enforce that both models predict similar heights for **corresponding pixels**
- Proper semi-supervised learning will leverage unlabeled data effectively
- Should see improved performance, especially at low labeled data percentages (5%, 10%, 20%)

---

## ✅ What Doesn't Need Fixing

### 1. Labeled Data Loss (Lines 779-819)
The labeled data loss is **correctly implemented** for pixel-wise:
```python
if is_pixelwise:
    # Keeps spatial dimensions
    mean = mean_raw.squeeze(1)  # (B, H, W)
    var = var_raw.squeeze(1)
    
    # Pixel-wise loss
    loss_mse = (mean - outcome_norm) ** 2  # (B, H, W)
    loss = 0.5 * (exp(-var) * loss_mse + var)
    loss_reg_0 = loss.mean()  # Average over all pixels and batch
```

### 2. Validation Loop (Lines 961-975)
The validation loop **correctly handles** pixel-wise outputs:
```python
if is_pixelwise:
    mean1 = mean1_raw.squeeze(1).flatten(1)  # (B, H*W) - keeps batch separate
    var1 = var1_raw.squeeze(1).flatten(1)
```
This is correct because it flattens spatial dims but keeps batch dimension.

### 3. Model Architecture
- UNET architecture is appropriate for pixel-wise regression
- Skip connections help preserve spatial details
- Dual output heads (mean, variance) are correct

### 4. Metrics Computation
- Predictions are correctly flattened for metric computation
- MAE, RMSE, R² are computed on all pixels

---

## 🎯 Recommendations

### Priority 1: Fix SSL Consistency Loss (CRITICAL)
The `.view(-1)` operations in the SSL consistency loss computation must be replaced with proper spatial dimension handling. This is causing the SSL component to be completely broken for pixel-wise regression.

### Priority 2: Verify Training Results
After fixing:
1. Re-run experiments with 5%, 10%, 20% labeled data
2. Compare results before/after fix
3. Expect to see improvement in SSL scenarios

### Priority 3: Add Assertions
Add shape assertions in the training loop to catch dimension mismatches:
```python
assert mean1_0.shape == (B, H, W), f"Expected (B,H,W), got {mean1_0.shape}"
```

### Priority 4: Consider Additional Improvements
1. **Spatial consistency**: Could add spatial smoothness regularization
2. **Edge handling**: Current padding uses zeros - could use reflection padding
3. **Data augmentation**: Could add rotation, color jitter (though geometric is most important)
4. **Batch size**: Currently 16 - could experiment with larger batches if memory allows

---

## 📈 Training Configuration Analysis

### Current Settings (20% labeled)
```yaml
training:
  num_epochs: 60
  lr: 0.0001
  weight_decay: 0.001
  batch_size: 16
  
ssl:
  ssl_mult: 3      # Repeat labeled 3x
  w_ulb: 10.0      # SSL loss weight (10x supervised!)
  samp_ssl: 5      # MC samples for SSL
```

### Concerns
1. **w_ulb = 10.0**: Very high weight for broken SSL loss
   - After fix, this might be too high - consider tuning
   - Typical values: 1.0 - 5.0
   
2. **ssl_mult = 3**: Repeats labeled data 3× per epoch
   - With 20% labeled (2,310 samples) → 6,930 samples
   - Unlabeled: 9,242 samples
   - Ratio is reasonable

3. **Learning rate**: 0.0001 is conservative but safe for pixel-wise regression

---

## 🔍 Summary

### What's Correct ✅
- Data augmentation (synchronized RGB + NDSM)
- Model architecture (UNET with uncertainty)
- Labeled data loss (pixel-wise NLL)
- Validation loop (proper spatial handling)
- Metrics computation

### What Was Broken (Now Fixed) ✅
- **SSL consistency loss**: Was using `.view(-1)` which destroyed spatial structure
- **Impact**: SSL component was not working for pixel-wise regression
- **Severity**: HIGH - affected all SSL experiments (5%, 10%, 20% labeled)

### Action Taken ✅
- Fixed SSL consistency loss to properly handle spatial dimensions
- Maintains backward compatibility with image-level regression
- All tests passed
- Ready for re-running experiments
