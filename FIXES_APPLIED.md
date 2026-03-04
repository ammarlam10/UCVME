# Fixes Applied to So2Sat_POP Training Pipeline

## Date: February 15, 2026

## Summary of Changes

All critical issues identified in the analysis have been fixed. The model should now train correctly on the So2Sat_POP dataset.

---

## ✅ Fix #1: Removed Double Target Normalization

**Files Modified:** `datasets/so2sat_pop_custom.py`

**Changes:**
- **Line 218-220**: Removed normalization from dataset loader
- Target values are now returned as raw population values
- Normalization is handled only in the training loop (`ucvme_age.py` line 761)

**Before:**
```python
raw_target = np.float32(self.outcome[index][self.header.index(t)])
normalized_target = (raw_target - self.normalize_mean) / self.normalize_std
target.append(normalized_target)
```

**After:**
```python
raw_target = np.float32(self.outcome[index][self.header.index(t)])
target.append(raw_target)
```

---

## ✅ Fix #2: Corrected Normalization Parameters

**Files Modified:** `configs/so2sat_pop_efficientnetb0_20percent_fixed.yaml`

**Changes:**
- Calculated actual mean and std from filtered training data in `FileList.csv`
- Updated config with correct values

**Before:**
```yaml
y_mean: 1085.0
y_std: 2800.0
```

**After:**
```yaml
y_mean: 1872.5640   # Calculated from 60,952 training samples
y_std: 3398.8516    # Calculated from filtered FileList.csv
```

**Calculation Method:**
```bash
awk -F',' 'NR>1 && $2=="TRAIN" {sum+=$3; sumsq+=$3*$3; count++} \
  END {mean=sum/count; printf "mean: %.4f, std: %.4f\n", mean, sqrt(sumsq/count - mean*mean)}' \
  FileList.csv
```

---

## ✅ Fix #3: Corrected Band Extraction (No Change Needed)

**Files Modified:** `datasets/so2sat_pop_custom.py`

**Analysis:**
- Current code uses indices `[3, 2, 1]` which extracts:
  - Index 3 = Band 4 (Red)
  - Index 2 = Band 3 (Green)
  - Index 1 = Band 2 (Blue)
- This is correct for RGB order as confirmed by user

**Updated Comments:**
```python
# Band 4 (index 3) = Red, Band 3 (index 2) = Green, Band 2 (index 1) = Blue
image_bands = data[:, :, [3, 2, 1]].astype(np.float32)  # RGB order
```

---

## ✅ Fix #4: Removed Double Image Normalization

**Files Modified:** `datasets/so2sat_pop_custom.py`

**Changes:**
- **Lines 150-158**: Kept only clipping and min-max normalization
- **Lines 205-206**: Removed mean/std normalization completely

**Before:**
```python
# Step 1: Clip to [0, 4000]
image_bands = np.clip(image_bands, 0, 4000)

# Step 2: Divide by 4000
image_bands = image_bands / 4000.0

# Step 3: Min-max normalization
arr_min = image_bands.min()
arr_max = image_bands.max()
if arr_max > arr_min:
    image_bands = (image_bands - arr_min) / (arr_max - arr_min)

# ... later in code ...
# Apply mean/std normalization
photo = (photo - mean) / std
```

**After:**
```python
# Step 1: Clip to [0, 4000]
image_bands = np.clip(image_bands, 0, 4000)

# Step 2: Min-max normalization to [0, 1]
arr_min = image_bands.min()
arr_max = image_bands.max()
if arr_max > arr_min:
    image_bands = (image_bands - arr_min) / (arr_max - arr_min)

# Note: Image normalization (clipping and min-max) is already applied above
# No additional mean/std normalization needed
```

---

## ✅ Fix #5: GPU Configuration

**Changes:**
- Removed `-e CUDA_VISIBLE_DEVICES=5` from docker command
- Kept only `--gpus device=5` flag

**Before:**
```bash
docker run --rm --gpus device=5 -e CUDA_VISIBLE_DEVICES=5 ...
```

**After:**
```bash
docker run --rm --gpus device=5 ...
```

**Explanation:**
- `--gpus device=5` maps host GPU 5 to container GPU 0
- Setting `CUDA_VISIBLE_DEVICES=5` inside container looks for GPU 5 (doesn't exist)
- This caused the model to fall back to CPU

---

## ✅ Fix #6: Reduced Batch Size

**Files Modified:** `configs/so2sat_pop_efficientnetb0_20percent_fixed.yaml`

**Changes:**
- Reduced batch_size from 256 to 128

**Before:**
```yaml
batch_size: 256
```

**After:**
```yaml
batch_size: 128
```

**Rationale:**
- With 12,190 labeled samples (24,380 after duplication)
- Batch size 128 gives 190 batches per epoch (vs 95 with batch size 256)
- More batches = better gradient estimates and more stable training

---

## New Configuration File

Created: `configs/so2sat_pop_efficientnetb0_20percent_fixed.yaml`

This config includes all fixes and uses 20% SSL (20% labeled, 80% unlabeled).

---

## Expected Improvements

With these fixes, you should see:

1. **Correct target scaling**: Model trains on properly normalized targets
2. **Better feature extraction**: Correct RGB bands with appropriate normalization
3. **GPU acceleration**: Model runs on GPU, ~10-20x faster than CPU
4. **More stable training**: Reduced batch size improves gradient estimates
5. **Meaningful metrics**: R², MAE, and RMSE should be in reasonable ranges

**Previous Results (with bugs):**
- R²: -0.136
- MAE: 1160.42
- RMSE: 3347.90

**Expected Results (after fixes):**
- R²: Should be positive (> 0.5 for good performance)
- MAE: Should be much lower (< 500 for reasonable performance)
- RMSE: Should be much lower (< 1000 for reasonable performance)

---

## Files Modified

1. `datasets/so2sat_pop_custom.py` - Fixed normalization and band extraction
2. `configs/so2sat_pop_efficientnetb0_20percent_fixed.yaml` - New config with correct parameters

## Files Created

1. `configs/so2sat_pop_efficientnetb0_20percent_fixed.yaml` - Fixed configuration
2. `FIXES_APPLIED.md` - This documentation file
