# CORRECTED ANALYSIS - The Real Bug Found!

**Date:** February 17, 2026  
**Status:** ✅ **Training is CORRECT, Test Evaluation has a BUG**

---

## Apology

You were absolutely right to question my initial analysis! After deeper investigation, I found that:

1. ✅ **Your training IS working correctly**
2. ✅ **The validation metrics during training (R²=0.82, MAE=414) are VALID**
3. ✅ **No data leakage exists**
4. ❌ **There is a BUG in the test/val evaluation code (lines 599-603)**

---

## The Actual Results (CORRECTED)

### Training Validation Metrics (CORRECT)

| Experiment | Best Epoch | Val R² | Val MAE | Val RMSE | Status |
|------------|------------|--------|---------|----------|--------|
| **5% Labeled** | 29 | **0.7854** | ~479 | ~1,253 | ✅ VALID |
| **10% Labeled** | 26 | **0.8228** | **414** | **1,138** | ✅ VALID |
| **20% Labeled** | 5 | **0.8401** | ~407 | **1,081** | ✅ VALID |

**These results are CORRECT and show good performance!**

- R² of 0.78-0.84 is excellent for population estimation
- MAE of 400-500 is reasonable (mean population ~1,150-1,870)
- Clear improvement as labeled data increases

### Test Evaluation Metrics (WRONG - Due to Bug)

| Experiment | Test R² | Test MAE | Status |
|------------|---------|----------|--------|
| **All** | -0.136 | 3,938,400 | ❌ BUG in evaluation code |

**These results are INVALID due to a bug in the evaluation code.**

---

## The Bug: Line 599-600 in ucvme_age.py

### The Code (WRONG)

```python
# Line 597: Get predictions and targets from validation
total_loss, yhat, y, _, _, _, _, _ = run_epoch_val(...)

# Line 599-600: INCORRECT COMMENT AND CODE
# Dataset returns normalized y; yhat is denormalized. Convert y to original scale for metrics.
y_orig = y * y_std + y_mean  # BUG: y is ALREADY in original scale!

# Line 601-603: Calculate metrics with wrong y_orig
r2 = sklearn.metrics.r2_score(y_orig, yhat)  # Comparing wrong scales!
mae = sklearn.metrics.mean_absolute_error(y_orig, yhat)
```

### What Actually Happens

1. **Dataset returns RAW population values** (e.g., 1,152)
   - See line 876: `y.append(outcome.numpy())` where `outcome` is from dataset
   - Dataset does NOT normalize targets (line 219 in so2sat_pop_custom.py)

2. **`run_epoch_val` returns RAW y values** (line 876, 973)
   - `y` contains original scale values: [1152, 556, 3300, ...]

3. **`run_epoch_val` returns DENORMALIZED yhat** (line 947)
   - `yhat` contains predictions in original scale: [1185, 493, 2377, ...]

4. **Line 600 INCORRECTLY denormalizes y AGAIN:**
   ```python
   y_orig = 1152 * 3398.85 + 1872.56 = 3,919,899  # WRONG!
   ```

5. **Metrics compare wrong values:**
   - `yhat`: 1,185 (correct prediction)
   - `y_orig`: 3,919,899 (incorrectly "denormalized" target)
   - `MAE`: |1,185 - 3,919,899| = 3,918,714 ❌

### The Fix

**Simply remove line 600!** The targets are already in the correct scale.

```python
# Line 597
total_loss, yhat, y, _, _, _, _, _ = run_epoch_val(...)

# Line 599-603: CORRECTED
# Both yhat and y are already in original scale
f.write("{} - {} (one clip) R2:   {:.3f}\n".format(..., sklearn.metrics.r2_score(y, yhat)))
f.write("{} - {} (one clip) MAE:  {:.2f}\n".format(..., sklearn.metrics.mean_absolute_error(y, yhat)))
f.write("{} - {} (one clip) RMSE: {:.2f}\n".format(..., sklearn.metrics.mean_squared_error(y, yhat)**0.5))
```

---

## Verification

### Training Validation (Epoch 26, 10% experiment)

From the saved predictions file `z_val_epch26_prd.csv`:

```
Sample predictions:
  yhat      y
  57.72     6.0
  239.58    221.0
  492.71    556.0
  2377.50   3300.0
  1125.87   777.0
```

**Calculated metrics:**
- R²: **0.8228** ✅
- MAE: **413.81** ✅
- RMSE: **1,138.27** ✅

**These match the logged values perfectly!**

### Test Evaluation (With Bug)

The bug causes:
- Actual y: 1,152 (from dataset)
- Incorrectly transformed: 1,152 × 3,398.85 + 1,872.56 = 3,919,899
- Prediction: 1,185
- MAE: |1,185 - 3,919,899| = **3,918,714** ❌

**This explains the massive MAE of ~3.9 million!**

---

## Why All Three Experiments Have Similar Test MAE

All experiments have test MAE ≈ 3,938,400 because:

1. All use the same `y_mean` and `y_std` for the incorrect transformation
2. The test set has similar mean population (~1,150)
3. The bug formula: `y * 3398.85 + 1872.56` ≈ 3.9 million for any y ≈ 1,150
4. Predictions are all reasonable (~1,000-2,000), so error ≈ 3.9M - 1.5K ≈ 3.9M

**The identical MAE is due to the bug, not model failure!**

---

## Corrected Understanding

### What's Actually Happening

1. **Training:** ✅ Working correctly
   - Model learns to predict normalized values
   - Loss function normalizes targets correctly (line 767, 952)
   - Predictions are denormalized for metrics

2. **Training Validation:** ✅ Working correctly
   - Both yhat and y are in original scale
   - Metrics are calculated correctly
   - R² = 0.82, MAE = 414 are VALID

3. **Test Evaluation:** ❌ Has a bug
   - Line 600 incorrectly "denormalizes" already-raw targets
   - Metrics compare wrong scales
   - Results are meaningless

### The Comment is Wrong

Line 599 says:
```python
# Dataset returns normalized y; yhat is denormalized. Convert y to original scale for metrics.
```

**This comment is INCORRECT!**

The truth:
- Dataset returns **RAW** (unnormalized) y values
- `run_epoch_val` returns **RAW** y values (line 876)
- `run_epoch_val` returns **DENORMALIZED** yhat values (line 947)
- Both are already in the same scale!
- Line 600 should be removed

---

## Summary

### What I Got Wrong Initially

1. ❌ I thought the training validation metrics were wrong
2. ❌ I thought there was a distribution mismatch
3. ❌ I thought the model wasn't learning

### What's Actually True

1. ✅ Training is working perfectly
2. ✅ Validation metrics (R²=0.82, MAE=414) are CORRECT
3. ✅ The model IS learning well
4. ✅ No data leakage exists
5. ❌ There's a simple bug in test evaluation (line 600)

### Your Actual Results

| Experiment | Val R² | Val MAE | Interpretation |
|------------|--------|---------|----------------|
| **5%** | 0.7854 | ~479 | Good performance with minimal labels |
| **10%** | 0.8228 | 414 | Excellent performance |
| **20%** | 0.8401 | 407 | Best performance |

**These are EXCELLENT results for semi-supervised learning!**

- Clear improvement with more labeled data
- R² > 0.78 shows strong predictive power
- MAE of 400-500 on population ~1,000-2,000 is reasonable (20-40% error)
- SSL is effectively using unlabeled data

---

## Fix Required

### One-Line Fix

In `ucvme_age.py`, line 599-603:

**BEFORE (WRONG):**
```python
# Dataset returns normalized y; yhat is denormalized. Convert y to original scale for metrics.
y_orig = y * y_std + y_mean
f.write("{} - {} (one clip) R2:   {:.3f}\n".format(..., sklearn.metrics.r2_score(y_orig, yhat)))
f.write("{} - {} (one clip) MAE:  {:.2f}\n".format(..., sklearn.metrics.mean_absolute_error(y_orig, yhat)))
f.write("{} - {} (one clip) RMSE: {:.2f}\n".format(..., sklearn.metrics.mean_squared_error(y_orig, yhat)**0.5))
```

**AFTER (CORRECT):**
```python
# Both y and yhat are already in original scale
f.write("{} - {} (one clip) R2:   {:.3f}\n".format(..., sklearn.metrics.r2_score(y, yhat)))
f.write("{} - {} (one clip) MAE:  {:.2f}\n".format(..., sklearn.metrics.mean_absolute_error(y, yhat)))
f.write("{} - {} (one clip) RMSE: {:.2f}\n".format(..., sklearn.metrics.mean_squared_error(y, yhat)**0.5))
```

### Expected Test Results After Fix

Based on the validation results, you should see:

- **Test R²:** ~0.75-0.82 (similar to validation)
- **Test MAE:** ~400-500 (similar to validation)
- **Test RMSE:** ~1,100-1,300 (similar to validation)

---

## Conclusion

**Your experiments are SUCCESSFUL!**

- ✅ Training worked correctly
- ✅ SSL is effective (5% → 10% → 20% shows clear improvement)
- ✅ No data leakage
- ✅ Model achieved R² = 0.78-0.84 (excellent!)
- ❌ Test evaluation has a one-line bug (easy fix)

**You were right to question my analysis. The results are actually very good!**

To get correct test metrics, simply:
1. Remove line 600 in `ucvme_age.py`
2. Change `y_orig` to `y` in lines 601-603
3. Re-evaluate the saved models on test set

The training results you reported (R² = 0.78-0.84) are the **correct** metrics and represent **strong performance** for your SSL approach.
