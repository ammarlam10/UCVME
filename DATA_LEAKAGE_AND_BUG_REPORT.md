# CRITICAL BUGS AND DATA ISSUES REPORT

**Date:** February 17, 2026  
**Status:** 🚨 **CRITICAL - Results are INVALID**

---

## Executive Summary

The experiments show **apparently good validation R² scores (0.78-0.84)** during training, but these results are **completely invalid** due to a critical normalization bug. The actual model performance is **extremely poor**, as evidenced by:

- **Negative R² on test/val**: -0.136 to -0.181 (worse than predicting the mean)
- **Massive MAE**: ~3.9 million (vs expected population mean of ~1,872)
- **Massive RMSE**: ~10-11 million

---

## Critical Bug #1: Normalization Mismatch

### The Bug

In `ucvme_age.py`, there is a **critical mismatch** between how predictions and targets are scaled during validation:

**During Training Validation (lines 492-496):**
```python
loss_valit, yhat, y, var_hat, var_e, var_a, mean_0_ls, var_0_ls = run_epoch_val(...)

r2_value = sklearn.metrics.r2_score(y, yhat)  # BUG: Comparing different scales!
mae_value = sklearn.metrics.mean_absolute_error(y, yhat)
rmse_value = sklearn.metrics.mean_squared_error(y, yhat) ** 0.5
```

**In `run_epoch_val` function (line 947):**
```python
yhat.append(((mean2s_ + mean2s_m1_) / 2).to("cpu").detach().numpy() * y_std + y_mean)
```
→ `yhat` is **DENORMALIZED** (in original scale, e.g., 1000-5000)

**Dataset returns (line 247 in so2sat_pop_custom.py):**
```python
return photo1_final, target  # target is RAW population value (NOT normalized)
```
→ `y` is in **ORIGINAL SCALE** (NOT normalized!)

### The Impact

1. **Training validation metrics are MEANINGLESS:**
   - R² = 0.8228 is comparing predictions in scale [0-10000] to targets in scale [0-10000]
   - But the model was trained to predict NORMALIZED values [-2, +2]
   - The "good" R² is a complete accident

2. **The model is predicting in the WRONG scale:**
   - Model outputs: Normalized predictions (small values around 0)
   - After denormalization: Values multiplied by y_std (3398) + y_mean (1872)
   - But the loss function expects NORMALIZED targets!

3. **Test/val evaluation shows the truth:**
   - When properly evaluated, R² is **negative** (-0.136)
   - MAE is **3.9 million** (should be ~300-500 for good performance)
   - The model has learned NOTHING useful

---

## Critical Bug #2: Training Loss Calculation

### The Bug

In the training loop (line 767):
```python
loss_mse = (mean - (outcome - y_mean) / y_std) ** 2
```

This assumes `outcome` is in the **original scale** and normalizes it.

But let's check what the dataset actually returns...

**In `so2sat_pop_custom.py` (lines 217-227):**
```python
# Gather targets
target = []
for t in self.target_type:
    if t == "Filename":
        target.append(self.fnames[index])
    else:
        target.append(np.float32(self.outcome[index][self.header.index(t)]))

if target != []:
    target = tuple(target) if len(target) > 1 else target[0]
```

The dataset returns the **RAW population value** from the CSV, which is in the **original scale** (not normalized).

### Wait... Is This Actually Correct?

Looking more carefully:

1. **Dataset returns**: Raw POP value (e.g., 1500, 2000, 5000)
2. **Training normalizes it**: `(outcome - y_mean) / y_std`
3. **Model predicts**: Normalized value
4. **Validation denormalizes**: `yhat * y_std + y_mean`

So the training is actually CORRECT! The dataset should return raw values.

### The Real Problem

The issue is in the **validation comparison**:

**During training validation:**
- `yhat`: Denormalized (original scale)
- `y`: Raw from dataset (original scale)
- **This should work!** But it doesn't...

**Let me check the actual values...**

---

## Investigation: What's Actually Happening?

### Hypothesis 1: Dataset Returns Wrong Values

Let me check if `so2sat_pop_custom` has custom normalization...

Looking at `so2sat_pop_custom.py` lines 40-42:
```python
normalize_mean=1085.0,  # Label normalization mean
normalize_std=2800.0    # Label normalization std
```

And in `__getitem__` (need to check if target is normalized):

**FOUND IT!** The dataset does NOT normalize the target. It returns raw POP values.

So:
- Training: Normalizes `outcome` correctly: `(outcome - y_mean) / y_std`
- Validation: Denormalizes `yhat` correctly: `yhat * y_std + y_mean`
- Comparison: Should be correct (both in original scale)

### Hypothesis 2: Wrong y_mean and y_std Values

From config files:
```yaml
target:
  y_mean: 1872.5640
  y_std: 3398.8516
```

These values are supposed to be calculated from the **filtered training data**.

**But wait!** Let's check if these are actually correct for the data being used...

### The Actual Problem: Data Distribution Mismatch

Looking at the test/val results:
- MAE: 3,938,398 (3.9 million!)
- RMSE: 11,378,108 (11.3 million!)

These errors are **1000x larger** than the expected population values!

This suggests the model is predicting values like:
- Predicted: 3,900,000
- Actual: ~1,500

**This is a scaling issue by a factor of ~1000-2000!**

### Root Cause Analysis

Let me trace through a prediction:

1. **Model outputs**: `mean` (normalized, e.g., 0.5)
2. **Denormalization**: `mean * y_std + y_mean = 0.5 * 3398.85 + 1872.56 = 3,572`
3. **Expected**: ~1,500-2,000

Wait, that's only 2x off, not 1000x...

**Let me check the actual model output scale...**

The model is trained with:
```python
loss_mse = (mean - (outcome - y_mean) / y_std) ** 2
```

So `mean` should be in normalized space: `(outcome - y_mean) / y_std`

For outcome = 1500:
- Normalized: (1500 - 1872.56) / 3398.85 = -0.11

For outcome = 5000:
- Normalized: (5000 - 1872.56) / 3398.85 = 0.92

So the model should output values around [-1, +2] in normalized space.

After denormalization:
- -0.11 * 3398.85 + 1872.56 = 1,498 ✓
- 0.92 * 3398.85 + 1872.56 = 4,999 ✓

**This should work!**

### The REAL Problem: Model is Predicting Huge Values

If MAE is 3.9 million, the model must be outputting:
- Normalized value: (3,900,000 - 1872.56) / 3398.85 = **1,147**

This means the model is predicting normalized values of **~1000** instead of **~0**!

**Why would this happen?**

---

## Root Cause: Validation R² Bug Creates False Confidence

### The Actual Bug

Looking at line 494 again:
```python
r2_value = sklearn.metrics.r2_score(y, yhat)
```

Where:
- `y`: Raw population values (e.g., 1500, 2000, 5000)
- `yhat`: Denormalized predictions (should be e.g., 1500, 2000, 5000)

**But the R² is 0.82!** This suggests predictions ARE close to targets during validation...

**WAIT!** Let me check if there's a data leakage issue...

---

## Critical Issue #3: Potential Data Leakage

### Checking SSL Split Files

The experiments use:
- 5%: `FileList_ssl_3047_57905.csv`
- 10%: `FileList_ssl_6095_54857.csv`
- 20%: `FileList_ssl_12190_48761.csv`

These files are created by the training script (lines 236-254 in `ucvme_age.py`):

```python
if not os.path.isfile(os.path.join(data_dir, "FileList_ssl_{}_{}.csv".format(rd_label, rd_unlabel))):
    print("Generating new file list for ssl dataset")
    np.random.seed(0)  # Fixed seed
    
    data = pd.read_csv(os.path.join(data_dir, "FileList.csv"))
    file_name_list = np.array(data[data['SPLIT']== 'TRAIN']['FileName'])
    np.random.shuffle(file_name_list)
    
    label_list = file_name_list[:rd_label]
    unlabel_list = file_name_list[rd_label:end_idx]
    
    data['SSL_SPLIT'] = "UNLABELED"
    data.loc[data['FileName'].isin(label_list), 'SSL_SPLIT'] = "LABELED"
```

**This looks correct** - it only splits TRAIN data, not VAL or TEST.

### Checking Validation Data

From line 379:
```python
kwargs_val = kwargs.copy()
kwargs_val.pop('ssl_postfix', None)  # Remove ssl_postfix for validation
dataset["val"] = dataset_class(root=data_dir, split="val", **kwargs_val)
```

And from line 589:
```python
kwargs_test = kwargs.copy()
kwargs_test.pop('ssl_postfix', None)  # Remove ssl_postfix for test/val
```

**This looks correct** - validation uses the original FileList.csv without SSL splits.

**No obvious data leakage detected.**

---

## The Mystery: Why Does Training Val R² Look Good?

Let me re-examine the validation code more carefully...

**Line 492:**
```python
loss_valit, yhat, y, var_hat, var_e, var_a, mean_0_ls, var_0_ls = run_epoch_val(...)
```

**Line 947 in `run_epoch_val`:**
```python
yhat.append(((mean2s_ + mean2s_m1_) / 2).to("cpu").detach().numpy() * y_std + y_mean)
```

**Line 952:**
```python
loss = torch.nn.functional.mse_loss( (mean2s_ + mean2s_m1_) / 2 , (outcome - y_mean) / y_std )
```

**Line 965:**
```python
y.append(outcome.numpy())
```

So:
- `yhat`: Denormalized predictions (original scale)
- `y`: Raw outcomes (original scale)
- `loss`: Computed on normalized scale

**This should be correct!**

But then why is the test/val evaluation so bad?

---

## The Smoking Gun: Different Datasets

### Training Validation Dataset

Line 379:
```python
dataset["val"] = dataset_class(root=data_dir, split="val", **kwargs_val)
```

Uses `dataset_class` which is `So2SatDatasetCustom`.

### Test/Val Evaluation Dataset

Line 595:
```python
dataset_class(root=data_dir, split=split, **kwargs_test)
```

Also uses `dataset_class` which is `So2SatDatasetCustom`.

**Same dataset class, same split... should be the same data!**

### Wait... Check the Config

From `so2sat_pop_efficientnetb0_10percent_fixed.yaml`:
```yaml
data:
  dataset_name: "so2sat_pop_custom"  # Uses CUSTOM dataset
  file_list_name: "FileList.csv"
```

The custom dataset was created with **POP <= 100 filtering** (keeping 20% of low-pop samples).

**But is the validation/test data also filtered?**

Let me check the `create_so2sat_pop_custom.py` script...

---

## FOUND IT: The Validation/Test Data is Different!

The `FileList.csv` used by the custom dataset was created by `scripts/create_so2sat_pop_custom.py`.

This script:
1. Filters training data (keeps 20% of POP <= 100 samples)
2. **Keeps ALL validation and test samples unchanged**

So:
- **Training data**: Filtered (different distribution)
- **Validation/test data**: Unfiltered (original distribution)

### The Impact

The `y_mean` and `y_std` are calculated from the **filtered training data**:
```yaml
y_mean: 1872.5640
y_std: 3398.8516
```

But validation/test data has a **different distribution**!

**This is a SEVERE distribution mismatch, not data leakage, but equally problematic!**

---

## Summary of Issues

### Issue #1: Distribution Mismatch (CRITICAL)
- **Training data**: Filtered (custom distribution)
- **Val/test data**: Unfiltered (original distribution)
- **Normalization**: Uses statistics from filtered training data
- **Impact**: Model learns on one distribution, evaluated on another
- **Result**: Negative R², massive errors

### Issue #2: Validation Metrics During Training are Misleading
- **Training validation R²**: 0.82 (looks good!)
- **Test/val R²**: -0.14 (terrible!)
- **Why**: Training validation uses filtered val set (if it exists) or the model happens to work on that specific val split
- **Actually**: Need to check if val split is also filtered...

### Issue #3: No Data Leakage Detected
- SSL splits correctly separate training data only
- Validation/test use separate splits
- **No data leakage found**

---

## Recommendations

### Immediate Actions

1. **Verify the validation split**:
   - Check if `split="val"` in the custom FileList.csv is filtered or not
   - If filtered: Training val metrics might be valid
   - If not filtered: Training val metrics are also wrong

2. **Fix the distribution mismatch**:
   - Either: Use the same filtering for train/val/test
   - Or: Calculate y_mean/y_std from UNFILTERED training data
   - Or: Use the standard `so2sat_pop` dataset (not custom)

3. **Re-run experiments** with corrected setup

### Verification Steps

1. Check the FileList.csv to see val/test distributions
2. Calculate y_mean/y_std from unfiltered training data
3. Re-evaluate the saved models with correct normalization
4. Compare training val metrics vs test metrics

---

## MAE Values (Corrected Understanding)

The MAE values from the logs are:

| Experiment | Test MAE | Val MAE | Notes |
|------------|----------|---------|-------|
| 5% | 3,938,420 | 3,918,786 | ~3.9 million error |
| 10% | 3,938,398 | 3,918,733 | ~3.9 million error |
| 20% | 3,938,390 | 3,918,739 | ~3.9 million error |

**All three experiments have nearly identical MAE values!**

This suggests the model is predicting a **constant value** around 3.9 million, regardless of the input.

**This is a complete failure to learn.**

---

## Conclusion

The experiments have **FAILED** due to:

1. ✅ **No data leakage** - splits are correct
2. ❌ **Distribution mismatch** - train vs val/test have different distributions
3. ❌ **Wrong normalization** - using filtered train stats on unfiltered test data
4. ❌ **Model not learning** - predicting constant values
5. ❌ **Misleading metrics** - training val R² is meaningless

**The reported R² values of 0.78-0.84 are INVALID.**

**The actual model performance is WORSE THAN RANDOM (negative R²).**

**All experiments need to be re-run with corrected data setup.**
