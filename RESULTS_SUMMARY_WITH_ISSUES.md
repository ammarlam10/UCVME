# Experiment Results Summary - WITH CRITICAL ISSUES

**Date:** February 17, 2026  
**Status:** 🚨 **RESULTS ARE INVALID - DO NOT USE**

---

## Quick Summary

| Experiment | Training Val R² | Training Val MAE | Test R² | Test MAE | Status |
|------------|-----------------|------------------|---------|----------|--------|
| **5% Labeled** | 0.7854 | Not logged | **-0.136** | **3,938,420** | ❌ FAILED |
| **10% Labeled** | 0.8228 | Not logged | **-0.136** | **3,938,398** | ❌ FAILED |
| **20% Labeled** | 0.8401 | Not logged | **-0.136** | **3,938,390** | ❌ FAILED |

**Expected MAE for good performance:** ~300-500  
**Actual MAE:** ~3.9 million (10,000x worse!)

---

## Critical Findings

### ✅ No Data Leakage Detected
- SSL splits correctly separate labeled/unlabeled within training set only
- Validation and test sets are properly isolated
- No overlap between train/val/test splits

### ❌ Model Has NOT Learned Anything
- **Test R² is negative** (-0.136): Worse than predicting the mean
- **MAE is ~3.9 million**: Expected population values are ~1,000-5,000
- **All three experiments have identical MAE**: Model predicts constant value
- **Training validation R² is misleading**: Shows 0.78-0.84 but test shows -0.14

### ❌ Severe Distribution Mismatch
The custom dataset (`so2sat_pop_custom`) has:
- **Training data**: Filtered (keeps only 20% of low-population samples)
- **Validation/test data**: Likely unfiltered (original distribution)
- **Normalization parameters**: Calculated from filtered training data
  - `y_mean: 1872.5640`
  - `y_std: 3398.8516`
- **Impact**: Model trained on one distribution, evaluated on another

### ❌ Validation Metrics During Training Are Misleading
- Training validation shows R² = 0.78-0.84 (looks good!)
- Test evaluation shows R² = -0.14 (terrible!)
- **Why the discrepancy?**
  - If validation set is also filtered: Metrics might be valid for that distribution
  - If validation set is unfiltered: Metrics are also wrong (need investigation)
  - The model may be overfitting to the filtered distribution

---

## Detailed Results

### Experiment 1: 5% Labeled

**Training Validation (Epoch 29):**
- Val Loss: 0.1359
- Val R²: **0.7854**
- Val MAE: Not logged during training

**Test Evaluation (After Training):**
- Test R²: **-0.136** ❌
- Test MAE: **3,938,420** ❌
- Test RMSE: **11,378,254** ❌

**Validation Re-evaluation:**
- Val R²: **-0.181** ❌
- Val MAE: **3,918,786** ❌
- Val RMSE: **9,990,530** ❌

### Experiment 2: 10% Labeled

**Training Validation (Epoch 26 - Best):**
- Val Loss: 0.1122
- Val R²: **0.8228**
- Val MAE: Not logged during training

**Test Evaluation (After Training):**
- Test R²: **-0.136** ❌
- Test MAE: **3,938,398** ❌
- Test RMSE: **11,378,109** ❌

**Validation Re-evaluation:**
- Val R²: **-0.181** ❌
- Val MAE: **3,918,733** ❌
- Val RMSE: **9,990,301** ❌

### Experiment 3: 20% Labeled

**Training Validation (Epoch 5 - Best):**
- Val Loss: 0.1012
- Val R²: **0.8401**
- Val MAE: Not logged during training

**Test Evaluation (After Training):**
- Test R²: **-0.136** ❌
- Test MAE: **3,938,390** ❌
- Test RMSE: **11,378,104** ❌

**Validation Re-evaluation:**
- Val R²: **-0.181** ❌
- Val MAE: **3,918,739** ❌
- Val RMSE: **9,990,308** ❌

---

## Key Observations

### 1. Identical Test Performance Across All Experiments
All three experiments have nearly **identical test MAE** (~3,938,400):
- This suggests the model is predicting a **constant value**
- The amount of labeled data (5%, 10%, 20%) makes **no difference**
- The model has **not learned** from the data

### 2. Huge Discrepancy Between Training and Test Metrics
- Training Val R²: 0.78-0.84 (good)
- Test R²: -0.14 (terrible)
- **1,000x difference** in scale

This indicates:
- Training validation is on a **different distribution**
- Or training validation has a **bug** in metric calculation
- The "good" training metrics are **misleading**

### 3. MAE Values Are Absurdly Large
- Expected population range: 0-10,000 (with mean ~1,872)
- Expected MAE for good model: 300-500
- Actual MAE: **3.9 million**

The model is predicting values like:
- Predicted: ~3,900,000
- Actual: ~1,500
- Error: **2,600x the actual value!**

---

## Root Cause Analysis

### Primary Issue: Distribution Mismatch

The `so2sat_pop_custom` dataset was created with **filtering**:
- **Training**: Keeps only 20% of samples with POP <= 100
- **Val/Test**: Likely uses original unfiltered data

**Normalization parameters** (`y_mean=1872.56`, `y_std=3398.85`) are calculated from the **filtered training data**.

When the model is evaluated on **unfiltered test data**:
1. Test data has different population distribution
2. Normalization uses wrong mean/std
3. Predictions are completely off-scale
4. Result: Negative R², massive errors

### Secondary Issue: Model Architecture or Training Problem

Even accounting for distribution mismatch:
- All three experiments produce **identical predictions**
- This suggests the model is **not learning at all**
- Possible causes:
  - Learning rate too high/low
  - Loss function issue
  - Gradient flow problem
  - Data preprocessing error

---

## What Went Wrong: Step-by-Step

1. **Custom dataset created** with filtered training data
2. **Normalization calculated** from filtered data: `y_mean=1872.56`, `y_std=3398.85`
3. **Model trained** on filtered data with these normalization parameters
4. **Training validation** shows good R² (0.78-0.84)
   - If val set is also filtered: Metrics are valid for that distribution
   - If val set is unfiltered: Metrics are also wrong (bug)
5. **Test evaluation** uses unfiltered data
   - Different distribution than training
   - Wrong normalization parameters
   - Result: Negative R², massive errors

---

## Recommendations

### Immediate Actions

1. **Verify the validation split:**
   ```bash
   # Check if val split in FileList.csv is filtered
   python3 -c "
   import pandas as pd
   df = pd.read_csv('/work/ammar/sslrp/data/So2Sat_POP/FileList.csv')
   val_data = df[df['SPLIT'] == 'VAL']
   print(f'Val samples: {len(val_data)}')
   print(f'Val POP mean: {val_data[\"POP\"].mean():.2f}')
   print(f'Val POP std: {val_data[\"POP\"].std():.2f}')
   print(f'Val POP min: {val_data[\"POP\"].min():.2f}')
   print(f'Val POP max: {val_data[\"POP\"].max():.2f}')
   "
   ```

2. **Check training data distribution:**
   ```bash
   python3 -c "
   import pandas as pd
   df = pd.read_csv('/work/ammar/sslrp/data/So2Sat_POP/FileList.csv')
   train_data = df[df['SPLIT'] == 'TRAIN']
   print(f'Train samples: {len(train_data)}')
   print(f'Train POP mean: {train_data[\"POP\"].mean():.2f}')
   print(f'Train POP std: {train_data[\"POP\"].std():.2f}')
   "
   ```

3. **Compare with config values:**
   - Config `y_mean`: 1872.5640
   - Config `y_std`: 3398.8516
   - Do these match the training data?

### Fix Options

#### Option 1: Use Consistent Filtering (Recommended)
- Filter val/test data the same way as training data
- Recalculate y_mean/y_std from filtered data
- Re-run experiments

#### Option 2: Use Unfiltered Data
- Use the standard `so2sat_pop` dataset (not custom)
- Calculate y_mean/y_std from full unfiltered training data
- Re-run experiments

#### Option 3: Use Original Distribution Normalization
- Keep filtered training data
- Calculate y_mean/y_std from **unfiltered** training data
- This allows the model to generalize to the full distribution
- Re-run experiments

### Verification Steps

1. **Check the FileList.csv:**
   - Verify train/val/test splits
   - Check population distributions
   - Confirm filtering was applied correctly

2. **Recalculate normalization:**
   - Calculate y_mean/y_std from appropriate data
   - Update config files

3. **Re-run ONE experiment first:**
   - Use 10% labeled as test case
   - Monitor training and test metrics
   - Verify test R² is positive and MAE is reasonable

4. **If successful, run all experiments:**
   - 5%, 10%, 20% labeled
   - Compare results

---

## Expected Results After Fix

If the fix is successful, you should see:

- **Test R²**: 0.6-0.8 (positive!)
- **Test MAE**: 300-800 (reasonable)
- **Test RMSE**: 500-1,200 (reasonable)
- **Training vs Test metrics**: Similar values (no huge discrepancy)

---

## Lessons Learned

1. **Always verify train/val/test distributions match**
2. **Calculate normalization from the same distribution used for evaluation**
3. **Monitor both training and test metrics during development**
4. **Sanity check: If training looks good but test is terrible, investigate immediately**
5. **Log MAE/RMSE during training validation, not just R²**

---

## Next Steps

1. ✅ Investigate validation split distribution
2. ✅ Recalculate y_mean/y_std
3. ✅ Update config files
4. ✅ Re-run experiments
5. ✅ Verify results are reasonable
6. ✅ Document the fix

**DO NOT use the current results for any analysis or publication.**
