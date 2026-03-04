# Validation Metrics Enhancement

## Changes Made

Added validation MAE and RMSE metrics to be logged after each epoch, alongside the existing R² metric.

### Files Modified
- `ucvme_age.py` (lines 494-499, 526-531)

### What Was Added

1. **MAE (Mean Absolute Error)** calculation for validation
2. **RMSE (Root Mean Squared Error)** calculation for validation
3. **Console output** showing R², MAE, and RMSE after each validation epoch
4. **CSV logging** of MAE and RMSE in the log.csv file

### Changes in Detail

#### Before:
```python
r2_value = sklearn.metrics.r2_score(y, yhat)
loss = loss_valit

# Only R2 was logged to CSV
f.write("{},{},{},{},{},{},{},{},{},{},{}".format(epoch, phase, loss, r2_value, ...))
```

#### After:
```python
r2_value = sklearn.metrics.r2_score(y, yhat)
mae_value = sklearn.metrics.mean_absolute_error(y, yhat)
rmse_value = sklearn.metrics.mean_squared_error(y, yhat) ** 0.5
loss = loss_valit

# Console output for easy monitoring
print(f"Epoch {epoch} - {phase}: R2={r2_value:.4f}, MAE={mae_value:.2f}, RMSE={rmse_value:.2f}", flush=True)

# R2, MAE, and RMSE logged to CSV
f.write("{},{},{},{},{},{},{},{},{},{},{},{},{}".format(epoch, phase, loss, r2_value, mae_value, rmse_value, ...))
```

### Output Format

#### Console Output (during training):
```
Epoch 0 - val: R2=0.6543, MAE=234.56, RMSE=456.78
Epoch 1 - val: R2=0.7012, MAE=198.34, RMSE=389.45
...
```

#### CSV Log File Format:
The log.csv file now includes additional columns:
```
epoch,phase,loss,r2,mae,rmse,time,samples,memory_allocated,memory_reserved,batch_size,...
0,val,0.0123,0.6543,234.56,456.78,120.5,11980,...
1,val,0.0098,0.7012,198.34,389.45,118.2,11980,...
```

### Benefits

1. **Real-time Monitoring**: See R², MAE, and RMSE printed to console after each epoch
2. **Better Evaluation**: MAE and RMSE provide interpretable error metrics in original units (population count)
3. **Historical Tracking**: All metrics saved to log.csv for later analysis
4. **Model Comparison**: Easy to compare different model runs using multiple metrics

### Metrics Explanation

- **R² (R-squared)**: Proportion of variance explained (0 to 1, higher is better)
  - < 0: Worse than predicting the mean
  - 0 to 0.5: Poor to moderate fit
  - 0.5 to 0.8: Good fit
  - > 0.8: Excellent fit

- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual values
  - In population units (e.g., 234.56 means average error of ~235 people)
  - Lower is better

- **RMSE (Root Mean Squared Error)**: Square root of average squared errors
  - Penalizes larger errors more than MAE
  - Also in population units
  - Lower is better

### Example Interpretation

If you see:
```
Epoch 10 - val: R2=0.7500, MAE=180.25, RMSE=320.45
```

This means:
- Model explains 75% of variance in population (good!)
- Average prediction error is ~180 people
- Typical prediction error (with larger errors weighted more) is ~320 people
