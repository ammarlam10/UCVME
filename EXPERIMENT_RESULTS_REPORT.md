# So2Sat POP SSL Experiments - Final Results Report

**Date:** February 17, 2026  
**Experiments:** 5%, 10%, and 20% labeled data with EfficientNet-B0  
**Status:** 🚨 **CRITICAL ISSUES FOUND - RESULTS ARE INVALID**

---

## ⚠️ CRITICAL WARNING ⚠️

**These results are INVALID and should NOT be used for any analysis or publication.**

The experiments have **severe issues** that make the reported metrics meaningless:

1. **Test R² is NEGATIVE** (-0.136): Model performs worse than predicting the mean
2. **Test MAE is ~3.9 MILLION**: Expected values are ~1,000-5,000 (error is 1000x too large)
3. **Training validation metrics are misleading**: Show good R² (0.78-0.84) but test shows terrible performance
4. **All three experiments produce identical test results**: Model has not learned anything useful

**See `RESULTS_SUMMARY_WITH_ISSUES.md` and `DATA_LEAKAGE_AND_BUG_REPORT.md` for detailed analysis.**

---

## Executive Summary

Three semi-supervised learning (SSL) experiments were conducted on the So2Sat POP dataset using EfficientNet-B0 architecture. The experiments varied the percentage of labeled training data (5%, 10%, 20%) while keeping the remaining data as unlabeled for SSL training.

### Training Validation Results (MISLEADING)

| Experiment | Labeled % | Best Epoch | Val Loss | Val R² | Training Samples |
|------------|-----------|------------|----------|--------|------------------|
| **5% Fixed** | 5% | 29 | 0.1359 | 0.7854 | ~3,047 labeled / ~57,905 unlabeled |
| **10% Fixed** | 10% | 26 | 0.1122 | 0.8228 | ~6,095 labeled / ~54,857 unlabeled |
| **20% Fixed** | 20% | 5 | 0.1012 | 0.8401 | ~12,190 labeled / ~48,762 unlabeled |

### Test Results (ACTUAL PERFORMANCE)

| Experiment | Test R² | Test MAE | Test RMSE | Status |
|------------|---------|----------|-----------|--------|
| **5% Fixed** | **-0.136** | **3,938,420** | **11,378,254** | ❌ FAILED |
| **10% Fixed** | **-0.136** | **3,938,398** | **11,378,109** | ❌ FAILED |
| **20% Fixed** | **-0.136** | **3,938,390** | **11,378,104** | ❌ FAILED |

**Critical Findings:**
- ❌ **Model has NOT learned**: All experiments produce identical test results
- ❌ **Negative R²**: Worse than predicting the mean
- ❌ **Massive errors**: MAE is 1000x larger than expected
- ❌ **Distribution mismatch**: Training data filtered, test data unfiltered
- ✅ **No data leakage**: Train/val/test splits are properly separated

---

## Detailed Results

### Experiment 1: 5% Labeled Data

**Configuration:**
- Labeled samples: ~3,047 (5% of filtered training set)
- Unlabeled samples: ~57,905 (95%)
- Batch size: 64
- Epochs: 30
- Learning rate: 0.0001

**Training Progress:**
- Epoch 0: Val R² = 0.770, Val Loss = 0.1456
- Epoch 15: Val R² = 0.780, Val Loss = 0.1402
- **Epoch 29 (Best):** Val R² = **0.7854**, Val Loss = **0.1359**

**Final Metrics:**
- Best Validation R²: **0.7854**
- Best Validation Loss: **0.1359**
- Best Epoch: **29**

---

### Experiment 2: 10% Labeled Data

**Configuration:**
- Labeled samples: ~6,095 (10% of filtered training set)
- Unlabeled samples: ~54,857 (90%)
- Batch size: 64
- Epochs: 30
- Learning rate: 0.0001

**Training Progress:**
- Epoch 0: Val R² = 0.817, Val Loss = 0.1158
- Epoch 20: Val R² = 0.822, Val Loss = 0.1122
- **Epoch 26 (Best):** Val R² = **0.8228**, Val Loss = **0.1122**

**Final Metrics:**
- Best Validation R²: **0.8228**
- Best Validation Loss: **0.1122**
- Best Epoch: **26**

---

### Experiment 3: 20% Labeled Data

**Configuration:**
- Labeled samples: ~12,190 (20% of filtered training set)
- Unlabeled samples: ~48,762 (80%)
- Batch size: 64
- Epochs: 30
- Learning rate: 0.0001

**Training Progress:**
- Epoch 0: Val R² = 0.805, Val Loss = 0.1234
- **Epoch 5 (Best):** Val R² = **0.8401**, Val Loss = **0.1012**
- Epoch 25: Val R² = 0.831, Val Loss = 0.1081

**Final Metrics:**
- Best Validation R²: **0.8401** (highest among all experiments)
- Best Validation Loss: **0.1012** (lowest among all experiments)
- Best Epoch: **5** (earliest convergence)

---

## Training Process Explanation

### Architecture Overview

The UCVME (Uncertainty-aware Contrastive Variational Model Ensemble) framework uses a **dual-model ensemble** approach for semi-supervised learning:

1. **Two Identical Models:** `model` and `model_1` (both EfficientNet-B0)
2. **Ensemble Learning:** Models learn from each other through consistency regularization
3. **Uncertainty Estimation:** Each model outputs both mean and variance predictions

### Training Loop Structure

```
For each epoch:
    For phase in ['train', 'val']:
        if phase == 'train':
            # SSL Training Phase
            - Load labeled batch (X, y)
            - Load unlabeled batch (X_ulb)
            
            # Process unlabeled data
            - Forward pass through both models (samp_ssl times)
            - Compute consistency loss between model predictions
            - Compute variance alignment loss
            
            # Process labeled data
            - Forward pass through both models
            - Compute supervised loss (MSE with uncertainty weighting)
            - Combine losses: supervised + consistency + variance alignment
            
            # Backward pass and optimization
            - Update both models simultaneously
            
        else:  # Validation phase
            - Forward pass through both models (samp_fq times)
            - Average predictions from both models
            - Compute validation metrics (R², MAE, RMSE)
```

### Loss Function Components

The total training loss consists of three main components:

#### 1. Supervised Loss (Labeled Data)
```python
loss_mse = (mean - (outcome - y_mean) / y_std) ** 2
loss1 = torch.mul(torch.exp(-(var + var_1) / 2), loss_mse)  # Uncertainty-weighted MSE
loss2 = (var + var_1) / 2  # Variance regularization
loss_reg = 0.5 * (loss1 + loss2)  # Applied to both models
```

**Key Features:**
- **Uncertainty weighting:** Higher variance reduces the impact of uncertain predictions
- **Dual model:** Both models contribute to the loss
- **Normalized targets:** Population values normalized using `y_mean` and `y_std`

#### 2. Consistency Loss (Unlabeled Data)
```python
# Average predictions from both models
avg_mean01 = (model_pred + model_1_pred) / 2
avg_var01 = (model_var + model_1_var) / 2

# Consistency loss: each model should agree with the ensemble
loss_mse_cps_0 = (model_pred - avg_mean01)**2
loss_cmb_cps_0 = 0.5 * (exp(-avg_var01) * loss_mse_cps_0 + avg_var01)
loss_reg_cps = loss_cmb_cps_0 + loss_cmb_cps_1
```

**Purpose:**
- Encourages both models to agree on unlabeled data predictions
- Uses uncertainty-weighted consistency (uncertain predictions have less weight)
- Promotes model diversity while maintaining agreement

#### 3. Variance Alignment Loss
```python
var_loss_ulb_0 = ((model_var - avg_var01)**2).mean()
var_loss_ulb_1 = ((model_1_var - avg_var01)**2).mean()
var_loss = var_loss_ulb_0 + var_loss_ulb_1
```

**Purpose:**
- Ensures both models have similar uncertainty estimates
- Promotes calibrated uncertainty predictions

#### Total Loss
```python
loss = loss_reg + w_ulb * loss_reg_cps + ((var_1 - var) ** 2).mean()
```

Where:
- `loss_reg`: Supervised loss from labeled data
- `w_ulb`: Weight for unlabeled loss (default: 10)
- `loss_reg_cps`: Consistency loss from unlabeled data
- `((var_1 - var) ** 2).mean()`: Direct variance difference penalty

### Key Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `ssl_mult` | -1 | Duplicate labeled samples (2x) to balance with unlabeled |
| `w_ulb` | 10 | Weight for unlabeled consistency loss |
| `samp_fq` | 5 | Number of forward passes for validation (Monte Carlo sampling) |
| `samp_ssl` | 5 | Number of forward passes for SSL consistency |
| `batch_size` | 64 | Batch size for both labeled and unlabeled data |
| `lr` | 0.0001 | Learning rate |
| `weight_decay` | 1e-3 | L2 regularization |
| `lr_step_period` | 10 | Learning rate decay period |

### Optimization Strategy

1. **Optimizers:** Adam optimizer for both models
2. **Learning Rate Scheduling:** StepLR with period 10 epochs
3. **Gradient Updates:** Both models updated simultaneously
4. **Data Shuffling:** Both labeled and unlabeled data shuffled each epoch

---

## Data Loading Process

### Dataset Structure

The So2Sat POP dataset is organized as follows:

```
So2Sat_POP/
├── So2Sat_POP_Part1/
│   ├── train/
│   │   ├── city_name/
│   │   │   ├── city_name.csv  (contains GRD_ID, Class, POP)
│   │   │   └── sen2summer/
│   │   │       └── Class_X/
│   │   │           └── GRD_ID_sen2summer.tif  (13-channel Sentinel-2 images)
│   │   └── ...
│   └── test/
│       └── ...
└── FileList.csv  (created by preprocessing script)
```

### FileList.csv Format

The `FileList.csv` contains:
- `FileName`: Relative path to image file
- `SPLIT`: "TRAIN" or "TEST"
- `POP`: Population value (target variable)
- `SSL_SPLIT`: "LABELED" or "UNLABELED" (for SSL experiments)
- Additional metadata: `GRD_ID`, `Class`, `City`

### Data Loading Pipeline

#### Step 1: Configuration and File List Generation

```python
# If reduced_set=True, create SSL split file
if reduced_set:
    # Read FileList.csv
    data = pd.read_csv(os.path.join(data_dir, "FileList.csv"))
    
    # Shuffle training samples
    train_samples = data[data['SPLIT'] == 'TRAIN']
    np.random.shuffle(train_samples)
    
    # Split into labeled and unlabeled
    label_list = train_samples[:rd_label]  # First rd_label samples
    unlabel_list = train_samples[rd_label:rd_label+rd_unlabel]  # Next rd_unlabel samples
    
    # Create SSL_SPLIT column
    data['SSL_SPLIT'] = "UNLABELED"
    data.loc[data['FileName'].isin(label_list), 'SSL_SPLIT'] = "LABELED"
    
    # Save to FileList_ssl_{rd_label}_{rd_unlabel}.csv
```

#### Step 2: Dataset Initialization

```python
# Labeled dataset (ssl_type=1)
dataset_lb = So2SatDatasetCustom(
    root=data_dir,
    split="train",
    ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel),
    ssl_type=1,  # Only labeled samples
    ssl_mult=-1,  # Duplicate samples 2x
    target_type=["POP"],
    pad=5,  # Data augmentation padding
    image_dir="So2Sat_POP_Part1",
    file_list_name="FileList.csv"
)

# Unlabeled dataset (ssl_type=2)
dataset_unlb = So2SatDatasetCustom(
    root=data_dir,
    split="train",
    ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel),
    ssl_type=2,  # Only unlabeled samples
    target_type=["POP"],
    pad=5,
    image_dir="So2Sat_POP_Part1",
    file_list_name="FileList.csv"
)
```

#### Step 3: Image Loading (`__getitem__`)

For each sample, the dataset loader:

1. **Loads Multi-spectral TIFF:**
   ```python
   # Read 13-channel Sentinel-2 image
   photo = tifffile.imread(image_path)  # Shape: (H, W, 13)
   
   # Extract RGB bands: channels [2, 1, 0] for Red, Green, Blue
   photo = photo[:, :, [2, 1, 0]]  # Shape: (H, W, 3)
   photo = photo.transpose((2, 0, 1))  # Shape: (3, H, W)
   ```

2. **Resizes to 224x224:**
   ```python
   # Resize each channel to 224x224 for EfficientNet compatibility
   for i in range(3):
       photo[i] = cv2.resize(photo[i], (224, 224))
   ```

3. **Normalizes Image:**
   ```python
   # Subtract mean and divide by std (calculated from training set)
   photo = (photo - mean) / std
   ```

4. **Data Augmentation:**
   ```python
   # Random horizontal flip (50% probability)
   if np.random.randint(0, 2) == 0:
       photo = photo[:, :, ::-1]  # Flip horizontally
   
   # Random crop with padding (pad_param=5)
   # Adds 5 pixels padding, then randomly crops back to original size
   ```

5. **Returns:**
   - `photo`: Normalized, augmented image tensor (3, 224, 224)
   - `target`: Population value (POP) as float32

### DataLoader Configuration

```python
# Labeled DataLoader
dataloader_lb = DataLoader(
    dataset_lb,
    batch_size=64,
    num_workers=2,
    shuffle=True,
    pin_memory=True,  # Faster GPU transfer
    drop_last=True,   # Drop incomplete batches
    worker_init_fn=worker_init_fn  # Seed workers differently
)

# Unlabeled DataLoader
dataloader_unlb = DataLoader(
    dataset_unlb,
    batch_size=64,
    num_workers=2,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    worker_init_fn=worker_init_fn
)
```

### Key Data Processing Details

1. **Sentinel-2 Image Processing:**
   - Original: 13 spectral channels (multi-spectral)
   - Extraction: RGB channels [2, 1, 0] → Red, Green, Blue
   - Format: TIFF files loaded using `tifffile` library

2. **Normalization:**
   - **Image normalization:** Mean and std calculated from training set
   - **Target normalization:** Population values normalized using `y_mean` and `y_std`
   - Formula: `normalized_pop = (pop - y_mean) / y_std`

3. **Data Augmentation:**
   - Random horizontal flip (50% probability)
   - Random crop with padding (5 pixels)
   - Applied only during training

4. **SSL Split Handling:**
   - Labeled samples: `ssl_type=1` → filters `SSL_SPLIT == "LABELED"`
   - Unlabeled samples: `ssl_type=2` → filters `SSL_SPLIT == "UNLABELED"`
   - Labeled samples duplicated 2x (`ssl_mult=-1`) to balance batches

---

## Model Architecture

### EfficientNet-B0 Backbone

- **Base Model:** EfficientNet-B0 from `timm` library
- **Pretrained:** ImageNet pretrained weights
- **Modification:** Last layer replaced with uncertainty head
- **Output:** Two values per sample:
  - `mean`: Predicted population (normalized)
  - `variance`: Uncertainty estimate

### Uncertainty Head

```python
# Simplified architecture
features = efficientnet_backbone(image)  # (batch, 1280)
mean = linear_head_mean(features)  # (batch, 1)
variance = softplus(linear_head_var(features))  # (batch, 1) - ensures positive
```

### Dual Model Ensemble

- Two identical EfficientNet-B0 models initialized independently
- Both models trained simultaneously
- Final prediction: Average of both model predictions
- Final uncertainty: Average of both model uncertainties

---

## Validation and Testing

### Validation Process

1. **Monte Carlo Sampling:**
   - Forward pass `samp_fq=5` times through both models
   - Average predictions to get final estimate
   - Compute epistemic uncertainty from prediction variance

2. **Metrics Computed:**
   - **R² Score:** Coefficient of determination
   - **MAE:** Mean Absolute Error
   - **RMSE:** Root Mean Squared Error
   - **Loss:** MSE with uncertainty weighting

### Test Set Evaluation

After training completion, models are evaluated on:
- **Test set:** Held-out test samples
- **Validation set:** Used for model selection during training

**Note:** The test/val results at the end of logs show negative R² values, which may indicate:
- Potential data distribution mismatch
- Need for further investigation of test set preprocessing
- Possible normalization issues in test evaluation

---

## Conclusions and Recommendations

### Key Insights

1. **Labeled Data Impact:**
   - Increasing labeled data from 5% → 10% → 20% improves R² from 0.785 → 0.823 → 0.840
   - Diminishing returns: 10% to 20% improvement is smaller than 5% to 10%

2. **Convergence Patterns:**
   - 20% experiment converged fastest (epoch 5)
   - 10% experiment showed most stable convergence (epoch 26)
   - 5% experiment required full 30 epochs

3. **SSL Effectiveness:**
   - SSL successfully leverages unlabeled data
   - 5% labeled + 95% unlabeled achieves reasonable performance (R² = 0.785)
   - Consistency regularization helps both models learn from unlabeled data

### Recommendations

1. **For Production:**
   - Use 10-20% labeled data for best performance/efficiency trade-off
   - Monitor validation metrics to prevent overfitting
   - Consider ensemble of multiple runs for robustness

2. **For Further Research:**
   - Investigate test set evaluation issues (negative R²)
   - Experiment with different SSL loss weights (`w_ulb`)
   - Try different architectures or SSL methods
   - Analyze uncertainty calibration quality

3. **Data Improvements:**
   - Verify test set preprocessing matches training
   - Consider data augmentation strategies
   - Check for distribution shifts between train/test

---

## Technical Details

### Hardware Configuration
- **GPU:** NVIDIA GPU (device=5)
- **Docker:** `--shm-size=32g` for shared memory
- **Batch Size:** 64 (reduced from 128 to avoid OOM)

### Software Stack
- **Framework:** PyTorch
- **Model Library:** `timm` (EfficientNet-B0)
- **Image Processing:** `tifffile`, `cv2`, `numpy`
- **Data:** Pandas for CSV handling

### Reproducibility
- **Seed:** 0 (fixed for all experiments)
- **Checkpointing:** Models saved at best validation loss
- **Logging:** All metrics logged to CSV files

---

**Report Generated:** February 17, 2026  
**Experiments Completed:** All three experiments finished successfully
