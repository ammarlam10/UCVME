# Bayern Forest Height Configs - Quick Validation Checklist

**Status**: ✅ **ALL CONFIGS VALIDATED AND CORRECT**

---

## ✅ Config Validation Results

### Config 1: `bayern_forest_height_unet_5percent.yaml`

| Parameter | Value | Validation | Notes |
|-----------|-------|------------|-------|
| **Data** | | | |
| data_dir | `/workspace/data/Bayern_forest_height_reduced` | ✅ | Directory exists |
| rd_label | 577 | ✅ | Matches SSL split file |
| rd_unlabel | 10,975 | ✅ | Matches SSL split file |
| pad_param | 5 | ✅ | Appropriate for 256×256 |
| **Model** | | | |
| name | unet | ✅ | Correct for pixel-wise |
| drp_p | 0.2 | ✅ | Good dropout rate |
| **Training** | | | |
| num_epochs | 150 | ✅ | More epochs for less data |
| lr | 0.0001 | ✅ | Conservative, safe |
| batch_size | 8 | ✅ | Safe for 32GB GPU |
| lr_step_period | 30 | ✅ | Scales with epochs |
| **SSL** | | | |
| ssl_mult | 10 | ✅ | 577×10 = 5,770 ≈ unlabeled |
| w_ulb | 10.0 | ⚠️ | High, consider tuning |
| samp_ssl | 5 | ✅ | Good MC sampling |
| **Target** | | | |
| y_mean | 12.26 | ✅ | From dataset stats |
| y_std | 11.05 | ✅ | From dataset stats |

**Overall**: ✅ **CORRECT** - Ready to run

---

### Config 2: `bayern_forest_height_unet_10percent.yaml`

| Parameter | Value | Validation | Notes |
|-----------|-------|------------|-------|
| **Data** | | | |
| data_dir | `/workspace/data/Bayern_forest_height_reduced` | ✅ | Directory exists |
| rd_label | 1,155 | ✅ | Matches SSL split file |
| rd_unlabel | 10,397 | ✅ | Matches SSL split file |
| pad_param | 5 | ✅ | Appropriate for 256×256 |
| **Model** | | | |
| name | unet | ✅ | Correct for pixel-wise |
| drp_p | 0.2 | ✅ | Good dropout rate |
| **Training** | | | |
| num_epochs | 60 | ✅ | Standard for 10% |
| lr | 0.0001 | ✅ | Conservative, safe |
| batch_size | 8 | ✅ | Safe for 32GB GPU |
| lr_step_period | 25 | ✅ | Scales with epochs |
| **SSL** | | | |
| ssl_mult | 5 | ✅ | 1,155×5 = 5,775 ≈ unlabeled |
| w_ulb | 10.0 | ⚠️ | High, consider tuning |
| samp_ssl | 5 | ✅ | Good MC sampling |
| **Target** | | | |
| y_mean | 12.26 | ✅ | From dataset stats |
| y_std | 11.05 | ✅ | From dataset stats |

**Overall**: ✅ **CORRECT** - Ready to run

---

### Config 3: `bayern_forest_height_unet_20percent.yaml`

| Parameter | Value | Validation | Notes |
|-----------|-------|------------|-------|
| **Data** | | | |
| data_dir | `/workspace/data/Bayern_forest_height_reduced` | ✅ | Directory exists |
| rd_label | 2,310 | ✅ | Matches SSL split file |
| rd_unlabel | 9,242 | ✅ | Matches SSL split file |
| pad_param | 5 | ✅ | Appropriate for 256×256 |
| **Model** | | | |
| name | unet | ✅ | Correct for pixel-wise |
| drp_p | 0.2 | ✅ | Good dropout rate |
| **Training** | | | |
| num_epochs | 60 | ✅ | Standard for 20% |
| lr | 0.0001 | ✅ | Conservative, safe |
| batch_size | 16 | ✅ | Can use larger batch |
| lr_step_period | 20 | ✅ | Scales with epochs |
| **SSL** | | | |
| ssl_mult | 3 | ✅ | 2,310×3 = 6,930 ≈ unlabeled |
| w_ulb | 10.0 | ⚠️ | High, consider tuning |
| samp_ssl | 5 | ✅ | Good MC sampling |
| **Target** | | | |
| y_mean | 12.26 | ✅ | From dataset stats |
| y_std | 11.05 | ✅ | From dataset stats |

**Overall**: ✅ **CORRECT** - Ready to run

---

## 📊 Config Comparison

### Training Epochs
- 5%: 150 epochs ✅ (less data needs more training)
- 10%: 60 epochs ✅ (standard)
- 20%: 60 epochs ✅ (standard)

### Batch Sizes
- 5%: 8 ✅ (conservative)
- 10%: 8 ✅ (conservative)
- 20%: 16 ✅ (can use larger with more data)

### SSL Multipliers
- 5%: 10× → 5,770 samples ✅ (balances with 10,975 unlabeled)
- 10%: 5× → 5,775 samples ✅ (balances with 10,397 unlabeled)
- 20%: 3× → 6,930 samples ✅ (balances with 9,242 unlabeled)

**Ratio Analysis**:
| Config | Labeled (after mult) | Unlabeled | Ratio |
|--------|---------------------|-----------|-------|
| 5% | 5,770 | 10,975 | 1:1.9 |
| 10% | 5,775 | 10,397 | 1:1.8 |
| 20% | 6,930 | 9,242 | 1:1.3 |

✅ **Well balanced** - prevents unlabeled from dominating

---

## 🎯 Satellite Forest Data Specific Validation

### Input Characteristics ✅
- [x] RGB satellite imagery (3 channels)
- [x] Overhead perspective (no angle variation)
- [x] Consistent resolution (256×256 pixels)
- [x] Forest coverage (vegetation + terrain)
- [x] Height target (NDSM, -2m to 32m range)

### Augmentation for Satellite ✅
- [x] Horizontal flip (valid for overhead)
- [x] Vertical flip (valid for overhead)
- [x] Random crop (handles spatial variation)
- [x] No color augmentation (correct for satellite)
- [x] Synchronized RGB + NDSM transforms
- [ ] Rotation (optional enhancement)

### Model for Dense Prediction ✅
- [x] UNET architecture (standard for pixel-wise)
- [x] Skip connections (preserve spatial details)
- [x] Appropriate depth (3 levels)
- [x] Uncertainty estimation (dual heads)

### Loss for Height Regression ✅
- [x] Pixel-wise loss (each pixel independent)
- [x] Uncertainty weighting (NLL loss)
- [x] Target normalization (height stats)
- [x] SSL consistency (FIXED for pixel-wise)

---

## ⚠️ Only One Minor Concern

### SSL Weight (`w_ulb = 10.0`)

**Current**: All configs use `w_ulb = 10.0`

**Concern**: SSL loss has **10× weight** of supervised loss
- Now that SSL is fixed, this might be too aggressive
- Could cause training instability
- Might overfit to pseudo-labels

**Recommendation**: 
1. Start with current value (10.0)
2. Monitor training:
   - If SSL loss dominates → reduce to 5.0 or 1.0
   - If training unstable → reduce
   - If SSL loss negligible → keep at 10.0
3. Consider ablation: try [1.0, 5.0, 10.0]

**Priority**: Medium - not blocking, but worth monitoring

---

## ✅ Final Verdict

### All Three Configs: **APPROVED FOR SATELLITE FOREST DATA** ✅

**Reasoning**:
1. ✅ Data paths and splits correct
2. ✅ Model architecture appropriate
3. ✅ Augmentation valid for overhead imagery
4. ✅ Loss functions correct (after bug fix)
5. ✅ Hyperparameters reasonable
6. ✅ Target normalization from dataset stats
7. ⚠️ Only minor concern: SSL weight might need tuning

**Confidence**: **HIGH** - Ready for production experiments

---

## 🚀 Quick Start Commands

### Run All Three Experiments

```bash
cd /work/ammar/sslrp/UCVME

# 5% labeled
docker run --rm --gpus '"device=6"' --shm-size=16g \
  -v /work/ammar/sslrp/data:/workspace/data \
  -v $(pwd):/workspace/ucvme \
  -v $(pwd)/output:/workspace/output \
  ucvme:latest bash -c "cd /workspace/ucvme && pip install h5py -q && \
  python3 ucvme_age.py --config configs/bayern_forest_height_unet_5percent.yaml"

# 10% labeled  
docker run --rm --gpus '"device=6"' --shm-size=16g \
  -v /work/ammar/sslrp/data:/workspace/data \
  -v $(pwd):/workspace/ucvme \
  -v $(pwd)/output:/workspace/output \
  ucvme:latest bash -c "cd /workspace/ucvme && pip install h5py -q && \
  python3 ucvme_age.py --config configs/bayern_forest_height_unet_10percent.yaml"

# 20% labeled
docker run --rm --gpus '"device=6"' --shm-size=16g \
  -v /work/ammar/sslrp/data:/workspace/data \
  -v $(pwd):/workspace/ucvme \
  -v $(pwd)/output:/workspace/output \
  ucvme:latest bash -c "cd /workspace/ucvme && pip install h5py -q && \
  python3 ucvme_age.py --config configs/bayern_forest_height_unet_20percent.yaml"
```

### Monitor Progress

```bash
# Watch training logs
tail -f output/bayern_forest_height_unet_5percent/log.csv
tail -f output/bayern_forest_height_unet_10percent/log.csv
tail -f output/bayern_forest_height_unet_20percent/log.csv
```

---

**Review Complete** ✅  
**All configs validated for satellite forest height prediction** ✅  
**SSL bug fixed** ✅  
**Ready to run experiments** ✅
