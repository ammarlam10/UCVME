# SSL Experiments - Docker Run Commands

All configurations have been created with the following fixes applied:
1. ✅ Removed double target normalization
2. ✅ Corrected y_mean (1872.5640) and y_std (3398.8516)
3. ✅ Fixed band extraction (RGB: indices 3,2,1)
4. ✅ Removed double image normalization (kept only clipping + min-max)
5. ✅ Reduced batch_size to 64 to avoid CUDA OOM
6. ✅ Fixed GPU configuration (no CUDA_VISIBLE_DEVICES conflict)
7. ✅ Added validation R2, MAE, and RMSE logging

---

## Experiment 1: 5% SSL (3,047 labeled + 57,905 unlabeled)

**Config:** `configs/so2sat_pop_efficientnetb0_5percent_fixed.yaml`

**Docker Command:**
```bash
docker run --rm \
    --gpus device=5 \
    --shm-size=32g \
    -v /work/ammar/sslrp/data/So2Sat_POP:/workspace/data/So2Sat_POP \
    -v /work/ammar/sslrp/UCVME:/workspace \
    -v /work/ammar/sslrp/UCVME/output:/workspace/output \
    ucvme:latest \
    python3 /workspace/ucvme_age.py \
        --config=/workspace/configs/so2sat_pop_efficientnetb0_5percent_fixed.yaml \
        --output=/workspace/output/so2sat_pop_efficientnetb0_5percent_fixed
```

**Expected Output:**
- Training samples: 6,094 (after duplication with ssl_mult=-1)
- Unlabeled samples: 57,905
- Batches per epoch: ~95 (labeled) + ~904 (unlabeled)
- Output directory: `output/so2sat_pop_efficientnetb0_5percent_fixed/`

---

## Experiment 2: 10% SSL (6,095 labeled + 54,857 unlabeled)

**Config:** `configs/so2sat_pop_efficientnetb0_10percent_fixed.yaml`

**Docker Command:**
```bash
docker run --rm \
    --gpus device=5 \
    --shm-size=32g \
    -v /work/ammar/sslrp/data/So2Sat_POP:/workspace/data/So2Sat_POP \
    -v /work/ammar/sslrp/UCVME:/workspace \
    -v /work/ammar/sslrp/UCVME/output:/workspace/output \
    ucvme:latest \
    python3 /workspace/ucvme_age.py \
        --config=/workspace/configs/so2sat_pop_efficientnetb0_10percent_fixed.yaml \
        --output=/workspace/output/so2sat_pop_efficientnetb0_10percent_fixed
```

**Expected Output:**
- Training samples: 12,190 (after duplication with ssl_mult=-1)
- Unlabeled samples: 54,857
- Batches per epoch: ~190 (labeled) + ~857 (unlabeled)
- Output directory: `output/so2sat_pop_efficientnetb0_10percent_fixed/`

---

## Experiment 3: 20% SSL (12,190 labeled + 48,762 unlabeled)

**Config:** `configs/so2sat_pop_efficientnetb0_20percent_fixed.yaml`

**Docker Command:**
```bash
docker run --rm \
    --gpus device=5 \
    --shm-size=32g \
    -v /work/ammar/sslrp/data/So2Sat_POP:/workspace/data/So2Sat_POP \
    -v /work/ammar/sslrp/UCVME:/workspace \
    -v /work/ammar/sslrp/UCVME/output:/workspace/output \
    ucvme:latest \
    python3 /workspace/ucvme_age.py \
        --config=/workspace/configs/so2sat_pop_efficientnetb0_20percent_fixed.yaml \
        --output=/workspace/output/so2sat_pop_efficientnetb0_20percent_fixed
```

**Expected Output:**
- Training samples: 24,380 (after duplication with ssl_mult=-1)
- Unlabeled samples: 48,762
- Batches per epoch: ~381 (labeled) + ~762 (unlabeled)
- Output directory: `output/so2sat_pop_efficientnetb0_20percent_fixed/`

---

## Running in tmux (Recommended)

To run experiments in separate tmux sessions:

### Experiment 1 (5% SSL):
```bash
tmux new -s ssl5percent
docker run --rm \
    --gpus device=5 \
    --shm-size=32g \
    -v /work/ammar/sslrp/data/So2Sat_POP:/workspace/data/So2Sat_POP \
    -v /work/ammar/sslrp/UCVME:/workspace \
    -v /work/ammar/sslrp/UCVME/output:/workspace/output \
    ucvme:latest \
    python3 /workspace/ucvme_age.py \
        --config=/workspace/configs/so2sat_pop_efficientnetb0_5percent_fixed.yaml \
        --output=/workspace/output/so2sat_pop_efficientnetb0_5percent_fixed
# Detach: Ctrl+b, then d
```

### Experiment 2 (10% SSL):
```bash
tmux new -s ssl10percent
docker run --rm \
    --gpus device=5 \
    --shm-size=32g \
    -v /work/ammar/sslrp/data/So2Sat_POP:/workspace/data/So2Sat_POP \
    -v /work/ammar/sslrp/UCVME:/workspace \
    -v /work/ammar/sslrp/UCVME/output:/workspace/output \
    ucvme:latest \
    python3 /workspace/ucvme_age.py \
        --config=/workspace/configs/so2sat_pop_efficientnetb0_10percent_fixed.yaml \
        --output=/workspace/output/so2sat_pop_efficientnetb0_10percent_fixed
# Detach: Ctrl+b, then d
```

### Experiment 3 (20% SSL):
```bash
tmux new -s ssl20percent
docker run --rm \
    --gpus device=5 \
    --shm-size=32g \
    -v /work/ammar/sslrp/data/So2Sat_POP:/workspace/data/So2Sat_POP \
    -v /work/ammar/sslrp/UCVME:/workspace \
    -v /work/ammar/sslrp/UCVME/output:/workspace/output \
    ucvme:latest \
    python3 /workspace/ucvme_age.py \
        --config=/workspace/configs/so2sat_pop_efficientnetb0_20percent_fixed.yaml \
        --output=/workspace/output/so2sat_pop_efficientnetb0_20percent_fixed
# Detach: Ctrl+b, then d
```

---

## Monitoring Progress

### Check tmux sessions:
```bash
tmux ls
```

### Attach to a session:
```bash
tmux attach -t ssl5percent   # or ssl10percent, ssl20percent
```

### Monitor log files:
```bash
# 5% SSL
tail -f /work/ammar/sslrp/UCVME/output/so2sat_pop_efficientnetb0_5percent_fixed/log.csv

# 10% SSL
tail -f /work/ammar/sslrp/UCVME/output/so2sat_pop_efficientnetb0_10percent_fixed/log.csv

# 20% SSL
tail -f /work/ammar/sslrp/UCVME/output/so2sat_pop_efficientnetb0_20percent_fixed/log.csv
```

### Check GPU usage:
```bash
nvidia-smi
```

---

## Expected Training Time

With batch_size=64 on V100S GPU:
- **5% SSL**: ~2-3 hours per epoch (95 labeled batches)
- **10% SSL**: ~3-4 hours per epoch (190 labeled batches)
- **20% SSL**: ~5-6 hours per epoch (381 labeled batches)

Total training time (30 epochs):
- **5% SSL**: ~60-90 hours
- **10% SSL**: ~90-120 hours
- **20% SSL**: ~150-180 hours

---

## Output Files

Each experiment will create:
- `log.csv` - Training/validation metrics (loss, R2, MAE, RMSE)
- `checkpoint.pt` - Latest model checkpoint
- `best.pt` - Best model (lowest validation loss)
- `train_pred_*.csv` - Training predictions per epoch
- `val_predmcd0_*.csv` - Validation predictions per epoch
- `z_val_epch*_prd.csv` - Validation predictions with uncertainty

---

## Comparing Results

After all experiments complete, compare performance:

```bash
# Extract validation metrics from all experiments
grep "^[0-9]*,val," output/so2sat_pop_efficientnetb0_5percent_fixed/log.csv | tail -1
grep "^[0-9]*,val," output/so2sat_pop_efficientnetb0_10percent_fixed/log.csv | tail -1
grep "^[0-9]*,val," output/so2sat_pop_efficientnetb0_20percent_fixed/log.csv | tail -1
```

Expected improvement with more labeled data:
- 5% SSL: Baseline performance
- 10% SSL: ~10-20% improvement in R2 and MAE
- 20% SSL: ~20-30% improvement in R2 and MAE

---

## Files Created

1. `configs/so2sat_pop_efficientnetb0_5percent_fixed.yaml`
2. `configs/so2sat_pop_efficientnetb0_10percent_fixed.yaml`
3. `configs/so2sat_pop_efficientnetb0_20percent_fixed.yaml`
4. `RUN_ALL_SSL_EXPERIMENTS.md` (this file)

All configurations use the same corrected normalization parameters and fixes!
