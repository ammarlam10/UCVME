# Output Files After Experiment Completion

This document describes **what is saved in the output directory** when a training run completes (e.g. Bayern Forest Height or any `ucvme_age.py` run).

---

## Output Directory

Controlled by config:

- **Bayern 5%**: `output/bayern_forest_height_unet_5percent/`
- **Bayern 10%**: `output/bayern_forest_height_unet_10percent/`
- **Bayern 20%**: `output/bayern_forest_height_unet_20percent/`

Or by CLI: `--output <path>`.

---

## Files Saved (Complete Run)

### 1. **log.csv** (main log)

**Purpose**: Training and validation metrics, one row per epoch phase.

**Contents**:
- **Header (first run only)**:
  - `Run timestamp: YYYYMMDD_HHMMSS`
  - `Starting run from scratch` or `Resuming from epoch N`
  - `# train row: epoch,phase,loss,r2_0,r2_1,time_sec,n_samples,mem_allocated,mem_reserved,batch_size,loss_reg_0,cps`
  - `# val row:   epoch,phase,loss,r2,mae,rmse,time_sec,n_samples,mem_allocated,mem_reserved,batch_size,0,0`

- **Per epoch – train row**:  
  `epoch,train,loss,r2_0,r2_1,time_sec,n_samples,mem_allocated,mem_reserved,batch_size,loss_reg_0,cps`

- **Per epoch – val row**:  
  `epoch,val,loss,r2,mae,rmse,time_sec,n_samples,mem_allocated,mem_reserved,batch_size,0,0`

- **After training**:  
  `Best validation loss X from epoch Y, R2 Z`

- **If `run_test: true`** (after loading best model):
  - `YYYYMMDD_HHMMSS - test (one clip) R2:   X.XXX`
  - `YYYYMMDD_HHMMSS - test (one clip) MAE:  X.XX`
  - `YYYYMMDD_HHMMSS - test (one clip) RMSE: X.XX`
  - Same three lines for `val`.

**Use**: Plot learning curves, compare runs, report final R²/MAE/RMSE.

---

### 2. **checkpoint.pt** (latest checkpoint)

**Purpose**: Resume training from the last epoch.

**Saved**: Every epoch (overwrites previous).

**Contents** (Python dict):
- `epoch` – last completed epoch index
- `state_dict` – model 0 weights
- `state_dict_1` – model 1 weights
- `best_loss` – best validation loss so far
- `loss` – last validation loss
- `best_model_loss` – validation loss of best model
- `r2` – R² of best model (validation)
- `opt_dict`, `scheduler_dict` – optimizer and scheduler for model 0
- `opt_dict_1`, `scheduler_dict_1` – for model 1
- `np_rndstate`, `trch_rndstate` – random states for reproducibility

**Use**: Resume with same config; no need to change script.

---

### 3. **best.pt** (best model)

**Purpose**: Best model by validation loss for evaluation and deployment.

**Saved**: Only when validation loss improves (overwrites previous).

**Contents**: Same structure as `checkpoint.pt`, but from the epoch with lowest validation loss.

**Use**:
- Load for final test/val metrics (script does this when `run_test: true`).
- Load for inference:
  ```python
  ckpt = torch.load("output/.../best.pt")
  model.load_state_dict(ckpt["state_dict"])
  model_1.load_state_dict(ckpt["state_dict_1"])
  ```

---

### 4. **train_pred_{epoch}.csv** (training predictions, per epoch) — **disabled by default**

**Purpose**: Training-set predictions and uncertainties (both models, MC samples).

**Saved**: Only if `SAVE_PREDICTION_CSVS = True` in `ucvme_age.py` (default: `False`).

**Columns**:
- `m_0_0`, `m_0_1`, … `m_0_{samp_fq-1}` – model 0 mean over MC samples
- `m_1_0`, … – model 1 mean
- `v_0_0`, … – model 0 variance
- `v_1_0`, … – model 1 variance  

(So 4 × `samp_fq` columns.)

**Rows**:
- **Image-level**: one row per training sample.
- **Pixel-wise (e.g. Bayern)**: one row per **pixel** (all batches, flattened).  
  So row count = (train samples) × (H×W). For 256×256 and thousands of samples this is huge (e.g. tens of GB per file).

**Use**: Debug training, analyze uncertainties. For pixel-wise, consider disabling or saving only a subset if disk is limited.

---

### 5. **val_predmcd0_{epoch}.csv** (validation predictions, model 0, per epoch) — **disabled by default**

**Purpose**: Validation-set predictions from model 0 over MC samples.

**Saved**: Only if `SAVE_PREDICTION_CSVS = True` in `ucvme_age.py` (default: `False`).

**Columns**: `m_0_0`, …, `m_0_{samp_fq-1}`, `v_0_0`, …, `v_0_{samp_fq-1}`.

**Rows**: Same as above (one per sample for image-level, one per pixel for pixel-wise).

**Use**: Validation uncertainty analysis. Again, can be very large for pixel-wise.

---

### 6. **z_val_epch{epoch}_prd.csv** (validation predictions summary, per epoch) — **disabled by default**

**Purpose**: Per-sample (or per-pixel) validation summary: prediction, target, and variances.

**Saved**: Only if `SAVE_PREDICTION_CSVS = True` in `ucvme_age.py` (default: `False`).

**Columns**: `yhat,y,var_hat,var_e,var_a`
- `yhat` – predicted value (denormalized)
- `y` – ground truth
- `var_hat` – total predictive variance (normalized)
- `var_e` – epistemic variance
- `var_a` – aleatoric variance

**Rows**: One per validation sample (image-level) or per pixel (pixel-wise). For pixel-wise, file can be very large.

**Use**: Plot predictions vs targets, analyze calibration of uncertainty.

---

## Summary Table

| File | When | Size (typical) | Main use |
|------|------|----------------|----------|
| **log.csv** | Always | KB | Metrics, curves, final R²/MAE/RMSE |
| **checkpoint.pt** | Every epoch | ~500MB–1GB+ | Resume training |
| **best.pt** | When val improves | ~500MB–1GB+ | Best model, inference |
| **train_pred_{epoch}.csv** | When `SAVE_PREDICTION_CSVS` | Huge for pixel-wise | Debug / analysis (off by default) |
| **val_predmcd0_{epoch}.csv** | When `SAVE_PREDICTION_CSVS` | Huge for pixel-wise | Val uncertainty (off by default) |
| **z_val_epch{epoch}_prd.csv** | When `SAVE_PREDICTION_CSVS` | Large for pixel-wise | Val predictions vs y (off by default) |

---

## After a Full Run You Can Rely On

- **log.csv** – full training/validation history and final test/val metrics.
- **best.pt** – best model (and best model’s epoch, loss, R² in the same dict).

Optional (and heavy for pixel-wise):

- **checkpoint.pt** – resume from last epoch.
- **train_pred_*.csv**, **val_predmcd0_*.csv**, **z_val_epch*_prd.csv** – detailed predictions and uncertainties.

---

## Pixel-wise (Bayern) Note

For Bayern Forest Height (and any pixel-wise setup), the prediction CSVs have **one row per pixel** (e.g. 256×256 per image × number of images). So:

- `train_pred_0.csv` can be tens of GB.
- Val prediction files scale similarly.

If you only need metrics and the best model, you can:

- Rely on **log.csv** and **best.pt**.
- Optionally change the code to skip writing `train_pred_*.csv`, `val_predmcd0_*.csv`, and `z_val_epch*_prd.csv` for pixel-wise, or write them only for a subset of epochs/samples.

---

## Quick Commands

```bash
# View training progress
tail -20 output/bayern_forest_height_unet_20percent/log.csv

# Final test/val metrics (when run_test: true)
grep -E "Best validation|R2:|MAE:|RMSE:" output/bayern_forest_height_unet_20percent/log.csv

# List all outputs for a run
ls -lh output/bayern_forest_height_unet_20percent/
```
