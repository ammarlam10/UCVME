# Bayern Forest Height UNET SSL Experiments

## SSL Splits Created

✓ **5% labeled**: 577 labeled + 10,975 unlabeled = 11,552 training samples
✓ **10% labeled**: 1,155 labeled + 10,397 unlabeled = 11,552 training samples  
✓ **20% labeled**: 2,310 labeled + 9,242 unlabeled = 11,552 training samples

## Hardware Configuration

- **GPUs**: 0 and 3 (Tesla V100 32GB each)
- **Batch size**: 16 (optimized for 2x V100 32GB)
- **Workers**: 6 (with 16GB shared memory)
- **Total memory**: ~64GB GPU + 16GB shared memory

## Docker Commands

### 5% Labeled Data (577 samples)

```bash
cd /work/ammar/sslrp/UCVME

docker run --rm --gpus '"device=0,3"' \
  --shm-size=16g \
  -e CUDA_VISIBLE_DEVICES=0,3 \
  -v /work/ammar/sslrp/data:/workspace/data \
  -v $(pwd):/workspace/ucvme \
  -v $(pwd)/output:/workspace/output \
  ucvme:latest \
  bash -c "cd /workspace/ucvme && pip install h5py -q && \
           python3 ucvme_age.py --config configs/bayern_forest_height_unet_5percent.yaml"
```

**Expected time**: ~20-25 hours (150 epochs × ~8-10 min/epoch)

---

### 10% Labeled Data (1,155 samples)

```bash
cd /work/ammar/sslrp/UCVME

docker run --rm --gpus '"device=0,3"' \
  --shm-size=16g \
  -e CUDA_VISIBLE_DEVICES=0,3 \
  -v /work/ammar/sslrp/data:/workspace/data \
  -v $(pwd):/workspace/ucvme \
  -v $(pwd)/output:/workspace/output \
  ucvme:latest \
  bash -c "cd /workspace/ucvme && pip install h5py -q && \
           python3 ucvme_age.py --config configs/bayern_forest_height_unet_10percent.yaml"
```

**Expected time**: ~16-20 hours (120 epochs × ~8-10 min/epoch)

---

### 20% Labeled Data (2,310 samples)

```bash
cd /work/ammar/sslrp/UCVME

docker run --rm --gpus '"device=0,3"' \
  --shm-size=16g \
  -e CUDA_VISIBLE_DEVICES=0,3 \
  -v /work/ammar/sslrp/data:/workspace/data \
  -v $(pwd):/workspace/ucvme \
  -v $(pwd)/output:/workspace/output \
  ucvme:latest \
  bash -c "cd /workspace/ucvme && pip install h5py -q && \
           python3 ucvme_age.py --config configs/bayern_forest_height_unet_20percent.yaml"
```

**Expected time**: ~13-17 hours (100 epochs × ~8-10 min/epoch)

---

## Run All Experiments Sequentially

```bash
cd /work/ammar/sslrp/UCVME

# 5% experiment
docker run --rm --gpus '"device=0,3"' --shm-size=16g -e CUDA_VISIBLE_DEVICES=0,3 \
  -v /work/ammar/sslrp/data:/workspace/data -v $(pwd):/workspace/ucvme -v $(pwd)/output:/workspace/output \
  ucvme:latest bash -c "cd /workspace/ucvme && pip install h5py -q && \
  python3 ucvme_age.py --config configs/bayern_forest_height_unet_5percent.yaml"

# 10% experiment
docker run --rm --gpus '"device=0,3"' --shm-size=16g -e CUDA_VISIBLE_DEVICES=0,3 \
  -v /work/ammar/sslrp/data:/workspace/data -v $(pwd):/workspace/ucvme -v $(pwd)/output:/workspace/output \
  ucvme:latest bash -c "cd /workspace/ucvme && pip install h5py -q && \
  python3 ucvme_age.py --config configs/bayern_forest_height_unet_10percent.yaml"

# 20% experiment
docker run --rm --gpus '"device=0,3"' --shm-size=16g -e CUDA_VISIBLE_DEVICES=0,3 \
  -v /work/ammar/sslrp/data:/workspace/data -v $(pwd):/workspace/ucvme -v $(pwd)/output:/workspace/output \
  ucvme:latest bash -c "cd /workspace/ucvme && pip install h5py -q && \
  python3 ucvme_age.py --config configs/bayern_forest_height_unet_20percent.yaml"
```

**Total time**: ~50-62 hours for all three experiments

---

## Monitor Training

### Watch logs in real-time

```bash
# 5% experiment
tail -f output/bayern_forest_height_unet_5percent/log.csv

# 10% experiment
tail -f output/bayern_forest_height_unet_10percent/log.csv

# 20% experiment
tail -f output/bayern_forest_height_unet_20percent/log.csv
```

### Check GPU usage

```bash
watch -n 1 nvidia-smi
```

### View results

```bash
# View final test results
tail -20 output/bayern_forest_height_unet_5percent/log.csv
tail -20 output/bayern_forest_height_unet_10percent/log.csv
tail -20 output/bayern_forest_height_unet_20percent/log.csv
```

---

## Configuration Details

| Experiment | Labeled | Unlabeled | SSL Mult | Epochs | Batch Size | Workers | Shared Mem |
|------------|---------|-----------|----------|--------|------------|---------|------------|
| 5%         | 577     | 10,975    | 10x      | 150    | 16         | 6       | 16GB       |
| 10%        | 1,155   | 10,397    | 5x       | 120    | 16         | 6       | 16GB       |
| 20%        | 2,310   | 9,242     | 3x       | 100    | 16         | 6       | 16GB       |

### Key Parameters

- **Model**: UNET with 3 downsampling layers (~7.2M parameters)
- **Dropout**: 0.2 (for MC Dropout uncertainty)
- **Learning rate**: 0.0001
- **Weight decay**: 0.001
- **SSL weight (w_ulb)**: 10.0
- **Target normalization**: mean=12.26, std=11.05

---

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size:
```bash
python3 ucvme_age.py --config configs/bayern_forest_height_unet_10percent.yaml --batch_size 8
```

### Shared Memory Issues

If you see "Bus error" or shared memory errors, increase `--shm-size`:
```bash
--shm-size=24g  # or even 32g
```

Or reduce workers:
```bash
python3 ucvme_age.py --config configs/bayern_forest_height_unet_10percent.yaml --num_workers 4
```

### Check if experiment is running

```bash
# Check Docker containers
docker ps

# Check GPU usage
nvidia-smi

# Check if output directory is being updated
ls -lth output/bayern_forest_height_unet_10percent/
```

---

## Expected Results

Based on similar SSL experiments:

| Labeled % | Expected Val R² | Expected Test R² | Expected MAE (m) |
|-----------|----------------|------------------|------------------|
| 5%        | 0.75-0.80      | 0.73-0.78        | 4.5-5.5          |
| 10%       | 0.80-0.85      | 0.78-0.83        | 3.5-4.5          |
| 20%       | 0.85-0.88      | 0.83-0.86        | 3.0-4.0          |

*Note: These are estimates based on similar pixel-wise regression tasks. Actual results may vary.*

---

## Output Files

Each experiment creates:

```
output/bayern_forest_height_unet_{5,10,20}percent/
├── log.csv                    # Training metrics (epoch, loss, R², MAE, RMSE)
├── checkpoint.pt              # Latest checkpoint
├── best.pt                    # Best model (lowest val loss)
├── train_pred_{epoch}.csv     # Training predictions per epoch
├── val_predmcd0_{epoch}.csv   # Validation predictions (MC Dropout)
└── z_val_epch{epoch}_prd.csv  # Validation results with uncertainty
```

---

## Quick Copy-Paste Commands

### Just run 10% experiment:
```bash
cd /work/ammar/sslrp/UCVME && docker run --rm --gpus '"device=0,3"' --shm-size=16g -e CUDA_VISIBLE_DEVICES=0,3 -v /work/ammar/sslrp/data:/workspace/data -v $(pwd):/workspace/ucvme -v $(pwd)/output:/workspace/output ucvme:latest bash -c "cd /workspace/ucvme && pip install h5py -q && python3 ucvme_age.py --config configs/bayern_forest_height_unet_10percent.yaml"
```

### Monitor 10% experiment:
```bash
tail -f output/bayern_forest_height_unet_10percent/log.csv
```
