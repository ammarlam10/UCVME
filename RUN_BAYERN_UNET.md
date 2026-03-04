# Quick Start: Bayern Forest Height UNET Training

## Prerequisites

```bash
# Install h5py if not already installed
pip install h5py>=3.0.0
```

## Run Training

### Option 1: Direct Python
```bash
cd /work/ammar/sslrp/UCVME
python3 ucvme_age.py --config configs/bayern_forest_height_unet.yaml
```

### Option 2: Using Docker
```bash
cd /work/ammar/sslrp/UCVME

# Run training in Docker (with increased shared memory for large datasets)
docker run --rm --gpus '"device=0,1"' \
  --shm-size=8g \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -v /work/ammar/sslrp/data:/workspace/data \
  -v $(pwd):/workspace/ucvme \
  -v $(pwd)/output:/workspace/output \
  ucvme:latest \
  bash -c "cd /workspace/ucvme && pip install h5py -q && python3 ucvme_age.py --config configs/bayern_forest_height_unet.yaml"
```

## Test Implementation First

Before running full training, test the implementation:

```bash
# Test model and dataset
python3 test_bayern_unet.py

# Inspect dataset
python3 inspect_bayern_dataset.py
```

## Monitor Training

Training outputs will be saved to:
```
output/bayern_forest_height_unet/
├── log.csv                    # Training metrics
├── checkpoint.pt              # Latest checkpoint
├── best.pt                    # Best model (lowest val loss)
├── train_pred_{epoch}.csv     # Training predictions
├── val_predmcd0_{epoch}.csv   # Validation predictions
└── z_val_epch{epoch}_prd.csv  # Validation results
```

## Expected Training Time

- **Dataset loading**: ~2 minutes (loads all data into memory)
- **Per epoch**: ~5-10 minutes (depends on GPU)
- **Total (50 epochs)**: ~4-8 hours

## Adjust Configuration

Edit `configs/bayern_forest_height_unet.yaml` to change:

```yaml
training:
  num_epochs: 50        # Number of epochs
  batch_size: 8         # Batch size (reduce if OOM)
  lr: 0.0001           # Learning rate
  
model:
  drp_p: 0.2           # Dropout rate
```

## Troubleshooting

### Out of Memory (OOM)
Reduce batch size in config:
```yaml
training:
  batch_size: 4  # or 2
```

### Slow Dataset Loading
The dataset loads all 14,440 samples into memory (~5GB). This is normal and happens once at startup.

### Check GPU Usage
```bash
nvidia-smi
```

## View Results

After training completes, check:

```bash
# View training log
cat output/bayern_forest_height_unet/log.csv

# View test results (at end of log)
tail -20 output/bayern_forest_height_unet/log.csv
```

## Compare with Other Models

To compare UNET with ResNet50 or EfficientNetB0 on the same dataset, you would need to:
1. Modify the dataset to output average height per image (instead of height maps)
2. Create a new config with the desired model
3. Run training with the new config

The current implementation supports both automatically!
