# Docker Quick Start Guide

This guide provides quick instructions for running UCVME in a Docker container.

## Quick Start

### 1. Build the Docker Image

```bash
docker-compose build
# or
docker build -t ucvme:latest .
```

### 2. Run Training

```bash
# Using the helper script
./docker-run.sh train /workspace/output

# Or using docker-compose directly
docker-compose run --rm ucvme python3 ucvme_age.py --output=/workspace/output
```

### 3. Run Testing

```bash
# Using the helper script
./docker-run.sh test /workspace/output

# Or using docker-compose directly
docker-compose run --rm ucvme python3 ucvme_age.py --output=/workspace/output --test_only
```

### 4. Interactive Shell

```bash
# Using the helper script
./docker-run.sh shell

# Or using docker-compose directly
docker-compose run --rm ucvme bash
```

## Directory Structure

### For UTKFace Dataset

Make sure your directory structure looks like this:

```
UCVME/
├── DATA_DIR/
│   ├── FileList.csv
│   └── UTKFace/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── output/          # Will be created automatically
└── ...
```

### For So2Sat_POP Dataset

The So2Sat_POP data should be accessible at `/work/ammar/sslrp/data/So2Sat_POP` (or update the path in docker-compose.yml):

```
/work/ammar/sslrp/data/So2Sat_POP/
├── So2Sat_POP_Part1/
│   ├── train/
│   └── test/
├── So2Sat_POP_Part2/
│   ├── train/
│   └── test/
└── FileList.csv  (created by helper script)
```

## GPU Support

### For Docker Compose

Edit `docker-compose.yml` and uncomment the GPU section:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### For Docker Directly

```bash
docker run -it --rm \
  --gpus all \
  -v $(pwd)/DATA_DIR:/workspace/DATA_DIR \
  -v $(pwd)/output:/workspace/output \
  ucvme:latest bash
```

## CPU-Only Usage

The Docker image works with CPU as well. The code will automatically detect if CUDA is available and fall back to CPU if not.

## Troubleshooting

### Permission Issues

If you encounter permission issues with mounted volumes:

```bash
# Fix permissions (Linux/Mac)
sudo chown -R $USER:$USER DATA_DIR output
```

### Out of Memory

If you run out of memory during training, reduce the batch size:

```bash
docker-compose run --rm ucvme python3 ucvme_age.py \
  --output=/workspace/output \
  --batch_size=16
```

### CUDA Errors

If you see CUDA-related errors but want to use CPU:

```bash
# Set CUDA_VISIBLE_DEVICES to empty
docker-compose run --rm -e CUDA_VISIBLE_DEVICES="" ucvme \
  python3 ucvme_age.py --output=/workspace/output
```

## Environment Details

- **Base Image**: `pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime`
- **Python**: 3.9 (from PyTorch base image)
- **PyTorch**: 1.11.0
- **CUDA**: 11.3 (if GPU support is enabled)

## Running with So2Sat_POP Dataset

### Step 1: Create FileList.csv (inside or outside Docker)

**Option A: Inside Docker container:**
```bash
# Start interactive shell
docker-compose run --rm ucvme bash

# Inside container, create FileList.csv
python scripts/create_so2sat_filelist.py \
    --data_dir /workspace/data/So2Sat_POP \
    --output /workspace/data/So2Sat_POP/FileList.csv
```

**Option B: Outside Docker (on host):**
```bash
# Run from host machine
python scripts/create_so2sat_filelist.py \
    --data_dir /work/ammar/sslrp/data/So2Sat_POP \
    --output /work/ammar/sslrp/data/So2Sat_POP/FileList.csv
```

### Step 2: Update Config File

Edit `configs/so2sat_pop_resnet50.yaml` or `configs/so2sat_pop_efficientnetb0.yaml` and ensure:
- `data_dir: "/workspace/data/So2Sat_POP"` (Docker path)
- Calculate and update `y_mean` and `y_std` from your data

### Step 3: Run Training

```bash
# ResNet50
docker-compose run --rm ucvme python3 ucvme_age.py \
    --config=/workspace/configs/so2sat_pop_resnet50.yaml \
    --output=/workspace/output/so2sat_pop_resnet50

# EfficientNetB0
docker-compose run --rm ucvme python3 ucvme_age.py \
    --config=/workspace/configs/so2sat_pop_efficientnetb0.yaml \
    --output=/workspace/output/so2sat_pop_effnetb0
```

### Step 4: Run Testing

```bash
docker-compose run --rm ucvme python3 ucvme_age.py \
    --config=/workspace/configs/so2sat_pop_resnet50.yaml \
    --output=/workspace/output/so2sat_pop_resnet50 \
    --weights=/workspace/output/so2sat_pop_resnet50/best.pt \
    --test_only
```

## Additional Options

You can pass any command-line arguments to the training script:

```bash
docker-compose run --rm ucvme python3 ucvme_age.py \
  --output=/workspace/output \
  --num_epochs=50 \
  --batch_size=16 \
  --lr=0.0001
```

See the main README.md and SO2SAT_POP_USAGE.md for all available options.

