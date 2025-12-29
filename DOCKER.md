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

## Additional Options

You can pass any command-line arguments to the training script:

```bash
docker-compose run --rm ucvme python3 ucvme_age.py \
  --output=/workspace/output \
  --num_epochs=50 \
  --batch_size=16 \
  --lr=0.0001
```

See the main README.md for all available options.

