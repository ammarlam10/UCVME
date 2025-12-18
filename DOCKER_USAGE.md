# Docker Usage Guide for UCVME

This guide explains how to build and run UCVME training in Docker containers.

## Prerequisites

1. **Docker** installed (version 20.10+)
2. **Docker Compose** installed (version 1.29+)
3. **NVIDIA Docker** (nvidia-docker2) for GPU support
   ```bash
   # Install nvidia-docker2
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

## Quick Start

### 1. Build Docker Image

```bash
docker-compose build
```

Or build directly:

```bash
docker build -t ucvme:latest .
```

### 2. Run Training

#### Option A: Using docker-compose (Recommended)

```bash
# Using helper script
bash scripts/docker_train.sh so2sat_pop /work/ammar/sslrp/data ./outputs/experiment 1000 5000

# Or directly with docker-compose
docker-compose run --rm ucvme python3 ucvme.py \
    --dataset so2sat_pop \
    --data_root /data \
    --output /workspace/outputs/experiment \
    --rd_label 1000 \
    --rd_unlabel 5000
```

#### Option B: Using docker run

```bash
docker run --rm --gpus all \
    -v /work/ammar/sslrp/data:/data:ro \
    -v $(pwd)/outputs:/workspace/outputs \
    -v $(pwd)/configs:/workspace/configs \
    ucvme:latest \
    python3 ucvme.py \
        --dataset so2sat_pop \
        --data_root /data \
        --output /workspace/outputs/experiment \
        --rd_label 1000 \
        --rd_unlabel 5000
```

### 3. Interactive Shell

To get an interactive shell in the container:

```bash
docker-compose run --rm ucvme /bin/bash
```

Or with docker run:

```bash
docker run --rm -it --gpus all \
    -v /work/ammar/sslrp/data:/data:ro \
    -v $(pwd)/outputs:/workspace/outputs \
    -v $(pwd)/configs:/workspace/configs \
    ucvme:latest \
    /bin/bash
```

## Volume Mounts

The Docker setup mounts:

1. **Data directory** (`/work/ammar/sslrp/data` → `/data`)
   - Read-only mount for dataset files
   - Contains So2Sat_POP, Bayern_forest_height, UTKFace_all

2. **Outputs directory** (`./outputs` → `/workspace/outputs`)
   - Read-write mount for training outputs
   - Contains checkpoints, logs, predictions

3. **Configs directory** (`./configs` → `/workspace/configs`)
   - Read-write mount for configuration files
   - Allows editing configs without rebuilding image

## GPU Support

The Dockerfile and docker-compose.yml are configured for GPU support:

- Uses `nvidia/cuda:11.8.0` base image
- Installs PyTorch with CUDA 11.8 support
- Uses `--gpus all` flag for GPU access

To check GPU availability in container:

```bash
docker-compose run --rm ucvme nvidia-smi
```

## Common Commands

### Build image
```bash
docker-compose build
```

### Run training
```bash
docker-compose run --rm ucvme python3 ucvme.py --dataset so2sat_pop --data_root /data --output /workspace/outputs/exp1
```

### Test only
```bash
docker-compose run --rm ucvme python3 ucvme.py \
    --dataset so2sat_pop \
    --data_root /data \
    --output /workspace/outputs/exp1 \
    --test_only \
    --weights /workspace/outputs/exp1/best.pt
```

### Generate FileList
```bash
docker-compose run --rm ucvme python3 scripts/generate_filelist_so2sat.py \
    --data_root /data \
    --season spring
```

### Check Python environment
```bash
docker-compose run --rm ucvme python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Troubleshooting

### Issue: "nvidia-docker not found"
**Solution**: Install nvidia-docker2 (see Prerequisites)

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or use fewer GPUs
```bash
# Use specific GPU
docker run --gpus '"device=0"' ...
```

### Issue: "Permission denied" on volumes
**Solution**: Check volume permissions or use sudo
```bash
sudo docker-compose run --rm ucvme ...
```

### Issue: "Module not found" errors
**Solution**: Rebuild the image to ensure all dependencies are installed
```bash
docker-compose build --no-cache
```

### Issue: Data not found
**Solution**: Verify data path is correct and volume is mounted
```bash
docker-compose run --rm ucvme ls -la /data
```

## Customization

### Change CUDA Version

Edit `Dockerfile`:
```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
```

And update PyTorch installation:
```dockerfile
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Add Additional Dependencies

Edit `requirements-docker.txt` and rebuild:
```bash
docker-compose build
```

### Change Base Image

Modify the `FROM` line in `Dockerfile` to use a different base image.

## Best Practices

1. **Use docker-compose** for easier volume management
2. **Mount outputs as volumes** to persist results
3. **Use read-only mounts** for data directories
4. **Rebuild image** when dependencies change
5. **Use .dockerignore** to exclude unnecessary files
6. **Tag images** for different experiments:
   ```bash
   docker build -t ucvme:v1.0 .
   ```

## Multi-GPU Training

For multi-GPU training, ensure all GPUs are accessible:

```bash
docker run --rm --gpus all \
    -v /work/ammar/sslrp/data:/data:ro \
    -v $(pwd)/outputs:/workspace/outputs \
    ucvme:latest \
    python3 ucvme.py \
        --dataset so2sat_pop \
        --data_root /data \
        --output /workspace/outputs/exp1 \
        --device cuda
```

The code uses `torch.nn.DataParallel` which will automatically use all available GPUs.

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Train UCVME

on: [push]

jobs:
  train:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v2
      - name: Build Docker image
        run: docker-compose build
      - name: Run training
        run: docker-compose run --rm ucvme python3 ucvme.py --dataset so2sat_pop --data_root /data --output /workspace/outputs/test
```

