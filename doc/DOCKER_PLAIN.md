# Running UCVME with Plain Docker (No docker-compose)

This guide shows how to build and run UCVME using plain `docker` commands when `docker-compose` is not available.

## Quick Start

### Step 1: Build the Docker Image

```bash
./docker-build.sh
```

Or manually:
```bash
docker build -t ucvme:latest .
```

### Step 2: Create FileList.csv (for So2Sat_POP)

```bash
./docker-run-plain.sh create-filelist
```

Or manually:
```bash
docker run -it --rm \
    -v /work/ammar/sslrp/data/So2Sat_POP:/workspace/data/So2Sat_POP \
    -v ./scripts:/workspace/scripts \
    -w /workspace \
    ucvme:latest \
    python scripts/create_so2sat_filelist.py \
        --data_dir /workspace/data/So2Sat_POP \
        --output /workspace/data/So2Sat_POP/FileList.csv
```

### Step 3: Run Training

**Using the script:**
```bash
./docker-run-plain.sh train \
    /workspace/configs/so2sat_pop_resnet50.yaml \
    /workspace/output/so2sat_pop_resnet50
```

**Manually:**
```bash
docker run -it --rm \
    --gpus all \
    -v /work/ammar/sslrp/data/So2Sat_POP:/workspace/data/So2Sat_POP \
    -v ./output:/workspace/output \
    -v ./configs:/workspace/configs \
    -v ./scripts:/workspace/scripts \
    -w /workspace \
    ucvme:latest \
    python3 ucvme_age.py \
        --config=/workspace/configs/so2sat_pop_resnet50.yaml \
        --output=/workspace/output/so2sat_pop_resnet50
```

### Step 4: Run Testing

```bash
./docker-run-plain.sh test \
    /workspace/configs/so2sat_pop_resnet50.yaml \
    /workspace/output/so2sat_pop_resnet50 \
    /workspace/output/so2sat_pop_resnet50/best.pt
```

## Available Commands

### Build Image
```bash
./docker-run-plain.sh build
```

### Interactive Shell
```bash
./docker-run-plain.sh shell
```

### Training
```bash
./docker-run-plain.sh train [config] [output_dir]
```

### Testing
```bash
./docker-run-plain.sh test [config] [output_dir] [weights_path]
```

### Create FileList.csv
```bash
./docker-run-plain.sh create-filelist
```

## Manual Docker Commands

If you prefer to run docker commands directly:

### Build
```bash
docker build -t ucvme:latest .
```

### Interactive Shell
```bash
docker run -it --rm \
    --gpus all \
    -v /work/ammar/sslrp/data/So2Sat_POP:/workspace/data/So2Sat_POP \
    -v ./output:/workspace/output \
    -v ./configs:/workspace/configs \
    -v ./scripts:/workspace/scripts \
    -v ./DATA_DIR:/workspace/DATA_DIR \
    -w /workspace \
    ucvme:latest bash
```

### Training
```bash
docker run -it --rm \
    --gpus all \
    -v /work/ammar/sslrp/data/So2Sat_POP:/workspace/data/So2Sat_POP \
    -v ./output:/workspace/output \
    -v ./configs:/workspace/configs \
    -v ./scripts:/workspace/scripts \
    -w /workspace \
    ucvme:latest \
    python3 ucvme_age.py \
        --config=/workspace/configs/so2sat_pop_resnet50.yaml \
        --output=/workspace/output/so2sat_pop_resnet50
```

### Testing
```bash
docker run -it --rm \
    --gpus all \
    -v /work/ammar/sslrp/data/So2Sat_POP:/workspace/data/So2Sat_POP \
    -v ./output:/workspace/output \
    -v ./configs:/workspace/configs \
    -w /workspace \
    ucvme:latest \
    python3 ucvme_age.py \
        --config=/workspace/configs/so2sat_pop_resnet50.yaml \
        --output=/workspace/output/so2sat_pop_resnet50 \
        --weights=/workspace/output/so2sat_pop_resnet50/best.pt \
        --test_only
```

## GPU Support

### With GPU (NVIDIA)
Add `--gpus all` flag:
```bash
docker run -it --rm --gpus all ...
```

### Without GPU (CPU only)
Remove the `--gpus all` flag:
```bash
docker run -it --rm ...
```

## Volume Mounts Explained

- `/work/ammar/sslrp/data/So2Sat_POP:/workspace/data/So2Sat_POP` - So2Sat_POP dataset
- `./output:/workspace/output` - Output directory for results
- `./configs:/workspace/configs` - Configuration files
- `./scripts:/workspace/scripts` - Helper scripts
- `./DATA_DIR:/workspace/DATA_DIR` - UTKFace dataset (if using)

## Troubleshooting

### Permission Denied
If you get permission errors:
```bash
sudo docker run ...
```

Or add your user to docker group:
```bash
sudo usermod -aG docker $USER
# Then logout and login again
```

### Image Not Found
Make sure you've built the image:
```bash
docker build -t ucvme:latest .
```

### Path Not Found
Check that the data directory exists:
```bash
ls -la /work/ammar/sslrp/data/So2Sat_POP
```

### GPU Not Working
Check if nvidia-docker is installed:
```bash
docker run --rm --gpus all nvidia/cuda:11.3.0-base-ubuntu20.04 nvidia-smi
```

If that fails, you may need to install nvidia-docker2 or use CPU mode.

## Complete Example Workflow

```bash
# 1. Build image
./docker-build.sh

# 2. Create FileList.csv
./docker-run-plain.sh create-filelist

# 3. Calculate normalization (optional)
docker run -it --rm \
    -v /work/ammar/sslrp/data/So2Sat_POP:/workspace/data/So2Sat_POP \
    -w /workspace \
    ucvme:latest \
    python -c "import pandas as pd; df=pd.read_csv('/workspace/data/So2Sat_POP/FileList.csv'); train=df[df['SPLIT']=='TRAIN']; print(f'y_mean: {train[\"POP\"].mean():.2f}, y_std: {train[\"POP\"].std():.2f}')"

# 4. Train
./docker-run-plain.sh train \
    /workspace/configs/so2sat_pop_resnet50.yaml \
    /workspace/output/so2sat_pop_run1

# 5. Test
./docker-run-plain.sh test \
    /workspace/configs/so2sat_pop_resnet50.yaml \
    /workspace/output/so2sat_pop_run1 \
    /workspace/output/so2sat_pop_run1/best.pt
```

