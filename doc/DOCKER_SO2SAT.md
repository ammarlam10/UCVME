# Running So2Sat_POP with Docker

Quick guide for running So2Sat_POP dataset training in Docker.

## Prerequisites

1. Docker and docker-compose installed
2. So2Sat_POP data available at `/work/ammar/sslrp/data/So2Sat_POP` (or update path in `docker-compose.yml`)

## Quick Start

### Step 1: Build Docker Image (if not already built)

```bash
docker-compose build
```

### Step 2: Create FileList.csv

**Option A: Inside Docker (Recommended)**

```bash
# Start interactive shell
docker-compose run --rm ucvme bash

# Inside container, create FileList.csv
python scripts/create_so2sat_filelist.py \
    --data_dir /workspace/data/So2Sat_POP \
    --output /workspace/data/So2Sat_POP/FileList.csv

# Exit container
exit
```

**Option B: On Host Machine**

```bash
python scripts/create_so2sat_filelist.py \
    --data_dir /work/ammar/sslrp/data/So2Sat_POP \
    --output /work/ammar/sslrp/data/So2Sat_POP/FileList.csv
```

### Step 3: Calculate Target Normalization (Optional)

```bash
# Inside Docker container
docker-compose run --rm ucvme python -c "
import pandas as pd
df = pd.read_csv('/workspace/data/So2Sat_POP/FileList.csv')
train = df[df['SPLIT'] == 'TRAIN']
print(f'y_mean: {train[\"POP\"].mean():.2f}')
print(f'y_std: {train[\"POP\"].std():.2f}')
"
```

Update `y_mean` and `y_std` in your config file.

### Step 4: Run Training

**Using ResNet50:**

```bash
docker-compose run --rm ucvme python3 ucvme_age.py \
    --config=/workspace/configs/so2sat_pop_resnet50.yaml \
    --output=/workspace/output/so2sat_pop_resnet50
```

**Using EfficientNetB0:**

```bash
docker-compose run --rm ucvme python3 ucvme_age.py \
    --config=/workspace/configs/so2sat_pop_efficientnetb0.yaml \
    --output=/workspace/output/so2sat_pop_effnetb0
```

**With Custom Parameters:**

```bash
docker-compose run --rm ucvme python3 ucvme_age.py \
    --config=/workspace/configs/so2sat_pop_resnet50.yaml \
    --output=/workspace/output/so2sat_pop_custom \
    --batch_size=64 \
    --num_epochs=50
```

### Step 5: Run Testing

```bash
docker-compose run --rm ucvme python3 ucvme_age.py \
    --config=/workspace/configs/so2sat_pop_resnet50.yaml \
    --output=/workspace/output/so2sat_pop_resnet50 \
    --weights=/workspace/output/so2sat_pop_resnet50/best.pt \
    --test_only
```

## Using the Example Script

A convenience script is provided:

```bash
./docker-so2sat-example.sh
```

This script will:
1. Check if FileList.csv exists, create it if missing
2. Run training with ResNet50

## Interactive Shell

For debugging or manual operations:

```bash
docker-compose run --rm ucvme bash
```

Inside the container:
- Data: `/workspace/data/So2Sat_POP`
- Configs: `/workspace/configs`
- Scripts: `/workspace/scripts`
- Output: `/workspace/output`

## GPU Support

To enable GPU support, edit `docker-compose.yml` and uncomment:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Then rebuild:
```bash
docker-compose build
```

## Troubleshooting

### FileList.csv Not Found

Make sure you've created FileList.csv in the data directory:
```bash
docker-compose run --rm ucvme ls -la /workspace/data/So2Sat_POP/FileList.csv
```

### Permission Issues

If you get permission errors:
```bash
# On host
sudo chown -R $USER:$USER /work/ammar/sslrp/data/So2Sat_POP
```

### Data Directory Not Mounted

Check `docker-compose.yml` volume mounts. The So2Sat_POP directory should be mounted at `/workspace/data/So2Sat_POP`.

### Update Data Path

If your data is in a different location, update `docker-compose.yml`:

```yaml
volumes:
  - /your/path/to/So2Sat_POP:/workspace/data/So2Sat_POP
```

And update the config file `data_dir` accordingly.

## Complete Example Workflow

```bash
# 1. Build (first time only)
docker-compose build

# 2. Create FileList.csv
docker-compose run --rm ucvme python scripts/create_so2sat_filelist.py \
    --data_dir /workspace/data/So2Sat_POP \
    --output /workspace/data/So2Sat_POP/FileList.csv

# 3. Calculate normalization values
docker-compose run --rm ucvme python -c "
import pandas as pd
df = pd.read_csv('/workspace/data/So2Sat_POP/FileList.csv')
train = df[df['SPLIT'] == 'TRAIN']
print(f'y_mean: {train[\"POP\"].mean():.2f}, y_std: {train[\"POP\"].std():.2f}')
"

# 4. Update config with calculated values, then train
docker-compose run --rm ucvme python3 ucvme_age.py \
    --config=/workspace/configs/so2sat_pop_resnet50.yaml \
    --output=/workspace/output/so2sat_pop_run1

# 5. Test
docker-compose run --rm ucvme python3 ucvme_age.py \
    --config=/workspace/configs/so2sat_pop_resnet50.yaml \
    --output=/workspace/output/so2sat_pop_run1 \
    --weights=/workspace/output/so2sat_pop_run1/best.pt \
    --test_only
```

## Notes

- All paths in config files use Docker paths (`/workspace/...`)
- Output will be saved to `./output/` on your host machine
- The container automatically detects GPU if available
- For CPU-only, the code will automatically fall back

