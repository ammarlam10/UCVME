# Configuration and Model Usage Guide

This guide explains how to use the config-based system and switch between ResNet50 and EfficientNetB0 models.

## Quick Start

### Using Config Files (Recommended)

**Run with ResNet50:**
```bash
python3 ucvme_age.py --config=configs/resnet50.yaml --output=./output/resnet50_run
```

**Run with EfficientNetB0:**
```bash
python3 ucvme_age.py --config=configs/efficientnetb0.yaml --output=./output/effnetb0_run
```

### Using Command-Line Arguments (Backward Compatible)

**Run with ResNet50 (default):**
```bash
python3 ucvme_age.py --output=./output/resnet50_run --model=resnet50
```

**Run with EfficientNetB0:**
```bash
python3 ucvme_age.py --output=./output/effnetb0_run --model=efficientnetb0
```

## Configuration Files

Configuration files are located in the `configs/` directory and use YAML format.

### Available Config Files

- `configs/resnet50.yaml` - Full ResNet50 configuration
- `configs/efficientnetb0.yaml` - Full EfficientNetB0 configuration
- `configs/resnet50_small.yaml` - Quick test config for ResNet50
- `configs/efficientnetb0_small.yaml` - Quick test config for EfficientNetB0
- `configs/resnet50_10percent.yaml` - ResNet50 with 10% labeled (using percentages)
- `configs/efficientnetb0_10percent.yaml` - EfficientNetB0 with 10% labeled (using percentages)

### Config File Structure

```yaml
model:
  name: "resnet50"  # or "efficientnetb0"
  pretrained: true
  drp_p: 0.05

data:
  data_dir: "DATA_DIR"
  reduced_set: true
  rd_label: 1000
  rd_unlabel: 9518
  # OR use percentages (takes precedence over absolute numbers):
  # label_percentage: 0.10    # 10% labeled
  # unlabel_percentage: 0.90  # 90% unlabeled
  pad_param: 5

training:
  num_epochs: 30
  lr: 0.0001
  weight_decay: 1e-3
  lr_step_period: 10
  batch_size: 32
  num_workers: 4
  seed: 0

ssl:
  ssl_mult: -1
  w_ulb: 10
  samp_fq: 5
  samp_ssl: 5

target:
  y_mean: 35
  y_std: 11

misc:
  device: null  # null = auto-detect, "cuda", or "cpu"
  run_test: true
  test_only: false
```

## Usage Examples

### 1. Training with Config File

```bash
# ResNet50 training
python3 ucvme_age.py \
    --config=configs/resnet50.yaml \
    --output=./output/resnet50_experiment

# EfficientNetB0 training
python3 ucvme_age.py \
    --config=configs/efficientnetb0.yaml \
    --output=./output/effnetb0_experiment
```

### 2. Override Config Values with CLI Args

```bash
# Use config but override output directory and batch size
python3 ucvme_age.py \
    --config=configs/resnet50.yaml \
    --output=./output/custom_run \
    --batch_size=64
```

### 3. Testing Only

```bash
# Test with ResNet50
python3 ucvme_age.py \
    --config=configs/resnet50.yaml \
    --output=./output/resnet50_test \
    --weights=./output/resnet50_test/best.pt \
    --test_only

# Test with EfficientNetB0
python3 ucvme_age.py \
    --config=configs/efficientnetb0.yaml \
    --output=./output/effnetb0_test \
    --weights=./output/effnetb0_test/best.pt \
    --test_only
```

### 4. Quick Test (Small Dataset)

```bash
# Quick test with ResNet50
python3 ucvme_age.py \
    --config=configs/resnet50_small.yaml \
    --output=./output/resnet50_quick_test

# Quick test with EfficientNetB0
python3 ucvme_age.py \
    --config=configs/efficientnetb0_small.yaml \
    --output=./output/effnetb0_quick_test
```

### 5. Using Percentage-Based Labeled/Unlabeled Split

You can specify the labeled/unlabeled split using percentages instead of absolute numbers:

```yaml
data:
  data_dir: "DATA_DIR"
  reduced_set: true
  # Option 1: Use percentages (recommended for experimentation)
  label_percentage: 0.10    # 10% labeled
  unlabel_percentage: 0.90  # 90% unlabeled
  # Option 2: Use absolute numbers (original method)
  # rd_label: 1000
  # rd_unlabel: 9518
  pad_param: 5
```

**Examples:**

```bash
# Use 10% labeled, 90% unlabeled (from config)
python3 ucvme_age.py \
    --config=configs/resnet50_10percent.yaml \
    --output=./output/resnet50_10pct

# Use 5% labeled, 95% unlabeled (custom config)
# Create config with: label_percentage: 0.05, unlabel_percentage: 0.95
python3 ucvme_age.py \
    --config=configs/my_5percent_config.yaml \
    --output=./output/resnet50_5pct
```

**Notes:**
- Percentages take precedence over absolute numbers (`rd_label`/`rd_unlabel`)
- If only one percentage is provided, the other is calculated automatically
- Percentages are calculated from total TRAIN samples in FileList.csv
- The calculated values are printed during execution for verification

## Docker Usage

### Building the Docker Image

```bash
# Build using docker-compose
docker-compose build

# Or build directly
docker build -t ucvme:latest .
```

### Running with Docker

**Using Config Files:**

```bash
# Start container
docker-compose up -d

# Run with ResNet50 config
docker-compose exec ucvme python3 ucvme_age.py \
    --config=/workspace/configs/resnet50.yaml \
    --output=/workspace/output/resnet50_run

# Run with EfficientNetB0 config
docker-compose exec ucvme python3 ucvme_age.py \
    --config=/workspace/configs/efficientnetb0.yaml \
    --output=/workspace/output/effnetb0_run
```

**Using Command-Line Arguments:**

```bash
# Run with ResNet50
docker-compose exec ucvme python3 ucvme_age.py \
    --output=/workspace/output/resnet50_run \
    --model=resnet50 \
    --data_dir=/workspace/DATA_DIR

# Run with EfficientNetB0
docker-compose exec ucvme python3 ucvme_age.py \
    --output=/workspace/output/effnetb0_run \
    --model=efficientnetb0 \
    --data_dir=/workspace/DATA_DIR
```

### Direct Docker Run

```bash
# With GPU support
docker run -it --rm \
    --gpus all \
    -v $(pwd)/DATA_DIR:/workspace/DATA_DIR \
    -v $(pwd)/output:/workspace/output \
    -v $(pwd)/configs:/workspace/configs \
    ucvme:latest \
    python3 ucvme_age.py \
    --config=/workspace/configs/efficientnetb0.yaml \
    --output=/workspace/output/effnetb0_run
```

## Creating Custom Config Files

1. Copy an existing config file:
```bash
cp configs/resnet50.yaml configs/my_custom_config.yaml
```

2. Edit the values as needed

3. Run with your custom config:
```bash
python3 ucvme_age.py --config=configs/my_custom_config.yaml --output=./output/my_experiment
```

## Model Comparison

| Feature | ResNet50 | EfficientNetB0 |
|---------|----------|----------------|
| Backbone | Custom ResNet50 | timm EfficientNetB0 |
| Feature Dim | 2048 | 1280 |
| Pretrained | ImageNet | ImageNet |
| Dropout | After each layer | After feature extraction |
| MC Dropout | Yes | Yes |

## Notes

- Config files use YAML format - be careful with indentation
- CLI arguments always override config file values
- The `--output` argument is required (either in config or CLI)
- Model selection can be done via `--model` CLI arg or `model.name` in config
- All existing CLI arguments remain functional for backward compatibility

## Troubleshooting

**Issue: "Unknown model" error**
- Solution: Make sure model name is exactly "resnet50" or "efficientnetb0" (case-insensitive)

**Issue: Config file not found**
- Solution: Use absolute path or relative path from project root

**Issue: timm not found**
- Solution: Install dependencies: `pip install -r requirements_pip.txt`

**Issue: YAML parsing error**
- Solution: Check YAML syntax and indentation (must use spaces, not tabs)



