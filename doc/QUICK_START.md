# Quick Start Guide: EfficientNetB0 and Config Support

## ✅ What's Been Implemented

1. **EfficientNetB0 Model** (`models/efficientnetb0.py`)
   - Uses timm EfficientNetB0 as backbone
   - Implements MC Dropout for uncertainty estimation
   - Drop-in replacement for ResNet50

2. **Config File Support**
   - YAML-based configuration files
   - CLI arguments can override config values
   - Backward compatible with existing CLI usage

3. **Docker Compatibility**
   - All dependencies included in requirements
   - Configs folder mounted in docker-compose
   - Ready to use in containerized environment

## 🚀 Quick Start

### Option 1: Using Config Files (Recommended)

```bash
# ResNet50
python3 ucvme_age.py --config=configs/resnet50.yaml --output=./output/resnet50

# EfficientNetB0
python3 ucvme_age.py --config=configs/efficientnetb0.yaml --output=./output/effnetb0
```

### Option 2: Using CLI Arguments

```bash
# ResNet50 (default)
python3 ucvme_age.py --output=./output/resnet50 --model=resnet50

# EfficientNetB0
python3 ucvme_age.py --output=./output/effnetb0 --model=efficientnetb0
```

## 🐳 Docker Usage

### Build Docker Image

```bash
docker-compose build
```

### Run with Config Files

```bash
# Start container
docker-compose up -d

# Run ResNet50
docker-compose exec ucvme python3 ucvme_age.py \
    --config=/workspace/configs/resnet50.yaml \
    --output=/workspace/output/resnet50_run

# Run EfficientNetB0
docker-compose exec ucvme python3 ucvme_age.py \
    --config=/workspace/configs/efficientnetb0.yaml \
    --output=/workspace/output/effnetb0_run
```

### Run with CLI Arguments

```bash
docker-compose exec ucvme python3 ucvme_age.py \
    --output=/workspace/output/effnetb0_run \
    --model=efficientnetb0 \
    --data_dir=/workspace/DATA_DIR \
    --pretrained \
    --num_epochs=30
```

## 📋 Available Config Files

- `configs/resnet50.yaml` - Full ResNet50 config
- `configs/efficientnetb0.yaml` - Full EfficientNetB0 config
- `configs/resnet50_small.yaml` - Quick test (ResNet50)
- `configs/efficientnetb0_small.yaml` - Quick test (EfficientNetB0)

## 🔧 Installation

### Local Installation

```bash
pip install -r requirements_pip.txt
```

This will install:
- `timm>=0.6.0` (for EfficientNetB0)
- `pyyaml>=5.4.0` (for config support)
- All other existing dependencies

### Docker Installation

Dependencies are automatically installed when building the Docker image:

```bash
docker-compose build
```

## 📝 Example Commands

### Training

```bash
# ResNet50 with config
python3 ucvme_age.py \
    --config=configs/resnet50.yaml \
    --output=./output/resnet50_exp

# EfficientNetB0 with config
python3 ucvme_age.py \
    --config=configs/efficientnetb0.yaml \
    --output=./output/effnetb0_exp

# Override config values
python3 ucvme_age.py \
    --config=configs/resnet50.yaml \
    --output=./output/custom \
    --batch_size=64 \
    --num_epochs=50
```

### Testing

```bash
# Test ResNet50
python3 ucvme_age.py \
    --config=configs/resnet50.yaml \
    --output=./output/resnet50_test \
    --weights=./output/resnet50_test/best.pt \
    --test_only

# Test EfficientNetB0
python3 ucvme_age.py \
    --config=configs/efficientnetb0.yaml \
    --output=./output/effnetb0_test \
    --weights=./output/effnetb0_test/best.pt \
    --test_only
```

## 🔍 Verification

### Check Model Import

```python
import models
print(dir(models))  # Should show resnet50_unc and efficientnetb0_unc
```

### Check Config Loading

```python
import yaml
with open('configs/resnet50.yaml') as f:
    config = yaml.safe_load(f)
    print(config['model']['name'])  # Should print 'resnet50'
```

## 📚 More Information

See `CONFIG_USAGE.md` for detailed documentation on:
- Config file structure
- Advanced usage examples
- Troubleshooting
- Model comparison

## ⚠️ Important Notes

1. **Output Directory**: `--output` is required (either in config or CLI)
2. **Model Selection**: Use `--model` CLI arg or `model.name` in config
3. **Backward Compatibility**: All existing CLI arguments still work
4. **Docker**: Configs folder is automatically mounted in docker-compose
5. **Dependencies**: Make sure `timm` and `pyyaml` are installed

## 🎯 Next Steps

1. Choose your model (ResNet50 or EfficientNetB0)
2. Select or create a config file
3. Run training/test with the commands above
4. Check output directory for results

Happy training! 🚀




