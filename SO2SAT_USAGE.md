# So2Sat_POP Dataset Usage Guide

This guide explains how to use the So2Sat_POP dataset with the UCVME framework.

## Dataset Structure

The So2Sat_POP dataset is located at `/work/ammar/sslrp/data/So2Sat_POP` and has the following structure:

```
So2Sat_POP/
├── So2Sat_POP_Part1/
│   ├── train/
│   │   ├── {city_id}_{pop}_{cityname}/
│   │   │   ├── {city_id}_{pop}_{cityname}.csv  # Contains GRD_ID, Class, POP
│   │   │   ├── sen2spring/Class_X/{GRD_ID}_sen2spring.tif
│   │   │   ├── sen2summer/Class_X/{GRD_ID}_sen2summer.tif
│   │   │   └── ...
│   └── test/
│       └── (similar structure)
└── So2Sat_POP_Part2/
    └── (similar structure)
```

## Quick Start

### 1. Generate FileList.csv (Optional but Recommended)

The FileList.csv will be auto-generated on first run, but you can pre-generate it:

```bash
python scripts/generate_filelist_so2sat.py \
    --data_root /work/ammar/sslrp/data \
    --season spring
```

This will create `FileList.csv` in the So2Sat_POP directory.

### 2. Run Training

```bash
python ucvme.py \
    --dataset so2sat_pop \
    --data_root /work/ammar/sslrp/data \
    --output ./outputs/so2sat_experiment \
    --rd_label 1000 \
    --rd_unlabel 5000 \
    --num_epochs 30 \
    --batch_size 32
```

### 3. Test Only

```bash
python ucvme.py \
    --dataset so2sat_pop \
    --data_root /work/ammar/sslrp/data \
    --output ./outputs/so2sat_experiment \
    --test_only \
    --weights ./outputs/so2sat_experiment/best.pt
```

## Configuration

The dataset configuration is in `configs/datasets/so2sat_pop.yaml`. Key settings:

- **season**: Which Sentinel-2 season to use (spring, summer, autumn, winter)
- **use_rgb_only**: If True, uses only first 3 channels (for ResNet compatibility)
- **parts**: Which parts to include (Part1, Part2, or both)

## Dataset Details

- **Target**: Population count (integer)
- **Image Format**: TIFF, 100x100 pixels
- **Channels**: 10 (Sentinel-2 bands), can be reduced to 3 for RGB
- **Splits**: train/test (validation split can be created from train)

## Notes

1. The dataset will automatically generate FileList.csv on first run if it doesn't exist
2. SSL splits are generated automatically based on `--rd_label` and `--rd_unlabel` parameters
3. Target normalization (mean/std) is computed automatically from the data
4. Image normalization (mean/std) is computed from a sample of training images

## Troubleshooting

**Issue**: "FileList.csv not found"
- **Solution**: The file will be auto-generated on first run, or run `scripts/generate_filelist_so2sat.py`

**Issue**: "Image not found" errors
- **Solution**: Check that the season specified in config matches available seasons in the data directory

**Issue**: "Config file not found"
- **Solution**: Ensure `configs/datasets/so2sat_pop.yaml` exists

