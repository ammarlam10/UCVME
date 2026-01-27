# So2Sat_POP Dataset Usage Guide

This guide explains how to use the UCVME codebase with the So2Sat_POP dataset.

## Dataset Structure

The So2Sat_POP dataset should be organized as follows:

```
/work/ammar/sslrp/data/So2Sat_POP/
├── So2Sat_POP_Part1/
│   ├── train/
│   │   ├── city_name/
│   │   │   ├── city_name.csv  (contains GRD_ID, Class, POP)
│   │   │   └── ...
│   │   └── ...
│   └── test/
│       └── ...
└── So2Sat_POP_Part2/
    ├── train/
    │   ├── city_name/
    │   │   └── dem/
    │   │       └── Class_X/
    │   │           └── GRD_ID_dem.tif
    │   └── ...
    └── test/
        └── ...
```

## Step 1: Create FileList.csv

Before training, you need to create a `FileList.csv` file that maps images to their population labels.

### Option A: Use the Helper Script (Recommended)

```bash
python scripts/create_so2sat_filelist.py \
    --data_dir /work/ammar/sslrp/data/So2Sat_POP \
    --output /work/ammar/sslrp/data/So2Sat_POP/FileList.csv
```

This script will:
- Scan both Part1 (labels) and Part2 (images)
- Match GRD_IDs to image files
- Create a FileList.csv with columns: `FileName`, `SPLIT`, `POP`, `GRD_ID`, `Class`, `City`

### Option B: Create Manually

Create a CSV file with the following columns:
- `FileName`: Relative path to image from data_dir (e.g., `So2Sat_POP_Part2/train/city/dem/Class_7/1kmN2988E4238_dem.tif`)
- `SPLIT`: `TRAIN` or `TEST`
- `POP`: Population value (target variable)
- `SSL_SPLIT`: (Optional) `LABELED` or `UNLABELED` for SSL - will be auto-generated if using `reduced_set`

Example:
```csv
FileName,SPLIT,POP,GRD_ID,Class
So2Sat_POP_Part2/train/city1/dem/Class_7/1kmN2988E4238_dem.tif,TRAIN,869,1kmN2978E4212,10
So2Sat_POP_Part2/train/city1/dem/Class_7/1kmN2979E4212_dem.tif,TRAIN,3783,1kmN2979E4212,12
...
```

## Step 2: Calculate Target Normalization

The config files include example `y_mean` and `y_std` values. You should calculate these from your training data:

```python
import pandas as pd
data = pd.read_csv("/work/ammar/sslrp/data/So2Sat_POP/FileList.csv")
train_data = data[data['SPLIT'] == 'TRAIN']
y_mean = train_data['POP'].mean()
y_std = train_data['POP'].std()
print(f"y_mean: {y_mean:.2f}, y_std: {y_std:.2f}")
```

Update these values in your config file under `target:` section.

## Step 3: Run Training

### Using ResNet50:

```bash
python3 ucvme_age.py \
    --config=configs/so2sat_pop_resnet50.yaml \
    --output=./output/so2sat_pop_resnet50
```

### Using EfficientNetB0:

```bash
python3 ucvme_age.py \
    --config=configs/so2sat_pop_efficientnetb0.yaml \
    --output=./output/so2sat_pop_effnetb0
```

### Override Config Values:

```bash
python3 ucvme_age.py \
    --config=configs/so2sat_pop_resnet50.yaml \
    --output=./output/so2sat_pop_custom \
    --batch_size=64 \
    --num_epochs=50
```

## Step 4: Testing

```bash
python3 ucvme_age.py \
    --config=configs/so2sat_pop_resnet50.yaml \
    --output=./output/so2sat_pop_resnet50 \
    --weights=./output/so2sat_pop_resnet50/best.pt \
    --test_only
```

## Configuration Options

Key configuration parameters in the YAML files:

```yaml
data:
  dataset_name: "so2sat_pop"  # Must be "so2sat_pop"
  data_dir: "/work/ammar/sslrp/data/So2Sat_POP"
  target_column: "POP"  # Target variable name
  image_dir: "So2Sat_POP_Part2"  # Directory containing images
  file_list_name: "FileList.csv"
  rd_label: 1000  # Number of labeled samples
  rd_unlabel: 5000  # Number of unlabeled samples

target:
  y_mean: 2000.0  # Calculate from your data
  y_std: 3000.0   # Calculate from your data
```

## Notes

1. **Image Format**: The dataset loader supports `.tif` files (satellite imagery). If images are single-channel, they will be converted to 3-channel RGB.

2. **File Paths**: The `FileName` column in FileList.csv should contain relative paths from `data_dir`.

3. **SSL Splits**: If using `reduced_set: true`, the script will automatically generate `FileList_ssl_{rd_label}_{rd_unlabel}.csv` with `SSL_SPLIT` column.

4. **Population Range**: Population values can vary widely. Make sure to set appropriate `y_mean` and `y_std` for normalization.

## Troubleshooting

### "FileList not found"
- Make sure you've created FileList.csv in the data_dir
- Check the path in your config file

### "Missing files" error
- Verify that image paths in FileList.csv are correct
- Check that Part2 directory structure matches the FileName paths

### "Unknown dataset" error
- Make sure `dataset_name: "so2sat_pop"` in your config
- Check that the dataset is registered in `datasets/__init__.py`

## Example: Complete Workflow

```bash
# 1. Create FileList.csv
python scripts/create_so2sat_filelist.py \
    --data_dir /work/ammar/sslrp/data/So2Sat_POP \
    --output /work/ammar/sslrp/data/So2Sat_POP/FileList.csv

# 2. Calculate normalization (optional, update config manually)
python -c "import pandas as pd; df=pd.read_csv('/work/ammar/sslrp/data/So2Sat_POP/FileList.csv'); train=df[df['SPLIT']=='TRAIN']; print(f'y_mean: {train[\"POP\"].mean():.2f}, y_std: {train[\"POP\"].std():.2f}')"

# 3. Update config with calculated values, then train
python3 ucvme_age.py \
    --config=configs/so2sat_pop_resnet50.yaml \
    --output=./output/so2sat_pop_run1
```

