# Preloading Implementation for So2Sat POP Dataset

## Summary

Implemented in-memory preloading for the So2Sat POP dataset to avoid repeated disk I/O and image processing across 100 epochs.

## Changes Made

### 1. Dataset Class (`datasets/so2sat_pop_custom.py`)

- Added `preload` parameter (default: `False`) to `So2SatDatasetCustom.__init__`
- Added `preloaded_images` list to store preprocessed images in RAM
- Refactored image loading into `_load_and_process_image()` method
- At init, if `preload=True`:
  - Loads all images from disk
  - Applies all preprocessing (TIFF decode, RGB band extraction, clip, normalize, resize to 224×224)
  - Stores processed images in `self.preloaded_images` list
  - Shows progress bar via tqdm
- In `__getitem__`:
  - If preloaded: returns copy from `preloaded_images[index]`
  - Otherwise: loads from disk as before
  - Augmentation (flip, pad/crop) still applied on-the-fly

### 2. Training Script (`ucvme_age.py`)

- Reads `preload` flag from config: `cfg['data']['preload']`
- If enabled, passes `preload=True` to all dataset constructors
- Prints confirmation message at startup

### 3. Config (`configs/so2sat_pop_efficientnetb0_5percent_fixed.yaml`)

- Added `preload: true` to `data:` section
- Updated `num_epochs: 100` (was 80)
- Comment added: "Preload all images into RAM at startup (recommended for 100 epochs)"

### 4. Run Script (`run_so2sat_5pct_4repeats.sh`)

- Updated to install `tqdm` (for preload progress bar) alongside `h5py`
- Fixed loop to run 4 reps (2, 3, 4, 5) as originally intended
- Updated time estimates for 100 epochs with preload

## Memory Usage

For the 5% config:
- **Labeled:** ~6,094 samples (after 2× duplication)
- **Unlabeled:** ~57,905 samples
- **Val:** ~11,980 samples
- **Total:** ~76k samples × 3 × 224 × 224 × 4 bytes ≈ **43 GB** (float32)

System has **359 GiB available**, so preloading is well within limits.

## Expected Performance

### Without preload (30 epochs)
- ~73-74 s/epoch
- Total: ~1h 4m for 30 epochs

### With preload (100 epochs, estimated)
- One-time preload: ~15-25 min
- Epoch time: ~60-70 s/epoch (may be faster, no disk I/O)
- Total per run: ~2.5-3 hours
- 4 runs: ~10-12 hours total

## Benefits

1. **No repeated I/O:** Each image is read from disk only once (at startup)
2. **No repeated preprocessing:** TIFF decode, band extraction, resize, normalize done once
3. **Faster epochs:** GPU waits less for data (if I/O was a bottleneck)
4. **More stable timing:** No variance from disk/filesystem performance
5. **Amortized over 100 epochs:** One-time load cost is negligible vs. 100× disk reads

## Usage

The preload feature is controlled by the config file:

```yaml
data:
  preload: true  # Enable preloading
```

Or disable by setting to `false` or removing the line (defaults to `false`).

## Running the Experiment

```bash
cd /work/ammar/sslrp/UCVME
./run_so2sat_5pct_4repeats.sh
```

This will:
1. Install h5py and tqdm in the Docker container
2. Run 4 repetitions (rep2, rep3, rep4, rep5) sequentially on GPU 5
3. Each run: preload data → train 100 epochs → test
4. Output dirs: `output/so2sat_pop_efficientnetb0_5percent_fixed_rep{2,3,4,5}/`

## Monitoring

Watch progress:
```bash
# Overall script progress
tail -f /home/ammar/.cursor/projects/work-ammar-sslrp-UCVME/terminals/*.txt

# Individual run logs
tail -f output/so2sat_pop_efficientnetb0_5percent_fixed_rep2/log.csv
tail -f output/so2sat_pop_efficientnetb0_5percent_fixed_rep3/log.csv
# etc.
```
