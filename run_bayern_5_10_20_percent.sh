#!/bin/bash
#
# Run Bayern Forest Height UNET SSL experiments: 5%, 10%, and 20% labeled.
# Output: log.csv, checkpoint.pt, best.pt (no prediction CSVs).
#
# Configs:
#   - 5%:  150 epochs, batch 8,  ~15-25h
#   - 10%: 60 epochs,  batch 8,  ~6-10h
#   - 20%: 60 epochs,  batch 16, ~5-8h
#
# Usage:
#   ./run_bayern_5_10_20_percent.sh              # run all three (sequential)
#   GPU_ID=6 ./run_bayern_5_10_20_percent.sh     # use GPU 6
#

set -e
GPU_ID="${GPU_ID:-0}"
DATA_DIR="${DATA_DIR:-/work/ammar/sslrp/data}"
WORKSPACE="${WORKSPACE:-/work/ammar/sslrp/UCVME}"
OUTPUT_DIR="${OUTPUT_DIR:-/work/ammar/sslrp/UCVME/output}"

RUN_ONE() {
  local config="$1"
  local name="$2"
  echo ""
  echo "=============================================="
  echo "[$(date -Iseconds)] Starting: $name"
  echo "  config: $config"
  echo "  output: output/$(basename "$config" .yaml)"
  echo "=============================================="
  docker run --rm \
    --gpus "device=${GPU_ID}" \
    --shm-size=16g \
    -v "${DATA_DIR}:/workspace/data" \
    -v "${WORKSPACE}:/workspace/ucvme" \
    -v "${OUTPUT_DIR}:/workspace/output" \
    ucvme:latest \
    bash -c "cd /workspace/ucvme && pip install -q h5py && python3 ucvme_age.py --config $config"
  echo "[$(date -Iseconds)] Finished: $name"
}

cd "$WORKSPACE"

# 5% labeled (577 labeled, 10,975 unlabeled, 150 epochs)
RUN_ONE "configs/bayern_forest_height_unet_5percent.yaml" "Bayern 5%"

# 10% labeled (1,155 labeled, 10,397 unlabeled, 60 epochs)
RUN_ONE "configs/bayern_forest_height_unet_10percent.yaml" "Bayern 10%"

# 20% labeled (2,310 labeled, 9,242 unlabeled, 60 epochs)
RUN_ONE "configs/bayern_forest_height_unet_20percent.yaml" "Bayern 20%"

echo ""
echo "=============================================="
echo "All three experiments finished."
echo "  output/bayern_forest_height_unet_5percent/"
echo "  output/bayern_forest_height_unet_10percent/"
echo "  output/bayern_forest_height_unet_20percent/"
echo "=============================================="
