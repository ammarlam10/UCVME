#!/bin/bash
# Example script for running So2Sat_POP with Docker

set -e

echo "=== So2Sat_POP Docker Training Example ==="
echo ""

# Configuration
DATA_DIR="/workspace/data/So2Sat_POP"
OUTPUT_DIR="/workspace/output/so2sat_pop_resnet50"
CONFIG="/workspace/configs/so2sat_pop_resnet50.yaml"

# Step 1: Check if FileList.csv exists
echo "Step 1: Checking for FileList.csv..."
if docker-compose run --rm ucvme test -f "${DATA_DIR}/FileList.csv"; then
    echo "✓ FileList.csv found"
else
    echo "✗ FileList.csv not found. Creating it..."
    docker-compose run --rm ucvme python scripts/create_so2sat_filelist.py \
        --data_dir "${DATA_DIR}" \
        --output "${DATA_DIR}/FileList.csv"
    echo "✓ FileList.csv created"
fi

echo ""
echo "Step 2: Starting training..."
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Step 2: Run training
docker-compose run --rm ucvme python3 ucvme_age.py \
    --config="${CONFIG}" \
    --output="${OUTPUT_DIR}"

echo ""
echo "=== Training Complete ==="
echo "Results saved to: ${OUTPUT_DIR}"

