#!/bin/bash
# Helper script to run training in Docker container

set -e

# Default values
DATASET=${1:-"so2sat_pop"}
DATA_ROOT=${2:-"/work/ammar/sslrp/data"}
OUTPUT_DIR=${3:-"./outputs/docker_experiment"}
RD_LABEL=${4:-1000}
RD_UNLABEL=${5:-5000}

echo "=========================================="
echo "UCVME Docker Training"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Data root: $DATA_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Labeled samples: $RD_LABEL"
echo "Unlabeled samples: $RD_UNLABEL"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training in Docker
docker-compose run --rm ucvme python3 ucvme.py \
    --dataset "$DATASET" \
    --data_root /data \
    --output /workspace/outputs/$(basename "$OUTPUT_DIR") \
    --rd_label "$RD_LABEL" \
    --rd_unlabel "$RD_UNLABEL" \
    --num_epochs 30 \
    --batch_size 32 \
    --num_workers 4

echo "Training completed! Check outputs in: $OUTPUT_DIR"

