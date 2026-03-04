#!/bin/bash
# Run So2Sat POP Custom training with Docker
# This script runs training with the custom configuration:
# - sen2spring RGB bands (2-4)
# - 90/10 train/val split
# - 80% reduction of labels where POP <= 100
# - Custom normalization

set -e

echo "=== So2Sat POP Custom Training ==="
echo ""

# Configuration
DATA_DIR="/workspace/data/So2Sat_POP"
OUTPUT_DIR="/workspace/output/so2sat_pop_custom"
CONFIG="/workspace/configs/so2sat_pop_custom.yaml"

echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "Data: ${DATA_DIR}"
echo ""

# Run training with Docker
docker run --rm \
    --gpus device=5 \
    -v /work/ammar/sslrp/data/So2Sat_POP:/workspace/data/So2Sat_POP \
    -v /work/ammar/sslrp/UCVME:/workspace \
    -v /work/ammar/sslrp/UCVME/output:/workspace/output \
    -e CUDA_VISIBLE_DEVICES=5 \
    ucvme:latest \
    python3 /workspace/ucvme_age.py \
        --config="${CONFIG}" \
        --output="${OUTPUT_DIR}"

echo ""
echo "=== Training Complete ==="
echo "Results saved to: ./output/so2sat_pop_custom"
