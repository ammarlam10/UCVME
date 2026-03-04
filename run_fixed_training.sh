#!/bin/bash

# Run So2Sat_POP training with all fixes applied
# Date: February 15, 2026
# 
# Fixes applied:
# 1. Removed double target normalization
# 2. Corrected y_mean and y_std (calculated from filtered data)
# 3. Confirmed correct band extraction (RGB: indices 3,2,1)
# 4. Removed double image normalization (kept only clipping + min-max)
# 5. Fixed GPU configuration (removed CUDA_VISIBLE_DEVICES)
# 6. Reduced batch size from 256 to 128

docker run --rm \
    --gpus device=5 \
    --shm-size=32g \
    -v /work/ammar/sslrp/data/So2Sat_POP:/workspace/data/So2Sat_POP \
    -v /work/ammar/sslrp/UCVME:/workspace \
    -v /work/ammar/sslrp/UCVME/output:/workspace/output \
    ucvme:latest \
    python3 /workspace/ucvme_age.py \
        --config=/workspace/configs/so2sat_pop_efficientnetb0_20percent_fixed.yaml \
        --output=/workspace/output/so2sat_pop_efficientnetb0_20percent_fixed
