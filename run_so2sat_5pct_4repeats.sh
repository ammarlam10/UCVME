#!/bin/bash
#
# Run so2sat_pop_efficientnetb0_5percent_fixed 4 additional times (repetitions 2–5).
# Config and hyperparameters are unchanged; each run writes to a separate output dir.
# Uses GPU 5 only.
#
# Previous run duration (30 epochs, from log):
#   Start: 2026-02-17 08:47:49 (Run timestamp: 20260217_084749)
#   End:   2026-02-17 09:52:13 (test/val one-clip logging)
#   Duration: ~1 hour 4 minutes per run (30 epochs)
#
# Current run (100 epochs with preloading):
#   Preload time: ~15-25 min (one-time per run)
#   Epoch time: ~60-70 s/epoch (estimated, may be faster with preload)
#   Total per run: ~2.5-3 hours
#   → 4 runs ≈ 10-12 hours total (sequential)
#

set -e
CONFIG="/workspace/configs/so2sat_pop_efficientnetb0_5percent_fixed.yaml"
BASE_OUTPUT="/workspace/output/so2sat_pop_efficientnetb0_5percent_fixed"

# Optional: run from project root on host (adjust paths if not using Docker)
# If you use Docker, leave these as /workspace; otherwise set DATA_DIR and WORKSPACE
DATA_DIR="${DATA_DIR:-/work/ammar/sslrp/data/So2Sat_POP}"
WORKSPACE="${WORKSPACE:-/work/ammar/sslrp/UCVME}"
OUTPUT_HOST="${OUTPUT_HOST:-/work/ammar/sslrp/UCVME/output}"

echo "=============================================="
echo "So2Sat POP 5% fixed – 4 repeats (GPU 5)"
echo "=============================================="
echo "Config: $CONFIG (100 epochs, preload enabled)"
echo "Base output: $BASE_OUTPUT"
echo "Estimated time: ~2.5-3h per run, ~10-12h total"
echo "=============================================="

for rep in 2 3 4 5; do
  OUT="${BASE_OUTPUT}_rep${rep}"
  echo ""
  echo "[$(date -Iseconds)] Starting repetition $rep → $OUT"
  docker run --rm \
    --gpus device=5 \
    --shm-size=32g \
    -v "${DATA_DIR}:/workspace/data/So2Sat_POP" \
    -v "${WORKSPACE}:/workspace" \
    -v "${OUTPUT_HOST}:/workspace/output" \
    ucvme:latest \
    bash -c "pip install --quiet h5py tqdm && python3 /workspace/ucvme_age.py \
      --config=$CONFIG \
      --output=$OUT"
  echo "[$(date -Iseconds)] Finished repetition $rep"
done

echo ""
echo "=============================================="
echo "All 4 repeats finished."
echo "Outputs: ${BASE_OUTPUT}_rep2, _rep3, _rep4, _rep5"
echo "=============================================="
