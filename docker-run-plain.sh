#!/bin/bash
# Run UCVME using plain docker commands (no docker-compose needed)

set -e

# Configuration
IMAGE_NAME="ucvme:latest"
DATA_DIR_HOST="/work/ammar/sslrp/data/So2Sat_POP"
DATA_DIR_CONTAINER="/workspace/data/So2Sat_POP"
OUTPUT_DIR_HOST="./output"
OUTPUT_DIR_CONTAINER="/workspace/output"
CONFIGS_DIR_HOST="./configs"
CONFIGS_DIR_CONTAINER="/workspace/configs"
SCRIPTS_DIR_HOST="./scripts"
SCRIPTS_DIR_CONTAINER="/workspace/scripts"

# Parse command
CMD="${1:-shell}"

# GPU support (uncomment if you have GPU)
GPU_FLAGS=""
# GPU_FLAGS="--gpus all"

case "$CMD" in
    build)
        echo "Building Docker image..."
        docker build -t "${IMAGE_NAME}" .
        ;;
    shell|bash)
        echo "Starting interactive shell..."
        docker run -it --rm \
            ${GPU_FLAGS} \
            -v "${DATA_DIR_HOST}:${DATA_DIR_CONTAINER}" \
            -v "${OUTPUT_DIR_HOST}:${OUTPUT_DIR_CONTAINER}" \
            -v "${CONFIGS_DIR_HOST}:${CONFIGS_DIR_CONTAINER}" \
            -v "${SCRIPTS_DIR_HOST}:${SCRIPTS_DIR_CONTAINER}" \
            -v "$(pwd)/DATA_DIR:/workspace/DATA_DIR" \
            -w /workspace \
            "${IMAGE_NAME}" \
            bash
        ;;
    train)
        CONFIG="${2:-/workspace/configs/so2sat_pop_resnet50.yaml}"
        OUTPUT="${3:-${OUTPUT_DIR_CONTAINER}/so2sat_pop_run}"
        echo "Starting training..."
        echo "  Config: ${CONFIG}"
        echo "  Output: ${OUTPUT}"
        docker run -it --rm \
            ${GPU_FLAGS} \
            -v "${DATA_DIR_HOST}:${DATA_DIR_CONTAINER}" \
            -v "${OUTPUT_DIR_HOST}:${OUTPUT_DIR_CONTAINER}" \
            -v "${CONFIGS_DIR_HOST}:${CONFIGS_DIR_CONTAINER}" \
            -v "${SCRIPTS_DIR_HOST}:${SCRIPTS_DIR_CONTAINER}" \
            -v "$(pwd)/DATA_DIR:/workspace/DATA_DIR" \
            -w /workspace \
            "${IMAGE_NAME}" \
            python3 ucvme_age.py --config="${CONFIG}" --output="${OUTPUT}" "${@:4}"
        ;;
    test)
        CONFIG="${2:-/workspace/configs/so2sat_pop_resnet50.yaml}"
        OUTPUT="${3:-${OUTPUT_DIR_CONTAINER}/so2sat_pop_run}"
        WEIGHTS="${4:-${OUTPUT}/best.pt}"
        echo "Running tests..."
        echo "  Config: ${CONFIG}"
        echo "  Output: ${OUTPUT}"
        echo "  Weights: ${WEIGHTS}"
        docker run -it --rm \
            ${GPU_FLAGS} \
            -v "${DATA_DIR_HOST}:${DATA_DIR_CONTAINER}" \
            -v "${OUTPUT_DIR_HOST}:${OUTPUT_DIR_CONTAINER}" \
            -v "${CONFIGS_DIR_HOST}:${CONFIGS_DIR_CONTAINER}" \
            -v "${SCRIPTS_DIR_HOST}:${SCRIPTS_DIR_CONTAINER}" \
            -v "$(pwd)/DATA_DIR:/workspace/DATA_DIR" \
            -w /workspace \
            "${IMAGE_NAME}" \
            python3 ucvme_age.py --config="${CONFIG}" --output="${OUTPUT}" --weights="${WEIGHTS}" --test_only "${@:5}"
        ;;
    create-filelist)
        echo "Creating FileList.csv..."
        docker run -it --rm \
            -v "${DATA_DIR_HOST}:${DATA_DIR_CONTAINER}" \
            -v "${SCRIPTS_DIR_HOST}:${SCRIPTS_DIR_CONTAINER}" \
            -w /workspace \
            "${IMAGE_NAME}" \
            python scripts/create_so2sat_filelist.py \
                --data_dir "${DATA_DIR_CONTAINER}" \
                --output "${DATA_DIR_CONTAINER}/FileList.csv"
        ;;
    *)
        echo "Usage: $0 {build|shell|train|test|create-filelist} [options]"
        echo ""
        echo "Commands:"
        echo "  build                    - Build the Docker image"
        echo "  shell                    - Start interactive bash shell"
        echo "  train [config] [output]  - Run training"
        echo "  test [config] [output] [weights] - Run testing"
        echo "  create-filelist          - Create FileList.csv for So2Sat_POP"
        echo ""
        echo "Examples:"
        echo "  $0 build"
        echo "  $0 shell"
        echo "  $0 train /workspace/configs/so2sat_pop_resnet50.yaml /workspace/output/so2sat_pop"
        echo "  $0 test /workspace/configs/so2sat_pop_resnet50.yaml /workspace/output/so2sat_pop /workspace/output/so2sat_pop/best.pt"
        echo "  $0 create-filelist"
        exit 1
        ;;
esac

