#!/bin/bash
# Helper script to run UCVME commands in Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo -e "${RED}Error: docker-compose not found${NC}"
    exit 1
fi

# Parse command
CMD="${1:-bash}"

case "$CMD" in
    build)
        echo -e "${GREEN}Building Docker image...${NC}"
        $COMPOSE_CMD build
        ;;
    train)
        OUTPUT_DIR="${2:-/workspace/output}"
        echo -e "${GREEN}Starting training with output directory: $OUTPUT_DIR${NC}"
        $COMPOSE_CMD run --rm ucvme python3 ucvme_age.py --output="$OUTPUT_DIR" "${@:3}"
        ;;
    test)
        OUTPUT_DIR="${2:-/workspace/output}"
        echo -e "${GREEN}Running tests with output directory: $OUTPUT_DIR${NC}"
        $COMPOSE_CMD run --rm ucvme python3 ucvme_age.py --output="$OUTPUT_DIR" --test_only "${@:3}"
        ;;
    shell|bash)
        echo -e "${GREEN}Starting interactive shell...${NC}"
        $COMPOSE_CMD run --rm ucvme bash
        ;;
    up)
        echo -e "${GREEN}Starting container in background...${NC}"
        $COMPOSE_CMD up -d
        ;;
    down)
        echo -e "${GREEN}Stopping container...${NC}"
        $COMPOSE_CMD down
        ;;
    logs)
        $COMPOSE_CMD logs -f ucvme
        ;;
    *)
        echo -e "${YELLOW}Usage: $0 {build|train|test|shell|up|down|logs} [options]${NC}"
        echo ""
        echo "Commands:"
        echo "  build              - Build the Docker image"
        echo "  train [OUTPUT_DIR] - Run training (default: /workspace/output)"
        echo "  test [OUTPUT_DIR]  - Run testing (default: /workspace/output)"
        echo "  shell              - Start interactive bash shell"
        echo "  up                 - Start container in background"
        echo "  down               - Stop container"
        echo "  logs               - Show container logs"
        echo ""
        echo "Examples:"
        echo "  $0 build"
        echo "  $0 train /workspace/output"
        echo "  $0 test /workspace/output"
        echo "  $0 shell"
        exit 1
        ;;
esac

