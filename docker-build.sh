#!/bin/bash
# Build Docker image using plain docker commands (no docker-compose needed)

set -e

echo "Building UCVME Docker image..."
docker build -t ucvme:latest .

echo ""
echo "✓ Build complete!"
echo ""
echo "To run the container, use:"
echo "  ./docker-run-plain.sh shell    # Interactive shell"
echo "  ./docker-run-plain.sh train    # Run training"
echo "  ./docker-run-plain.sh test     # Run testing"

