#!/bin/bash
# Build base images for Dockerfile consolidation
# Author: Ultra System Architect
# Date: August 10, 2025

set -e


# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

echo "=========================================="
echo "BUILDING CONSOLIDATED BASE IMAGES"
echo "=========================================="

BASE_DIR="/opt/sutazaiapp"
DOCKER_BASE_DIR="$BASE_DIR/docker/base"

# Build Python agent master base
echo "Building Python agent master base image..."
docker build \
  -f "$DOCKER_BASE_DIR/Dockerfile.python-agent-master" \
  -t sutazai-python-agent-master:latest \
  --target production \
  "$DOCKER_BASE_DIR"

if [ $? -eq 0 ]; then
  echo "✅ Python base image built successfully"
else
  echo "❌ Failed to build Python base image"
  exit 1
fi

# Build Node.js agent master base
echo "Building Node.js agent master base image..."
docker build \
  -f "$DOCKER_BASE_DIR/Dockerfile.nodejs-agent-master" \
  -t sutazai-nodejs-agent-master:latest \
  --target production \
  "$DOCKER_BASE_DIR"

if [ $? -eq 0 ]; then
  echo "✅ Node.js base image built successfully"
else
  echo "❌ Failed to build Node.js base image"
  exit 1
fi

# List built images
echo ""
echo "Built images:"
docker images | grep -E "sutazai.*agent-master"

echo ""
echo "Base images ready for migration!"