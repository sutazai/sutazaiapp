#!/bin/bash
set -e

# Build SutazAI base images
echo "🏗️  Building SutazAI base images..."

# Build Python agent base
echo "Building python-agent-base..."
docker build -t sutazai/python-agent-base:latest -f docker/base/Dockerfile.python-agent-base docker/base/

# Build Node.js base  
echo "Building nodejs-base..."
docker build -t sutazai/nodejs-base:latest -f docker/base/Dockerfile.nodejs-base docker/base/

# Build monitoring base
echo "Building monitoring-base..."
docker build -t sutazai/monitoring-base:latest -f docker/base/Dockerfile.monitoring-base docker/base/

# Build GPU base (if NVIDIA runtime available)
if docker info | grep -q nvidia; then
    echo "Building gpu-python-base..."
    docker build -t sutazai/gpu-python-base:latest -f docker/base/Dockerfile.gpu-python-base docker/base/
else
    echo "⚠️  NVIDIA runtime not available, skipping GPU base image"
fi

echo "✅ Base images built successfully!"

# Tag with version
VERSION=${1:-latest}
if [ "$VERSION" != "latest" ]; then
    docker tag sutazai/python-agent-base:latest sutazai/python-agent-base:$VERSION
    docker tag sutazai/nodejs-base:latest sutazai/nodejs-base:$VERSION
    docker tag sutazai/monitoring-base:latest sutazai/monitoring-base:$VERSION
    if docker info | grep -q nvidia; then
        docker tag sutazai/gpu-python-base:latest sutazai/gpu-python-base:$VERSION
    fi
fi

echo "🎯 Base images ready for use!"
