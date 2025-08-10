#!/bin/bash

# SutazAI Missing Services Preparation Script
# Ensures all required directories and configurations exist before deployment

set -euo pipefail


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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔧 Preparing SutazAI Missing Services deployment..."

# Create required directories
echo "📁 Creating required directories..."
mkdir -p "$PROJECT_ROOT/configs/"{neo4j,kong,consul,rabbitmq,loki,alertmanager,backend,frontend,resource-manager}
mkdir -p "$PROJECT_ROOT/services/"{resource-manager,faiss-vector,ai-metrics}

# Create service directories if they don't exist
if [ ! -d "$PROJECT_ROOT/backend" ]; then
    echo "⚠️  Warning: backend directory doesn't exist. Creating placeholder..."
    mkdir -p "$PROJECT_ROOT/backend"
    cat > "$PROJECT_ROOT/backend/main.py" << 'EOF'
# Placeholder backend service
from fastapi import FastAPI

app = FastAPI(title="SutazAI Backend API")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "backend-api"}

@app.get("/")
async def root():
    return {"message": "SutazAI Backend API", "version": "1.0.0"}
EOF
fi

if [ ! -d "$PROJECT_ROOT/frontend" ]; then
    echo "⚠️  Warning: frontend directory doesn't exist. Creating placeholder..."
    mkdir -p "$PROJECT_ROOT/frontend"
    cat > "$PROJECT_ROOT/frontend/package.json" << 'EOF'
{
  "name": "sutazai-frontend",
  "version": "1.0.0",
  "main": "index.js",
  "scripts": {
    "build": "echo 'Building frontend...' && mkdir -p build && echo '<h1>SutazAI Frontend</h1>' > build/index.html",
    "start": "serve -s build -l 3000"
  },
  "dependencies": {
    "serve": "^14.0.0"
  }
}
EOF
fi

# Create service files if they don't exist
for service in resource-manager faiss-vector ai-metrics; do
    if [ ! -f "$PROJECT_ROOT/services/$service/main.py" ]; then
        echo "⚠️  Warning: $service service doesn't exist. Creating placeholder..."
        mkdir -p "$PROJECT_ROOT/services/$service"
        cat > "$PROJECT_ROOT/services/$service/main.py" << EOF
# Placeholder $service service
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="SutazAI ${service^} Service")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "$service"}

@app.get("/")
async def root():
    return {"message": "SutazAI ${service^} Service", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF
    fi
done

# Test Docker image availability
echo "🐳 Testing Docker image availability..."
images=(
    "neo4j:5-community"
    "kong:3.5"
    "hashicorp/consul:1.17"
    "rabbitmq:3.12-management-alpine"
    "python:3.11-slim"
    "node:18-alpine"
    "grafana/loki:3.0.0"
    "prom/alertmanager:v0.27.0"
)

failed_images=()
for image in "${images[@]}"; do
    echo "  Testing $image..."
    if ! docker pull "$image" >/dev/null 2>&1; then
        failed_images+=("$image")
        echo "    ❌ Failed to pull $image"
    else
        echo "    ✅ $image available"
    fi
done

if [ ${#failed_images[@]} -ne 0 ]; then
    echo "❌ Failed to pull the following images:"
    printf '  - %s\n' "${failed_images[@]}"
    exit 1
fi

# Check for required external network
if ! docker network inspect sutazai-network >/dev/null 2>&1; then
    echo "📡 Creating sutazai-network..."
    docker network create sutazai-network --driver bridge
else
    echo "✅ sutazai-network already exists"
fi

# Check for external volume
if ! docker volume inspect shared_runtime_data >/dev/null 2>&1; then
    echo "💾 Creating shared_runtime_data volume..."
    docker volume create shared_runtime_data
else
    echo "✅ shared_runtime_data volume already exists"
fi

echo "✅ All preparations completed successfully!"
echo "🚀 You can now run: docker-compose -f docker-compose.missing-services.yml up -d"