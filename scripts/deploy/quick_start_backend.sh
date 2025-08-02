#!/bin/bash
# Quick start script for SutazAI backend while Docker builds

set -euo pipefail

PROJECT_ROOT="/opt/sutazaiapp"
BACKEND_DIR="${PROJECT_ROOT}/backend"

echo "🚀 SutazAI Quick Start - Backend"
echo "=================================="

# Kill any existing simple backend
docker stop simple-backend 2>/dev/null || true
docker rm simple-backend 2>/dev/null || true

# Start working backend container
echo "Starting comprehensive backend..."

docker run -d \
  --name sutazai-working-backend \
  --network sutazaiapp_sutazai-network \
  -p 8000:8000 \
  -v "${BACKEND_DIR}/app/working_main.py:/app/main.py" \
  -e DATABASE_URL="postgresql://sutazai:sutazai_password@sutazai-postgres:5432/sutazai" \
  -e NEO4J_URI="bolt://sutazai-neo4j:7687" \
  -e NEO4J_USER="neo4j" \
  -e NEO4J_PASSWORD="sutazai_neo4j_password" \
  python:3.11-slim bash -c "
    pip install fastapi uvicorn httpx psutil pydantic &&
    cd /app &&
    python main.py
  "

echo "Backend starting... waiting 30 seconds"
sleep 30

# Test the backend
echo "Testing backend health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend is healthy!"
    echo "🌐 Frontend should now work at: http://172.31.77.193:8501/"
    echo "🔧 Backend API available at: http://localhost:8000"
    echo "📚 API docs at: http://localhost:8000/docs"
else
    echo "❌ Backend health check failed"
    echo "Checking logs..."
    docker logs sutazai-working-backend --tail 20
fi