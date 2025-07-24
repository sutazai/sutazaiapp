#!/bin/bash
# SutazAI Baseline Deployment Script
# Automated deployment with consistent setup

set -e  # Exit on error

echo "========================================="
echo "SutazAI AGI/ASI System Deployment"
echo "========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."
if ! command_exists docker; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

if ! command_exists docker compose && ! command_exists docker-compose; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    exit 1
fi

# Stop and clean existing containers
echo "Cleaning up existing containers..."
docker compose down -v 2>/dev/null || docker-compose down -v 2>/dev/null || true
docker stop sutazai-backend-minimal 2>/dev/null || true
docker rm sutazai-backend-minimal 2>/dev/null || true

# Create necessary directories
echo "Creating directories..."
mkdir -p backend/app frontend monitoring/prometheus monitoring/grafana/dashboards
mkdir -p data logs models scripts

# Fix docker-compose.yml if needed
echo "Preparing docker-compose configuration..."
if [ -f docker-compose.yml.backup ]; then
    cp docker-compose.yml.backup docker-compose.yml.original-backup
fi

# Start core services
echo "Starting core infrastructure services..."
docker compose up -d postgres redis

# Wait for services to be healthy
echo "Waiting for PostgreSQL and Redis to be healthy..."
sleep 10

# Start vector databases
echo "Starting vector databases..."
docker compose up -d chromadb qdrant

# Start Ollama (without GPU)
echo "Starting Ollama..."
docker compose up -d ollama

# Build and start minimal backend
echo "Building minimal backend..."
docker build -t sutazaiapp-backend-minimal -f backend/Dockerfile.minimal backend/

echo "Starting minimal backend..."
docker run -d --name sutazai-backend-minimal \
    --network sutazaiapp_sutazai-network \
    -p 8000:8000 \
    -e PYTHONPATH=/app \
    sutazaiapp-backend-minimal \
    uvicorn app.main_minimal:app --host 0.0.0.0 --port 8000

# Start frontend
echo "Starting frontend..."
docker compose up -d frontend --no-deps

# Start monitoring
echo "Starting monitoring services..."
docker compose up -d prometheus grafana

# Wait for services to start
echo "Waiting for services to initialize..."
sleep 10

# Check service status
echo -e "\n${GREEN}Service Status:${NC}"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep sutazai

# Test endpoints
echo -e "\n${GREEN}Testing endpoints:${NC}"
echo -n "Backend Health: "
curl -s http://localhost:8000/health | jq -r '.status' 2>/dev/null || echo "Failed"

echo -n "Frontend Status: "
curl -s -o /dev/null -w "%{http_code}" http://localhost:8501 || echo "Failed"

echo -e "\n${GREEN}=========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "Access points:"
echo "- Frontend: http://localhost:8501"
echo "- Backend API: http://localhost:8000"
echo "- Prometheus: http://localhost:9090"
echo "- Grafana: http://localhost:3000"
echo "- ChromaDB: http://localhost:8001"
echo "- Qdrant: http://localhost:6333"
echo "- Ollama: http://localhost:11434"
echo ""
echo "To view logs: docker logs <container-name>"
echo "To stop all services: docker compose down"