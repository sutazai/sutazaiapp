#!/bin/bash
set -e

echo "🚀 Starting SutazAI Core Services..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Creating from .env.example..."
    cp .env.example .env
fi

# Source environment variables
source .env

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs/backend logs/frontend data uploads models
mkdir -p monitoring/prometheus monitoring/grafana/dashboards

# Start core services only (without agents)
echo "🐳 Starting Docker services..."
docker compose -f docker-compose.yml up -d postgres redis

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL..."
sleep 10

# Start vector databases
docker compose -f docker-compose.yml up -d chromadb qdrant

# Start Ollama for model serving
docker compose -f docker-compose.yml up -d ollama

# Wait for services to be ready
echo "⏳ Waiting for services to initialize..."
sleep 10

# Start backend
docker compose -f docker-compose.yml up -d backend

# Wait for backend to be ready
echo "⏳ Waiting for backend to start..."
sleep 10

# Start frontend
docker compose -f docker-compose.yml up -d frontend

# Check status
echo ""
echo "✅ Core services started!"
echo ""
echo "📊 Service Status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "sutazai|NAMES"

echo ""
echo "🌐 Access Points:"
echo "  - Frontend: http://localhost:8501"
echo "  - Backend API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo ""
echo "💡 To view logs: docker compose logs -f [service-name]"
echo "💡 To stop all services: docker compose down"