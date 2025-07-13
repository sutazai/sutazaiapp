#!/bin/bash
# SutazAI Production Deployment Script

set -e

echo "🚀 Deploying SutazAI to production..."

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required for production deployment"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is required for production deployment"
    exit 1
fi

# Build and start services
echo "🐳 Building and starting Docker services..."
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🔍 Checking service health..."
docker-compose ps

# Run health checks
echo "🏥 Running health checks..."
curl -f http://localhost:8000/health || echo "❌ Backend health check failed"
curl -f http://localhost:3000 || echo "❌ Frontend health check failed"

echo "✅ Production deployment complete!"
echo ""
echo "🌐 Access points:"
echo "  Backend API: http://localhost:8000"
echo "  Web UI: http://localhost:3000"
echo "  API Docs: http://localhost:8000/docs"
