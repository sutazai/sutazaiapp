#!/bin/bash

set -e

echo "🚀 Starting SutazAI with Real Agents"
echo "===================================="

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if real agent configuration exists
if [ ! -f "docker-compose-real-agents.yml" ]; then
    warning "Real agent configuration not found. Running automated deployment..."
    ./deploy_automated_sutazai_system.sh
    exit 0
fi

# Check if services are already running
if docker-compose -f docker-compose-real-agents.yml ps | grep -q "Up"; then
    info "Some services already running. Checking status..."
    
    # Show current status
    docker-compose -f docker-compose-real-agents.yml ps
    
    read -p "Do you want to restart all services? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Restarting all services..."
        docker-compose -f docker-compose-real-agents.yml down
        sleep 5
    else
        log "Starting missing services..."
    fi
fi

# Start services in proper order
log "Starting core infrastructure..."
docker-compose -f docker-compose-real-agents.yml up -d postgres redis qdrant chromadb ollama

log "Waiting for core services to be ready..."
sleep 20

log "Starting real AI agents..."
docker-compose -f docker-compose-real-agents.yml up -d tabby open-webui langflow dify-api browserless

log "Waiting for agents to initialize..."
sleep 30

log "Starting application services..."
docker-compose -f docker-compose-real-agents.yml up -d sutazai-backend sutazai-streamlit

log "Starting monitoring..."
docker-compose -f docker-compose-real-agents.yml up -d prometheus grafana

log "Waiting for all services to stabilize..."
sleep 15

# Verify services are running
log "Verifying service status..."
docker-compose -f docker-compose-real-agents.yml ps

# Test key endpoints
log "Testing system health..."

# Test backend
if curl -s http://localhost:8000/health | grep -q "healthy"; then
    log "✅ Backend is healthy"
else
    warning "Backend may still be starting"
fi

# Test Streamlit
if curl -s http://localhost:8501 > /dev/null; then
    log "✅ Streamlit frontend is accessible"
else
    warning "Frontend may still be starting"
fi

# Test OpenWebUI
if curl -s http://localhost:8089 > /dev/null; then
    log "✅ OpenWebUI is accessible"
else
    warning "OpenWebUI may still be starting"
fi

# Show final status
echo ""
echo "🎉 SutazAI Real Agent System Started!"
echo "====================================="
echo ""
echo "📊 Access Points:"
echo "   Main Interface:    http://localhost:8501"
echo "   Backend API:       http://localhost:8000"
echo "   OpenWebUI:         http://localhost:8089"
echo "   LangFlow:          http://localhost:7860"
echo "   Monitoring:        http://localhost:3000"
echo ""
echo "🔧 Management:"
echo "   View status:       docker-compose -f docker-compose-real-agents.yml ps"
echo "   View logs:         docker-compose -f docker-compose-real-agents.yml logs -f"
echo "   Stop system:       docker-compose -f docker-compose-real-agents.yml down"
echo ""
echo "All agents are REAL - no mocks! 🤖"