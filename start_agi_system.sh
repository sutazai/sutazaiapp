#!/bin/bash

echo "🚀 Starting SutazAI AGI/ASI System"
echo "=================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }

# Change to project directory
cd /opt/sutazaiapp

# Check if core services are running
log "Checking core services..."
if ! docker ps | grep -q sutazai-postgres; then
    error "PostgreSQL is not running. Please start core services first."
fi

# Start backend services
log "Starting AGI backend services..."

# 1. Start the main backend
if docker ps | grep -q sutazai-backend; then
    info "Backend already running"
else
    log "Starting backend service..."
    docker run -d \
        --name sutazai-backend \
        --network sutazaiapp_sutazai-network \
        -p 8000:8000 \
        -v $PWD/backend:/app \
        -v $PWD/workspace:/workspace \
        -e DATABASE_URL=postgresql://sutazai:sutazai_password@sutazai-postgres:5432/sutazai \
        -e REDIS_URL=redis://sutazai-redis:6379 \
        -e QDRANT_URL=http://sutazai-qdrant:6333 \
        -e CHROMADB_URL=http://sutazai-chromadb:8000 \
        -e OLLAMA_URL=http://sutazai-ollama:11434 \
        python:3.11-slim \
        bash -c "cd /app && pip install -r requirements.txt && python intelligent_backend.py"
fi

# 2. Start Streamlit UI
if docker ps | grep -q sutazai-streamlit; then
    info "Streamlit already running"
else
    log "Starting Streamlit UI..."
    docker run -d \
        --name sutazai-streamlit \
        --network sutazaiapp_sutazai-network \
        -p 8501:8501 \
        -v $PWD/frontend:/app \
        -e BACKEND_URL=http://sutazai-backend:8000 \
        python:3.11-slim \
        bash -c "cd /app && pip install streamlit requests pandas plotly && streamlit run enhanced_streamlit_app.py"
fi

# 3. Start monitoring (optional for low-resource systems)
if [ "${START_MONITORING:-false}" = "true" ]; then
    log "Starting monitoring services..."
    docker-compose up -d prometheus grafana
fi

# Wait for services to be ready
log "Waiting for services to initialize..."
sleep 10

# Check service health
log "Checking service health..."
services=(
    "Backend:http://localhost:8000/health"
    "Streamlit:http://localhost:8501"
    "Ollama:http://localhost:11434/api/health"
)

for service in "${services[@]}"; do
    name="${service%%:*}"
    url="${service##*:}"
    
    if curl -s -f "$url" > /dev/null 2>&1; then
        info "✓ $name is healthy"
    else
        warning "✗ $name is not responding"
    fi
done

# Pull a model if none exist
log "Checking AI models..."
models=$(docker exec sutazai-ollama ollama list | tail -n +2 | wc -l)
if [ "$models" -eq 0 ]; then
    log "No models found. Pulling llama3.2:1b..."
    docker exec sutazai-ollama ollama pull llama3.2:1b &
    info "Model download started in background"
fi

echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ SutazAI AGI System Started!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "\n${BLUE}Access Points:${NC}"
echo -e "  • Web UI:    ${YELLOW}http://localhost:8501${NC}"
echo -e "  • API:       ${YELLOW}http://localhost:8000${NC}"
echo -e "  • API Docs:  ${YELLOW}http://localhost:8000/docs${NC}"
echo -e "\n${BLUE}Available Services:${NC}"
echo -e "  • Chat with AI models"
echo -e "  • Agent orchestration"
echo -e "  • Knowledge management"
echo -e "  • Self-improvement system"
echo -e "\n${BLUE}Quick Commands:${NC}"
echo -e "  • View logs:     ${YELLOW}docker logs -f sutazai-backend${NC}"
echo -e "  • Stop system:   ${YELLOW}docker stop sutazai-backend sutazai-streamlit${NC}"
echo -e "  • List models:   ${YELLOW}docker exec sutazai-ollama ollama list${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"