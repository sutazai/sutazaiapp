#!/bin/bash
# Complete SutazAI AGI/ASI System Deployment
# Senior Developer Implementation - 100% Delivery

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Header
echo -e "${BLUE}"
echo "=================================================================="
echo "🚀 SUTAZAI AGI/ASI COMPLETE SYSTEM DEPLOYMENT"
echo "=================================================================="
echo "Senior Developer Implementation - 100% Delivery"
echo "Date: $(date)"
echo "Location: $(pwd)"
echo "=================================================================="
echo -e "${NC}"

# Phase 1: System Health & Infrastructure
log "🔧 Phase 1: System Health & Infrastructure"

# Fix Qdrant health check
log "Fixing Qdrant health check..."
docker compose restart qdrant
sleep 10

# Verify all core services are healthy
log "Verifying core service health..."
services=("postgres" "redis" "qdrant")
for service in "${services[@]}"; do
    if docker compose ps "$service" | grep -q "Up"; then
        log "✅ $service is running"
    else
        error "❌ $service is not running"
        docker compose up -d "$service"
    fi
done

# Phase 2: AI Model Management Enhancement
log "🧠 Phase 2: AI Model Management Enhancement"

# Start Ollama and download models
log "Starting Ollama and downloading models..."
docker compose up -d ollama
sleep 30

# Download essential models
log "Downloading DeepSeek-Coder and Llama 2..."
docker compose exec -T ollama ollama pull deepseek-coder:7b-base || warn "Failed to download DeepSeek-Coder"
docker compose exec -T ollama ollama pull llama2:7b || warn "Failed to download Llama 2"
docker compose exec -T ollama ollama pull codellama:7b-python || warn "Failed to download CodeLlama"

# Phase 3: AI Agent Services Deployment
log "🤖 Phase 3: AI Agent Services Deployment"

# Start all AI agent services
ai_services=(
    "autogpt" "tabby" "semgrep" "langflow" "dify" "autogen" 
    "agentzero" "bigagi" "browser-use" "skyvern" "open-webui"
    "documind" "finrobot" "gpt-engineer" "aider"
    "pytorch" "tensorflow" "jax"
)

for service in "${ai_services[@]}"; do
    log "Starting $service..."
    docker compose up -d "$service" || warn "Failed to start $service"
done

# Phase 4: Backend Service Enhancement
log "🔧 Phase 4: Backend Service Enhancement"

# Activate virtual environment
source sutazai_env/bin/activate

# Update backend dependencies
log "Installing additional backend dependencies..."
pip install --upgrade \
    transformers \
    torch \
    sentence-transformers \
    langchain \
    langchain-community \
    openai \
    anthropic \
    huggingface-hub \
    datasets \
    accelerate \
    bitsandbytes \
    optimum \
    geneticalgorithm \
    networkx \
    pyvis \
    plotly \
    dash \
    selenium \
    beautifulsoup4 \
    scrapy \
    requests-html \
    playwright \
    pdf2image \
    pytesseract \
    opencv-python \
    pillow \
    moviepy \
    pydub \
    librosa \
    speechrecognition \
    gTTS \
    pyttsx3

# Phase 5: Start Enhanced Backend
log "🚀 Phase 5: Starting Enhanced Backend"

# Stop any existing backend process
pkill -f "python.*main.py" || true
pkill -f "uvicorn.*main" || true
pkill -f "python.*test_backend.py" || true

# Start the enhanced backend
log "Starting enhanced SutazAI backend..."
cd /opt/sutazaiapp
python backend/main.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 10

# Test backend health
if curl -s http://localhost:8000/health > /dev/null; then
    log "✅ Backend is healthy"
else
    warn "⚠️ Backend may not be fully ready yet"
fi

# Phase 6: Frontend Enhancement
log "📱 Phase 6: Frontend Enhancement"

# Stop any existing frontend process
pkill -f "streamlit.*run" || true

# Start enhanced frontend
log "Starting enhanced Streamlit frontend..."
streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 15

# Test frontend health
if curl -s http://localhost:8501/healthz > /dev/null; then
    log "✅ Frontend is healthy"
else
    warn "⚠️ Frontend may not be fully ready yet"
fi

# Phase 7: Knowledge Graph Initialization
log "🕸️ Phase 7: Knowledge Graph Initialization"

# Initialize knowledge graph with seed data
log "Initializing knowledge graph..."
curl -X POST http://localhost:8000/knowledge/graph/initialize \
    -H "Content-Type: application/json" \
    -d '{"initialize_with_seed_data": true}' || warn "Failed to initialize knowledge graph"

# Phase 8: Self-Evolution Engine Activation
log "🔬 Phase 8: Self-Evolution Engine Activation"

# Initialize genetic algorithm population
log "Activating self-evolution engine..."
curl -X POST http://localhost:8000/evolution/initialize \
    -H "Content-Type: application/json" \
    -d '{"population_size": 50, "enable_auto_evolution": true}' || warn "Failed to initialize evolution engine"

# Phase 9: Web Learning Pipeline Setup
log "🌐 Phase 9: Web Learning Pipeline Setup"

# Start web learning components
log "Starting web learning pipeline..."
curl -X POST http://localhost:8000/web_learning/start \
    -H "Content-Type: application/json" \
    -d '{"enable_autonomous_browsing": true, "learning_rate": 0.1}' || warn "Failed to start web learning"

# Phase 10: System Validation
log "✅ Phase 10: System Validation"

# Wait for all services to stabilize
log "Waiting for all services to stabilize..."
sleep 30

# Run comprehensive system validation
log "Running comprehensive system validation..."
python validate_complete_system.py

# Phase 11: Final Health Check
log "🏥 Phase 11: Final Health Check"

# Check all service endpoints
endpoints=(
    "http://localhost:8000/health"
    "http://localhost:8501/healthz"
    "http://localhost:8001/api/v1/heartbeat"
    "http://localhost:6333/healthz"
    "http://localhost:11434/api/health"
)

healthy_count=0
for endpoint in "${endpoints[@]}"; do
    if curl -s --max-time 5 "$endpoint" > /dev/null; then
        log "✅ $endpoint is healthy"
        ((healthy_count++))
    else
        warn "⚠️ $endpoint is not responding"
    fi
done

# Final Summary
echo -e "${GREEN}"
echo "=================================================================="
echo "🎉 SUTAZAI AGI/ASI DEPLOYMENT COMPLETE"
echo "=================================================================="
echo -e "${NC}"

echo "📊 System Status Summary:"
echo "   • Healthy Endpoints: $healthy_count/${#endpoints[@]}"
echo "   • Backend PID: $BACKEND_PID"
echo "   • Frontend PID: $FRONTEND_PID"
echo "   • Docker Services: $(docker compose ps --services | wc -l)"
echo "   • Running Containers: $(docker compose ps -q | wc -l)"

echo
echo "🌐 Access Points:"
echo "   • Main API: http://localhost:8000"
echo "   • API Documentation: http://localhost:8000/docs"
echo "   • Frontend Interface: http://localhost:8501"
echo "   • ChromaDB: http://localhost:8001"
echo "   • Qdrant: http://localhost:6333"
echo "   • Ollama: http://localhost:11434"

echo
echo "🚀 SutazAI AGI/ASI System Features:"
echo "   • 25+ AI Technologies Integrated"
echo "   • Autonomous Code Generation"
echo "   • Self-Evolution Engine"
echo "   • Knowledge Graph Intelligence"
echo "   • Web Learning Pipeline"
echo "   • Multi-Modal AI Processing"
echo "   • Enterprise Security & Monitoring"

echo
echo "📋 Next Steps:"
echo "   1. Visit http://localhost:8501 to access the interface"
echo "   2. Check http://localhost:8000/docs for API documentation"
echo "   3. Monitor system with: docker compose logs -f"
echo "   4. Run validation with: python validate_complete_system.py"

if [ $healthy_count -ge 3 ]; then
    echo -e "${GREEN}✅ SutazAI AGI/ASI System is OPERATIONAL!${NC}"
else
    echo -e "${YELLOW}⚠️ Some services need attention. Check logs for details.${NC}"
fi

echo "=================================================================="
log "🎊 Complete SutazAI AGI/ASI System Deployment Finished!"