#!/bin/bash

# JARVIS Production Deployment Script
# Deploys the complete JARVIS system with all features from analyzed repositories

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}           JARVIS PRODUCTION DEPLOYMENT${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Function to check prerequisites
check_prerequisites() {
    echo -e "\n${YELLOW}Checking prerequisites...${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed${NC}"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}Docker Compose is not installed${NC}"
        exit 1
    fi
    
    # Check for required environment variables
    required_vars=("OPENAI_API_KEY" "DB_PASSWORD")
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            echo -e "${YELLOW}Warning: $var is not set. Some features may not work.${NC}"
        fi
    done
    
    echo -e "${GREEN}Prerequisites check passed${NC}"
}

# Function to setup environment
setup_environment() {
    echo -e "\n${YELLOW}Setting up environment...${NC}"
    
    # Create necessary directories
    mkdir -p models/{whisper,vosk,ollama}
    mkdir -p voices
    mkdir -p logs
    mkdir -p audio_cache
    mkdir -p monitoring/{prometheus,grafana/dashboards,grafana/datasources}
    mkdir -p nginx/ssl
    
    # Create .env file if not exists
    if [ ! -f .env ]; then
        echo -e "${YELLOW}Creating .env file...${NC}"
        cat > .env <<EOF
# JARVIS Environment Configuration
DB_PASSWORD=jarvis_secure_2024
GRAFANA_PASSWORD=admin_secure_2024

# AI Provider Keys (Optional - for cloud models)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
HUGGINGFACE_TOKEN=

# Voice Services (Optional)
PORCUPINE_ACCESS_KEY=
ELEVENLABS_API_KEY=

# Feature Flags
ENABLE_VOICE=true
ENABLE_LOCAL_MODELS=true
ENABLE_WEB_SEARCH=true
EOF
        echo -e "${GREEN}.env file created${NC}"
    fi
    
    echo -e "${GREEN}Environment setup completed${NC}"
}

# Function to download models
download_models() {
    echo -e "\n${YELLOW}Downloading AI models...${NC}"
    
    # Download Whisper model
    if [ ! -f "models/whisper/base.pt" ]; then
        echo "Downloading Whisper base model..."
        docker run --rm -v $(pwd)/models/whisper:/models python:3.10 \
            pip install openai-whisper && \
            python -c "import whisper; whisper.load_model('base', download_root='/models')"
    fi
    
    # Download Vosk model (English)
    if [ ! -d "models/vosk/vosk-model-en-us-0.22" ]; then
        echo "Downloading Vosk model..."
        cd models/vosk
        wget -q https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
        unzip -q vosk-model-en-us-0.22.zip
        rm vosk-model-en-us-0.22.zip
        cd ../..
    fi
    
    echo -e "${GREEN}Models downloaded${NC}"
}

# Function to pull Ollama models
setup_ollama_models() {
    echo -e "\n${YELLOW}Setting up Ollama models...${NC}"
    
    # Start Ollama temporarily
    docker-compose -f docker-compose-jarvis.yml up -d ollama
    sleep 5
    
    # Pull models
    models=("llama3" "mistral" "codellama")
    for model in "${models[@]}"; do
        echo "Pulling $model..."
        docker exec jarvis-ollama ollama pull $model || true
    done
    
    echo -e "${GREEN}Ollama models ready${NC}"
}

# Function to build Docker images
build_images() {
    echo -e "\n${YELLOW}Building Docker images...${NC}"
    
    # Create Dockerfiles if not exist
    mkdir -p docker/jarvis
    
    # Main JARVIS Dockerfile
    cat > docker/jarvis/Dockerfile <<'EOF'
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    portaudio19-dev \
    python3-pyaudio \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/requirements.txt backend/requirements_jarvis.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements_jarvis.txt

# Copy application code
COPY backend/app ./app
COPY backend/app/services/jarvis_orchestrator.py ./app/services/
COPY backend/app/services/voice_pipeline.py ./app/services/
COPY backend/app/api/v1/endpoints/jarvis_websocket.py ./app/api/v1/endpoints/

# Expose ports
EXPOSE 8888 8889

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8888", "--reload"]
EOF

    # Build images
    docker-compose -f docker-compose-jarvis.yml build
    
    echo -e "${GREEN}Docker images built${NC}"
}

# Function to start services
start_services() {
    echo -e "\n${YELLOW}Starting JARVIS services...${NC}"
    
    # Start core services first
    docker-compose -f docker-compose-jarvis.yml up -d jarvis-postgres jarvis-redis
    sleep 5
    
    # Start JARVIS services
    docker-compose -f docker-compose-jarvis.yml up -d
    
    echo -e "${GREEN}All services started${NC}"
}

# Function to run health checks
health_check() {
    echo -e "\n${YELLOW}Running health checks...${NC}"
    
    sleep 10
    
    # Check WebSocket endpoint
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8888/health | grep -q "200"; then
        echo -e "${GREEN}✓ WebSocket service healthy${NC}"
    else
        echo -e "${RED}✗ WebSocket service not responding${NC}"
    fi
    
    # Check Ollama
    if curl -s http://localhost:11434/api/tags | grep -q "models"; then
        echo -e "${GREEN}✓ Ollama service healthy${NC}"
    else
        echo -e "${RED}✗ Ollama service not responding${NC}"
    fi
    
    # Check Redis
    if docker exec jarvis-redis redis-cli ping | grep -q "PONG"; then
        echo -e "${GREEN}✓ Redis healthy${NC}"
    else
        echo -e "${RED}✗ Redis not responding${NC}"
    fi
    
    # Check PostgreSQL
    if docker exec jarvis-postgres pg_isready | grep -q "accepting connections"; then
        echo -e "${GREEN}✓ PostgreSQL healthy${NC}"
    else
        echo -e "${RED}✗ PostgreSQL not responding${NC}"
    fi
}

# Function to display access information
display_info() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}           JARVIS DEPLOYMENT COMPLETE${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    
    echo -e "\n${GREEN}Access Points:${NC}"
    echo -e "  WebSocket Interface: ${BLUE}ws://localhost:8888${NC}"
    echo -e "  REST API: ${BLUE}http://localhost:8889${NC}"
    echo -e "  Voice Service: ${BLUE}http://localhost:8890${NC}"
    echo -e "  Grafana Dashboard: ${BLUE}http://localhost:3000${NC}"
    echo -e "  Prometheus: ${BLUE}http://localhost:9090${NC}"
    
    echo -e "\n${GREEN}Features Enabled:${NC}"
    echo -e "  ✓ Voice Recognition (Whisper + Vosk + Google)"
    echo -e "  ✓ Wake Word Detection ('Jarvis')"
    echo -e "  ✓ Multi-Model AI (GPT-4, Claude, Local)"
    echo -e "  ✓ Real-time WebSocket Communication"
    echo -e "  ✓ Local LLM Support (Ollama)"
    echo -e "  ✓ HuggingFace Model Integration"
    
    echo -e "\n${GREEN}Quick Test:${NC}"
    echo -e "  Text: ${YELLOW}curl -X POST http://localhost:8889/jarvis/chat -d '{\"message\":\"Hello JARVIS\"}'${NC}"
    echo -e "  WebSocket: ${YELLOW}wscat -c ws://localhost:8888${NC}"
    
    echo -e "\n${GREEN}Logs:${NC}"
    echo -e "  ${YELLOW}docker-compose -f docker-compose-jarvis.yml logs -f jarvis-core${NC}"
}

# Main deployment flow
main() {
    echo -e "${YELLOW}Starting JARVIS deployment...${NC}"
    
    check_prerequisites
    setup_environment
    download_models
    build_images
    setup_ollama_models
    start_services
    health_check
    display_info
    
    echo -e "\n${GREEN}Deployment completed successfully!${NC}"
    echo -e "${YELLOW}JARVIS is ready to serve.${NC}"
}

# Handle command line arguments
case "${1:-}" in
    stop)
        echo -e "${YELLOW}Stopping JARVIS services...${NC}"
        docker-compose -f docker-compose-jarvis.yml down
        echo -e "${GREEN}Services stopped${NC}"
        ;;
    restart)
        echo -e "${YELLOW}Restarting JARVIS services...${NC}"
        docker-compose -f docker-compose-jarvis.yml restart
        echo -e "${GREEN}Services restarted${NC}"
        ;;
    logs)
        docker-compose -f docker-compose-jarvis.yml logs -f
        ;;
    status)
        docker-compose -f docker-compose-jarvis.yml ps
        ;;
    test)
        echo -e "${YELLOW}Running JARVIS tests...${NC}"
        pytest tests/test_jarvis.py -v
        ;;
    *)
        main
        ;;
esac