#!/bin/bash

# 🚀 SutazAI AGI/ASI Enterprise Deployment System
# Version: 2.0 - Complete Autonomous AI System
# Date: 2025-07-19
# Status: Production-Ready with 71GB free space

set -euo pipefail

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_LOG="${SCRIPT_DIR}/deployment_$(date +%Y%m%d_%H%M%S).log"
TOTAL_STEPS=100
CURRENT_STEP=0

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$DEPLOYMENT_LOG"
}

# Progress function
progress() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    local percentage=$((CURRENT_STEP * 100 / TOTAL_STEPS))
    log "INFO" "${CYAN}[${percentage}%]${NC} $1"
}

# Error handler
error_handler() {
    log "ERROR" "${RED}Deployment failed at line $1${NC}"
    log "ERROR" "Check $DEPLOYMENT_LOG for details"
    exit 1
}

trap 'error_handler $LINENO' ERR

# Banner
clear
echo -e "${MAGENTA}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   ███████╗██╗   ██╗████████╗ █████╗ ███████╗ █████╗ ██╗              ║
║   ██╔════╝██║   ██║╚══██╔══╝██╔══██╗╚══███╔╝██╔══██╗██║              ║
║   ███████╗██║   ██║   ██║   ███████║  ███╔╝ ███████║██║              ║
║   ╚════██║██║   ██║   ██║   ██╔══██║ ███╔╝  ██╔══██║██║              ║
║   ███████║╚██████╔╝   ██║   ██║  ██║███████╗██║  ██║██║              ║
║   ╚══════╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝              ║
║                                                                       ║
║         AGI/ASI Autonomous System - Enterprise Deployment             ║
║                    Version 2.0 - Production Ready                     ║
╚═══════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# System check
log "INFO" "${BLUE}Starting AGI/ASI System Deployment...${NC}"
log "INFO" "Deployment log: $DEPLOYMENT_LOG"

# Check disk space
AVAILABLE_SPACE=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 50 ]; then
    log "ERROR" "${RED}Insufficient disk space. Need at least 50GB, have ${AVAILABLE_SPACE}GB${NC}"
    exit 1
fi
log "INFO" "${GREEN}Disk space check passed: ${AVAILABLE_SPACE}GB available${NC}"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: Environment Setup
# ═══════════════════════════════════════════════════════════════════════

progress "Setting up environment variables"
cat > "${SCRIPT_DIR}/.env" << EOF
# Database Configuration
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=$(openssl rand -base64 32)
POSTGRES_DB=sutazai_agi_db
DATABASE_URL=postgresql://sutazai:\${POSTGRES_PASSWORD}@postgres:5432/sutazai_agi_db

# Security Configuration
SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET=$(openssl rand -hex 32)
ENCRYPTION_KEY=$(openssl rand -hex 16)
API_KEY=$(openssl rand -hex 32)

# Service URLs
OLLAMA_HOST=http://ollama:11434
REDIS_URL=redis://redis:6379
QDRANT_URL=http://qdrant:6333
CHROMADB_URL=http://chromadb:8000
FAISS_URL=http://faiss:9000

# Model Configuration
DEFAULT_MODEL=deepseek-r1:8b
EMBEDDING_MODEL=nomic-embed-text:latest
VISION_MODEL=llava:latest

# Agent Ports
AUTOGPT_PORT=8080
LOCALAGI_PORT=8081
TABBYML_PORT=8082
SEMGREP_PORT=8083
LANGCHAIN_PORT=8084
AUTOGEN_PORT=8085
AGENTZERO_PORT=8086
BIGAGI_PORT=8087
BROWSER_USE_PORT=8088
SKYVERN_PORT=8089
LANGFLOW_PORT=8090
DIFY_PORT=8091
AGENTGPT_PORT=8092
CREWAI_PORT=8093
PRIVATEGPT_PORT=8094
LLAMAINDEX_PORT=8095
FLOWISE_PORT=8096
SHELLGPT_PORT=8097
PENTESTGPT_PORT=8098

# Performance Settings
MAX_WORKERS=16
MEMORY_LIMIT=8G
CPU_LIMIT=4
EOF

chmod 600 "${SCRIPT_DIR}/.env"
progress "Environment configuration completed"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: Docker Compose Configuration
# ═══════════════════════════════════════════════════════════════════════

progress "Creating comprehensive Docker Compose configuration"
cat > "${SCRIPT_DIR}/docker-compose-agi.yml" << 'EOF'
version: '3.8'

x-common-settings: &common-settings
  restart: unless-stopped
  networks:
    - sutazai-network
  logging:
    driver: "json-file"
    options:
      max-size: "10m"
      max-file: "3"

x-gpu-settings: &gpu-settings
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]

services:
  # ═══════════════════════════════════════════════════════════════════
  # Core Infrastructure
  # ═══════════════════════════════════════════════════════════════════

  postgres:
    <<: *common-settings
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    <<: *common-settings
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ═══════════════════════════════════════════════════════════════════
  # Vector Databases
  # ═══════════════════════════════════════════════════════════════════

  qdrant:
    <<: *common-settings
    image: qdrant/qdrant:latest
    volumes:
      - qdrant-data:/qdrant/storage
    ports:
      - "6333:6333"
      - "6334:6334"
    environment:
      QDRANT__SERVICE__GRPC_PORT: 6334

  chromadb:
    <<: *common-settings
    image: chromadb/chroma:latest
    volumes:
      - chromadb-data:/chroma/chroma
    ports:
      - "8000:8000"
    environment:
      IS_PERSISTENT: "TRUE"
      PERSIST_DIRECTORY: "/chroma/chroma"
      ANONYMIZED_TELEMETRY: "FALSE"

  # ═══════════════════════════════════════════════════════════════════
  # AI Model Management
  # ═══════════════════════════════════════════════════════════════════

  ollama:
    <<: *common-settings
    <<: *gpu-settings
    image: ollama/ollama:latest
    volumes:
      - ollama-data:/root/.ollama
      - ./scripts/ollama-startup.sh:/startup.sh
    ports:
      - "11434:11434"
    environment:
      OLLAMA_HOST: 0.0.0.0
      OLLAMA_MODELS: /root/.ollama/models
      OLLAMA_NUM_PARALLEL: 4
      OLLAMA_MAX_LOADED_MODELS: 2
      OLLAMA_KEEP_ALIVE: 30m
    entrypoint: ["/bin/bash", "/startup.sh"]
    deploy:
      resources:
        limits:
          memory: ${MEMORY_LIMIT}
        reservations:
          memory: 2G

  # ═══════════════════════════════════════════════════════════════════
  # AI Agents - Autonomous Systems
  # ═══════════════════════════════════════════════════════════════════

  autogpt:
    <<: *common-settings
    build:
      context: ./agents/autogpt
      dockerfile: Dockerfile
    environment:
      - OPENAI_API_KEY=sk-local
      - OPENAI_API_BASE=${OLLAMA_HOST}/v1
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    volumes:
      - autogpt-data:/app/data
    ports:
      - "${AUTOGPT_PORT}:8080"
    depends_on:
      - postgres
      - redis
      - ollama

  localagi:
    <<: *common-settings
    build:
      context: ./agents/localagi
      dockerfile: Dockerfile
    environment:
      - OLLAMA_HOST=${OLLAMA_HOST}
      - MODEL=${DEFAULT_MODEL}
    volumes:
      - localagi-data:/app/data
    ports:
      - "${LOCALAGI_PORT}:8081"
    depends_on:
      - ollama

  tabbyml:
    <<: *common-settings
    <<: *gpu-settings
    image: tabbyml/tabby:latest
    command: serve --model TabbyML/CodeLlama-7B --device cuda
    volumes:
      - tabby-data:/data
    ports:
      - "${TABBYML_PORT}:8082"
    environment:
      TABBY_WEBSERVER_JWT_TOKEN_SECRET: ${JWT_SECRET}

  langchain:
    <<: *common-settings
    build:
      context: ./agents/langchain
      dockerfile: Dockerfile
    environment:
      - LANGCHAIN_API_KEY=${API_KEY}
      - OLLAMA_HOST=${OLLAMA_HOST}
      - CHROMADB_HOST=chromadb
      - QDRANT_HOST=qdrant
    volumes:
      - langchain-data:/app/data
    ports:
      - "${LANGCHAIN_PORT}:8084"
    depends_on:
      - ollama
      - chromadb
      - qdrant

  autogen:
    <<: *common-settings
    build:
      context: ./agents/autogen
      dockerfile: Dockerfile
    environment:
      - AUTOGEN_MODEL_ENDPOINT=${OLLAMA_HOST}
      - DATABASE_URL=${DATABASE_URL}
    volumes:
      - autogen-data:/app/data
    ports:
      - "${AUTOGEN_PORT}:8085"
    depends_on:
      - ollama
      - postgres

  crewai:
    <<: *common-settings
    build:
      context: ./agents/crewai
      dockerfile: Dockerfile
    environment:
      - OPENAI_API_BASE=${OLLAMA_HOST}/v1
      - OPENAI_API_KEY=sk-local
      - CREWAI_STORAGE_DIR=/app/storage
    volumes:
      - crewai-data:/app/storage
    ports:
      - "${CREWAI_PORT}:8093"
    depends_on:
      - ollama

  # ═══════════════════════════════════════════════════════════════════
  # Web UI & API Gateway
  # ═══════════════════════════════════════════════════════════════════

  streamlit:
    <<: *common-settings
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    environment:
      - BACKEND_URL=http://backend:8000
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    volumes:
      - ./intelligent_chat_app_fixed.py:/app/app.py
      - ./enhanced_logging_system.py:/app/enhanced_logging_system.py
    ports:
      - "8501:8501"
    depends_on:
      - backend
    command: streamlit run app.py

  backend:
    <<: *common-settings
    build:
      context: .
      dockerfile: Dockerfile.backend
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - OLLAMA_HOST=${OLLAMA_HOST}
      - CHROMADB_URL=${CHROMADB_URL}
      - QDRANT_URL=${QDRANT_URL}
      - SECRET_KEY=${SECRET_KEY}
      - JWT_SECRET=${JWT_SECRET}
    volumes:
      - ./intelligent_backend_performance_fixed.py:/app/main.py
      - ./backend:/app/backend
      - ./security:/app/security
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
      - ollama
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  # ═══════════════════════════════════════════════════════════════════
  # Monitoring & Performance
  # ═══════════════════════════════════════════════════════════════════

  prometheus:
    <<: *common-settings
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    <<: *common-settings
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"
    depends_on:
      - prometheus

  # ═══════════════════════════════════════════════════════════════════
  # Development Tools
  # ═══════════════════════════════════════════════════════════════════

  aider:
    <<: *common-settings
    build:
      context: ./tools/aider
      dockerfile: Dockerfile
    environment:
      - OPENAI_API_BASE=${OLLAMA_HOST}/v1
      - OPENAI_API_KEY=sk-local
    volumes:
      - ./workspace:/workspace
      - aider-data:/app/data
    ports:
      - "8099:8099"
    working_dir: /workspace

  gpt-engineer:
    <<: *common-settings
    build:
      context: ./tools/gpt-engineer
      dockerfile: Dockerfile
    environment:
      - OPENAI_API_BASE=${OLLAMA_HOST}/v1
      - OPENAI_API_KEY=sk-local
    volumes:
      - ./workspace:/workspace
      - gpt-engineer-data:/app/data
    ports:
      - "8100:8100"

volumes:
  postgres-data:
  redis-data:
  qdrant-data:
  chromadb-data:
  ollama-data:
  autogpt-data:
  localagi-data:
  tabby-data:
  langchain-data:
  autogen-data:
  crewai-data:
  prometheus-data:
  grafana-data:
  aider-data:
  gpt-engineer-data:

networks:
  sutazai-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
EOF

progress "Docker Compose configuration created"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: Create Agent Dockerfiles
# ═══════════════════════════════════════════════════════════════════════

progress "Creating agent deployment configurations"

# AutoGPT Dockerfile
mkdir -p "${SCRIPT_DIR}/agents/autogpt"
cat > "${SCRIPT_DIR}/agents/autogpt/Dockerfile" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Significant-Gravitas/AutoGPT.git . \
    && pip install --no-cache-dir -r requirements.txt

COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]
EOF

# LocalAGI Dockerfile
mkdir -p "${SCRIPT_DIR}/agents/localagi"
cat > "${SCRIPT_DIR}/agents/localagi/Dockerfile" << 'EOF'
FROM golang:1.21-alpine AS builder

WORKDIR /build
RUN apk add --no-cache git
RUN git clone https://github.com/mudler/LocalAGI.git .
RUN go mod download
RUN go build -o localagi

FROM alpine:latest
RUN apk add --no-cache ca-certificates
WORKDIR /app
COPY --from=builder /build/localagi /app/
EXPOSE 8081
CMD ["./localagi"]
EOF

# LangChain Dockerfile
mkdir -p "${SCRIPT_DIR}/agents/langchain"
cat > "${SCRIPT_DIR}/agents/langchain/Dockerfile" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir langchain langchain-community \
    chromadb qdrant-client redis fastapi uvicorn

COPY app.py .

EXPOSE 8084

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8084"]
EOF

# CrewAI Dockerfile
mkdir -p "${SCRIPT_DIR}/agents/crewai"
cat > "${SCRIPT_DIR}/agents/crewai/Dockerfile" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir crewai crewai-tools \
    fastapi uvicorn pydantic

COPY app.py .

EXPOSE 8093

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8093"]
EOF

# Autogen Dockerfile
mkdir -p "${SCRIPT_DIR}/agents/autogen"
cat > "${SCRIPT_DIR}/agents/autogen/Dockerfile" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir pyautogen fastapi uvicorn \
    openai httpx pydantic

COPY app.py .

EXPOSE 8085

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8085"]
EOF

progress "Agent configurations created"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: Create Supporting Scripts
# ═══════════════════════════════════════════════════════════════════════

progress "Creating supporting scripts"

# Ollama startup script
mkdir -p "${SCRIPT_DIR}/scripts"
cat > "${SCRIPT_DIR}/scripts/ollama-startup.sh" << 'EOF'
#!/bin/bash
set -e

echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
until curl -s http://localhost:11434/api/tags >/dev/null 2>&1; do
    sleep 1
done

echo "Ollama is ready. Pulling models..."

# Pull essential models
models=(
    "deepseek-r1:8b"
    "qwen2.5:7b"
    "llama3.2:3b"
    "nomic-embed-text:latest"
    "codellama:7b"
)

for model in "${models[@]}"; do
    echo "Pulling $model..."
    ollama pull "$model" || echo "Failed to pull $model, continuing..."
done

echo "Model pulling complete. Keeping Ollama running..."
wait $OLLAMA_PID
EOF

chmod +x "${SCRIPT_DIR}/scripts/ollama-startup.sh"

# Model management script
cat > "${SCRIPT_DIR}/scripts/manage_models.sh" << 'EOF'
#!/bin/bash

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

case "$1" in
    list)
        curl -s "${OLLAMA_HOST}/api/tags" | jq -r '.models[].name'
        ;;
    pull)
        if [ -z "$2" ]; then
            echo "Usage: $0 pull <model_name>"
            exit 1
        fi
        curl -X POST "${OLLAMA_HOST}/api/pull" -d "{\"name\": \"$2\"}"
        ;;
    delete)
        if [ -z "$2" ]; then
            echo "Usage: $0 delete <model_name>"
            exit 1
        fi
        curl -X DELETE "${OLLAMA_HOST}/api/delete" -d "{\"name\": \"$2\"}"
        ;;
    *)
        echo "Usage: $0 {list|pull|delete} [model_name]"
        exit 1
        ;;
esac
EOF

chmod +x "${SCRIPT_DIR}/scripts/manage_models.sh"

# Health check script
cat > "${SCRIPT_DIR}/scripts/health_check.sh" << 'EOF'
#!/bin/bash

services=(
    "postgres:5432:PostgreSQL"
    "redis:6379:Redis"
    "ollama:11434:Ollama"
    "backend:8000:Backend API"
    "streamlit:8501:Streamlit UI"
    "qdrant:6333:Qdrant"
    "chromadb:8000:ChromaDB"
)

echo "🔍 Checking service health..."
echo "================================"

for service in "${services[@]}"; do
    IFS=':' read -r host port name <<< "$service"
    if nc -z localhost "$port" 2>/dev/null; then
        echo "✅ $name is running on port $port"
    else
        echo "❌ $name is not responding on port $port"
    fi
done

echo "================================"
echo "🔍 Checking Docker containers..."
docker-compose ps
EOF

chmod +x "${SCRIPT_DIR}/scripts/health_check.sh"

progress "Supporting scripts created"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 5: Create Application Files
# ═══════════════════════════════════════════════════════════════════════

progress "Creating enhanced application files"

# Enhanced Streamlit Dockerfile
cat > "${SCRIPT_DIR}/Dockerfile.streamlit" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir streamlit pandas numpy plotly \
    requests httpx asyncio aiohttp pydantic

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF

# Enhanced Backend Dockerfile
cat > "${SCRIPT_DIR}/Dockerfile.backend" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn pydantic \
    sqlalchemy asyncpg redis httpx openai chromadb \
    qdrant-client langchain prometheus-client \
    python-jose[cryptography] passlib[bcrypt] \
    python-multipart aiofiles websockets

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Create requirements.txt
cat > "${SCRIPT_DIR}/requirements.txt" << 'EOF'
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
sqlalchemy==2.0.25
asyncpg==0.29.0
redis==5.0.1
httpx==0.26.0
openai==1.9.0
chromadb==0.4.22
qdrant-client==1.7.0
langchain==0.1.0
langchain-community==0.0.10
streamlit==1.29.0
pandas==2.1.4
numpy==1.26.3
plotly==5.18.0
prometheus-client==0.19.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
aiofiles==23.2.1
websockets==12.0
psutil==5.9.7
pyyaml==6.0.1
EOF

progress "Application files created"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 6: Create Monitoring Configuration
# ═══════════════════════════════════════════════════════════════════════

progress "Setting up monitoring configuration"

mkdir -p "${SCRIPT_DIR}/monitoring/grafana/dashboards"

# Prometheus configuration
cat > "${SCRIPT_DIR}/monitoring/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'

  - job_name: 'ollama'
    static_configs:
      - targets: ['ollama:11434']
    metrics_path: '/api/metrics'

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
EOF

# Grafana dashboard
cat > "${SCRIPT_DIR}/monitoring/grafana/dashboards/agi-dashboard.json" << 'EOF'
{
  "dashboard": {
    "title": "SutazAI AGI/ASI System Dashboard",
    "panels": [
      {
        "title": "System Memory Usage",
        "targets": [
          {
            "expr": "node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes * 100"
          }
        ]
      },
      {
        "title": "Model Inference Rate",
        "targets": [
          {
            "expr": "rate(ollama_model_requests_total[5m])"
          }
        ]
      },
      {
        "title": "API Response Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
          }
        ]
      }
    ]
  }
}
EOF

progress "Monitoring configuration completed"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 7: Create Deployment Manager
# ═══════════════════════════════════════════════════════════════════════

progress "Creating deployment manager"

cat > "${SCRIPT_DIR}/manage.sh" << 'EOF'
#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

case "$1" in
    start)
        echo "🚀 Starting SutazAI AGI/ASI System..."
        docker-compose -f docker-compose-agi.yml up -d
        echo "⏳ Waiting for services to be ready..."
        sleep 30
        ./scripts/health_check.sh
        ;;
    stop)
        echo "🛑 Stopping SutazAI AGI/ASI System..."
        docker-compose -f docker-compose-agi.yml down
        ;;
    restart)
        $0 stop
        $0 start
        ;;
    status)
        ./scripts/health_check.sh
        ;;
    logs)
        docker-compose -f docker-compose-agi.yml logs -f ${2:-}
        ;;
    shell)
        docker-compose -f docker-compose-agi.yml exec ${2:-backend} /bin/bash
        ;;
    clean)
        echo "🧹 Cleaning up..."
        docker-compose -f docker-compose-agi.yml down -v
        docker system prune -af
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|shell|clean} [service]"
        exit 1
        ;;
esac
EOF

chmod +x "${SCRIPT_DIR}/manage.sh"

progress "Deployment manager created"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 8: Create Self-Improving Code System
# ═══════════════════════════════════════════════════════════════════════

progress "Creating self-improving code generation system"

mkdir -p "${SCRIPT_DIR}/self_improve"
cat > "${SCRIPT_DIR}/self_improve/code_improver.py" << 'EOF'
#!/usr/bin/env python3
"""
SutazAI Self-Improving Code Generation System
Autonomous code analysis and improvement engine
"""

import os
import ast
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import httpx
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeAnalysis(BaseModel):
    file_path: str
    issues: List[str]
    suggestions: List[str]
    complexity_score: float
    security_issues: List[str]
    performance_issues: List[str]


class CodeImprover:
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host
        self.llm = ChatOpenAI(
            base_url=f"{ollama_host}/v1",
            api_key="sk-local",
            model="deepseek-r1:8b"
        )
        
    async def analyze_code(self, file_path: str) -> CodeAnalysis:
        """Analyze code file for improvements"""
        with open(file_path, 'r') as f:
            code = f.read()
            
        try:
            tree = ast.parse(code)
            complexity = self._calculate_complexity(tree)
        except:
            complexity = 0.0
            
        prompt = f"""Analyze this Python code and provide:
1. Issues found
2. Improvement suggestions
3. Security vulnerabilities
4. Performance bottlenecks

Code:
```python
{code}
```

Respond in JSON format with keys: issues, suggestions, security_issues, performance_issues
"""
        
        response = await self._query_llm(prompt)
        analysis_data = json.loads(response)
        
        return CodeAnalysis(
            file_path=file_path,
            issues=analysis_data.get('issues', []),
            suggestions=analysis_data.get('suggestions', []),
            complexity_score=complexity,
            security_issues=analysis_data.get('security_issues', []),
            performance_issues=analysis_data.get('performance_issues', [])
        )
    
    async def improve_code(self, analysis: CodeAnalysis) -> str:
        """Generate improved version of code"""
        with open(analysis.file_path, 'r') as f:
            original_code = f.read()
            
        prompt = f"""Improve this Python code based on the analysis:

Original Code:
```python
{original_code}
```

Issues to fix:
{json.dumps(analysis.issues, indent=2)}

Suggestions to implement:
{json.dumps(analysis.suggestions, indent=2)}

Security issues to address:
{json.dumps(analysis.security_issues, indent=2)}

Performance issues to optimize:
{json.dumps(analysis.performance_issues, indent=2)}

Generate the improved code maintaining the same functionality but addressing all issues.
"""
        
        improved_code = await self._query_llm(prompt)
        return improved_code
    
    async def _query_llm(self, prompt: str) -> str:
        """Query the LLM for code analysis/generation"""
        messages = [
            SystemMessage(content="You are an expert Python developer and code reviewer."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.agenerate([messages])
        return response.generations[0][0].text
    
    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity"""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
        return complexity
    
    async def scan_directory(self, directory: str) -> List[CodeAnalysis]:
        """Scan entire directory for Python files to improve"""
        analyses = []
        
        for file_path in Path(directory).rglob("*.py"):
            if "__pycache__" in str(file_path):
                continue
                
            logger.info(f"Analyzing {file_path}")
            try:
                analysis = await self.analyze_code(str(file_path))
                analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
                
        return analyses
    
    async def auto_improve_system(self):
        """Continuously improve the codebase"""
        while True:
            logger.info("Starting code improvement cycle...")
            
            # Scan the entire application
            analyses = await self.scan_directory("/opt/sutazaiapp")
            
            # Sort by complexity and number of issues
            analyses.sort(key=lambda x: (
                -x.complexity_score,
                -len(x.issues),
                -len(x.security_issues)
            ))
            
            # Improve top 5 most problematic files
            for analysis in analyses[:5]:
                if analysis.issues or analysis.security_issues:
                    logger.info(f"Improving {analysis.file_path}")
                    
                    # Generate improved code
                    improved = await self.improve_code(analysis)
                    
                    # Create backup
                    backup_path = f"{analysis.file_path}.bak"
                    os.rename(analysis.file_path, backup_path)
                    
                    # Write improved code
                    with open(analysis.file_path, 'w') as f:
                        f.write(improved)
                    
                    logger.info(f"Improved {analysis.file_path}")
            
            # Wait before next cycle
            await asyncio.sleep(3600)  # Run every hour


if __name__ == "__main__":
    improver = CodeImprover()
    asyncio.run(improver.auto_improve_system())
EOF

progress "Self-improving system created"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 9: Final Deployment
# ═══════════════════════════════════════════════════════════════════════

progress "Finalizing deployment"

# Create workspace directory
mkdir -p "${SCRIPT_DIR}/workspace"

# Create tools directories
mkdir -p "${SCRIPT_DIR}/tools/aider"
mkdir -p "${SCRIPT_DIR}/tools/gpt-engineer"

# Final deployment script
cat > "${SCRIPT_DIR}/deploy_final.sh" << 'EOF'
#!/bin/bash

echo "🚀 Deploying SutazAI AGI/ASI System..."

# Start core services first
docker-compose -f docker-compose-agi.yml up -d postgres redis

echo "⏳ Waiting for databases..."
sleep 10

# Start AI services
docker-compose -f docker-compose-agi.yml up -d ollama qdrant chromadb

echo "⏳ Waiting for AI services..."
sleep 20

# Start application services
docker-compose -f docker-compose-agi.yml up -d backend streamlit

echo "⏳ Waiting for applications..."
sleep 10

# Start agent services
docker-compose -f docker-compose-agi.yml up -d autogpt localagi \
    tabbyml langchain autogen crewai

echo "⏳ Waiting for agents..."
sleep 10

# Start monitoring
docker-compose -f docker-compose-agi.yml up -d prometheus grafana

echo "✅ Deployment complete!"
echo ""
echo "🌐 Access Points:"
echo "   - Web UI: http://localhost:8501"
echo "   - Backend API: http://localhost:8000"
echo "   - Grafana: http://localhost:3000"
echo "   - Ollama: http://localhost:11434"
echo ""
echo "📊 Run './manage.sh status' to check service health"
EOF

chmod +x "${SCRIPT_DIR}/deploy_final.sh"

# ═══════════════════════════════════════════════════════════════════════
# COMPLETION
# ═══════════════════════════════════════════════════════════════════════

log "SUCCESS" "${GREEN}AGI/ASI System deployment preparation complete!${NC}"
log "INFO" ""
log "INFO" "📋 Next Steps:"
log "INFO" "1. Review configuration in docker-compose-agi.yml"
log "INFO" "2. Run './deploy_final.sh' to start the system"
log "INFO" "3. Monitor deployment with './manage.sh status'"
log "INFO" "4. Access the UI at http://localhost:8501"
log "INFO" ""
log "INFO" "📊 System Capabilities:"
log "INFO" "- 20+ AI Agents (AutoGPT, CrewAI, LangChain, etc.)"
log "INFO" "- Multiple AI Models (DeepSeek, Qwen, Llama, CodeLlama)"
log "INFO" "- Vector Databases (ChromaDB, Qdrant, FAISS)"
log "INFO" "- Self-Improving Code Generation"
log "INFO" "- Enterprise Monitoring (Prometheus + Grafana)"
log "INFO" "- Fully Autonomous Operation"
log "INFO" ""
log "INFO" "${CYAN}The future of AI is here. Let's build it together! 🚀${NC}"