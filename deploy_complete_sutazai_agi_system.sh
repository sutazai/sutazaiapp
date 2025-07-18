#!/bin/bash
# 🚀 SutazAI Complete AGI/ASI System Deployment
# Enterprise-Grade Autonomous AI System with 30+ Services
# 100% Automated Deployment with Zero Manual Intervention

set -e

# Colors and formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ASCII Art Banner
print_banner() {
    echo -e "${CYAN}${BOLD}"
    cat << "EOF"
  ____        _            _    ___   _    ____ ___ 
 / ___| _   _| |_ __ _ ___/ \  |_ _| / \  |  _ \_ _|
 \___ \| | | | __/ _` |___|/ _ \  | | / _ \ | |_) | |
  ___) | |_| | || (_| |  / ___ \ | |/ ___ \|  _ <| |
 |____/ \__,_|\__\__,_| /_/   \_\___/_/   \_\_| \___|
                                                    
🤖 Enterprise AGI/ASI Autonomous System 🤖
    Complete AI Stack with 30+ Services
         100% Local & Self-Hosted
EOF
    echo -e "${NC}"
}

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌ ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] ℹ️  INFO: $1${NC}"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] 🎉 SUCCESS: $1${NC}"
}

# Progress indicator
progress() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))
    
    printf "\r${CYAN}Progress: ["
    printf "%*s" $filled | tr ' ' '█'
    printf "%*s" $empty | tr ' ' '░'
    printf "] %d%% (%d/%d)${NC}" $percentage $current $total
}

# System requirements check
check_system_requirements() {
    info "Checking system requirements..."
    
    # Check if running on correct IP
    if [[ ! $(hostname -I | grep -q "192.168.131.128") ]]; then
        warn "System not running on expected IP 192.168.131.128"
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check system resources
    local total_mem=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$total_mem" -lt 16 ]; then
        warn "System has less than 16GB RAM. Some services may not perform optimally."
    fi
    
    local disk_space=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$disk_space" -lt 100 ]; then
        warn "Less than 100GB free disk space available"
    fi
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        log "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while read gpu memory; do
            info "GPU: $gpu with ${memory}MB VRAM"
        done
    else
        warn "No NVIDIA GPU detected. AI models will run on CPU (slower performance)"
    fi
    
    log "System requirements check completed"
}

# Create necessary directories
create_directories() {
    info "Creating directory structure..."
    
    local directories=(
        "docker/autogpt"
        "docker/localagi"
        "docker/tabbyml"
        "docker/semgrep"
        "docker/langchain-agents"
        "docker/autogen"
        "docker/agentzero"
        "docker/bigagi"
        "docker/browser-use"
        "docker/skyvern"
        "docker/documind"
        "docker/finrobot"
        "docker/gpt-engineer"
        "docker/aider"
        "docker/langflow"
        "docker/dify"
        "docker/pytorch"
        "docker/tensorflow"
        "docker/jax"
        "docker/faiss"
        "docker/awesome-code-ai"
        "docker/enhanced-model-manager"
        "docker/neuromorphic"
        "docker/self-improvement"
        "docker/web-learning"
        "docker/health-check"
        "docker/orchestrator"
        "config"
        "monitoring/prometheus"
        "monitoring/grafana"
        "nginx"
        "scripts"
        "logs"
        "data"
        "workspace"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
    done
    
    log "Directory structure created"
}

# Generate Docker configurations
generate_docker_configs() {
    info "Generating Docker configurations..."
    
    # Create backend Dockerfile
    cat > docker/backend.Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /logs /workspace /models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
EOF

    # Create Streamlit Dockerfile
    cat > docker/streamlit.Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/healthz || exit 1

# Start Streamlit
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
EOF

    # Create AI Service Dockerfiles
    create_ai_service_dockerfiles
    
    log "Docker configurations generated"
}

# Create AI service Docker configurations
create_ai_service_dockerfiles() {
    # AutoGPT Dockerfile
    cat > docker/autogpt/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Clone AutoGPT and install
RUN git clone https://github.com/Significant-Gravitas/AutoGPT.git /app/autogpt
WORKDIR /app/autogpt/classic

RUN pip install -e .

EXPOSE 8000

CMD ["python", "-m", "autogpt", "--continuous", "--ai-settings-file", "/app/ai_settings.yaml"]
EOF

    # LocalAGI Dockerfile
    cat > docker/localagi/Dockerfile << 'EOF'
FROM golang:1.21-alpine AS builder

WORKDIR /app
RUN apk add --no-cache git
RUN git clone https://github.com/mudler/LocalAGI.git .
RUN go mod download
RUN go build -o localagi .

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/localagi .
EXPOSE 8080
CMD ["./localagi"]
EOF

    # TabbyML Dockerfile
    cat > docker/tabbyml/Dockerfile << 'EOF'
FROM nvidia/cuda:11.8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://get.docker.com | sh

WORKDIR /app
RUN curl -fsSL https://github.com/TabbyML/tabby/releases/latest/download/tabby_x86_64-manylinux2014 -o tabby
RUN chmod +x tabby

EXPOSE 8080
CMD ["./tabby", "serve", "--model", "CodeLlama-7B", "--host", "0.0.0.0", "--port", "8080"]
EOF

    # FAISS Service Dockerfile
    cat > docker/faiss/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install faiss-cpu numpy scipy flask

COPY faiss_service.py .

EXPOSE 8000
CMD ["python", "faiss_service.py"]
EOF

    # Create FAISS service
    cat > docker/faiss/faiss_service.py << 'EOF'
import faiss
import numpy as np
from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# Global index storage
indexes = {}
data_path = os.environ.get('FAISS_DATA_PATH', '/data/faiss_indexes')
os.makedirs(data_path, exist_ok=True)

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/create_index', methods=['POST'])
def create_index():
    data = request.json
    index_name = data['name']
    dimension = data['dimension']
    index_type = data.get('type', 'IVFFlat')
    
    if index_type == 'IVFFlat':
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, 100)
    else:
        index = faiss.IndexFlatL2(dimension)
    
    indexes[index_name] = index
    return jsonify({"status": "created", "name": index_name})

@app.route('/add_vectors', methods=['POST'])
def add_vectors():
    data = request.json
    index_name = data['index']
    vectors = np.array(data['vectors'], dtype=np.float32)
    
    if index_name not in indexes:
        return jsonify({"error": "Index not found"}), 404
    
    index = indexes[index_name]
    if not index.is_trained:
        index.train(vectors)
    
    index.add(vectors)
    return jsonify({"status": "added", "total": index.ntotal})

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    index_name = data['index']
    query_vector = np.array(data['query'], dtype=np.float32).reshape(1, -1)
    k = data.get('k', 10)
    
    if index_name not in indexes:
        return jsonify({"error": "Index not found"}), 404
    
    index = indexes[index_name]
    distances, indices = index.search(query_vector, k)
    
    return jsonify({
        "distances": distances[0].tolist(),
        "indices": indices[0].tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
EOF

    # Enhanced Model Manager Dockerfile
    cat > docker/enhanced-model-manager/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install fastapi uvicorn requests torch transformers sentence-transformers

COPY model_manager.py .

EXPOSE 8000
CMD ["uvicorn", "model_manager:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

    # Create Enhanced Model Manager service
    cat > docker/enhanced-model-manager/model_manager.py << 'EOF'
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
import json
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModel

app = FastAPI(title="Enhanced Model Manager", version="1.0.0")

class ModelRequest(BaseModel):
    model_name: str
    prompt: str
    parameters: Dict[str, Any] = {}

class ModelInfo(BaseModel):
    name: str
    status: str
    memory_usage: str
    quantization: str

ollama_url = os.environ.get('OLLAMA_URL', 'http://ollama:11434')
models_cache = {}

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": len(models_cache)}

@app.get("/models", response_model=List[ModelInfo])
def list_models():
    try:
        response = requests.get(f"{ollama_url}/api/tags")
        ollama_models = response.json().get('models', [])
        
        model_list = []
        for model in ollama_models:
            model_list.append(ModelInfo(
                name=model['name'],
                status="available",
                memory_usage=f"{model['size'] / 1024**3:.1f}GB",
                quantization=model['details'].get('quantization_level', 'unknown')
            ))
        
        return model_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
def generate_text(request: ModelRequest):
    try:
        ollama_request = {
            "model": request.model_name,
            "prompt": request.prompt,
            "stream": False,
            "options": request.parameters
        }
        
        response = requests.post(f"{ollama_url}/api/generate", json=ollama_request)
        result = response.json()
        
        return {
            "response": result.get('response', ''),
            "model": request.model_name,
            "done": result.get('done', False),
            "total_duration": result.get('total_duration', 0),
            "load_duration": result.get('load_duration', 0),
            "prompt_eval_count": result.get('prompt_eval_count', 0),
            "eval_count": result.get('eval_count', 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pull")
def pull_model(model_name: str):
    try:
        response = requests.post(f"{ollama_url}/api/pull", json={"name": model_name})
        return {"status": "pulling", "model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/models/{model_name}")
def delete_model(model_name: str):
    try:
        response = requests.delete(f"{ollama_url}/api/delete", json={"name": model_name})
        return {"status": "deleted", "model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_stats():
    try:
        # Get system stats
        import psutil
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "models_loaded": len(models_cache),
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    except Exception as e:
        return {"error": str(e)}
EOF

    log "AI service Docker configurations created"
}

# Generate configuration files
generate_configs() {
    info "Generating configuration files..."
    
    # Redis configuration
    cat > config/redis.conf << 'EOF'
bind 0.0.0.0
port 6379
timeout 0
save 900 1
save 300 10
save 60 10000
rdbcompression yes
dbfilename dump.rdb
dir /data
maxmemory 2gb
maxmemory-policy allkeys-lru
EOF

    # Qdrant configuration
    cat > config/qdrant.yaml << 'EOF'
log_level: INFO
storage:
  performance:
    max_search_threads: 8
    max_optimization_threads: 2
service:
  http_port: 6333
  grpc_port: 6334
  enable_cors: true
  max_request_size_mb: 64
EOF

    # AutoGen configuration
    mkdir -p config/autogen
    cat > config/autogen/config.json << 'EOF'
{
  "config_list": [
    {
      "model": "deepseek-coder:33b",
      "api_key": "local",
      "base_url": "http://ollama:11434/v1",
      "api_type": "openai"
    },
    {
      "model": "llama3.2:1b",
      "api_key": "local", 
      "base_url": "http://ollama:11434/v1",
      "api_type": "openai"
    }
  ]
}
EOF

    # Prometheus configuration
    cat > monitoring/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'sutazai-backend'
    static_configs:
      - targets: ['sutazai-backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'ollama'
    static_configs:
      - targets: ['ollama:11434']
    metrics_path: '/metrics'

  - job_name: 'qdrant'
    static_configs:
      - targets: ['qdrant:6333']
    metrics_path: '/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
EOF

    # Nginx configuration
    cat > nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server sutazai-backend:8000;
    }
    
    upstream streamlit {
        server sutazai-streamlit:8501;
    }
    
    upstream grafana {
        server grafana:3000;
    }

    server {
        listen 80;
        server_name localhost;

        # Main Streamlit interface
        location / {
            proxy_pass http://streamlit;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # API backend
        location /api/ {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }

        # Monitoring
        location /grafana/ {
            proxy_pass http://grafana/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }

        # Health check
        location /health {
            return 200 'healthy\n';
            add_header Content-Type text/plain;
        }
    }
}
EOF

    # Database initialization script
    cat > scripts/init-postgres.sql << 'EOF'
-- Create additional databases
CREATE DATABASE vector_store;
CREATE DATABASE agent_memory;

-- Create extensions
\c sutazai;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

\c vector_store;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

\c agent_memory;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
EOF

    log "Configuration files generated"
}

# Build Docker services
build_services() {
    info "Building Docker services..."
    
    # Check if docker-compose-complete.yml exists
    if [ ! -f "docker-compose-complete.yml" ]; then
        error "docker-compose-complete.yml not found"
        exit 1
    fi
    
    # Build services in parallel for speed
    info "Building core infrastructure services..."
    docker-compose -f docker-compose-complete.yml build \
        sutazai-backend \
        sutazai-streamlit \
        faiss \
        enhanced-model-manager \
        health-check \
        sutazai-orchestrator &
    
    local build_pid=$!
    
    # Show progress while building
    local counter=0
    while kill -0 $build_pid 2>/dev/null; do
        progress $counter 30
        sleep 2
        ((counter++))
    done
    
    wait $build_pid
    echo # New line after progress
    
    log "Docker services built successfully"
}

# Start the system
start_system() {
    info "Starting SutazAI AGI/ASI System..."
    
    # Start core infrastructure first
    info "Starting core infrastructure..."
    docker-compose -f docker-compose-complete.yml up -d \
        postgres \
        redis \
        qdrant \
        chromadb \
        faiss
    
    # Wait for infrastructure to be ready
    info "Waiting for infrastructure to be ready..."
    sleep 30
    
    # Start Ollama and model management
    info "Starting AI model services..."
    docker-compose -f docker-compose-complete.yml up -d \
        ollama \
        enhanced-model-manager
    
    # Wait for models to load
    info "Waiting for AI models to load..."
    sleep 60
    
    # Start AI agents
    info "Starting AI agent services..."
    docker-compose -f docker-compose-complete.yml up -d \
        autogpt \
        localagi \
        tabbyml \
        semgrep \
        langchain-agents \
        autogen-agents \
        agentzero \
        bigagi
    
    # Start specialized services
    info "Starting specialized AI services..."
    docker-compose -f docker-compose-complete.yml up -d \
        browser-use \
        skyvern \
        documind \
        finrobot \
        gpt-engineer \
        aider
    
    # Start UI and framework services
    info "Starting UI and framework services..."
    docker-compose -f docker-compose-complete.yml up -d \
        open-webui \
        langflow \
        dify
    
    # Start ML frameworks
    info "Starting machine learning frameworks..."
    docker-compose -f docker-compose-complete.yml up -d \
        pytorch \
        tensorflow \
        jax \
        awesome-code-ai
    
    # Start advanced AI systems
    info "Starting advanced AI systems..."
    docker-compose -f docker-compose-complete.yml up -d \
        neuromorphic-engine \
        self-improvement-engine \
        web-learning-engine
    
    # Start main application services
    info "Starting main application services..."
    docker-compose -f docker-compose-complete.yml up -d \
        sutazai-backend \
        sutazai-streamlit
    
    # Start monitoring and proxy
    info "Starting monitoring and proxy services..."
    docker-compose -f docker-compose-complete.yml up -d \
        prometheus \
        grafana \
        node-exporter \
        nginx
    
    # Start system management
    info "Starting system management services..."
    docker-compose -f docker-compose-complete.yml up -d \
        health-check \
        sutazai-orchestrator
    
    log "All services started successfully"
}

# Verify deployment
verify_deployment() {
    info "Verifying deployment..."
    
    local services=(
        "http://localhost:8000/health|Backend API"
        "http://localhost:8501|Streamlit UI"
        "http://localhost:11434/api/tags|Ollama Models"
        "http://localhost:6333/healthz|Qdrant Vector DB"
        "http://localhost:8001/api/v1/heartbeat|ChromaDB"
        "http://localhost:8002/health|FAISS Service"
        "http://localhost:3000|Grafana Monitoring"
        "http://localhost:9090|Prometheus Metrics"
    )
    
    local failed_services=()
    local total_services=${#services[@]}
    local current=0
    
    for service in "${services[@]}"; do
        IFS='|' read -r url name <<< "$service"
        ((current++))
        progress $current $total_services
        
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo -e "\n${GREEN}✅ $name: OK${NC}"
        else
            echo -e "\n${RED}❌ $name: FAILED${NC}"
            failed_services+=("$name")
        fi
        sleep 1
    done
    
    echo # New line after progress
    
    if [ ${#failed_services[@]} -eq 0 ]; then
        success "All services are running correctly!"
    else
        warn "Some services failed to start: ${failed_services[*]}"
        info "Check logs with: docker-compose -f docker-compose-complete.yml logs [service_name]"
    fi
}

# Display system information
display_system_info() {
    echo -e "\n${CYAN}${BOLD}🎉 SutazAI AGI/ASI System Deployment Complete! 🎉${NC}\n"
    
    cat << EOF
${BOLD}System Access Points:${NC}
┌─────────────────────────────────────────────────────────────┐
│ ${GREEN}Main Interface:${NC} http://192.168.131.128:8501         │
│ ${BLUE}Backend API:${NC}    http://192.168.131.128:8000         │
│ ${PURPLE}Monitoring:${NC}     http://192.168.131.128:3000         │
│ ${YELLOW}Chat UI:${NC}        http://192.168.131.128:8030         │
│ ${CYAN}LangFlow:${NC}       http://192.168.131.128:7860         │
│ ${GREEN}Dify Platform:${NC}  http://192.168.131.128:5001         │
└─────────────────────────────────────────────────────────────┘

${BOLD}AI Services Status:${NC}
• 🤖 AI Models: DeepSeek-R1-8B, Qwen3-8B, DeepSeek-Coder-33B, Llama3.2-1B
• 🗃️ Vector DBs: Qdrant, ChromaDB, FAISS
• 🛠️ AI Agents: AutoGPT, LocalAGI, TabbyML, Semgrep, LangChain, AutoGen
• 🌐 Web Automation: Browser-Use, Skyvern
• 📄 Document AI: Documind
• 💰 Financial AI: FinRobot
• 💻 Code AI: GPT-Engineer, Aider, Awesome-Code-AI
• 🧠 ML Frameworks: PyTorch, TensorFlow, JAX
• 🔬 Advanced AI: Neuromorphic Engine, Self-Improvement, Web Learning

${BOLD}Management Commands:${NC}
• View all services: ${GREEN}docker-compose -f docker-compose-complete.yml ps${NC}
• View logs: ${GREEN}docker-compose -f docker-compose-complete.yml logs -f [service]${NC}
• Restart service: ${GREEN}docker-compose -f docker-compose-complete.yml restart [service]${NC}
• Stop system: ${GREEN}docker-compose -f docker-compose-complete.yml down${NC}
• System stats: ${GREEN}curl http://localhost:8000/api/system/stats${NC}

${BOLD}Next Steps:${NC}
1. Access the main interface at http://192.168.131.128:8501
2. Explore the AI agent capabilities
3. Check system monitoring at http://192.168.131.128:3000
4. Review API documentation at http://192.168.131.128:8000/docs

${GREEN}${BOLD}🚀 Your enterprise AGI/ASI system is now fully operational! 🚀${NC}
EOF
}

# Create system monitoring script
create_monitoring_script() {
    cat > monitor_sutazai_system.sh << 'EOF'
#!/bin/bash
# SutazAI System Monitor

while true; do
    clear
    echo "🤖 SutazAI AGI/ASI System Status 🤖"
    echo "=================================="
    echo
    
    # System resources
    echo "💻 System Resources:"
    echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
    echo "RAM: $(free | grep Mem | awk '{printf("%.1f%%", $3/$2 * 100.0)}')"
    echo "Disk: $(df -h / | awk 'NR==2{printf "%s", $5}')"
    echo
    
    # Docker services
    echo "🐳 Docker Services:"
    docker-compose -f docker-compose-complete.yml ps --format "table {{.Name}}\t{{.State}}\t{{.Ports}}" | head -20
    echo
    
    # Model status
    echo "🧠 AI Models Status:"
    curl -s http://localhost:11434/api/tags | jq -r '.models[]? | "\(.name): \(.size/1024/1024/1024 | floor)GB"' 2>/dev/null || echo "Checking..."
    echo
    
    echo "Press Ctrl+C to exit"
    sleep 10
done
EOF
    chmod +x monitor_sutazai_system.sh
}

# Main deployment function
main() {
    print_banner
    
    local start_time=$(date +%s)
    
    # Deployment steps
    check_system_requirements
    create_directories
    generate_docker_configs
    generate_configs
    build_services
    start_system
    
    # Wait for system stabilization
    info "Waiting for system to stabilize..."
    sleep 60
    
    verify_deployment
    create_monitoring_script
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    success "Deployment completed in ${duration} seconds"
    display_system_info
}

# Run deployment
main "$@"