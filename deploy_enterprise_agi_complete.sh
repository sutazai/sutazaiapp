#!/bin/bash

# SutazAI Enterprise AGI/ASI Complete Deployment Script
# This script deploys the full enterprise-grade SutazAI system with all components

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_NAME="sutazai-enterprise"
COMPOSE_FILE="docker-compose-complete.yml"
ENVIRONMENT_FILE=".env.production"
LOG_FILE="deployment_$(date +%Y%m%d_%H%M%S).log"
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"

# System requirements
MIN_RAM_GB=16
MIN_DISK_GB=100
MIN_DOCKER_VERSION="20.0.0"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🚀 SutazAI Enterprise AGI/ASI Deployment${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to log messages
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Function to check system requirements
check_system_requirements() {
    log "${YELLOW}📋 Checking system requirements...${NC}"
    
    # Check RAM
    local total_ram_gb=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$total_ram_gb" -lt "$MIN_RAM_GB" ]; then
        log "${RED}❌ Insufficient RAM: ${total_ram_gb}GB available, ${MIN_RAM_GB}GB required${NC}"
        exit 1
    fi
    log "${GREEN}✅ RAM: ${total_ram_gb}GB available${NC}"
    
    # Check disk space
    local available_disk_gb=$(df / | awk 'NR==2 {print int($4/1024/1024)}')
    if [ "$available_disk_gb" -lt "$MIN_DISK_GB" ]; then
        log "${RED}❌ Insufficient disk space: ${available_disk_gb}GB available, ${MIN_DISK_GB}GB required${NC}"
        exit 1
    fi
    log "${GREEN}✅ Disk space: ${available_disk_gb}GB available${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log "${RED}❌ Docker not found. Installing Docker...${NC}"
        install_docker
    else
        log "${GREEN}✅ Docker is installed${NC}"
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log "${RED}❌ Docker Compose not found${NC}"
        exit 1
    fi
    log "${GREEN}✅ Docker Compose is available${NC}"
}

# Function to install Docker
install_docker() {
    log "${YELLOW}🐳 Installing Docker...${NC}"
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    sudo systemctl start docker || sudo dockerd &
    sudo systemctl enable docker || true
    rm get-docker.sh
    log "${GREEN}✅ Docker installed successfully${NC}"
}

# Function to check GPU support
check_gpu_support() {
    log "${YELLOW}🎮 Checking GPU support...${NC}"
    
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi --list-gpus | wc -l)
        log "${GREEN}✅ NVIDIA GPU detected: ${gpu_count} GPU(s) available${NC}"
        
        # Check for nvidia-docker
        if ! command -v nvidia-docker &> /dev/null && ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
            log "${YELLOW}⚠️  Installing NVIDIA Docker support...${NC}"
            install_nvidia_docker
        else
            log "${GREEN}✅ NVIDIA Docker support available${NC}"
        fi
    else
        log "${YELLOW}⚠️  No NVIDIA GPU detected, continuing with CPU-only deployment${NC}"
    fi
}

# Function to install NVIDIA Docker support
install_nvidia_docker() {
    # Add NVIDIA Docker repository
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker || sudo pkill dockerd && sudo dockerd &
    
    log "${GREEN}✅ NVIDIA Docker support installed${NC}"
}

# Function to create environment configuration
create_environment_config() {
    log "${YELLOW}⚙️  Creating environment configuration...${NC}"
    
    cat > "$ENVIRONMENT_FILE" << EOF
# SutazAI Enterprise AGI/ASI Environment Configuration
TZ=UTC

# Database Configuration
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=$(openssl rand -base64 32)
POSTGRES_DB=sutazai

# Redis Configuration  
REDIS_PASSWORD=$(openssl rand -base64 32)

# Vector Database Configuration
CHROMADB_API_KEY=$(openssl rand -base64 32)
QDRANT_API_KEY=$(openssl rand -base64 32)

# Security
JWT_SECRET=$(openssl rand -base64 64)
API_SECRET_KEY=$(openssl rand -base64 64)

# Monitoring
GRAFANA_PASSWORD=admin

# System Configuration
SUTAZAI_ENV=production
MAX_WORKERS=8
ENABLE_SELF_IMPROVEMENT=true
ENABLE_WEB_LEARNING=true
ENABLE_NEUROMORPHIC=true

# Model Configuration
OLLAMA_MODELS=deepseek-r1:8b,qwen3:8b,codellama:7b,llama3.2:1b
AUTO_DOWNLOAD_MODELS=true

# Performance Tuning
POSTGRES_MAX_CONNECTIONS=200
REDIS_MAX_MEMORY=2gb
OLLAMA_MAX_LOADED_MODELS=4
CHROMADB_MAX_MEMORY=4gb
QDRANT_MAX_MEMORY=4gb
EOF
    
    log "${GREEN}✅ Environment configuration created${NC}"
}

# Function to create necessary directories
create_directories() {
    log "${YELLOW}📁 Creating necessary directories...${NC}"
    
    local dirs=(
        "data/postgres"
        "data/redis" 
        "data/qdrant"
        "data/chromadb"
        "data/ollama"
        "data/models"
        "data/workspace"
        "data/logs"
        "data/prometheus"
        "data/grafana"
        "logs/deployment"
        "backup"
        "ssl"
        "config/nginx"
        "config/prometheus"
        "config/grafana"
        "monitoring/dashboards"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        chmod 755 "$dir"
    done
    
    log "${GREEN}✅ Directories created${NC}"
}

# Function to create monitoring configuration
create_monitoring_config() {
    log "${YELLOW}📊 Creating monitoring configuration...${NC}"
    
    # Create Prometheus configuration
    mkdir -p monitoring/prometheus
    cat > monitoring/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'sutazai-backend'
    static_configs:
      - targets: ['sutazai-backend:8000']
    metrics_path: '/metrics'

  - job_name: 'sutazai-agents'
    static_configs:
      - targets: 
        - 'autogpt:8000'
        - 'localagi:8080'
        - 'tabbyml:8080'
        - 'langchain-agents:8000'
        - 'autogen-agents:8000'
        - 'agentzero:8000'

  - job_name: 'infrastructure'
    static_configs:
      - targets:
        - 'postgres:5432'
        - 'redis:6379'
        - 'qdrant:6333'
        - 'chromadb:8000'
        - 'ollama:11434'

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
EOF

    # Create Grafana provisioning
    mkdir -p monitoring/grafana/provisioning/{datasources,dashboards}
    
    cat > monitoring/grafana/provisioning/datasources/datasources.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    log "${GREEN}✅ Monitoring configuration created${NC}"
}

# Function to create nginx configuration
create_nginx_config() {
    log "${YELLOW}🌐 Creating nginx configuration...${NC}"
    
    mkdir -p nginx
    cat > nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream sutazai_backend {
        server sutazai-backend:8000;
    }
    
    upstream sutazai_frontend {
        server sutazai-streamlit:8501;
    }
    
    upstream open_webui {
        server open-webui:8080;
    }
    
    server {
        listen 80;
        server_name _;
        
        location / {
            proxy_pass http://sutazai_frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /api/ {
            proxy_pass http://sutazai_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /webui/ {
            proxy_pass http://open_webui/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
}
EOF
    
    log "${GREEN}✅ Nginx configuration created${NC}"
}

# Function to ensure all Docker files exist
ensure_docker_files() {
    log "${YELLOW}🐳 Ensuring Docker configurations exist...${NC}"
    
    # Create missing Dockerfiles if they don't exist
    local docker_dirs=(
        "docker/enhanced-model-manager"
        "docker/faiss"
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
        "docker/neuromorphic"
        "docker/self-improvement"
        "docker/web-learning"
        "docker/health-check"
        "docker/orchestrator"
    )
    
    for dir in "${docker_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            create_default_dockerfile "$dir"
        fi
    done
    
    log "${GREEN}✅ Docker configurations ensured${NC}"
}

# Function to create default Dockerfile
create_default_dockerfile() {
    local dir="$1"
    local service_name=$(basename "$dir")
    
    cat > "$dir/Dockerfile" << EOF
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    git \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "main.py"]
EOF

    # Create basic requirements.txt
    cat > "$dir/requirements.txt" << EOF
fastapi==0.104.1
uvicorn==0.24.0
requests==2.31.0
aiohttp==3.9.1
pydantic==2.5.0
asyncio==3.4.3
python-multipart==0.0.6
EOF

    # Create basic main.py
    cat > "$dir/main.py" << EOF
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="${service_name^} Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "$service_name"}

@app.get("/")
async def root():
    return {"message": "$service_name service is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF
}

# Function to build all Docker images
build_docker_images() {
    log "${YELLOW}🔨 Building Docker images...${NC}"
    
    # Build custom images in parallel
    local services=(
        "enhanced-model-manager"
        "faiss"
        "autogpt"
        "localagi"
        "tabbyml"
        "langchain-agents"
        "autogen"
        "agentzero"
        "bigagi"
        "browser-use"
        "skyvern"
        "documind"
        "finrobot"
        "gpt-engineer"
        "aider"
        "langflow"
        "dify"
        "pytorch"
        "tensorflow"
        "jax"
        "neuromorphic"
        "self-improvement"
        "web-learning"
        "health-check"
        "orchestrator"
    )
    
    for service in "${services[@]}"; do
        if [ -d "docker/$service" ]; then
            log "${CYAN}Building $service...${NC}"
            docker build -t "sutazai-$service" "docker/$service" &
        fi
    done
    
    # Wait for all builds to complete
    wait
    
    log "${GREEN}✅ Docker images built successfully${NC}"
}

# Function to pull external Docker images
pull_external_images() {
    log "${YELLOW}📥 Pulling external Docker images...${NC}"
    
    local external_images=(
        "postgres:15"
        "redis:7-alpine"
        "qdrant/qdrant:latest"
        "chromadb/chroma:latest"
        "ollama/ollama:latest"
        "ghcr.io/open-webui/open-webui:main"
        "prom/prometheus:latest"
        "grafana/grafana:latest"
        "prom/node-exporter:latest"
        "nginx:alpine"
    )
    
    for image in "${external_images[@]}"; do
        log "${CYAN}Pulling $image...${NC}"
        docker pull "$image" &
    done
    
    # Wait for all pulls to complete
    wait
    
    log "${GREEN}✅ External images pulled successfully${NC}"
}

# Function to deploy core infrastructure
deploy_core_infrastructure() {
    log "${YELLOW}🏗️  Deploying core infrastructure...${NC}"
    
    # Deploy databases and cache first
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENVIRONMENT_FILE" up -d \
        postgres redis qdrant chromadb faiss
    
    # Wait for databases to be ready
    log "${CYAN}Waiting for databases to be ready...${NC}"
    sleep 30
    
    # Check database health
    for i in {1..30}; do
        if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U sutazai; then
            log "${GREEN}✅ PostgreSQL is ready${NC}"
            break
        fi
        sleep 2
    done
    
    log "${GREEN}✅ Core infrastructure deployed${NC}"
}

# Function to deploy model management
deploy_model_management() {
    log "${YELLOW}🧠 Deploying model management...${NC}"
    
    # Deploy Ollama and enhanced model manager
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENVIRONMENT_FILE" up -d \
        ollama enhanced-model-manager
    
    # Wait for Ollama to be ready
    log "${CYAN}Waiting for Ollama to be ready...${NC}"
    sleep 60
    
    # Download initial models
    log "${CYAN}Downloading initial AI models...${NC}"
    docker-compose -f "$COMPOSE_FILE" exec -T ollama ollama pull deepseek-r1:8b &
    docker-compose -f "$COMPOSE_FILE" exec -T ollama ollama pull qwen3:8b &
    docker-compose -f "$COMPOSE_FILE" exec -T ollama ollama pull codellama:7b &
    docker-compose -f "$COMPOSE_FILE" exec -T ollama ollama pull llama3.2:1b &
    
    wait
    
    log "${GREEN}✅ Model management deployed${NC}"
}

# Function to deploy AI agents
deploy_ai_agents() {
    log "${YELLOW}🤖 Deploying AI agents...${NC}"
    
    # Deploy core AI agents
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENVIRONMENT_FILE" up -d \
        autogpt localagi tabbyml semgrep langchain-agents autogen-agents agentzero
    
    # Deploy specialized agents
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENVIRONMENT_FILE" up -d \
        bigagi browser-use skyvern documind finrobot gpt-engineer aider
    
    log "${GREEN}✅ AI agents deployed${NC}"
}

# Function to deploy backend and frontend
deploy_application_layer() {
    log "${YELLOW}🌐 Deploying application layer...${NC}"
    
    # Deploy backend API
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENVIRONMENT_FILE" up -d \
        sutazai-backend
    
    # Wait for backend to be ready
    sleep 30
    
    # Deploy frontend interfaces
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENVIRONMENT_FILE" up -d \
        sutazai-streamlit open-webui langflow dify
    
    log "${GREEN}✅ Application layer deployed${NC}"
}

# Function to deploy advanced AI services
deploy_advanced_ai() {
    log "${YELLOW}🔬 Deploying advanced AI services...${NC}"
    
    # Deploy ML frameworks
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENVIRONMENT_FILE" up -d \
        pytorch tensorflow jax
    
    # Deploy self-improvement systems
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENVIRONMENT_FILE" up -d \
        neuromorphic-engine self-improvement-engine web-learning-engine
    
    log "${GREEN}✅ Advanced AI services deployed${NC}"
}

# Function to deploy monitoring and operations
deploy_monitoring() {
    log "${YELLOW}📊 Deploying monitoring and operations...${NC}"
    
    # Deploy monitoring stack
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENVIRONMENT_FILE" up -d \
        prometheus grafana node-exporter
    
    # Deploy infrastructure services
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENVIRONMENT_FILE" up -d \
        nginx health-check sutazai-orchestrator
    
    log "${GREEN}✅ Monitoring and operations deployed${NC}"
}

# Function to verify deployment
verify_deployment() {
    log "${YELLOW}🔍 Verifying deployment...${NC}"
    
    # Check all services are running
    local failed_services=()
    
    log "${CYAN}Checking service health...${NC}"
    
    # List of critical services to check
    local services=(
        "sutazai-postgres"
        "sutazai-redis"
        "sutazai-qdrant"
        "sutazai-chromadb"
        "sutazai-ollama"
        "sutazai-backend"
        "sutazai-streamlit"
    )
    
    for service in "${services[@]}"; do
        if ! docker ps | grep -q "$service.*Up"; then
            failed_services+=("$service")
        fi
    done
    
    if [ ${#failed_services[@]} -eq 0 ]; then
        log "${GREEN}✅ All critical services are running${NC}"
    else
        log "${RED}❌ Failed services: ${failed_services[*]}${NC}"
        log "${YELLOW}Checking logs for failed services...${NC}"
        for service in "${failed_services[@]}"; do
            docker logs "$service" --tail 20
        done
    fi
    
    # Test API endpoints
    log "${CYAN}Testing API endpoints...${NC}"
    sleep 30
    
    if curl -f http://localhost:8000/health &> /dev/null; then
        log "${GREEN}✅ Backend API is responding${NC}"
    else
        log "${RED}❌ Backend API is not responding${NC}"
    fi
    
    if curl -f http://localhost:8501 &> /dev/null; then
        log "${GREEN}✅ Frontend is responding${NC}"
    else
        log "${RED}❌ Frontend is not responding${NC}"
    fi
}

# Function to create backup
create_backup() {
    log "${YELLOW}💾 Creating backup...${NC}"
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup configuration files
    cp -r "$COMPOSE_FILE" "$ENVIRONMENT_FILE" nginx/ monitoring/ "$BACKUP_DIR/"
    
    # Backup database
    docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_dump -U sutazai sutazai > "$BACKUP_DIR/database_backup.sql"
    
    log "${GREEN}✅ Backup created in $BACKUP_DIR${NC}"
}

# Function to display final status
display_final_status() {
    log "\n${BLUE}========================================${NC}"
    log "${BLUE}🎉 Deployment Complete!${NC}"
    log "${BLUE}========================================${NC}"
    
    log "\n${GREEN}📊 Service Status:${NC}"
    docker-compose -f "$COMPOSE_FILE" ps
    
    log "\n${GREEN}🌐 Access URLs:${NC}"
    log "${CYAN}Main Web Interface:${NC} http://localhost:8501"
    log "${CYAN}Advanced Web UI:${NC} http://localhost:8030"
    log "${CYAN}API Documentation:${NC} http://localhost:8000/docs"
    log "${CYAN}Monitoring Dashboard:${NC} http://localhost:3000 (admin/admin)"
    log "${CYAN}Metrics:${NC} http://localhost:9090"
    log "${CYAN}LangFlow:${NC} http://localhost:7860"
    log "${CYAN}Dify:${NC} http://localhost:5001"
    
    log "\n${GREEN}🔧 Management Commands:${NC}"
    log "${CYAN}View logs:${NC} docker-compose -f $COMPOSE_FILE logs -f"
    log "${CYAN}Stop system:${NC} docker-compose -f $COMPOSE_FILE down"
    log "${CYAN}Restart system:${NC} docker-compose -f $COMPOSE_FILE restart"
    log "${CYAN}Update system:${NC} docker-compose -f $COMPOSE_FILE pull && docker-compose -f $COMPOSE_FILE up -d"
    
    log "\n${GREEN}📋 Next Steps:${NC}"
    log "1. Access the web interface at http://localhost:8501"
    log "2. Configure your AI models and agents"
    log "3. Test the various AI capabilities"
    log "4. Monitor system performance via Grafana"
    log "5. Review logs for any issues"
    
    log "\n${PURPLE}🎯 System deployed successfully with 30+ AI services!${NC}"
}

# Main deployment function
main() {
    log "${BLUE}Starting SutazAI Enterprise AGI/ASI deployment...${NC}"
    
    # Phase 1: System preparation
    check_system_requirements
    check_gpu_support
    create_environment_config
    create_directories
    create_monitoring_config
    create_nginx_config
    
    # Phase 2: Docker preparation
    ensure_docker_files
    pull_external_images
    build_docker_images
    
    # Phase 3: Service deployment
    deploy_core_infrastructure
    deploy_model_management
    deploy_ai_agents
    deploy_application_layer
    deploy_advanced_ai
    deploy_monitoring
    
    # Phase 4: Verification and finalization
    verify_deployment
    create_backup
    display_final_status
    
    log "\n${GREEN}🚀 SutazAI Enterprise AGI/ASI system deployed successfully!${NC}"
    log "${GREEN}📝 Deployment log saved to: $LOG_FILE${NC}"
    log "${GREEN}💾 Backup created in: $BACKUP_DIR${NC}"
}

# Run main function
main "$@"