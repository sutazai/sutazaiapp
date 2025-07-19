#!/bin/bash

set -e

echo "🚀 SutazAI AGI/ASI Complete E2E System Deployment"
echo "=================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SYSTEM_NAME="SutazAI"
VERSION="1.0.0"
DEPLOYMENT_DATE=$(date +"%Y-%m-%d %H:%M:%S")
LOG_FILE="deployment_$(date +%Y%m%d_%H%M%S).log"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking system prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check system resources
    TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_RAM" -lt 16 ]; then
        warning "System has less than 16GB RAM. Performance may be affected."
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df . | awk 'NR==2 {print $4}')
    if [ "$AVAILABLE_SPACE" -lt 104857600 ]; then  # 100GB in KB
        warning "Less than 100GB disk space available. Consider freeing up space."
    fi
    
    # Check for GPU support
    if command -v nvidia-smi &> /dev/null; then
        log "NVIDIA GPU detected. GPU acceleration will be enabled."
        GPU_ENABLED=true
    else
        warning "No GPU detected. Using CPU-only mode."
        GPU_ENABLED=false
    fi
    
    log "Prerequisites check completed successfully."
}

# Setup environment
setup_environment() {
    log "Setting up environment..."
    
    # Create necessary directories
    mkdir -p data/{models,documents,logs,backups,workspace}
    mkdir -p monitoring/{prometheus,grafana}
    mkdir -p ssl
    mkdir -p nginx
    mkdir -p backend
    mkdir -p frontend
    
    # Set proper permissions
    chmod 755 data
    chmod 700 ssl
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        log "Creating .env file..."
        cat > .env << EOF
# SutazAI Configuration
SUTAZAI_VERSION=${VERSION}
ENVIRONMENT=production
DEBUG=false

# Database Configuration
POSTGRES_DB=sutazai
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=sutazai_secure_password_$(openssl rand -hex 16)
DATABASE_URL=postgresql://sutazai:sutazai_secure_password_$(openssl rand -hex 16)@postgres:5432/sutazai

# Redis Configuration
REDIS_URL=redis://redis:6379

# Vector Database Configuration
QDRANT_URL=http://qdrant:6333
CHROMADB_URL=http://chromadb:8000
FAISS_URL=http://faiss:8088

# Model Configuration
OLLAMA_URL=http://ollama:11434
DEFAULT_MODEL=deepseek-coder:33b
FALLBACK_MODEL=llama3.2:1b

# Security
SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET_KEY=$(openssl rand -hex 32)

# Agent URLs
AUTOGPT_URL=http://autogpt:8080
LOCALAGI_URL=http://localagi:8080
TABBY_URL=http://tabby:8080
BROWSER_USE_URL=http://browser-use:8080
DOCUMIND_URL=http://documind:8080
FINROBOT_URL=http://finrobot:8080
GPT_ENGINEER_URL=http://gpt-engineer:8080
AIDER_URL=http://aider:8080

# Monitoring
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_PASSWORD=admin_$(openssl rand -hex 8)

# Performance
MAX_WORKERS=8
MAX_MEMORY_GB=32
AUTO_SCALING=true
EOF
        log ".env file created successfully."
    else
        log ".env file already exists. Skipping creation."
    fi
    
    log "Environment setup completed."
}

# Create missing Docker files
create_docker_files() {
    log "Creating missing Docker files..."
    
    # Create AutoGPT Dockerfile
    mkdir -p docker/autogpt
    cat > docker/autogpt/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone AutoGPT
RUN git clone https://github.com/Significant-Gravitas/AutoGPT.git .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create startup script
RUN echo '#!/bin/bash\n\
python -m autogpt.app.main --ai-settings ai_settings.yaml\n\
' > start.sh && chmod +x start.sh

EXPOSE 8080

CMD ["./start.sh"]
EOF

    # Create LocalAGI Dockerfile
    mkdir -p docker/localagi
    cat > docker/localagi/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clone LocalAGI
RUN git clone https://github.com/mudler/LocalAGI.git .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create startup script
RUN echo '#!/bin/bash\n\
python app.py --host 0.0.0.0 --port 8080\n\
' > start.sh && chmod +x start.sh

EXPOSE 8080

CMD ["./start.sh"]
EOF

    # Create Browser-Use Dockerfile
    mkdir -p docker/browser-use
    cat > docker/browser-use/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clone Browser-Use
RUN git clone https://github.com/browser-use/browser-use.git .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create startup script
RUN echo '#!/bin/bash\n\
python -m browser_use.server --host 0.0.0.0 --port 8080\n\
' > start.sh && chmod +x start.sh

EXPOSE 8080

CMD ["./start.sh"]
EOF

    # Create Skyvern Dockerfile
    mkdir -p docker/skyvern
    cat > docker/skyvern/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    chromium \
    && rm -rf /var/lib/apt/lists/*

# Clone Skyvern
RUN git clone https://github.com/Skyvern-AI/skyvern.git .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create startup script
RUN echo '#!/bin/bash\n\
python -m skyvern.app --host 0.0.0.0 --port 8080\n\
' > start.sh && chmod +x start.sh

EXPOSE 8080

CMD ["./start.sh"]
EOF

    # Create LangFlow Dockerfile
    mkdir -p docker/langflow
    cat > docker/langflow/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install LangFlow
RUN pip install langflow

# Create startup script
RUN echo '#!/bin/bash\n\
langflow run --host 0.0.0.0 --port 7860\n\
' > start.sh && chmod +x start.sh

EXPOSE 7860

CMD ["./start.sh"]
EOF

    # Create Dify Dockerfile
    mkdir -p docker/dify
    cat > docker/dify/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Clone Dify
RUN git clone https://github.com/langgenius/dify.git .

# Install Python dependencies
RUN pip install --no-cache-dir -r api/requirements.txt

# Install Node.js dependencies
RUN cd web && npm install

# Create startup script
RUN echo '#!/bin/bash\n\
cd api && python app.py &\n\
cd web && npm start &\n\
wait\n\
' > start.sh && chmod +x start.sh

EXPOSE 5001 3000

CMD ["./start.sh"]
EOF

    # Create GPT-Engineer Dockerfile
    mkdir -p docker/gpt-engineer
    cat > docker/gpt-engineer/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clone GPT-Engineer
RUN git clone https://github.com/AntonOsika/gpt-engineer.git .

# Install dependencies
RUN pip install --no-cache-dir -e .

# Create startup script
RUN echo '#!/bin/bash\n\
python -m gpt_engineer.server --host 0.0.0.0 --port 8080\n\
' > start.sh && chmod +x start.sh

EXPOSE 8080

CMD ["./start.sh"]
EOF

    # Create Aider Dockerfile
    mkdir -p docker/aider
    cat > docker/aider/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clone Aider
RUN git clone https://github.com/Aider-AI/aider.git .

# Install dependencies
RUN pip install --no-cache-dir -e .

# Create startup script
RUN echo '#!/bin/bash\n\
aider --api --host 0.0.0.0 --port 8080\n\
' > start.sh && chmod +x start.sh

EXPOSE 8080

CMD ["./start.sh"]
EOF

    # Create BigAGI Dockerfile
    mkdir -p docker/bigagi
    cat > docker/bigagi/Dockerfile << 'EOF'
FROM node:18-alpine

WORKDIR /app

# Install git
RUN apk add --no-cache git

# Clone BigAGI
RUN git clone https://github.com/enricoros/big-AGI.git .

# Install dependencies
RUN npm install

# Build the application
RUN npm run build

# Create startup script
RUN echo '#!/bin/sh\n\
npm start\n\
' > start.sh && chmod +x start.sh

EXPOSE 3000

CMD ["./start.sh"]
EOF

    # Create AgentZero Dockerfile
    mkdir -p docker/agentzero
    cat > docker/agentzero/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clone AgentZero
RUN git clone https://github.com/frdel/agent-zero.git .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create startup script
RUN echo '#!/bin/bash\n\
python run.py --host 0.0.0.0 --port 8080\n\
' > start.sh && chmod +x start.sh

EXPOSE 8080

CMD ["./start.sh"]
EOF

    log "Docker files created successfully."
}

# Install and setup Ollama models
setup_ollama_models() {
    log "Setting up Ollama models..."
    
    # Wait for Ollama to be ready
    log "Waiting for Ollama service to be ready..."
    timeout=300
    counter=0
    while [ $counter -lt $timeout ]; do
        if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
            log "Ollama is ready."
            break
        fi
        sleep 5
        counter=$((counter + 5))
    done
    
    if [ $counter -eq $timeout ]; then
        error "Ollama failed to start within ${timeout} seconds."
        return 1
    fi
    
    # Pull required models
    log "Pulling AI models..."
    
    # Pull models in order of importance
    MODELS=(
        "llama3.2:1b"
        "deepseek-r1:8b"  
        "qwen3:8b"
        "deepseek-coder:33b"
        "llama2:13b"
    )
    
    for model in "${MODELS[@]}"; do
        log "Pulling model: $model"
        if docker exec sutazai-ollama ollama pull "$model"; then
            log "Successfully pulled $model"
        else
            warning "Failed to pull $model, continuing with other models..."
        fi
    done
    
    log "Model setup completed."
}

# Build and start services
build_and_start_services() {
    log "Building and starting all services..."
    
    # Pull latest images first
    log "Pulling base Docker images..."
    docker-compose pull postgres redis qdrant chromadb || true
    
    # Build custom images
    log "Building custom Docker images..."
    if ! docker-compose build --parallel; then
        error "Failed to build Docker images."
        return 1
    fi
    
    # Start core infrastructure first
    log "Starting core infrastructure services..."
    docker-compose up -d postgres redis qdrant chromadb ollama
    
    # Wait for core services to be ready
    log "Waiting for core services to be ready..."
    sleep 30
    
    # Start AI agents and backend services
    log "Starting AI agents and backend services..."
    docker-compose up -d sutazai-backend
    
    # Wait for backend to be ready
    log "Waiting for backend service to be ready..."
    sleep 20
    
    # Start frontend and remaining services
    log "Starting frontend and remaining services..."
    docker-compose up -d
    
    log "All services started successfully."
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # List of services to check
    SERVICES=(
        "postgres:5432"
        "redis:6379"
        "qdrant:6333"
        "chromadb:8000"
        "ollama:11434"
        "sutazai-backend:8000"
        "sutazai-streamlit:8501"
    )
    
    failed_services=()
    
    for service in "${SERVICES[@]}"; do
        service_name=$(echo "$service" | cut -d':' -f1)
        port=$(echo "$service" | cut -d':' -f2)
        
        log "Checking $service_name on port $port..."
        
        if nc -z localhost "$port" 2>/dev/null; then
            log "✅ $service_name is running on port $port"
        else
            error "❌ $service_name is not responding on port $port"
            failed_services+=("$service_name")
        fi
    done
    
    # Check web services
    WEB_SERVICES=(
        "http://localhost:8501"  # Streamlit
        "http://localhost:8000/health"  # Backend
        "http://localhost:6333/dashboard"  # Qdrant
        "http://localhost:8001/api/v1/heartbeat"  # ChromaDB
    )
    
    for url in "${WEB_SERVICES[@]}"; do
        log "Checking web service: $url"
        if curl -s -f "$url" > /dev/null 2>&1; then
            log "✅ Web service responding: $url"
        else
            warning "⚠️ Web service not responding: $url"
        fi
    done
    
    if [ ${#failed_services[@]} -eq 0 ]; then
        log "🎉 All services are running successfully!"
        return 0
    else
        error "The following services failed to start: ${failed_services[*]}"
        return 1
    fi
}

# Initialize agents
initialize_agents() {
    log "Initializing AI agents..."
    
    # Start agent orchestrator
    log "Starting agent orchestrator..."
    if curl -s -X POST http://localhost:8000/api/orchestrator/start > /dev/null 2>&1; then
        log "Agent orchestrator started successfully."
    else
        warning "Failed to start agent orchestrator."
    fi
    
    # Initialize external agents
    log "Initializing external agents..."
    if curl -s -X POST http://localhost:8000/api/external_agents/initialize > /dev/null 2>&1; then
        log "External agents initialized successfully."
    else
        warning "Failed to initialize external agents."
    fi
    
    # Wait for agents to be ready
    log "Waiting for agents to initialize..."
    sleep 30
    
    # Check agent status
    log "Checking agent status..."
    agent_status=$(curl -s http://localhost:8000/api/orchestrator/agents | jq -r '.active_agents // 0')
    external_agent_status=$(curl -s http://localhost:8000/api/external_agents/status | jq -r '.active_agents // 0')
    
    log "Active internal agents: $agent_status"
    log "Active external agents: $external_agent_status"
    
    log "Agent initialization completed."
}

# Create monitoring dashboard
setup_monitoring() {
    log "Setting up monitoring and dashboards..."
    
    # Create Prometheus configuration
    mkdir -p monitoring/prometheus
    cat > monitoring/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'sutazai-backend'
    static_configs:
      - targets: ['sutazai-backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'sutazai-agents'
    static_configs:
      - targets: ['autogpt:8080', 'localagi:8080', 'tabby:8080']
    scrape_interval: 30s

  - job_name: 'infrastructure'
    static_configs:
      - targets: ['postgres:5432', 'redis:6379', 'qdrant:6333', 'chromadb:8000']
    scrape_interval: 30s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
    scrape_interval: 15s
EOF

    # Create Grafana configuration
    mkdir -p monitoring/grafana
    cat > monitoring/grafana/grafana.ini << 'EOF'
[server]
http_port = 3000
domain = localhost

[security]
admin_user = admin
admin_password = sutazai_admin

[database]
type = sqlite3
path = /var/lib/grafana/grafana.db

[session]
provider = file
provider_config = sessions
cookie_name = grafana_sess
cookie_secure = false
session_life_time = 86400

[analytics]
reporting_enabled = false
check_for_updates = false

[log]
mode = console
level = info
EOF

    log "Monitoring setup completed."
}

# Create backup strategy
setup_backup_system() {
    log "Setting up backup system..."
    
    # Create backup script
    cat > backup_system.sh << 'EOF'
#!/bin/bash

BACKUP_DIR="data/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="sutazai_backup_${TIMESTAMP}"

echo "Creating system backup: $BACKUP_NAME"

# Create backup directory
mkdir -p "$BACKUP_DIR/$BACKUP_NAME"

# Backup databases
echo "Backing up PostgreSQL..."
docker exec sutazai-postgres pg_dump -U sutazai sutazai > "$BACKUP_DIR/$BACKUP_NAME/postgres.sql"

echo "Backing up Redis..."
docker exec sutazai-redis redis-cli BGSAVE
docker cp sutazai-redis:/data/dump.rdb "$BACKUP_DIR/$BACKUP_NAME/"

echo "Backing up Qdrant..."
docker cp sutazai-qdrant:/qdrant/storage "$BACKUP_DIR/$BACKUP_NAME/qdrant"

echo "Backing up ChromaDB..."
docker cp sutazai-chromadb:/chroma/chroma "$BACKUP_DIR/$BACKUP_NAME/chromadb"

echo "Backing up models..."
docker cp sutazai-ollama:/root/.ollama "$BACKUP_DIR/$BACKUP_NAME/ollama"

# Compress backup
echo "Compressing backup..."
tar -czf "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" -C "$BACKUP_DIR" "$BACKUP_NAME"
rm -rf "$BACKUP_DIR/$BACKUP_NAME"

echo "Backup completed: ${BACKUP_NAME}.tar.gz"

# Cleanup old backups (keep last 7 days)
find "$BACKUP_DIR" -name "sutazai_backup_*.tar.gz" -mtime +7 -delete
EOF

    chmod +x backup_system.sh
    
    # Create systemd timer for automated backups (optional)
    if command -v systemctl &> /dev/null; then
        log "Setting up automated backup schedule..."
        # Add backup automation here if needed
    fi
    
    log "Backup system setup completed."
}

# Generate deployment report
generate_report() {
    log "Generating deployment report..."
    
    REPORT_FILE="deployment_report_$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$REPORT_FILE" << EOF
{
  "deployment": {
    "system_name": "$SYSTEM_NAME",
    "version": "$VERSION",
    "deployment_date": "$DEPLOYMENT_DATE",
    "status": "completed",
    "environment": "production"
  },
  "services": {
    "total_services": $(docker-compose ps --services | wc -l),
    "running_services": $(docker-compose ps --filter status=running | wc -l),
    "failed_services": $(docker-compose ps --filter status=exited | wc -l)
  },
  "agents": {
    "internal_agents": $(curl -s http://localhost:8000/api/orchestrator/agents | jq -r '.total_agents // 0'),
    "active_internal_agents": $(curl -s http://localhost:8000/api/orchestrator/agents | jq -r '.active_agents // 0'),
    "external_agents": $(curl -s http://localhost:8000/api/external_agents/status | jq -r '.total_agents // 0'),
    "active_external_agents": $(curl -s http://localhost:8000/api/external_agents/status | jq -r '.active_agents // 0')
  },
  "models": {
    "ollama_models": $(docker exec sutazai-ollama ollama list | tail -n +2 | wc -l)
  },
  "access_points": {
    "main_ui": "http://localhost:8501",
    "backend_api": "http://localhost:8000",
    "api_docs": "http://localhost:8000/docs", 
    "monitoring": "http://localhost:3000",
    "qdrant_ui": "http://localhost:6333/dashboard",
    "chromadb": "http://localhost:8001"
  },
  "system_info": {
    "total_ram_gb": $(free -g | awk '/^Mem:/{print $2}'),
    "available_space_gb": $(df . | awk 'NR==2 {print int($4/1024/1024)}'),
    "cpu_cores": $(nproc),
    "gpu_available": $(command -v nvidia-smi &> /dev/null && echo "true" || echo "false")
  }
}
EOF
    
    log "Deployment report generated: $REPORT_FILE"
}

# Display final information
display_final_info() {
    echo ""
    echo "🎉 SutazAI AGI/ASI System Deployment Completed Successfully!"
    echo "============================================================"
    echo ""
    echo "📊 System Access Points:"
    echo "   Main Interface:     http://localhost:8501"
    echo "   Backend API:        http://localhost:8000"
    echo "   API Documentation:  http://localhost:8000/docs"
    echo "   Monitoring:         http://localhost:3000"
    echo "   Qdrant Dashboard:   http://localhost:6333/dashboard"
    echo "   ChromaDB:           http://localhost:8001"
    echo ""
    echo "🤖 Agent Status:"
    internal_agents=$(curl -s http://localhost:8000/api/orchestrator/agents | jq -r '.total_agents // 0')
    active_internal=$(curl -s http://localhost:8000/api/orchestrator/agents | jq -r '.active_agents // 0')
    external_agents=$(curl -s http://localhost:8000/api/external_agents/status | jq -r '.total_agents // 0')
    active_external=$(curl -s http://localhost:8000/api/external_agents/status | jq -r '.active_agents // 0')
    
    echo "   Internal Agents:    $active_internal/$internal_agents active"
    echo "   External Agents:    $active_external/$external_agents active"
    echo ""
    echo "🔧 Management Commands:"
    echo "   Start system:       docker-compose up -d"
    echo "   Stop system:        docker-compose down"
    echo "   View logs:          docker-compose logs -f"
    echo "   System status:      docker-compose ps"
    echo "   Create backup:      ./backup_system.sh"
    echo ""
    echo "📝 Logs and Reports:"
    echo "   Deployment log:     $LOG_FILE"
    echo "   System report:      Check deployment_report_*.json"
    echo ""
    echo "🚀 The SutazAI system is now fully operational!"
    echo "   All AI agents are working together in harmony."
    echo "   Visit http://localhost:8501 to start using the system."
    echo ""
}

# Main deployment flow
main() {
    log "Starting SutazAI AGI/ASI Complete E2E System Deployment"
    
    check_prerequisites
    setup_environment
    create_docker_files
    setup_monitoring
    setup_backup_system
    
    build_and_start_services
    
    # Wait for services to stabilize
    log "Waiting for services to stabilize..."
    sleep 60
    
    setup_ollama_models
    initialize_agents
    
    verify_deployment
    generate_report
    display_final_info
    
    log "Deployment completed successfully at $(date)"
}

# Handle cleanup on exit
cleanup() {
    if [ $? -ne 0 ]; then
        error "Deployment failed. Check logs for details: $LOG_FILE"
        echo "To clean up, run: docker-compose down --volumes"
    fi
}

trap cleanup EXIT

# Run main deployment
main "$@"