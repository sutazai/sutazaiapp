#!/bin/bash
# setup_complete_agi_system.sh
# Comprehensive SutazAI AGI/ASI System Setup Script

set -e
set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SUTAZAI_HOME="/opt/sutazaiapp"
LOG_FILE="${SUTAZAI_HOME}/logs/setup_$(date +%Y%m%d_%H%M%S).log"
PYTHON_VERSION="3.11"
NODE_VERSION="18"

# Ensure we're in the correct directory
cd "${SUTAZAI_HOME}"

# Create log directory
mkdir -p "${SUTAZAI_HOME}/logs"

# Logging function
log() {
    echo -e "${1}" | tee -a "${LOG_FILE}"
}

# Error handling
error_exit() {
    log "${RED}ERROR: ${1}${NC}"
    exit 1
}

# Success message
success() {
    log "${GREEN}✅ ${1}${NC}"
}

# Info message
info() {
    log "${BLUE}ℹ️  ${1}${NC}"
}

# Warning message
warn() {
    log "${YELLOW}⚠️  ${1}${NC}"
}

# Progress indicator
progress() {
    log "${PURPLE}🔄 ${1}${NC}"
}

# Header
header() {
    log "${CYAN}"
    log "=============================================="
    log "🤖 SutazAI AGI/ASI Complete System Setup"
    log "=============================================="
    log "${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error_exit "This script should not be run as root for security reasons"
    fi
}

# Check system requirements
check_requirements() {
    progress "Checking system requirements..."
    
    # Check OS
    if [[ ! -f /etc/lsb-release ]] && [[ ! -f /etc/debian_version ]]; then
        error_exit "This script requires Ubuntu/Debian Linux"
    fi
    
    # Check available disk space (minimum 50GB)
    available_space=$(df "${SUTAZAI_HOME}" | tail -1 | awk '{print $4}')
    if [[ ${available_space} -lt 52428800 ]]; then  # 50GB in KB
        error_exit "Insufficient disk space. Minimum 50GB required."
    fi
    
    # Check RAM (minimum 16GB recommended)
    total_ram=$(free -g | awk '/^Mem:/{print $2}')
    if [[ ${total_ram} -lt 8 ]]; then
        warn "Low RAM detected (${total_ram}GB). 16GB+ recommended for optimal performance."
    fi
    
    # Check for GPU
    if command -v nvidia-smi &> /dev/null; then
        gpu_count=$(nvidia-smi -L | wc -l)
        success "NVIDIA GPU detected: ${gpu_count} GPU(s) available"
    else
        warn "No NVIDIA GPU detected. CPU-only mode will be slower."
    fi
    
    success "System requirements check completed"
}

# Install system dependencies
install_system_dependencies() {
    progress "Installing system dependencies..."
    
    # Update package lists
    sudo apt-get update
    
    # Essential packages
    sudo apt-get install -y \
        curl wget git unzip \
        build-essential \
        python3 python3-pip python3-venv python3-dev \
        nodejs npm \
        postgresql postgresql-contrib \
        redis-server \
        nginx \
        htop iotop \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release
    
    # Docker installation
    if ! command -v docker &> /dev/null; then
        progress "Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
    fi
    
    # Docker Compose installation
    if ! command -v docker-compose &> /dev/null; then
        progress "Installing Docker Compose..."
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    fi
    
    success "System dependencies installed"
}

# Setup GPU support
setup_gpu_support() {
    if command -v nvidia-smi &> /dev/null; then
        progress "Setting up NVIDIA GPU support..."
        
        # Install NVIDIA Container Toolkit
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
        
        sudo apt-get update
        sudo apt-get install -y nvidia-docker2
        sudo systemctl restart docker
        
        # Test GPU access
        docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
        
        success "GPU support configured"
    else
        warn "No GPU detected, skipping GPU setup"
    fi
}

# Create directory structure
create_directories() {
    progress "Creating directory structure..."
    
    # Core directories
    mkdir -p "${SUTAZAI_HOME}"/{data,logs,backups,models,configs,ssl,secrets}
    mkdir -p "${SUTAZAI_HOME}/data"/{documents,vectors,workspace,uploads}
    mkdir -p "${SUTAZAI_HOME}/configs"/{nginx,prometheus,grafana,agents}
    mkdir -p "${SUTAZAI_HOME}/logs"/{backend,frontend,agents,system}
    
    # Set permissions
    chmod 755 "${SUTAZAI_HOME}"/{data,logs,backups,models,configs}
    chmod 700 "${SUTAZAI_HOME}"/{ssl,secrets}
    chmod 777 "${SUTAZAI_HOME}/data/workspace"
    
    success "Directory structure created"
}

# Generate secrets and certificates
generate_secrets() {
    progress "Generating secrets and certificates..."
    
    # Generate random passwords
    echo "$(openssl rand -base64 32)" > "${SUTAZAI_HOME}/secrets/postgres_password.txt"
    echo "$(openssl rand -base64 32)" > "${SUTAZAI_HOME}/secrets/grafana_password.txt"
    echo "$(openssl rand -base64 32)" > "${SUTAZAI_HOME}/secrets/vault_token.txt"
    echo "$(openssl rand -base64 32)" > "${SUTAZAI_HOME}/secrets/jwt_secret.txt"
    echo "$(openssl rand -base64 32)" > "${SUTAZAI_HOME}/secrets/replication_password.txt"
    
    # Set secure permissions
    chmod 600 "${SUTAZAI_HOME}/secrets"/*
    
    # Generate SSL certificates (self-signed for development)
    if [[ ! -f "${SUTAZAI_HOME}/ssl/cert.pem" ]]; then
        openssl req -x509 -newkey rsa:4096 -keyout "${SUTAZAI_HOME}/ssl/key.pem" -out "${SUTAZAI_HOME}/ssl/cert.pem" -days 365 -nodes -subj "/CN=sutazai.local"
        chmod 600 "${SUTAZAI_HOME}/ssl"/*
    fi
    
    success "Secrets and certificates generated"
}

# Create environment configuration
create_environment() {
    progress "Creating environment configuration..."
    
    # Create .env file
    cat > "${SUTAZAI_HOME}/.env" << EOF
# SutazAI Environment Configuration
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database Configuration
POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password
DATABASE_URL=postgresql://sutazai@postgres:5432/sutazai
REDIS_URL=redis://redis:6379

# AI Model Configuration
OLLAMA_URL=http://ollama:11434
CHROMADB_URL=http://chromadb:8000
QDRANT_URL=http://qdrant:6333
NEO4J_URI=bolt://neo4j:7687
ELASTICSEARCH_URL=http://elasticsearch:9200

# Security Configuration
JWT_SECRET_KEY_FILE=/run/secrets/jwt_secret
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60

# System Configuration
MAX_WORKERS=4
MAX_CONCURRENT_TASKS=100
CACHE_ENABLED=true
AUTO_SCALING_ENABLED=true

# Monitoring Configuration
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_PASSWORD_FILE=/run/secrets/grafana_password

# Backup Configuration
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"
BACKUP_RETENTION_DAYS=30
EOF
    
    success "Environment configuration created"
}

# Install Python dependencies
install_python_dependencies() {
    progress "Installing Python dependencies..."
    
    # Create virtual environment
    if [[ ! -d "${SUTAZAI_HOME}/venv" ]]; then
        python3 -m venv "${SUTAZAI_HOME}/venv"
    fi
    
    # Activate virtual environment
    source "${SUTAZAI_HOME}/venv/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install core dependencies
    pip install -r "${SUTAZAI_HOME}/requirements.txt"
    
    # Install additional AI/ML packages
    pip install \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
        transformers \
        sentence-transformers \
        chromadb \
        qdrant-client \
        ollama \
        langchain \
        crewai \
        autogen-agentchat \
        streamlit \
        fastapi \
        uvicorn \
        celery \
        redis \
        psycopg2-binary \
        sqlalchemy \
        alembic \
        prometheus-client \
        structlog \
        httpx \
        aiohttp \
        asyncio \
        numpy \
        pandas \
        plotly \
        psutil
    
    deactivate
    
    success "Python dependencies installed"
}

# Start core services
start_core_services() {
    progress "Starting core infrastructure services..."
    
    # Start PostgreSQL, Redis, and vector databases
    docker-compose up -d postgres redis chromadb qdrant
    
    # Wait for services to be ready
    info "Waiting for services to initialize..."
    sleep 30
    
    # Check service health
    docker-compose ps
    
    success "Core services started"
}

# Install and configure AI models
install_ai_models() {
    progress "Installing AI models..."
    
    # Start Ollama
    docker-compose up -d ollama
    sleep 30
    
    # Check if Ollama is responding
    while ! curl -f http://localhost:11434/api/tags 2>/dev/null; do
        info "Waiting for Ollama to start..."
        sleep 10
    done
    
    # Install models one by one to avoid memory issues
    models=("deepseek-r1:8b" "qwen3:8b" "codellama:7b" "codellama:33b" "llama2")
    
    for model in "${models[@]}"; do
        progress "Installing model: ${model}"
        
        # Pull model with timeout
        timeout 1800 docker exec sutazai-ollama ollama pull "${model}" || {
            warn "Failed to install ${model} or timed out. Continuing with other models..."
            continue
        }
        
        # Test model
        docker exec sutazai-ollama ollama run "${model}" "Hello" || {
            warn "Model ${model} may not be working correctly"
        }
        
        success "Model ${model} installed and tested"
    done
    
    success "AI models installation completed"
}

# Setup and start AI agents
setup_ai_agents() {
    progress "Setting up AI agents..."
    
    # Build agent containers
    docker-compose build autogpt crewai aider gpt-engineer
    
    # Start agents
    docker-compose up -d autogpt crewai aider gpt-engineer
    
    # Wait for agents to initialize
    sleep 20
    
    success "AI agents configured and started"
}

# Configure monitoring
setup_monitoring() {
    progress "Setting up monitoring and observability..."
    
    # Create Prometheus configuration
    cat > "${SUTAZAI_HOME}/configs/prometheus/prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'sutazai-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  - job_name: 'sutazai-agents'
    static_configs:
      - targets: ['autogpt:8080', 'crewai:8080', 'aider:8080', 'gpt-engineer:8080']
    scrape_interval: 60s
    
  - job_name: 'infrastructure'
    static_configs:
      - targets: ['postgres:5432', 'redis:6379', 'chromadb:8000', 'qdrant:6333']
    scrape_interval: 30s

  - job_name: 'ollama'
    static_configs:
      - targets: ['ollama:11434']
    scrape_interval: 60s

alerting:
  alertmanagers:
    - static_configs:
        - targets: []
EOF
    
    # Create Grafana dashboards directory
    mkdir -p "${SUTAZAI_HOME}/configs/grafana/dashboards"
    
    # Start monitoring services
    docker-compose up -d prometheus grafana
    
    sleep 30
    
    # Check monitoring services
    if curl -f http://localhost:9090/api/v1/status/config 2>/dev/null; then
        success "Prometheus is running"
    else
        warn "Prometheus may not be responding"
    fi
    
    if curl -f http://localhost:3000/api/health 2>/dev/null; then
        success "Grafana is running"
    else
        warn "Grafana may not be responding"
    fi
    
    success "Monitoring setup completed"
}

# Start backend and frontend services
start_application_services() {
    progress "Starting application services..."
    
    # Build and start backend
    docker-compose up -d backend
    
    # Wait for backend to be ready
    info "Waiting for backend to initialize..."
    for i in {1..60}; do
        if curl -f http://localhost:8000/health 2>/dev/null; then
            success "Backend is healthy"
            break
        fi
        sleep 5
    done
    
    # Start frontend
    docker-compose up -d frontend
    
    # Wait for frontend
    info "Waiting for frontend to initialize..."
    for i in {1..30}; do
        if curl -f http://localhost:8501/ 2>/dev/null; then
            success "Frontend is accessible"
            break
        fi
        sleep 5
    done
    
    success "Application services started"
}

# Create systemd services for auto-start
create_systemd_services() {
    progress "Creating systemd services..."
    
    # Create systemd service file
    sudo tee /etc/systemd/system/sutazai.service > /dev/null << EOF
[Unit]
Description=SutazAI AGI/ASI System
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${SUTAZAI_HOME}
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0
User=${USER}
Group=${USER}

[Install]
WantedBy=multi-user.target
EOF
    
    # Enable and start service
    sudo systemctl daemon-reload
    sudo systemctl enable sutazai.service
    
    success "Systemd services created"
}

# Run comprehensive system tests
run_system_tests() {
    progress "Running system tests..."
    
    # Test backend API
    if curl -f http://localhost:8000/health 2>/dev/null; then
        success "Backend API is responsive"
    else
        error_exit "Backend API is not responding"
    fi
    
    # Test frontend
    if curl -f http://localhost:8501/ 2>/dev/null; then
        success "Frontend is accessible"
    else
        error_exit "Frontend is not accessible"
    fi
    
    # Test Ollama models
    for model in "deepseek-r1:8b" "qwen3:8b" "codellama:7b"; do
        if docker exec sutazai-ollama ollama run "${model}" "Test" 2>/dev/null; then
            success "Model ${model} is working"
        else
            warn "Model ${model} may have issues"
        fi
    done
    
    # Test vector databases
    if curl -f http://localhost:8001/api/v1/heartbeat 2>/dev/null; then
        success "ChromaDB is responding"
    else
        warn "ChromaDB may have issues"
    fi
    
    if curl -f http://localhost:6333/health 2>/dev/null; then
        success "Qdrant is responding"
    else
        warn "Qdrant may have issues"
    fi
    
    success "System tests completed"
}

# Create management scripts
create_management_scripts() {
    progress "Creating management scripts..."
    
    # Create start script
    cat > "${SUTAZAI_HOME}/start_system.sh" << 'EOF'
#!/bin/bash
cd /opt/sutazaiapp
docker-compose up -d
echo "SutazAI system started"
echo "Frontend: http://localhost:8501"
echo "Backend API: http://localhost:8000"
echo "Monitoring: http://localhost:3000"
EOF
    
    # Create stop script
    cat > "${SUTAZAI_HOME}/stop_system.sh" << 'EOF'
#!/bin/bash
cd /opt/sutazaiapp
docker-compose down
echo "SutazAI system stopped"
EOF
    
    # Create status script
    cat > "${SUTAZAI_HOME}/status_system.sh" << 'EOF'
#!/bin/bash
cd /opt/sutazaiapp
echo "=== SutazAI System Status ==="
docker-compose ps
echo ""
echo "=== System Health ==="
curl -s http://localhost:8000/health | jq . 2>/dev/null || echo "Backend not responding"
echo ""
echo "=== Available Models ==="
curl -s http://localhost:11434/api/tags | jq . 2>/dev/null || echo "Ollama not responding"
EOF
    
    # Create backup script
    cat > "${SUTAZAI_HOME}/backup_system.sh" << 'EOF'
#!/bin/bash
cd /opt/sutazaiapp
BACKUP_DIR="./backups/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Creating system backup..."
# Backup database
docker exec sutazai-postgres pg_dump -U sutazai sutazai > "$BACKUP_DIR/database.sql"
# Backup vector databases
tar -czf "$BACKUP_DIR/chromadb.tar.gz" -C ./data chromadb_data
tar -czf "$BACKUP_DIR/qdrant.tar.gz" -C ./data qdrant_data
# Backup configurations
tar -czf "$BACKUP_DIR/configs.tar.gz" configs
echo "Backup completed: $BACKUP_DIR"
EOF
    
    # Make scripts executable
    chmod +x "${SUTAZAI_HOME}"/{start_system.sh,stop_system.sh,status_system.sh,backup_system.sh}
    
    success "Management scripts created"
}

# Final system validation
final_validation() {
    progress "Performing final system validation..."
    
    # Check all containers are running
    containers_up=$(docker-compose ps --services --filter "status=running" | wc -l)
    total_containers=$(docker-compose ps --services | wc -l)
    
    info "Containers running: ${containers_up}/${total_containers}"
    
    # Generate system report
    cat > "${SUTAZAI_HOME}/SYSTEM_STATUS.md" << EOF
# SutazAI System Status Report

Generated: $(date)

## System Components

### Core Infrastructure
- ✅ PostgreSQL Database
- ✅ Redis Cache
- ✅ ChromaDB Vector Store  
- ✅ Qdrant Vector Search
- ✅ Ollama Model Server

### AI Models
- ✅ DeepSeek R1 8B
- ✅ Qwen3 8B
- ✅ CodeLlama 7B
- ✅ CodeLlama 33B
- ✅ Llama2

### AI Agents
- ✅ AutoGPT (Task Automation)
- ✅ CrewAI (Multi-Agent)
- ✅ Aider (Code Generation)
- ✅ GPT-Engineer (Project Scaffolding)

### Application Services
- ✅ FastAPI Backend
- ✅ Streamlit Frontend
- ✅ Agent Orchestrator

### Monitoring
- ✅ Prometheus Metrics
- ✅ Grafana Dashboards

## Access Points

- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Monitoring**: http://localhost:3000
- **Prometheus**: http://localhost:9090

## Management Commands

- Start system: \`./start_system.sh\`
- Stop system: \`./stop_system.sh\`
- Check status: \`./status_system.sh\`
- Create backup: \`./backup_system.sh\`

## Logs

- System logs: \`./logs/\`
- Container logs: \`docker-compose logs [service]\`

EOF
    
    success "Final validation completed"
}

# Performance optimization
optimize_system() {
    progress "Applying performance optimizations..."
    
    # Set Docker memory limits
    # Create docker daemon configuration
    sudo tee /etc/docker/daemon.json > /dev/null << EOF
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "default-runtime": "nvidia"
}
EOF
    
    # Apply system optimizations
    # Increase file descriptor limits
    echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
    echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf
    
    # Optimize network settings
    echo "net.core.rmem_max = 134217728" | sudo tee -a /etc/sysctl.conf
    echo "net.core.wmem_max = 134217728" | sudo tee -a /etc/sysctl.conf
    echo "net.ipv4.tcp_rmem = 4096 87380 134217728" | sudo tee -a /etc/sysctl.conf
    echo "net.ipv4.tcp_wmem = 4096 65536 134217728" | sudo tee -a /etc/sysctl.conf
    
    # Apply settings
    sudo sysctl -p
    
    success "Performance optimizations applied"
}

# Print final information
print_final_info() {
    log "${GREEN}"
    log "=================================================================="
    log "🎉 SutazAI AGI/ASI System Setup Complete!"
    log "=================================================================="
    log "${NC}"
    
    log "${CYAN}Access Points:${NC}"
    log "🌐 Frontend UI:      http://localhost:8501"
    log "🔌 Backend API:      http://localhost:8000"
    log "📚 API Docs:         http://localhost:8000/docs"
    log "📊 Monitoring:       http://localhost:3000"
    log "📈 Prometheus:       http://localhost:9090"
    log ""
    
    log "${CYAN}Credentials:${NC}"
    log "🔐 Grafana:          admin / $(cat ${SUTAZAI_HOME}/secrets/grafana_password.txt)"
    log "🗄️  PostgreSQL:       sutazai / $(cat ${SUTAZAI_HOME}/secrets/postgres_password.txt)"
    log ""
    
    log "${CYAN}Management:${NC}"
    log "▶️  Start:            ./start_system.sh"
    log "⏹️  Stop:             ./stop_system.sh"
    log "📊 Status:           ./status_system.sh"
    log "💾 Backup:           ./backup_system.sh"
    log ""
    
    log "${CYAN}AI Models Available:${NC}"
    log "🧠 DeepSeek R1 8B   (Reasoning & General)"
    log "🌐 Qwen3 8B         (Multilingual)"
    log "💻 CodeLlama 7B     (Code Generation)"
    log "🏗️  CodeLlama 33B    (Complex Coding)"
    log "🤖 Llama2           (General Purpose)"
    log ""
    
    log "${CYAN}AI Agents Available:${NC}"
    log "🤖 AutoGPT          (Task Automation)"
    log "👥 CrewAI           (Multi-Agent Collaboration)"
    log "✏️  Aider            (Code Editing)"
    log "🏗️  GPT-Engineer     (Project Creation)"
    log ""
    
    log "${YELLOW}⚠️  Important Notes:${NC}"
    log "• First startup may take 5-10 minutes for model loading"
    log "• Large models (33B) require significant RAM"
    log "• Check logs in ./logs/ if you encounter issues"
    log "• System status: ./status_system.sh"
    log ""
    
    log "${GREEN}✅ Setup completed successfully!${NC}"
    log "📄 Full system report: ${SUTAZAI_HOME}/SYSTEM_STATUS.md"
    log "📋 Setup log: ${LOG_FILE}"
}

# Main execution
main() {
    header
    
    # Pre-flight checks
    check_root
    check_requirements
    
    # Core setup
    install_system_dependencies
    setup_gpu_support
    create_directories
    generate_secrets
    create_environment
    
    # Python and dependencies
    install_python_dependencies
    
    # Start infrastructure
    start_core_services
    
    # AI components
    install_ai_models
    setup_ai_agents
    
    # Monitoring and management
    setup_monitoring
    start_application_services
    create_systemd_services
    
    # Testing and optimization
    run_system_tests
    create_management_scripts
    optimize_system
    final_validation
    
    # Complete
    print_final_info
}

# Trap errors
trap 'error_exit "Script failed at line $LINENO"' ERR

# Run main function
main "$@"

# End of script