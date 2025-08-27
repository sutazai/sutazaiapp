#!/bin/bash
# SutazaiApp Multi-Agent AI System Deployment Script
# Version: 2.0
# Date: 2025-08-27
# This script deploys the complete Jarvis AI multi-agent system

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/logs/deployment-$(date +%Y%m%d-%H%M%S).log"
ENV_FILE="${SCRIPT_DIR}/.env"

# Ensure log directory exists
mkdir -p "${SCRIPT_DIR}/logs"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

# Header
print_header() {
    echo -e "${BLUE}"
    cat << "EOF"
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║          🤖 SutazaiApp Multi-Agent AI System v2.0 🤖               ║
║                                                                    ║
║               Jarvis Voice-Controlled AI Platform                  ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check CPU cores
    CPU_CORES=$(nproc)
    if [ $CPU_CORES -lt 4 ]; then
        warning "System has only $CPU_CORES cores. Minimum 4 cores recommended."
    else
        log "✓ CPU cores: $CPU_CORES"
    fi
    
    # Check RAM
    TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
    if [ $TOTAL_RAM -lt 8 ]; then
        warning "System has only ${TOTAL_RAM}GB RAM. Minimum 8GB recommended."
    else
        log "✓ RAM: ${TOTAL_RAM}GB"
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ $AVAILABLE_SPACE -lt 50 ]; then
        warning "Only ${AVAILABLE_SPACE}GB free disk space. Minimum 50GB recommended."
    else
        log "✓ Disk space: ${AVAILABLE_SPACE}GB available"
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    else
        log "✓ Docker: $(docker --version)"
    fi
    
    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose v2."
    else
        log "✓ Docker Compose: $(docker compose version)"
    fi
}

# Generate environment file
generate_env() {
    if [ -f "$ENV_FILE" ]; then
        warning "Environment file already exists. Backing up..."
        cp "$ENV_FILE" "${ENV_FILE}.backup-$(date +%Y%m%d-%H%M%S)"
    fi
    
    log "Generating secure environment variables..."
    
    cat > "$ENV_FILE" << EOF
# SutazaiApp Environment Configuration
# Generated: $(date)

# Database Passwords
POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
NEO4J_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)

# Service Passwords
RABBITMQ_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
GRAFANA_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)

# JWT and Security
JWT_SECRET=$(openssl rand -base64 64)
API_KEY=$(openssl rand -base64 32)

# GitHub Token (optional - for GitHub integration)
GITHUB_TOKEN=

# Model Configuration
OLLAMA_MAX_LOADED_MODELS=2
OLLAMA_MODELS_PATH=/root/.ollama/models

# Resource Limits
MAX_WORKERS=4
TASK_TIMEOUT=300
MEMORY_LIMIT_MB=7168
CPU_LIMIT_PERCENT=80

# Feature Flags
ENABLE_VOICE=true
WAKE_WORD=jarvis
TTS_ENGINE=pyttsx3
STT_MODEL=whisper-tiny
EOF
    
    chmod 600 "$ENV_FILE"
    log "✓ Environment file generated"
}

# Install system dependencies
install_dependencies() {
    log "Installing system dependencies..."
    
    # Check if running as root or with sudo
    if [ "$EUID" -ne 0 ]; then 
        SUDO="sudo"
    else
        SUDO=""
    fi
    
    # Update package list
    $SUDO apt-get update -qq
    
    # Install required packages
    PACKAGES="curl wget git python3 python3-pip python3-venv portaudio19-dev ffmpeg jq"
    
    for package in $PACKAGES; do
        if ! dpkg -l | grep -q "^ii  $package"; then
            log "Installing $package..."
            $SUDO apt-get install -y -qq "$package"
        fi
    done
    
    log "✓ System dependencies installed"
}

# Install Ollama
install_ollama() {
    log "Installing Ollama..."
    
    if command -v ollama &> /dev/null; then
        log "Ollama is already installed: $(ollama --version)"
    else
        log "Downloading and installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        
        # Start Ollama service
        sudo systemctl enable ollama
        sudo systemctl start ollama
        
        sleep 5  # Give Ollama time to start
    fi
    
    # Pull TinyLlama (default model)
    log "Pulling TinyLlama model (this may take a few minutes)..."
    ollama pull tinyllama:latest || warning "Could not pull TinyLlama - will retry later"
    
    log "✓ Ollama setup complete"
}

# Clone agent repositories
clone_agents() {
    log "Cloning AI agent repositories..."
    
    cd "${SCRIPT_DIR}/agents"
    
    # List of repositories to clone
    declare -A repos=(
        ["letta"]="https://github.com/mysuperai/letta.git"
        ["autogpt"]="https://github.com/Significant-Gravitas/AutoGPT.git"
        ["localagi"]="https://github.com/mudler/LocalAGI.git"
        ["agent-zero"]="https://github.com/frdel/agent-zero.git"
        ["langchain"]="https://github.com/langchain-ai/langchain.git"
        ["autogen"]="https://github.com/ag2ai/ag2.git"
        ["crewai"]="https://github.com/crewAIInc/crewAI.git"
        ["gpt-engineer"]="https://github.com/AntonOsika/gpt-engineer.git"
        ["opendevin"]="https://github.com/OpenDevin/OpenDevin.git"
        ["aider"]="https://github.com/Aider-AI/aider.git"
        ["deep-researcher"]="https://github.com/langchain-ai/local-deep-researcher.git"
        ["finrobot"]="https://github.com/AI4Finance-Foundation/FinRobot.git"
        ["semgrep"]="https://github.com/semgrep/semgrep.git"
        ["browser-use"]="https://github.com/browser-use/browser-use.git"
        ["skyvern"]="https://github.com/Skyvern-AI/skyvern.git"
        ["big-agi"]="https://github.com/enricoros/big-agi.git"
        ["privategpt"]="https://github.com/zylon-ai/private-gpt.git"
        ["llamaindex"]="https://github.com/run-llama/llama_index.git"
        ["flowise"]="https://github.com/FlowiseAI/Flowise.git"
        ["langflow"]="https://github.com/langflow-ai/langflow.git"
        ["dify"]="https://github.com/langgenius/dify.git"
    )
    
    for name in "${!repos[@]}"; do
        if [ ! -d "$name" ]; then
            log "Cloning $name..."
            git clone --depth 1 "${repos[$name]}" "$name" 2>/dev/null || warning "Could not clone $name"
        else
            info "$name already exists"
        fi
    done
    
    cd "${SCRIPT_DIR}"
    log "✓ Agent repositories cloned"
}

# Create configuration files
create_configs() {
    log "Creating configuration files..."
    
    # Kong configuration
    mkdir -p "${SCRIPT_DIR}/config/kong"
    cat > "${SCRIPT_DIR}/config/kong/kong.yml" << 'EOF'
_format_version: "3.0"

services:
  - name: jarvis-gateway
    url: http://172.20.0.30:8000
    routes:
      - name: jarvis-main
        paths:
          - /api/v1/jarvis
        strip_path: false

  - name: agent-orchestrator
    url: http://172.20.0.103:8000
    routes:
      - name: agent-route
        paths:
          - /api/v1/agents
        plugins:
          - name: rate-limiting
            config:
              minute: 60
              policy: local

  - name: ollama-service
    url: http://172.20.0.22:11434
    routes:
      - name: llm-route
        paths:
          - /api/v1/llm
        plugins:
          - name: request-size-limiting
            config:
              allowed_payload_size: 10
EOF
    
    # Prometheus configuration
    mkdir -p "${SCRIPT_DIR}/config/prometheus"
    cat > "${SCRIPT_DIR}/config/prometheus/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'node'
    static_configs:
      - targets: ['172.20.0.42:9100']
      
  - job_name: 'blackbox'
    metrics_path: /probe
    params:
      module: [http_2xx]
    static_configs:
      - targets:
          - http://172.20.0.30:8000/health
          - http://172.20.0.31:8501
          - http://172.20.0.22:11434/api/tags
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - source_labels: [__param_target]
        target_label: instance
      - target_label: __address__
        replacement: 172.20.0.43:9115
EOF
    
    log "✓ Configuration files created"
}

# Start Docker services
start_services() {
    log "Starting Docker services..."
    
    cd "${SCRIPT_DIR}"
    
    # Create network if it doesn't exist
    docker network create --subnet=172.20.0.0/16 sutazai-network 2>/dev/null || true
    
    # Start core services first
    log "Starting core infrastructure services..."
    docker compose up -d postgres redis neo4j consul rabbitmq kong
    
    # Wait for core services to be ready
    log "Waiting for core services to initialize..."
    sleep 15
    
    # Start AI services
    log "Starting AI and vector services..."
    docker compose up -d chromadb qdrant ollama
    
    sleep 10
    
    # Start application services
    log "Starting application services..."
    docker compose up -d backend frontend
    
    # Start monitoring stack
    log "Starting monitoring services..."
    docker compose up -d prometheus grafana node-exporter blackbox-exporter alertmanager
    
    # Start Portainer for management
    log "Starting Portainer..."
    docker compose up -d portainer
    
    log "✓ All services started"
}

# Initialize databases
initialize_databases() {
    log "Initializing databases..."
    
    # Wait for PostgreSQL to be ready
    log "Waiting for PostgreSQL to be ready..."
    until docker exec sutazai-postgres pg_isready -U jarvis; do
        sleep 2
    done
    
    # Create database schema
    log "Creating database schema..."
    docker exec -i sutazai-postgres psql -U jarvis -d jarvis_ai << 'EOF'
-- Agent registry table
CREATE TABLE IF NOT EXISTS agents (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100),
    capabilities JSONB,
    status VARCHAR(50) DEFAULT 'inactive',
    port INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Task history table
CREATE TABLE IF NOT EXISTS tasks (
    id SERIAL PRIMARY KEY,
    task_id VARCHAR(255) UNIQUE NOT NULL,
    type VARCHAR(100),
    input_data JSONB,
    output_data JSONB,
    status VARCHAR(50),
    agent_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Conversation history
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    messages JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System metrics
CREATE TABLE IF NOT EXISTS metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(255),
    metric_value NUMERIC,
    metadata JSONB,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_agents_status ON agents(status);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_agent ON tasks(agent_id);
CREATE INDEX idx_conversations_session ON conversations(session_id);
EOF
    
    log "✓ Databases initialized"
}

# Health check
health_check() {
    log "Running health checks..."
    
    declare -A services=(
        ["PostgreSQL"]="http://localhost:10000"
        ["Redis"]="http://localhost:10001"
        ["Kong Gateway"]="http://localhost:10005"
        ["Consul"]="http://localhost:10006/v1/status/leader"
        ["RabbitMQ Management"]="http://localhost:10008"
        ["ChromaDB"]="http://localhost:10100/api/v1/heartbeat"
        ["Qdrant"]="http://localhost:10101/health"
        ["Ollama"]="http://localhost:10104/api/tags"
        ["Prometheus"]="http://localhost:10200/-/healthy"
        ["Grafana"]="http://localhost:10201/api/health"
        ["Portainer"]="http://localhost:9000"
    )
    
    failed=0
    for name in "${!services[@]}"; do
        url="${services[$name]}"
        if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "200\|301\|302"; then
            log "✓ $name: OK"
        else
            warning "✗ $name: Not responding"
            ((failed++))
        fi
    done
    
    if [ $failed -eq 0 ]; then
        log "✓ All services are healthy!"
    else
        warning "$failed services are not responding. They may still be starting up."
    fi
}

# Print access information
print_access_info() {
    echo -e "${GREEN}"
    cat << EOF

╔════════════════════════════════════════════════════════════════════╗
║                    🎉 DEPLOYMENT COMPLETE! 🎉                      ║
╚════════════════════════════════════════════════════════════════════╝

Access Points:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🌐 Jarvis Frontend:        http://localhost:10011
🔧 Backend API:           http://localhost:10010
📊 Grafana Dashboard:     http://localhost:10201
🐰 RabbitMQ Management:   http://localhost:10008
🔍 Consul UI:            http://localhost:10006
📈 Prometheus:           http://localhost:10200
🐋 Portainer:           http://localhost:9000

Default Credentials:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📧 RabbitMQ:  jarvis / (check .env file)
📊 Grafana:   admin / (check .env file)

Quick Commands:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 View logs:        docker compose logs -f [service_name]
🔄 Restart service:  docker compose restart [service_name]
📊 Check status:     docker compose ps
🛑 Stop all:        docker compose down
🚀 Start all:       docker compose up -d

Voice Control:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Open http://localhost:10011 in your browser
2. Click the microphone icon
3. Say "Hey Jarvis" to activate voice control

EOF
    echo -e "${NC}"
}

# Main deployment function
main() {
    print_header
    
    log "Starting SutazaiApp deployment..."
    
    # Run deployment steps
    check_requirements
    generate_env
    install_dependencies
    install_ollama
    create_configs
    clone_agents
    start_services
    initialize_databases
    
    # Wait a bit for everything to stabilize
    log "Waiting for services to stabilize..."
    sleep 20
    
    health_check
    print_access_info
    
    log "Deployment completed successfully!"
    log "Log file saved to: $LOG_FILE"
}

# Run main function
main "$@"