#!/bin/bash

# SutazAI AGI/ASI System - Automated Deployment Script
# ===================================================
# This script automates the complete deployment of the SutazAI system
# with all required components, models, and services.

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="sutazai"
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"
LOG_FILE="deployment.log"
BACKUP_DIR="backups"
MODELS_DIR="models"
DATA_DIR="data"

# Functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
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
    log "Checking prerequisites..."
    
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
    
    # Check NVIDIA Docker (optional)
    if command -v nvidia-docker &> /dev/null; then
        info "NVIDIA Docker detected. GPU acceleration will be enabled."
    else
        warning "NVIDIA Docker not found. GPU acceleration will be disabled."
    fi
    
    # Check available disk space (minimum 50GB)
    available_space=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    if [[ $available_space -lt 50 ]]; then
        error "Insufficient disk space. At least 50GB is required."
        exit 1
    fi
    
    log "Prerequisites check completed successfully."
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    directories=(
        "$BACKUP_DIR"
        "$MODELS_DIR"
        "$DATA_DIR"
        "logs"
        "workspace"
        "uploads"
        "cache"
        "ssl"
        "monitoring/prometheus"
        "monitoring/grafana"
        "nginx/ssl"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        info "Created directory: $dir"
    done
    
    log "Directory creation completed."
}

# Generate SSL certificates
generate_ssl_certificates() {
    log "Generating SSL certificates..."
    
    if [[ ! -f "ssl/cert.pem" || ! -f "ssl/key.pem" ]]; then
        openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes \
            -subj "/C=US/ST=State/L=City/O=SutazAI/CN=localhost"
        
        # Copy to nginx directory
        cp ssl/cert.pem nginx/ssl/
        cp ssl/key.pem nginx/ssl/
        
        info "SSL certificates generated successfully."
    else
        info "SSL certificates already exist."
    fi
}

# Setup environment variables
setup_environment() {
    log "Setting up environment variables..."
    
    if [[ ! -f "$ENV_FILE" ]]; then
        error "Environment file $ENV_FILE not found!"
        exit 1
    fi
    
    # Load environment variables
    source "$ENV_FILE"
    
    # Generate secret keys if not set
    if [[ "${SECRET_KEY}" == "your-secret-key-here-change-in-production" ]]; then
        new_secret=$(openssl rand -hex 32)
        sed -i "s/SECRET_KEY=.*/SECRET_KEY=${new_secret}/" "$ENV_FILE"
        info "Generated new SECRET_KEY"
    fi
    
    if [[ "${JWT_SECRET_KEY}" == "your-jwt-secret-key-here-change-in-production" ]]; then
        new_jwt_secret=$(openssl rand -hex 32)
        sed -i "s/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=${new_jwt_secret}/" "$ENV_FILE"
        info "Generated new JWT_SECRET_KEY"
    fi
    
    log "Environment setup completed."
}

# Pull required Docker images
pull_images() {
    log "Pulling required Docker images..."
    
    # Core images
    images=(
        "postgres:15"
        "redis:7-alpine"
        "qdrant/qdrant:latest"
        "chromadb/chroma:latest"
        "ollama/ollama:latest"
        "prom/prometheus:latest"
        "grafana/grafana:latest"
        "prom/node-exporter:latest"
        "nginx:alpine"
        "significant/autogpt:latest"
        "tabbyml/tabby:latest"
        "returntocorp/semgrep:latest"
        "ghcr.io/open-webui/open-webui:main"
    )
    
    for image in "${images[@]}"; do
        info "Pulling image: $image"
        docker pull "$image" || warning "Failed to pull $image"
    done
    
    log "Image pulling completed."
}

# Build custom Docker images
build_images() {
    log "Building custom Docker images..."
    
    # Build backend
    info "Building SutazAI backend..."
    docker build -t sutazai-backend:latest -f docker/backend.Dockerfile ./backend/
    
    # Build frontend
    info "Building SutazAI frontend..."
    docker build -t sutazai-frontend:latest -f docker/streamlit.Dockerfile ./frontend/
    
    # Build AI agents
    info "Building AI agent images..."
    docker build -t sutazai-browser-use:latest -f docker/browser-use/Dockerfile ./docker/browser-use/
    docker build -t sutazai-skyvern:latest -f docker/skyvern/Dockerfile ./docker/skyvern/
    docker build -t sutazai-documind:latest -f docker/documind/Dockerfile ./docker/documind/
    docker build -t sutazai-finrobot:latest -f docker/finrobot/Dockerfile ./docker/finrobot/
    docker build -t sutazai-gpt-engineer:latest -f docker/gpt-engineer/Dockerfile ./docker/gpt-engineer/
    docker build -t sutazai-aider:latest -f docker/aider/Dockerfile ./docker/aider/
    docker build -t sutazai-bigagi:latest -f docker/bigagi/Dockerfile ./docker/bigagi/
    docker build -t sutazai-agentzero:latest -f docker/agentzero/Dockerfile ./docker/agentzero/
    docker build -t sutazai-health-check:latest -f docker/health-check/Dockerfile ./docker/health-check/
    
    log "Image building completed."
}

# Download required models
download_models() {
    log "Downloading required AI models..."
    
    # Start Ollama temporarily to download models
    docker run -d --name ollama-temp -v ollama-data:/root/.ollama ollama/ollama:latest
    
    # Wait for Ollama to be ready
    sleep 10
    
    # Download models
    models=(
        "deepseek-coder:33b"
        "llama2:13b"
        "codellama:7b"
        "mistral:7b"
    )
    
    for model in "${models[@]}"; do
        info "Downloading model: $model"
        docker exec ollama-temp ollama pull "$model" || warning "Failed to download $model"
    done
    
    # Stop temporary container
    docker stop ollama-temp
    docker rm ollama-temp
    
    log "Model downloading completed."
}

# Setup monitoring configuration
setup_monitoring() {
    log "Setting up monitoring configuration..."
    
    # Create Prometheus config
    cat > monitoring/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'sutazai-backend'
    static_configs:
      - targets: ['sutazai-backend:8000']
        labels:
          service: 'backend'
  
  - job_name: 'sutazai-streamlit'
    static_configs:
      - targets: ['sutazai-streamlit:8501']
        labels:
          service: 'frontend'
  
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
        labels:
          service: 'system'
  
  - job_name: 'qdrant'
    static_configs:
      - targets: ['qdrant:6333']
        labels:
          service: 'vector-db'
  
  - job_name: 'ollama'
    static_configs:
      - targets: ['ollama:11434']
        labels:
          service: 'llm'
EOF
    
    # Create Grafana dashboard
    mkdir -p monitoring/grafana/dashboards
    cat > monitoring/grafana/dashboards/sutazai-dashboard.json << EOF
{
  "dashboard": {
    "title": "SutazAI System Dashboard",
    "panels": [
      {
        "title": "System Overview",
        "type": "stat",
        "targets": [
          {
            "expr": "up",
            "legendFormat": "Services Up"
          }
        ]
      }
    ]
  }
}
EOF
    
    log "Monitoring configuration completed."
}

# Start services
start_services() {
    log "Starting SutazAI services..."
    
    # Start core infrastructure first
    info "Starting core infrastructure..."
    docker-compose up -d postgres redis qdrant chromadb
    
    # Wait for databases to be ready
    sleep 30
    
    # Start model services
    info "Starting model services..."
    docker-compose up -d ollama
    
    # Wait for Ollama to be ready
    sleep 20
    
    # Start AI agents
    info "Starting AI agents..."
    docker-compose up -d autogpt localagi tabby browser-use skyvern documind finrobot gpt-engineer aider open-webui bigagi agentzero
    
    # Start main application
    info "Starting main application..."
    docker-compose up -d sutazai-backend sutazai-streamlit
    
    # Start monitoring
    info "Starting monitoring services..."
    docker-compose up -d prometheus grafana node-exporter
    
    # Start reverse proxy
    info "Starting reverse proxy..."
    docker-compose up -d nginx
    
    # Start health check
    info "Starting health check service..."
    docker-compose up -d health-check
    
    log "All services started successfully."
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Check service health
    services=(
        "sutazai-backend:8000/health"
        "sutazai-streamlit:8501/healthz"
        "qdrant:6333/healthz"
        "chromadb:8000/api/v1/heartbeat"
        "ollama:11434/api/health"
        "prometheus:9090/-/healthy"
        "grafana:3000/api/health"
    )
    
    for service in "${services[@]}"; do
        service_name=$(echo "$service" | cut -d: -f1)
        endpoint="http://$service"
        
        info "Checking $service_name..."
        
        # Try to connect to the service
        if curl -f -s --max-time 10 "$endpoint" > /dev/null 2>&1; then
            info "$service_name is healthy"
        else
            warning "$service_name is not responding"
        fi
    done
    
    log "Deployment verification completed."
}

# Create backup
create_backup() {
    log "Creating system backup..."
    
    timestamp=$(date +%Y%m%d_%H%M%S)
    backup_file="$BACKUP_DIR/sutazai_backup_$timestamp.tar.gz"
    
    # Create backup of data directories
    tar -czf "$backup_file" \
        data/ \
        logs/ \
        workspace/ \
        .env \
        docker-compose.yml \
        --exclude="*.log" \
        --exclude="*.tmp" \
        2>/dev/null || true
    
    info "Backup created: $backup_file"
    
    # Keep only last 5 backups
    cd "$BACKUP_DIR"
    ls -t sutazai_backup_*.tar.gz | tail -n +6 | xargs -r rm
    cd ..
    
    log "Backup creation completed."
}

# Setup systemd service (optional)
setup_systemd() {
    if [[ "$EUID" -eq 0 ]]; then
        log "Setting up systemd service..."
        
        cat > /etc/systemd/system/sutazai.service << EOF
[Unit]
Description=SutazAI AGI/ASI System
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF
        
        systemctl daemon-reload
        systemctl enable sutazai.service
        
        info "Systemd service created and enabled."
    else
        info "Skipping systemd setup (requires root privileges)."
    fi
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    
    # Remove temporary containers
    docker ps -a --filter "name=temp" -q | xargs -r docker rm -f
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    log "Cleanup completed."
}

# Main deployment function
main() {
    log "Starting SutazAI AGI/ASI System deployment..."
    
    # Trap cleanup on exit
    trap cleanup EXIT
    
    # Run deployment steps
    check_prerequisites
    create_directories
    generate_ssl_certificates
    setup_environment
    pull_images
    build_images
    download_models
    setup_monitoring
    start_services
    
    # Wait for services to stabilize
    log "Waiting for services to stabilize..."
    sleep 60
    
    verify_deployment
    create_backup
    setup_systemd
    
    log "SutazAI deployment completed successfully!"
    
    # Display access information
    echo ""
    echo "🚀 SutazAI AGI/ASI System is now running!"
    echo ""
    echo "Access URLs:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🌐 Main Interface:        http://localhost"
    echo "📊 Backend API:           http://localhost/api"
    echo "🤖 Chat Interface:        http://localhost/chat"
    echo "📈 Monitoring:            http://localhost/grafana"
    echo "📊 Metrics:               http://localhost/prometheus"
    echo "🔍 Vector Search:         http://localhost/qdrant"
    echo "🧠 Model Management:      http://localhost/ollama"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Default credentials:"
    echo "  Grafana: admin/admin"
    echo ""
    echo "To stop the system: docker-compose down"
    echo "To view logs: docker-compose logs -f"
    echo "To restart: docker-compose restart"
    echo ""
    echo "For support, check the documentation or visit: https://github.com/sutazai/sutazaiapp"
}

# Run main function
main "$@"