#!/bin/bash

# SutazAI AGI/ASI Autonomous System Management Script
# Comprehensive management tool for all system operations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
PROJECT_NAME="sutazaiapp"
BACKEND_URL="http://localhost:8000"
FRONTEND_URL="http://localhost:8501"
COMMAND=$1
SERVICE=$2

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Health check function
check_service_health() {
    local service_name=$1
    local health_url=$2
    local max_attempts=30
    local attempt=1
    
    log "Checking health of $service_name..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$health_url" > /dev/null 2>&1; then
            success "$service_name is healthy"
            return 0
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            error "$service_name health check failed after $max_attempts attempts"
            return 1
        fi
        
        log "Attempt $attempt/$max_attempts - waiting for $service_name..."
        sleep 5
        ((attempt++))
    done
}

function print_help() {
    echo "SutazAI AGI/ASI Autonomous System Management"
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  start          Start the entire SutazAI system"
    echo "  stop           Stop the entire SutazAI system"
    echo "  restart        Restart the entire SutazAI system"
    echo "  status         Show system status and health"
    echo "  update         Update system images and restart"
    echo "  backup         Create a backup of all system data"
    echo "  install-models Install AI models in Ollama"
    echo "  logs [service] View logs (optionally for specific service)"
    echo "  cleanup        Clean up system and remove unused resources"
    echo "  help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start              # Start the system"
    echo "  $0 status             # Check system health"
    echo "  $0 logs backend       # View backend logs"
    echo "  $0 install-models     # Install AI models"
}

function check_docker() {
    if ! [ -x "$(command -v docker)" ]; then
        echo "Error: docker is not installed. Please install docker." >&2
        exit 1
    fi
    if ! [ -x "$(command -v docker-compose)" ]; then
        echo "Error: docker-compose is not installed. Please install docker-compose." >&2
        exit 1
    fi
}

# --- Main Logic ---

check_docker

# System status function
system_status() {
    log "SutazAI System Status"
    echo "===================="
    
    # Docker containers
    echo -e "\n${YELLOW}Docker Containers:${NC}"
    docker ps --filter "name=sutazai" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    # System health
    echo -e "\n${YELLOW}Service Health:${NC}"
    
    # Backend health
    if curl -s -f "$BACKEND_URL/health" > /dev/null 2>&1; then
        echo -e "Backend API: ${GREEN}✓ Healthy${NC}"
        # Get system metrics
        curl -s "$BACKEND_URL/api/metrics" | jq -r '
            "CPU Usage: " + (.system.cpu.usage_percent | tostring) + "%",
            "Memory Usage: " + (.system.memory.usage_percent | tostring) + "%",
            "Disk Usage: " + (.system.disk.usage_percent | tostring) + "%",
            "Active Agents: " + (.active_agents | tostring),
            "Uptime: " + (.uptime | tostring) + " seconds"
        ' 2>/dev/null || echo "Metrics unavailable"
    else
        echo -e "Backend API: ${RED}✗ Unhealthy${NC}"
    fi
    
    # Frontend health
    if curl -s -f "$FRONTEND_URL" > /dev/null 2>&1; then
        echo -e "Frontend UI: ${GREEN}✓ Accessible${NC}"
    else
        echo -e "Frontend UI: ${RED}✗ Inaccessible${NC}"
    fi
    
    echo -e "\n${YELLOW}System Access:${NC}"
    echo "Frontend: $FRONTEND_URL"
    echo "Backend API: $BACKEND_URL"
    echo "API Docs: $BACKEND_URL/docs"
}

case "$COMMAND" in
    start)
        log "Starting SutazAI AGI/ASI Autonomous System..."
        
        # Create required directories
        mkdir -p secrets logs workspace
        
        # Generate secrets if they don't exist
        if [ ! -f secrets/postgres_password.txt ]; then
            openssl rand -base64 32 > secrets/postgres_password.txt
            log "Generated PostgreSQL password"
        fi
        
        # Start core infrastructure first
        log "Starting core infrastructure..."
        docker-compose up -d postgres redis chromadb qdrant ollama
        
        # Wait for databases to be ready
        log "Waiting for databases to initialize..."
        sleep 30
        
        # Start backend
        log "Starting backend API..."
        docker-compose up -d backend
        
        # Wait for backend to be ready
        check_service_health "Backend API" "$BACKEND_URL/health"
        
        # Start frontend
        log "Starting frontend UI..."
        docker-compose up -d frontend
        
        # Wait for frontend to be ready
        check_service_health "Frontend UI" "$FRONTEND_URL"
        
        # Start AI agents
        log "Starting AI agents..."
        docker-compose up -d aider gpt-engineer autogpt crewai
        
        success "SutazAI system started successfully!"
        echo ""
        echo "Access points:"
        echo "- Frontend UI: $FRONTEND_URL"
        echo "- Backend API: $BACKEND_URL"
        echo "- API Documentation: $BACKEND_URL/docs"
        ;;
    stop)
        log "Stopping SutazAI system..."
        docker-compose down
        success "SutazAI system stopped"
        ;;
    restart)
        log "Restarting SutazAI system..."
        docker-compose down
        sleep 5
        $0 start
        ;;
    status)
        system_status
        ;;
    logs)
        if [ -z "$SERVICE" ]; then
            echo "Tailing logs for all services... (Press Ctrl+C to exit)"
            docker-compose logs -f
        else
            echo "Tailing logs for $SERVICE... (Press Ctrl+C to exit)"
            docker-compose logs -f "$SERVICE"
        fi
        ;;
    install-models)
        log "Installing AI models..."
        
        # Wait for Ollama to be ready
        check_service_health "Ollama" "http://localhost:11434"
        
        # Install models one by one
        local models=("llama3.2:1b" "qwen2.5:3b")
        
        for model in "${models[@]}"; do
            log "Installing model: $model"
            docker exec sutazai-ollama ollama pull "$model" || warning "Failed to install $model"
        done
        
        # List installed models
        log "Installed models:"
        docker exec sutazai-ollama ollama list
        ;;
    backup)
        local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$backup_dir"
        
        log "Creating system backup in $backup_dir..."
        
        # Backup database
        docker exec sutazai-postgres pg_dump -U sutazai sutazai > "$backup_dir/postgres_backup.sql"
        
        success "Backup completed: $backup_dir"
        ;;
    help|*)
        print_help
        ;;
esac

exit 0
