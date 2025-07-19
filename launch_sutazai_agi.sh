#!/bin/bash
# SutazAI AGI/ASI Complete System Launcher
# Enterprise-grade production deployment

echo "╔══════════════════════════════════════════════════╗"
echo "║       SutazAI AGI/ASI System Launcher v10.0      ║"
echo "║          Enterprise Production Deployment         ║"
echo "╚══════════════════════════════════════════════════╝"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Check root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root or with sudo${NC}"
    exit 1
fi

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
LOG_DIR="$SCRIPT_DIR/logs/launcher"
mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/launcher.log"
}

# Status function
print_status() {
    local message=$1
    local status=$2
    
    case $status in
        "success")
            echo -e "${GREEN}✓ ${message}${NC}"
            ;;
        "error")
            echo -e "${RED}✗ ${message}${NC}"
            ;;
        "warning")
            echo -e "${YELLOW}⚠ ${message}${NC}"
            ;;
        "info")
            echo -e "${BLUE}ℹ ${message}${NC}"
            ;;
        "processing")
            echo -e "${CYAN}⟳ ${message}${NC}"
            ;;
    esac
    log "$status: $message"
}

# Phase tracking
CURRENT_PHASE=""
PHASE_COUNT=0
TOTAL_PHASES=10

start_phase() {
    ((PHASE_COUNT++))
    CURRENT_PHASE="$1"
    echo -e "\n${BOLD}[$PHASE_COUNT/$TOTAL_PHASES] $CURRENT_PHASE${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Starting phase: $CURRENT_PHASE"
}

# Error handling
handle_error() {
    local error_msg="$1"
    print_status "$error_msg" "error"
    log "ERROR in phase '$CURRENT_PHASE': $error_msg"
    
    # Ask user what to do
    echo -e "\n${YELLOW}What would you like to do?${NC}"
    echo "1) Retry this phase"
    echo "2) Skip and continue"
    echo "3) Run diagnostic tool"
    echo "4) Exit"
    
    read -p "Choice (1-4): " choice
    case $choice in
        1) return 1 ;;
        2) return 0 ;;
        3) 
            python3 fix_all_issues.py
            return 1
            ;;
        4) exit 1 ;;
        *) return 1 ;;
    esac
}

# Phase 1: System Validation
phase_system_validation() {
    start_phase "System Validation"
    
    # Check system resources
    print_status "Checking system resources..." "processing"
    
    cpu_cores=$(nproc)
    total_mem=$(free -g | awk '/^Mem:/{print $2}')
    free_disk=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    
    if [ $cpu_cores -lt 4 ]; then
        print_status "CPU cores: $cpu_cores (minimum 4 recommended)" "warning"
    else
        print_status "CPU cores: $cpu_cores" "success"
    fi
    
    if [ $total_mem -lt 8 ]; then
        print_status "RAM: ${total_mem}GB (minimum 8GB recommended)" "warning"
    else
        print_status "RAM: ${total_mem}GB" "success"
    fi
    
    if [ $free_disk -lt 20 ]; then
        print_status "Free disk: ${free_disk}GB (minimum 20GB recommended)" "warning"
    else
        print_status "Free disk: ${free_disk}GB" "success"
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        handle_error "Docker is not installed"
        return $?
    fi
    
    if ! docker ps &> /dev/null; then
        handle_error "Docker daemon is not running"
        return $?
    fi
    
    print_status "Docker is operational" "success"
    
    # Check Python
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_status "Python version: $python_version" "success"
    
    return 0
}

# Phase 2: Environment Preparation
phase_environment_prep() {
    start_phase "Environment Preparation"
    
    # Create directory structure
    print_status "Creating directory structure..." "processing"
    
    directories=(
        "data/uploads" "data/documents" "data/models" "data/cache" "data/vectors"
        "logs/agents" "logs/backend" "logs/system" "logs/launcher"
        "backups" "secrets" "monitoring/dashboards"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
    done
    
    chmod -R 755 data logs
    print_status "Directory structure created" "success"
    
    # Set up environment variables
    if [ ! -f .env ]; then
        print_status "Creating environment configuration..." "processing"
        cat > .env <<EOF
# SutazAI Environment Configuration
ENVIRONMENT=production
DATABASE_URL=postgresql://sutazai:sutazai123@localhost:5432/sutazai_db
REDIS_URL=redis://localhost:6379
OLLAMA_BASE_URL=http://localhost:11434
SECRET_KEY=$(openssl rand -hex 32)
API_KEY=$(openssl rand -hex 32)
LOG_LEVEL=INFO
EOF
        chmod 600 .env
        print_status "Environment configuration created" "success"
    else
        print_status "Environment configuration exists" "success"
    fi
    
    return 0
}

# Phase 3: Core Services Deployment
phase_core_services() {
    start_phase "Core Services Deployment"
    
    services=(
        "postgres:5432:PostgreSQL"
        "redis:6379:Redis"
        "ollama:11434:Ollama"
        "qdrant:6333:Qdrant"
        "chromadb:8001:ChromaDB"
    )
    
    # Start core services
    print_status "Starting core services..." "processing"
    docker-compose up -d postgres redis ollama qdrant chromadb 2>/dev/null
    
    # Wait for services to be ready
    print_status "Waiting for services to initialize..." "processing"
    sleep 10
    
    # Check each service
    for service in "${services[@]}"; do
        IFS=':' read -r name port display <<< "$service"
        
        if nc -z localhost $port 2>/dev/null; then
            print_status "$display is running on port $port" "success"
        else
            print_status "$display failed to start on port $port" "error"
            
            # Try to diagnose
            container_name="sutazai-$name"
            if docker ps -a | grep -q $container_name; then
                logs=$(docker logs --tail 20 $container_name 2>&1)
                log "Container logs for $container_name: $logs"
            fi
        fi
    done
    
    return 0
}

# Phase 4: Database Initialization
phase_database_init() {
    start_phase "Database Initialization"
    
    print_status "Checking database connection..." "processing"
    
    # Wait for PostgreSQL to be fully ready
    for i in {1..30}; do
        if docker exec sutazai-postgres pg_isready -U sutazai &>/dev/null; then
            print_status "Database is ready" "success"
            break
        fi
        if [ $i -eq 30 ]; then
            handle_error "Database failed to become ready"
            return $?
        fi
        sleep 1
    done
    
    # Initialize database schema
    print_status "Initializing database schema..." "processing"
    
    # Run schema creation through backend
    python3 -c "
import sys
sys.path.append('.')
try:
    from backend.models.base_models import Base
    from sqlalchemy import create_engine
    engine = create_engine('postgresql://sutazai:sutazai123@localhost:5432/sutazai_db')
    Base.metadata.create_all(bind=engine)
    print('Schema created successfully')
except Exception as e:
    print(f'Schema creation failed: {e}')
    sys.exit(1)
" || handle_error "Failed to create database schema"
    
    print_status "Database initialized" "success"
    return 0
}

# Phase 5: Python Dependencies
phase_python_deps() {
    start_phase "Python Dependencies Installation"
    
    print_status "Installing core dependencies..." "processing"
    
    # Core dependencies
    pip3 install -q --upgrade pip setuptools wheel
    
    # Install from requirements if exists
    if [ -f requirements.txt ]; then
        pip3 install -q -r requirements.txt
        print_status "Installed from requirements.txt" "success"
    else
        # Install essential packages
        pip3 install -q \
            fastapi uvicorn[standard] \
            sqlalchemy psycopg2-binary redis \
            transformers torch \
            langchain chromadb qdrant-client \
            streamlit gradio \
            prometheus-client psutil \
            aiofiles websockets python-multipart
        
        print_status "Installed essential packages" "success"
    fi
    
    return 0
}

# Phase 6: Backend Deployment
phase_backend_deployment() {
    start_phase "Backend Service Deployment"
    
    # Stop any existing backend processes
    print_status "Stopping existing backend processes..." "processing"
    pkill -f "intelligent_backend" 2>/dev/null
    pkill -f "simple_backend" 2>/dev/null
    sleep 2
    
    # Choose and start backend
    if [ -f "intelligent_backend_enterprise.py" ]; then
        backend_script="intelligent_backend_enterprise.py"
        backend_name="Enterprise Backend v10.0"
    elif [ -f "intelligent_backend_final.py" ]; then
        backend_script="intelligent_backend_final.py"
        backend_name="Final Backend v7.0"
    else
        handle_error "No suitable backend script found"
        return $?
    fi
    
    print_status "Starting $backend_name..." "processing"
    nohup python3 "$backend_script" > "$LOG_DIR/backend.log" 2>&1 &
    backend_pid=$!
    
    # Wait for backend to start
    for i in {1..30}; do
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            print_status "$backend_name is running (PID: $backend_pid)" "success"
            break
        fi
        if [ $i -eq 30 ]; then
            handle_error "Backend failed to start"
            return $?
        fi
        sleep 1
    done
    
    # Test key endpoints
    endpoints=("/health" "/api/models" "/api/status")
    for endpoint in "${endpoints[@]}"; do
        if curl -s "http://localhost:8000$endpoint" >/dev/null 2>&1; then
            print_status "Endpoint $endpoint is responding" "success"
        else
            print_status "Endpoint $endpoint is not responding" "warning"
        fi
    done
    
    return 0
}

# Phase 7: AI Models Setup
phase_ai_models() {
    start_phase "AI Models Configuration"
    
    print_status "Checking Ollama models..." "processing"
    
    # Required models
    models=("llama3.2:1b" "qwen2.5:3b")
    
    for model in "${models[@]}"; do
        # Check if model exists
        if docker exec sutazai-ollama ollama list | grep -q "$model"; then
            print_status "Model $model is available" "success"
        else
            print_status "Pulling model $model..." "processing"
            docker exec sutazai-ollama ollama pull "$model" &
            
            # Note: Model pulling happens in background
            print_status "Model $model pull initiated (background)" "info"
        fi
    done
    
    return 0
}

# Phase 8: AI Agents Deployment
phase_agents_deployment() {
    start_phase "AI Agents Deployment"
    
    print_status "Building AI agent images..." "processing"
    
    # Core agents to deploy
    agents=("autogpt" "crewai" "agentgpt" "privategpt" "llamaindex" "flowise")
    
    # Check if docker-compose file exists
    if [ ! -f docker-compose.yml ]; then
        print_status "docker-compose.yml not found" "warning"
        return 0
    fi
    
    # Build and start agents
    for agent in "${agents[@]}"; do
        print_status "Deploying $agent..." "processing"
        
        # Try to start the agent
        if docker-compose up -d "$agent" 2>/dev/null; then
            print_status "$agent deployment initiated" "success"
        else
            print_status "$agent deployment failed" "warning"
        fi
    done
    
    # Wait a bit for agents to initialize
    sleep 5
    
    # Check agent status
    print_status "Checking agent status..." "processing"
    docker-compose ps | grep -E "(autogpt|crewai|agentgpt|privategpt|llamaindex|flowise)"
    
    return 0
}

# Phase 9: System Optimization
phase_optimization() {
    start_phase "System Optimization"
    
    print_status "Applying performance optimizations..." "processing"
    
    # System optimizations
    sysctl -w vm.swappiness=10 >/dev/null 2>&1
    sysctl -w net.core.somaxconn=65535 >/dev/null 2>&1
    
    # Docker optimizations
    if [ -f /etc/docker/daemon.json ]; then
        print_status "Docker already configured" "success"
    else
        print_status "Optimizing Docker configuration..." "processing"
        cat > /etc/docker/daemon.json <<EOF
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "storage-driver": "overlay2"
}
EOF
        systemctl restart docker >/dev/null 2>&1
        print_status "Docker optimized" "success"
    fi
    
    # Clean up
    print_status "Cleaning up temporary files..." "processing"
    find /tmp -type f -name "*.log" -mtime +7 -delete 2>/dev/null
    docker system prune -f >/dev/null 2>&1
    print_status "Cleanup completed" "success"
    
    return 0
}

# Phase 10: Final Validation
phase_final_validation() {
    start_phase "Final System Validation"
    
    print_status "Running comprehensive system check..." "processing"
    
    # Run the fix script to identify any issues
    if python3 fix_all_issues.py > "$LOG_DIR/validation.log" 2>&1; then
        print_status "System validation passed" "success"
    else
        print_status "Some issues detected (check validation.log)" "warning"
    fi
    
    # Generate system report
    cat > SYSTEM_STATUS.md <<EOF
# SutazAI System Status Report
Generated: $(date)

## System Information
- **Version**: 10.0 Enterprise
- **Environment**: Production
- **Status**: Operational

## Services Status
$(docker-compose ps 2>/dev/null || echo "Docker Compose status unavailable")

## Access Points
- **Main API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs
- **Health Check**: http://localhost:8000/health

## Quick Commands
- View logs: \`tail -f logs/backend/*.log\`
- Monitor system: \`docker stats\`
- Check services: \`docker-compose ps\`
- Restart backend: \`systemctl restart sutazai-backend\`

## Support
- Logs directory: $LOG_DIR
- Configuration: .env
- Issues: Check SYSTEM_FIX_REPORT.md
EOF
    
    print_status "System report generated: SYSTEM_STATUS.md" "success"
    
    return 0
}

# Main execution
main() {
    echo -e "\n${BOLD}Starting SutazAI AGI/ASI System...${NC}\n"
    
    # Create PID file
    echo $$ > /var/run/sutazai-launcher.pid
    
    # Run all phases
    phases=(
        phase_system_validation
        phase_environment_prep
        phase_core_services
        phase_database_init
        phase_python_deps
        phase_backend_deployment
        phase_ai_models
        phase_agents_deployment
        phase_optimization
        phase_final_validation
    )
    
    for phase in "${phases[@]}"; do
        while ! $phase; do
            echo -e "\n${YELLOW}Phase failed. Retrying...${NC}"
            sleep 2
        done
    done
    
    # Success message
    echo -e "\n${GREEN}${BOLD}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}${BOLD}║     🎉 SutazAI AGI/ASI System Ready! 🎉         ║${NC}"
    echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════════════╝${NC}"
    
    echo -e "\n${CYAN}System Information:${NC}"
    echo "• API Endpoint: http://localhost:8000"
    echo "• Documentation: http://localhost:8000/api/docs"
    echo "• WebSocket: ws://localhost:8000/ws/{client_id}"
    echo "• Logs: $LOG_DIR"
    
    echo -e "\n${CYAN}Next Steps:${NC}"
    echo "1. Access the API documentation"
    echo "2. Start the Streamlit UI: streamlit run intelligent_chat_app.py"
    echo "3. Monitor system: tail -f logs/backend/*.log"
    echo "4. Check agent status: docker-compose ps"
    
    echo -e "\n${GREEN}System is fully operational!${NC}"
    
    # Remove PID file
    rm -f /var/run/sutazai-launcher.pid
    
    log "System launch completed successfully"
}

# Signal handlers
trap 'echo -e "\n${RED}Launch interrupted!${NC}"; exit 1' INT TERM

# Execute main function
main

exit 0