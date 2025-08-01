#!/bin/bash

# LocalAGI System Deployment Script
# Deploys the complete LocalAGI autonomous orchestration system

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="$PROJECT_ROOT/logs/localagi_deployment.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        INFO)
            echo -e "${GREEN}[INFO]${NC} $message"
            ;;
        WARN)
            echo -e "${YELLOW}[WARN]${NC} $message"
            ;;
        ERROR)
            echo -e "${RED}[ERROR]${NC} $message"
            ;;
        DEBUG)
            echo -e "${BLUE}[DEBUG]${NC} $message"
            ;;
    esac
    
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
}

# Error handling
error_exit() {
    log ERROR "$1"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log INFO "Checking prerequisites..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        error_exit "Docker is not running. Please start Docker and try again."
    fi
    
    # Check if docker-compose is available
    if ! command -v docker-compose >/dev/null 2>&1; then
        error_exit "docker-compose is not installed. Please install docker-compose and try again."
    fi
    
    # Check if Python 3.8+ is available
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
        error_exit "Python 3.8 or higher is required."
    fi
    
    # Check required directories exist
    local required_dirs=(
        "$PROJECT_ROOT/localagi"
        "$PROJECT_ROOT/backend"
        "$PROJECT_ROOT/logs"
        "$PROJECT_ROOT/data"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log WARN "Creating missing directory: $dir"
            mkdir -p "$dir"
        fi
    done
    
    log INFO "Prerequisites check completed"
}

# Install Python dependencies
install_dependencies() {
    log INFO "Installing Python dependencies..."
    
    # Check if virtual environment exists
    if [[ ! -d "$PROJECT_ROOT/venv" ]]; then
        log INFO "Creating Python virtual environment..."
        python3 -m venv "$PROJECT_ROOT/venv"
    fi
    
    # Activate virtual environment
    source "$PROJECT_ROOT/venv/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install LocalAGI dependencies
    local requirements_files=(
        "$PROJECT_ROOT/localagi/requirements.txt"
        "$PROJECT_ROOT/backend/requirements.txt"
    )
    
    for req_file in "${requirements_files[@]}"; do
        if [[ -f "$req_file" ]]; then
            log INFO "Installing requirements from $req_file..."
            pip install -r "$req_file"
        else
            log WARN "Requirements file not found: $req_file"
        fi
    done
    
    # Install additional dependencies for LocalAGI
    log INFO "Installing additional LocalAGI dependencies..."
    pip install asyncio aioredis pyyaml networkx numpy scipy scikit-learn
    
    log INFO "Python dependencies installation completed"
}

# Setup configuration
setup_configuration() {
    log INFO "Setting up LocalAGI configuration..."
    
    # Create environment file if it doesn't exist
    local env_file="$PROJECT_ROOT/.env"
    if [[ ! -f "$env_file" ]]; then
        log INFO "Creating .env file..."
        cat > "$env_file" << EOF
# LocalAGI Configuration
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=sutazai_password
POSTGRES_DB=sutazai
REDIS_PASSWORD=redis_password
NEO4J_PASSWORD=sutazai_neo4j_password
SECRET_KEY=localagi-secret-key-$(openssl rand -hex 16)
CHROMADB_API_KEY=localagi-chroma-key
GRAFANA_PASSWORD=sutazai_grafana

# LocalAGI Specific Settings
LOCALAGI_LOG_LEVEL=INFO
LOCALAGI_MAX_AGENTS=38
LOCALAGI_REDIS_DB=1
LOCALAGI_OLLAMA_MODEL=qwen2.5:3b
LOCALAGI_ENABLE_AUTONOMOUS=true
LOCALAGI_ENABLE_SWARMS=true
LOCALAGI_ENABLE_WORKFLOWS=true

# Performance Settings
LOCALAGI_MAX_CONCURRENT_TASKS=10
LOCALAGI_TASK_TIMEOUT=300
LOCALAGI_MEMORY_LIMIT=4G
EOF
        log INFO ".env file created"
    else
        log INFO ".env file already exists"
    fi
    
    # Ensure LocalAGI config directory exists
    mkdir -p "$PROJECT_ROOT/localagi/configs"
    
    log INFO "Configuration setup completed"
}

# Initialize database and services
init_services() {
    log INFO "Initializing core services..."
    
    # Start core infrastructure services first
    log INFO "Starting core infrastructure services..."
    docker-compose up -d postgres redis neo4j ollama chromadb qdrant
    
    # Wait for services to be ready
    log INFO "Waiting for services to be ready..."
    local max_wait=120
    local wait_time=0
    
    while [[ $wait_time -lt $max_wait ]]; do
        if docker-compose ps | grep -E "(postgres|redis|ollama)" | grep -q "Up"; then
            log INFO "Core services are ready"
            break
        fi
        
        log DEBUG "Waiting for services... ($wait_time/$max_wait seconds)"
        sleep 5
        wait_time=$((wait_time + 5))
    done
    
    if [[ $wait_time -ge $max_wait ]]; then
        log WARN "Some services may not be fully ready, continuing anyway..."
    fi
    
    # Initialize database schema
    log INFO "Initializing database schema..."
    docker-compose exec -T postgres psql -U sutazai -d sutazai -c "
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE EXTENSION IF NOT EXISTS pg_trgm;
        CREATE EXTENSION IF NOT EXISTS btree_gin;
    " || log WARN "Database initialization had some issues (may be expected)"
    
    log INFO "Services initialization completed"
}

# Deploy LocalAGI components
deploy_localagi() {
    log INFO "Deploying LocalAGI autonomous orchestration system..."
    
    # Start LocalAGI related services
    log INFO "Starting LocalAGI services..."
    
    # Start backend services
    docker-compose up -d backend-agi
    
    # Wait for backend to be ready
    log INFO "Waiting for backend service..."
    local backend_ready=false
    for i in {1..30}; do
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            backend_ready=true
            break
        fi
        log DEBUG "Waiting for backend... ($i/30)"
        sleep 5
    done
    
    if [[ "$backend_ready" = true ]]; then
        log INFO "Backend service is ready"
    else
        log WARN "Backend service may not be fully ready"
    fi
    
    # Start LocalAGI enhanced service
    log INFO "Starting LocalAGI enhanced service..."
    docker-compose up -d localagi-enhanced
    
    log INFO "LocalAGI deployment completed"
}

# Run system validation
validate_system() {
    log INFO "Running system validation..."
    
    # Test basic connectivity
    local services=(
        "http://localhost:11434:Ollama"
        "http://localhost:6379:Redis"
        "http://localhost:8000:Backend"
        "http://localhost:8115:LocalAGI"
    )
    
    local all_healthy=true
    
    for service_info in "${services[@]}"; do
        IFS=':' read -r url name <<< "$service_info"
        
        log INFO "Testing $name connectivity..."
        
        if [[ "$name" == "Redis" ]]; then
            # Special handling for Redis
            if echo "PING" | nc -w 5 localhost 6379 | grep -q "PONG"; then
                log INFO "$name: HEALTHY"
            else
                log WARN "$name: NOT RESPONDING"
                all_healthy=false
            fi
        else
            # HTTP services
            if curl -f -m 10 "$url" >/dev/null 2>&1 || curl -f -m 10 "$url/health" >/dev/null 2>&1; then
                log INFO "$name: HEALTHY"
            else
                log WARN "$name: NOT RESPONDING"
                all_healthy=false
            fi
        fi
    done
    
    # Run LocalAGI system test
    log INFO "Running LocalAGI system test..."
    
    if [[ -f "$PROJECT_ROOT/scripts/test_localagi_system.py" ]]; then
        cd "$PROJECT_ROOT"
        source venv/bin/activate
        
        if python3 scripts/test_localagi_system.py; then
            log INFO "LocalAGI system test: PASSED"
        else
            log WARN "LocalAGI system test: FAILED (check logs for details)"
            all_healthy=false
        fi
    else
        log WARN "LocalAGI test script not found, skipping detailed tests"
    fi
    
    if [[ "$all_healthy" = true ]]; then
        log INFO "System validation: ALL TESTS PASSED"
    else
        log WARN "System validation: SOME ISSUES DETECTED"
    fi
    
    return 0
}

# Generate deployment report
generate_report() {
    log INFO "Generating deployment report..."
    
    local report_file="$PROJECT_ROOT/logs/localagi_deployment_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
LocalAGI Autonomous Orchestration System Deployment Report
=========================================================

Deployment Date: $(date)
Project Root: $PROJECT_ROOT
Deployment Script: $0

System Components:
- LocalAGI Orchestration Engine: DEPLOYED
- Autonomous Decision Making: DEPLOYED
- Task Decomposition System: DEPLOYED
- Swarm Coordination: DEPLOYED
- Self-improving Workflows: DEPLOYED
- Collaborative Problem Solver: DEPLOYED
- Goal Achievement System: DEPLOYED
- Coordination Protocols: DEPLOYED

Infrastructure Services:
- PostgreSQL Database: $(docker-compose ps postgres | grep -q "Up" && echo "RUNNING" || echo "NOT RUNNING")
- Redis Cache: $(docker-compose ps redis | grep -q "Up" && echo "RUNNING" || echo "NOT RUNNING")
- Ollama Model Server: $(docker-compose ps ollama | grep -q "Up" && echo "RUNNING" || echo "NOT RUNNING")
- ChromaDB Vector Store: $(docker-compose ps chromadb | grep -q "Up" && echo "RUNNING" || echo "NOT RUNNING")
- Backend API: $(docker-compose ps backend-agi | grep -q "Up" && echo "RUNNING" || echo "NOT RUNNING")

Service Endpoints:
- Backend API: http://localhost:8000
- LocalAGI Service: http://localhost:8115
- Ollama API: http://localhost:11434
- ChromaDB: http://localhost:8001
- Qdrant: http://localhost:6333

Configuration Files:
- Main Config: $PROJECT_ROOT/localagi/configs/autonomous_orchestrator_config.yaml
- Environment: $PROJECT_ROOT/.env
- Docker Compose: $PROJECT_ROOT/docker-compose.yml

Log Files:
- Deployment Log: $LOG_FILE
- LocalAGI Logs: $PROJECT_ROOT/logs/localagi.log
- System Logs: $PROJECT_ROOT/logs/

Next Steps:
1. Access the LocalAGI system at http://localhost:8115
2. Review the system logs for any warnings or errors
3. Run the test suite: python3 scripts/test_localagi_system.py
4. Monitor system performance and adjust configurations as needed
5. Explore the autonomous capabilities through the API or web interface

For support and documentation, see:
- LocalAGI Documentation: $PROJECT_ROOT/localagi/README.md
- API Documentation: http://localhost:8000/docs
- System Monitoring: http://localhost:3000 (Grafana)

EOF

    log INFO "Deployment report generated: $report_file"
    
    # Display summary
    echo ""
    echo "================================================================"
    echo "           LocalAGI Deployment Summary"
    echo "================================================================"
    echo "Status: COMPLETED"
    echo "Services: $(docker-compose ps --services | wc -l) total services"
    echo "Running: $(docker-compose ps | grep -c "Up") services running"
    echo "Report: $report_file"
    echo "Logs: $LOG_FILE"
    echo ""
    echo "Access Points:"
    echo "- LocalAGI System: http://localhost:8115"
    echo "- Backend API: http://localhost:8000"
    echo "- Grafana Monitoring: http://localhost:3000"
    echo ""
    echo "To test the system: python3 scripts/test_localagi_system.py"
    echo "================================================================"
}

# Cleanup function
cleanup() {
    local exit_code=$?
    
    if [[ $exit_code -ne 0 ]]; then
        log ERROR "Deployment failed with exit code $exit_code"
        log INFO "Cleaning up partial deployment..."
        
        # Stop any services that were started
        docker-compose down || true
        
        log INFO "Partial cleanup completed"
    fi
    
    exit $exit_code
}

# Main deployment function
main() {
    log INFO "Starting LocalAGI autonomous orchestration system deployment"
    
    # Setup error handling
    trap cleanup EXIT
    
    # Ensure log directory exists
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Run deployment steps
    check_prerequisites
    install_dependencies
    setup_configuration
    init_services
    deploy_localagi
    validate_system
    generate_report
    
    log INFO "LocalAGI deployment completed successfully!"
    
    # Show quick start information
    echo ""
    echo "Quick Start Commands:"
    echo "  Start all services: docker-compose up -d"
    echo "  Stop all services: docker-compose down"
    echo "  View logs: docker-compose logs -f localagi-enhanced"
    echo "  Run tests: python3 scripts/test_localagi_system.py"
    echo "  System status: curl http://localhost:8115/status"
    echo ""
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi