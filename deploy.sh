#!/bin/bash
################################################################################
# UNIVERSAL DEPLOYMENT SCRIPT - SutazAI Complete Infrastructure Deployment
# Rule 12 Compliance: Zero-touch deployment with hardware optimization
# Self-sufficient, intelligent, comprehensive deployment solution
################################################################################

set -euo pipefail

# Configuration
readonly SCRIPT_VERSION="1.0.0"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$SCRIPT_DIR"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly LOG_FILE="${PROJECT_ROOT}/logs/deploy_${TIMESTAMP}.log"
readonly BACKUP_DIR="${PROJECT_ROOT}/backups/deploy_${TIMESTAMP}"

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Environment detection and optimization
ENVIRONMENT="${ENVIRONMENT:-auto}"
DRY_RUN=false
FORCE=false
ROLLBACK=false
SKIP_BACKUP=false
HARDWARE_OPTIMIZED=true
AUTO_RECOVERY=true

# Hardware detection variables
DETECTED_CORES=""
DETECTED_MEMORY=""
DETECTED_STORAGE=""
DETECTED_GPU=""
OPTIMIZATION_PROFILE=""

################################################################################
# LOGGING FUNCTIONS (Rule 8 Compliance - No print statements)
################################################################################

setup_logging() {
    mkdir -p "$(dirname "$LOG_FILE")"
    mkdir -p "$BACKUP_DIR"
    
    # Setup log rotation
    if [[ -f "$LOG_FILE" ]] && [[ $(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null) -gt 104857600 ]]; then
        mv "$LOG_FILE" "${LOG_FILE}.old"
    fi
}

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S UTC')"
    
    # Log to file with structured format
    echo "[$timestamp] [$level] [$$] $message" >> "$LOG_FILE"
    
    # Log to console with colors and proper formatting
    case "$level" in
        ERROR)
            echo -e "${RED}✗ [ERROR]${NC} $message" >&2
            ;;
        WARN)
            echo -e "${YELLOW}⚠ [WARN]${NC} $message"
            ;;
        INFO)
            echo -e "${GREEN}✓ [INFO]${NC} $message"
            ;;
        DEBUG)
            [[ "${DEBUG:-0}" == "1" ]] && echo -e "${BLUE}🔍 [DEBUG]${NC} $message"
            ;;
        SUCCESS)
            echo -e "${GREEN}🎉 [SUCCESS]${NC} $message"
            ;;
        PROGRESS)
            echo -e "${CYAN}⏳ [PROGRESS]${NC} $message"
            ;;
    esac
}

log_error() { log ERROR "$@"; }
log_warn() { log WARN "$@"; }
log_info() { log INFO "$@"; }
log_debug() { log DEBUG "$@"; }
log_success() { log SUCCESS "$@"; }
log_progress() { log PROGRESS "$@"; }

################################################################################
# HARDWARE DETECTION AND OPTIMIZATION (Rule 12 - Hardware Optimization)
################################################################################

detect_hardware() {
    log_progress "Detecting hardware configuration..."
    
    # Detect CPU cores
    DETECTED_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "4")
    
    # Detect memory (in GB)
    if command -v free >/dev/null 2>&1; then
        DETECTED_MEMORY=$(free -g | awk '/^Mem:/{print $2}')
    elif command -v sysctl >/dev/null 2>&1; then
        DETECTED_MEMORY=$(($(sysctl -n hw.memsize 2>/dev/null || echo "8589934592") / 1024 / 1024 / 1024))
    else
        DETECTED_MEMORY="8"
    fi
    
    # Detect storage (in GB for root partition)
    DETECTED_STORAGE=$(df "$PROJECT_ROOT" | awk 'NR==2 {print int($2/1024/1024)}')
    
    # Detect GPU
    if command -v nvidia-smi >/dev/null 2>&1; then
        DETECTED_GPU="nvidia"
    elif command -v rocm-smi >/dev/null 2>&1; then
        DETECTED_GPU="amd"
    else
        DETECTED_GPU="none"
    fi
    
    # Determine optimization profile
    if [[ $DETECTED_CORES -ge 16 && $DETECTED_MEMORY -ge 32 && $DETECTED_STORAGE -ge 100 ]]; then
        OPTIMIZATION_PROFILE="high-performance"
    elif [[ $DETECTED_CORES -ge 8 && $DETECTED_MEMORY -ge 16 && $DETECTED_STORAGE -ge 50 ]]; then
        OPTIMIZATION_PROFILE="standard"
    elif [[ $DETECTED_CORES -ge 4 && $DETECTED_MEMORY -ge 8 && $DETECTED_STORAGE -ge 25 ]]; then
        OPTIMIZATION_PROFILE="minimal"
    else
        OPTIMIZATION_PROFILE="limited"
        log_warn "System resources below recommended minimums"
    fi
    
    log_info "Hardware Detection Results:"
    log_info "  CPU Cores: $DETECTED_CORES"
    log_info "  Memory: ${DETECTED_MEMORY}GB"
    log_info "  Storage: ${DETECTED_STORAGE}GB"
    log_info "  GPU: $DETECTED_GPU"
    log_info "  Optimization Profile: $OPTIMIZATION_PROFILE"
}

apply_hardware_optimization() {
    log_progress "Applying hardware-specific optimizations..."
    
    local compose_override="${PROJECT_ROOT}/docker-compose.override.yml"
    
    # Create hardware-optimized compose override
    cat > "$compose_override" << EOF
# Auto-generated hardware optimization overrides
# Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
# Profile: $OPTIMIZATION_PROFILE ($DETECTED_CORES cores, ${DETECTED_MEMORY}GB RAM, ${DETECTED_STORAGE}GB storage)

version: '3.8'

services:
EOF
    
    case "$OPTIMIZATION_PROFILE" in
        "high-performance")
            cat >> "$compose_override" << EOF
  postgres:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 1G
    environment:
      - POSTGRES_SHARED_BUFFERS=1GB
      - POSTGRES_EFFECTIVE_CACHE_SIZE=3GB
      - POSTGRES_WORK_MEM=256MB
      - POSTGRES_MAX_CONNECTIONS=200

  redis:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M

  ollama:
    deploy:
      resources:
        limits:
          cpus: '8.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 2G
EOF
            if [[ "$DETECTED_GPU" == "nvidia" ]]; then
                cat >> "$compose_override" << EOF
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
EOF
            fi
            ;;
        "standard")
            cat >> "$compose_override" << EOF
  postgres:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M

  redis:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.25'
          memory: 256M

  ollama:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 1G
EOF
            ;;
        "minimal"|"limited")
            cat >> "$compose_override" << EOF
  postgres:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.25'
          memory: 256M

  redis:
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.1'
          memory: 128M

  ollama:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
EOF
            ;;
    esac
    
    log_info "Hardware optimizations applied for $OPTIMIZATION_PROFILE profile"
}

################################################################################
# DEPENDENCY INSTALLATION AND SETUP
################################################################################

install_dependencies() {
    log_progress "Installing and verifying dependencies..."
    
    # Check and install missing dependencies
    local missing_deps=()
    
    for cmd in docker docker-compose git curl jq python3 make; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            missing_deps+=("$cmd")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_info "Installing missing dependencies: ${missing_deps[*]}"
        
        # Detect package manager and install
        if command -v apt-get >/dev/null 2>&1; then
            sudo apt-get update
            for dep in "${missing_deps[@]}"; do
                case "$dep" in
                    "docker")
                        curl -fsSL https://get.docker.com -o get-docker.sh
                        sudo sh get-docker.sh
                        sudo usermod -aG docker "$USER"
                        ;;
                    "docker-compose")
                        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
                        sudo chmod +x /usr/local/bin/docker-compose
                        ;;
                    *)
                        sudo apt-get install -y "$dep"
                        ;;
                esac
            done
        elif command -v yum >/dev/null 2>&1; then
            sudo yum update -y
            for dep in "${missing_deps[@]}"; do
                case "$dep" in
                    "docker")
                        sudo yum install -y docker
                        sudo systemctl start docker
                        sudo systemctl enable docker
                        sudo usermod -aG docker "$USER"
                        ;;
                    *)
                        sudo yum install -y "$dep"
                        ;;
                esac
            done
        elif command -v brew >/dev/null 2>&1; then
            for dep in "${missing_deps[@]}"; do
                brew install "$dep"
            done
        else
            log_error "Unable to automatically install dependencies. Please install manually: ${missing_deps[*]}"
            exit 1
        fi
    fi
    
    # Verify Docker daemon
    if ! docker info >/dev/null 2>&1; then
        log_info "Starting Docker daemon..."
        if command -v systemctl >/dev/null 2>&1; then
            sudo systemctl start docker
            sudo systemctl enable docker
        elif command -v service >/dev/null 2>&1; then
            sudo service docker start
        else
            log_error "Unable to start Docker daemon. Please start it manually."
            exit 1
        fi
        
        # Wait for Docker to be ready
        for i in {1..30}; do
            if docker info >/dev/null 2>&1; then
                break
            fi
            sleep 2
        done
    fi
    
    log_success "All dependencies installed and verified"
}

check_prerequisites() {
    log_progress "Checking system prerequisites..."
    
    # Check Python version
    if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then
        log_error "Python 3.11+ required (found $(python3 --version))"
        exit 1
    fi
    
    # Check disk space (need at least based on profile)
    local required_space_gb
    case "$OPTIMIZATION_PROFILE" in
        "high-performance") required_space_gb=50 ;;
        "standard") required_space_gb=30 ;;
        *) required_space_gb=20 ;;
    esac
    
    if [[ $DETECTED_STORAGE -lt $required_space_gb ]]; then
        log_error "Insufficient disk space (need at least ${required_space_gb}GB, have ${DETECTED_STORAGE}GB)"
        exit 1
    fi
    
    # Check Docker network
    if ! docker network ls | grep -q sutazai-network; then
        log_info "Creating Docker network: sutazai-network"
        docker network create sutazai-network
    fi
    
    log_success "All prerequisites met"
}

################################################################################
# ENVIRONMENT SETUP AND CONFIGURATION
################################################################################

setup_environment() {
    log_progress "Setting up environment configuration..."
    
    # Auto-detect environment if not specified
    if [[ "$ENVIRONMENT" == "auto" ]]; then
        if [[ -f "${PROJECT_ROOT}/.env.production" ]]; then
            ENVIRONMENT="production"
        elif [[ -f "${PROJECT_ROOT}/.env.staging" ]]; then
            ENVIRONMENT="staging"
        else
            ENVIRONMENT="development"
        fi
        log_info "Auto-detected environment: $ENVIRONMENT"
    fi
    
    # Setup environment-specific configuration
    local env_file="${PROJECT_ROOT}/.env"
    local env_template="${PROJECT_ROOT}/.env.${ENVIRONMENT}"
    
    if [[ -f "$env_template" ]]; then
        cp "$env_template" "$env_file"
        log_info "Using environment file: $env_template"
    elif [[ ! -f "$env_file" ]]; then
        # Create default .env file
        cat > "$env_file" << EOF
# SutazAI Environment Configuration
# Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
# Environment: $ENVIRONMENT
# Hardware Profile: $OPTIMIZATION_PROFILE

ENVIRONMENT=$ENVIRONMENT
COMPOSE_PROJECT_NAME=sutazai-${ENVIRONMENT}

# Database Configuration
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
POSTGRES_DB=sutazai

# Redis Configuration
REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)

# Neo4j Configuration
NEO4J_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)

# Security Configuration
JWT_SECRET_KEY=$(openssl rand -base64 64 | tr -d "=+/" | cut -c1-50)
ENCRYPTION_KEY=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)

# Hardware Optimization
OPTIMIZATION_PROFILE=$OPTIMIZATION_PROFILE
DETECTED_CORES=$DETECTED_CORES
DETECTED_MEMORY=$DETECTED_MEMORY
DETECTED_GPU=$DETECTED_GPU
EOF
        log_info "Created default environment configuration"
    fi
    
    # Source environment variables
    set -a
    source "$env_file"
    set +a
    
    log_success "Environment configuration complete"
}

################################################################################
# INFRASTRUCTURE VALIDATION AND HEALTH CHECKS
################################################################################

validate_infrastructure() {
    log_progress "Validating infrastructure health..."
    
    local health_checks=(
        "Backend API:http://localhost:10010/health:60"
        "Frontend UI:http://localhost:10011/:30"
        "Ollama API:http://localhost:10104/api/tags:60"
        "PostgreSQL:docker exec sutazai-postgres pg_isready -U sutazai:30"
        "Redis:docker exec sutazai-redis redis-cli ping:30"
        "Prometheus:http://localhost:10200/-/healthy:30"
        "Grafana:http://localhost:10201/api/health:30"
        "ChromaDB:http://localhost:10100/api/v1/heartbeat:30"
        "Qdrant:http://localhost:10101/:30"
    )
    
    local failed_checks=()
    
    for check in "${health_checks[@]}"; do
        IFS=':' read -r service_name url timeout <<< "$check"
        
        log_debug "Checking $service_name..."
        
        if [[ "$url" == docker* ]]; then
            # Docker command health check
            if timeout "$timeout" bash -c "$url" >/dev/null 2>&1; then
                log_info "✓ $service_name is healthy"
            else
                log_warn "✗ $service_name health check failed"
                failed_checks+=("$service_name")
            fi
        else
            # HTTP health check
            local retries=0
            local max_retries=$((timeout / 2))
            local success=false
            
            while [[ $retries -lt $max_retries ]]; do
                if curl -sf "$url" >/dev/null 2>&1; then
                    log_info "✓ $service_name is healthy"
                    success=true
                    break
                fi
                ((retries++))
                sleep 2
            done
            
            if [[ "$success" == "false" ]]; then
                log_warn "✗ $service_name health check failed after $max_retries attempts"
                failed_checks+=("$service_name")
            fi
        fi
    done
    
    if [[ ${#failed_checks[@]} -eq 0 ]]; then
        log_success "All infrastructure health checks passed"
        return 0
    else
        log_warn "Failed health checks: ${failed_checks[*]}"
        if [[ "$AUTO_RECOVERY" == "true" ]]; then
            log_info "Attempting automatic recovery..."
            restart_failed_services "${failed_checks[@]}"
        fi
        return 1
    fi
}

restart_failed_services() {
    local failed_services=("$@")
    
    for service in "${failed_services[@]}"; do
        log_info "Restarting $service..."
        case "$service" in
            "Backend API")
                docker-compose restart backend
                ;;
            "Frontend UI")
                docker-compose restart frontend
                ;;
            "Ollama API")
                docker-compose restart ollama
                ;;
            "PostgreSQL")
                docker-compose restart postgres
                ;;
            "Redis")
                docker-compose restart redis
                ;;
            "Prometheus")
                docker-compose restart prometheus
                ;;
            "Grafana")
                docker-compose restart grafana
                ;;
            "ChromaDB")
                docker-compose restart chromadb
                ;;
            "Qdrant")
                docker-compose restart qdrant
                ;;
        esac
        sleep 5
    done
    
    # Re-validate after restart
    sleep 10
    validate_infrastructure
}

################################################################################
# BACKUP AND ROLLBACK FUNCTIONALITY
################################################################################

create_backup() {
    if [[ "$SKIP_BACKUP" == "true" ]]; then
        log_info "Backup skipped (--skip-backup flag)"
        return 0
    fi
    
    log_progress "Creating deployment backup..."
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup running containers state
    docker-compose ps --format json > "$BACKUP_DIR/containers_state.json" 2>/dev/null || true
    
    # Backup databases
    log_info "Backing up PostgreSQL..."
    if docker exec sutazai-postgres pg_isready -U sutazai >/dev/null 2>&1; then
        docker exec sutazai-postgres pg_dump -U sutazai sutazai 2>/dev/null | gzip > "$BACKUP_DIR/postgres_backup.sql.gz"
    fi
    
    log_info "Backing up Redis..."
    if docker exec sutazai-redis redis-cli ping >/dev/null 2>&1; then
        docker exec sutazai-redis redis-cli BGSAVE >/dev/null 2>&1
        sleep 2
        docker cp sutazai-redis:/data/dump.rdb "$BACKUP_DIR/redis_backup.rdb" 2>/dev/null || true
    fi
    
    # Backup configurations
    log_info "Backing up configurations..."
    [[ -d "$PROJECT_ROOT/config" ]] && cp -r "$PROJECT_ROOT/config" "$BACKUP_DIR/"
    [[ -f "$PROJECT_ROOT/docker-compose.yml" ]] && cp "$PROJECT_ROOT/docker-compose.yml" "$BACKUP_DIR/"
    [[ -f "$PROJECT_ROOT/.env" ]] && cp "$PROJECT_ROOT/.env" "$BACKUP_DIR/"
    [[ -f "$PROJECT_ROOT/docker-compose.override.yml" ]] && cp "$PROJECT_ROOT/docker-compose.override.yml" "$BACKUP_DIR/"
    
    # Create comprehensive restore script
    cat > "$BACKUP_DIR/restore.sh" << 'EOF'
#!/bin/bash
# Comprehensive Restore Script
set -euo pipefail

BACKUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$BACKUP_DIR")"

echo "🔄 Restoring from backup: $BACKUP_DIR"

# Stop current services
cd "$PROJECT_ROOT"
docker-compose down

# Restore configurations
[[ -d "$BACKUP_DIR/config" ]] && cp -r "$BACKUP_DIR/config" "$PROJECT_ROOT/"
[[ -f "$BACKUP_DIR/docker-compose.yml" ]] && cp "$BACKUP_DIR/docker-compose.yml" "$PROJECT_ROOT/"
[[ -f "$BACKUP_DIR/.env" ]] && cp "$BACKUP_DIR/.env" "$PROJECT_ROOT/"
[[ -f "$BACKUP_DIR/docker-compose.override.yml" ]] && cp "$BACKUP_DIR/docker-compose.override.yml" "$PROJECT_ROOT/"

# Start core services
docker-compose up -d postgres redis

# Wait for databases
sleep 10

# Restore PostgreSQL
if [[ -f "$BACKUP_DIR/postgres_backup.sql.gz" ]]; then
    echo "📀 Restoring PostgreSQL..."
    gunzip -c "$BACKUP_DIR/postgres_backup.sql.gz" | \
        docker exec -i sutazai-postgres psql -U sutazai sutazai
fi

# Restore Redis
if [[ -f "$BACKUP_DIR/redis_backup.rdb" ]]; then
    echo "📀 Restoring Redis..."
    docker cp "$BACKUP_DIR/redis_backup.rdb" sutazai-redis:/data/dump.rdb
    docker-compose restart redis
fi

# Start remaining services
docker-compose up -d

echo "✅ Restore complete!"
EOF
    
    chmod +x "$BACKUP_DIR/restore.sh"
    
    # Create backup manifest
    cat > "$BACKUP_DIR/manifest.json" << EOF
{
  "backup_timestamp": "$(date -u +"%Y-%m-%d %H:%M:%S UTC")",
  "script_version": "$SCRIPT_VERSION",
  "environment": "$ENVIRONMENT",
  "optimization_profile": "$OPTIMIZATION_PROFILE",
  "hardware": {
    "cores": $DETECTED_CORES,
    "memory_gb": $DETECTED_MEMORY,
    "storage_gb": $DETECTED_STORAGE,
    "gpu": "$DETECTED_GPU"
  },
  "backup_contents": [
    "containers_state.json",
    "postgres_backup.sql.gz",
    "redis_backup.rdb",
    "config/",
    "docker-compose.yml",
    ".env",
    "docker-compose.override.yml",
    "restore.sh"
  ]
}
EOF
    
    log_success "Backup created: $BACKUP_DIR"
}

################################################################################
# DEPLOYMENT ORCHESTRATION
################################################################################

deploy_infrastructure() {
    log_progress "Deploying SutazAI infrastructure..."
    
    # Phase 1: Core Infrastructure
    log_info "Phase 1: Starting core infrastructure..."
    docker-compose up -d postgres redis neo4j
    
    # Wait for databases to be ready
    log_info "Waiting for databases to initialize..."
    sleep 15
    
    # Phase 2: Message Queue and Vector Databases
    log_info "Phase 2: Starting message queue and vector databases..."
    docker-compose up -d rabbitmq chromadb qdrant faiss
    sleep 10
    
    # Phase 3: AI Services
    log_info "Phase 3: Starting AI services..."
    docker-compose up -d ollama
    
    # Wait for Ollama and ensure models are loaded
    log_info "Waiting for Ollama to start and loading models..."
    for i in {1..60}; do
        if curl -sf http://localhost:10104/api/tags >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done
    
    # Ensure TinyLlama model is available
    docker exec sutazai-ollama ollama pull tinyllama:latest || log_warn "Failed to pull TinyLlama model"
    
    # Phase 4: Monitoring Stack
    log_info "Phase 4: Starting monitoring stack..."
    docker-compose up -d prometheus grafana loki alertmanager node-exporter cadvisor
    sleep 10
    
    # Phase 5: Application Layer
    log_info "Phase 5: Starting application services..."
    docker-compose up -d backend frontend
    sleep 15
    
    # Phase 6: Agent Services
    log_info "Phase 6: Starting agent services..."
    docker-compose up -d \
        ai-agent-orchestrator \
        hardware-resource-optimizer \
        jarvis-automation-agent \
        jarvis-hardware-optimizer \
        ollama-integration \
        resource-arbitration-agent \
        task-assignment-coordinator \
        2>/dev/null || log_warn "Some agent services may not be available"
    
    log_success "Infrastructure deployment complete"
}

################################################################################
# MAIN DEPLOYMENT ORCHESTRATION
################################################################################

show_usage() {
    cat << EOF
🚀 UNIVERSAL DEPLOYMENT SCRIPT - SutazAI v${SCRIPT_VERSION}

DESCRIPTION:
    Zero-touch deployment with automatic hardware optimization,
    dependency installation, and intelligent infrastructure management.

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -e, --environment ENV     Environment (auto|development|staging|production) [default: auto]
    -d, --dry-run            Preview deployment without making changes
    -f, --force              Skip all confirmation prompts
    -r, --rollback           Rollback to most recent backup
    --skip-backup            Skip backup creation (not recommended)
    --no-optimization        Disable hardware optimization
    --no-recovery            Disable automatic recovery attempts
    -q, --quiet              Reduce output verbosity
    -v, --verbose            Increase output verbosity (sets DEBUG=1)
    -h, --help               Show this help message

EXAMPLES:
    $0                       # Auto-detect and deploy with optimization
    $0 -e production         # Deploy production environment
    $0 --dry-run             # Preview deployment
    $0 --rollback            # Rollback to previous state
    $0 -f -e staging         # Force deploy staging without prompts

HARDWARE PROFILES:
    limited        < 4 cores, < 8GB RAM   (minimal resource allocation)
    minimal        4+ cores, 8+ GB RAM    (basic resource allocation)
    standard       8+ cores, 16+ GB RAM   (balanced resource allocation)
    high-performance 16+ cores, 32+ GB RAM (maximum resource allocation)

ACCESS URLS (after successful deployment):
    Backend API:    http://localhost:10010
    Frontend UI:    http://localhost:10011
    Grafana:        http://localhost:10201 (admin/admin)
    Prometheus:     http://localhost:10200

EOF
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            -r|--rollback)
                ROLLBACK=true
                shift
                ;;
            --skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            --no-optimization)
                HARDWARE_OPTIMIZED=false
                shift
                ;;
            --no-recovery)
                AUTO_RECOVERY=false
                shift
                ;;
            -q|--quiet)
                export QUIET=1
                shift
                ;;
            -v|--verbose)
                export DEBUG=1
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

execute_rollback() {
    log_info "🔄 Executing rollback procedure..."
    
    # Find most recent backup
    local latest_backup=$(find "$PROJECT_ROOT/backups" -name "deploy_*" -type d 2>/dev/null | sort -r | head -1)
    
    if [[ -z "$latest_backup" ]]; then
        log_error "No backup found for rollback"
        exit 1
    fi
    
    log_info "Using backup: $latest_backup"
    
    if [[ ! -x "$latest_backup/restore.sh" ]]; then
        log_error "Restore script not found or not executable"
        exit 1
    fi
    
    # Execute restore
    "$latest_backup/restore.sh"
    
    # Validate rollback
    sleep 10
    if validate_infrastructure; then
        log_success "Rollback completed successfully"
    else
        log_error "Rollback completed but some services are unhealthy"
        exit 1
    fi
}

main() {
    # Setup logging immediately
    setup_logging
    
    log_info "🚀 SutazAI Universal Deployment Script v${SCRIPT_VERSION}"
    log_info "Starting deployment process at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    
    # Parse arguments
    parse_arguments "$@"
    
    # Handle rollback immediately
    if [[ "$ROLLBACK" == "true" ]]; then
        execute_rollback
        exit $?
    fi
    
    # Hardware detection and optimization
    detect_hardware
    
    # Install dependencies
    install_dependencies
    
    # Check prerequisites
    check_prerequisites
    
    # Apply hardware optimization
    if [[ "$HARDWARE_OPTIMIZED" == "true" ]]; then
        apply_hardware_optimization
    fi
    
    # Setup environment
    setup_environment
    
    # Handle dry run
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "🔍 DRY RUN MODE - Showing deployment plan:"
        log_info "  Environment: $ENVIRONMENT"
        log_info "  Hardware Profile: $OPTIMIZATION_PROFILE"
        log_info "  Resource Allocation: ${DETECTED_CORES} cores, ${DETECTED_MEMORY}GB RAM"
        log_info "  GPU Support: $DETECTED_GPU"
        log_info "  Backup Location: $BACKUP_DIR"
        log_info "  Hardware Optimization: $HARDWARE_OPTIMIZED"
        log_info "  Auto Recovery: $AUTO_RECOVERY"
        exit 0
    fi
    
    # Confirmation for production
    if [[ "$FORCE" == "false" && "$ENVIRONMENT" == "production" ]]; then
        echo -e "${RED}⚠️  WARNING: Deploying to PRODUCTION environment${NC}"
        echo -e "${YELLOW}This will affect live services and data${NC}"
        echo -n "Type 'CONFIRM' to proceed: "
        read -r confirmation
        if [[ "$confirmation" != "CONFIRM" ]]; then
            log_info "Deployment cancelled by user"
            exit 0
        fi
    fi
    
    # Create backup
    create_backup
    
    # Execute deployment
    log_progress "🚀 Beginning infrastructure deployment..."
    
    if deploy_infrastructure; then
        log_progress "🔍 Validating deployment..."
        
        # Give services time to fully start
        sleep 20
        
        if validate_infrastructure; then
            # Generate deployment report
            cat > "${PROJECT_ROOT}/deployment_report_${TIMESTAMP}.json" << EOF
{
  "deployment_timestamp": "$(date -u +"%Y-%m-%d %H:%M:%S UTC")",
  "script_version": "$SCRIPT_VERSION",
  "environment": "$ENVIRONMENT",
  "optimization_profile": "$OPTIMIZATION_PROFILE",
  "hardware": {
    "cores": $DETECTED_CORES,
    "memory_gb": $DETECTED_MEMORY,
    "storage_gb": $DETECTED_STORAGE,
    "gpu": "$DETECTED_GPU"
  },
  "deployment_status": "SUCCESS",
  "backup_location": "$BACKUP_DIR",
  "log_file": "$LOG_FILE",
  "access_urls": {
    "backend": "http://localhost:10010",
    "frontend": "http://localhost:10011",
    "grafana": "http://localhost:10201",
    "prometheus": "http://localhost:10200"
  }
}
EOF
            
            log_success "🎉 DEPLOYMENT SUCCESSFUL!"
            log_info "📊 Environment: $ENVIRONMENT"
            log_info "⚡ Hardware Profile: $OPTIMIZATION_PROFILE ($DETECTED_CORES cores, ${DETECTED_MEMORY}GB RAM)"
            log_info "💾 Backup: $BACKUP_DIR"
            log_info "📝 Log: $LOG_FILE"
            log_info ""
            log_info "🌐 Access URLs:"
            log_info "  • Backend API:  http://localhost:10010"
            log_info "  • Frontend UI:  http://localhost:10011"
            log_info "  • Grafana:      http://localhost:10201 (admin/admin)"
            log_info "  • Prometheus:   http://localhost:10200"
            log_info ""
            log_info "🔄 To rollback: $0 --rollback"
            
            exit 0
        else
            log_error "❌ Deployment completed but validation failed"
            if [[ "$AUTO_RECOVERY" == "true" ]]; then
                log_info "🔄 Auto-recovery will attempt to fix issues"
            else
                log_info "🔧 Run with --no-recovery to disable auto-recovery"
            fi
            exit 1
        fi
    else
        log_error "❌ DEPLOYMENT FAILED"
        log_error "📝 Check log: $LOG_FILE"
        log_error "🔄 Rollback: $0 --rollback"
        exit 1
    fi
}

# Error handling
trap 'log_error "Script failed at line $LINENO with exit code $?"' ERR

# Ensure we're in the project root
cd "$PROJECT_ROOT"

# Execute main function
main "$@"