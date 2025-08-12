#!/bin/bash
################################################################################
# ULTRA-DEPLOYMENT SCRIPT - SutazAI Master Deployment System
# Following all 19 CLAUDE.md rules - Single source of truth for deployment
# Self-updating, intelligent, comprehensive deployment solution
################################################################################

set -euo pipefail

# Configuration
readonly SCRIPT_VERSION="1.0.0"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly LOG_FILE="${PROJECT_ROOT}/logs/deploy_${TIMESTAMP}.log"
readonly BACKUP_DIR="${PROJECT_ROOT}/backups/deploy_${TIMESTAMP}"

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Environment detection
ENVIRONMENT="${ENVIRONMENT:-dev}"
DRY_RUN=false
FORCE=false
ROLLBACK=false
SELF_UPDATE=true

################################################################################
# LOGGING FUNCTIONS
################################################################################

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    
    # Create log directory if it doesn't exist
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Log to file
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    # Log to console with colors
    case "$level" in
        ERROR)
            echo -e "${RED}[ERROR]${NC} $message" >&2
            ;;
        WARN)
            echo -e "${YELLOW}[WARN]${NC} $message"
            ;;
        INFO)
            echo -e "${GREEN}[INFO]${NC} $message"
            ;;
        DEBUG)
            [[ "${DEBUG:-0}" == "1" ]] && echo -e "${BLUE}[DEBUG]${NC} $message"
            ;;
    esac
}

log_error() { log ERROR "$@"; }
log_warn() { log WARN "$@"; }
log_info() { log INFO "$@"; }
log_debug() { log DEBUG "$@"; }

################################################################################
# SELF-UPDATE MECHANISM
################################################################################

self_update() {
    if [[ "$SELF_UPDATE" == "false" ]]; then
        log_info "Self-update skipped (--no-update flag)"
        return 0
    fi
    
    log_info "Checking for script updates..."
    
    cd "$PROJECT_ROOT"
    
    # Fetch latest changes
    if git fetch origin main &>/dev/null; then
        local LOCAL_HASH=$(git rev-parse HEAD)
        local REMOTE_HASH=$(git rev-parse origin/main)
        
        if [[ "$LOCAL_HASH" != "$REMOTE_HASH" ]]; then
            log_info "Updates available. Pulling latest changes..."
            
            # Stash any local changes
            git stash push -m "deploy.sh auto-stash ${TIMESTAMP}" &>/dev/null || true
            
            # Pull latest
            if git pull origin main &>/dev/null; then
                log_info "Repository updated successfully"
                
                # Check if this script was updated
                if git diff HEAD~1 HEAD --name-only | grep -q "scripts/deploy.sh"; then
                    log_warn "Deploy script was updated. Restarting with new version..."
                    exec "$0" "$@"
                fi
            else
                log_error "Failed to pull updates"
                return 1
            fi
        else
            log_info "Already up to date"
        fi
    else
        log_warn "Could not fetch updates (offline mode?)"
    fi
}

################################################################################
# HEALTH CHECK FUNCTIONS
################################################################################

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local prereqs_met=true
    
    # Check required commands
    for cmd in docker docker-compose git curl jq python3; do
        if ! command -v "$cmd" &>/dev/null; then
            log_error "Required command not found: $cmd"
            prereqs_met=false
        fi
    done
    
    # Check Docker daemon
    if ! docker info &>/dev/null; then
        log_error "Docker daemon is not running"
        prereqs_met=false
    fi
    
    # Check Python version
    if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)" 2>/dev/null; then
        log_info "Python 3.12+ detected"
    else
        log_error "Python 3.12+ required"
        prereqs_met=false
    fi
    
    # Check disk space (need at least 10GB free)
    local available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 10485760 ]]; then
        log_error "Insufficient disk space (need at least 10GB free)"
        prereqs_met=false
    fi
    
    if [[ "$prereqs_met" == "false" ]]; then
        log_error "Prerequisites not met. Please fix the issues above."
        exit 1
    fi
    
    log_info "All prerequisites met"
}

check_service_health() {
    local service="$1"
    local url="$2"
    local max_retries="${3:-30}"
    local retry_delay="${4:-2}"
    
    log_info "Checking health of $service..."
    
    for i in $(seq 1 $max_retries); do
        if curl -s -f "$url" &>/dev/null; then
            log_info "$service is healthy"
            return 0
        fi
        
        if [[ $i -lt $max_retries ]]; then
            log_debug "Attempt $i/$max_retries failed, retrying in ${retry_delay}s..."
            sleep "$retry_delay"
        fi
    done
    
    log_error "$service health check failed after $max_retries attempts"
    return 1
}

verify_system_health() {
    log_info "Verifying system health..."
    
    local all_healthy=true
    
    # Core services
    check_service_health "Backend API" "http://localhost:10010/health" || all_healthy=false
    check_service_health "Frontend UI" "http://localhost:10011/" || all_healthy=false
    check_service_health "Ollama" "http://localhost:10104/api/tags" || all_healthy=false
    
    # Databases
    if docker exec sutazai-postgres pg_isready &>/dev/null; then
        log_info "PostgreSQL is healthy"
    else
        log_error "PostgreSQL is not healthy"
        all_healthy=false
    fi
    
    if [[ "$(docker exec sutazai-redis redis-cli ping 2>/dev/null)" == "PONG" ]]; then
        log_info "Redis is healthy"
    else
        log_error "Redis is not healthy"
        all_healthy=false
    fi
    
    if [[ "$all_healthy" == "true" ]]; then
        log_info "All services are healthy"
        return 0
    else
        log_error "Some services are unhealthy"
        return 1
    fi
}

################################################################################
# BACKUP FUNCTIONS
################################################################################

create_backup() {
    log_info "Creating backup..."
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup database
    log_info "Backing up PostgreSQL..."
    docker exec sutazai-postgres pg_dump -U sutazai sutazai \
        | gzip > "$BACKUP_DIR/postgres_backup.sql.gz"
    
    # Backup Redis
    log_info "Backing up Redis..."
    docker exec sutazai-redis redis-cli BGSAVE &>/dev/null
    sleep 2
    docker cp sutazai-redis:/data/dump.rdb "$BACKUP_DIR/redis_backup.rdb"
    
    # Backup configurations
    log_info "Backing up configurations..."
    cp -r "$PROJECT_ROOT/config" "$BACKUP_DIR/config"
    cp "$PROJECT_ROOT/docker-compose.yml" "$BACKUP_DIR/"
    cp "$PROJECT_ROOT/.env" "$BACKUP_DIR/.env" 2>/dev/null || true
    
    # Create restore script
    cat > "$BACKUP_DIR/restore.sh" << 'EOF'
#!/bin/bash
# Restore script for backup
set -e

BACKUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$BACKUP_DIR")")"

echo "Restoring from backup: $BACKUP_DIR"

# Restore database
echo "Restoring PostgreSQL..."
gunzip -c "$BACKUP_DIR/postgres_backup.sql.gz" | \
    docker exec -i sutazai-postgres psql -U sutazai sutazai

# Restore Redis
echo "Restoring Redis..."
docker cp "$BACKUP_DIR/redis_backup.rdb" sutazai-redis:/data/dump.rdb
docker exec sutazai-redis redis-cli SHUTDOWN NOSAVE
docker restart sutazai-redis

echo "Restore complete!"
EOF
    
    chmod +x "$BACKUP_DIR/restore.sh"
    
    log_info "Backup created: $BACKUP_DIR"
}

################################################################################
# DEPLOYMENT FUNCTIONS
################################################################################

deploy_environment() {
    local env="$1"
    
    log_info "Deploying environment: $env"
    
    # Set environment-specific variables
    case "$env" in
        dev)
            export COMPOSE_FILE="docker-compose.yml"
            export COMPOSE_PROJECT_NAME="sutazai-dev"
            ;;
        staging)
            export COMPOSE_FILE="docker-compose.yml:docker-compose.staging.yml"
            export COMPOSE_PROJECT_NAME="sutazai-staging"
            ;;
        production)
            export COMPOSE_FILE="docker-compose.yml:docker-compose.production.yml"
            export COMPOSE_PROJECT_NAME="sutazai-prod"
            
            # Generate secure secrets for production
            if [[ ! -f "$PROJECT_ROOT/.env.production.secure" ]]; then
                log_info "Generating secure production secrets..."
                python3 "$PROJECT_ROOT/scripts/generate_secure_secrets.py"
            fi
            
            # Use production env
            cp "$PROJECT_ROOT/.env.production.secure" "$PROJECT_ROOT/.env"
            ;;
        *)
            log_error "Unknown environment: $env"
            exit 1
            ;;
    esac
    
    # Pull latest images
    log_info "Pulling latest Docker images..."
    docker-compose pull --quiet
    
    # Start services with proper order
    log_info "Starting core databases..."
    docker-compose up -d postgres redis neo4j
    sleep 10
    
    log_info "Starting message queue..."
    docker-compose up -d rabbitmq
    sleep 5
    
    log_info "Starting vector databases..."
    docker-compose up -d qdrant chromadb faiss
    sleep 5
    
    log_info "Starting Ollama..."
    docker-compose up -d ollama
    
    # Wait for Ollama and load model
    log_info "Waiting for Ollama to start..."
    check_service_health "Ollama" "http://localhost:10104/api/tags" 60 2
    
    # Ensure TinyLlama model is loaded
    log_info "Loading TinyLlama model..."
    docker exec sutazai-ollama ollama pull tinyllama:latest || true
    
    log_info "Starting monitoring stack..."
    docker-compose up -d prometheus grafana loki alertmanager
    
    log_info "Starting application services..."
    docker-compose up -d backend frontend
    
    log_info "Starting agent services..."
    docker-compose up -d ai-agent-orchestrator hardware-resource-optimizer \
        jarvis-automation-agent jarvis-hardware-optimizer \
        ollama-integration resource-arbitration-agent \
        task-assignment-coordinator
    
    # Verify deployment
    sleep 10
    if verify_system_health; then
        log_info "Deployment successful!"
    else
        log_error "Deployment completed but some services are unhealthy"
        return 1
    fi
}

rollback_deployment() {
    log_info "Rolling back deployment..."
    
    # Find latest backup
    local latest_backup=$(ls -td "$PROJECT_ROOT"/backups/deploy_* 2>/dev/null | head -1)
    
    if [[ -z "$latest_backup" ]]; then
        log_error "No backup found for rollback"
        exit 1
    fi
    
    log_info "Using backup: $latest_backup"
    
    # Execute restore script
    if [[ -x "$latest_backup/restore.sh" ]]; then
        "$latest_backup/restore.sh"
    else
        log_error "Restore script not found or not executable"
        exit 1
    fi
    
    # Restart services
    docker-compose restart
    
    log_info "Rollback complete"
}

################################################################################
# MAIN EXECUTION
################################################################################

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

ULTRA-DEPLOYMENT SCRIPT for SutazAI System

Options:
    -e, --environment ENV    Set environment (dev|staging|production) [default: dev]
    -d, --dry-run           Show what would be deployed without doing it
    -f, --force             Skip confirmation prompts
    -r, --rollback          Rollback to previous deployment
    -n, --no-update         Skip self-update check
    -h, --help              Show this help message

Examples:
    $0                      # Deploy dev environment
    $0 -e production        # Deploy production environment
    $0 --rollback           # Rollback to previous deployment
    $0 -e staging --dry-run # Preview staging deployment

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
            -n|--no-update)
                SELF_UPDATE=false
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

main() {
    log_info "=== ULTRA-DEPLOYMENT SCRIPT v${SCRIPT_VERSION} ==="
    log_info "Starting deployment process..."
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Self-update
    self_update "$@"
    
    # Check prerequisites
    check_prerequisites
    
    # Handle rollback
    if [[ "$ROLLBACK" == "true" ]]; then
        rollback_deployment
        exit $?
    fi
    
    # Dry run mode
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN MODE - No changes will be made"
        log_info "Would deploy environment: $ENVIRONMENT"
        log_info "Would create backup in: $BACKUP_DIR"
        exit 0
    fi
    
    # Confirmation prompt
    if [[ "$FORCE" == "false" && "$ENVIRONMENT" == "production" ]]; then
        echo -e "${YELLOW}⚠️  WARNING: Deploying to PRODUCTION${NC}"
        echo -n "Type 'DEPLOY' to confirm: "
        read confirmation
        if [[ "$confirmation" != "DEPLOY" ]]; then
            log_info "Deployment cancelled"
            exit 0
        fi
    fi
    
    # Create backup before deployment
    create_backup
    
    # Deploy
    if deploy_environment "$ENVIRONMENT"; then
        log_info "=== DEPLOYMENT SUCCESSFUL ==="
        log_info "Environment: $ENVIRONMENT"
        log_info "Backup location: $BACKUP_DIR"
        log_info "Log file: $LOG_FILE"
        
        # Show access URLs
        echo -e "\n${GREEN}Access URLs:${NC}"
        echo "  Backend API: http://localhost:10010"
        echo "  Frontend UI: http://localhost:10011"
        echo "  Grafana: http://localhost:10201 (admin/admin)"
        echo "  Prometheus: http://localhost:10200"
        
        exit 0
    else
        log_error "=== DEPLOYMENT FAILED ==="
        log_error "Check log file: $LOG_FILE"
        log_error "To rollback, run: $0 --rollback"
        exit 1
    fi
}

# Trap errors
trap 'log_error "Script failed at line $LINENO"' ERR

# Run main function
main "$@"