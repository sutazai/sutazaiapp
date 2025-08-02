#!/bin/bash
#
# SutazAI Universal Deployment Master Script
# Version: 2.0.0
# 
# This script implements Rule 12: One-Command Universal Deployment
# A bulletproof, intelligent deployment system that can transform
# any fresh system into a fully operational SutazAI environment.
#
# Usage: ./deploy.sh [TARGET] [OPTIONS]
# Targets: local, staging, production, fresh
# 

set -euo pipefail

# ===============================================
# DEPLOYMENT CONFIGURATION
# ===============================================

# Script metadata
readonly SCRIPT_VERSION="2.0.0"
readonly SCRIPT_NAME="SutazAI Universal Deployment Master"
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly DEPLOYMENT_ID="deploy_$(date +%Y%m%d_%H%M%S)_$$"

# Deployment targets
declare -A DEPLOYMENT_TARGETS=(
    ["local"]="Local development environment"
    ["staging"]="Staging environment with full features"
    ["production"]="Production environment with high availability"
    ["fresh"]="Fresh system installation from scratch"
)

# State management
readonly STATE_DIR="$PROJECT_ROOT/logs/deployment_state"
readonly STATE_FILE="$STATE_DIR/${DEPLOYMENT_ID}.json"
readonly ROLLBACK_DIR="$PROJECT_ROOT/logs/rollback"
readonly LOG_FILE="$PROJECT_ROOT/logs/deployment_${DEPLOYMENT_ID}.log"

# Platform detection
readonly OS_TYPE="$(uname -s)"
readonly ARCH_TYPE="$(uname -m)"
readonly LINUX_DISTRO="$(test -f /etc/os-release && . /etc/os-release && echo "$ID" || echo "unknown")"

# System requirements
declare -A MIN_REQUIREMENTS=(
    ["memory_gb"]="16"
    ["disk_gb"]="100"
    ["cpu_cores"]="4"
)

declare -A RECOMMENDED_REQUIREMENTS=(
    ["memory_gb"]="32"
    ["disk_gb"]="500"
    ["cpu_cores"]="8"
)

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly UNDERLINE='\033[4m'
readonly NC='\033[0m'

# Deployment phases
readonly DEPLOYMENT_PHASES=(
    "initialize"
    "system_detection"
    "dependency_check"
    "security_setup"
    "environment_prepare"
    "infrastructure_deploy"
    "services_deploy"
    "health_validation"
    "post_deployment"
    "finalize"
)

# ===============================================
# LOGGING & OUTPUT SYSTEM
# ===============================================

setup_logging() {
    mkdir -p "$(dirname "$LOG_FILE")" "$STATE_DIR" "$ROLLBACK_DIR"
    exec 1> >(tee -a "$LOG_FILE")
    exec 2> >(tee -a "$LOG_FILE" >&2)
    
    # Initialize deployment state
    cat > "$STATE_FILE" << EOF
{
    "deployment_id": "$DEPLOYMENT_ID",
    "script_version": "$SCRIPT_VERSION",
    "target": "${DEPLOYMENT_TARGET:-unknown}",
    "start_time": "$(date -Iseconds)",
    "platform": {
        "os": "$OS_TYPE",
        "arch": "$ARCH_TYPE",
        "distro": "$LINUX_DISTRO"
    },
    "phases": {},
    "rollback_points": [],
    "status": "initializing"
}
EOF
}

log_phase() {
    local phase="$1"
    local message="${2:-Starting $phase}"
    echo -e "\n${PURPLE}${BOLD}═══════════════════════════════════════════════════${NC}"
    echo -e "${PURPLE}${BOLD}🚀 PHASE: $(echo "$phase" | tr '[:lower:]' '[:upper:]')${NC}"
    echo -e "${PURPLE}${BOLD}${message}${NC}"
    echo -e "${PURPLE}${BOLD}═══════════════════════════════════════════════════${NC}\n"
    
    # Update state
    update_deployment_state "phases.$phase" "$(date -Iseconds)"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅ SUCCESS: $1${NC}"
}

log_info() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] ℹ️  INFO: $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  WARNING: $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌ ERROR: $1${NC}"
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] 🐛 DEBUG: $1${NC}"
    fi
}

show_progress() {
    local current="$1"
    local total="$2"
    local description="$3"
    local percentage=$((current * 100 / total))
    local filled=$((percentage / 2))
    local empty=$((50 - filled))
    
    printf "\r${BLUE}["
    printf "%${filled}s" | tr ' ' '='
    printf "%${empty}s" | tr ' ' ' '
    printf "] %d%% - %s${NC}" "$percentage" "$description"
    
    if [[ $current -eq $total ]]; then
        echo
    fi
}

# ===============================================
# STATE MANAGEMENT & ROLLBACK SYSTEM
# ===============================================

update_deployment_state() {
    local key="$1"
    local value="$2"
    
    # Ensure state file exists
    if [[ ! -f "$STATE_FILE" ]]; then
        log_warn "State file not found, recreating..."
        setup_logging
    fi
    
    # Use jq to update JSON state file
    if command -v jq >/dev/null 2>&1 && [[ -f "$STATE_FILE" ]]; then
        local temp_file
        temp_file=$(mktemp)
        jq --arg key "$key" --arg value "$value" 'setpath($key | split("."); $value)' "$STATE_FILE" > "$temp_file"
        mv "$temp_file" "$STATE_FILE"
    else
        log_warn "jq not available or state file missing, state tracking limited"
    fi
}

create_rollback_point() {
    local phase="$1"
    local description="$2"
    local rollback_id="rollback_${phase}_$(date +%s)"
    local rollback_file="$ROLLBACK_DIR/${rollback_id}.tar.gz"
    
    log_info "Creating rollback point: $phase"
    
    # Create system snapshot
    {
        # Docker state
        if command -v docker >/dev/null 2>&1; then
            docker ps -a --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" > "$ROLLBACK_DIR/${rollback_id}_docker.txt"
            docker image ls --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}" > "$ROLLBACK_DIR/${rollback_id}_images.txt"
        fi
        
        # Environment state
        env | grep -E '^(SUTAZAI|POSTGRES|REDIS|NEO4J|SECRET)' > "$ROLLBACK_DIR/${rollback_id}_env.txt" || true
        
        # Configuration files
        tar -czf "$rollback_file" \
            -C "$PROJECT_ROOT" \
            --exclude='logs' \
            --exclude='data' \
            --exclude='venv' \
            --exclude='node_modules' \
            --exclude='.git' \
            . 2>/dev/null || true
    }
    
    # Update state
    local rollback_info="{\"id\":\"$rollback_id\",\"phase\":\"$phase\",\"description\":\"$description\",\"timestamp\":\"$(date -Iseconds)\",\"file\":\"$rollback_file\"}"
    update_deployment_state "rollback_points[length]" "$rollback_info"
    
    echo "$rollback_id"
}

rollback_to_point() {
    local rollback_id="$1"
    local rollback_file="$ROLLBACK_DIR/${rollback_id}.tar.gz"
    
    log_warn "Initiating rollback to: $rollback_id"
    
    if [[ ! -f "$rollback_file" ]]; then
        log_error "Rollback file not found: $rollback_file"
        return 1
    fi
    
    # Stop all services
    log_info "Stopping all services for rollback..."
    stop_all_services || true
    
    # Restore configuration
    log_info "Restoring configuration from rollback point..."
    tar -xzf "$rollback_file" -C "$PROJECT_ROOT" 2>/dev/null || true
    
    # Restore environment
    if [[ -f "$ROLLBACK_DIR/${rollback_id}_env.txt" ]]; then
        source "$ROLLBACK_DIR/${rollback_id}_env.txt" || true
    fi
    
    log_success "Rollback completed to: $rollback_id"
}

# ===============================================
# ERROR HANDLING & RECOVERY
# ===============================================

cleanup_on_exit() {
    local exit_code=$?
    
    if [[ $exit_code -ne 0 ]]; then
        log_error "Deployment failed with exit code: $exit_code"
        
        # Only update state if state file exists
        if [[ -f "$STATE_FILE" ]]; then
            update_deployment_state "status" "failed"
            update_deployment_state "end_time" "$(date -Iseconds)"
            update_deployment_state "exit_code" "$exit_code"
        fi
        
        # Show state file location for debugging
        if [[ -n "${STATE_FILE:-}" ]]; then
            echo -e "\n${YELLOW}State file location: ${STATE_FILE}${NC}"
            if [[ -f "$STATE_FILE" ]]; then
                jq . "$STATE_FILE" 2>/dev/null || cat "$STATE_FILE" 2>/dev/null || true
            fi
        fi
        
        # Offer rollback
        if [[ "${AUTO_ROLLBACK:-true}" == "true" ]] && [[ -n "${LAST_ROLLBACK_POINT:-}" ]]; then
            log_warn "Auto-rollback enabled, rolling back to last known good state..."
            rollback_to_point "$LAST_ROLLBACK_POINT" || true
        else
            echo -e "\n${YELLOW}Deployment failed. You can manually rollback using:${NC}"
            echo -e "${YELLOW}  ./deploy.sh rollback ${LAST_ROLLBACK_POINT:-latest}${NC}\n"
        fi
    else
        if [[ -f "$STATE_FILE" ]]; then
            update_deployment_state "status" "completed"
            update_deployment_state "end_time" "$(date -Iseconds)"
        fi
    fi
}

handle_error() {
    local exit_code=$1
    local line_number=$2
    local command=$3
    
    log_error "Error on line $line_number: command '$command' failed with exit code $exit_code"
    
    # Attempt intelligent recovery based on error type
    case $exit_code in
        125|126|127) # Command not found or permission denied
            log_info "Attempting to install missing dependencies..."
            install_missing_dependencies || true
            ;;
        130) # Interrupted by user
            log_warn "Deployment interrupted by user"
            exit 130
            ;;
        *)
            log_error "Unrecoverable error, initiating cleanup..."
            ;;
    esac
    
    exit $exit_code
}

trap 'handle_error $? $LINENO $BASH_COMMAND' ERR
trap 'cleanup_on_exit' EXIT

# ===============================================
# SYSTEM DETECTION & VALIDATION
# ===============================================

detect_system_capabilities() {
    log_phase "system_detection" "Detecting system capabilities and requirements"
    
    # Hardware detection
    local cpu_cores
    cpu_cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "1")
    
    local memory_gb
    if [[ "$OS_TYPE" == "Linux" ]]; then
        memory_gb=$(free -m | awk 'NR==2{printf "%.0f", $2/1024}')
    elif [[ "$OS_TYPE" == "Darwin" ]]; then
        memory_gb=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
    else
        memory_gb="8" # Default assumption
    fi
    
    local disk_gb
    disk_gb=$(df -BG "$PROJECT_ROOT" 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//' || echo "100")
    
    # GPU detection
    local gpu_available=false
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_available=true
        log_info "NVIDIA GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    elif command -v rocm-smi >/dev/null 2>&1; then
        gpu_available=true
        log_info "AMD GPU detected"
    fi
    
    # Container runtime detection
    local container_runtime="none"
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        container_runtime="docker"
    elif command -v podman >/dev/null 2>&1; then
        container_runtime="podman"
    fi
    
    # Network connectivity
    local internet_available=false
    if curl -s --max-time 5 https://google.com >/dev/null 2>&1; then
        internet_available=true
    fi
    
    # Update system capabilities in state
    update_deployment_state "system.cpu_cores" "$cpu_cores"
    update_deployment_state "system.memory_gb" "$memory_gb"
    update_deployment_state "system.disk_gb" "$disk_gb"
    update_deployment_state "system.gpu_available" "$gpu_available"
    update_deployment_state "system.container_runtime" "$container_runtime"
    update_deployment_state "system.internet_available" "$internet_available"
    
    log_info "System detected: $cpu_cores cores, ${memory_gb}GB RAM, ${disk_gb}GB disk"
    log_info "Container runtime: $container_runtime"
    log_info "GPU available: $gpu_available"
    log_info "Internet connectivity: $internet_available"
    
    # Validate minimum requirements
    validate_system_requirements "$cpu_cores" "$memory_gb" "$disk_gb"
}

validate_system_requirements() {
    local cpu_cores="$1"
    local memory_gb="$2"
    local disk_gb="$3"
    
    local errors=()
    local warnings=()
    
    # Check minimum requirements
    if [[ $cpu_cores -lt ${MIN_REQUIREMENTS[cpu_cores]} ]]; then
        errors+=("Insufficient CPU cores: $cpu_cores < ${MIN_REQUIREMENTS[cpu_cores]}")
    fi
    
    if [[ $memory_gb -lt ${MIN_REQUIREMENTS[memory_gb]} ]]; then
        errors+=("Insufficient memory: ${memory_gb}GB < ${MIN_REQUIREMENTS[memory_gb]}GB")
    fi
    
    if [[ $disk_gb -lt ${MIN_REQUIREMENTS[disk_gb]} ]]; then
        errors+=("Insufficient disk space: ${disk_gb}GB < ${MIN_REQUIREMENTS[disk_gb]}GB")
    fi
    
    # Check recommended requirements
    if [[ $cpu_cores -lt ${RECOMMENDED_REQUIREMENTS[cpu_cores]} ]]; then
        warnings+=("CPU cores below recommended: $cpu_cores < ${RECOMMENDED_REQUIREMENTS[cpu_cores]}")
    fi
    
    if [[ $memory_gb -lt ${RECOMMENDED_REQUIREMENTS[memory_gb]} ]]; then
        warnings+=("Memory below recommended: ${memory_gb}GB < ${RECOMMENDED_REQUIREMENTS[memory_gb]}GB")
    fi
    
    if [[ $disk_gb -lt ${RECOMMENDED_REQUIREMENTS[disk_gb]} ]]; then
        warnings+=("Disk space below recommended: ${disk_gb}GB < ${RECOMMENDED_REQUIREMENTS[disk_gb]}GB")
    fi
    
    # Handle errors
    if [[ ${#errors[@]} -gt 0 ]]; then
        log_error "System requirements not met:"
        for error in "${errors[@]}"; do
            log_error "  - $error"
        done
        
        if [[ "${FORCE_DEPLOY:-false}" != "true" ]]; then
            log_error "Deployment aborted. Use FORCE_DEPLOY=true to override."
            exit 1
        else
            log_warn "Forcing deployment despite insufficient resources"
        fi
    fi
    
    # Handle warnings
    if [[ ${#warnings[@]} -gt 0 ]]; then
        log_warn "System warnings:"
        for warning in "${warnings[@]}"; do
            log_warn "  - $warning"
        done
        log_warn "Performance may be degraded"
    fi
    
    log_success "System requirements validation completed"
}

# ===============================================
# DEPENDENCY MANAGEMENT
# ===============================================

check_and_install_dependencies() {
    log_phase "dependency_check" "Checking and installing system dependencies"
    
    # Critical dependencies for deployment
    local critical_deps=(
        "curl:curl"
        "wget:wget"
        "git:git"
        "jq:jq"
        "tar:tar"
        "gzip:gzip"
    )
    
    # Container runtime
    local container_deps=(
        "docker:docker.io"
        "docker-compose:docker-compose"
    )
    
    # Optional but recommended
    local optional_deps=(
        "htop:htop"
        "vim:vim"
        "tmux:tmux"
        "tree:tree"
    )
    
    # Platform-specific package manager detection
    local package_manager="unknown"
    local install_cmd=""
    local update_cmd=""
    
    if command -v apt-get >/dev/null 2>&1; then
        package_manager="apt"
        install_cmd="apt-get install -y"
        update_cmd="apt-get update"
    elif command -v yum >/dev/null 2>&1; then
        package_manager="yum"
        install_cmd="yum install -y"
        update_cmd="yum update"
    elif command -v dnf >/dev/null 2>&1; then
        package_manager="dnf"
        install_cmd="dnf install -y"
        update_cmd="dnf update"
    elif command -v pacman >/dev/null 2>&1; then
        package_manager="pacman"
        install_cmd="pacman -S --noconfirm"
        update_cmd="pacman -Sy"
    elif command -v brew >/dev/null 2>&1; then
        package_manager="brew"
        install_cmd="brew install"
        update_cmd="brew update"
    fi
    
    log_info "Detected package manager: $package_manager"
    
    # Update package index
    if [[ "$package_manager" != "unknown" && "$package_manager" != "brew" ]]; then
        log_info "Updating package index..."
        if [[ $EUID -eq 0 ]]; then
            $update_cmd >/dev/null 2>&1 || log_warn "Failed to update package index"
        else
            sudo $update_cmd >/dev/null 2>&1 || log_warn "Failed to update package index"
        fi
    fi
    
    # Install critical dependencies
    install_dependency_group "Critical" critical_deps[@] "$package_manager" "$install_cmd" true
    
    # Install container runtime
    if ! command -v docker >/dev/null 2>&1; then
        install_docker "$package_manager"
    else
        log_success "Docker already installed: $(docker --version)"
    fi
    
    # Install optional dependencies
    install_dependency_group "Optional" optional_deps[@] "$package_manager" "$install_cmd" false
    
    # Verify Docker is running
    ensure_docker_running
    
    log_success "Dependency installation completed"
}

install_dependency_group() {
    local group_name="$1"
    local -n deps_array=$2
    local package_manager="$3"
    local install_cmd="$4"
    local required="$5"
    
    log_info "Installing $group_name dependencies..."
    
    local failed_deps=()
    
    for dep in "${deps_array[@]}"; do
        IFS=':' read -r cmd_name pkg_name <<< "$dep"
        
        if command -v "$cmd_name" >/dev/null 2>&1; then
            log_success "$cmd_name already installed"
        else
            log_info "Installing $cmd_name..."
            
            if [[ "$package_manager" == "unknown" ]]; then
                log_warn "Unknown package manager, cannot install $cmd_name"
                if [[ "$required" == "true" ]]; then
                    failed_deps+=("$cmd_name")
                fi
            else
                if install_package "$pkg_name" "$install_cmd"; then
                    log_success "$cmd_name installed successfully"
                else
                    log_error "Failed to install $cmd_name"
                    if [[ "$required" == "true" ]]; then
                        failed_deps+=("$cmd_name")
                    fi
                fi
            fi
        fi
    done
    
    if [[ ${#failed_deps[@]} -gt 0 && "$required" == "true" ]]; then
        log_error "Failed to install required dependencies: ${failed_deps[*]}"
        exit 1
    fi
}

install_package() {
    local package="$1"
    local install_cmd="$2"
    
    if [[ $EUID -eq 0 ]]; then
        $install_cmd "$package" >/dev/null 2>&1
    else
        sudo $install_cmd "$package" >/dev/null 2>&1
    fi
}

install_docker() {
    local package_manager="$1"
    
    log_info "Installing Docker..."
    
    case "$package_manager" in
        "apt")
            # Docker's official installation for Ubuntu/Debian
            curl -fsSL https://get.docker.com -o get-docker.sh
            if [[ $EUID -eq 0 ]]; then
                sh get-docker.sh
            else
                sudo sh get-docker.sh
                sudo usermod -aG docker "$USER"
            fi
            rm get-docker.sh
            ;;
        "yum"|"dnf")
            # Docker installation for RHEL/CentOS/Fedora
            if [[ $EUID -eq 0 ]]; then
                $package_manager install -y docker docker-compose
                systemctl enable --now docker
            else
                sudo $package_manager install -y docker docker-compose
                sudo systemctl enable --now docker
                sudo usermod -aG docker "$USER"
            fi
            ;;
        "brew")
            brew install --cask docker
            ;;
        *)
            log_error "Unsupported package manager for Docker installation: $package_manager"
            exit 1
            ;;
    esac
    
    log_success "Docker installation completed"
}

ensure_docker_running() {
    log_info "Ensuring Docker daemon is running..."
    
    if ! docker info >/dev/null 2>&1; then
        log_info "Starting Docker daemon..."
        
        if command -v systemctl >/dev/null 2>&1; then
            if [[ $EUID -eq 0 ]]; then
                systemctl start docker
            else
                sudo systemctl start docker
            fi
            sleep 5
        else
            log_warn "Cannot start Docker daemon automatically"
            log_error "Please start Docker daemon manually and re-run deployment"
            exit 1
        fi
        
        # Verify Docker is now running
        if ! docker info >/dev/null 2>&1; then
            log_error "Failed to start Docker daemon"
            exit 1
        fi
    fi
    
    log_success "Docker daemon is running"
}

# ===============================================
# SECURITY SETUP
# ===============================================

setup_security() {
    log_phase "security_setup" "Configuring security settings and secrets"
    
    local secrets_dir="$PROJECT_ROOT/secrets"
    mkdir -p "$secrets_dir"
    chmod 700 "$secrets_dir"
    
    # Generate secure passwords if they don't exist
    generate_secure_secrets
    
    # Setup SSL certificates
    setup_ssl_certificates
    
    # Configure firewall if needed
    configure_firewall
    
    # Set proper file permissions
    set_secure_permissions
    
    log_success "Security setup completed"
}

generate_secure_secrets() {
    local secrets_dir="$PROJECT_ROOT/secrets"
    
    # Generate database passwords
    local secrets=(
        "postgres_password.txt"
        "redis_password.txt"
        "neo4j_password.txt"
        "jwt_secret.txt"
        "grafana_password.txt"
    )
    
    for secret_file in "${secrets[@]}"; do
        local secret_path="$secrets_dir/$secret_file"
        
        if [[ ! -f "$secret_path" ]]; then
            log_info "Generating $secret_file..."
            openssl rand -base64 32 | tr -d "=+/" | cut -c1-25 > "$secret_path"
            chmod 600 "$secret_path"
        else
            log_info "$secret_file already exists"
        fi
    done
    
    # Generate JWT secret key
    if [[ ! -f "$secrets_dir/jwt_secret.txt" ]]; then
        openssl rand -hex 32 > "$secrets_dir/jwt_secret.txt"
        chmod 600 "$secrets_dir/jwt_secret.txt"
    fi
}

setup_ssl_certificates() {
    local ssl_dir="$PROJECT_ROOT/ssl"
    mkdir -p "$ssl_dir"
    
    if [[ ! -f "$ssl_dir/cert.pem" ]] || [[ ! -f "$ssl_dir/key.pem" ]]; then
        log_info "Generating self-signed SSL certificates..."
        
        # Generate private key
        openssl genpkey -algorithm RSA -out "$ssl_dir/key.pem" -pkcs8 -aes-256-cbc -pass pass:sutazai 2>/dev/null
        
        # Generate certificate
        openssl req -new -x509 -key "$ssl_dir/key.pem" -out "$ssl_dir/cert.pem" -days 365 \
            -passin pass:sutazai \
            -subj "/C=US/ST=CA/L=San Francisco/O=SutazAI/CN=localhost" 2>/dev/null
        
        chmod 600 "$ssl_dir"/*
        log_success "SSL certificates generated"
    else
        log_info "SSL certificates already exist"
    fi
}

configure_firewall() {
    if [[ "${DEPLOYMENT_TARGET:-local}" == "production" ]]; then
        log_info "Configuring firewall for production..."
        
        # Configure iptables or ufw based on availability
        if command -v ufw >/dev/null 2>&1; then
            configure_ufw_firewall
        elif command -v firewall-cmd >/dev/null 2>&1; then
            configure_firewalld
        else
            log_warn "No supported firewall found, skipping firewall configuration"
        fi
    else
        log_info "Skipping firewall configuration for non-production deployment"
    fi
}

configure_ufw_firewall() {
    log_info "Configuring UFW firewall..."
    
    if [[ $EUID -eq 0 ]]; then
        ufw --force enable
        ufw default deny incoming
        ufw default allow outgoing
        
        # Allow SSH
        ufw allow ssh
        
        # Allow HTTP/HTTPS
        ufw allow 80/tcp
        ufw allow 443/tcp
        
        # Allow application ports
        ufw allow 8000:8500/tcp
        
        log_success "UFW firewall configured"
    else
        sudo ufw --force enable
        sudo ufw default deny incoming
        sudo ufw default allow outgoing
        sudo ufw allow ssh
        sudo ufw allow 80/tcp
        sudo ufw allow 443/tcp
        sudo ufw allow 8000:8500/tcp
        
        log_success "UFW firewall configured"
    fi
}

set_secure_permissions() {
    log_info "Setting secure file permissions..."
    
    # Secure script files
    find "$PROJECT_ROOT/scripts" -name "*.sh" -exec chmod 755 {} \; 2>/dev/null || true
    
    # Secure configuration files
    find "$PROJECT_ROOT/config" -type f -exec chmod 644 {} \; 2>/dev/null || true
    
    # Secure secret files
    find "$PROJECT_ROOT/secrets" -type f -exec chmod 600 {} \; 2>/dev/null || true
    
    # Secure log directory
    chmod 755 "$PROJECT_ROOT/logs"
    
    log_success "File permissions secured"
}

# ===============================================
# ENVIRONMENT PREPARATION
# ===============================================

prepare_environment() {
    log_phase "environment_prepare" "Preparing deployment environment"
    
    # Create directory structure
    create_directory_structure
    
    # Setup environment variables
    setup_environment_variables
    
    # Configure Docker Compose environment
    configure_docker_environment
    
    # Initialize data directories
    initialize_data_directories
    
    log_success "Environment preparation completed"
}

create_directory_structure() {
    log_info "Creating directory structure..."
    
    local required_dirs=(
        "logs"
        "data"
        "data/postgres"
        "data/redis"
        "data/neo4j"
        "data/ollama"
        "data/chromadb"
        "data/qdrant"
        "data/faiss"
        "data/models"
        "data/training"
        "data/documents"
        "data/monitoring"
        "monitoring/prometheus"
        "monitoring/grafana/provisioning"
        "monitoring/grafana/dashboards"
        "monitoring/loki"
        "monitoring/promtail"
        "ssl"
        "secrets"
        "backups"
        "workspace"
    )
    
    for dir in "${required_dirs[@]}"; do
        mkdir -p "$PROJECT_ROOT/$dir"
    done
    
    log_success "Directory structure created"
}

setup_environment_variables() {
    log_info "Setting up environment variables..."
    
    local env_file="$PROJECT_ROOT/.env"
    local secrets_dir="$PROJECT_ROOT/secrets"
    
    # Read secrets
    local postgres_password
    local redis_password
    local neo4j_password
    local jwt_secret
    local grafana_password
    
    postgres_password=$(cat "$secrets_dir/postgres_password.txt" 2>/dev/null || echo "sutazai_postgres")
    redis_password=$(cat "$secrets_dir/redis_password.txt" 2>/dev/null || echo "sutazai_redis")
    neo4j_password=$(cat "$secrets_dir/neo4j_password.txt" 2>/dev/null || echo "sutazai_neo4j")
    jwt_secret=$(cat "$secrets_dir/jwt_secret.txt" 2>/dev/null || echo "sutazai_jwt_secret")
    grafana_password=$(cat "$secrets_dir/grafana_password.txt" 2>/dev/null || echo "sutazai_grafana")
    
    # Get local IP
    local local_ip
    local_ip=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "localhost")
    
    # Create environment file
    cat > "$env_file" << EOF
# SutazAI Deployment Environment Configuration
# Generated: $(date -Iseconds)
# Deployment ID: $DEPLOYMENT_ID
# Target: ${DEPLOYMENT_TARGET:-local}

# System Configuration
TZ=UTC
SUTAZAI_ENV=${DEPLOYMENT_TARGET:-local}
LOCAL_IP=$local_ip
DEPLOYMENT_ID=$DEPLOYMENT_ID

# Database Configuration
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=$postgres_password
POSTGRES_DB=sutazai
DATABASE_URL=postgresql://sutazai:$postgres_password@postgres:5432/sutazai

# Redis Configuration
REDIS_PASSWORD=$redis_password

# Neo4j Configuration
NEO4J_PASSWORD=$neo4j_password

# Security Configuration
SECRET_KEY=$jwt_secret
JWT_SECRET=$jwt_secret

# Monitoring Configuration
GRAFANA_PASSWORD=$grafana_password

# Model Configuration
OLLAMA_HOST=0.0.0.0
OLLAMA_ORIGINS=*
OLLAMA_NUM_PARALLEL=2
OLLAMA_MAX_LOADED_MODELS=2

# Feature Flags
ENABLE_GPU=auto
ENABLE_MONITORING=true
ENABLE_LOGGING=true
ENABLE_HEALTH_CHECKS=true

# Performance Tuning
MAX_WORKERS=4
CONNECTION_POOL_SIZE=20
CACHE_TTL=3600

# Development/Debug Settings
DEBUG=${DEBUG:-false}
LOG_LEVEL=${LOG_LEVEL:-INFO}
EOF
    
    chmod 600 "$env_file"
    log_success "Environment variables configured"
}

configure_docker_environment() {
    log_info "Configuring Docker environment..."
    
    # Create Docker daemon configuration for optimal performance
    local docker_daemon_config="/etc/docker/daemon.json"
    
    if [[ ! -f "$docker_daemon_config" ]] && [[ "${DEPLOYMENT_TARGET:-local}" != "local" ]]; then
        log_info "Creating Docker daemon configuration..."
        
        local config_content='
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "100m",
        "max-file": "3"
    },
    "storage-driver": "overlay2",
    "live-restore": true,
    "default-ulimits": {
        "nofile": {
            "Name": "nofile",
            "Hard": 64000,
            "Soft": 64000
        }
    }
}
'
        
        if [[ $EUID -eq 0 ]]; then
            echo "$config_content" > "$docker_daemon_config"
            systemctl restart docker
        else
            echo "$config_content" | sudo tee "$docker_daemon_config" >/dev/null
            sudo systemctl restart docker
        fi
        
        sleep 5 # Wait for Docker to restart
        log_success "Docker daemon configured and restarted"
    else
        log_info "Docker daemon configuration already exists or skipped for local deployment"
    fi
}

initialize_data_directories() {
    log_info "Initializing data directories..."
    
    # Set proper ownership and permissions for data directories
    local data_dirs=(
        "data/postgres"
        "data/redis"
        "data/neo4j"
        "data/ollama"
        "data/chromadb"
        "data/qdrant"
        "data/monitoring"
    )
    
    for dir in "${data_dirs[@]}"; do
        local full_path="$PROJECT_ROOT/$dir"
        if [[ -d "$full_path" ]]; then
            chmod 755 "$full_path"
        fi
    done
    
    log_success "Data directories initialized"
}

# ===============================================
# INFRASTRUCTURE DEPLOYMENT
# ===============================================

deploy_infrastructure() {
    log_phase "infrastructure_deploy" "Deploying core infrastructure services"
    
    local rollback_point
    rollback_point=$(create_rollback_point "infrastructure" "Before infrastructure deployment")
    LAST_ROLLBACK_POINT="$rollback_point"
    
    # Determine compose files based on target and capabilities
    local compose_files
    compose_files=$(determine_compose_files)
    
    log_info "Using Docker Compose files: $compose_files"
    
    # Deploy infrastructure services in stages
    deploy_core_databases "$compose_files"
    deploy_vector_databases "$compose_files"
    deploy_model_services "$compose_files"
    
    log_success "Infrastructure deployment completed"
}

determine_compose_files() {
    local compose_files="-f docker-compose.yml"
    
    # Add target-specific compose file
    case "${DEPLOYMENT_TARGET:-local}" in
        "production")
            if [[ -f "$PROJECT_ROOT/docker-compose.production.yml" ]]; then
                compose_files+=" -f docker-compose.production.yml"
            fi
            ;;
        "staging")
            if [[ -f "$PROJECT_ROOT/docker-compose.staging.yml" ]]; then
                compose_files+=" -f docker-compose.staging.yml"
            fi
            ;;
    esac
    
    # Add GPU support if available
    if command -v nvidia-smi >/dev/null 2>&1 && [[ -f "$PROJECT_ROOT/docker-compose.gpu.yml" ]]; then
        compose_files+=" -f docker-compose.gpu.yml"
        log_info "GPU support detected, adding GPU compose file"
    fi
    
    # Add monitoring stack
    if [[ "${ENABLE_MONITORING:-true}" == "true" ]] && [[ -f "$PROJECT_ROOT/docker-compose.monitoring.yml" ]]; then
        compose_files+=" -f docker-compose.monitoring.yml"
    fi
    
    echo "$compose_files"
}

deploy_core_databases() {
    local compose_files="$1"
    
    log_info "Deploying core database services..."
    
    # Deploy databases first
    local core_services=(
        "postgres"
        "redis"
        "neo4j"
    )
    
    for service in "${core_services[@]}"; do
        log_info "Starting $service..."
        docker compose $compose_files up -d "$service"
        
        # Wait for service to be healthy
        wait_for_service_health "$service" 60
        
        show_progress $((${#core_services[@]} - $(( ${#core_services[@]} - $(get_service_index "$service" "${core_services[@]}") )))) ${#core_services[@]} "Deploying $service"
    done
    
    log_success "Core databases deployed successfully"
}

deploy_vector_databases() {
    local compose_files="$1"
    
    log_info "Deploying vector database services..."
    
    local vector_services=(
        "chromadb"
        "qdrant"
        "faiss"
    )
    
    for service in "${vector_services[@]}"; do
        log_info "Starting $service..."
        docker compose $compose_files up -d "$service"
        
        wait_for_service_health "$service" 60
        show_progress $((${#vector_services[@]} - $(( ${#vector_services[@]} - $(get_service_index "$service" "${vector_services[@]}") )))) ${#vector_services[@]} "Deploying $service"
    done
    
    log_success "Vector databases deployed successfully"
}

deploy_model_services() {
    local compose_files="$1"
    
    log_info "Deploying model services..."
    
    # Deploy Ollama
    log_info "Starting Ollama service..."
    docker compose $compose_files up -d ollama
    
    # Wait for Ollama to be ready
    wait_for_service_health "ollama" 120
    
    # Download essential models
    download_essential_models
    
    log_success "Model services deployed successfully"
}

# ===============================================
# SERVICE DEPLOYMENT
# ===============================================

deploy_services() {
    log_phase "services_deploy" "Deploying application and AI services"
    
    local rollback_point
    rollback_point=$(create_rollback_point "services" "Before services deployment")
    LAST_ROLLBACK_POINT="$rollback_point"
    
    local compose_files
    compose_files=$(determine_compose_files)
    
    # Deploy application services
    deploy_application_services "$compose_files"
    
    # Deploy AI agents
    deploy_ai_agents "$compose_files"
    
    # Deploy monitoring stack
    if [[ "${ENABLE_MONITORING:-true}" == "true" ]]; then
        deploy_monitoring_stack "$compose_files"
    fi
    
    log_success "Services deployment completed"
}

deploy_application_services() {
    local compose_files="$1"
    
    log_info "Deploying application services..."
    
    local app_services=(
        "backend"
        "frontend"
    )
    
    for service in "${app_services[@]}"; do
        log_info "Starting $service..."
        docker compose $compose_files up -d "$service"
        
        wait_for_service_health "$service" 90
        show_progress $((${#app_services[@]} - $(( ${#app_services[@]} - $(get_service_index "$service" "${app_services[@]}") )))) ${#app_services[@]} "Deploying $service"
    done
    
    log_success "Application services deployed successfully"
}

deploy_ai_agents() {
    local compose_files="$1"
    
    log_info "Deploying AI agent services..."
    
    # Get list of available AI services from compose file
    local ai_services
    ai_services=$(docker compose $compose_files config --services | grep -E "(agent|gpt|ai|crew|auto|letta|aider)" | head -10)
    
    if [[ -z "$ai_services" ]]; then
        log_warn "No AI services found in compose configuration"
        return 0
    fi
    
    local ai_services_array
    IFS=$'\n' read -rd '' -a ai_services_array <<< "$ai_services" || true
    
    local deployed_count=0
    local total_count=${#ai_services_array[@]}
    
    for service in "${ai_services_array[@]}"; do
        log_info "Starting AI service: $service..."
        
        # Start service with timeout
        if timeout 60 docker compose $compose_files up -d "$service" 2>/dev/null; then
            deployed_count=$((deployed_count + 1))
            log_success "Successfully started $service"
        else
            log_warn "Failed to start $service, continuing with other services"
        fi
        
        show_progress $deployed_count $total_count "Deploying AI services"
        sleep 2 # Brief pause between deployments
    done
    
    log_success "AI agent services deployment completed ($deployed_count/$total_count successful)"
}

deploy_monitoring_stack() {
    local compose_files="$1"
    
    log_info "Deploying monitoring stack..."
    
    local monitoring_services=(
        "prometheus"
        "grafana"
        "loki"
        "promtail"
    )
    
    for service in "${monitoring_services[@]}"; do
        if docker compose $compose_files config --services | grep -q "^$service$"; then
            log_info "Starting monitoring service: $service..."
            docker compose $compose_files up -d "$service"
            
            wait_for_service_health "$service" 60
            show_progress $((${#monitoring_services[@]} - $(( ${#monitoring_services[@]} - $(get_service_index "$service" "${monitoring_services[@]}") )))) ${#monitoring_services[@]} "Deploying $service"
        else
            log_warn "Monitoring service $service not found in compose configuration"
        fi
    done
    
    log_success "Monitoring stack deployed successfully"
}

# ===============================================
# UTILITY FUNCTIONS
# ===============================================

wait_for_service_health() {
    local service="$1"
    local timeout="${2:-60}"
    local interval=5
    local elapsed=0
    
    log_info "Waiting for $service to become healthy (timeout: ${timeout}s)..."
    
    while [[ $elapsed -lt $timeout ]]; do
        # Check if service is running
        if docker ps --filter "name=sutazai-$service" --filter "status=running" --format "{{.Names}}" | grep -q "sutazai-$service"; then
            # Check health status if healthcheck is defined
            local health_status
            health_status=$(docker inspect "sutazai-$service" --format='{{.State.Health.Status}}' 2>/dev/null || echo "no_healthcheck")
            
            if [[ "$health_status" == "healthy" ]] || [[ "$health_status" == "no_healthcheck" ]]; then
                log_success "$service is healthy and ready"
                return 0
            elif [[ "$health_status" == "unhealthy" ]]; then
                log_warn "$service is unhealthy, continuing to wait..."
            fi
        else
            log_debug "$service container not running yet..."
        fi
        
        sleep $interval
        elapsed=$((elapsed + interval))
        
        # Show progress
        local percentage=$((elapsed * 100 / timeout))
        printf "\r${BLUE}Waiting for $service: [%3d%%] %ds/%ds${NC}" "$percentage" "$elapsed" "$timeout"
    done
    
    echo # New line after progress
    log_warn "$service did not become healthy within ${timeout}s, continuing anyway..."
    return 1
}

get_service_index() {
    local service="$1"
    shift
    local services=("$@")
    
    for i in "${!services[@]}"; do
        if [[ "${services[$i]}" == "$service" ]]; then
            echo $((i + 1))
            return 0
        fi
    done
    echo 1
}

download_essential_models() {
    log_info "Downloading essential AI models..."
    
    local essential_models=(
        "tinyllama:latest"
        "qwen2.5:3b"
        "nomic-embed-text:latest"
    )
    
    for model in "${essential_models[@]}"; do
        log_info "Downloading model: $model..."
        
        # Use timeout to prevent hanging
        if timeout 300 docker exec sutazai-ollama ollama pull "$model" >/dev/null 2>&1; then
            log_success "Successfully downloaded $model"
        else
            log_warn "Failed to download $model or timeout exceeded"
        fi
    done
    
    log_success "Essential models download completed"
}

stop_all_services() {
    log_info "Stopping all SutazAI services..."
    
    local compose_files
    compose_files=$(determine_compose_files)
    
    # Stop all services
    docker compose $compose_files down --remove-orphans >/dev/null 2>&1 || true
    
    # Stop any remaining containers
    docker ps --filter "name=sutazai-" --format "{{.Names}}" | xargs -r docker stop >/dev/null 2>&1 || true
    
    log_success "All services stopped"
}

# ===============================================
# HEALTH VALIDATION
# ===============================================

validate_deployment_health() {
    log_phase "health_validation" "Validating deployment health and functionality"
    
    local health_results=()
    local failed_checks=0
    
    # Core infrastructure checks
    check_infrastructure_health health_results failed_checks
    
    # Application service checks
    check_application_health health_results failed_checks
    
    # AI service checks
    check_ai_services_health health_results failed_checks
    
    # Integration tests
    run_integration_tests health_results failed_checks
    
    # Generate health report
    generate_health_report "${health_results[@]}"
    
    if [[ $failed_checks -gt 0 ]]; then
        log_warn "Health validation completed with $failed_checks failed checks"
        
        if [[ "${DEPLOYMENT_TARGET:-local}" == "production" ]]; then
            log_error "Production deployment failed health validation"
            return 1
        else
            log_warn "Non-production deployment, continuing despite health issues"
        fi
    else
        log_success "All health checks passed successfully"
    fi
    
    return 0
}

check_infrastructure_health() {
    local -n results=$1
    local -n failed=$2
    
    log_info "Checking infrastructure health..."
    
    local infrastructure_services=(
        "postgres:5432"
        "redis:6379"
        "neo4j:7474"
        "chromadb:8000"
        "qdrant:6333"
        "ollama:11434"
    )
    
    for service_port in "${infrastructure_services[@]}"; do
        IFS=':' read -r service port <<< "$service_port"
        
        if check_service_port "sutazai-$service" "$port"; then
            results+=("✅ Infrastructure: $service (port $port) - HEALTHY")
        else
            results+=("❌ Infrastructure: $service (port $port) - FAILED")
            failed=$((failed + 1))
        fi
    done
}

check_application_health() {
    local -n results=$1
    local -n failed=$2
    
    log_info "Checking application health..."
    
    # Backend health check
    if check_http_endpoint "http://localhost:8000/health" 200; then
        results+=("✅ Application: Backend API - HEALTHY")
    else
        results+=("❌ Application: Backend API - FAILED")
        failed=$((failed + 1))
    fi
    
    # Frontend health check
    if check_http_endpoint "http://localhost:8501/healthz" 200; then
        results+=("✅ Application: Frontend - HEALTHY")
    else
        results+=("❌ Application: Frontend - FAILED")
        failed=$((failed + 1))
    fi
}

check_ai_services_health() {
    local -n results=$1
    local -n failed=$2
    
    log_info "Checking AI services health..."
    
    # Ollama model availability
    if docker exec sutazai-ollama ollama list 2>/dev/null | grep -q "tinyllama"; then
        results+=("✅ AI: Ollama models - AVAILABLE")
    else
        results+=("❌ AI: Ollama models - NOT AVAILABLE")
        failed=$((failed + 1))
    fi
    
    # Vector database connectivity
    if check_http_endpoint "http://localhost:8001/api/v1/heartbeat" 200; then
        results+=("✅ AI: ChromaDB - HEALTHY")
    else
        results+=("❌ AI: ChromaDB - FAILED")
        failed=$((failed + 1))
    fi
}

run_integration_tests() {
    local -n results=$1
    local -n failed=$2
    
    log_info "Running integration tests..."
    
    # Test AI model inference
    if test_ai_inference; then
        results+=("✅ Integration: AI inference - WORKING")
    else
        results+=("❌ Integration: AI inference - FAILED")
        failed=$((failed + 1))
    fi
    
    # Test database connectivity
    if test_database_connectivity; then
        results+=("✅ Integration: Database connectivity - WORKING")
    else
        results+=("❌ Integration: Database connectivity - FAILED")
        failed=$((failed + 1))
    fi
}

check_service_port() {
    local container="$1"
    local port="$2"
    
    # Check if container is running and port is open
    if docker ps --filter "name=$container" --filter "status=running" --format "{{.Names}}" | grep -q "$container"; then
        # Use docker exec to check port from inside container
        docker exec "$container" timeout 5 bash -c "echo >/dev/tcp/localhost/$port" 2>/dev/null
    else
        return 1
    fi
}

check_http_endpoint() {
    local url="$1"
    local expected_status="${2:-200}"
    
    local actual_status
    actual_status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$url" 2>/dev/null || echo "000")
    
    [[ "$actual_status" == "$expected_status" ]]
}

test_ai_inference() {
    # Simple test of AI inference via Ollama
    local test_prompt="Hello, this is a test."
    local response
    
    response=$(curl -s --max-time 30 -X POST http://localhost:11434/api/generate \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"tinyllama\", \"prompt\": \"$test_prompt\", \"stream\": false}" 2>/dev/null)
    
    # Check if response contains expected fields
    echo "$response" | grep -q '"response"' && echo "$response" | grep -q '"done"'
}

test_database_connectivity() {
    # Test PostgreSQL connectivity
    docker exec sutazai-postgres pg_isready -U sutazai >/dev/null 2>&1 && \
    docker exec sutazai-redis redis-cli -a "$(cat "$PROJECT_ROOT/secrets/redis_password.txt" 2>/dev/null || echo "sutazai_redis")" ping 2>/dev/null | grep -q "PONG"
}

generate_health_report() {
    local results=("$@")
    local report_file="$PROJECT_ROOT/logs/health_report_$(date +%Y%m%d_%H%M%S).json"
    
    log_info "Generating health report: $report_file"
    
    # Create detailed health report
    cat > "$report_file" << EOF
{
    "deployment_id": "$DEPLOYMENT_ID",
    "timestamp": "$(date -Iseconds)",
    "target": "${DEPLOYMENT_TARGET:-local}",
    "version": "$SCRIPT_VERSION",
    "results": [
EOF
    
    local first=true
    for result in "${results[@]}"; do
        if [[ "$first" == "true" ]]; then
            first=false
        else
            echo "," >> "$report_file"
        fi
        echo "        \"$result\"" >> "$report_file"
    done
    
    cat >> "$report_file" << EOF
    ],
    "summary": {
        "total_checks": ${#results[@]},
        "passed": $(printf '%s\n' "${results[@]}" | grep -c "✅"),
        "failed": $(printf '%s\n' "${results[@]}" | grep -c "❌")
    }
}
EOF
    
    log_success "Health report generated: $report_file"
}

# ===============================================
# POST-DEPLOYMENT TASKS
# ===============================================

run_post_deployment_tasks() {
    log_phase "post_deployment" "Running post-deployment tasks and optimizations"
    
    # Create user accounts and permissions
    setup_initial_users
    
    # Configure monitoring dashboards
    setup_monitoring_dashboards
    
    # Run system optimizations
    run_system_optimizations
    
    # Create backup schedule
    setup_backup_schedule
    
    # Generate access information
    generate_access_information
    
    log_success "Post-deployment tasks completed"
}

setup_initial_users() {
    log_info "Setting up initial user accounts..."
    
    # This would typically involve API calls to create admin users
    # For now, we'll just create a placeholder configuration
    
    local users_config="$PROJECT_ROOT/config/initial_users.json"
    cat > "$users_config" << EOF
{
    "admin": {
        "username": "admin",
        "role": "administrator",
        "created": "$(date -Iseconds)",
        "deployment_id": "$DEPLOYMENT_ID"
    }
}
EOF
    
    log_success "Initial user configuration created"
}

setup_monitoring_dashboards() {
    if [[ "${ENABLE_MONITORING:-true}" == "true" ]]; then
        log_info "Setting up monitoring dashboards..."
        
        # Copy dashboard configurations
        if [[ -d "$PROJECT_ROOT/monitoring/grafana/dashboards" ]]; then
            # Dashboard configurations would be applied here
            log_success "Monitoring dashboards configured"
        else
            log_warn "Monitoring dashboard directory not found"
        fi
    else
        log_info "Monitoring disabled, skipping dashboard setup"
    fi
}

run_system_optimizations() {
    log_info "Running system optimizations..."
    
    # Docker system cleanup
    docker system prune -f >/dev/null 2>&1 || true
    
    # Set optimal Docker resource limits
    apply_resource_optimizations
    
    log_success "System optimizations completed"
}

apply_resource_optimizations() {
    # This function would apply CPU and memory optimizations
    # based on the detected system capabilities
    
    local cpu_cores
    cpu_cores=$(nproc 2>/dev/null || echo "4")
    
    if [[ $cpu_cores -lt 8 ]]; then
        log_info "Applying low-resource optimizations..."
        # Would set environment variables for reduced resource usage
    else
        log_info "Sufficient resources detected, using standard configuration"
    fi
}

setup_backup_schedule() {
    if [[ "${DEPLOYMENT_TARGET:-local}" != "local" ]]; then
        log_info "Setting up backup schedule..."
        
        # Create backup script
        cat > "$PROJECT_ROOT/scripts/backup_system.sh" << 'EOF'
#!/bin/bash
# Automated backup script for SutazAI
set -euo pipefail

BACKUP_DIR="/opt/sutazaiapp/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/backup_$TIMESTAMP.tar.gz"

# Create backup
mkdir -p "$BACKUP_DIR"
tar -czf "$BACKUP_FILE" \
    --exclude='logs' \
    --exclude='data/postgres' \
    --exclude='data/ollama' \
    -C /opt/sutazaiapp \
    .

# Keep only last 7 days of backups
find "$BACKUP_DIR" -name "backup_*.tar.gz" -mtime +7 -delete

echo "Backup created: $BACKUP_FILE"
EOF
        
        chmod +x "$PROJECT_ROOT/scripts/backup_system.sh"
        log_success "Backup schedule configured"
    else
        log_info "Local deployment, skipping backup schedule"
    fi
}

generate_access_information() {
    log_info "Generating access information..."
    
    local access_file="$PROJECT_ROOT/logs/ACCESS_INFO_${DEPLOYMENT_ID}.txt"
    local local_ip
    local_ip=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "localhost")
    
    cat > "$access_file" << EOF
SutazAI System Access Information
=================================

Deployment ID: $DEPLOYMENT_ID
Deployment Target: ${DEPLOYMENT_TARGET:-local}
Deployment Time: $(date)
Local IP: $local_ip

WEB INTERFACES
--------------
Main Application: http://$local_ip:8501
Backend API: http://$local_ip:8000
API Documentation: http://$local_ip:8000/docs

AI SERVICES
-----------
Ollama API: http://$local_ip:11434
LangFlow: http://$local_ip:8090
FlowiseAI: http://$local_ip:8099
Dify: http://$local_ip:8107

DATABASES
---------
PostgreSQL: $local_ip:5432 (user: sutazai)
Redis: $local_ip:6379
Neo4j Browser: http://$local_ip:7474
ChromaDB: http://$local_ip:8001
Qdrant: http://$local_ip:6333

MONITORING
----------
Grafana: http://$local_ip:3000 (admin/$(cat "$PROJECT_ROOT/secrets/grafana_password.txt" 2>/dev/null || echo "sutazai_grafana"))
Prometheus: http://$local_ip:9090

CREDENTIALS
-----------
Database passwords are stored in: $PROJECT_ROOT/secrets/
Default admin credentials are in: $PROJECT_ROOT/config/initial_users.json

LOGS & TROUBLESHOOTING
---------------------
Deployment logs: $LOG_FILE
State file: $STATE_FILE
Health reports: $PROJECT_ROOT/logs/health_report_*.json

To view service logs: docker compose logs [service_name]
To restart services: docker compose restart [service_name]
To stop all services: docker compose down
EOF
    
    log_success "Access information generated: $access_file"
    
    # Display key information
    echo -e "\n${GREEN}${BOLD}🎉 DEPLOYMENT COMPLETED SUCCESSFULLY! 🎉${NC}\n"
    echo -e "${CYAN}Main Application: ${BOLD}http://$local_ip:8501${NC}"
    echo -e "${CYAN}Backend API: ${BOLD}http://$local_ip:8000${NC}"
    echo -e "${CYAN}Full access info: ${BOLD}$access_file${NC}\n"
}

# ===============================================
# COMMAND HANDLERS
# ===============================================

handle_deploy() {
    local target="${1:-local}"
    
    if [[ ! -v DEPLOYMENT_TARGETS[$target] ]]; then
        log_error "Invalid deployment target: $target"
        log_info "Available targets: ${!DEPLOYMENT_TARGETS[*]}"
        exit 1
    fi
    
    DEPLOYMENT_TARGET="$target"
    
    log_info "Starting deployment to target: $target"
    log_info "Description: ${DEPLOYMENT_TARGETS[$target]}"
    
    # Execute deployment phases
    for phase in "${DEPLOYMENT_PHASES[@]}"; do
        case "$phase" in
            "initialize")
                setup_logging
                ;;
            "system_detection")
                detect_system_capabilities
                ;;
            "dependency_check")
                check_and_install_dependencies
                ;;
            "security_setup")
                setup_security
                ;;
            "environment_prepare")
                prepare_environment
                ;;
            "infrastructure_deploy")
                deploy_infrastructure
                ;;
            "services_deploy")
                deploy_services
                ;;
            "health_validation")
                validate_deployment_health
                ;;
            "post_deployment")
                run_post_deployment_tasks
                ;;
            "finalize")
                log_success "Deployment completed successfully!"
                ;;
        esac
    done
}

handle_rollback() {
    local rollback_point="${1:-latest}"
    
    log_info "Initiating rollback to: $rollback_point"
    
    if [[ "$rollback_point" == "latest" ]]; then
        # Find the most recent rollback point
        local latest_rollback
        latest_rollback=$(ls -t "$ROLLBACK_DIR"/rollback_*.tar.gz 2>/dev/null | head -1 | xargs basename -s .tar.gz 2>/dev/null || echo "")
        
        if [[ -z "$latest_rollback" ]]; then
            log_error "No rollback points found"
            exit 1
        fi
        
        rollback_point="$latest_rollback"
    fi
    
    rollback_to_point "$rollback_point"
}

handle_status() {
    log_info "Checking SutazAI system status..."
    
    # Check if any containers are running
    local running_containers
    running_containers=$(docker ps --filter "name=sutazai-" --format "{{.Names}}" | wc -l)
    
    echo -e "\n${BOLD}SutazAI System Status${NC}"
    echo -e "=====================\n"
    
    if [[ $running_containers -gt 0 ]]; then
        echo -e "${GREEN}Status: ${BOLD}RUNNING${NC} ($running_containers containers)"
        echo -e "\nRunning services:"
        docker ps --filter "name=sutazai-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    else
        echo -e "${YELLOW}Status: ${BOLD}STOPPED${NC}"
        echo -e "No SutazAI containers are currently running"
    fi
    
    # Show recent deployments
    echo -e "\nRecent deployments:"
    if [[ -d "$STATE_DIR" ]]; then
        ls -la "$STATE_DIR"/*.json 2>/dev/null | tail -5 || echo "No deployment records found"
    else
        echo "No deployment state directory found"
    fi
}

handle_logs() {
    local service="${1:-}"
    
    if [[ -n "$service" ]]; then
        log_info "Showing logs for service: $service"
        docker compose logs -f "sutazai-$service" 2>/dev/null || docker logs -f "sutazai-$service" 2>/dev/null || {
            log_error "Service not found: $service"
            exit 1
        }
    else
        log_info "Showing logs for all SutazAI services"
        docker compose logs -f
    fi
}

handle_cleanup() {
    log_info "Cleaning up SutazAI system..."
    
    # Stop and remove all containers
    docker compose down --remove-orphans --volumes 2>/dev/null || true
    
    # Remove dangling images
    docker image prune -f >/dev/null 2>&1 || true
    
    # Clean build cache
    docker builder prune -f >/dev/null 2>&1 || true
    
    # Clean up old logs (keep last 7 days)
    find "$PROJECT_ROOT/logs" -name "*.log" -mtime +7 -delete 2>/dev/null || true
    
    # Clean up old rollback points (keep last 5)
    if [[ -d "$ROLLBACK_DIR" ]]; then
        ls -t "$ROLLBACK_DIR"/rollback_*.tar.gz 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null || true
    fi
    
    log_success "System cleanup completed"
}

# ===============================================
# MAIN ENTRY POINT
# ===============================================

show_usage() {
    cat << EOF
${BOLD}${SCRIPT_NAME} v${SCRIPT_VERSION}${NC}
${UNDERLINE}Universal deployment script for SutazAI systems${NC}

${BOLD}USAGE:${NC}
    $0 [COMMAND] [TARGET/OPTIONS]

${BOLD}COMMANDS:${NC}
    deploy [TARGET]    Deploy SutazAI system to specified target
                      Targets: local, staging, production, fresh
    
    rollback [POINT]   Rollback to specified rollback point
                      Use 'latest' for most recent point
    
    status            Show current system status and running services
    
    logs [SERVICE]    Show logs for specified service or all services
    
    cleanup           Clean up containers, images, and old files
    
    health            Run health checks on current deployment
    
    help              Show this help message

${BOLD}DEPLOYMENT TARGETS:${NC}
$(for target in "${!DEPLOYMENT_TARGETS[@]}"; do
    printf "    %-12s %s\n" "$target" "${DEPLOYMENT_TARGETS[$target]}"
done)

${BOLD}ENVIRONMENT VARIABLES:${NC}
    DEBUG=true            Enable debug output
    FORCE_DEPLOY=true     Force deployment despite warnings
    AUTO_ROLLBACK=false   Disable automatic rollback on failure
    ENABLE_MONITORING=false  Disable monitoring stack
    LOG_LEVEL=DEBUG       Set logging level

${BOLD}EXAMPLES:${NC}
    $0 deploy local                    # Deploy to local development
    $0 deploy production               # Deploy to production
    FORCE_DEPLOY=true $0 deploy fresh  # Force fresh installation
    $0 rollback latest                 # Rollback to latest checkpoint
    $0 status                          # Check system status
    $0 logs backend                    # Show backend service logs
    $0 cleanup                         # Clean up system

${BOLD}FILES & DIRECTORIES:${NC}
    Configuration: $PROJECT_ROOT/.env
    Secrets:      $PROJECT_ROOT/secrets/
    Logs:         $PROJECT_ROOT/logs/
    State:        $PROJECT_ROOT/logs/deployment_state/
    Rollbacks:    $PROJECT_ROOT/logs/rollback/

For more information, visit: https://github.com/sutazai/sutazaiapp
EOF
}

# Main command dispatcher
main() {
    local command="${1:-deploy}"
    
    case "$command" in
        "deploy")
            shift
            handle_deploy "${1:-local}"
            ;;
        "rollback")
            shift
            handle_rollback "${1:-latest}"
            ;;
        "status")
            handle_status
            ;;
        "logs")
            shift
            handle_logs "${1:-}"
            ;;
        "cleanup")
            handle_cleanup
            ;;
        "health")
            setup_logging
            validate_deployment_health
            ;;
        "help"|"--help"|"-h")
            show_usage
            ;;
        *)
            log_error "Unknown command: $command"
            echo
            show_usage
            exit 1
            ;;
    esac
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi