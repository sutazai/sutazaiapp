#!/bin/bash
#
# SutazAI Startup Optimization Integration Script
# Integrates optimized startup with existing deployment infrastructure
#
# DESCRIPTION:
#   This script provides seamless integration between the new startup optimization
#   system and existing deployment scripts, allowing users to choose between
#   traditional and optimized startup methods.
#

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Color codes
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

log_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

install_optimization_hooks() {
    log_info "Installing startup optimization hooks..."
    
    # Create backup of original deploy.sh if it exists
    if [[ -f "$PROJECT_ROOT/deploy.sh" ]]; then
        if [[ ! -f "$PROJECT_ROOT/deploy.sh.backup" ]]; then
            cp "$PROJECT_ROOT/deploy.sh" "$PROJECT_ROOT/deploy.sh.backup"
            log_success "Created backup: deploy.sh.backup"
        fi
    fi
    
    # Add optimization integration to deploy.sh
    if [[ -f "$PROJECT_ROOT/deploy.sh" ]]; then
        # Check if optimization hooks are already present
        if ! grep -q "ENABLE_FAST_STARTUP" "$PROJECT_ROOT/deploy.sh"; then
            log_info "Adding fast startup integration to deploy.sh..."
            
            # Create temporary file with integration
            cat > "/tmp/startup_integration.patch" << 'EOF'

# ===============================================
# STARTUP OPTIMIZATION INTEGRATION
# ===============================================

# Add fast startup option to environment variables section
readonly ENABLE_FAST_STARTUP="${ENABLE_FAST_STARTUP:-true}"
readonly FAST_STARTUP_MODE="${FAST_STARTUP_MODE:-full}"
readonly FAST_STARTUP_PARALLEL="${FAST_STARTUP_PARALLEL:-auto}"

# Function to use optimized startup
use_fast_startup() {
    local mode="${1:-$FAST_STARTUP_MODE}"
    local parallel_jobs="${2:-$FAST_STARTUP_PARALLEL}"
    
    log_info "Using optimized fast startup (mode: $mode)"
    
    local fast_start_script="$PROJECT_ROOT/scripts/fast_start.sh"
    
    if [[ ! -f "$fast_start_script" ]]; then
        log_error "Fast startup script not found: $fast_start_script"
        log_warn "Falling back to traditional startup..."
        return 1
    fi
    
    local fast_start_args=("$mode")
    
    if [[ "$parallel_jobs" != "auto" ]]; then
        fast_start_args+=(--parallel "$parallel_jobs")
    fi
    
    if [[ "${ENABLE_MONITORING:-false}" == "true" ]]; then
        fast_start_args+=(--monitor)
    fi
    
    if [[ "${DEBUG:-false}" == "true" ]]; then
        fast_start_args+=(--debug)
    fi
    
    # Execute fast startup
    if "$fast_start_script" "${fast_start_args[@]}"; then
        log_success "Fast startup completed successfully"
        return 0
    else
        log_error "Fast startup failed, falling back to traditional method..."
        return 1
    fi
}

# Modify the services deployment section
deploy_services_optimized() {
    log_phase "services_deploy" "Deploying application and AI services (optimized)"
    
    local rollback_point
    rollback_point=$(create_rollback_point "services" "Before optimized services deployment")
    LAST_ROLLBACK_POINT="$rollback_point"
    
    # Try fast startup first if enabled
    if [[ "${ENABLE_FAST_STARTUP:-true}" == "true" ]]; then
        if use_fast_startup; then
            log_success "Services deployed using optimized startup"
            return 0
        else
            log_warn "Fast startup failed, falling back to traditional deployment..."
        fi
    fi
    
    # Fallback to original deployment method
    log_info "Using traditional service deployment..."
    deploy_services_traditional
}

# Preserve original deploy_services function
deploy_services_traditional() {
    # This will contain the original deploy_services logic
    log_phase "services_deploy" "Deploying application and AI services (traditional)"
    
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

EOF
            
            # Note: In a real implementation, we would need to carefully patch the deploy.sh
            # For now, we'll create integration instructions
            log_warn "Manual integration required for deploy.sh"
            log_info "Please see STARTUP_OPTIMIZATION_GUIDE.md for integration instructions"
        else
            log_success "Fast startup integration already present in deploy.sh"
        fi
    fi
}

create_startup_aliases() {
    log_info "Creating convenient startup aliases..."
    
    local alias_file="$PROJECT_ROOT/scripts/startup_aliases.sh"
    
    cat > "$alias_file" << 'EOF'
#!/bin/bash
#
# SutazAI Startup Command Aliases
# Source this file to get convenient startup commands
#

SUTAZAI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Fast startup aliases
alias sutazai-start='${SUTAZAI_ROOT}/scripts/fast_start.sh full'
alias sutazai-start-core='${SUTAZAI_ROOT}/scripts/fast_start.sh core'
alias sutazai-start-critical='${SUTAZAI_ROOT}/scripts/fast_start.sh critical-only'
alias sutazai-start-agents='${SUTAZAI_ROOT}/scripts/fast_start.sh agents-only'

# Monitoring aliases
alias sutazai-start-monitor='${SUTAZAI_ROOT}/scripts/fast_start.sh full --monitor'
alias sutazai-status='docker ps --filter "name=sutazai-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"'
alias sutazai-logs='docker compose -f ${SUTAZAI_ROOT}/docker-compose.yml logs -f'

# Optimization and validation
alias sutazai-optimize='python3 ${SUTAZAI_ROOT}/scripts/startup_optimizer.py'
alias sutazai-validate='python3 ${SUTAZAI_ROOT}/scripts/startup_validator.py'
alias sutazai-benchmark='${SUTAZAI_ROOT}/scripts/startup_validator.py && echo "Benchmark completed, check logs/"'

# System management
alias sutazai-stop='docker compose -f ${SUTAZAI_ROOT}/docker-compose.yml down'
alias sutazai-restart='sutazai-stop && sleep 5 && sutazai-start'
alias sutazai-health='curl -s http://localhost:8000/health && curl -s http://localhost:11434/api/tags'

# Development helpers
alias sutazai-dev='${SUTAZAI_ROOT}/scripts/fast_start.sh core --monitor --parallel 4'
alias sutazai-test='${SUTAZAI_ROOT}/scripts/fast_start.sh full --dry-run'

echo "SutazAI startup aliases loaded!"
echo "Available commands:"
echo "  sutazai-start         - Full optimized startup"
echo "  sutazai-start-core    - Core services only"
echo "  sutazai-start-monitor - Full startup with monitoring"
echo "  sutazai-status        - Show running services"
echo "  sutazai-validate      - Run performance validation"
echo "  sutazai-stop          - Stop all services"
echo ""
echo "To use these aliases, run: source scripts/startup_aliases.sh"
EOF
    
    chmod +x "$alias_file"
    log_success "Startup aliases created: $alias_file"
}

create_systemd_service() {
    log_info "Creating systemd service for automatic startup..."
    
    local service_file="/tmp/sutazai-startup.service"
    
    cat > "$service_file" << EOF
[Unit]
Description=SutazAI Optimized Startup Service
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$PROJECT_ROOT
ExecStart=$PROJECT_ROOT/scripts/fast_start.sh full
ExecStop=$PROJECT_ROOT/scripts/fast_start.sh stop
User=root
Group=root
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    log_info "Systemd service file created: $service_file"
    log_info "To install system-wide, run as root:"
    log_info "  sudo cp $service_file /etc/systemd/system/"
    log_info "  sudo systemctl daemon-reload"
    log_info "  sudo systemctl enable sutazai-startup.service"
}

create_docker_healthcheck() {
    log_info "Creating Docker healthcheck integration..."
    
    local healthcheck_script="$PROJECT_ROOT/scripts/docker_healthcheck.sh"
    
    cat > "$healthcheck_script" << 'EOF'
#!/bin/bash
#
# Docker Healthcheck Script for SutazAI
# Used by Docker Compose health checks
#

set -euo pipefail

SERVICE_NAME="${1:-backend}"
CHECK_TYPE="${2:-http}"

case "$CHECK_TYPE" in
    "http")
        case "$SERVICE_NAME" in
            "backend")
                curl -f http://localhost:8000/health >/dev/null 2>&1
                ;;
            "frontend")
                curl -f http://localhost:8501 >/dev/null 2>&1
                ;;
            "ollama")
                curl -f http://localhost:11434/api/tags >/dev/null 2>&1
                ;;
            *)
                exit 0  # Default to healthy for unknown services
                ;;
        esac
        ;;
    "tcp")
        nc -z localhost "${3:-8000}"
        ;;
    "postgres")
        pg_isready -h localhost -p 5432 -U sutazai
        ;;
    "redis")
        redis-cli -h localhost -p 6379 ping | grep -q PONG
        ;;
    *)
        exit 0  # Default to healthy
        ;;
esac
EOF
    
    chmod +x "$healthcheck_script"
    log_success "Docker healthcheck script created: $healthcheck_script"
}

validate_installation() {
    log_info "Validating startup optimization installation..."
    
    local errors=0
    
    # Check required files
    local required_files=(
        "scripts/fast_start.sh"
        "scripts/startup_optimizer.py"
        "scripts/startup_validator.py"
        "STARTUP_OPTIMIZATION_GUIDE.md"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$PROJECT_ROOT/$file" ]]; then
            log_warn "Missing required file: $file"
            errors=$((errors + 1))
        fi
    done
    
    # Check dependencies
    local required_commands=(
        "docker"
        "docker-compose"
        "python3"
        "curl"
    )
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            log_warn "Missing required command: $cmd"
            errors=$((errors + 1))
        fi
    done
    
    # Check Python dependencies
    if command -v python3 >/dev/null 2>&1; then
        local python_deps=("docker" "psutil" "requests" "pyyaml")
        for dep in "${python_deps[@]}"; do
            if ! python3 -c "import $dep" >/dev/null 2>&1; then
                log_warn "Missing Python dependency: $dep"
                log_info "Install with: pip3 install $dep"
                errors=$((errors + 1))
            fi
        done
    fi
    
    if [[ $errors -eq 0 ]]; then
        log_success "✅ Installation validation passed!"
        log_info "You can now use optimized startup with:"
        log_info "  ./scripts/fast_start.sh"
        return 0
    else
        log_warn "⚠️  Installation validation found $errors issues"
        log_info "Please resolve the above issues before using optimized startup"
        return 1
    fi
}

install_python_dependencies() {
    log_info "Installing Python dependencies for startup optimization..."
    
    local requirements_file="/tmp/startup_optimization_requirements.txt"
    
    cat > "$requirements_file" << EOF
# Python dependencies for SutazAI startup optimization
docker>=6.0.0
psutil>=5.8.0
requests>=2.25.0
PyYAML>=6.0
asyncio-backport>=0.1.0;python_version<"3.7"
EOF
    
    if command -v pip3 >/dev/null 2>&1; then
        pip3 install -r "$requirements_file"
        log_success "Python dependencies installed"
    else
        log_warn "pip3 not found, please install dependencies manually:"
        cat "$requirements_file"
    fi
    
    rm -f "$requirements_file"
}

show_usage() {
    cat << EOF
${BOLD}SutazAI Startup Optimization Integration${NC}

This script integrates the startup optimization system with existing deployment infrastructure.

${BOLD}USAGE:${NC}
    $0 [COMMAND]

${BOLD}COMMANDS:${NC}
    install          Install optimization hooks and integration
    aliases          Create convenient startup command aliases
    systemd          Create systemd service for automatic startup
    healthcheck      Create Docker healthcheck integration
    validate         Validate installation and dependencies
    deps             Install Python dependencies
    all              Run all installation steps
    help             Show this help message

${BOLD}EXAMPLES:${NC}
    $0 all                    # Complete installation and setup
    $0 install                # Install optimization hooks only
    $0 aliases                # Create command aliases only
    $0 validate               # Check installation status

${BOLD}FILES CREATED:${NC}
    scripts/startup_aliases.sh      # Convenient command aliases
    scripts/docker_healthcheck.sh   # Docker health check integration
    /tmp/sutazai-startup.service    # Systemd service template

EOF
}

main() {
    local command="${1:-help}"
    
    case "$command" in
        "install")
            install_optimization_hooks
            ;;
        "aliases")
            create_startup_aliases
            ;;
        "systemd")
            create_systemd_service
            ;;
        "healthcheck")
            create_docker_healthcheck
            ;;
        "validate")
            validate_installation
            ;;
        "deps")
            install_python_dependencies
            ;;
        "all")
            log_info "Running complete startup optimization installation..."
            install_python_dependencies
            install_optimization_hooks
            create_startup_aliases
            create_docker_healthcheck
            create_systemd_service
            validate_installation
            log_success "🎉 Startup optimization installation completed!"
            log_info "Quick start: source scripts/startup_aliases.sh && sutazai-start"
            ;;
        "help"|"--help"|"-h")
            show_usage
            ;;
        *)
            echo "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi