#!/bin/bash
#
# Ultimate Deployment System Setup Script
# Version: 1.0.0
#
# DESCRIPTION:
#   Setup script for the Ultimate Deployment System that prepares all
#   components, installs dependencies, and verifies the system is ready
#   for bulletproof deployments.
#
# USAGE:
#   ./setup-ultimate-deployment.sh [--install-deps] [--verify-only]
#
# REQUIREMENTS:
#   - Bash 4.0+
#   - Python 3.8+
#   - Docker and Docker Compose v2
#   - Internet connectivity for package downloads

set -euo pipefail

# Configuration
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly SCRIPTS_DIR="$PROJECT_ROOT/scripts"
readonly LOG_DIR="$PROJECT_ROOT/logs"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Setup logging
mkdir -p "$LOG_DIR"
readonly SETUP_LOG="$LOG_DIR/ultimate-deployment-setup.log"

log_info() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1" >> "$SETUP_LOG"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $1${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $1" >> "$SETUP_LOG"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1" >> "$SETUP_LOG"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1" >> "$SETUP_LOG"
}

show_banner() {
    echo -e "${PURPLE}${BOLD}"
    echo "================================================================================================"
    echo "🚀 ULTIMATE DEPLOYMENT SYSTEM SETUP"
    echo "   SutazAI 131-Agent Ecosystem | Zero Downtime | 1000% Reliability"
    echo "   Version: 1.0.0"
    echo "================================================================================================"
    echo -e "${NC}"
}

check_prerequisites() {
    log_info "Checking system prerequisites..."
    
    local errors=0
    
    # Check Python version
    if command -v python3 >/dev/null 2>&1; then
        local python_version
        python_version=$(python3 --version | cut -d' ' -f2)
        local major_version
        major_version=$(echo "$python_version" | cut -d'.' -f1)
        local minor_version
        minor_version=$(echo "$python_version" | cut -d'.' -f2)
        
        if [[ $major_version -eq 3 && $minor_version -ge 8 ]]; then
            log_success "Python $python_version detected"
        else
            log_error "Python 3.8+ required, found $python_version"
            errors=$((errors + 1))
        fi
    else
        log_error "Python 3 not found"
        errors=$((errors + 1))
    fi
    
    # Check Docker
    if command -v docker >/dev/null 2>&1; then
        if docker info >/dev/null 2>&1; then
            local docker_version
            docker_version=$(docker --version | cut -d' ' -f3 | tr -d ',')
            log_success "Docker $docker_version detected and running"
        else
            log_error "Docker found but not running"
            errors=$((errors + 1))
        fi
    else
        log_error "Docker not found"
        errors=$((errors + 1))
    fi
    
    # Check Docker Compose
    if docker compose version >/dev/null 2>&1; then
        local compose_version
        compose_version=$(docker compose version --short)
        log_success "Docker Compose $compose_version detected"
    else
        log_error "Docker Compose v2 not found"
        errors=$((errors + 1))
    fi
    
    # Check system resources
    if command -v free >/dev/null 2>&1; then
        local memory_gb
        memory_gb=$(free -m | awk 'NR==2{printf "%.0f", $2/1024}')
        if [[ $memory_gb -ge 16 ]]; then
            log_success "Memory: ${memory_gb}GB (sufficient)"
        else
            log_warn "Memory: ${memory_gb}GB (recommended: 16GB+)"
        fi
    fi
    
    if [[ $errors -gt 0 ]]; then
        log_error "System prerequisites check failed with $errors errors"
        return 1
    fi
    
    log_success "All prerequisites satisfied"
    return 0
}

install_python_dependencies() {
    log_info "Installing Python dependencies..."
    
    local required_packages=(
        "aiohttp>=3.8.0"
        "websockets>=10.0"
        "cryptography>=3.4.0"
        "PyYAML>=6.0"
        "psutil>=5.8.0"
        "jinja2>=3.0.0"
    )
    
    # Check if pip is available
    if ! command -v pip3 >/dev/null 2>&1; then
        log_error "pip3 not found"
        return 1
    fi
    
    # Install packages
    for package in "${required_packages[@]}"; do
        log_info "Installing $package..."
        if pip3 install "$package" --user --quiet; then
            log_success "Installed $package"
        else
            log_error "Failed to install $package"
            return 1
        fi
    done
    
    log_success "Python dependencies installed"
}

setup_directory_structure() {
    log_info "Setting up directory structure..."
    
    local required_dirs=(
        "logs"
        "logs/deployment_state"
        "logs/rollback"
        "config"
        "config/environments"
        "config/templates"
        "secrets"
        "backups"
        "data"
        "data/monitoring"
    )
    
    for dir in "${required_dirs[@]}"; do
        local full_path="$PROJECT_ROOT/$dir"
        if [[ ! -d "$full_path" ]]; then
            mkdir -p "$full_path"
            log_info "Created directory: $dir"
        fi
    done
    
    # Set secure permissions
    chmod 700 "$PROJECT_ROOT/secrets" 2>/dev/null || true
    chmod 755 "$PROJECT_ROOT/logs"
    chmod 755 "$PROJECT_ROOT/config"
    
    log_success "Directory structure setup completed"
}

make_scripts_executable() {
    log_info "Making deployment scripts executable..."
    
    local deployment_scripts=(
        "ultimate-deployment-master.py"
        "ultimate-deployment-orchestrator.py"
        "comprehensive-agent-health-monitor.py"
        "advanced-rollback-system.py"
        "multi-environment-config-manager.py"
        "setup-ultimate-deployment.sh"
    )
    
    for script in "${deployment_scripts[@]}"; do
        local script_path="$SCRIPTS_DIR/$script"
        if [[ -f "$script_path" ]]; then
            chmod +x "$script_path"
            log_success "Made executable: $script"
        else
            log_warn "Script not found: $script"
        fi
    done
    
    # Also make the main deploy script executable
    if [[ -f "$PROJECT_ROOT/deploy.sh" ]]; then
        chmod +x "$PROJECT_ROOT/deploy.sh"
        log_success "Made executable: deploy.sh"
    fi
}

verify_deployment_system() {
    log_info "Verifying Ultimate Deployment System..."
    
    # Test Python imports
    log_info "Testing Python dependencies..."
    if python3 -c "
import asyncio
import aiohttp
import websockets
import cryptography
import yaml
import psutil
import jinja2
print('All Python dependencies available')
" 2>/dev/null; then
        log_success "Python dependencies verified"
    else
        log_error "Python dependencies verification failed"
        return 1
    fi
    
    # Test script syntax
    local main_scripts=(
        "ultimate-deployment-master.py"
        "comprehensive-agent-health-monitor.py"
        "advanced-rollback-system.py"
        "multi-environment-config-manager.py"
    )
    
    for script in "${main_scripts[@]}"; do
        local script_path="$SCRIPTS_DIR/$script"
        if [[ -f "$script_path" ]]; then
            if python3 -m py_compile "$script_path" 2>/dev/null; then
                log_success "Syntax verified: $script"
            else
                log_error "Syntax error in: $script"
                return 1
            fi
        fi
    done
    
    # Test deployment master help
    if cd "$PROJECT_ROOT" && python3 scripts/ultimate-deployment-master.py --help >/dev/null 2>&1; then
        log_success "Ultimate Deployment Master responds correctly"
    else
        log_error "Ultimate Deployment Master not working properly"
        return 1
    fi
    
    log_success "Ultimate Deployment System verification completed"
}

create_quick_start_script() {
    log_info "Creating quick start script..."
    
    local quickstart_script="$PROJECT_ROOT/start-ultimate-deployment.sh"
    
    cat > "$quickstart_script" << 'EOF'
#!/bin/bash
#
# Quick Start Script for Ultimate Deployment System
#
# Usage: ./start-ultimate-deployment.sh [environment]
#

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVIRONMENT="${1:-local}"

echo "🚀 Starting Ultimate Deployment System"
echo "Environment: $ENVIRONMENT"
echo "Dashboard: http://localhost:7777"
echo "WebSocket: ws://localhost:7778"
echo "API: http://localhost:7779"
echo ""

cd "$PROJECT_ROOT"
python3 scripts/ultimate-deployment-master.py deploy --environment "$ENVIRONMENT"
EOF
    
    chmod +x "$quickstart_script"
    log_success "Quick start script created: start-ultimate-deployment.sh"
}

generate_system_info() {
    log_info "Generating system information report..."
    
    local info_file="$LOG_DIR/ultimate-deployment-system-info.txt"
    
    cat > "$info_file" << EOF
Ultimate Deployment System - System Information
==============================================

Generated: $(date)
Project Root: $PROJECT_ROOT

System Information:
------------------
OS: $(uname -s) $(uname -r)
Architecture: $(uname -m)
Python: $(python3 --version 2>&1 || echo "Not available")
Docker: $(docker --version 2>&1 || echo "Not available")
Docker Compose: $(docker compose version 2>&1 || echo "Not available")

Memory: $(free -h 2>/dev/null | grep Mem || echo "Not available")
Disk Space: $(df -h . 2>/dev/null | tail -1 || echo "Not available")

Deployment Scripts:
------------------
EOF
    
    # List all deployment scripts
    find "$SCRIPTS_DIR" -name "*.py" -o -name "*.sh" | sort | while read -r script; do
        echo "$(basename "$script") - $(stat -c %s "$script" 2>/dev/null || echo "0") bytes" >> "$info_file"
    done
    
    cat >> "$info_file" << EOF

Quick Start Commands:
--------------------
# Local deployment
./start-ultimate-deployment.sh local

# Production deployment  
./start-ultimate-deployment.sh production

# Dashboard only
python3 scripts/ultimate-deployment-master.py dashboard

# System status
python3 scripts/ultimate-deployment-master.py status

# Emergency rollback
python3 scripts/ultimate-deployment-master.py emergency

Documentation:
-------------
See ULTIMATE_DEPLOYMENT_SYSTEM_DOCUMENTATION.md for complete documentation.

Setup Log:
---------
$(tail -20 "$SETUP_LOG" 2>/dev/null || echo "Setup log not available")
EOF
    
    log_success "System information saved to: $info_file"
}

show_completion_message() {
    echo -e "\n${GREEN}${BOLD}================================================================================================"
    echo "🎉 ULTIMATE DEPLOYMENT SYSTEM SETUP COMPLETED SUCCESSFULLY!"
    echo "================================================================================================${NC}"
    echo ""
    echo -e "${CYAN}${BOLD}Quick Start:${NC}"
    echo -e "${CYAN}  ./start-ultimate-deployment.sh local${NC}     # Local development deployment"
    echo -e "${CYAN}  ./start-ultimate-deployment.sh production${NC} # Production deployment"
    echo ""
    echo -e "${CYAN}${BOLD}Dashboard:${NC}"
    echo -e "${CYAN}  http://localhost:7777${NC}                    # Real-time monitoring dashboard"
    echo ""
    echo -e "${CYAN}${BOLD}Manual Commands:${NC}"
    echo -e "${CYAN}  python3 scripts/ultimate-deployment-master.py deploy --environment local${NC}"
    echo -e "${CYAN}  python3 scripts/ultimate-deployment-master.py dashboard${NC}"
    echo -e "${CYAN}  python3 scripts/ultimate-deployment-master.py status${NC}"
    echo ""
    echo -e "${CYAN}${BOLD}Documentation:${NC}"
    echo -e "${CYAN}  ULTIMATE_DEPLOYMENT_SYSTEM_DOCUMENTATION.md${NC}  # Complete documentation"
    echo -e "${CYAN}  logs/ultimate-deployment-system-info.txt${NC}     # System information"
    echo ""
    echo -e "${GREEN}${BOLD}Ready to deploy 131 AI agents with 1000% reliability! 🚀${NC}"
    echo ""
}

main() {
    local install_deps=false
    local verify_only=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --install-deps)
                install_deps=true
                shift
                ;;
            --verify-only)
                verify_only=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [--install-deps] [--verify-only]"
                echo ""
                echo "Options:"
                echo "  --install-deps  Install Python dependencies"
                echo "  --verify-only   Only verify system, don't make changes"
                echo "  --help         Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    show_banner
    
    # Check prerequisites
    if ! check_prerequisites; then
        log_error "Prerequisites check failed"
        exit 1
    fi
    
    if [[ "$verify_only" == "true" ]]; then
        log_info "Verification mode - skipping setup steps"
        verify_deployment_system
        exit 0
    fi
    
    # Install Python dependencies if requested
    if [[ "$install_deps" == "true" ]]; then
        install_python_dependencies
    fi
    
    # Setup system
    setup_directory_structure
    make_scripts_executable
    create_quick_start_script
    
    # Verify everything works
    verify_deployment_system
    
    # Generate system info
    generate_system_info
    
    # Show completion message
    show_completion_message
    
    log_success "Ultimate Deployment System setup completed successfully!"
}

# Execute main function
main "$@"