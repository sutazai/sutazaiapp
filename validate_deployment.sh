#!/bin/bash
#
# SutazAI Deployment Validation Script
# This script validates that the deployment system is properly configured
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log_info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

log_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

log_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

validate_files() {
    log_info "Validating deployment files..."
    
    local required_files=(
        "deploy.sh"
        "scripts/deploy-production.sh"
        "docker-compose.yml"
        "docker-compose.cpu-only.yml"
        "docker-compose.gpu.yml"
        "docker-compose.monitoring.yml"
        "DEPLOYMENT_README.md"
    )
    
    local missing_files=()
    
    for file in "${required_files[@]}"; do
        if [[ -f "$PROJECT_ROOT/$file" ]]; then
            log_success "Found: $file"
        else
            log_error "Missing: $file"
            missing_files+=("$file")
        fi
    done
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        log_error "Missing required files: ${missing_files[*]}"
        return 1
    fi
    
    log_success "All required files present"
}

validate_permissions() {
    log_info "Validating file permissions..."
    
    local executable_files=(
        "deploy.sh"
        "scripts/deploy-production.sh"
    )
    
    for file in "${executable_files[@]}"; do
        if [[ -x "$PROJECT_ROOT/$file" ]]; then
            log_success "Executable: $file"
        else
            log_warn "Not executable: $file (fixing...)"
            chmod +x "$PROJECT_ROOT/$file"
            log_success "Fixed permissions: $file"
        fi
    done
}

validate_docker_compose() {
    log_info "Validating Docker Compose files..."
    
    local compose_files=(
        "docker-compose.yml"
        "docker-compose.cpu-only.yml"
        "docker-compose.gpu.yml"
        "docker-compose.monitoring.yml"
    )
    
    for file in "${compose_files[@]}"; do
        if docker compose -f "$PROJECT_ROOT/$file" config >/dev/null 2>&1; then
            log_success "Valid: $file"
        else
            log_error "Invalid: $file"
            return 1
        fi
    done
}

validate_directories() {
    log_info "Validating directory structure..."
    
    local required_dirs=(
        "scripts"
        "monitoring"
        "monitoring/prometheus"
        "logs"
        "secrets"
        "ssl"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [[ -d "$PROJECT_ROOT/$dir" ]]; then
            log_success "Found directory: $dir"
        else
            log_warn "Creating directory: $dir"
            mkdir -p "$PROJECT_ROOT/$dir"
            log_success "Created: $dir"
        fi
    done
}

validate_dependencies() {
    log_info "Validating system dependencies..."
    
    local required_commands=(
        "docker"
        "curl"
        "wget"
        "git"
        "jq"
    )
    
    local missing_commands=()
    
    for cmd in "${required_commands[@]}"; do
        if command -v "$cmd" >/dev/null 2>&1; then
            log_success "Found: $cmd"
        else
            log_warn "Missing: $cmd"
            missing_commands+=("$cmd")
        fi
    done
    
    if [[ ${#missing_commands[@]} -gt 0 ]]; then
        log_warn "Missing dependencies: ${missing_commands[*]}"
        log_info "The deployment script will attempt to install these automatically"
    fi
}

test_deployment_script() {
    log_info "Testing deployment script help function..."
    
    if "$PROJECT_ROOT/deploy.sh" help >/dev/null 2>&1; then
        log_success "Deployment script help works"
    else
        log_error "Deployment script help failed"
        return 1
    fi
    
    log_info "Testing deployment script status function..."
    
    if "$PROJECT_ROOT/deploy.sh" status >/dev/null 2>&1; then
        log_success "Deployment script status works"
    else
        log_warn "Deployment script status returned non-zero (normal if no containers running)"
    fi
}

main() {
    echo -e "\n${BLUE}SutazAI Deployment System Validation${NC}"
    echo -e "${BLUE}====================================${NC}\n"
    
    local validation_steps=(
        "validate_files"
        "validate_permissions"
        "validate_directories"
        "validate_dependencies"
        "validate_docker_compose"
        "test_deployment_script"
    )
    
    local failed_steps=()
    
    for step in "${validation_steps[@]}"; do
        if $step; then
            echo
        else
            failed_steps+=("$step")
            echo
        fi
    done
    
    if [[ ${#failed_steps[@]} -eq 0 ]]; then
        echo -e "${GREEN}🎉 All validation checks passed!${NC}"
        echo -e "${GREEN}The deployment system is ready to use.${NC}\n"
        
        echo -e "${BLUE}Quick start commands:${NC}"
        echo -e "  ${YELLOW}./deploy.sh deploy local${NC}     # Deploy locally"
        echo -e "  ${YELLOW}./deploy.sh status${NC}           # Check status"
        echo -e "  ${YELLOW}./deploy.sh help${NC}             # Show help"
        echo
        
        return 0
    else
        echo -e "${RED}❌ Validation failed for: ${failed_steps[*]}${NC}"
        echo -e "${RED}Please fix the issues above before deploying.${NC}\n"
        return 1
    fi
}

# Run validation
main "$@"