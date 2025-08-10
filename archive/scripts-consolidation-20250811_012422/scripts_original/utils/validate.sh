#!/bin/bash
# SutazAI Docker Configuration Validation Script
# Validates the Docker Excellence structure and configuration

set -euo pipefail


# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

VALIDATION_ERRORS=0

log() {
    echo -e "${BLUE}[VALIDATE]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
    ((VALIDATION_ERRORS++))
}

# Validate directory structure
validate_structure() {
    log "Validating Docker Excellence directory structure..."
    
    local required_dirs=(
        "docker/base"
        "docker/services"
        "docker/services/frontend"
        "docker/services/backend"
        "docker/services/agents"
        "docker/services/monitoring"
        "docker/services/infrastructure"
        "docker/compose"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [[ -d "${PROJECT_ROOT}/${dir}" ]]; then
            log_success "Directory exists: $dir"
        else
            log_error "Missing directory: $dir"
        fi
    done
}

# Validate required files
validate_files() {
    log "Validating required Docker files..."
    
    local required_files=(
        "docker/README.md"
        "docker/build.sh"
        "docker/deploy.sh"
        "docker/base/python-base.Dockerfile"
        "docker/base/agent-base.Dockerfile" 
        "docker/base/security-base.Dockerfile"
        "docker/services/frontend/Dockerfile"
        "docker/services/backend/Dockerfile"
        "docker/compose/docker-compose.yml"
        "docker/compose/docker-compose.dev.yml"
        "docker/compose/docker-compose.test.yml"
        "docker/compose/docker-compose.agents.yml"
    )
    
    for file in "${required_files[@]}"; do
        if [[ -f "${PROJECT_ROOT}/${file}" ]]; then
            log_success "File exists: $file"
        else
            log_error "Missing file: $file"
        fi
    done
}

# Validate Dockerfile best practices
validate_dockerfile_practices() {
    log "Validating Dockerfile best practices..."
    
    local dockerfiles=(
        "docker/base/python-base.Dockerfile"
        "docker/base/agent-base.Dockerfile"
        "docker/services/frontend/Dockerfile"
        "docker/services/backend/Dockerfile"
    )
    
    for dockerfile in "${dockerfiles[@]}"; do
        local filepath="${PROJECT_ROOT}/${dockerfile}"
        
        if [[ -f "$filepath" ]]; then
            # Check for multi-stage builds
            if grep -q "FROM.*as.*" "$filepath"; then
                log_success "$dockerfile: Uses multi-stage builds"
            else
                log_warning "$dockerfile: No multi-stage build detected"
            fi
            
            # Check for non-root user
            if grep -q "USER.*" "$filepath"; then
                log_success "$dockerfile: Uses non-root user"
            else
                log_error "$dockerfile: Missing non-root user"
            fi
            
            # Check for health checks
            if grep -q "HEALTHCHECK" "$filepath"; then
                log_success "$dockerfile: Has health check"
            else
                log_error "$dockerfile: Missing health check"
            fi
            
            # Check for .dockerignore equivalent practices
            if grep -q "COPY.*--chown" "$filepath"; then
                log_success "$dockerfile: Uses proper file ownership"
            else
                log_warning "$dockerfile: Consider using --chown in COPY commands"
            fi
        fi
    done
}

# Validate Docker Compose configuration
validate_compose_config() {
    log "Validating Docker Compose configurations..."
    
    local compose_files=(
        "docker/compose/docker-compose.yml"
        "docker/compose/docker-compose.dev.yml"
        "docker/compose/docker-compose.test.yml"
        "docker/compose/docker-compose.agents.yml"
    )
    
    for compose_file in "${compose_files[@]}"; do
        local filepath="${PROJECT_ROOT}/${compose_file}"
        
        if [[ -f "$filepath" ]]; then
            # Check syntax - override files need base file
            if [[ "$compose_file" == *"dev.yml" ]]; then
                # Check dev file as override
                if docker-compose -f "${PROJECT_ROOT}/docker/compose/docker-compose.yml" -f "$filepath" config >/dev/null 2>&1; then
                    log_success "$compose_file: Valid YAML syntax (as override)"
                else
                    log_error "$compose_file: Invalid YAML syntax (as override)"
                fi
            else
                # Check standalone file
                if docker-compose -f "$filepath" config >/dev/null 2>&1; then
                    log_success "$compose_file: Valid YAML syntax"
                else
                    log_error "$compose_file: Invalid YAML syntax"
                fi
            fi
            
            # Check for health checks
            if grep -q "healthcheck:" "$filepath"; then
                log_success "$compose_file: Has service health checks"
            else
                log_warning "$compose_file: Missing health checks"
            fi
            
            # Check for resource limits
            if grep -q "resources:" "$filepath"; then
                log_success "$compose_file: Has resource limits"
            else
                log_warning "$compose_file: Missing resource limits"
            fi
            
            # Check for restart policies
            if grep -q "restart:" "$filepath"; then
                log_success "$compose_file: Has restart policies"
            else
                log_warning "$compose_file: Missing restart policies"
            fi
            
            # Check for networks
            if grep -q "networks:" "$filepath"; then
                log_success "$compose_file: Uses custom networks"
            else
                log_warning "$compose_file: Using default network"
            fi
        fi
    done
}

# Validate build scripts
validate_build_scripts() {
    log "Validating build and deployment scripts..."
    
    local scripts=(
        "docker/build.sh"
        "docker/deploy.sh"
    )
    
    for script in "${scripts[@]}"; do
        local filepath="${PROJECT_ROOT}/${script}"
        
        if [[ -f "$filepath" ]]; then
            # Check if executable
            if [[ -x "$filepath" ]]; then
                log_success "$script: Is executable"
            else
                log_error "$script: Not executable"
            fi
            
            # Check for error handling
            if grep -q "set -euo pipefail" "$filepath"; then
                log_success "$script: Has proper error handling"
            else
                log_error "$script: Missing error handling (set -euo pipefail)"
            fi
            
            # Check for logging
            if grep -q "log.*(" "$filepath"; then
                log_success "$script: Has logging functions"
            else
                log_warning "$script: Consider adding logging"
            fi
        fi
    done
}

# Check for legacy files that should be moved/removed
validate_legacy_cleanup() {
    log "Checking for legacy Docker files that need cleanup..."
    
    local legacy_files=(
        "docker-compose.complete-agents.yml"
        "docker-compose.agents-simple.yml"
        "frontend/Dockerfile"  # Should be moved to docker/services/frontend/
        "backend/Dockerfile"   # Should be moved to docker/services/backend/
    )
    
    for file in "${legacy_files[@]}"; do
        if [[ -f "${PROJECT_ROOT}/${file}" ]]; then
            log_warning "Legacy file should be reviewed: $file"
        else
            log_success "No legacy file: $file"
        fi
    done
}

# Check security best practices
validate_security() {
    log "Validating security best practices..."
    
    # Check for secrets management
    if [[ -d "${PROJECT_ROOT}/secrets" ]]; then
        log_success "Secrets directory exists"
        
        if ls "${PROJECT_ROOT}/secrets"/*.txt >/dev/null 2>&1; then
            log_success "Secret files found"
            
            # Check permissions
            for secret in "${PROJECT_ROOT}/secrets"/*.txt; do
                if [[ $(stat -c %a "$secret" 2>/dev/null || stat -f %Mp%Lp "$secret" 2>/dev/null) == "600" ]]; then
                    log_success "Secret file has correct permissions: $(basename "$secret")"
                else
                    log_warning "Secret file permissions should be 600: $(basename "$secret")"
                fi
            done
        else
            log_warning "No secret files found - generate for production"
        fi
    else
        log_error "Secrets directory missing"
    fi
    
    # Check for .env files in git
    if find "${PROJECT_ROOT}" -name ".env" -not -path "*/node_modules/*" | grep -q .; then
        log_warning "Found .env files - ensure they're in .gitignore"
    else
        log_success "No .env files found in git"
    fi
}

# Generate validation report
generate_report() {
    echo ""
    echo "======================================"
    echo "Docker Excellence Validation Report"
    echo "======================================"
    echo "Timestamp: $(date)"
    echo "Project: SutazAI"
    echo "Validation Errors: $VALIDATION_ERRORS"
    echo ""
    
    if [[ $VALIDATION_ERRORS -eq 0 ]]; then
        log_success "All validations passed! Docker configuration follows Docker Excellence standards."
        echo ""
        echo "Next steps:"
        echo "1. Build base images: ./docker/build.sh base-only"
        echo "2. Build all services: ./docker/build.sh"
        echo "3. Deploy development: ENVIRONMENT=development ./docker/deploy.sh"
        echo "4. Deploy production: ENVIRONMENT=production ./docker/deploy.sh"
        return 0
    else
        log_error "Validation failed with $VALIDATION_ERRORS errors."
        echo ""
        echo "Please fix the errors above before proceeding with deployment."
        return 1
    fi
}

# Main validation
main() {
    echo "SutazAI Docker Configuration Validator"
    echo "Checking compliance with Docker Excellence standards..."
    echo ""
    
    validate_structure
    validate_files
    validate_dockerfile_practices
    validate_compose_config
    validate_build_scripts
    validate_legacy_cleanup
    validate_security
    
    generate_report
}

# Run validation
main "$@"