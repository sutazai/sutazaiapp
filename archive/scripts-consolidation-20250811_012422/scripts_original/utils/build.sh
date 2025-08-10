#!/bin/bash
# SutazAI Docker Build Script - Docker Excellence Compliant
# Builds all Docker images following best practices

set -euo pipefail

# Configuration

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
BUILD_LOG="${SCRIPT_DIR}/build.log"
BUILD_PARALLEL=${BUILD_PARALLEL:-4}
REGISTRY=${DOCKER_REGISTRY:-"sutazai"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$BUILD_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$BUILD_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$BUILD_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$BUILD_LOG"
}

# Build base images first (dependencies for other images)
build_base_images() {
    log "Building base images..."
    
    local base_images=(
        "python-base:base/python-base.Dockerfile"
        "agent-base:base/agent-base.Dockerfile"
        "security-base:base/security-base.Dockerfile"
    )
    
    for image_def in "${base_images[@]}"; do
        IFS=':' read -r image_name dockerfile <<< "$image_def"
        
        log "Building base image: ${REGISTRY}/${image_name}"
        
        if docker build \
            -t "${REGISTRY}/${image_name}:latest" \
            -f "${SCRIPT_DIR}/${dockerfile}" \
            "${SCRIPT_DIR}/base" >> "$BUILD_LOG" 2>&1; then
            log_success "Built ${REGISTRY}/${image_name}"
        else
            log_error "Failed to build ${REGISTRY}/${image_name}"
            return 1
        fi
    done
}

# Build service images
build_service_images() {
    log "Building service images..."
    
    local service_images=(
        "frontend:services/frontend/Dockerfile:${PROJECT_ROOT}/frontend"
        "backend:services/backend/Dockerfile:${PROJECT_ROOT}/backend"
        "ollama:services/infrastructure/ollama/Dockerfile:${SCRIPT_DIR}/services/infrastructure/ollama"
        "prometheus:services/monitoring/prometheus/Dockerfile:${SCRIPT_DIR}/services/monitoring/prometheus"
    )
    
    for image_def in "${service_images[@]}"; do
        IFS=':' read -r image_name dockerfile context <<< "$image_def"
        
        log "Building service image: ${REGISTRY}/${image_name}"
        
        if docker build \
            -t "${REGISTRY}/${image_name}:latest" \
            -f "${SCRIPT_DIR}/${dockerfile}" \
            "$context" >> "$BUILD_LOG" 2>&1; then
            log_success "Built ${REGISTRY}/${image_name}"
        else
            log_error "Failed to build ${REGISTRY}/${image_name}"
            return 1
        fi
    done
}

# Build agent images (optional - only if agent directories exist)
build_agent_images() {
    log "Building agent images..."
    
    local agent_images=(
        "senior-ai-engineer:services/agents/senior-ai-engineer/Dockerfile:${PROJECT_ROOT}/agents/senior-ai-engineer"
        "deployment-automation-master:services/agents/deployment-automation-master/Dockerfile:${PROJECT_ROOT}/agents/deployment-automation-master"
        "infrastructure-devops-manager:services/agents/infrastructure-devops-manager/Dockerfile:${PROJECT_ROOT}/agents/infrastructure-devops-manager"
    )
    
    for image_def in "${agent_images[@]}"; do
        IFS=':' read -r image_name dockerfile context <<< "$image_def"
        
        # Skip if agent directory doesn't exist
        if [[ ! -d "$context" ]]; then
            log_warning "Skipping ${image_name} - directory ${context} does not exist"
            continue
        fi
        
        log "Building agent image: ${REGISTRY}/${image_name}"
        
        if docker build \
            -t "${REGISTRY}/${image_name}:latest" \
            -f "${SCRIPT_DIR}/${dockerfile}" \
            "$context" >> "$BUILD_LOG" 2>&1; then
            log_success "Built ${REGISTRY}/${image_name}"
        else
            log_warning "Failed to build ${REGISTRY}/${image_name} (optional)"
        fi
    done
}

# Validate images
validate_images() {
    log "Validating built images..."
    
    local images=(
        "${REGISTRY}/python-base:latest"
        "${REGISTRY}/agent-base:latest"
        "${REGISTRY}/security-base:latest"
        "${REGISTRY}/frontend:latest"
        "${REGISTRY}/backend:latest"
        "${REGISTRY}/ollama:latest"
        "${REGISTRY}/prometheus:latest"
    )
    
    for image in "${images[@]}"; do
        if docker image inspect "$image" >/dev/null 2>&1; then
            local size=$(docker image inspect "$image" --format='{{.Size}}' | numfmt --to=iec)
            log_success "Image $image exists (size: $size)"
        else
            log_error "Image $image not found"
            return 1
        fi
    done
}

# Security scan (if trivy is available)
security_scan() {
    if command -v trivy >/dev/null 2>&1; then
        log "Running security scans..."
        
        local critical_images=(
            "${REGISTRY}/backend:latest"
            "${REGISTRY}/frontend:latest"
        )
        
        for image in "${critical_images[@]}"; do
            log "Scanning $image for vulnerabilities..."
            trivy image --severity HIGH,CRITICAL --no-progress --quiet "$image" || {
                log_warning "Security issues found in $image"
            }
        done
    else
        log_warning "Trivy not found - skipping security scans"
    fi
}

# Tag images for different environments
tag_images() {
    local version=${1:-"latest"}
    log "Tagging images with version: $version"
    
    local images=(
        "python-base" "agent-base" "security-base"
        "frontend" "backend" "ollama" "prometheus"
    )
    
    for image in "${images[@]}"; do
        if docker image inspect "${REGISTRY}/${image}:latest" >/dev/null 2>&1; then
            docker tag "${REGISTRY}/${image}:latest" "${REGISTRY}/${image}:${version}"
            log_success "Tagged ${REGISTRY}/${image}:${version}"
        fi
    done
}

# Clean up build artifacts
cleanup() {
    log "Cleaning up build artifacts..."
    
    # Remove dangling images
    docker image prune -f >> "$BUILD_LOG" 2>&1 || true
    
    # Remove build cache (optional)
    if [[ "${CLEAN_CACHE:-false}" == "true" ]]; then
        docker builder prune -f >> "$BUILD_LOG" 2>&1 || true
        log "Cleaned build cache"
    fi
}

# Main execution
main() {
    log "Starting Docker build process for SutazAI"
    log "Build log: $BUILD_LOG"
    
    # Initialize build log
    echo "Docker Build Log - $(date)" > "$BUILD_LOG"
    
    # Check Docker availability
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running or not accessible"
        exit 1
    fi
    
    # Build in dependency order
    build_base_images || exit 1
    build_service_images || exit 1
    build_agent_images || true  # Optional
    
    # Validate and scan
    validate_images || exit 1
    security_scan || true  # Optional
    
    # Tag for environments
    if [[ -n "${BUILD_VERSION:-}" ]]; then
        tag_images "$BUILD_VERSION"
    fi
    
    # Cleanup
    cleanup
    
    log_success "Docker build completed successfully!"
    log "All images tagged with registry: $REGISTRY"
    log "Build artifacts available in: $BUILD_LOG"
}

# Handle script arguments
case "${1:-build}" in
    "build")
        main
        ;;
    "base-only")
        build_base_images
        ;;
    "services-only")
        build_service_images
        ;;
    "agents-only")
        build_agent_images
        ;;
    "validate")
        validate_images
        ;;
    "clean")
        docker system prune -af
        ;;
    *)
        echo "Usage: $0 [build|base-only|services-only|agents-only|validate|clean]"
        exit 1
        ;;
esac