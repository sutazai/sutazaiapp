#!/bin/bash
# SutazAI Production Docker Build Script
# Builds all required Docker images for the SutazAI application
# Author: ULTRAFIX Deployment Specialist
# Date: August 12, 2025

set -euo pipefail

# Configuration
PROJECT_ROOT="/opt/sutazaiapp"
DOCKER_BUILD_OPTS="--no-cache"
BUILD_LOG_DIR="${PROJECT_ROOT}/logs/docker-builds"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create log directory
mkdir -p "${BUILD_LOG_DIR}"

# Function to build Docker image
build_image() {
    local image_name="$1"
    local dockerfile_path="$2"
    local context_path="$3"
    local log_file="${BUILD_LOG_DIR}/build_${image_name//:/_}.log"
    
    log "Building image: ${image_name}"
    log "Dockerfile: ${dockerfile_path}"
    log "Context: ${context_path}"
    
    if docker build ${DOCKER_BUILD_OPTS} -t "${image_name}" -f "${dockerfile_path}" "${context_path}" > "${log_file}" 2>&1; then
        log_success "Built ${image_name}"
        return 0
    else
        log_error "Failed to build ${image_name}. Check log: ${log_file}"
        return 1
    fi
}

# Function to check if image exists
image_exists() {
    docker images | grep -q "$1"
}

# Main build function
main() {
    cd "${PROJECT_ROOT}"
    
    log "Starting SutazAI Docker image build process"
    log "Project root: ${PROJECT_ROOT}"
    
    # Build order - dependencies first
    local images_to_build=(
        # Base image first
        "sutazai-python-agent-master:latest|docker/base/Dockerfile.simple-base|docker/base"
        
        # Core services
        "sutazaiapp-faiss:latest|docker/faiss/Dockerfile.standalone|docker/faiss"
        "sutazaiapp-hardware-resource-optimizer:latest|agents/hardware-resource-optimizer/Dockerfile.standalone|agents/hardware-resource-optimizer"
        "sutazaiapp-jarvis-hardware-resource-optimizer:latest|agents/jarvis-hardware-resource-optimizer/Dockerfile.standalone|agents/jarvis-hardware-resource-optimizer"
        "sutazaiapp-ollama-integration:latest|agents/ollama_integration/Dockerfile.standalone|agents/ollama_integration"
        "sutazaiapp-resource-arbitration-agent:latest|agents/resource_arbitration_agent/Dockerfile.standalone|agents/resource_arbitration_agent"
        "sutazaiapp-task-assignment-coordinator:latest|agents/task_assignment_coordinator/Dockerfile.standalone|agents/task_assignment_coordinator"
    )
    
    local build_count=0
    local success_count=0
    local failed_images=()
    
    for image_spec in "${images_to_build[@]}"; do
        IFS='|' read -r image_name dockerfile_path context_path <<< "${image_spec}"
        
        if image_exists "${image_name%:*}"; then
            log_warning "Image ${image_name} already exists, skipping..."
            continue
        fi
        
        ((build_count++))
        
        if build_image "${image_name}" "${dockerfile_path}" "${context_path}"; then
            ((success_count++))
        else
            failed_images+=("${image_name}")
        fi
    done
    
    # Summary
    log "=========================================="
    log "Build Summary:"
    log "Total images to build: ${build_count}"
    log "Successfully built: ${success_count}"
    log "Failed: $((build_count - success_count))"
    
    if [ ${#failed_images[@]} -gt 0 ]; then
        log_error "Failed images:"
        for img in "${failed_images[@]}"; do
            log_error "  - ${img}"
        done
        return 1
    else
        log_success "All images built successfully!"
        
        # List built images
        log "Built images:"
        docker images | grep -E "sutazai|sutazaiapp" | grep -v "<none>"
        
        return 0
    fi
}

# Cleanup function
cleanup() {
    log "Cleaning up build artifacts..."
    docker system prune -f > /dev/null 2>&1 || true
}

# Trap cleanup
trap cleanup EXIT

# Check prerequisites
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    log_error "Docker daemon is not running"
    exit 1
fi

# Run main function
main "$@"