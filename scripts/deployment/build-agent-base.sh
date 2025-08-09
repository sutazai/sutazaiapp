#!/bin/bash

# SutazAI Agent Base Image Builder
# Builds optimized base image with pre-installed dependencies

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_CONTEXT="$PROJECT_ROOT/docker/agent-base"
IMAGE_NAME="sutazai/agent-base"
IMAGE_TAG="latest"
BUILD_ARGS=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
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

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build optimized SutazAI agent base image with pre-installed dependencies.

OPTIONS:
    -t, --tag TAG           Set custom image tag (default: latest)
    -n, --name NAME         Set custom image name (default: sutazai/agent-base)
    --no-cache              Build without using Docker cache
    --push                  Push image to registry after build
    --multi-platform        Build for multiple platforms (linux/amd64,linux/arm64)
    -v, --verbose           Enable verbose output
    -h, --help              Show this help message

EXAMPLES:
    $0                      # Build with default settings
    $0 -t v1.0             # Build with custom tag
    $0 --no-cache          # Build without cache
    $0 --push              # Build and push to registry

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            -n|--name)
                IMAGE_NAME="$2"
                shift 2
                ;;
            --no-cache)
                BUILD_ARGS="$BUILD_ARGS --no-cache"
                shift
                ;;
            --push)
                PUSH_IMAGE=true
                shift
                ;;
            --multi-platform)
                MULTI_PLATFORM=true
                BUILD_ARGS="$BUILD_ARGS --platform linux/amd64,linux/arm64"
                shift
                ;;
            -v|--verbose)
                set -x
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Validate prerequisites
validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check if build context exists
    if [[ ! -d "$DOCKER_CONTEXT" ]]; then
        log_error "Docker build context not found: $DOCKER_CONTEXT"
        exit 1
    fi
    
    # Check if Dockerfile exists
    if [[ ! -f "$DOCKER_CONTEXT/Dockerfile" ]]; then
        log_error "Dockerfile not found: $DOCKER_CONTEXT/Dockerfile"
        exit 1
    fi
    
    # Check if requirements file exists
    if [[ ! -f "$DOCKER_CONTEXT/agent-requirements.txt" ]]; then
        log_error "Requirements file not found: $DOCKER_CONTEXT/agent-requirements.txt"
        exit 1
    fi
    
    log_success "Prerequisites validated"
}

# Build Docker image
build_image() {
    local full_image_name="${IMAGE_NAME}:${IMAGE_TAG}"
    
    log_info "Building Docker image: $full_image_name"
    log_info "Build context: $DOCKER_CONTEXT"
    log_info "Build arguments: $BUILD_ARGS"
    
    # Add build timestamp and version labels
    local build_date=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    
    BUILD_ARGS="$BUILD_ARGS --build-arg BUILD_DATE=$build_date"
    BUILD_ARGS="$BUILD_ARGS --build-arg GIT_COMMIT=$git_commit"
    BUILD_ARGS="$BUILD_ARGS --label org.opencontainers.image.created=$build_date"
    BUILD_ARGS="$BUILD_ARGS --label org.opencontainers.image.revision=$git_commit"
    BUILD_ARGS="$BUILD_ARGS --label org.opencontainers.image.title=SutazAI-Agent-Base"
    BUILD_ARGS="$BUILD_ARGS --label org.opencontainers.image.description=Optimized-base-image-for-SutazAI-agents"
    
    # Build the image
    if [[ "${MULTI_PLATFORM:-false}" == "true" ]]; then
        # Multi-platform build requires buildx
        if ! docker buildx version &> /dev/null; then
            log_error "Docker buildx is required for multi-platform builds"
            exit 1
        fi
        
        docker buildx build \
            $BUILD_ARGS \
            -t "$full_image_name" \
            "$DOCKER_CONTEXT"
    else
        # Standard build
        docker build \
            $BUILD_ARGS \
            -t "$full_image_name" \
            "$DOCKER_CONTEXT"
    fi
    
    if [[ $? -eq 0 ]]; then
        log_success "Image built successfully: $full_image_name"
    else
        log_error "Failed to build image"
        exit 1
    fi
}

# Verify built image
verify_image() {
    local full_image_name="${IMAGE_NAME}:${IMAGE_TAG}"
    
    log_info "Verifying built image..."
    
    # Check if image exists
    if ! docker image inspect "$full_image_name" &> /dev/null; then
        log_error "Image not found after build: $full_image_name"
        exit 1
    fi
    
    # Get image size
    local image_size=$(docker image inspect "$full_image_name" --format='{{.Size}}' | numfmt --to=iec)
    log_info "Image size: $image_size"
    
    # Test basic functionality
    log_info "Testing image functionality..."
    if docker run --rm "$full_image_name" python -c "import fastapi, uvicorn, redis, pydantic; print('Dependencies verified')"; then
        log_success "Image functionality verified"
    else
        log_error "Image functionality test failed"
        exit 1
    fi
}

# Push image to registry
push_image() {
    if [[ "${PUSH_IMAGE:-false}" == "true" ]]; then
        local full_image_name="${IMAGE_NAME}:${IMAGE_TAG}"
        
        log_info "Pushing image to registry: $full_image_name"
        
        if docker push "$full_image_name"; then
            log_success "Image pushed successfully"
        else
            log_error "Failed to push image"
            exit 1
        fi
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up build artifacts..."
    # Remove dangling images
    docker image prune -f &> /dev/null || true
}

# Main execution function
main() {
    log_info "Starting SutazAI agent base image build"
    log_info "Project root: $PROJECT_ROOT"
    
    parse_args "$@"
    validate_prerequisites
    build_image
    verify_image
    push_image
    cleanup
    
    log_success "Agent base image build completed successfully!"
    log_info "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
    log_info "You can now update docker-compose files to use this optimized base image"
}

# Set trap for cleanup on exit
trap cleanup EXIT

# Execute main function
main "$@"