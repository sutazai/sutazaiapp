#!/bin/bash
#
# Optimized Build Script for SutazAI
# Handles large ML packages and build timeouts
#

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly LOG_FILE="$PROJECT_ROOT/logs/build_$(date +%Y%m%d_%H%M%S).log"

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Services that require builds (in dependency order)
readonly BUILD_SERVICES=(
    "backend"
    "frontend"
    "faiss"
    "letta"
    "autogpt"
    "crewai"
    "aider"
    "gpt-engineer"
)

# Logging
setup_logging() {
    mkdir -p "$(dirname "$LOG_FILE")"
    exec 1> >(tee -a "$LOG_FILE")
    exec 2> >(tee -a "$LOG_FILE" >&2)
}

log_info() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] INFO: $1${NC}"
}

log_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] SUCCESS: $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARNING: $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ERROR: $1${NC}"
}

# Build individual service with optimizations
build_service() {
    local service="$1"
    local build_context=""
    local dockerfile_path=""
    
    # Determine build context and Dockerfile path
    case "$service" in
        "backend"|"frontend")
            build_context="$PROJECT_ROOT/$service"
            dockerfile_path="$build_context/Dockerfile"
            ;;
        *)
            build_context="$PROJECT_ROOT/docker/$service"
            dockerfile_path="$build_context/Dockerfile"
            ;;
    esac
    
    if [[ ! -f "$dockerfile_path" ]]; then
        log_warn "Dockerfile not found for $service at $dockerfile_path"
        return 1
    fi
    
    log_info "Building $service..."
    
    # Enable BuildKit for better caching and performance
    export DOCKER_BUILDKIT=1
    
    # Build with optimizations
    local build_args=(
        "--target" "production"
        "--cache-from" "sutazai/$service:cache"
        "--build-arg" "BUILDKIT_INLINE_CACHE=1"
        "--tag" "sutazai/$service:latest"
        "--tag" "sutazai/$service:cache"
    )
    
    # Add additional optimization for large builds
    if [[ "$service" == "backend" || "$service" == "faiss" ]]; then
        build_args+=("--build-arg" "PIP_NO_CACHE_DIR=1")
        build_args+=("--build-arg" "PIP_TIMEOUT=300")
    fi
    
    # Execute build with timeout
    if timeout 1800 docker build "${build_args[@]}" "$build_context"; then
        log_success "$service built successfully"
        return 0
    else
        log_warn "$service build failed or timed out, trying fallback..."
        
        # Fallback: build without cache
        if docker build --no-cache --tag "sutazai/$service:latest" "$build_context"; then
            log_success "$service built with fallback strategy"
            return 0
        else
            log_error "$service build failed completely"
            return 1
        fi
    fi
}

# Build services in parallel where possible
build_services_parallel() {
    log_info "Building services in optimized order..."
    
    local failed_builds=()
    local successful_builds=()
    
    # Build core services first (sequential for dependencies)
    for service in "backend" "frontend"; do
        if build_service "$service"; then
            successful_builds+=("$service")
        else
            failed_builds+=("$service")
        fi
    done
    
    # Build other services in parallel batches
    local batch_size=2
    local current_batch=()
    
    for service in "${BUILD_SERVICES[@]:2}"; do
        current_batch+=("$service")
        
        if [[ ${#current_batch[@]} -eq $batch_size ]]; then
            # Build current batch in parallel
            log_info "Building batch: ${current_batch[*]}"
            
            local pids=()
            for batch_service in "${current_batch[@]}"; do
                build_service "$batch_service" &
                pids+=($!)
            done
            
            # Wait for batch to complete
            for pid in "${pids[@]}"; do
                if wait "$pid"; then
                    successful_builds+=("$(get_service_from_pid "$pid")")
                else
                    failed_builds+=("$(get_service_from_pid "$pid")")
                fi
            done
            
            current_batch=()
        fi
    done
    
    # Build remaining services
    for service in "${current_batch[@]}"; do
        if build_service "$service"; then
            successful_builds+=("$service")
        else
            failed_builds+=("$service")
        fi
    done
    
    # Report results
    echo -e "\n${BLUE}BUILD SUMMARY${NC}"
    echo -e "============="
    echo -e "${GREEN}Successful builds (${#successful_builds[@]}): ${successful_builds[*]}${NC}"
    
    if [[ ${#failed_builds[@]} -gt 0 ]]; then
        echo -e "${RED}Failed builds (${#failed_builds[@]}): ${failed_builds[*]}${NC}"
        return 1
    else
        echo -e "${GREEN}All builds completed successfully!${NC}"
        return 0
    fi
}

# Clean up old images to free space
cleanup_old_images() {
    log_info "Cleaning up old Docker images..."
    
    # Remove dangling images
    docker image prune -f >/dev/null 2>&1 || true
    
    # Remove old SutazAI images (keep latest 2 versions)
    for service in "${BUILD_SERVICES[@]}"; do
        local old_images
        old_images=$(docker images "sutazai/$service" --format "{{.ID}}" | tail -n +3)
        if [[ -n "$old_images" ]]; then
            echo "$old_images" | xargs docker rmi -f >/dev/null 2>&1 || true
        fi
    done
    
    log_success "Cleanup completed"
}

# Pre-build optimizations
setup_build_environment() {
    log_info "Setting up optimized build environment..."
    
    # Enable BuildKit
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    
    # Create buildx builder if not exists
    if ! docker buildx inspect sutazai-builder >/dev/null 2>&1; then
        docker buildx create --name sutazai-builder --use >/dev/null 2>&1 || true
    fi
    
    # Configure BuildKit settings for better performance
    export BUILDKIT_PROGRESS=plain
    
    log_success "Build environment configured"
}

# Check system resources before building
check_build_requirements() {
    log_info "Checking system requirements for building..."
    
    local memory_gb
    memory_gb=$(free -m | awk 'NR==2{printf "%.0f", $2/1024}')
    local disk_gb
    disk_gb=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
    
    log_info "Available resources: ${memory_gb}GB RAM, ${disk_gb}GB disk"
    
    if [[ $memory_gb -lt 4 ]]; then
        log_warn "Low memory detected (${memory_gb}GB). Consider using sequential builds."
        export BUILD_MODE="sequential"
    fi
    
    if [[ $disk_gb -lt 20 ]]; then
        log_warn "Low disk space (${disk_gb}GB). Running cleanup first."
        cleanup_old_images
    fi
    
    log_success "System requirements check completed"
}

# Main build function
main() {
    local command="${1:-build}"
    
    case "$command" in
        "build"|"all")
            setup_logging
            log_info "Starting optimized build process..."
            
            setup_build_environment
            check_build_requirements
            
            if [[ "${BUILD_MODE:-parallel}" == "sequential" ]]; then
                log_info "Using sequential build mode due to resource constraints"
                for service in "${BUILD_SERVICES[@]}"; do
                    build_service "$service" || log_warn "$service build failed"
                done
            else
                build_services_parallel
            fi
            
            cleanup_old_images
            log_success "Build process completed!"
            ;;
        "service")
            if [[ -z "${2:-}" ]]; then
                log_error "Service name required. Usage: $0 service <service_name>"
                exit 1
            fi
            setup_logging
            build_service "$2"
            ;;
        "cleanup")
            setup_logging
            cleanup_old_images
            ;;
        "help"|"--help"|"-h")
            cat << EOF
SutazAI Optimized Build Script

USAGE:
    $0 [COMMAND] [OPTIONS]

COMMANDS:
    build, all      Build all services (default)
    service <name>  Build specific service
    cleanup         Clean up old Docker images
    help           Show this help

SERVICES:
    ${BUILD_SERVICES[*]}

EXAMPLES:
    $0 build
    $0 service backend
    $0 cleanup

ENVIRONMENT VARIABLES:
    BUILD_MODE      Set to 'sequential' for resource-constrained systems
    DOCKER_BUILDKIT Enable BuildKit (automatically set)

EOF
            ;;
        *)
            log_error "Unknown command: $command"
            exit 1
            ;;
    esac
}

# Helper function for parallel builds
get_service_from_pid() {
    # This would need to be implemented based on how we track PIDs
    echo "unknown"
}

# Execute main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi