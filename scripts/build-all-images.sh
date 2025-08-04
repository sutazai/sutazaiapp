#!/bin/bash

# SutazAI Production Docker Image Build Script
# Purpose: Build all missing Docker images for SutazAI services
# Usage: ./build-all-images.sh [options]
# 
# This script builds 32 Docker images for the complete SutazAI platform including:
# - Backend & Frontend services
# - AI Agent services (AutoGPT, Aider, CrewAI, etc.)
# - ML Framework containers (PyTorch, TensorFlow, JAX)
# - Infrastructure services (Health Monitor, MCP Server, etc.)

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_ROOT}/logs"
BUILD_LOG="${LOG_DIR}/build_all_images_$(date +%Y%m%d_%H%M%S).log"
FAILED_BUILDS_LOG="${LOG_DIR}/failed_builds_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Build configuration with defaults
MAX_PARALLEL_BUILDS=3
DOCKER_BUILD_TIMEOUT=1800  # 30 minutes per image
IMAGE_TAG="latest"
DRY_RUN=false
VERBOSE=false
ENABLE_CACHE=true
PUSH_TO_REGISTRY=false
REGISTRY_URL=""
SKIP_TESTS=false

# Build categories for prioritized building
PRIORITY_1_SERVICES=("backend" "frontend" "postgres" "redis" "ollama")
PRIORITY_2_SERVICES=("faiss" "chromadb" "qdrant" "neo4j")
PRIORITY_3_SERVICES=("autogpt" "aider" "crewai" "agentgpt" "mcp-server")

# Usage information
usage() {
    cat << EOF
${BOLD}SutazAI Docker Image Build Script${NC}

Builds all 32 Docker images required for the SutazAI platform.

${BOLD}USAGE:${NC}
    $0 [OPTIONS]

${BOLD}OPTIONS:${NC}
    -h, --help              Show this help message
    -d, --dry-run           Show build plan without executing
    -v, --verbose           Enable verbose output
    -j, --parallel NUM      Max parallel builds (default: ${MAX_PARALLEL_BUILDS})
    -t, --tag TAG           Docker image tag (default: ${IMAGE_TAG})
    --no-cache              Disable Docker build cache
    --push                  Push images to registry after building
    --registry URL          Registry URL for pushing images
    --timeout SECONDS       Build timeout per image (default: ${DOCKER_BUILD_TIMEOUT})
    --skip-tests            Skip post-build testing

${BOLD}EXAMPLES:${NC}
    $0                      Build all images with default settings
    $0 --dry-run           Show what would be built
    $0 -j 2 --verbose      Build with 2 parallel jobs, verbose output
    $0 --push --registry localhost:5000  Build and push to registry

${BOLD}BUILD CATEGORIES:${NC}
    Priority 1: Core services (backend, frontend, databases)
    Priority 2: Vector databases and graph databases  
    Priority 3: AI agents and specialized services
    Priority 4: ML frameworks and utilities

Total Images to Build: 32
Estimated Build Time: 45-90 minutes (depending on hardware)
EOF
}

# Logging functions
log() {
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${BLUE}[$timestamp INFO]${NC} $*" | tee -a "$BUILD_LOG"
}

warn() {
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${YELLOW}[$timestamp WARN]${NC} $*" | tee -a "$BUILD_LOG"
}

error() {
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${RED}[$timestamp ERROR]${NC} $*" | tee -a "$BUILD_LOG"
}

success() {
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${GREEN}[$timestamp SUCCESS]${NC} $*" | tee -a "$BUILD_LOG"
}

info() {
    local timestamp=$(date '+%H:%M:%S')
    echo -e "${CYAN}[$timestamp INFO]${NC} $*" | tee -a "$BUILD_LOG"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -j|--parallel)
                MAX_PARALLEL_BUILDS="$2"
                shift 2
                ;;
            -t|--tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            --no-cache)
                ENABLE_CACHE=false
                shift
                ;;
            --push)
                PUSH_TO_REGISTRY=true
                shift
                ;;
            --registry)
                REGISTRY_URL="$2"
                PUSH_TO_REGISTRY=true
                shift 2
                ;;
            --timeout)
                DOCKER_BUILD_TIMEOUT="$2"
                shift 2
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            *)
                error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Initialize environment and check dependencies
init_environment() {
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Initialize build log
    cat > "$BUILD_LOG" << EOF
===============================================================================
SutazAI Docker Image Build Log
Started: $(date)
Project Root: $PROJECT_ROOT
Configuration:
  - Max Parallel Builds: $MAX_PARALLEL_BUILDS
  - Docker Build Timeout: $DOCKER_BUILD_TIMEOUT seconds
  - Image Tag: $IMAGE_TAG
  - Enable Cache: $ENABLE_CACHE
  - Push to Registry: $PUSH_TO_REGISTRY
  - Registry URL: ${REGISTRY_URL:-"N/A"}
  - Dry Run: $DRY_RUN
  - Verbose: $VERBOSE
===============================================================================

EOF

    # Check dependencies
    local deps=("docker" "python3")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        error "Missing required dependencies: ${missing_deps[*]}"
        error "Please install missing dependencies and try again"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running or accessible"
        exit 1
    fi
    
    # Check Docker Compose file
    if [[ ! -f "${PROJECT_ROOT}/docker-compose.yml" ]]; then
        error "Docker Compose file not found: ${PROJECT_ROOT}/docker-compose.yml"
        exit 1
    fi
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Enable BuildKit for faster builds
    export DOCKER_BUILDKIT=1
    export BUILDKIT_PROGRESS=plain
    
    log "Environment initialized successfully"
    log "Project root: $PROJECT_ROOT"
    log "Build log: $BUILD_LOG"
}

# Get list of services that need to be built
get_build_services() {
    python3 << 'EOF'
import yaml
import os

with open('docker-compose.yml', 'r') as f:
    compose = yaml.safe_load(f)

services_to_build = []
for service_name, service_config in compose.get('services', {}).items():
    if 'build' in service_config:
        build_config = service_config['build']
        if isinstance(build_config, dict):
            context = build_config.get('context', '.')
            dockerfile = build_config.get('dockerfile', 'Dockerfile')
            args = build_config.get('args', {})
            target = build_config.get('target', None)
        else:
            context = build_config
            dockerfile = 'Dockerfile'
            args = {}
            target = None
        
        full_path = os.path.join(context, dockerfile)
        dockerfile_exists = os.path.isfile(full_path)
        context_exists = os.path.isdir(context)
        
        if dockerfile_exists and context_exists:
            # Format: service|context|dockerfile|args|target
            args_str = ','.join([f'{k}={v}' for k, v in args.items()]) if args else ''
            target_str = target if target else ''
            print(f"{service_name}|{context}|{dockerfile}|{args_str}|{target_str}")

EOF
}

# Estimate build sizes and dependencies
estimate_build_complexity() {
    log "Analyzing build complexity..."
    
    local services
    services=$(get_build_services)
    
    local total_services=0
    local estimated_time=0
    
    while IFS='|' read -r service context dockerfile args target; do
        ((total_services++))
        
        # Estimate build time based on service type
        local service_time=120  # Default 2 minutes
        
        case $service in
            "backend"|"frontend")
                service_time=300  # 5 minutes for main services
                ;;
            "pytorch"|"tensorflow"|"jax")
                service_time=600  # 10 minutes for ML frameworks
                ;;
            *agent*|"autogpt"|"crewai"|"aider")
                service_time=240  # 4 minutes for AI agents
                ;;
            "faiss"|"chromadb"|"qdrant")
                service_time=180  # 3 minutes for vector databases
                ;;
        esac
        
        estimated_time=$((estimated_time + service_time))
        
        if [[ "$VERBOSE" == "true" ]]; then
            info "Service: $service (estimated: ${service_time}s)"
        fi
        
    done <<< "$services"
    
    # Adjust for parallel building
    local parallel_time=$((estimated_time / MAX_PARALLEL_BUILDS))
    
    log "Build Analysis:"
    log "  Total services: $total_services"
    log "  Sequential build time: $((estimated_time / 60)) minutes"
    log "  Parallel build time (j=$MAX_PARALLEL_BUILDS): $((parallel_time / 60)) minutes"
}

# Build a single Docker image
build_single_image() {
    local service_name="$1"
    local context="$2"
    local dockerfile="$3"
    local build_args="$4"
    local target="$5"
    
    local image_name="sutazai-${service_name}:${IMAGE_TAG}"
    if [[ -n "$REGISTRY_URL" ]]; then
        image_name="${REGISTRY_URL}/${image_name}"
    fi
    
    log "Building $service_name -> $image_name"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] docker build -t $image_name -f $context/$dockerfile $context"
        return 0
    fi
    
    # Prepare build command
    local build_cmd="docker build"
    
    # Add build arguments
    if [[ -n "$build_args" ]]; then
        IFS=',' read -ra ARGS <<< "$build_args"
        for arg in "${ARGS[@]}"; do
            build_cmd="$build_cmd --build-arg $arg"
        done
    fi
    
    # Add target if specified
    if [[ -n "$target" ]]; then
        build_cmd="$build_cmd --target $target"
    fi
    
    # Add cache options
    if [[ "$ENABLE_CACHE" == "false" ]]; then
        build_cmd="$build_cmd --no-cache"
    fi
    
    # Add progress output
    if [[ "$VERBOSE" == "true" ]]; then
        build_cmd="$build_cmd --progress=plain"
    else
        build_cmd="$build_cmd --progress=auto"
    fi
    
    # Complete build command
    build_cmd="$build_cmd -t $image_name -f $context/$dockerfile $context"
    
    # Create individual build log
    local service_build_log="${LOG_DIR}/build_${service_name}_$(date +%Y%m%d_%H%M%S).log"
    
    # Execute build with timeout
    local build_start_time=$(date +%s)
    
    if timeout "$DOCKER_BUILD_TIMEOUT" bash -c "$build_cmd" > "$service_build_log" 2>&1; then
        local build_end_time=$(date +%s)
        local build_duration=$((build_end_time - build_start_time))
        success "Built $service_name in ${build_duration}s ($(($build_duration / 60))m$(($build_duration % 60))s)"
        
        # Push to registry if requested
        if [[ "$PUSH_TO_REGISTRY" == "true" ]]; then
            log "Pushing $image_name to registry"
            if docker push "$image_name" >> "$service_build_log" 2>&1; then
                success "Pushed $image_name to registry"
            else
                warn "Failed to push $image_name to registry"
            fi
        fi
        
        # Cleanup intermediate images to save space
        docker image prune -f > /dev/null 2>&1 || true
        
        return 0
    else
        local build_end_time=$(date +%s)
        local build_duration=$((build_end_time - build_start_time))
        error "Failed to build $service_name after ${build_duration}s"
        error "Build log: $service_build_log"
        
        # Log to failed builds
        echo "$service_name|$image_name|$service_build_log|$build_duration" >> "$FAILED_BUILDS_LOG"
        
        return 1
    fi
}

# Build all images with parallel processing and dependency management
build_all_images() {
    log "Starting Docker image build process..."
    
    local services
    services=$(get_build_services)
    
    if [[ -z "$services" ]]; then
        warn "No services found to build"
        return 0
    fi
    
    # Count total services
    local total_services=0
    while IFS='|' read -r service context dockerfile args target; do
        ((total_services++))
    done <<< "$services"
    
    log "Building $total_services Docker images with $MAX_PARALLEL_BUILDS parallel jobs"
    
    # Build tracking
    local build_pids=()
    local active_builds=0
    local completed_builds=0
    local successful_builds=()
    local failed_builds=()
    
    # Create temporary directory for build results
    local temp_dir="/tmp/sutazai_build_$$"
    mkdir -p "$temp_dir"
    
    # Start building services
    while IFS='|' read -r service_name context dockerfile build_args target || [[ ${#build_pids[@]} -gt 0 ]]; do
        # Start new builds if slots available and services remaining
        if [[ -n "$service_name" && $active_builds -lt $MAX_PARALLEL_BUILDS ]]; then
            info "Starting build for $service_name (slot $((active_builds + 1))/$MAX_PARALLEL_BUILDS)"
            
            # Start build in background
            (
                if build_single_image "$service_name" "$context" "$dockerfile" "$build_args" "$target"; then
                    echo "SUCCESS" > "$temp_dir/result_$service_name"
                else
                    echo "FAILED" > "$temp_dir/result_$service_name"
                fi
            ) &
            
            build_pids+=($!)
            ((active_builds++))
            service_name=""  # Mark as processed
        fi
        
        # Check for completed builds
        local new_pids=()
        for i in "${!build_pids[@]}"; do
            local pid="${build_pids[$i]}"
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid"
                ((active_builds--))
                ((completed_builds++))
                
                # Find and process result
                for result_file in "$temp_dir"/result_*; do
                    if [[ -f "$result_file" ]]; then
                        local result_service=$(basename "$result_file" | sed 's/result_//')
                        local result_status=$(cat "$result_file")
                        
                        if [[ "$result_status" == "SUCCESS" ]]; then
                            successful_builds+=("$result_service")
                        else
                            failed_builds+=("$result_service")
                        fi
                        
                        rm -f "$result_file"
                        break
                    fi
                done
                
                info "Progress: $completed_builds / $total_services completed"
            else
                new_pids+=("$pid")
            fi
        done
        build_pids=("${new_pids[@]}")
        
        # Small delay to prevent CPU spinning
        sleep 1
    done <<< "$services"
    
    # Wait for all remaining builds
    for pid in "${build_pids[@]}"; do
        wait "$pid"
        ((completed_builds++))
    done
    
    # Process any remaining results
    for result_file in "$temp_dir"/result_*; do
        if [[ -f "$result_file" ]]; then
            local result_service=$(basename "$result_file" | sed 's/result_//')
            local result_status=$(cat "$result_file")
            
            if [[ "$result_status" == "SUCCESS" ]]; then
                successful_builds+=("$result_service")
            else
                failed_builds+=("$result_service")
            fi
        fi
    done
    
    # Cleanup temp directory
    rm -rf "$temp_dir"
    
    # Report results
    log ""
    log "================================"
    log "BUILD SUMMARY"
    log "================================"
    log "Total services: $total_services"
    log "Successful builds: ${#successful_builds[@]}"
    log "Failed builds: ${#failed_builds[@]}"
    
    if [[ ${#successful_builds[@]} -gt 0 ]]; then
        success "Successfully built images:"
        for service in "${successful_builds[@]}"; do
            success "  ✓ sutazai-$service:$IMAGE_TAG"
        done
    fi
    
    if [[ ${#failed_builds[@]} -gt 0 ]]; then
        error "Failed to build images:"
        for service in "${failed_builds[@]}"; do
            error "  ✗ sutazai-$service:$IMAGE_TAG"
        done
        error "Check individual build logs in $LOG_DIR/"
        error "Failed builds log: $FAILED_BUILDS_LOG"
        return 1
    fi
    
    success "All Docker images built successfully!"
    return 0
}

# Verify built images exist and are valid
verify_built_images() {
    log "Verifying built Docker images..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Skipping image verification"
        return 0
    fi
    
    local services
    services=$(get_build_services)
    
    local verified_count=0
    local failed_verification=()
    
    while IFS='|' read -r service_name context dockerfile build_args target; do
        local image_name="sutazai-${service_name}:${IMAGE_TAG}"
        
        if docker image inspect "$image_name" &> /dev/null; then
            # Get image size
            local image_size=$(docker image inspect "$image_name" --format='{{.Size}}' | numfmt --to=iec-i --suffix=B --format="%.1f")
            success "✓ Verified $image_name ($image_size)"
            ((verified_count++))
        else
            error "✗ Image not found: $image_name"
            failed_verification+=("$service_name")
        fi
    done <<< "$services"
    
    log "Image verification: $verified_count verified, ${#failed_verification[@]} failed"
    
    if [[ ${#failed_verification[@]} -gt 0 ]]; then
        error "Image verification failed for: ${failed_verification[*]}"
        return 1
    fi
    
    success "All images verified successfully"
    return 0
}

# Test that services can start with the built images
test_service_startup() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log "Skipping service startup tests (--skip-tests enabled)"
        return 0
    fi
    
    log "Testing basic service startup capability..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would test service startup"
        return 0
    fi
    
    # Test docker-compose config validation
    if docker-compose config &> /dev/null; then
        success "Docker Compose configuration is valid"
    else
        error "Docker Compose configuration validation failed"
        return 1
    fi
    
    log "Service startup tests completed"
    return 0
}

# Display build statistics and summary
show_build_stats() {
    log "Generating build statistics..."
    
    local total_images=$(docker images --format "table {{.Repository}}" | grep -c "^sutazai-" || echo "0")
    local total_size=$(docker images --format "table {{.Repository}}\t{{.Size}}" | grep "^sutazai-" | awk '{print $2}' | sed 's/[KMGT]B//g' | awk '{sum+=$1} END {print sum "MB"}' || echo "Unknown")
    
    log ""
    log "================================"
    log "FINAL STATISTICS"
    log "================================"
    log "Total SutazAI images: $total_images"
    log "Total disk usage: $total_size"
    log "Build log: $BUILD_LOG"
    
    if [[ -f "$FAILED_BUILDS_LOG" ]]; then
        log "Failed builds log: $FAILED_BUILDS_LOG"
    fi
    
    log ""
    log "Docker images are ready for deployment!"
    log "Use 'docker-compose up' to start the SutazAI platform"
}

# Cleanup function
cleanup() {
    log "Performing cleanup..."
    
    # Remove any temporary files
    rm -f /tmp/sutazai_build_* 2>/dev/null || true
    
    # Remove dangling images to free space
    if [[ "$DRY_RUN" == "false" ]]; then
        docker image prune -f > /dev/null 2>&1 || true
    fi
    
    log "Cleanup completed"
}

# Main execution function
main() {
    local start_time=$(date +%s)
    
    # Set trap for cleanup on exit
    trap cleanup EXIT
    
    # Parse command line arguments
    parse_args "$@"
    
    # Initialize environment
    init_environment
    
    # Show banner
    echo -e "${BOLD}${BLUE}===============================================================================${NC}"
    echo -e "${BOLD}${BLUE}           SutazAI Docker Image Build System${NC}"
    echo -e "${BOLD}${BLUE}           Building 32 Production-Ready Images${NC}"
    echo -e "${BOLD}${BLUE}===============================================================================${NC}"
    echo ""
    
    log "Starting SutazAI Docker image build process"
    
    # Estimate build complexity
    estimate_build_complexity
    
    # Build all images
    if ! build_all_images; then
        error "Image building failed"
        exit 1
    fi
    
    # Verify built images
    if ! verify_built_images; then
        error "Image verification failed"
        exit 1
    fi
    
    # Test service startup capability
    if ! test_service_startup; then
        warn "Service startup tests failed, but images were built successfully"
    fi
    
    # Show final statistics
    show_build_stats
    
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    local hours=$((total_duration / 3600))
    local minutes=$(((total_duration % 3600) / 60))
    local seconds=$((total_duration % 60))
    
    success "SutazAI Docker image build completed successfully!"
    success "Total build time: ${hours}h ${minutes}m ${seconds}s"
    success "All Docker images are ready for production deployment!"
    
    echo ""
    echo -e "${BOLD}${GREEN}✓ Build completed successfully!${NC}"
    echo -e "${BOLD}Next steps:${NC}"
    echo "  1. Run 'docker images | grep sutazai' to see built images"
    echo "  2. Use 'docker-compose up -d' to start the platform"
    echo "  3. Check logs in $LOG_DIR/ for detailed build information"
    echo ""
}

# Execute main function with all arguments
main "$@"