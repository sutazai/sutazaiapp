#!/bin/bash
#
# SutazAI Docker Images Build Script
# Version: 1.0.0
#
# DESCRIPTION:
#   Builds all required Docker images for the SutazAI system following
#   proper dependency order and error handling.
#
# PURPOSE:
#   - Build all custom Docker images before deployment
#   - Handle build dependencies and sequence
#   - Validate successful builds
#   - Provide detailed build reporting
#
# USAGE:
#   ./scripts/build_all_images.sh [OPTIONS]
#
# OPTIONS:
#   --parallel     Build images in parallel (faster but uses more resources)
#   --force        Force rebuild all images (ignore cache)
#   --quiet        Suppress verbose output
#   --validate     Validate images after building
#   --cleanup      Clean up build cache after completion
#
# REQUIREMENTS:
#   - Docker and Docker Compose v2
#   - Sufficient disk space for image builds
#   - Internet connectivity for base image downloads
#

set -euo pipefail

# Script configuration
readonly SCRIPT_VERSION="1.0.0"
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly BUILD_LOG="$PROJECT_ROOT/logs/build_$(date +%Y%m%d_%H%M%S).log"
readonly BUILD_STATE_FILE="$PROJECT_ROOT/logs/build_state.json"

# Build options
PARALLEL_BUILD=false
FORCE_REBUILD=false
QUIET_MODE=false
VALIDATE_IMAGES=false
CLEANUP_AFTER=false

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Services that require local builds (from docker-compose.yml analysis)
declare -A BUILD_SERVICES=(
    # Core Application Services
    ["backend"]="./backend"
    ["frontend"]="./frontend"
    
    # Vector and AI Services
    ["faiss"]="./docker/faiss"
    
    # AI Agent Services
    ["autogpt"]="./docker/autogpt"
    ["crewai"]="./docker/crewai"
    ["letta"]="./docker/letta"
    ["aider"]="./docker/aider"
    ["gpt-engineer"]="./docker/gpt-engineer"
    ["agentgpt"]="./docker/agentgpt"
    ["privategpt"]="./docker/privategpt"
    ["llamaindex"]="./docker/llamaindex"
    ["shellgpt"]="./docker/shellgpt"
    ["pentestgpt"]="./docker/pentestgpt"
    ["documind"]="./docker/documind"
    ["browser-use"]="./docker/browser-use"
    ["skyvern"]="./docker/skyvern"
    
    # Development and ML Services
    ["pytorch"]="./docker/pytorch"
    ["tensorflow"]="./docker/tensorflow"
    ["jax"]="./docker/jax"
    
    # Monitoring and Infrastructure
    ["ai-metrics-exporter"]="./monitoring/ai-metrics-exporter"
    ["health-monitor"]="./docker/health-check"
    ["mcp-server"]="./mcp_server"
    
    # Additional Services
    ["context-framework"]="./docker/context-framework"
    ["autogen"]="./docker/autogen"
    ["opendevin"]="./docker/opendevin"
    ["finrobot"]="./docker/finrobot"
    ["code-improver"]="./docker/code-improver"
    ["service-hub"]="./docker/service-hub"
    ["awesome-code-ai"]="./docker/awesome-code-ai"
    ["fsdp"]="./docker/fsdp"
    ["agentzero"]="./docker/agentzero"
)

# Build dependency order (services that must be built first)
declare -a BUILD_ORDER=(
    # Base services first
    "backend"
    "frontend"
    "faiss"
    "health-monitor"
    "mcp-server"
    
    # AI infrastructure
    "context-framework"
    "ai-metrics-exporter"
    
    # Core AI agents
    "autogpt"
    "agentgpt"
    "crewai"
    "letta"
    "privategpt"
    
    # Development agents
    "aider"
    "gpt-engineer"
    "opendevin"
    "code-improver"
    
    # Specialized agents
    "llamaindex"
    "shellgpt"
    "pentestgpt"
    "documind"
    "browser-use"
    "skyvern"
    "autogen"
    "agentzero"
    "finrobot"
    "awesome-code-ai"
    "service-hub"
    "fsdp"
    
    # ML frameworks last (largest builds)
    "pytorch"
    "tensorflow"
    "jax"
)

# ===============================================
# LOGGING FUNCTIONS
# ===============================================

setup_logging() {
    mkdir -p "$(dirname "$BUILD_LOG")"
    exec 1> >(tee -a "$BUILD_LOG")
    exec 2> >(tee -a "$BUILD_LOG" >&2)
    
    # Initialize build state
    cat > "$BUILD_STATE_FILE" << EOF
{
    "build_start": "$(date -Iseconds)",
    "script_version": "$SCRIPT_VERSION",
    "parallel_build": $PARALLEL_BUILD,
    "force_rebuild": $FORCE_REBUILD,
    "total_services": ${#BUILD_ORDER[@]},
    "completed_builds": [],
    "failed_builds": [],
    "build_times": {},
    "status": "in_progress"
}
EOF
}

log_info() {
    if [[ "$QUIET_MODE" != "true" ]]; then
        echo -e "${CYAN}[$(date +'%H:%M:%S')] INFO: $1${NC}"
    fi
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

log_build_phase() {
    echo -e "\n${BLUE}${BOLD}═══════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}${BOLD}BUILDING: $1${NC}"
    echo -e "${BLUE}${BOLD}═══════════════════════════════════════════════════${NC}\n"
}

# ===============================================
# BUILD VALIDATION FUNCTIONS
# ===============================================

validate_build_environment() {
    log_info "Validating build environment..."
    
    # Check Docker
    if ! command -v docker >/dev/null 2>&1; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check Docker Compose
    if ! docker compose version >/dev/null 2>&1; then
        log_error "Docker Compose v2 is not available"
        exit 1
    fi
    
    # Check disk space (need at least 20GB free)
    local available_space
    available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    local available_gb=$((available_space / 1024 / 1024))
    
    if [[ $available_gb -lt 20 ]]; then
        log_warn "Low disk space: ${available_gb}GB available (20GB+ recommended)"
        if [[ "$FORCE_REBUILD" != "true" ]]; then
            log_error "Insufficient disk space for builds. Use --force to override."
            exit 1
        fi
    fi
    
    log_success "Build environment validation passed"
}

validate_dockerfile_exists() {
    local service="$1"
    local context_path="$2"
    local dockerfile_path="$context_path/Dockerfile"
    
    if [[ ! -f "$dockerfile_path" ]]; then
        log_error "Dockerfile not found for $service: $dockerfile_path"
        return 1
    fi
    
    # Basic Dockerfile validation
    if ! grep -q "FROM" "$dockerfile_path"; then
        log_error "Invalid Dockerfile for $service: missing FROM instruction"
        return 1
    fi
    
    return 0
}

# ===============================================
# BUILD FUNCTIONS
# ===============================================

build_service_image() {
    local service="$1"
    local context_path="$2"
    local build_start_time
    build_start_time=$(date +%s)
    
    log_build_phase "$service"
    
    # Validate Dockerfile exists
    if ! validate_dockerfile_exists "$service" "$context_path"; then
        update_build_state "failed_builds" "$service"
        return 1
    fi
    
    # Prepare build arguments
    local build_args=()
    if [[ "$FORCE_REBUILD" == "true" ]]; then
        build_args+=(--no-cache)
    fi
    
    if [[ "$QUIET_MODE" == "true" ]]; then
        build_args+=(--quiet)
    else
        build_args+=(--progress=plain)
    fi
    
    # Build the image
    log_info "Building $service from $context_path..."
    
    if docker build "${build_args[@]}" \
        -t "sutazai/$service:latest" \
        -f "$context_path/Dockerfile" \
        "$context_path"; then
        
        local build_end_time
        build_end_time=$(date +%s)
        local build_duration=$((build_end_time - build_start_time))
        
        log_success "Successfully built $service (${build_duration}s)"
        update_build_state "completed_builds" "$service"
        update_build_time "$service" "$build_duration"
        return 0
    else
        log_error "Failed to build $service"
        update_build_state "failed_builds" "$service"
        return 1
    fi
}

build_all_services_sequential() {
    log_info "Building services sequentially..."
    
    local failed_builds=()
    local total_services=${#BUILD_ORDER[@]}
    local current_service=1
    
    for service in "${BUILD_ORDER[@]}"; do
        if [[ -v BUILD_SERVICES[$service] ]]; then
            local context_path="${BUILD_SERVICES[$service]}"
            
            log_info "Building service $current_service/$total_services: $service"
            
            if ! build_service_image "$service" "$context_path"; then
                failed_builds+=("$service")
                log_warn "Build failed for $service, continuing with other services..."
            fi
        else
            log_warn "Service $service not found in BUILD_SERVICES, skipping..."
        fi
        
        current_service=$((current_service + 1))
        show_progress $((current_service - 1)) $total_services "Building images"
    done
    
    if [[ ${#failed_builds[@]} -gt 0 ]]; then
        log_error "Failed to build ${#failed_builds[@]} services: ${failed_builds[*]}"
        return 1
    fi
    
    return 0
}

build_all_services_parallel() {
    log_info "Building services in parallel..."
    
    # Create build batches to avoid overwhelming the system
    local batch_size=4
    local build_pids=()
    local failed_builds=()
    
    # Build in batches
    local batch_start=0
    while [[ $batch_start -lt ${#BUILD_ORDER[@]} ]]; do
        local batch_end=$((batch_start + batch_size))
        if [[ $batch_end -gt ${#BUILD_ORDER[@]} ]]; then
            batch_end=${#BUILD_ORDER[@]}
        fi
        
        log_info "Starting build batch: services $((batch_start + 1))-$batch_end"
        
        # Start builds for this batch
        for ((i = batch_start; i < batch_end; i++)); do
            local service="${BUILD_ORDER[$i]}"
            if [[ -v BUILD_SERVICES[$service] ]]; then
                local context_path="${BUILD_SERVICES[$service]}"
                
                log_info "Starting parallel build for $service..."
                (
                    if build_service_image "$service" "$context_path"; then
                        echo "SUCCESS:$service" > "/tmp/build_result_$$_$service"
                    else
                        echo "FAILED:$service" > "/tmp/build_result_$$_$service"
                    fi
                ) &
                
                build_pids+=($!)
            fi
        done
        
        # Wait for this batch to complete
        for pid in "${build_pids[@]}"; do
            wait $pid
        done
        
        # Check results
        for ((i = batch_start; i < batch_end; i++)); do
            local service="${BUILD_ORDER[$i]}"
            if [[ -f "/tmp/build_result_$$_$service" ]]; then
                local result
                result=$(cat "/tmp/build_result_$$_$service")
                if [[ "$result" == "FAILED:$service" ]]; then
                    failed_builds+=("$service")
                fi
                rm -f "/tmp/build_result_$$_$service"
            fi
        done
        
        build_pids=()
        batch_start=$batch_end
        
        # Brief pause between batches
        sleep 2
    done
    
    if [[ ${#failed_builds[@]} -gt 0 ]]; then
        log_error "Failed to build ${#failed_builds[@]} services: ${failed_builds[*]}"
        return 1
    fi
    
    return 0
}

# ===============================================
# STATE MANAGEMENT FUNCTIONS
# ===============================================

update_build_state() {
    local key="$1"
    local service="$2"
    
    if command -v jq >/dev/null 2>&1 && [[ -f "$BUILD_STATE_FILE" ]]; then
        local temp_file
        temp_file=$(mktemp)
        jq --arg service "$service" ".${key} += [\$service]" "$BUILD_STATE_FILE" > "$temp_file"
        mv "$temp_file" "$BUILD_STATE_FILE"
    fi
}

update_build_time() {
    local service="$1"
    local duration="$2"
    
    if command -v jq >/dev/null 2>&1 && [[ -f "$BUILD_STATE_FILE" ]]; then
        local temp_file
        temp_file=$(mktemp)
        jq --arg service "$service" --arg duration "$duration" \
           ".build_times[\$service] = \$duration" "$BUILD_STATE_FILE" > "$temp_file"
        mv "$temp_file" "$BUILD_STATE_FILE"
    fi
}

finalize_build_state() {
    local status="$1"
    
    if command -v jq >/dev/null 2>&1 && [[ -f "$BUILD_STATE_FILE" ]]; then
        local temp_file
        temp_file=$(mktemp)
        jq --arg status "$status" --arg end_time "$(date -Iseconds)" \
           '.status = $status | .build_end = $end_time' "$BUILD_STATE_FILE" > "$temp_file"
        mv "$temp_file" "$BUILD_STATE_FILE"
    fi
}

# ===============================================
# VALIDATION AND CLEANUP FUNCTIONS
# ===============================================

validate_built_images() {
    if [[ "$VALIDATE_IMAGES" != "true" ]]; then
        return 0
    fi
    
    log_info "Validating built images..."
    
    local validation_failed=false
    
    for service in "${BUILD_ORDER[@]}"; do
        if [[ -v BUILD_SERVICES[$service] ]]; then
            local image_name="sutazai/$service:latest"
            
            # Check if image exists
            if docker images --format "table {{.Repository}}:{{.Tag}}" | grep -q "^$image_name$"; then
                # Try to run a basic test
                if docker run --rm --entrypoint="" "$image_name" echo "Image validation test" >/dev/null 2>&1; then
                    log_success "Image validation passed: $service"
                else
                    log_warn "Image validation failed: $service (may be normal for some services)"
                fi
            else
                log_error "Image not found: $image_name"
                validation_failed=true
            fi
        fi
    done
    
    if [[ "$validation_failed" == "true" ]]; then
        log_error "Some image validations failed"
        return 1
    fi
    
    log_success "Image validation completed successfully"
    return 0
}

cleanup_build_artifacts() {
    if [[ "$CLEANUP_AFTER" != "true" ]]; then
        return 0
    fi
    
    log_info "Cleaning up build artifacts..."
    
    # Remove dangling images
    docker image prune -f >/dev/null 2>&1 || true
    
    # Remove build cache
    docker builder prune -f >/dev/null 2>&1 || true
    
    # Clean up temporary files
    rm -f /tmp/build_result_$$_* 2>/dev/null || true
    
    log_success "Build artifacts cleaned up"
}

# ===============================================
# UTILITY FUNCTIONS
# ===============================================

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

show_build_summary() {
    echo -e "\n${BOLD}${CYAN}BUILD SUMMARY${NC}"
    echo -e "${CYAN}=============${NC}\n"
    
    local total_services=${#BUILD_ORDER[@]}
    local completed_count=0
    local failed_count=0
    
    if [[ -f "$BUILD_STATE_FILE" ]] && command -v jq >/dev/null 2>&1; then
        completed_count=$(jq '.completed_builds | length' "$BUILD_STATE_FILE" 2>/dev/null || echo "0")
        failed_count=$(jq '.failed_builds | length' "$BUILD_STATE_FILE" 2>/dev/null || echo "0")
        
        echo -e "${GREEN}✅ Successfully built: $completed_count services${NC}"
        echo -e "${RED}❌ Failed builds: $failed_count services${NC}"
        echo -e "${BLUE}📦 Total services: $total_services${NC}"
        
        if [[ $failed_count -gt 0 ]]; then
            echo -e "\n${RED}Failed services:${NC}"
            jq -r '.failed_builds[]' "$BUILD_STATE_FILE" 2>/dev/null | sed 's/^/  - /' || true
        fi
        
        echo -e "\n${CYAN}Build times:${NC}"
        jq -r '.build_times | to_entries[] | "  \(.key): \(.value)s"' "$BUILD_STATE_FILE" 2>/dev/null || true
    else
        echo "Build state information not available"
    fi
    
    echo -e "\n${CYAN}Build log: $BUILD_LOG${NC}"
    echo -e "${CYAN}Build state: $BUILD_STATE_FILE${NC}"
}

# ===============================================
# ARGUMENT PARSING
# ===============================================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --parallel)
                PARALLEL_BUILD=true
                shift
                ;;
            --force)
                FORCE_REBUILD=true
                shift
                ;;
            --quiet)
                QUIET_MODE=true
                shift
                ;;
            --validate)
                VALIDATE_IMAGES=true
                shift
                ;;
            --cleanup)
                CLEANUP_AFTER=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

show_usage() {
    cat << EOF
${BOLD}SutazAI Docker Images Build Script v${SCRIPT_VERSION}${NC}

${BOLD}DESCRIPTION:${NC}
Builds all required Docker images for the SutazAI system following proper
dependency order and error handling.

${BOLD}USAGE:${NC}
    $0 [OPTIONS]

${BOLD}OPTIONS:${NC}
    --parallel      Build images in parallel (faster but uses more resources)
    --force         Force rebuild all images (ignore cache)
    --quiet         Suppress verbose output
    --validate      Validate images after building
    --cleanup       Clean up build cache after completion
    --help, -h      Show this help message

${BOLD}EXAMPLES:${NC}
    $0                                    # Build all images sequentially
    $0 --parallel --validate              # Fast parallel build with validation
    $0 --force --cleanup                  # Force rebuild and cleanup after
    $0 --quiet --parallel                 # Quiet parallel build

${BOLD}SERVICES TO BUILD:${NC}
$(printf "    %-20s %s\n" "SERVICE" "CONTEXT")
$(for service in "${BUILD_ORDER[@]}"; do
    if [[ -v BUILD_SERVICES[$service] ]]; then
        printf "    %-20s %s\n" "$service" "${BUILD_SERVICES[$service]}"
    fi
done)

${BOLD}REQUIREMENTS:${NC}
    - Docker and Docker Compose v2
    - 20GB+ free disk space
    - Internet connectivity for base images
    - Sufficient RAM for parallel builds (if using --parallel)

EOF
}

# ===============================================
# MAIN EXECUTION
# ===============================================

main() {
    parse_arguments "$@"
    
    log_info "Starting SutazAI Docker Images Build v$SCRIPT_VERSION"
    
    # Setup and validation
    setup_logging
    validate_build_environment
    
    # Show configuration
    log_info "Build configuration:"
    log_info "  Parallel build: $PARALLEL_BUILD"
    log_info "  Force rebuild: $FORCE_REBUILD"
    log_info "  Validate images: $VALIDATE_IMAGES"
    log_info "  Cleanup after: $CLEANUP_AFTER"
    log_info "  Total services: ${#BUILD_ORDER[@]}"
    
    # Execute builds
    local build_success=true
    local build_start_time
    build_start_time=$(date +%s)
    
    if [[ "$PARALLEL_BUILD" == "true" ]]; then
        if ! build_all_services_parallel; then
            build_success=false
        fi
    else
        if ! build_all_services_sequential; then
            build_success=false
        fi
    fi
    
    local build_end_time
    build_end_time=$(date +%s)
    local total_build_time=$((build_end_time - build_start_time))
    
    # Post-build tasks
    if [[ "$build_success" == "true" ]]; then
        validate_built_images || build_success=false
    fi
    
    cleanup_build_artifacts
    
    # Finalize and report
    if [[ "$build_success" == "true" ]]; then
        finalize_build_state "completed"
        log_success "All Docker images built successfully! (${total_build_time}s total)"
    else
        finalize_build_state "failed"
        log_error "Some Docker image builds failed (${total_build_time}s total)"
    fi
    
    show_build_summary
    
    if [[ "$build_success" != "true" ]]; then
        exit 1
    fi
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi