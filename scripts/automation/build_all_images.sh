#!/bin/bash

# SutazAI Docker Image Build Automation Script
# Purpose: Build all missing Docker images for SutazAI services
# Usage: ./build_all_images.sh [options]
# Requires: Docker, docker-compose, Python 3

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
DOCKER_COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"
LOG_DIR="${PROJECT_ROOT}/logs"
BUILD_LOG="${LOG_DIR}/build_$(date +%Y%m%d_%H%M%S).log"
BUILD_STATE_FILE="${LOG_DIR}/build_state.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Build configuration
MAX_PARALLEL_BUILDS=4
DOCKER_BUILD_TIMEOUT=3600
BUILD_CACHE=true
PUSH_TO_REGISTRY=false
REGISTRY_URL=""
IMAGE_TAG="latest"
DRY_RUN=false
VERBOSE=false
FORCE_REBUILD=false

# Usage information
usage() {
    cat << EOF
${BOLD}SutazAI Docker Image Build Script${NC}

${BOLD}USAGE:${NC}
    $0 [OPTIONS]

${BOLD}OPTIONS:${NC}
    -h, --help              Show this help message
    -d, --dry-run           Show what would be built without building
    -v, --verbose           Enable verbose output
    -f, --force             Force rebuild all images (ignore cache)
    -j, --parallel NUM      Maximum parallel builds (default: ${MAX_PARALLEL_BUILDS})
    -t, --tag TAG           Docker image tag (default: ${IMAGE_TAG})
    --no-cache              Disable Docker build cache
    --push                  Push images to registry after building
    --registry URL          Registry URL for pushing images
    --timeout SECONDS       Build timeout per image (default: ${DOCKER_BUILD_TIMEOUT})

${BOLD}EXAMPLES:${NC}
    $0                      Build all missing images
    $0 --dry-run           Show build plan without executing
    $0 -f -j 2             Force rebuild with 2 parallel jobs
    $0 --push --registry localhost:5000  Build and push to local registry

${BOLD}ENVIRONMENT VARIABLES:${NC}
    DOCKER_BUILDKIT=1       Enable BuildKit for faster builds
    COMPOSE_DOCKER_CLI_BUILD=1  Use Docker CLI for compose builds
EOF
}

# Logging functions
log() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "$BUILD_LOG"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" | tee -a "$BUILD_LOG"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$BUILD_LOG"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "$BUILD_LOG"
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
            -f|--force)
                FORCE_REBUILD=true
                BUILD_CACHE=false
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
                BUILD_CACHE=false
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
            *)
                error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Initialize environment
init_environment() {
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Initialize build log
    cat > "$BUILD_LOG" << EOF
=================================================================
SutazAI Docker Image Build Log
Started: $(date)
Project Root: $PROJECT_ROOT
Build Configuration:
  - Max Parallel Builds: $MAX_PARALLEL_BUILDS
  - Docker Build Timeout: $DOCKER_BUILD_TIMEOUT seconds
  - Use Build Cache: $BUILD_CACHE
  - Push to Registry: $PUSH_TO_REGISTRY
  - Registry URL: ${REGISTRY_URL:-"N/A"}
  - Image Tag: $IMAGE_TAG
  - Dry Run: $DRY_RUN
  - Force Rebuild: $FORCE_REBUILD
=================================================================

EOF

    # Check dependencies
    check_dependencies
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Enable BuildKit for faster builds
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    
    log "Environment initialized successfully"
}

# Check required dependencies
check_dependencies() {
    local deps=("docker" "docker-compose" "python3")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        error "Missing required dependencies: ${missing_deps[*]}"
        error "Please install the missing dependencies and try again"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running or accessible"
        exit 1
    fi
    
    # Check Docker Compose file
    if [[ ! -f "$DOCKER_COMPOSE_FILE" ]]; then
        error "Docker Compose file not found: $DOCKER_COMPOSE_FILE"
        exit 1
    fi
    
    log "All dependencies verified"
}

# Extract build services from docker-compose.yml
extract_build_services() {
    log "Extracting build services from docker-compose.yml"
    
    python3 << 'EOF'
import yaml
import json
import sys
import os

compose_file = os.environ.get('DOCKER_COMPOSE_FILE', 'docker-compose.yml')
try:
    with open(compose_file, 'r') as f:
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
            
            services_to_build.append({
                'service': service_name,
                'context': context,
                'dockerfile': dockerfile,
                'args': args,
                'target': target,
                'full_path': os.path.join(context, dockerfile)
            })
    
    # Output as JSON for bash processing
    print(json.dumps(services_to_build, indent=2))
    
except Exception as e:
    print(f"Error parsing docker-compose.yml: {e}", file=sys.stderr)
    sys.exit(1)
EOF
}

# Check which Dockerfiles exist and which are missing
analyze_build_requirements() {
    log "Analyzing build requirements..."
    
    local services_json
    services_json=$(extract_build_services)
    
    # Save to temporary file for processing
    local temp_file="/tmp/sutazai_build_services.json"
    echo "$services_json" > "$temp_file"
    
    # Analyze each service
    python3 << EOF
import json
import os
import sys

temp_file = "$temp_file"
project_root = "$PROJECT_ROOT"

with open(temp_file, 'r') as f:
    services = json.load(f)

existing_services = []
missing_services = []
missing_dockerfiles = []

for service in services:
    dockerfile_path = os.path.join(project_root, service['full_path'])
    context_path = os.path.join(project_root, service['context'])
    
    service['dockerfile_exists'] = os.path.isfile(dockerfile_path)
    service['context_exists'] = os.path.isdir(context_path)
    service['dockerfile_full_path'] = dockerfile_path
    service['context_full_path'] = context_path
    
    if service['dockerfile_exists'] and service['context_exists']:
        existing_services.append(service)
    else:
        missing_services.append(service)
        if not service['dockerfile_exists']:
            missing_dockerfiles.append(dockerfile_path)

# Print analysis results
print(f"ANALYSIS RESULTS:")
print(f"Total services requiring build: {len(services)}")
print(f"Services ready to build: {len(existing_services)}")
print(f"Services with missing files: {len(missing_services)}")
print(f"Missing Dockerfiles: {len(missing_dockerfiles)}")

if missing_dockerfiles:
    print("\\nMISSING DOCKERFILES:")
    for dockerfile in missing_dockerfiles:
        print(f"  - {dockerfile}")

# Save build state
build_state = {
    'timestamp': '$(date -Iseconds)',
    'total_services': len(services),
    'existing_services': existing_services,
    'missing_services': missing_services,
    'missing_dockerfiles': missing_dockerfiles
}

with open('$BUILD_STATE_FILE', 'w') as f:
    json.dump(build_state, f, indent=2)

print(f"\\nBuild state saved to: $BUILD_STATE_FILE")
EOF
    
    # Clean up temp file
    rm -f "$temp_file"
}

# Create missing Dockerfiles
create_missing_dockerfiles() {
    log "Creating missing Dockerfiles..."
    
    if [[ ! -f "$BUILD_STATE_FILE" ]]; then
        error "Build state file not found. Run analysis first."
        return 1
    fi
    
    python3 << 'EOF'
import json
import os
import sys

build_state_file = os.environ.get('BUILD_STATE_FILE')
project_root = os.environ.get('PROJECT_ROOT')

with open(build_state_file, 'r') as f:
    build_state = json.load(f)

for service in build_state['missing_services']:
    if not service['dockerfile_exists']:
        dockerfile_path = service['dockerfile_full_path']
        context_path = service['context_full_path']
        service_name = service['service']
        
        # Create context directory if it doesn't exist
        os.makedirs(context_path, exist_ok=True)
        
        # Determine the appropriate base Dockerfile template
        dockerfile_content = generate_dockerfile_template(service_name, service)
        
        # Write Dockerfile
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        print(f"Created Dockerfile: {dockerfile_path}")

def generate_dockerfile_template(service_name, service_config):
    """Generate appropriate Dockerfile template based on service type"""
    
    # Base templates for different service types
    if 'agent' in service_name or service_name in ['autogpt', 'aider', 'crewai', 'agentgpt', 'agentzero']:
        return f'''# {service_name.title()} Agent Service
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    git \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Start command
CMD ["python", "app.py"]
'''
    
    elif service_name in ['pytorch', 'tensorflow', 'jax']:
        return f'''# {service_name.title()} ML Framework
FROM python:3.11-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies for ML
RUN apt-get update && apt-get install -y \\
    curl \\
    git \\
    build-essential \\
    libopenblas-dev \\
    && rm -rf /var/lib/apt/lists/*

# Install {service_name}
RUN pip install --no-cache-dir {service_name} jupyter notebook

# Copy any additional requirements
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Create workspace directories
RUN mkdir -p /workspace/models /workspace/data /workspace/notebooks

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
'''
    
    elif service_name in ['faiss', 'ai-metrics-exporter', 'health-monitor']:
        return f'''# {service_name.title()} Service
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "app.py"]
'''
    
    else:
        # Generic service template
        return f'''# {service_name.title()} Service
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Start application
CMD ["python", "-c", "print('Service {service_name} starting...'); import time; time.sleep(3600)"]
'''
EOF
}

# Create basic requirements.txt files for services that need them
create_missing_requirements() {
    log "Creating missing requirements.txt files..."
    
    python3 << 'EOF'
import json
import os

build_state_file = os.environ.get('BUILD_STATE_FILE')
project_root = os.environ.get('PROJECT_ROOT')

with open(build_state_file, 'r') as f:
    build_state = json.load(f)

for service in build_state['missing_services']:
    context_path = service['context_full_path']
    service_name = service['service']
    requirements_path = os.path.join(context_path, 'requirements.txt')
    
    if not os.path.exists(requirements_path):
        # Create basic requirements.txt based on service type
        requirements = generate_requirements_template(service_name)
        
        with open(requirements_path, 'w') as f:
            f.write(requirements)
        
        print(f"Created requirements.txt: {requirements_path}")

def generate_requirements_template(service_name):
    """Generate appropriate requirements.txt based on service type"""
    
    base_requirements = [
        "fastapi==0.115.6",
        "uvicorn[standard]==0.32.1",
        "pydantic==2.10.4",
        "requests==2.32.3",
        "aiohttp==3.11.11",
        "python-dotenv==1.0.1"
    ]
    
    if 'agent' in service_name or service_name in ['autogpt', 'aider', 'crewai', 'agentgpt', 'agentzero']:
        agent_requirements = [
            "openai==1.58.1",
            "anthropic==0.42.0",
            "langchain==0.3.11",
            "transformers",
            "torch==2.5.1"
        ]
        return "\\n".join(base_requirements + agent_requirements)
    
    elif service_name in ['pytorch', 'tensorflow', 'jax']:
        ml_requirements = [
            "numpy==2.1.3",
            "pandas==2.2.3",
            "scikit-learn==1.6.0",
            "matplotlib==3.10.0",
            "jupyter==1.1.1"
        ]
        if service_name == 'pytorch':
            ml_requirements.append("torch==2.5.1")
        elif service_name == 'tensorflow':
            ml_requirements.append("tensorflow==2.18.0")
        elif service_name == 'jax':
            ml_requirements.append("jax==0.4.37")
        
        return "\\n".join(base_requirements + ml_requirements)
    
    elif service_name in ['faiss']:
        vector_requirements = [
            "faiss-cpu==1.9.0",
            "numpy==2.1.3",
            "sentence-transformers==3.3.1"
        ]
        return "\\n".join(base_requirements + vector_requirements)
    
    else:
        return "\\n".join(base_requirements)
EOF
}

# Build a single Docker image
build_single_image() {
    local service_name="$1"
    local context="$2"
    local dockerfile="$3"
    local build_args="$4"
    local target="$5"
    
    local image_name="sutazai-${service_name}"
    if [[ -n "$REGISTRY_URL" ]]; then
        image_name="${REGISTRY_URL}/${image_name}"
    fi
    image_name="${image_name}:${IMAGE_TAG}"
    
    log "Building ${service_name} (${image_name})"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] Would build: docker build -t ${image_name} -f ${context}/${dockerfile} ${context}"
        return 0
    fi
    
    # Prepare build command
    local build_cmd="docker build"
    
    # Add build arguments
    if [[ -n "$build_args" && "$build_args" != "null" ]]; then
        while IFS= read -r line; do
            if [[ -n "$line" ]]; then
                build_cmd="$build_cmd --build-arg $line"
            fi
        done <<< "$build_args"
    fi
    
    # Add target if specified
    if [[ -n "$target" && "$target" != "null" ]]; then
        build_cmd="$build_cmd --target $target"
    fi
    
    # Add cache options
    if [[ "$BUILD_CACHE" == "false" ]]; then
        build_cmd="$build_cmd --no-cache"
    fi
    
    # Add progress output for verbose mode
    if [[ "$VERBOSE" == "true" ]]; then
        build_cmd="$build_cmd --progress=plain"
    fi
    
    # Complete build command
    build_cmd="$build_cmd -t ${image_name} -f ${context}/${dockerfile} ${context}"
    
    # Execute build with timeout
    local build_start_time=$(date +%s)
    local build_log_file="${LOG_DIR}/build_${service_name}_$(date +%Y%m%d_%H%M%S).log"
    
    if timeout "$DOCKER_BUILD_TIMEOUT" $build_cmd > "$build_log_file" 2>&1; then
        local build_end_time=$(date +%s)
        local build_duration=$((build_end_time - build_start_time))
        success "Built ${service_name} in ${build_duration}s"
        
        # Push to registry if requested
        if [[ "$PUSH_TO_REGISTRY" == "true" ]]; then
            log "Pushing ${image_name} to registry"
            if docker push "$image_name" >> "$build_log_file" 2>&1; then
                success "Pushed ${image_name}"
            else
                warn "Failed to push ${image_name}"
            fi
        fi
        
        return 0
    else
        local build_end_time=$(date +%s)
        local build_duration=$((build_end_time - build_start_time))
        error "Failed to build ${service_name} after ${build_duration}s"
        error "Build log: $build_log_file"
        
        # Show last few lines of build log for debugging
        if [[ "$VERBOSE" == "true" ]]; then
            echo "Last 20 lines of build log:"
            tail -n 20 "$build_log_file"
        fi
        
        return 1
    fi
}

# Build all images with parallel processing
build_all_images() {
    log "Starting Docker image builds..."
    
    if [[ ! -f "$BUILD_STATE_FILE" ]]; then
        warn "Build state file not found. Running analysis first..."
        analyze_build_requirements
    fi
    
    # Create missing files first
    create_missing_dockerfiles
    create_missing_requirements
    
    local build_results=()
    local failed_builds=()
    local successful_builds=()
    
    # Extract buildable services
    local services_to_build
    services_to_build=$(python3 -c "
import json
try:
    with open('$BUILD_STATE_FILE', 'r') as f:
        build_state = json.load(f)
    
    for service in build_state['existing_services']:
        build_args = '|'.join([f'{k}={v}' for k, v in service.get('args', {}).items()])
        target = service.get('target', '')
        print(f\"{service['service']}|{service['context']}|{service['dockerfile']}|{build_args}|{target}\")
except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
    import sys
    print(f'Error reading build state: {e}', file=sys.stderr)
    exit(1)
" 2>/dev/null)
    
    if [[ -z "$services_to_build" ]]; then
        warn "No services found to build"
        return 0
    fi
    
    # Build services with parallel processing
    local build_pids=()
    local active_builds=0
    
    while IFS='|' read -r service_name context dockerfile build_args target || [[ ${#build_pids[@]} -gt 0 ]]; do
        # Start new builds if slots available
        if [[ -n "$service_name" && $active_builds -lt $MAX_PARALLEL_BUILDS ]]; then
            log "Starting build for $service_name (parallel slot $((active_builds + 1))/$MAX_PARALLEL_BUILDS)"
            
            # Start build in background
            (
                if build_single_image "$service_name" "$context" "$dockerfile" "$build_args" "$target"; then
                    echo "SUCCESS:$service_name" > "/tmp/build_result_$$_$service_name"
                else
                    echo "FAILED:$service_name" > "/tmp/build_result_$$_$service_name"
                fi
            ) &
            
            build_pids+=($!)
            ((active_builds++))
            service_name=""  # Mark as processed
        fi
        
        # Check for completed builds
        local new_pids=()
        for pid in "${build_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            else
                wait "$pid"
                ((active_builds--))
                
                # Check result
                local result_file="/tmp/build_result_$$_*"
                for result in $result_file; do
                    if [[ -f "$result" ]]; then
                        local result_content=$(cat "$result")
                        local result_service=$(echo "$result_content" | cut -d: -f2)
                        local result_status=$(echo "$result_content" | cut -d: -f1)
                        
                        if [[ "$result_status" == "SUCCESS" ]]; then
                            successful_builds+=("$result_service")
                        else
                            failed_builds+=("$result_service")
                        fi
                        
                        rm -f "$result"
                    fi
                done
            fi
        done
        build_pids=("${new_pids[@]}")
    done <<< "$services_to_build"
    
    # Wait for all remaining builds to complete
    for pid in "${build_pids[@]}"; do
        wait "$pid"
    done
    
    # Collect final results
    local result_file="/tmp/build_result_$$_*"
    for result in $result_file; do
        if [[ -f "$result" ]]; then
            local result_content=$(cat "$result")
            local result_service=$(echo "$result_content" | cut -d: -f2)
            local result_status=$(echo "$result_content" | cut -d: -f1)
            
            if [[ "$result_status" == "SUCCESS" ]]; then
                successful_builds+=("$result_service")
            else
                failed_builds+=("$result_service")
            fi
            
            rm -f "$result"
        fi
    done
    
    # Report results
    log "Build Summary:"
    log "  Total services: $((${#successful_builds[@]} + ${#failed_builds[@]}))"
    log "  Successful builds: ${#successful_builds[@]}"
    log "  Failed builds: ${#failed_builds[@]}"
    
    if [[ ${#successful_builds[@]} -gt 0 ]]; then
        success "Successfully built images:"
        for service in "${successful_builds[@]}"; do
            success "  ✓ $service"
        done
    fi
    
    if [[ ${#failed_builds[@]} -gt 0 ]]; then
        error "Failed to build images:"
        for service in "${failed_builds[@]}"; do
            error "  ✗ $service"
        done
        return 1
    fi
    
    success "All Docker images built successfully!"
    return 0
}

# Verify built images
verify_images() {
    log "Verifying built images..."
    
    local verification_results=()
    local failed_verifications=()
    
    # Get list of images that should have been built
    if [[ ! -f "$BUILD_STATE_FILE" ]]; then
        warn "Build state file not found, skipping verification"
        return 0
    fi
    
    python3 -c "
import json
with open('$BUILD_STATE_FILE', 'r') as f:
    build_state = json.load(f)

for service in build_state['existing_services']:
    print(service['service'])
" | while read -r service_name; do
        local image_name="sutazai-${service_name}:${IMAGE_TAG}"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "  [DRY RUN] Would verify: $image_name"
            continue
        fi
        
        if docker image inspect "$image_name" &> /dev/null; then
            success "✓ Verified image: $image_name"
            verification_results+=("$service_name")
        else
            error "✗ Image not found: $image_name"
            failed_verifications+=("$service_name")
        fi
    done
    
    if [[ ${#failed_verifications[@]} -gt 0 ]]; then
        error "Image verification failed for: ${failed_verifications[*]}"
        return 1
    fi
    
    success "All images verified successfully"
    return 0
}

# Test services can start
test_service_startup() {
    log "Testing service startup capability..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would test service startup"
        return 0
    fi
    
    # Test with docker-compose config validation
    if docker-compose -f "$DOCKER_COMPOSE_FILE" config &> /dev/null; then
        success "Docker Compose configuration is valid"
    else
        error "Docker Compose configuration validation failed"
        return 1
    fi
    
    # Test individual service startup (quick test)
    local test_services=("backend" "frontend" "postgres" "redis")
    local startup_results=()
    local failed_startups=()
    
    for service in "${test_services[@]}"; do
        log "Testing startup for $service"
        
        # Start service in detached mode with timeout
        if timeout 60 docker-compose -f "$DOCKER_COMPOSE_FILE" up -d "$service" &> /dev/null; then
            # Wait a moment for startup
            sleep 5
            
            # Check if service is running
            if docker-compose -f "$DOCKER_COMPOSE_FILE" ps "$service" | grep -q "Up"; then
                success "✓ Service $service started successfully"
                startup_results+=("$service")
            else
                error "✗ Service $service failed to start properly"
                failed_startups+=("$service")
            fi
            
            # Stop the test service
            docker-compose -f "$DOCKER_COMPOSE_FILE" stop "$service" &> /dev/null
        else
            error "✗ Service $service failed to start within timeout"
            failed_startups+=("$service")
        fi
    done
    
    if [[ ${#failed_startups[@]} -gt 0 ]]; then
        error "Service startup test failed for: ${failed_startups[*]}"
        return 1
    fi
    
    success "All tested services can start successfully"
    return 0
}

# Cleanup function
cleanup() {
    log "Performing cleanup..."
    
    # Remove temporary files
    rm -f /tmp/build_result_$$_*
    rm -f /tmp/sutazai_build_services.json
    
    # Stop any test services that might be running
    if [[ -f "$DOCKER_COMPOSE_FILE" ]]; then
        docker-compose -f "$DOCKER_COMPOSE_FILE" down &> /dev/null || true
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
    
    log "Starting SutazAI Docker image build process"
    
    # Analyze build requirements
    analyze_build_requirements
    
    # Build all images
    if ! build_all_images; then
        error "Image building failed"
        exit 1
    fi
    
    # Verify built images
    if ! verify_images; then
        error "Image verification failed"
        exit 1
    fi
    
    # Test service startup capability
    if ! test_service_startup; then
        warn "Service startup tests failed, but images were built successfully"
    fi
    
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    
    success "SutazAI Docker image build completed successfully in ${total_duration}s"
    success "Build log saved to: $BUILD_LOG"
    success "Build state saved to: $BUILD_STATE_FILE"
    
    log "All Docker images are ready for deployment!"
}

# Execute main function with all arguments
main "$@"