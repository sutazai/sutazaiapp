#!/bin/bash
# title        :start_sutazai.sh
# description  :This script starts all SutazAI services using Docker Compose
# author       :SutazAI Team
# version      :3.0
# usage        :bash scripts/start_sutazai.sh [--debug] [--gpu] [--skip-model-check]

# Change to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to display service status
status() {
    local service="$1"
    local status="$2"
    local color="$3"
    printf "  %-20s [%s%s\033[0m]\n" "$service" "$color" "$status"
}

# Parse command-line arguments
DEBUG=false
GPU_MODE=false
SKIP_MODEL_CHECK=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --debug) DEBUG=true; shift ;;
        --gpu) GPU_MODE=true; shift ;;
        --skip-model-check) SKIP_MODEL_CHECK=true; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Set Python path and activate virtual environment
export PYTHONPATH="${PROJECT_ROOT}"

# Activate virtual environment if it exists
if [ -f "/opt/venv-sutazaiapp/bin/activate" ]; then
    source /opt/venv-sutazaiapp/bin/activate
    log "Activated virtual environment"
else
    log "No virtual environment found at /opt/venv-sutazaiapp"
fi

# Check Python version
log "Checking system prerequisites..."
PYTHON_VERSION=$(python --version | cut -d' ' -f2)
log "Found Python version: $PYTHON_VERSION"

# Set model inference mode based on GPU availability
if [ "$GPU_MODE" = true ]; then
    log "GPU mode enabled - using GPU for model inference"
    export USE_GPU=1
else
    log "CPU mode enabled - using CPU for model inference"
    export USE_GPU=0
fi

# Function to check if port is in use
is_port_in_use() {
    nc -z localhost "$1" >/dev/null 2>&1
    return $?
}

# Function to wait for a service to be ready
wait_for_service() {
    local service_name="$1"
    local port="$2"
    local timeout="${3:-30}"  # Default timeout: 30 seconds
    
    log "Waiting for $service_name to be ready..."
    local start_time=$(date +%s)
    
    while true; do
        if nc -z localhost "$port" >/dev/null 2>&1; then
            log "$service_name is ready"
            return 0
        fi
        
        local current_time=$(date +%s)
        local elapsed=$(( current_time - start_time ))
        
        if [ "$elapsed" -ge "$timeout" ]; then
            log "Timeout waiting for $service_name"
            return 1
        fi
        
        sleep 1
    done
}

# Function to get PID of a process by pattern
get_process_pid() {
    local pattern="$1"
    local pid=$(pgrep -f "$pattern" | grep -v "$$" | head -n 1)
    echo "$pid"
}

# Function to start a service
start_service() {
    local service_name="$1"
    local command="$2"
    local pattern="$3"
    local port="$4"
    local pid_file="pids/${service_name}.pid"
    
    # Create pids directory if it doesn't exist
    mkdir -p pids
    
    # Check if service is already running by PID file
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            log "$service_name is already running with PID $pid"
            return 0
        else
            log "$service_name PID file exists but process is not running"
            rm -f "$pid_file"
        fi
    fi
    
    # Also check by pattern
    local existing_pid=$(get_process_pid "$pattern")
    if [ ! -z "$existing_pid" ]; then
        log "$service_name is already running with PID $existing_pid (detected by pattern)"
        echo $existing_pid > "$pid_file"
        return 0
    fi
    
    # Check if port is already in use
    if [ ! -z "$port" ] && is_port_in_use "$port"; then
        log "WARNING: Port $port is already in use, but $service_name is not running."
        log "Another application might be using port $port."
        return 1
    fi
    
    # Start the service
    log "Starting $service_name..."
    eval "$command" &
    local pid=$!
    echo $pid > "$pid_file"
    log "$service_name started with PID $pid"
    
    # Give it a moment to crash if it's going to
    sleep 1
    if ! ps -p "$pid" > /dev/null 2>&1; then
        log "ERROR: $service_name failed to start"
        return 1
    fi
    
    return 0
}

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    log "ERROR: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    log "ERROR: docker-compose.yml not found in $PROJECT_ROOT"
    exit 1
fi

# Create necessary directories
mkdir -p logs data tmp

# Start services using Docker Compose
log "Starting SutazAI services with Docker Compose..."

# Start core infrastructure first
log "Starting core infrastructure..."
docker compose up -d postgres redis neo4j chromadb qdrant ollama

# Wait for core services to be ready
log "Waiting for core services to initialize..."
sleep 20

# Start application services
log "Starting application services..."
docker compose up -d backend-agi frontend-agi

# Wait for services to be ready
sleep 10

# Start monitoring if requested
if [ "$DEBUG" = true ]; then
    log "Starting monitoring services..."
    docker compose up -d prometheus grafana loki promtail
fi

# Show service status
log "Checking service status..."
docker compose ps

# Display service URLs
log ""
log "==========================================="
log "SutazAI Services Started Successfully!"
log "==========================================="
log "🌐 Main Interface:    http://localhost:8501"
log "📚 API Documentation: http://localhost:8000/docs"
log "📊 Backend Health:    http://localhost:8000/health"
log "🗄️  Knowledge Graph:   http://localhost:7474"
log "🔍 Vector DB:         http://localhost:6333/dashboard"
if [ "$DEBUG" = true ]; then
    log "📈 Monitoring:        http://localhost:3000"
fi
log "==========================================="
log ""
log "To view logs: docker compose logs -f [service-name]"
log "To stop all services: ./scripts/stop_sutazai.sh"

exit 0