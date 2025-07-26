#!/bin/bash
# start_services.sh - A reliable service starter for SutazAI
# This script coordinates the startup of all SutazAI services with proper health checks

# Navigate to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Helper functions
print_status() {
    local symbol="$1"
    local color="$2"
    local message="$3"
    echo -e "${color}${symbol} ${message}${NC}"
}

print_success() {
    print_status "✓" "${GREEN}" "$1"
}

print_error() {
    print_status "✗" "${RED}" "$1"
}

print_warning() {
    print_status "!" "${YELLOW}" "$1"
}

print_info() {
    print_status "ℹ" "${BLUE}" "$1"
}

print_section() {
    echo -e "\n${BOLD}$1${NC}"
    echo -e "${BOLD}$(printf '%*s' ${#1} | tr ' ' '-')${NC}"
}

# Create common log directory
LOGS_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOGS_DIR"

# Check systemd service status
check_systemd_service() {
    local service_name="$1"
    
    if systemctl list-unit-files | grep -q "$service_name"; then
        if systemctl is-active --quiet "$service_name"; then
            print_success "$service_name is active"
            return 0
        elif systemctl is-enabled --quiet "$service_name"; then
            print_warning "$service_name is enabled but not active, starting..."
            if sudo systemctl start "$service_name"; then
                print_success "$service_name started successfully"
                return 0
            else
                print_error "Failed to start $service_name"
                return 1
            fi
        else
            print_warning "$service_name is installed but not enabled"
            if sudo systemctl enable --now "$service_name"; then
                print_success "$service_name enabled and started successfully"
                return 0
            else
                print_error "Failed to enable and start $service_name"
                return 1
            fi
        fi
    else
        print_warning "$service_name is not installed as a systemd service"
        return 2
    fi
}

# Check if a port is in use
check_port_in_use() {
    local port="$1"
    local service_name="$2"
    
    if netstat -tuln | grep -q ":$port "; then
        local pid=$(lsof -i :$port -t 2>/dev/null || echo "unknown")
        if [ "$pid" != "unknown" ]; then
            local process_name=$(ps -p $pid -o comm= 2>/dev/null || echo "unknown")
            print_warning "Port $port is already in use by $process_name (PID: $pid)"
        else
            print_warning "Port $port is already in use by an unknown process"
        fi
        return 0
    else
        return 1
    fi
}

# Check a service health endpoint
check_health_endpoint() {
    local url="$1"
    local service_name="$2"
    local max_attempts="${3:-30}"
    
    print_info "Checking $service_name health endpoint at $url"
    for ((i=1; i<=$max_attempts; i++)); do
        if curl -s "$url" > /dev/null; then
            print_success "$service_name health check passed"
            return 0
        else
            if [ $i -eq $max_attempts ]; then
                print_error "$service_name health check failed after $max_attempts attempts"
                return 1
            fi
            echo -n "."
            sleep 1
        fi
    done
}

# Start monitoring services
print_section "Starting Monitoring Services"

# Start Node Exporter service (via systemd)
print_info "Starting Node Exporter service"
check_systemd_service "sutazai-node-exporter.service"
node_exporter_status=$?

# Start Prometheus service (via systemd)
print_info "Starting Prometheus service"
check_systemd_service "sutazai-prometheus.service"
prometheus_status=$?

# Reload systemd daemon if needed
if [ $node_exporter_status -eq 1 ] || [ $prometheus_status -eq 1 ]; then
    print_info "Reloading systemd daemon and retrying service start"
    sudo systemctl daemon-reload
    
    if [ $node_exporter_status -eq 1 ]; then
        sudo systemctl start sutazai-node-exporter.service
    fi
    
    if [ $prometheus_status -eq 1 ]; then
        sudo systemctl start sutazai-prometheus.service
    fi
fi

# Start Backend Server
print_section "Starting Backend Server"

# Check if backend is already running on port 8000
if check_port_in_use 8000 "Backend Server"; then
    print_warning "Backend appears to be already running on port 8000"
    
    # Check the backend health to verify it's working properly
    if check_health_endpoint "http://localhost:8000/api/health" "Backend Server"; then
        print_success "Existing Backend Server is operational"
    else
        print_warning "Existing Backend Server not responding to health checks"
        print_info "Stopping any existing backend process before starting a new one"
        
        # Try to use the stop_backend script if it exists
        if [ -f "${PROJECT_ROOT}/scripts/stop_backend.sh" ] && [ -x "${PROJECT_ROOT}/scripts/stop_backend.sh" ]; then
            bash "${PROJECT_ROOT}/scripts/stop_backend.sh"
        # Otherwise try to stop by PID file
        elif [ -f "${PROJECT_ROOT}/.backend.pid" ]; then
            BACKEND_PID=$(cat "${PROJECT_ROOT}/.backend.pid")
            if ps -p $BACKEND_PID > /dev/null; then
                kill $BACKEND_PID
                sleep 2
                kill -9 $BACKEND_PID 2>/dev/null || true
            fi
            rm -f "${PROJECT_ROOT}/.backend.pid"
        fi
        
        # Start backend using the improved start_backend.sh script
        print_info "Starting Backend Server with start_backend.sh"
        if [ -f "${PROJECT_ROOT}/scripts/start_backend.sh" ] && [ -x "${PROJECT_ROOT}/scripts/start_backend.sh" ]; then
            bash "${PROJECT_ROOT}/scripts/start_backend.sh"
            if [ $? -ne 0 ]; then
                print_error "Failed to start Backend Server"
                exit 1
            fi
        else
            print_error "Backend start script not found: ${PROJECT_ROOT}/scripts/start_backend.sh"
            exit 1
        fi
    fi
else
    # Start backend using the improved start_backend.sh script
    print_info "Starting Backend Server with start_backend.sh"
    if [ -f "${PROJECT_ROOT}/scripts/start_backend.sh" ] && [ -x "${PROJECT_ROOT}/scripts/start_backend.sh" ]; then
        bash "${PROJECT_ROOT}/scripts/start_backend.sh"
        if [ $? -ne 0 ]; then
            print_error "Failed to start Backend Server"
            exit 1
        fi
    else
        print_error "Backend start script not found: ${PROJECT_ROOT}/scripts/start_backend.sh"
        exit 1
    fi
fi

# Start Vector Database (Qdrant)
print_section "Starting Vector Database (Qdrant)"

# Check if Qdrant is configured to run locally
QDRANT_HOST=$(grep "QDRANT_HOST" "${PROJECT_ROOT}/.env" 2>/dev/null | cut -d '=' -f2 || echo "localhost")
QDRANT_PORT=$(grep "QDRANT_PORT" "${PROJECT_ROOT}/.env" 2>/dev/null | cut -d '=' -f2 || echo "6333")

if [[ "$QDRANT_HOST" == "localhost" || "$QDRANT_HOST" == "127.0.0.1" ]]; then
    print_info "Checking local Qdrant on port $QDRANT_PORT"
    
    # Check if Qdrant is already running
    if check_port_in_use $QDRANT_PORT "Qdrant"; then
        print_success "Qdrant is already running on port $QDRANT_PORT"
    else
        # Start Qdrant using Docker
        print_info "Starting Qdrant using Docker"
        if docker ps -a | grep -q "qdrant"; then
            # Container exists but is not running
            if ! docker ps | grep -q "qdrant"; then
                print_info "Qdrant container exists but is not running. Starting..."
                docker start qdrant
            fi
        else
            # Create and start new container
            print_info "Creating new Qdrant container"
            docker run -d --name qdrant -p $QDRANT_PORT:6333 -v ${PROJECT_ROOT}/qdrant_storage:/qdrant/storage qdrant/qdrant
        fi
        
        # Verify Qdrant is running
        sleep 5
        if curl -s "http://localhost:$QDRANT_PORT/health" > /dev/null; then
            print_success "Qdrant started successfully"
        else
            print_warning "Qdrant container started but health check failed"
        fi
    fi
else
    print_info "Qdrant is configured to use a remote host: $QDRANT_HOST:$QDRANT_PORT"
fi

# Start Web UI
print_section "Starting Web UI"

# Check if Web UI is already running on port 3000
if check_port_in_use 3000 "Web UI"; then
    print_warning "Web UI appears to be already running on port 3000"
    print_info "Stopping any existing Web UI process before starting a new one"
    
    # Try to use the stop_webui script if it exists
    if [ -f "${PROJECT_ROOT}/scripts/stop_webui.sh" ] && [ -x "${PROJECT_ROOT}/scripts/stop_webui.sh" ]; then
        bash "${PROJECT_ROOT}/scripts/stop_webui.sh"
    # Otherwise try to stop by PID file
    elif [ -f "${PROJECT_ROOT}/.webui.pid" ]; then
        WEBUI_PID=$(cat "${PROJECT_ROOT}/.webui.pid")
        if ps -p $WEBUI_PID > /dev/null; then
            kill $WEBUI_PID
            sleep 2
            kill -9 $WEBUI_PID 2>/dev/null || true
        fi
        rm -f "${PROJECT_ROOT}/.webui.pid"
    fi
fi

# Start Web UI using the improved start_webui.sh script
print_info "Starting Web UI with start_webui.sh"
if [ -f "${PROJECT_ROOT}/scripts/start_webui.sh" ] && [ -x "${PROJECT_ROOT}/scripts/start_webui.sh" ]; then
    bash "${PROJECT_ROOT}/scripts/start_webui.sh"
    if [ $? -ne 0 ]; then
        print_error "Failed to start Web UI"
    else
        # Verify Web UI is running
        sleep 5
        if check_port_in_use 3000 "Web UI" && [ -f "${PROJECT_ROOT}/.webui.pid" ]; then
            WEBUI_PID=$(cat "${PROJECT_ROOT}/.webui.pid")
            if ps -p $WEBUI_PID > /dev/null; then
                print_success "Web UI started successfully with PID: $WEBUI_PID"
            else
                print_warning "Web UI PID file exists but process is not running"
            fi
        else
            print_warning "Web UI may not have started correctly"
        fi
    fi
else
    print_error "Web UI start script not found: ${PROJECT_ROOT}/scripts/start_webui.sh"
fi

# Run health check to verify all services
print_section "Running Health Check"
print_info "Verifying all services are operational"

# Use existing health check script
if [ -f "${PROJECT_ROOT}/scripts/health_check.sh" ] && [ -x "${PROJECT_ROOT}/scripts/health_check.sh" ]; then
    bash "${PROJECT_ROOT}/scripts/health_check.sh"
    HEALTH_STATUS=$?
    
    if [ $HEALTH_STATUS -eq 0 ]; then
        print_success "All services are healthy and operational"
    elif [ $HEALTH_STATUS -eq 1 ]; then
        print_warning "Services started with warnings (see health check output for details)"
    else
        print_error "Some services failed to start properly (see health check output for details)"
    fi
else
    print_warning "Health check script not found, skipping comprehensive health verification"
    
    # Perform minimal health checks
    check_health_endpoint "http://localhost:8000/api/health" "Backend Server"
    if check_port_in_use 3000 "Web UI"; then
        print_success "Web UI is running on port 3000"
    else
        print_error "Web UI is not running on port 3000"
    fi
    
    if [[ "$QDRANT_HOST" == "localhost" || "$QDRANT_HOST" == "127.0.0.1" ]]; then
        if curl -s "http://localhost:$QDRANT_PORT/health" > /dev/null; then
            print_success "Qdrant is operational"
        else
            print_warning "Qdrant health check failed"
        fi
    fi
fi

print_section "Service Startup Complete"
print_info "Access the Web UI at: http://localhost:3000"
print_info "Access the Backend API at: http://localhost:8000"
print_info "To stop all services, run: ./scripts/stop_all.sh"
print_info "To check service health, run: ./scripts/health_check.sh" 