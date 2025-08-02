#!/bin/bash
# title        :localagi_service.sh
# description  :This script manages the LocalAGI service
# author       :SutazAI Team
# version      :1.0
# usage        :sudo bash scripts/localagi_service.sh [start|stop|status]
# notes        :Requires bash 4.0+ and Docker/Docker Compose

# Navigate to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Set LocalAGI path
LOCALAGI_PATH="/opt/localagi"
LOCALAGI_LOG_FILE="${PROJECT_ROOT}/logs/localagi.log"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'
NC='\033[0m' # No Color

# Create log directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/logs"

# Function to print a header
print_header() {
    echo -e "${BOLD}${1}${RESET}"
}

# Function to print info message
print_info() {
    echo -e "${BLUE}│ ${CYAN}i${BLUE} ${1}${RESET}"
}

# Function to print success message
print_success() {
    echo -e "${BLUE}│ ${GREEN}✓${BLUE} ${1}${RESET}"
}

# Function to print error message
print_error() {
    echo -e "${BLUE}│ ${RED}✗${BLUE} ${1}${RESET}"
}

# Function to print warning message
print_warning() {
    echo -e "${BLUE}│ ${YELLOW}!${BLUE} ${1}${RESET}"
}

# Function to start LocalAGI
start_localagi() {
    print_header "Starting LocalAGI service..."
    
    # Check if LocalAGI directory exists
    if [ ! -d "$LOCALAGI_PATH" ]; then
        print_error "LocalAGI directory not found at $LOCALAGI_PATH"
        return 1
    fi
    
    # Navigate to LocalAGI directory
    cd "$LOCALAGI_PATH"
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        return 1
    fi
    
    # Check for docker-compose file
    if [ -f "docker-compose.yaml" ]; then
        DOCKER_COMPOSE_FILE="docker-compose.yaml"
    elif [ -f "docker-compose.yml" ]; then
        DOCKER_COMPOSE_FILE="docker-compose.yml"
    else
        print_error "Docker Compose file not found"
        return 1
    fi
    
    # Determine which docker compose command to use
    if ! command -v docker-compose &> /dev/null; then
        print_warning "docker-compose command not found, using docker compose"
        DOCKER_COMPOSE_CMD="docker compose"
    else
        DOCKER_COMPOSE_CMD="docker-compose"
    fi
    
    # Start API server if not already running
    if ! docker ps --format '{{.Names}}' | grep -q "localagi-api"; then
        print_info "Starting LocalAI API server..."
        $DOCKER_COMPOSE_CMD up -d api
        sleep 5
    else
        print_info "LocalAI API server is already running"
    fi
    
    # Check if API server is healthy
    for i in {1..12}; do
        if curl -s http://localhost:8090/health | grep -q "status" || curl -s http://localhost:8090/readyz > /dev/null; then
            print_success "LocalAI API server is ready"
            break
        fi
        if [ $i -eq 12 ]; then
            print_error "LocalAI API server is not responding after 60 seconds"
            return 1
        fi
        print_info "Waiting for LocalAI API server to be ready... ($i/12)"
        sleep 5
    done
    
    # Start LocalAGI in detached mode
    print_info "Starting LocalAGI agent..."
    $DOCKER_COMPOSE_CMD up -d localagi
    
    print_success "LocalAGI service started"
    return 0
}

# Function to stop LocalAGI
stop_localagi() {
    print_header "Stopping LocalAGI service..."
    
    # Check if LocalAGI directory exists
    if [ ! -d "$LOCALAGI_PATH" ]; then
        print_error "LocalAGI directory not found at $LOCALAGI_PATH"
        return 1
    fi
    
    # Navigate to LocalAGI directory
    cd "$LOCALAGI_PATH"
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        return 1
    fi
    
    # Check for docker-compose file
    if [ -f "docker-compose.yaml" ]; then
        DOCKER_COMPOSE_FILE="docker-compose.yaml"
    elif [ -f "docker-compose.yml" ]; then
        DOCKER_COMPOSE_FILE="docker-compose.yml"
    else
        print_error "Docker Compose file not found"
        return 1
    fi
    
    # Determine which docker compose command to use
    if ! command -v docker-compose &> /dev/null; then
        print_warning "docker-compose command not found, using docker compose"
        DOCKER_COMPOSE_CMD="docker compose"
    else
        DOCKER_COMPOSE_CMD="docker-compose"
    fi
    
    # Stop LocalAGI containers
    print_info "Stopping LocalAGI containers..."
    $DOCKER_COMPOSE_CMD down
    
    print_success "LocalAGI service stopped"
    return 0
}

# Function to check LocalAGI status
check_localagi_status() {
    print_header "Checking LocalAGI service status..."
    
    # Check if LocalAGI directory exists
    if [ ! -d "$LOCALAGI_PATH" ]; then
        print_error "LocalAGI directory not found at $LOCALAGI_PATH"
        return 1
    fi
    
    # Navigate to LocalAGI directory
    cd "$LOCALAGI_PATH"
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        return 1
    fi
    
    # Check for docker-compose file
    if [ -f "docker-compose.yaml" ]; then
        DOCKER_COMPOSE_FILE="docker-compose.yaml"
    elif [ -f "docker-compose.yml" ]; then
        DOCKER_COMPOSE_FILE="docker-compose.yml"
    else
        print_error "Docker Compose file not found"
        return 1
    fi
    
    # Check API server status
    if docker ps --format '{{.Names}}' | grep -q "localagi-api"; then
        print_success "LocalAI API server is running"
        
        # Check if API is responding
        if curl -s http://localhost:8090/health | grep -q "status" || curl -s http://localhost:8090/readyz > /dev/null; then
            print_success "LocalAI API server is healthy"
        else
            print_warning "LocalAI API server is running but not responding properly"
        fi
    else
        print_warning "LocalAI API server is not running"
    fi
    
    # Check LocalAGI status
    if docker ps --format '{{.Names}}' | grep -q "localagi-localagi"; then
        print_success "LocalAGI agent is running"
    else
        print_warning "LocalAGI agent is not running"
    fi
    
    print_info "Docker containers status:"
    docker ps --filter "name=localagi"
    
    return 0
}

# Main function
main() {
    case "$1" in
        start)
            start_localagi
            ;;
        stop)
            stop_localagi
            ;;
        status)
            check_localagi_status
            ;;
        *)
            echo "Usage: $0 {start|stop|status}"
            exit 1
            ;;
    esac
}

# Call main function with first argument
main "$1" 