#!/bin/bash
#
# Comprehensive Startup Script for Sutazai System
# Starts all services with health checks and validation
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/opt/sutazaiapp"
LOG_FILE="${PROJECT_ROOT}/startup.log"
VERIFICATION_SCRIPT="${SCRIPT_DIR}/verify_all_components.py"
PYTHON_VENV="${SCRIPT_DIR}/venv/bin/python"

# Start logging
echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting Sutazai System" | tee "${LOG_FILE}"

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    
    case $status in
        success)
            echo -e "${GREEN}✓${NC} ${message}" | tee -a "${LOG_FILE}"
            ;;
        error)
            echo -e "${RED}✗${NC} ${message}" | tee -a "${LOG_FILE}"
            ;;
        warning)
            echo -e "${YELLOW}⚠${NC} ${message}" | tee -a "${LOG_FILE}"
            ;;
        info)
            echo -e "${CYAN}ℹ${NC} ${message}" | tee -a "${LOG_FILE}"
            ;;
        *)
            echo "${message}" | tee -a "${LOG_FILE}"
            ;;
    esac
}

# Function to wait for service health
wait_for_health() {
    local service=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    print_status info "Waiting for ${service} to be healthy..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "${url}" > /dev/null 2>&1; then
            print_status success "${service} is healthy"
            return 0
        fi
        echo -n "." | tee -a "${LOG_FILE}"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_status error "${service} failed to become healthy after ${max_attempts} attempts"
    return 1
}

# Function to check Docker service
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_status error "Docker is not running"
        exit 1
    fi
    print_status success "Docker is running"
}

# Function to start core infrastructure
start_core_services() {
    echo -e "\n${BOLD}${CYAN}Starting Core Infrastructure Services...${NC}" | tee -a "${LOG_FILE}"
    
    cd "${PROJECT_ROOT}"
    
    # Start databases and infrastructure
    docker compose -f docker-compose-core.yml up -d
    
    # Wait for critical services
    wait_for_health "PostgreSQL" "http://localhost:10000" || true
    wait_for_health "Redis" "http://localhost:10001" || true
    wait_for_health "Neo4j" "http://localhost:10002" || true
    wait_for_health "RabbitMQ" "http://localhost:10005" || true
    
    print_status success "Core infrastructure started"
}

# Function to start vector databases
start_vector_services() {
    echo -e "\n${BOLD}${CYAN}Starting Vector Database Services...${NC}" | tee -a "${LOG_FILE}"
    
    cd "${PROJECT_ROOT}"
    docker compose -f docker-compose-vectors.yml up -d
    
    # Give vector DBs time to initialize
    sleep 5
    
    print_status success "Vector databases started"
}

# Function to start backend services
start_backend_services() {
    echo -e "\n${BOLD}${CYAN}Starting Backend Services...${NC}" | tee -a "${LOG_FILE}"
    
    cd "${PROJECT_ROOT}"
    docker compose -f docker-compose-backend.yml up -d
    
    # Wait for backend API
    wait_for_health "Backend API" "http://localhost:10200/health"
    
    print_status success "Backend services started"
}

# Function to start frontend
start_frontend() {
    echo -e "\n${BOLD}${CYAN}Starting Frontend Service...${NC}" | tee -a "${LOG_FILE}"
    
    cd "${PROJECT_ROOT}"
    docker compose -f docker-compose-frontend.yml up -d
    
    # Wait for Streamlit
    wait_for_health "Streamlit Frontend" "http://localhost:11000/_stcore/health"
    
    print_status success "Frontend started"
}

# Function to start AI agents
start_ai_agents() {
    echo -e "\n${BOLD}${CYAN}Starting AI Agent Services...${NC}" | tee -a "${LOG_FILE}"
    
    cd "${PROJECT_ROOT}"
    docker compose -f docker-compose-agents.yml up -d
    
    # Start MCP Bridge separately to ensure it starts last
    docker start sutazai-mcp-bridge || docker compose up -d sutazai-mcp-bridge
    
    wait_for_health "MCP Bridge" "http://localhost:11100/health"
    
    print_status success "AI agents started"
}

# Function to verify MCP servers
verify_mcp_servers() {
    echo -e "\n${BOLD}${CYAN}Verifying MCP Server Wrappers...${NC}" | tee -a "${LOG_FILE}"
    
    local all_healthy=true
    
    for wrapper in "${SCRIPT_DIR}"/mcp/wrappers/*.sh; do
        if [ -f "$wrapper" ]; then
            local server_name=$(basename "$wrapper" .sh)
            
            if "$wrapper" --selfcheck > /dev/null 2>&1; then
                print_status success "MCP server ${server_name} is healthy"
            else
                print_status warning "MCP server ${server_name} failed self-check"
                all_healthy=false
            fi
        fi
    done
    
    if [ "$all_healthy" = true ]; then
        print_status success "All MCP servers verified"
    else
        print_status warning "Some MCP servers failed verification"
    fi
}

# Function to handle OOM-killed services
restart_failed_services() {
    echo -e "\n${BOLD}${CYAN}Checking for failed services...${NC}" | tee -a "${LOG_FILE}"
    
    local failed_services=$(docker ps -a --filter "status=exited" --format "{{.Names}}" | grep "^sutazai-" || true)
    
    if [ -n "$failed_services" ]; then
        print_status warning "Found failed services, attempting restart..."
        
        for service in $failed_services; do
            print_status info "Restarting ${service}..."
            docker start "${service}"
            sleep 2
        done
    else
        print_status success "No failed services found"
    fi
}

# Function to run comprehensive verification
run_verification() {
    echo -e "\n${BOLD}${MAGENTA}Running Comprehensive System Verification...${NC}" | tee -a "${LOG_FILE}"
    
    if [ -f "${VERIFICATION_SCRIPT}" ] && [ -f "${PYTHON_VENV}" ]; then
        cd "${PROJECT_ROOT}"
        "${PYTHON_VENV}" "${VERIFICATION_SCRIPT}"
        
        # Check verification results
        if [ -f "${PROJECT_ROOT}/verification_report.json" ]; then
            local status=$(python3 -c "import json; print(json.load(open('${PROJECT_ROOT}/verification_report.json'))['overall_status'])")
            
            case $status in
                healthy)
                    print_status success "System verification: ALL SYSTEMS OPERATIONAL"
                    ;;
                degraded)
                    print_status warning "System verification: SYSTEM DEGRADED"
                    ;;
                critical)
                    print_status error "System verification: CRITICAL FAILURES DETECTED"
                    ;;
            esac
        fi
    else
        print_status warning "Verification script not available"
    fi
}

# Function to display service URLs
display_urls() {
    echo -e "\n${BOLD}${GREEN}Service URLs:${NC}" | tee -a "${LOG_FILE}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "${LOG_FILE}"
    echo "Frontend (Streamlit):     http://localhost:11000" | tee -a "${LOG_FILE}"
    echo "Backend API:              http://localhost:10200/docs" | tee -a "${LOG_FILE}"
    echo "MCP Bridge:               http://localhost:11100" | tee -a "${LOG_FILE}"
    echo "Neo4j Browser:            http://localhost:10002" | tee -a "${LOG_FILE}"
    echo "RabbitMQ Management:      http://localhost:10005" | tee -a "${LOG_FILE}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "${LOG_FILE}"
}

# Function to monitor resources
monitor_resources() {
    echo -e "\n${BOLD}${CYAN}Current Resource Usage:${NC}" | tee -a "${LOG_FILE}"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" | head -20 | tee -a "${LOG_FILE}"
}

# Main execution
main() {
    echo -e "${BOLD}${MAGENTA}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${MAGENTA}║       SUTAZAI SYSTEM COMPREHENSIVE STARTUP v1.0         ║${NC}"
    echo -e "${BOLD}${MAGENTA}╚══════════════════════════════════════════════════════════╝${NC}"
    
    # Pre-flight checks
    check_docker
    
    # Start services in order
    start_core_services
    start_vector_services
    start_backend_services
    start_frontend
    start_ai_agents
    
    # Post-startup tasks
    restart_failed_services
    verify_mcp_servers
    
    # Wait for everything to stabilize
    print_status info "Waiting for system stabilization..."
    sleep 10
    
    # Run verification
    run_verification
    
    # Display final status
    monitor_resources
    display_urls
    
    echo -e "\n${BOLD}${GREEN}Startup Complete!${NC}" | tee -a "${LOG_FILE}"
    echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
    
    # Return status based on verification
    if [ -f "${PROJECT_ROOT}/verification_report.json" ]; then
        local status=$(python3 -c "import json; print(json.load(open('${PROJECT_ROOT}/verification_report.json'))['overall_status'])")
        [ "$status" = "healthy" ] && exit 0 || exit 1
    fi
}

# Handle script termination
trap 'print_status error "Startup interrupted"; exit 1' INT TERM

# Run main function
main "$@"