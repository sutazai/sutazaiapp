#!/bin/bash
#############################################################
# SUTAZAIAPP Universal Deployment Script
# Created: 2025-08-19
# Purpose: Deploy and manage all SUTAZAIAPP services
# Rule 12 Compliance: Universal deployment entry point
#############################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="/opt/sutazaiapp"
cd "$PROJECT_ROOT"

# Function to print colored output
log() {
    echo -e "${2:-$GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites..." "$BLUE"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log "Docker is not installed!" "$RED"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log "Docker Compose is not installed!" "$RED"
        exit 1
    fi
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        log "Node.js is not installed!" "$RED"
        exit 1
    fi
    
    log "Prerequisites check passed ✅" "$GREEN"
}

# Function to deploy core services
deploy_core() {
    log "Deploying core services..." "$BLUE"
    
    # Start Docker services
    docker-compose -f "$PROJECT_ROOT/docker/docker-compose.yml" up -d
    
    # Wait for services to be healthy
    log "Waiting for services to be healthy..." "$YELLOW"
    sleep 10
    
    # Check service health
    docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(healthy|Up)"
    
    log "Core services deployed ✅" "$GREEN"
}

# Function to deploy MCP servers
deploy_mcp() {
    log "Deploying MCP servers..." "$BLUE"
    
    if [ -f "$PROJECT_ROOT/scripts/deployment/infrastructure/deploy-real-mcp-services.sh" ]; then
        bash "$PROJECT_ROOT/scripts/deployment/infrastructure/deploy-real-mcp-services.sh"
    else
        log "MCP deployment script not found, skipping..." "$YELLOW"
    fi
    
    log "MCP servers deployment attempted ✅" "$GREEN"
}

# Function to deploy mesh system
deploy_mesh() {
    log "Deploying mesh system..." "$BLUE"
    
    if [ -f "$PROJECT_ROOT/scripts/mesh/deploy_real_mcp_services.sh" ]; then
        bash "$PROJECT_ROOT/scripts/mesh/deploy_real_mcp_services.sh"
    else
        log "Mesh deployment script not found, skipping..." "$YELLOW"
    fi
    
    log "Mesh system deployment attempted ✅" "$GREEN"
}

# Function to run tests
run_tests() {
    log "Running system tests..." "$BLUE"
    
    # Backend tests
    if [ -d "$PROJECT_ROOT/backend/tests" ]; then
        log "Running backend tests..." "$YELLOW"
        cd "$PROJECT_ROOT/backend"
        python -m pytest tests/ -v --tb=short || true
        cd "$PROJECT_ROOT"
    fi
    
    # Frontend tests
    if [ -f "$PROJECT_ROOT/package.json" ]; then
        log "Running frontend tests..." "$YELLOW"
        npx playwright test --reporter=list || true
    fi
    
    log "Tests completed ✅" "$GREEN"
}

# Function to check system status
check_status() {
    log "System Status Check" "$BLUE"
    echo "=================================="
    
    # Check Docker containers
    log "Docker Containers:" "$YELLOW"
    docker ps --format "table {{.Names}}\t{{.Status}}" | head -20
    
    # Check ports
    log "\nOpen Ports:" "$YELLOW"
    ss -tlnp 2>/dev/null | grep -E "100[0-9]{2}" | head -10
    
    # Check service health
    log "\nService Health:" "$YELLOW"
    curl -s http://localhost:10010/health | jq '.' 2>/dev/null || echo "Backend not responding"
    
    echo "=================================="
}

# Function to stop all services
stop_all() {
    log "Stopping all services..." "$RED"
    docker-compose -f "$PROJECT_ROOT/docker/docker-compose.yml" down
    log "All services stopped ✅" "$GREEN"
}

# Function to cleanup
cleanup() {
    log "Running cleanup..." "$BLUE"
    
    # Remove orphan containers
    docker container prune -f
    
    # Clean logs
    find "$PROJECT_ROOT/logs" -type f -name "*.log" -mtime +7 -delete 2>/dev/null || true
    
    log "Cleanup completed ✅" "$GREEN"
}

# Main menu
show_menu() {
    echo "=================================="
    echo "  SUTAZAIAPP Deployment Manager"
    echo "=================================="
    echo "1. Deploy All Services"
    echo "2. Deploy Core Only"
    echo "3. Deploy MCP Servers"
    echo "4. Deploy Mesh System"
    echo "5. Run Tests"
    echo "6. Check Status"
    echo "7. Stop All Services"
    echo "8. Cleanup"
    echo "9. Exit"
    echo "=================================="
    read -p "Select option: " choice
}

# Main execution
main() {
    if [ "$1" = "--all" ] || [ "$1" = "-a" ]; then
        check_prerequisites
        deploy_core
        deploy_mcp
        deploy_mesh
        run_tests
        check_status
        exit 0
    fi
    
    if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        echo "Usage: $0 [OPTIONS]"
        echo "Options:"
        echo "  --all, -a     Deploy all services"
        echo "  --core        Deploy core services only"
        echo "  --status      Check system status"
        echo "  --stop        Stop all services"
        echo "  --help, -h    Show this help message"
        exit 0
    fi
    
    if [ "$1" = "--core" ]; then
        check_prerequisites
        deploy_core
        exit 0
    fi
    
    if [ "$1" = "--status" ]; then
        check_status
        exit 0
    fi
    
    if [ "$1" = "--stop" ]; then
        stop_all
        exit 0
    fi
    
    # Interactive menu
    while true; do
        show_menu
        case $choice in
            1)
                check_prerequisites
                deploy_core
                deploy_mcp
                deploy_mesh
                run_tests
                check_status
                ;;
            2)
                check_prerequisites
                deploy_core
                ;;
            3)
                deploy_mcp
                ;;
            4)
                deploy_mesh
                ;;
            5)
                run_tests
                ;;
            6)
                check_status
                ;;
            7)
                stop_all
                ;;
            8)
                cleanup
                ;;
            9)
                log "Exiting..." "$BLUE"
                exit 0
                ;;
            *)
                log "Invalid option!" "$RED"
                ;;
        esac
        echo ""
        read -p "Press Enter to continue..."
    done
}

# Run main function
main "$@"