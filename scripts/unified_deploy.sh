#!/bin/bash
# Unified Deployment Script - Consolidates 25 deployment scripts
# Created: 2025-08-21
# Replaces: deploy.sh, deploy-dev.sh, deploy-real-mcp.sh, deploy-monitoring.sh, etc.

set -e

# Configuration
COMPOSE_FILE="/opt/sutazaiapp/docker-compose.yml"
ENV_FILE="/opt/sutazaiapp/.env"
PROJECT_NAME="sutazai"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
TARGET=${1:-all}  # all, backend, frontend, mcp, monitoring, database
ACTION=${2:-deploy}  # deploy, restart, stop, status

echo -e "${BLUE}=== Unified Deployment Script ===${NC}"
echo "Target: $TARGET | Action: $ACTION"

# Function: Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed${NC}"
        exit 1
    fi
    
    # Check docker-compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}docker-compose is not installed${NC}"
        exit 1
    fi
    
    # Check compose file
    if [ ! -f "$COMPOSE_FILE" ]; then
        echo -e "${RED}docker-compose.yml not found${NC}"
        exit 1
    fi
    
    echo "Prerequisites OK"
}

# Function: Load environment
load_environment() {
    if [ -f "$ENV_FILE" ]; then
        export $(cat "$ENV_FILE" | grep -v '^#' | xargs)
        echo "Environment loaded"
    fi
}

# Function: Deploy services
deploy_services() {
    local services=""
    
    case "$TARGET" in
        all)
            echo -e "${GREEN}Deploying all services...${NC}"
            services=""
            ;;
        backend)
            services="backend postgres redis"
            ;;
        frontend)
            services="frontend"
            ;;
        mcp)
            services="mcp-orchestrator mcp-manager"
            ;;
        monitoring)
            services="prometheus grafana loki cadvisor"
            ;;
        database)
            services="postgres redis neo4j chromadb qdrant"
            ;;
        *)
            echo -e "${RED}Unknown target: $TARGET${NC}"
            exit 1
            ;;
    esac
    
    # Execute docker-compose command
    echo "Deploying: ${services:-all services}"
    docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d $services
    
    # Wait for services to be healthy
    sleep 5
    check_health "$services"
}

# Function: Stop services
stop_services() {
    local services=""
    
    case "$TARGET" in
        all)
            docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" down
            ;;
        *)
            services=$(get_service_list "$TARGET")
            docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" stop $services
            ;;
    esac
    
    echo "Services stopped"
}

# Function: Restart services
restart_services() {
    stop_services
    sleep 2
    deploy_services
}

# Function: Check service health
check_health() {
    local services=$1
    echo -e "${YELLOW}Checking service health...${NC}"
    
    # Check each service
    if [ -z "$services" ]; then
        services=$(docker-compose -f "$COMPOSE_FILE" ps --services)
    fi
    
    for service in $services; do
        container="${PROJECT_NAME}-${service}"
        if docker ps | grep -q "$container"; then
            status=$(docker inspect --format='{{.State.Status}}' "$container" 2>/dev/null || echo "unknown")
            if [ "$status" = "running" ]; then
                echo -e "  $service: ${GREEN}✓ Running${NC}"
            else
                echo -e "  $service: ${RED}✗ $status${NC}"
            fi
        else
            echo -e "  $service: ${YELLOW}Not found${NC}"
        fi
    done
}

# Function: Show status
show_status() {
    echo -e "${BLUE}=== Service Status ===${NC}"
    
    if [ "$TARGET" = "all" ]; then
        docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" ps
    else
        services=$(get_service_list "$TARGET")
        docker-compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" ps $services
    fi
    
    echo -e "\n${BLUE}=== Resource Usage ===${NC}"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" | \
        grep -E "CONTAINER|${PROJECT_NAME}" | head -20
}

# Function: Get service list for target
get_service_list() {
    case "$1" in
        backend) echo "backend postgres redis" ;;
        frontend) echo "frontend" ;;
        mcp) echo "mcp-orchestrator mcp-manager" ;;
        monitoring) echo "prometheus grafana loki cadvisor" ;;
        database) echo "postgres redis neo4j chromadb qdrant" ;;
        *) echo "" ;;
    esac
}

# Function: Post-deployment tasks
post_deployment() {
    echo -e "${YELLOW}Running post-deployment tasks...${NC}"
    
    # Wait for services to stabilize
    sleep 10
    
    # Check critical services
    if curl -s http://localhost:10010/health > /dev/null; then
        echo -e "Backend API: ${GREEN}✓ Healthy${NC}"
    else
        echo -e "Backend API: ${RED}✗ Not responding${NC}"
    fi
    
    if curl -s http://localhost:10011 > /dev/null; then
        echo -e "Frontend: ${GREEN}✓ Accessible${NC}"
    else
        echo -e "Frontend: ${RED}✗ Not accessible${NC}"
    fi
}

# Main execution
main() {
    check_prerequisites
    load_environment
    
    case "$ACTION" in
        deploy)
            deploy_services
            post_deployment
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            post_deployment
            ;;
        status)
            show_status
            ;;
        *)
            echo -e "${RED}Unknown action: $ACTION${NC}"
            echo "Usage: $0 [all|backend|frontend|mcp|monitoring|database] [deploy|stop|restart|status]"
            exit 1
            ;;
    esac
    
    echo -e "${GREEN}=== Operation Complete ===${NC}"
}

# Run main function
main