#!/bin/bash

# Live Logs Monitoring Script - RECONSTRUCTED
# Based on audit showing options 1-15 were working as of 2025-08-20
# Recreated by: observability-monitoring-engineer
# Date: 2025-08-21

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Function to display header
show_header() {
    clear
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║           SUTAZAI MONITORING & LOGS DASHBOARD                ║${NC}"
    echo -e "${BLUE}║                   $(date +'%Y-%m-%d %H:%M:%S')                        ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
}

# Option 1: System Overview
show_system_overview() {
    show_header
    echo -e "\n${YELLOW}═══ System Overview ═══${NC}\n"
    
    echo -e "${CYAN}Container Status:${NC}"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.State}}" | head -20
    
    echo -e "\n${CYAN}Resource Usage:${NC}"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -20
    
    echo -e "\n${CYAN}Disk Usage:${NC}"
    df -h | grep -E "^/dev|Filesystem"
    
    echo -e "\n${CYAN}Memory Usage:${NC}"
    free -h
}

# Option 2: Live Logs (All Services)
show_live_logs() {
    show_header
    echo -e "\n${YELLOW}═══ Live Logs - All Services ═══${NC}\n"
    echo -e "${CYAN}Starting combined log stream (Ctrl+C to stop)...${NC}\n"
    
    # Show logs from main services
    docker-compose logs -f --tail=50 2>/dev/null || \
    docker logs -f --tail=50 sutazai-backend sutazai-frontend 2>&1
}

# Option 3: Test API Endpoints
test_api_endpoints() {
    show_header
    echo -e "\n${YELLOW}═══ Testing API Endpoints ═══${NC}\n"
    
    endpoints=(
        "http://localhost:10010/health|Backend API"
        "http://localhost:10011|Frontend UI"
        "http://localhost:10200/-/healthy|Prometheus"
        "http://localhost:10201/api/health|Grafana"
        "http://localhost:10202/ready|Loki"
        "http://localhost:10006/v1/agent/self|Consul"
        "http://localhost:10000|PostgreSQL"
        "http://localhost:10001|Redis"
        "http://localhost:10002|Neo4j Browser"
        "http://localhost:10100/api/v1|ChromaDB"
        "http://localhost:10101/dashboard|Qdrant"
    )
    
    for endpoint in "${endpoints[@]}"; do
        IFS='|' read -r url name <<< "$endpoint"
        echo -n "Testing $name... "
        if curl -s -f -o /dev/null "$url" 2>/dev/null; then
            echo -e "${GREEN}✓ UP${NC}"
        else
            echo -e "${RED}✗ DOWN${NC}"
        fi
    done
}

# Option 4: Container Statistics
show_container_stats() {
    show_header
    echo -e "\n${YELLOW}═══ Container Statistics ═══${NC}\n"
    watch -n 2 docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
}

# Option 5: Log Management
manage_logs() {
    show_header
    echo -e "\n${YELLOW}═══ Log Management ═══${NC}\n"
    
    echo "1. View Backend Logs"
    echo "2. View Frontend Logs"
    echo "3. View Database Logs"
    echo "4. Clear Old Logs"
    echo "5. Export Logs"
    echo "6. Return to Main Menu"
    
    read -p "Select option: " log_choice
    
    case $log_choice in
        1) docker logs --tail=100 sutazai-backend ;;
        2) docker logs --tail=100 sutazai-frontend ;;
        3) docker logs --tail=100 sutazai-postgres ;;
        4) echo "Clearing logs older than 7 days..."; find /var/lib/docker/containers -name "*.log" -mtime +7 -delete ;;
        5) echo "Exporting logs..."; docker-compose logs > logs_export_$(date +%Y%m%d_%H%M%S).txt ;;
        6) return ;;
    esac
}

# Option 6: Debug Controls
debug_controls() {
    show_header
    echo -e "\n${YELLOW}═══ Debug Controls ═══${NC}\n"
    
    echo "1. Enable Debug Mode"
    echo "2. Disable Debug Mode"
    echo "3. View Debug Logs"
    echo "4. Set Log Level to DEBUG"
    echo "5. Set Log Level to INFO"
    echo "6. Set Log Level to ERROR"
    echo "7. Return to Main Menu"
    
    read -p "Select option: " debug_choice
    
    case $debug_choice in
        1) export DEBUG=true; echo "Debug mode enabled" ;;
        2) unset DEBUG; echo "Debug mode disabled" ;;
        3) journalctl -xe --no-pager | tail -100 ;;
        4) export LOG_LEVEL=DEBUG; echo "Log level set to DEBUG" ;;
        5) export LOG_LEVEL=INFO; echo "Log level set to INFO" ;;
        6) export LOG_LEVEL=ERROR; echo "Log level set to ERROR" ;;
        7) return ;;
    esac
}

# Option 7: Database Repair
repair_databases() {
    show_header
    echo -e "\n${YELLOW}═══ Database Repair ═══${NC}\n"
    
    echo "Checking database health..."
    
    # PostgreSQL
    echo -n "PostgreSQL: "
    if docker exec sutazai-postgres pg_isready 2>/dev/null; then
        echo -e "${GREEN}✓ Healthy${NC}"
    else
        echo -e "${RED}✗ Needs repair${NC}"
        echo "Attempting to repair PostgreSQL..."
        docker restart sutazai-postgres
    fi
    
    # Redis
    echo -n "Redis: "
    if docker exec sutazai-redis redis-cli ping 2>/dev/null | grep -q PONG; then
        echo -e "${GREEN}✓ Healthy${NC}"
    else
        echo -e "${RED}✗ Needs repair${NC}"
        echo "Attempting to repair Redis..."
        docker restart sutazai-redis
    fi
    
    # Neo4j
    echo -n "Neo4j: "
    if curl -s http://localhost:10002 2>/dev/null | grep -q neo4j; then
        echo -e "${GREEN}✓ Healthy${NC}"
    else
        echo -e "${RED}✗ Needs repair${NC}"
        echo "Attempting to repair Neo4j..."
        docker restart sutazai-neo4j
    fi
}

# Option 8: System Repair
repair_system() {
    show_header
    echo -e "\n${YELLOW}═══ System Repair ═══${NC}\n"
    
    echo "Running system diagnostics..."
    
    # Check and fix Docker
    echo "Checking Docker daemon..."
    if ! docker ps > /dev/null 2>&1; then
        echo "Docker daemon not responding. Attempting restart..."
        sudo systemctl restart docker
    fi
    
    # Clean up stopped containers
    echo "Cleaning up stopped containers..."
    docker container prune -f
    
    # Clean up unused images
    echo "Cleaning up unused images..."
    docker image prune -f
    
    # Clean up unused volumes
    echo "Cleaning up unused volumes..."
    docker volume prune -f
    
    # Clean up unused networks
    echo "Cleaning up unused networks..."
    docker network prune -f
    
    echo -e "\n${GREEN}System repair complete!${NC}"
}

# Option 9: Restart All Services
restart_all_services() {
    show_header
    echo -e "\n${YELLOW}═══ Restarting All Services ═══${NC}\n"
    
    echo "Stopping all services..."
    docker-compose down
    
    echo "Starting all services..."
    docker-compose up -d
    
    echo "Waiting for services to be ready..."
    sleep 10
    
    echo -e "\n${GREEN}All services restarted!${NC}"
}

# Option 10: Unified Live Logs
show_unified_live_logs() {
    show_header
    echo -e "\n${YELLOW}═══ Unified Live Logs Stream ═══${NC}\n"
    
    echo -e "${CYAN}Streaming logs from all containers (Ctrl+C to stop)...${NC}\n"
    
    # Stream logs from all containers with formatting
    docker ps --format '{{.Names}}' | xargs -I {} sh -c 'docker logs -f --tail=10 {} 2>&1 | sed "s/^/[{}] /"' 
}

# Option 11: Docker Troubleshooting
docker_troubleshooting() {
    show_header
    echo -e "\n${YELLOW}═══ Docker Troubleshooting ═══${NC}\n"
    
    echo "Docker Version:"
    docker --version
    
    echo -e "\nDocker Info:"
    docker info | head -20
    
    echo -e "\nDocker Networks:"
    docker network ls
    
    echo -e "\nDocker Volumes:"
    docker volume ls
    
    echo -e "\nProblematic Containers:"
    docker ps -a --filter "status=exited" --filter "status=dead"
}

# Option 12: Redeploy All Containers
redeploy_containers() {
    show_header
    echo -e "\n${YELLOW}═══ Redeploying All Containers ═══${NC}\n"
    
    echo "WARNING: This will recreate all containers!"
    read -p "Are you sure? (y/N): " confirm
    
    if [[ $confirm == "y" || $confirm == "Y" ]]; then
        echo "Stopping all containers..."
        docker-compose down
        
        echo "Removing all containers..."
        docker-compose rm -f
        
        echo "Rebuilding and starting containers..."
        docker-compose up -d --build
        
        echo -e "\n${GREEN}All containers redeployed!${NC}"
    else
        echo "Redeploy cancelled."
    fi
}

# Option 13: Smart Health Check
smart_health_check() {
    show_header
    echo -e "\n${YELLOW}═══ Smart Health Check ═══${NC}\n"
    
    # Check critical services
    services=(
        "sutazai-backend:10010:/health:Backend API"
        "sutazai-frontend:10011:/:Frontend UI"
        "sutazai-postgres:10000:/:PostgreSQL"
        "sutazai-redis:10001:/:Redis"
        "sutazai-prometheus:10200:/-/healthy:Prometheus"
        "sutazai-grafana:10201:/api/health:Grafana"
    )
    
    for service in "${services[@]}"; do
        IFS=':' read -r container port path name <<< "$service"
        echo -n "Checking $name... "
        
        # First check if container is running
        if docker ps | grep -q "$container"; then
            # Then check if service is responding
            if curl -s -f "http://localhost:$port$path" > /dev/null 2>&1; then
                echo -e "${GREEN}✓ Healthy${NC}"
            else
                echo -e "${YELLOW}⚠ Running but not responding${NC}"
            fi
        else
            echo -e "${RED}✗ Container not running${NC}"
        fi
    done
    
    # Check disk space
    echo -e "\n${CYAN}Disk Space:${NC}"
    df -h / | tail -1 | awk '{print "  Used: "$3" / "$2" ("$5")"}'
    
    # Check memory
    echo -e "\n${CYAN}Memory:${NC}"
    free -h | grep Mem | awk '{print "  Used: "$3" / "$2}'
}

# Option 14: Container Health Status
container_health_status() {
    show_header
    echo -e "\n${YELLOW}═══ Container Health Status ═══${NC}\n"
    
    # Get health status of all containers
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.State}}" | while read line; do
        if echo "$line" | grep -q "healthy"; then
            echo -e "${GREEN}$line${NC}"
        elif echo "$line" | grep -q "unhealthy"; then
            echo -e "${RED}$line${NC}"
        elif echo "$line" | grep -q "starting"; then
            echo -e "${YELLOW}$line${NC}"
        else
            echo "$line"
        fi
    done
}

# Option 15: Selective Service Deployment
selective_deployment() {
    show_header
    echo -e "\n${YELLOW}═══ Selective Service Deployment ═══${NC}\n"
    
    echo "Available services:"
    echo "1. Backend API"
    echo "2. Frontend UI"
    echo "3. PostgreSQL"
    echo "4. Redis"
    echo "5. Neo4j"
    echo "6. Monitoring Stack (Prometheus, Grafana, Loki)"
    echo "7. All Services"
    echo "8. Cancel"
    
    read -p "Select service to deploy: " service_choice
    
    case $service_choice in
        1) docker-compose up -d backend ;;
        2) docker-compose up -d frontend ;;
        3) docker-compose up -d postgres ;;
        4) docker-compose up -d redis ;;
        5) docker-compose up -d neo4j ;;
        6) docker-compose up -d prometheus grafana loki ;;
        7) docker-compose up -d ;;
        8) echo "Deployment cancelled" ;;
        *) echo "Invalid option" ;;
    esac
}

# Main menu
show_menu() {
    while true; do
        show_header
        echo -e "\n${CYAN}Main Menu:${NC}\n"
        echo "1.  System Overview"
        echo "2.  Live Logs (All Services)"
        echo "3.  Test API Endpoints"
        echo "4.  Container Statistics"
        echo "5.  Log Management"
        echo "6.  Debug Controls"
        echo "7.  Database Repair"
        echo "8.  System Repair"
        echo "9.  Restart All Services"
        echo "10. Unified Live Logs"
        echo "11. Docker Troubleshooting"
        echo "12. Redeploy All Containers"
        echo "13. Smart Health Check"
        echo "14. Container Health Status"
        echo "15. Selective Service Deployment"
        echo "0.  Exit"
        
        read -p $'\nSelect option: ' choice
        
        case $choice in
            1) show_system_overview; read -p "Press Enter to continue..." ;;
            2) show_live_logs ;;
            3) test_api_endpoints; read -p "Press Enter to continue..." ;;
            4) show_container_stats ;;
            5) manage_logs; read -p "Press Enter to continue..." ;;
            6) debug_controls; read -p "Press Enter to continue..." ;;
            7) repair_databases; read -p "Press Enter to continue..." ;;
            8) repair_system; read -p "Press Enter to continue..." ;;
            9) restart_all_services; read -p "Press Enter to continue..." ;;
            10) show_unified_live_logs ;;
            11) docker_troubleshooting; read -p "Press Enter to continue..." ;;
            12) redeploy_containers; read -p "Press Enter to continue..." ;;
            13) smart_health_check; read -p "Press Enter to continue..." ;;
            14) container_health_status; read -p "Press Enter to continue..." ;;
            15) selective_deployment; read -p "Press Enter to continue..." ;;
            0) echo "Exiting..."; exit 0 ;;
            *) echo "Invalid option. Please try again."; sleep 2 ;;
        esac
    done
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being executed directly
    show_menu
fi