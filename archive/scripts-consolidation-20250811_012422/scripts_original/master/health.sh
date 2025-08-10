#!/bin/bash
# Master Health Check Script for SutazAI System
# Version: 2.0 - Ultra-Consolidated
# Created: August 10, 2025
# Purpose: Comprehensive health monitoring for all services

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/health_check_$(date +%Y%m%d_%H%M%S).log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Service definitions with health endpoints
declare -A SERVICES=(
    ["PostgreSQL"]="10000:/health"
    ["Redis"]="10001:/health"
    ["Neo4j"]="10002:/"
    ["Backend API"]="10010:/health"
    ["Frontend"]="10011:/"
    ["Ollama"]="10104:/api/tags"
    ["Qdrant"]="10101:/"
    ["ChromaDB"]="10100:/api/v1"
    ["FAISS"]="10103:/health"
    ["RabbitMQ"]="10008:/"
    ["Prometheus"]="10200:/-/healthy"
    ["Grafana"]="10201:/api/health"
    ["Loki"]="10202:/ready"
    ["Hardware Optimizer"]="11110:/health"
    ["AI Orchestrator"]="8589:/health"
    ["Ollama Integration"]="8090:/health"
)

# Counters
TOTAL=0
HEALTHY=0
UNHEALTHY=0
UNKNOWN=0

# Logging
mkdir -p "$(dirname "$LOG_FILE")"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

# Functions
log_header() {
    echo "=========================================="
    echo "SutazAI System Health Check Report"
    echo "Date: $(date)"
    echo "=========================================="
    echo ""
}

check_docker() {
    echo -e "${BLUE}[CHECK]${NC} Docker Status"
    if systemctl is-active docker &>/dev/null; then
        echo -e "${GREEN}✓${NC} Docker daemon is running"
    else
        echo -e "${RED}✗${NC} Docker daemon is not running"
        exit 1
    fi
    echo ""
}

check_containers() {
    echo -e "${BLUE}[CHECK]${NC} Container Status"
    local running=$(docker ps -q | wc -l)
    local total=$(docker ps -aq | wc -l)
    echo "Running containers: $running/$total"
    
    # List unhealthy containers
    local unhealthy=$(docker ps --filter health=unhealthy --format "table {{.Names}}\t{{.Status}}" | tail -n +2)
    if [ -n "$unhealthy" ]; then
        echo -e "${YELLOW}[WARNING]${NC} Unhealthy containers:"
        echo "$unhealthy"
    fi
    echo ""
}

check_service() {
    local name=$1
    local port_endpoint=$2
    local port="${port_endpoint%%:*}"
    local endpoint="${port_endpoint#*:}"
    
    ((TOTAL++))
    
    printf "%-25s " "$name:"
    
    # Check if port is listening
    if ! nc -z localhost "$port" 2>/dev/null; then
        echo -e "${RED}✗ Port $port not accessible${NC}"
        ((UNHEALTHY++))
        return
    fi
    
    # Try health endpoint
    if curl -sf "http://localhost:${port}${endpoint}" -o /dev/null --max-time 5 2>/dev/null; then
        echo -e "${GREEN}✓ Healthy${NC}"
        ((HEALTHY++))
    else
        # Try alternate check for some services
        if curl -sf "http://localhost:${port}/" -o /dev/null --max-time 5 2>/dev/null; then
            echo -e "${YELLOW}⚠ Responding (no health endpoint)${NC}"
            ((HEALTHY++))
        else
            echo -e "${RED}✗ Not responding${NC}"
            ((UNHEALTHY++))
        fi
    fi
}

check_all_services() {
    echo -e "${BLUE}[CHECK]${NC} Service Health Status"
    echo "----------------------------------------"
    
    for service in "${!SERVICES[@]}"; do
        check_service "$service" "${SERVICES[$service]}"
    done
    
    echo "----------------------------------------"
    echo ""
}

check_resources() {
    echo -e "${BLUE}[CHECK]${NC} System Resources"
    
    # Memory
    local mem_total=$(free -h | awk '/^Mem:/ {print $2}')
    local mem_used=$(free -h | awk '/^Mem:/ {print $3}')
    local mem_percent=$(free | awk '/^Mem:/ {printf "%.1f", $3/$2 * 100}')
    echo "Memory: $mem_used / $mem_total ($mem_percent%)"
    
    # Disk
    local disk_usage=$(df -h /opt/sutazaiapp | awk 'NR==2 {print $3 " / " $2 " (" $5 ")"}')
    echo "Disk: $disk_usage"
    
    # Docker disk usage
    local docker_disk=$(docker system df --format "table {{.Type}}\t{{.Size}}\t{{.Reclaimable}}" | tail -n +2)
    echo -e "\nDocker disk usage:"
    echo "$docker_disk"
    echo ""
}

check_logs() {
    echo -e "${BLUE}[CHECK]${NC} Recent Errors in Logs"
    
    # Check for recent errors in Docker logs
    local errors=$(docker-compose logs --tail=100 2>&1 | grep -i "error\|exception\|fatal" | head -5)
    if [ -n "$errors" ]; then
        echo -e "${YELLOW}Recent errors found:${NC}"
        echo "$errors"
    else
        echo -e "${GREEN}No recent errors in logs${NC}"
    fi
    echo ""
}

generate_summary() {
    echo "=========================================="
    echo "SUMMARY"
    echo "=========================================="
    
    local health_percent=$((HEALTHY * 100 / TOTAL))
    
    echo "Total Services: $TOTAL"
    echo -e "Healthy: ${GREEN}$HEALTHY${NC}"
    echo -e "Unhealthy: ${RED}$UNHEALTHY${NC}"
    echo -e "Unknown: ${YELLOW}$UNKNOWN${NC}"
    echo "Health Score: $health_percent%"
    
    if [ $health_percent -ge 90 ]; then
        echo -e "\n${GREEN}✓ System is healthy and operational${NC}"
    elif [ $health_percent -ge 70 ]; then
        echo -e "\n${YELLOW}⚠ System is partially operational${NC}"
    else
        echo -e "\n${RED}✗ System has critical issues${NC}"
    fi
    
    echo ""
    echo "Full report saved to: $LOG_FILE"
}

# Run health checks based on mode
case "${1:-full}" in
    quick)
        log_header
        check_docker
        check_containers
        generate_summary
        ;;
    services)
        log_header
        check_all_services
        generate_summary
        ;;
    resources)
        log_header
        check_resources
        ;;
    full)
        log_header
        check_docker
        check_containers
        check_all_services
        check_resources
        check_logs
        generate_summary
        ;;
    *)
        echo "Usage: $0 {quick|services|resources|full}"
        echo ""
        echo "Modes:"
        echo "  quick     - Quick container status check"
        echo "  services  - Check all service health endpoints"
        echo "  resources - Check system resource usage"
        echo "  full      - Complete health assessment (default)"
        exit 1
        ;;
esac

# Return appropriate exit code
if [ $UNHEALTHY -gt 0 ]; then
    exit 1
else
    exit 0
fi