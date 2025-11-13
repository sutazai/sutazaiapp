#!/bin/bash
# SutazaiApp Health Check Script
# Version: 1.0.0
# Created: 2025-11-13 21:30:00 UTC
# Purpose: Comprehensive health check for all SutazaiApp services

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
HEALTHY=0
UNHEALTHY=0
WARNING=0

echo -e "${BLUE}"
echo "═══════════════════════════════════════════════════════════"
echo "        SutazaiApp System Health Check"
echo "        $(date +'%Y-%m-%d %H:%M:%S UTC')"
echo "═══════════════════════════════════════════════════════════"
echo -e "${NC}"

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}✗ Docker daemon is not running${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker daemon is running${NC}"
echo ""

# Function to check service
check_service() {
    local name=$1
    local port=$2
    local health_url=$3
    
    # Check if container is running
    if docker ps --filter "name=$name" --filter "status=running" | grep -q "$name"; then
        # Check health status
        health=$(docker inspect --format='{{.State.Health.Status}}' "$name" 2>/dev/null || echo "none")
        
        if [ "$health" = "healthy" ]; then
            echo -e "${GREEN}✓${NC} $name - Running (Healthy)"
            ((HEALTHY++))
        elif [ "$health" = "none" ]; then
            # No health check defined, try HTTP check if URL provided
            if [ -n "$health_url" ]; then
                if curl -sf "$health_url" > /dev/null 2>&1; then
                    echo -e "${GREEN}✓${NC} $name - Running (HTTP OK)"
                    ((HEALTHY++))
                else
                    echo -e "${YELLOW}⚠${NC} $name - Running (HTTP Failed)"
                    ((WARNING++))
                fi
            else
                echo -e "${GREEN}✓${NC} $name - Running (No Health Check)"
                ((HEALTHY++))
            fi
        else
            echo -e "${YELLOW}⚠${NC} $name - Running ($health)"
            ((WARNING++))
        fi
    else
        echo -e "${RED}✗${NC} $name - Not Running"
        ((UNHEALTHY++))
    fi
}

echo "Management:"
check_service "sutazai-portainer" "9000" "http://localhost:9000/api/status"
echo ""

echo "Core Infrastructure:"
check_service "sutazai-postgres" "10000"
check_service "sutazai-redis" "10001"
check_service "sutazai-neo4j" "10002" "http://localhost:10002"
check_service "sutazai-rabbitmq" "10004"
check_service "sutazai-consul" "10006" "http://localhost:10006/v1/status/leader"
check_service "sutazai-kong" "10008"
echo ""

echo "Vector Databases:"
check_service "sutazai-chromadb" "10100" "http://localhost:10100/api/v1/heartbeat"
check_service "sutazai-qdrant" "10101" "http://localhost:10102/collections"
check_service "sutazai-faiss" "10103" "http://localhost:10103/health"
echo ""

echo "AI Services:"
check_service "sutazai-ollama" "11434" "http://localhost:11434/api/tags"
echo ""

echo "Application:"
check_service "sutazai-backend" "10200" "http://localhost:10200/health"
check_service "sutazai-frontend" "11000" "http://localhost:11000/_stcore/health"
echo ""

echo "Monitoring:"
check_service "sutazai-prometheus" "10202" "http://localhost:10202/-/healthy"
check_service "sutazai-grafana" "10201" "http://localhost:10201/api/health"
echo ""

# Summary
echo "═══════════════════════════════════════════════════════════"
echo -e "${GREEN}Healthy Services: $HEALTHY${NC}"
echo -e "${YELLOW}Warning Services: $WARNING${NC}"
echo -e "${RED}Unhealthy Services: $UNHEALTHY${NC}"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Container resource usage
echo "Resource Usage:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" \
    $(docker ps --filter "name=sutazai-" --format "{{.Names}}") 2>/dev/null | head -20
echo ""

# Volume usage
echo "Volume Usage:"
echo "═══════════════════════════════════════════════════════════"
docker volume ls --filter "name=sutazaiapp_" --format "table {{.Name}}\t{{.Driver}}" | head -20
echo ""

# Network status
echo "Network Status:"
echo "═══════════════════════════════════════════════════════════"
docker network inspect sutazaiapp_sutazai-network --format '{{range .Containers}}{{.Name}} - {{.IPv4Address}}{{"\n"}}{{end}}' 2>/dev/null | head -20
echo ""

# Exit code based on health
if [ $UNHEALTHY -gt 0 ]; then
    echo -e "${RED}⚠ WARNING: Some services are not healthy!${NC}"
    exit 1
elif [ $WARNING -gt 0 ]; then
    echo -e "${YELLOW}⚠ CAUTION: Some services have warnings${NC}"
    exit 0
else
    echo -e "${GREEN}✓ All services are healthy!${NC}"
    exit 0
fi
