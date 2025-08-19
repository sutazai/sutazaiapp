#!/bin/bash

# Fix for Option 9 in live_logs.sh - Restart All Services
# This script provides a working alternative to the broken option 9

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

PROJECT_ROOT="/opt/sutazaiapp"

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              FIXED: RESTART ALL SERVICES                    ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if docker-compose.yml exists
if [[ ! -f "$PROJECT_ROOT/docker-compose.yml" ]]; then
    echo -e "${RED}Error: docker-compose.yml not found in $PROJECT_ROOT${NC}"
    echo -e "${YELLOW}Please ensure you're in the correct directory${NC}"
    exit 1
fi

# Check Docker daemon
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}Error: Docker daemon is not running${NC}"
    echo -e "${YELLOW}Please start Docker first${NC}"
    exit 1
fi

echo -e "${CYAN}Step 1: Checking current container status...${NC}"
running_count=$(docker ps --filter "name=sutazai-" --format "{{.Names}}" 2>/dev/null | wc -l || echo "0")
echo "Currently running containers: $running_count"
echo ""

echo -e "${CYAN}Step 2: Validating docker-compose configuration...${NC}"
cd "$PROJECT_ROOT"
if docker compose config >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Configuration is valid${NC}"
else
    echo -e "${RED}✗ Configuration has errors${NC}"
    echo -e "${YELLOW}Attempting to fix common issues...${NC}"
    
    # Fix common issues
    # Remove promtail user issue if it exists
    if grep -q "user: promtail" docker-compose.yml; then
        echo "Fixing promtail user configuration..."
        sed -i.bak 's/user: promtail/# user: promtail # Commented out - causing errors/' docker-compose.yml
    fi
fi
echo ""

echo -e "${CYAN}Step 3: Restarting services in dependency order...${NC}"

# Define service restart order (dependencies first)
declare -a services=(
    "postgres"
    "redis" 
    "neo4j"
    "chromadb"
    "qdrant"
    "ollama"
    "consul"
    "prometheus"
    "grafana"
    "backend"
    "frontend"
)

restart_count=0
failed_count=0

for service in "${services[@]}"; do
    container_name="sutazai-${service}"
    
    # Check if container exists
    if docker ps -a --format "{{.Names}}" | grep -q "^${container_name}$"; then
        echo -n "Restarting ${container_name}... "
        
        if docker restart "${container_name}" >/dev/null 2>&1; then
            echo -e "${GREEN}✓${NC}"
            ((restart_count++))
        else
            echo -e "${RED}✗ Failed${NC}"
            ((failed_count++))
            
            # Try to start if restart failed
            echo -n "  Attempting to start ${container_name}... "
            if docker start "${container_name}" >/dev/null 2>&1; then
                echo -e "${GREEN}✓${NC}"
                ((restart_count++))
                ((failed_count--))
            else
                echo -e "${RED}✗${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}Skipping ${container_name} (not found)${NC}"
    fi
    
    # Small delay between restarts to avoid overwhelming the system
    sleep 1
done

echo ""
echo -e "${CYAN}Step 4: Verifying service health...${NC}"
sleep 5

# Check health status
healthy_count=0
unhealthy_count=0

for service in "${services[@]}"; do
    container_name="sutazai-${service}"
    
    if docker ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
        health_status=$(docker inspect --format='{{.State.Health.Status}}' "${container_name}" 2>/dev/null || echo "none")
        
        case "$health_status" in
            "healthy")
                echo -e "  ${container_name}: ${GREEN}✓ Healthy${NC}"
                ((healthy_count++))
                ;;
            "unhealthy")
                echo -e "  ${container_name}: ${RED}✗ Unhealthy${NC}"
                ((unhealthy_count++))
                ;;
            "starting")
                echo -e "  ${container_name}: ${YELLOW}⟳ Starting${NC}"
                ;;
            *)
                echo -e "  ${container_name}: ${CYAN}○ Running (no health check)${NC}"
                ;;
        esac
    fi
done

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}                        RESTART SUMMARY                        ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Services restarted: ${GREEN}${restart_count}${NC}"
echo -e "  Failed restarts: ${RED}${failed_count}${NC}"
echo -e "  Healthy services: ${GREEN}${healthy_count}${NC}"
echo -e "  Unhealthy services: ${RED}${unhealthy_count}${NC}"
echo ""

if [[ $failed_count -eq 0 ]]; then
    echo -e "${GREEN}✅ All services restarted successfully!${NC}"
else
    echo -e "${YELLOW}⚠️  Some services failed to restart${NC}"
    echo ""
    echo -e "${CYAN}Troubleshooting tips:${NC}"
    echo "  • Check logs: docker logs sutazai-[service-name]"
    echo "  • Verify .env file exists with required configurations"
    echo "  • Ensure sufficient system resources are available"
    echo "  • Try starting services individually: docker start sutazai-[service-name]"
fi

echo ""
echo -e "${CYAN}Service URLs:${NC}"
echo "  • Frontend: http://localhost:10011"
echo "  • Backend API: http://localhost:10010"
echo "  • API Docs: http://localhost:10010/docs"

# Check if monitoring services are running
if docker ps --format "{{.Names}}" | grep -q "sutazai-grafana"; then
    echo "  • Grafana: http://localhost:10201"
fi
if docker ps --format "{{.Names}}" | grep -q "sutazai-prometheus"; then
    echo "  • Prometheus: http://localhost:10200"
fi

echo ""