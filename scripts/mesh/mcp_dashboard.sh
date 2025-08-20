#!/bin/bash

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

clear
echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║            MCP INFRASTRUCTURE STATUS DASHBOARD            ║${NC}"
echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Define all MCP services
declare -A MCP_PORTS=(
    ["extended-memory"]="3009"
    ["ultimatecoder"]="3011"
    ["claude-flow"]="3001"
    ["ruv-swarm"]="3002"
    ["files"]="3003"
    ["context7"]="3004"
    ["http-fetch"]="3005"
    ["ddg"]="3006"
    ["ssh"]="3010"
    ["knowledge-graph-mcp"]="3014"
    ["github"]="3016"
    ["language-server"]="3018"
    ["claude-task-runner"]="3019"
)

echo -e "${BLUE}Service Health Status:${NC}"
echo "────────────────────────────────────────────────"

healthy=0
unhealthy=0
total=0

for service in "${!MCP_PORTS[@]}"; do
    port="${MCP_PORTS[$service]}"
    total=$((total + 1))
    
    # Check container
    container="mcp-$service"
    if docker ps --format "{{.Names}}" | grep -q "^$container$"; then
        status=$(docker ps --format "{{.Status}}" --filter "name=^$container$" | head -1)
        
        # Check health endpoint
        if timeout 1 curl -s "http://localhost:$port/health" 2>/dev/null | grep -q "healthy"; then
            echo -e "  ${GREEN}✓${NC} $container:$port - ${GREEN}HEALTHY${NC}"
            healthy=$((healthy + 1))
        elif timeout 1 bash -c "echo > /dev/tcp/localhost/$port" 2>/dev/null; then
            echo -e "  ${YELLOW}⚠${NC} $container:$port - ${YELLOW}RUNNING${NC} ($status)"
            healthy=$((healthy + 1))
        else
            echo -e "  ${RED}✗${NC} $container:$port - ${RED}UNHEALTHY${NC} ($status)"
            unhealthy=$((unhealthy + 1))
        fi
    else
        echo -e "  ${RED}✗${NC} $container:$port - ${RED}NOT FOUND${NC}"
        unhealthy=$((unhealthy + 1))
    fi
done

echo ""
echo -e "${BLUE}Summary:${NC}"
echo "────────────────────────────────────────────────"
echo -e "  Total Services: ${CYAN}$total${NC}"
echo -e "  Healthy: ${GREEN}$healthy${NC}"
echo -e "  Unhealthy: ${RED}$unhealthy${NC}"
echo -e "  Health Rate: ${CYAN}$(( healthy * 100 / total ))%${NC}"

echo ""
echo -e "${BLUE}Docker Container Status:${NC}"
echo "────────────────────────────────────────────────"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "mcp-|NAMES" | head -15

echo ""
echo -e "${BLUE}Quick Actions:${NC}"
echo "────────────────────────────────────────────────"
echo "  • View logs: docker logs mcp-<service> --tail 50"
echo "  • Restart service: docker restart mcp-<service>"
echo "  • Check Consul: http://localhost:10006/ui/"
echo "  • Run fix script: bash /opt/sutazaiapp/scripts/mesh/fix_mcp_infrastructure.sh"

echo ""
echo -e "${CYAN}Press Ctrl+C to exit${NC}"
