#!/bin/bash
# Docker Container Cleanup Script for SutazAI
# Date: 2025-08-26
# Purpose: Clean duplicate containers and restore critical services

set -e

echo "=========================================="
echo "SutazAI Docker Container Cleanup Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Phase 1: Inventory
echo -e "${YELLOW}Phase 1: Current Container Inventory${NC}"
echo "Running containers:"
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" | head -20
echo ""
echo "Total running containers: $(docker ps -q | wc -l)"
echo "Duplicate MCP containers to remove: $(docker ps | grep -E 'mcp/(fetch|sequentialthinking|duckduckgo)' | wc -l)"
echo ""

# Confirmation
read -p "Do you want to proceed with cleanup? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 1
fi

# Phase 2: Stop duplicate MCP containers
echo -e "\n${YELLOW}Phase 2: Stopping duplicate MCP containers${NC}"

# Stop all mcp/fetch containers
echo "Stopping mcp/fetch containers..."
FETCH_CONTAINERS=$(docker ps -q --filter "ancestor=mcp/fetch" 2>/dev/null)
if [ ! -z "$FETCH_CONTAINERS" ]; then
    docker stop $FETCH_CONTAINERS || true
    echo -e "${GREEN}✓ Stopped $(echo $FETCH_CONTAINERS | wc -w) mcp/fetch containers${NC}"
else
    echo "No mcp/fetch containers running"
fi

# Stop all mcp/sequentialthinking containers
echo "Stopping mcp/sequentialthinking containers..."
SEQ_CONTAINERS=$(docker ps -q --filter "ancestor=mcp/sequentialthinking" 2>/dev/null)
if [ ! -z "$SEQ_CONTAINERS" ]; then
    docker stop $SEQ_CONTAINERS || true
    echo -e "${GREEN}✓ Stopped $(echo $SEQ_CONTAINERS | wc -w) mcp/sequentialthinking containers${NC}"
else
    echo "No mcp/sequentialthinking containers running"
fi

# Stop all mcp/duckduckgo containers
echo "Stopping mcp/duckduckgo containers..."
DDG_CONTAINERS=$(docker ps -q --filter "ancestor=mcp/duckduckgo" 2>/dev/null)
if [ ! -z "$DDG_CONTAINERS" ]; then
    docker stop $DDG_CONTAINERS || true
    echo -e "${GREEN}✓ Stopped $(echo $DDG_CONTAINERS | wc -w) mcp/duckduckgo containers${NC}"
else
    echo "No mcp/duckduckgo containers running"
fi

# Phase 3: Remove stopped containers
echo -e "\n${YELLOW}Phase 3: Removing stopped duplicate containers${NC}"

# Remove all stopped mcp containers
echo "Removing stopped MCP containers..."
docker rm $(docker ps -aq --filter "ancestor=mcp/fetch" --filter "status=exited") 2>/dev/null || true
docker rm $(docker ps -aq --filter "ancestor=mcp/sequentialthinking" --filter "status=exited") 2>/dev/null || true
docker rm $(docker ps -aq --filter "ancestor=mcp/duckduckgo" --filter "status=exited") 2>/dev/null || true
echo -e "${GREEN}✓ Removed stopped MCP containers${NC}"

# Phase 4: Check and restart critical services
echo -e "\n${YELLOW}Phase 4: Checking critical services${NC}"

# Function to check and start a service
check_and_start_service() {
    local SERVICE_NAME=$1
    local CONTAINER_NAME=$2
    local PORT=$3
    
    echo -n "Checking $SERVICE_NAME..."
    
    # Check if container exists
    if docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        # Check if running
        if docker ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
            echo -e " ${GREEN}✓ Running${NC}"
        else
            echo -e " ${YELLOW}Starting...${NC}"
            if docker start "${CONTAINER_NAME}" 2>/dev/null; then
                sleep 2
                # Verify it's running
                if docker ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
                    echo -e "  ${GREEN}✓ Started successfully${NC}"
                else
                    echo -e "  ${RED}✗ Failed to start (check logs: docker logs ${CONTAINER_NAME})${NC}"
                fi
            else
                echo -e "  ${RED}✗ Failed to start${NC}"
            fi
        fi
    else
        echo -e " ${RED}✗ Container not found${NC}"
    fi
}

# Check critical services
check_and_start_service "PostgreSQL" "sutazai-postgres" "10000"
check_and_start_service "Redis" "sutazai-redis" "10001"
check_and_start_service "ChromaDB" "sutazai-chromadb" "10100"
check_and_start_service "Qdrant" "sutazai-qdrant" "10101"

# Phase 5: Final inventory
echo -e "\n${YELLOW}Phase 5: Final Container Status${NC}"
echo "Running containers after cleanup:"
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" | grep -v "^NAMES" | head -20
echo ""
echo "Total running containers: $(docker ps -q | wc -l)"

# Phase 6: Recommendations
echo -e "\n${YELLOW}Recommendations:${NC}"
echo "1. Fix MCP wrapper scripts to use --rm flag"
echo "2. Add health checks to all containers"
echo "3. Implement proper container naming convention"
echo "4. Set up monitoring and alerting"
echo ""
echo -e "${GREEN}Cleanup complete!${NC}"
echo ""

# Show problematic wrapper scripts
echo -e "${YELLOW}MCP wrapper scripts that need fixing:${NC}"
ls -la /opt/sutazaiapp/scripts/mcp/wrappers/*.sh 2>/dev/null | tail -10 || echo "Wrapper scripts directory not found"

echo ""
echo "For detailed report, see: /opt/sutazaiapp/scripts/docker-cleanup-report.md"