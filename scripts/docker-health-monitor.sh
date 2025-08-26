#!/bin/bash
# Docker Container Health Monitoring Script for SutazAI
# Date: 2025-08-26
# Purpose: Monitor container health and add health checks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "SutazAI Docker Container Health Monitor"
echo "=========================================="
echo ""

# Function to check container health
check_container_health() {
    local CONTAINER_NAME=$1
    local PORT=$2
    local SERVICE_TYPE=$3
    
    # Check if container exists
    if ! docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${RED}✗ ${CONTAINER_NAME}${NC} - Container not found"
        return 1
    fi
    
    # Check if container is running
    if ! docker ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${RED}✗ ${CONTAINER_NAME}${NC} - Container stopped"
        return 1
    fi
    
    # Get container status
    STATUS=$(docker inspect --format='{{.State.Status}}' "${CONTAINER_NAME}" 2>/dev/null)
    HEALTH=$(docker inspect --format='{{.State.Health.Status}}' "${CONTAINER_NAME}" 2>/dev/null || echo "none")
    
    # Check port connectivity if provided
    PORT_STATUS="N/A"
    if [ ! -z "$PORT" ]; then
        if nc -z localhost $PORT 2>/dev/null; then
            PORT_STATUS="${GREEN}✓${NC}"
        else
            PORT_STATUS="${RED}✗${NC}"
        fi
    fi
    
    # Determine health indicator
    if [ "$HEALTH" = "healthy" ]; then
        HEALTH_INDICATOR="${GREEN}✓ Healthy${NC}"
    elif [ "$HEALTH" = "unhealthy" ]; then
        HEALTH_INDICATOR="${RED}✗ Unhealthy${NC}"
    elif [ "$HEALTH" = "starting" ]; then
        HEALTH_INDICATOR="${YELLOW}⟳ Starting${NC}"
    elif [ "$HEALTH" = "none" ]; then
        HEALTH_INDICATOR="${YELLOW}⚠ No health check${NC}"
    else
        HEALTH_INDICATOR="${YELLOW}? Unknown${NC}"
    fi
    
    echo -e "${BLUE}${CONTAINER_NAME}${NC}"
    echo -e "  Status: ${STATUS} | Health: ${HEALTH_INDICATOR} | Port ${PORT}: ${PORT_STATUS}"
    
    # Show last 3 log lines if unhealthy
    if [ "$HEALTH" = "unhealthy" ] || [ "$STATUS" != "running" ]; then
        echo -e "  ${YELLOW}Recent logs:${NC}"
        docker logs --tail 3 "${CONTAINER_NAME}" 2>&1 | sed 's/^/    /'
    fi
    
    return 0
}

# Function to add health check to a container
add_health_check() {
    local CONTAINER_NAME=$1
    local CHECK_CMD=$2
    
    echo -e "${YELLOW}Adding health check to ${CONTAINER_NAME}...${NC}"
    
    # Note: Docker doesn't support adding health checks to running containers
    # This would need to be done in the Dockerfile or docker-compose.yml
    echo "  Health checks must be added at container creation time."
    echo "  Recommended: Update docker-compose.yml or Dockerfile"
}

# Main monitoring section
echo -e "${BLUE}=== Critical Services Health ===${NC}"
echo ""

# Check critical services
check_container_health "sutazai-postgres" "10000" "database"
check_container_health "sutazai-redis" "10001" "cache"
check_container_health "sutazai-chromadb" "10100" "vectordb"
check_container_health "sutazai-qdrant" "10101" "vectordb"
check_container_health "sutazai-ollama" "" "llm"

echo ""
echo -e "${BLUE}=== Container Statistics ===${NC}"
echo ""

# Show container resource usage
echo "Container Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" | grep -E "sutazai-|CONTAINER"

echo ""
echo -e "${BLUE}=== Recommendations for Health Checks ===${NC}"
echo ""

# Generate docker-compose health check recommendations
cat << 'EOF'
Add these health checks to your docker-compose.yml:

  postgres:
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 30s

  redis:
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 30s

  chromadb:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 30s

  qdrant:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 30s
EOF

echo ""
echo -e "${BLUE}=== Container Naming Convention ===${NC}"
echo ""

# Show naming convention recommendations
cat << 'EOF'
Recommended container naming convention:
- Database services: sutazai-{service} (e.g., sutazai-postgres)
- MCP servers: sutazai-mcp-{name} (e.g., sutazai-mcp-fetch)
- Backend services: sutazai-backend-{service}
- Frontend services: sutazai-frontend-{service}
- Support services: sutazai-{service} (e.g., sutazai-consul)

To rename a container:
docker rename old_name sutazai-new-name
EOF

echo ""
echo -e "${GREEN}Health monitoring complete!${NC}"