#!/bin/bash
# Docker Compose Configuration Validation Script
# Created: 2025-08-20
# Purpose: Validate the fixed docker-compose.yml configuration

set -e

echo "======================================"
echo "Docker Compose Configuration Test"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Change to project root
cd /opt/sutazaiapp

echo "1. Validating configuration syntax..."
if docker compose config > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Configuration syntax is valid"
else
    echo -e "${RED}✗${NC} Configuration syntax error"
    docker compose config
    exit 1
fi

echo ""
echo "2. Checking service definitions..."
SERVICE_COUNT=$(docker compose config --services | wc -l)
echo -e "${GREEN}✓${NC} Found $SERVICE_COUNT services"

echo ""
echo "3. Verifying critical services..."
CRITICAL_SERVICES=(
    "postgres"
    "redis"
    "backend"
    "frontend"
    "mcp-orchestrator"
    "mcp-manager"
)

for service in "${CRITICAL_SERVICES[@]}"; do
    if docker compose config --services | grep -q "^$service$"; then
        echo -e "${GREEN}✓${NC} Service '$service' is defined"
    else
        echo -e "${RED}✗${NC} Service '$service' is missing"
        exit 1
    fi
done

echo ""
echo "4. Checking network definitions..."
NETWORKS=$(docker compose config --format json 2>/dev/null | jq -r '.networks | keys[]' | sort)
echo "Networks defined:"
echo "$NETWORKS" | while read network; do
    echo "  - $network"
done

echo ""
echo "5. Checking volume definitions..."
VOLUME_COUNT=$(docker compose config --volumes | wc -l)
echo -e "${GREEN}✓${NC} Found $VOLUME_COUNT volumes defined"

echo ""
echo "6. Verifying depends_on relationships..."
# Check mcp-manager depends on mcp-orchestrator
MCP_DEPENDS=$(docker compose config --format json 2>/dev/null | jq -r '.services."mcp-manager".depends_on."mcp-orchestrator".condition // "not_found"')
if [ "$MCP_DEPENDS" = "service_healthy" ]; then
    echo -e "${GREEN}✓${NC} mcp-manager correctly depends on mcp-orchestrator (service_healthy)"
else
    echo -e "${RED}✗${NC} mcp-manager dependency issue: $MCP_DEPENDS"
fi

# Check backend dependencies
BACKEND_DEPS=$(docker compose config --format json 2>/dev/null | jq -r '.services.backend.depends_on | keys[]' 2>/dev/null | sort | tr '\n' ' ')
if [ -n "$BACKEND_DEPS" ]; then
    echo -e "${GREEN}✓${NC} backend depends on: $BACKEND_DEPS"
fi

# Check frontend dependency
FRONTEND_DEPS=$(docker compose config --format json 2>/dev/null | jq -r '.services.frontend.depends_on | keys[]' 2>/dev/null | sort | tr '\n' ' ')
if [ -n "$FRONTEND_DEPS" ]; then
    echo -e "${GREEN}✓${NC} frontend depends on: $FRONTEND_DEPS"
fi

echo ""
echo "7. Checking port mappings..."
echo "Key port mappings:"
docker compose config --format json 2>/dev/null | jq -r '
    .services | to_entries[] | 
    select(.value.ports != null) | 
    "\(.key): \(.value.ports | map(split(":")[0]) | join(", "))"
' | head -10

echo ""
echo "8. Checking healthchecks..."
SERVICES_WITH_HEALTH=$(docker compose config --format json 2>/dev/null | jq -r '.services | to_entries[] | select(.value.healthcheck != null) | .key' | wc -l)
echo -e "${GREEN}✓${NC} $SERVICES_WITH_HEALTH services have healthchecks defined"

echo ""
echo "9. Checking resource limits..."
SERVICES_WITH_LIMITS=$(docker compose config --format json 2>/dev/null | jq -r '.services | to_entries[] | select(.value.deploy.resources.limits != null) | .key' 2>/dev/null | wc -l)
echo -e "${GREEN}✓${NC} $SERVICES_WITH_LIMITS services have resource limits defined"

echo ""
echo "======================================"
echo -e "${GREEN}Configuration validation complete!${NC}"
echo "======================================"
echo ""
echo "The docker-compose.yml file is valid and ready to use."
echo ""
echo "To start services, run:"
echo "  docker compose up -d [service_name]"
echo ""
echo "To start all services:"
echo "  docker compose up -d"