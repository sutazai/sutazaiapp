#!/bin/bash
# Service Mesh Deployment Script
# Deploys and configures the production-grade service mesh

set -e

echo "=================================================="
echo "SutazAI Service Mesh Deployment v2.0.0"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running from project root
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}Error: Must run from project root directory${NC}"
    exit 1
fi

# Function to check service health
check_service() {
    local service=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    echo -n "Checking $service..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e " ${RED}✗${NC}"
    return 1
}

# Step 1: Verify dependencies
echo "1. Verifying dependencies..."
echo -n "  - Docker: "
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓${NC} $(docker --version)"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

echo -n "  - Docker Compose: "
if command -v docker-compose &> /dev/null; then
    echo -e "${GREEN}✓${NC} $(docker-compose --version)"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

# Step 2: Start core infrastructure
echo ""
echo "2. Starting core infrastructure..."

# Start Redis (needed for backward compatibility)
echo "  - Starting Redis..."
docker-compose up -d redis
check_service "Redis" "http://localhost:10001"

# Start PostgreSQL
echo "  - Starting PostgreSQL..."
docker-compose up -d postgres
sleep 5  # Give PostgreSQL time to initialize

# Step 3: Start service mesh components
echo ""
echo "3. Starting service mesh components..."

# Start Consul for service discovery
echo "  - Starting Consul..."
docker-compose up -d consul
check_service "Consul" "http://localhost:8500/v1/status/leader"

# Start Kong API Gateway
echo "  - Starting Kong..."
docker-compose up -d kong
check_service "Kong Admin API" "http://localhost:8001"
check_service "Kong Proxy" "http://localhost:8000"

# Step 4: Start backend with mesh support
echo ""
echo "4. Starting backend API with service mesh..."
docker-compose up -d backend-api
check_service "Backend API" "http://localhost:10010/health"

# Step 5: Register backend service with mesh
echo ""
echo "5. Registering backend service with mesh..."

curl -X POST http://localhost:10010/api/v1/mesh/v2/register \
    -H "Content-Type: application/json" \
    -d '{
        "service_name": "backend-api",
        "address": "backend-api",
        "port": 8000,
        "tags": ["api", "v1", "backend"],
        "metadata": {
            "version": "2.0.0",
            "environment": "production"
        }
    }' > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "  - Backend service registered ${GREEN}✓${NC}"
else
    echo -e "  - Backend service registration ${YELLOW}failed (may already be registered)${NC}"
fi

# Step 6: Configure Kong routes
echo ""
echo "6. Configuring Kong API Gateway routes..."

# Create upstream for backend
curl -X PUT http://localhost:8001/upstreams/backend-upstream \
    -d "algorithm=round-robin" \
    -d "healthchecks.active.type=http" \
    -d "healthchecks.active.http_path=/health" \
    -d "healthchecks.active.healthy.interval=5" \
    -d "healthchecks.active.healthy.successes=1" \
    -d "healthchecks.active.unhealthy.interval=10" \
    -d "healthchecks.active.unhealthy.http_failures=3" > /dev/null 2>&1

# Add backend target
curl -X POST http://localhost:8001/upstreams/backend-upstream/targets \
    -d "target=backend-api:8000" \
    -d "weight=100" > /dev/null 2>&1

# Create service
curl -X PUT http://localhost:8001/services/backend-service \
    -d "name=backend-service" \
    -d "host=backend-upstream" > /dev/null 2>&1

# Create route
curl -X PUT http://localhost:8001/services/backend-service/routes/backend-route \
    -d "paths[]=/api" \
    -d "strip_path=false" > /dev/null 2>&1

echo -e "  - Kong routes configured ${GREEN}✓${NC}"

# Step 7: Start monitoring stack
echo ""
echo "7. Starting monitoring stack..."

# Start Prometheus
echo "  - Starting Prometheus..."
docker-compose up -d prometheus
check_service "Prometheus" "http://localhost:10200"

# Start Grafana
echo "  - Starting Grafana..."
docker-compose up -d grafana
check_service "Grafana" "http://localhost:10201"

# Step 8: Verify service mesh
echo ""
echo "8. Verifying service mesh..."

# Check mesh health
MESH_HEALTH=$(curl -s http://localhost:10010/api/v1/mesh/v2/health | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))")

if [ "$MESH_HEALTH" = "healthy" ]; then
    echo -e "  - Service mesh status: ${GREEN}healthy${NC}"
else
    echo -e "  - Service mesh status: ${YELLOW}$MESH_HEALTH${NC}"
fi

# Get topology
TOPOLOGY=$(curl -s http://localhost:10010/api/v1/mesh/v2/topology | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"Services: {d.get('total_services', 0)}, Instances: {d.get('total_instances', 0)}\")")
echo "  - Mesh topology: $TOPOLOGY"

# Step 9: Display access information
echo ""
echo "=================================================="
echo -e "${GREEN}Service Mesh Deployment Complete!${NC}"
echo "=================================================="
echo ""
echo "Access Points:"
echo "  - Backend API: http://localhost:10010"
echo "  - Kong Gateway: http://localhost:8000"
echo "  - Kong Admin: http://localhost:8001"
echo "  - Consul UI: http://localhost:8500"
echo "  - Prometheus: http://localhost:10200"
echo "  - Grafana: http://localhost:10201 (admin/admin)"
echo ""
echo "API Endpoints:"
echo "  - Service Registration: POST /api/v1/mesh/v2/register"
echo "  - Service Discovery: GET /api/v1/mesh/v2/discover/{service}"
echo "  - Service Call: POST /api/v1/mesh/v2/call"
echo "  - Mesh Topology: GET /api/v1/mesh/v2/topology"
echo "  - Mesh Health: GET /api/v1/mesh/v2/health"
echo "  - Mesh Metrics: GET /api/v1/mesh/v2/metrics"
echo ""
echo "Next Steps:"
echo "  1. Register additional services with the mesh"
echo "  2. Configure load balancing strategies"
echo "  3. Set up circuit breaker thresholds"
echo "  4. Create Grafana dashboards for monitoring"
echo "  5. Test service-to-service communication"
echo ""
echo -e "${YELLOW}Note: Legacy Redis-based mesh is still available at /api/v1/mesh/* for backward compatibility${NC}"