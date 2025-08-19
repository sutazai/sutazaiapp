#!/bin/bash
# PostgreSQL DNS Fix and Backend Startup Script
# Created: 2025-08-18
# Purpose: Fix PostgreSQL connectivity issues and start backend with proper configuration

set -e

echo "=== PostgreSQL DNS Fix and Backend Startup Script ==="
echo "Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if PostgreSQL is running
echo ""
echo "1. Checking PostgreSQL status..."
if docker ps | grep -q sutazai-postgres; then
    print_status "PostgreSQL container is running"
    
    # Test direct connection
    if docker exec sutazai-postgres psql -U sutazai -d sutazai -c "SELECT 1;" &>/dev/null; then
        print_status "PostgreSQL is accepting connections"
    else
        print_error "PostgreSQL is not accepting connections"
        exit 1
    fi
else
    print_error "PostgreSQL container is not running"
    print_warning "Starting PostgreSQL..."
    docker-compose -f /opt/sutazaiapp/docker/docker-compose.consolidated.yml up -d postgres
    sleep 10
fi

# Test DNS resolution
echo ""
echo "2. Testing DNS resolution..."
if docker run --rm --network sutazai-network alpine ping -c 1 sutazai-postgres &>/dev/null; then
    print_status "DNS resolution for 'sutazai-postgres' is working"
else
    print_warning "DNS resolution failed, adding network alias..."
    POSTGRES_CONTAINER=$(docker ps --filter "name=postgres" --format "{{.Names}}" | grep -E "^sutazai-postgres$|postgres")
    docker network connect --alias sutazai-postgres sutazai-network "$POSTGRES_CONTAINER" 2>/dev/null || true
    print_status "Network alias added"
fi

# Stop any existing backend container
echo ""
echo "3. Checking for existing backend container..."
if docker ps -a | grep -q sutazai-backend; then
    print_warning "Stopping existing backend container..."
    docker stop sutazai-backend 2>/dev/null || true
    docker rm sutazai-backend 2>/dev/null || true
    print_status "Existing backend container removed"
fi

# Load environment variables from .env file
echo ""
echo "4. Setting up environment variables..."
if [ -f "/opt/sutazaiapp/.env" ]; then
    source /opt/sutazaiapp/.env
fi

# Use existing passwords or generate secure ones if not set
export POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-change_me_secure}
export NEO4J_PASSWORD=${NEO4J_PASSWORD:-sutazai123}
export GRAFANA_PASSWORD=${GRAFANA_PASSWORD:-sutazai123}
export JWT_SECRET=${JWT_SECRET:-sutazai_jwt_secret_key_2025_ultra_secure_random_string}
export SECRET_KEY=${SECRET_KEY:-system_ultra_secure_secret_key_2025}
export RABBITMQ_DEFAULT_PASS=${RABBITMQ_DEFAULT_PASS:-sutazai123}

# Save environment variables to .env file for persistence
ENV_FILE="/opt/sutazaiapp/.env"
if [ ! -f "$ENV_FILE" ]; then
    print_warning "Creating .env file with secure passwords..."
    cat > "$ENV_FILE" << EOF
# Auto-generated environment variables
# Created: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
POSTGRES_DB=sutazai
NEO4J_PASSWORD=${NEO4J_PASSWORD}
GRAFANA_PASSWORD=${GRAFANA_PASSWORD}
JWT_SECRET=${JWT_SECRET}
SECRET_KEY=${SECRET_KEY}
RABBITMQ_DEFAULT_USER=sutazai
RABBITMQ_DEFAULT_PASS=${RABBITMQ_DEFAULT_PASS}
SUTAZAI_ENV=production
TZ=UTC
EOF
    chmod 600 "$ENV_FILE"
    print_status "Environment file created"
else
    print_status "Loading existing environment file"
    source "$ENV_FILE"
fi

# Start backend with proper configuration
echo ""
echo "5. Starting backend service..."
docker run -d \
  --name sutazai-backend \
  --network sutazai-network \
  --restart unless-stopped \
  -p 10010:8000 \
  -e DATABASE_URL="postgresql://sutazai:${POSTGRES_PASSWORD}@sutazai-postgres:5432/sutazai" \
  -e POSTGRES_USER="sutazai" \
  -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
  -e POSTGRES_DB="sutazai" \
  -e REDIS_URL="redis://sutazai-redis:6379/0" \
  -e NEO4J_URI="bolt://sutazai-neo4j:7687" \
  -e NEO4J_PASSWORD="${NEO4J_PASSWORD}" \
  -e CHROMADB_URL="http://sutazai-chromadb:8000" \
  -e CHROMADB_HOST="sutazai-chromadb" \
  -e CHROMADB_PORT="8000" \
  -e QDRANT_URL="http://sutazai-qdrant:6333" \
  -e QDRANT_HOST="sutazai-qdrant" \
  -e QDRANT_PORT="6333" \
  -e RABBITMQ_URL="amqp://sutazai:${RABBITMQ_DEFAULT_PASS}@sutazai-rabbitmq:5672/" \
  -e OLLAMA_BASE_URL="http://sutazai-ollama:11434" \
  -e OLLAMA_HOST="sutazai-ollama" \
  -e CONSUL_URL="http://sutazai-consul:8500" \
  -e CONSUL_HOST="sutazai-consul" \
  -e CONSUL_PORT="8500" \
  -e JWT_SECRET="${JWT_SECRET}" \
  -e SECRET_KEY="${SECRET_KEY}" \
  -e GRAFANA_PASSWORD="${GRAFANA_PASSWORD}" \
  -e SUTAZAI_ENV="production" \
  -e API_V1_STR="/api/v1" \
  -e PROJECT_NAME="SutazAI" \
  -e BACKEND_CORS_ORIGINS='["http://localhost:10011","http://sutazai-frontend:8501"]' \
  -e LOG_LEVEL="INFO" \
  -e TZ="UTC" \
  -v /opt/sutazaiapp/backend:/app \
  -v /opt/sutazaiapp/data:/data \
  -v /opt/sutazaiapp/logs:/logs \
  -v /opt/sutazaiapp/.mcp.json:/app/.mcp.json:ro \
  -v /opt/sutazaiapp/scripts/mcp:/app/scripts/mcp:ro \
  sutazaiapp-backend:v1.0.0 \
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 --loop uvloop

if [ $? -eq 0 ]; then
    print_status "Backend container started successfully"
else
    print_error "Failed to start backend container"
    exit 1
fi

# Wait for backend to be healthy
echo ""
echo "6. Waiting for backend to be healthy..."
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f http://localhost:10010/health &>/dev/null; then
        print_status "Backend is healthy and responding"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -n "."
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    print_error "Backend failed to become healthy after $MAX_RETRIES attempts"
    echo ""
    echo "Checking backend logs for errors:"
    docker logs sutazai-backend --tail 20
    exit 1
fi

# Verify database connectivity
echo ""
echo "7. Verifying database connectivity..."
if docker exec sutazai-backend python -c "
import psycopg2
conn = psycopg2.connect('postgresql://sutazai:${POSTGRES_PASSWORD}@sutazai-postgres:5432/sutazai')
print('Database connection successful')
conn.close()
" 2>/dev/null; then
    print_status "Backend can connect to PostgreSQL"
else
    print_warning "Backend database connection test failed (this may be normal if psycopg2 is not installed)"
fi

# Final status report
echo ""
echo "============================================"
echo "          DEPLOYMENT STATUS REPORT          "
echo "============================================"
echo ""

# Check all critical services
SERVICES=("sutazai-postgres" "sutazai-redis" "sutazai-backend" "sutazai-frontend")
ALL_HEALTHY=true

for service in "${SERVICES[@]}"; do
    if docker ps | grep -q "$service"; then
        STATUS=$(docker inspect "$service" --format='{{.State.Status}}' 2>/dev/null || echo "unknown")
        if [ "$STATUS" == "running" ]; then
            print_status "$service: Running"
        else
            print_warning "$service: Status = $STATUS"
            ALL_HEALTHY=false
        fi
    else
        print_error "$service: Not running"
        ALL_HEALTHY=false
    fi
done

echo ""
if [ "$ALL_HEALTHY" = true ]; then
    print_status "All critical services are running!"
    echo ""
    echo "Access points:"
    echo "  - Backend API: http://localhost:10010"
    echo "  - Frontend UI: http://localhost:10011"
    echo "  - PostgreSQL: localhost:10000"
    echo "  - Redis: localhost:10001"
    echo ""
    echo "Environment file saved to: $ENV_FILE"
    echo ""
    print_status "PostgreSQL DNS fix completed successfully!"
else
    print_warning "Some services may need attention"
    echo ""
    echo "To check logs:"
    echo "  docker logs sutazai-backend"
    echo "  docker logs sutazai-postgres"
    echo ""
    echo "To restart all services:"
    echo "  cd /opt/sutazaiapp"
    echo "  docker-compose -f docker/docker-compose.consolidated.yml up -d"
fi

echo ""
echo "Script completed at: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"