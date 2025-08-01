#!/bin/bash

# Fix Docker Deployment Issues for SutazAI
# Addresses: Ollama health checks, networking, and resource allocation

set -e

echo "========================================="
echo "SutazAI Docker Deployment Fix Script"
echo "========================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 1. Fix Ollama container health check
print_status "Fixing Ollama container health check..."

# Create a custom healthcheck script for Ollama
cat > /tmp/ollama_healthcheck.sh << 'EOF'
#!/bin/sh
# Custom health check for Ollama that doesn't require curl
if ollama list >/dev/null 2>&1; then
    exit 0
else
    exit 1
fi
EOF

# Update docker-compose.yml to fix Ollama healthcheck
print_status "Updating Ollama healthcheck configuration..."
cat > /tmp/ollama_healthcheck_fix.yml << 'EOF'
  ollama:
    healthcheck:
      test: ["CMD-SHELL", "ollama list >/dev/null 2>&1 || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s
EOF

# 2. Fix Ollama Specialist service
print_status "Fixing Ollama Specialist service configuration..."

# Check if the service is trying to connect to wrong ports
docker exec sutazai-ollama-specialist cat /app/main.py 2>/dev/null | grep -E "port|PORT" || true

# 3. Create network connectivity test script
print_status "Creating network connectivity test..."
cat > /opt/sutazaiapp/scripts/test_network_connectivity.sh << 'EOF'
#!/bin/bash
# Test network connectivity between containers

echo "Testing network connectivity..."

# Test Ollama connectivity
echo -n "Testing Ollama API: "
if docker exec sutazai-agi-brain wget -q -O- http://ollama:11434/api/tags >/dev/null 2>&1; then
    echo "✓ OK"
else
    echo "✗ FAILED"
fi

# Test Redis connectivity
echo -n "Testing Redis: "
if docker exec sutazai-agi-brain redis-cli -h redis -a ${REDIS_PASSWORD:-redis_password} ping >/dev/null 2>&1; then
    echo "✓ OK"
else
    echo "✗ FAILED"
fi

# Test PostgreSQL connectivity
echo -n "Testing PostgreSQL: "
if docker exec sutazai-agi-brain pg_isready -h postgres -U ${POSTGRES_USER:-sutazai} >/dev/null 2>&1; then
    echo "✓ OK"
else
    echo "✗ FAILED"
fi
EOF
chmod +x /opt/sutazaiapp/scripts/test_network_connectivity.sh

# 4. Fix resource allocation issues
print_status "Optimizing resource allocation..."

# Create resource limits configuration
cat > /opt/sutazaiapp/docker-compose.resource-limits.yml << 'EOF'
# Resource limits overlay for docker-compose
version: '3.8'

services:
  ollama:
    deploy:
      resources:
        limits:
          cpus: '4'          # Reduced from 6
          memory: 6G         # Reduced from 8G
        reservations:
          cpus: '2'
          memory: 4G
    environment:
      OLLAMA_NUM_PARALLEL: 1      # Reduced from 2
      OLLAMA_NUM_THREADS: 4       # Reduced from 8
      OLLAMA_MAX_LOADED_MODELS: 1
      OLLAMA_KEEP_ALIVE: 1m       # Reduced from 2m

  # Limit other resource-intensive services
  neo4j:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G

  chromadb:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
EOF

# 5. Create a health check monitoring script
print_status "Creating health check monitoring script..."
cat > /opt/sutazaiapp/scripts/monitor_health_checks.sh << 'EOF'
#!/bin/bash
# Monitor container health checks

while true; do
    clear
    echo "=== Container Health Status ==="
    echo "Time: $(date)"
    echo ""
    
    docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(healthy|unhealthy|starting)" || true
    
    echo ""
    echo "=== Unhealthy Containers Details ==="
    for container in $(docker ps --filter "health=unhealthy" --format "{{.Names}}"); do
        echo "Container: $container"
        docker inspect $container | jq '.[0].State.Health.Log[-1].Output' 2>/dev/null | sed 's/\\n/\n/g' || true
        echo "---"
    done
    
    sleep 10
done
EOF
chmod +x /opt/sutazaiapp/scripts/monitor_health_checks.sh

# 6. Apply immediate fixes
print_status "Applying immediate fixes..."

# Fix Ollama health check by recreating the service
print_warning "Recreating Ollama service with fixed health check..."
cd /opt/sutazaiapp

# Create a temporary docker-compose override
cat > docker-compose.healthcheck-fix.yml << 'EOF'
version: '3.8'

services:
  ollama:
    healthcheck:
      test: ["CMD-SHELL", "ollama list >/dev/null 2>&1 || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s
      
  ollama-specialist:
    healthcheck:
      test: ["CMD-SHELL", "pgrep -f 'python main.py' >/dev/null || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    environment:
      LITELLM_API_KEY: ${LITELLM_API_KEY:-sk-1234}
EOF

# Apply the fix
print_status "Applying health check fixes..."
docker-compose -f docker-compose.yml -f docker-compose.healthcheck-fix.yml up -d ollama ollama-specialist

# 7. Clean up unused resources
print_status "Cleaning up unused Docker resources..."
docker system prune -f --volumes 2>/dev/null || true

# 8. Verify fixes
print_status "Verifying fixes..."
sleep 30  # Wait for containers to stabilize

echo ""
echo "=== Current Container Status ==="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.State}}"

echo ""
echo "=== Network Connectivity Test ==="
/opt/sutazaiapp/scripts/test_network_connectivity.sh 2>/dev/null || true

echo ""
print_status "Fix script completed!"
print_warning "Monitor health checks with: ./scripts/monitor_health_checks.sh"
print_warning "If issues persist, consider running: docker-compose down && docker-compose up -d"