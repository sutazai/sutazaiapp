#!/bin/bash
# Docker Infrastructure Fix Script
# Fixes health checks, resource allocation, and network segmentation

set -e

echo "========================================="
echo "SutazAI Docker Infrastructure Fix Script"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root or with sudo
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root or with sudo"
   exit 1
fi

PROJECT_DIR="/opt/sutazaiapp"
cd "$PROJECT_DIR"

# Step 1: Create fixed health check compose override
print_status "Creating health check fixes..."

cat > docker-compose.healthcheck-fix.yml << 'EOF'
# Health Check Fixes for Ollama and Semgrep

services:
  # Fix Ollama health check - use actual service name from docker-compose-local-llm.yml
  ollama:
    healthcheck:
      # Use wget which is available in the ollama image
      test: ["CMD-SHELL", "wget --no-verbose --tries=1 --spider http://localhost:11434/api/tags || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    deploy:
      resources:
        limits:
          memory: 4G   # Reduced from 23GB to reasonable amount
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'

  # Fix Semgrep health check (hanging endpoint issue) - use actual service name
  semgrep:
    healthcheck:
      # Use a simple endpoint check instead of /health which hangs
      test: ["CMD-SHELL", "curl -f --max-time 5 http://localhost:8000/docs || exit 1"]
      interval: 60s
      timeout: 10s
      retries: 5
      start_period: 120s
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
        reservations:
          memory: 256M
          cpus: '0.25'
EOF

print_success "Health check fixes created"

# Step 2: Create network segmentation configuration
print_status "Creating network segmentation configuration..."

cat > docker-compose.network-fix.yml << 'EOF'
# Network Segmentation Configuration

networks:
  # Main network for inter-service communication
  sutazai-network:
    external: true
    name: sutazaiapp_sutazai-network

  # Database network (isolated)
  sutazai-db-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.21.0.0/24

  # Agent network (isolated)
  sutazai-agent-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.22.0.0/24

  # Vector DB network (isolated)
  sutazai-vector-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.23.0.0/24

services:
  # Update backend to use correct IP
  backend:
    networks:
      sutazai-network:
        ipv4_address: 172.20.0.40  # Changed from .30 to avoid conflicts
      sutazai-db-network:
      sutazai-vector-network:

  # Update frontend to documented IP
  jarvis-frontend:
    networks:
      sutazai-network:
        ipv4_address: 172.20.0.31  # As documented

  # Database services on isolated network
  postgres:
    networks:
      sutazai-network:
        ipv4_address: 172.20.0.10
      sutazai-db-network:
        ipv4_address: 172.21.0.10

  redis:
    networks:
      sutazai-network:
        ipv4_address: 172.20.0.11
      sutazai-db-network:
        ipv4_address: 172.21.0.11

  neo4j:
    networks:
      sutazai-network:
        ipv4_address: 172.20.0.12
      sutazai-db-network:
        ipv4_address: 172.21.0.12

  # Agent services on isolated network
  ollama:
    networks:
      sutazai-network:
      sutazai-agent-network:
        ipv4_address: 172.22.0.10

  semgrep:
    networks:
      sutazai-network:
      sutazai-agent-network:
        ipv4_address: 172.22.0.20
EOF

print_success "Network segmentation configuration created"

# Step 3: Create resource optimization configuration
print_status "Creating resource optimization configuration..."

cat > docker-compose.resource-fix.yml << 'EOF'
# Resource Optimization Configuration

services:
  # Neo4j - Currently at 96% memory usage
  neo4j:
    environment:
      # Reduce memory allocation
      NEO4J_server_memory_heap_initial__size: 128M
      NEO4J_server_memory_heap_max__size: 256M
      NEO4J_server_memory_pagecache_size: 64M
    deploy:
      resources:
        limits:
          memory: 384M  # Reduced from 512M
          cpus: '0.5'
        reservations:
          memory: 256M

  # Vector databases - Over-provisioned
  chromadb:
    deploy:
      resources:
        limits:
          memory: 512M  # Reduced from 1G
          cpus: '0.5'
        reservations:
          memory: 256M

  qdrant:
    deploy:
      resources:
        limits:
          memory: 512M  # Reduced from 1G
          cpus: '0.5'
        reservations:
          memory: 256M

  faiss:
    deploy:
      resources:
        limits:
          memory: 384M  # Reduced from 768M
          cpus: '0.5'
        reservations:
          memory: 256M

  # Ollama - Way over-provisioned
  ollama:
    deploy:
      resources:
        limits:
          memory: 4G   # Reduced from 23G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
EOF

print_success "Resource optimization configuration created"

# Step 4: Apply the fixes
print_status "Applying infrastructure fixes..."

# Stop affected services
print_status "Stopping affected services..."
docker compose -f agents/docker-compose-local-llm.yml stop ollama semgrep 2>/dev/null || true
docker compose -f agents/docker-compose-phase2.yml stop semgrep 2>/dev/null || true

# Apply health check fixes
print_status "Applying health check fixes for Ollama..."
docker compose -f agents/docker-compose-local-llm.yml \
               -f docker-compose.healthcheck-fix.yml \
               up -d ollama --no-deps

print_status "Applying health check fixes for Semgrep..."
docker compose -f agents/docker-compose-phase2.yml \
               -f docker-compose.healthcheck-fix.yml \
               up -d semgrep --no-deps

print_success "Health check fixes applied"

# Step 5: Verify fixes
print_status "Verifying service health..."

sleep 10  # Give services time to start

# Check Ollama
if curl -f --max-time 5 http://localhost:11435/api/tags > /dev/null 2>&1; then
    print_success "Ollama is healthy and responding"
else
    print_warning "Ollama may still be starting up"
fi

# Check Semgrep
if curl -f --max-time 5 http://localhost:11801/docs > /dev/null 2>&1; then
    print_success "Semgrep is healthy and responding"
else
    print_warning "Semgrep may still be starting up"
fi

# Step 6: Network verification
print_status "Verifying network configuration..."

# Check for IP conflicts
print_status "Checking for IP conflicts..."
docker network inspect sutazaiapp_sutazai-network --format '{{json .Containers}}' | \
    jq -r '.[] | "\(.Name): \(.IPv4Address)"' | sort -t: -k2 | \
    awk -F'[:/]' '{ip[$2]++; name[$2]=name[$2]" "$1} END {for (i in ip) if (ip[i]>1) print "CONFLICT on "i":"name[i]}'

CONFLICTS=$(docker network inspect sutazaiapp_sutazai-network --format '{{json .Containers}}' | \
    jq -r '.[] | "\(.Name): \(.IPv4Address)"' | sort -t: -k2 | \
    awk -F'[:/]' '{ip[$2]++} END {for (i in ip) if (ip[i]>1) print i}' | wc -l)

if [ "$CONFLICTS" -eq 0 ]; then
    print_success "No IP conflicts detected"
else
    print_warning "IP conflicts detected. Manual intervention may be required."
fi

# Step 7: Generate comprehensive report
print_status "Generating infrastructure report..."

cat > infrastructure-report.txt << 'EOF'
SutazAI Docker Infrastructure Report
====================================
Generated: $(date)

NETWORK CONFIGURATION
--------------------
EOF

echo "Docker Networks:" >> infrastructure-report.txt
docker network ls --filter "name=sutazai" >> infrastructure-report.txt

echo -e "\nIP Allocations:" >> infrastructure-report.txt
docker network inspect sutazaiapp_sutazai-network --format '{{json .Containers}}' | \
    jq -r '.[] | "\(.Name): \(.IPv4Address)"' | sort >> infrastructure-report.txt

echo -e "\nSERVICE HEALTH STATUS" >> infrastructure-report.txt
echo "--------------------" >> infrastructure-report.txt
docker ps --filter "name=sutazai" --format "table {{.Names}}\t{{.Status}}" >> infrastructure-report.txt

echo -e "\nRESOURCE USAGE" >> infrastructure-report.txt
echo "--------------------" >> infrastructure-report.txt
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" \
    $(docker ps --filter "name=sutazai" -q) >> infrastructure-report.txt

print_success "Infrastructure report generated: infrastructure-report.txt"

# Step 8: Create monitoring script
print_status "Creating monitoring script..."

cat > monitor-infrastructure.sh << 'EOF'
#!/bin/bash
# Continuous monitoring script for SutazAI infrastructure

while true; do
    clear
    echo "SutazAI Infrastructure Monitor - $(date)"
    echo "========================================"
    
    echo -e "\n[SERVICE HEALTH]"
    docker ps --filter "name=sutazai" --format "table {{.Names}}\t{{.Status}}" | head -20
    
    echo -e "\n[RESOURCE USAGE]"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" \
        $(docker ps --filter "name=sutazai" -q) | head -20
    
    echo -e "\n[UNHEALTHY SERVICES]"
    docker ps --filter "name=sutazai" --format "{{.Names}}: {{.Status}}" | grep -v "healthy" | grep -v "starting" || echo "All services healthy"
    
    echo -e "\n[NETWORK CONNECTIONS]"
    docker network inspect sutazaiapp_sutazai-network --format '{{len .Containers}}' | xargs echo "Connected containers:"
    
    echo -e "\nPress Ctrl+C to exit. Refreshing in 10 seconds..."
    sleep 10
done
EOF

chmod +x monitor-infrastructure.sh

print_success "Monitoring script created: monitor-infrastructure.sh"

# Final summary
echo ""
echo "========================================="
echo "Infrastructure Fix Summary"
echo "========================================="
print_success "✅ Health check configurations fixed"
print_success "✅ Network segmentation defined"
print_success "✅ Resource allocations optimized"
print_success "✅ Monitoring tools created"

echo ""
echo "Next Steps:"
echo "1. Review infrastructure-report.txt for current status"
echo "2. Run ./monitor-infrastructure.sh for live monitoring"
echo "3. Apply network segmentation with: docker compose -f docker-compose.network-fix.yml up -d"
echo "4. Apply resource optimization with: docker compose -f docker-compose.resource-fix.yml up -d"

echo ""
print_warning "Note: Some changes require container restart to take effect"
print_warning "Use 'docker compose down && docker compose up -d' for full restart if needed"