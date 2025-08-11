#!/bin/bash
# ULTRASECURITY: Secure Monitoring Services Migration Script
# Converts all monitoring services to run as non-root users
# Target: 100% non-root containers (29/29)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Header
echo "=================================================="
echo "ULTRASECURITY: Monitoring Services Migration"
echo "Converting 5 containers to non-root users"
echo "=================================================="
echo ""

# Step 1: Stop current monitoring services
log_info "Step 1: Stopping current monitoring services..."
docker compose stop promtail cadvisor blackbox-exporter consul redis-exporter 2>/dev/null || true
log_success "Services stopped"

# Step 2: Build secure images
log_info "Step 2: Building secure Docker images..."

# Build Promtail
log_info "Building secure Promtail image..."
docker build -t sutazai-promtail-secure:latest \
    -f docker/monitoring-secure/promtail/Dockerfile \
    docker/monitoring-secure/promtail/
log_success "Promtail image built"

# Build cAdvisor
log_info "Building secure cAdvisor image..."
docker build -t sutazai-cadvisor-secure:latest \
    -f docker/monitoring-secure/cadvisor/Dockerfile \
    docker/monitoring-secure/cadvisor/
log_success "cAdvisor image built"

# Build Blackbox Exporter
log_info "Building secure Blackbox Exporter image..."
docker build -t sutazai-blackbox-exporter-secure:latest \
    -f docker/monitoring-secure/blackbox-exporter/Dockerfile \
    docker/monitoring-secure/blackbox-exporter/
log_success "Blackbox Exporter image built"

# Build Consul
log_info "Building secure Consul image..."
docker build -t sutazai-consul-secure:latest \
    -f docker/monitoring-secure/consul/Dockerfile \
    docker/monitoring-secure/consul/
log_success "Consul image built"

# Build Redis Exporter
log_info "Building secure Redis Exporter image..."
docker build -t sutazai-redis-exporter-secure:latest \
    -f docker/monitoring-secure/redis-exporter/Dockerfile \
    docker/monitoring-secure/redis-exporter/
log_success "Redis Exporter image built"

# Step 3: Remove old containers
log_info "Step 3: Removing old containers..."
docker compose rm -f promtail cadvisor blackbox-exporter consul redis-exporter 2>/dev/null || true
log_success "Old containers removed"

# Step 4: Deploy with secure configuration
log_info "Step 4: Deploying secure monitoring services..."
docker compose -f docker-compose.yml -f docker-compose.security-monitoring.yml up -d \
    promtail cadvisor blackbox-exporter consul redis-exporter
log_success "Secure monitoring services deployed"

# Step 5: Wait for services to be healthy
log_info "Step 5: Waiting for services to become healthy..."
sleep 10

# Step 6: Verify non-root status
log_info "Step 6: Verifying security status..."
echo ""
echo "Container Security Status:"
echo "--------------------------"

check_container_user() {
    local container=$1
    local user_info=$(docker exec $container id 2>/dev/null || echo "Container not running")
    
    if [[ $user_info == *"uid=0(root)"* ]]; then
        log_error "$container: STILL RUNNING AS ROOT - $user_info"
        return 1
    elif [[ $user_info == "Container not running" ]]; then
        log_warning "$container: Container not running or command failed"
        return 1
    else
        log_success "$container: Running as non-root - $user_info"
        return 0
    fi
}

# Check each container
SECURE_COUNT=0
TOTAL_COUNT=5

check_container_user "sutazai-promtail" && ((SECURE_COUNT++))
check_container_user "sutazai-cadvisor" && ((SECURE_COUNT++))
check_container_user "sutazai-blackbox-exporter" && ((SECURE_COUNT++))
check_container_user "sutazai-consul" && ((SECURE_COUNT++))
check_container_user "sutazai-redis-exporter" && ((SECURE_COUNT++))

# Step 7: Test service functionality
echo ""
log_info "Step 7: Testing service functionality..."

test_service() {
    local service=$1
    local port=$2
    local endpoint=$3
    
    if curl -s -f "http://localhost:$port$endpoint" > /dev/null 2>&1; then
        log_success "$service is responding on port $port"
        return 0
    else
        log_error "$service is NOT responding on port $port"
        return 1
    fi
}

# Test each service
FUNCTIONAL_COUNT=0

test_service "Blackbox Exporter" 10204 "/health" && ((FUNCTIONAL_COUNT++))
test_service "Consul" 10006 "/v1/status/leader" && ((FUNCTIONAL_COUNT++))
test_service "cAdvisor" 10206 "/healthz" && ((FUNCTIONAL_COUNT++))
test_service "Redis Exporter" 10208 "/health" && ((FUNCTIONAL_COUNT++))
# Promtail doesn't expose a health endpoint on 3100 by default

# Final Report
echo ""
echo "=================================================="
echo "SECURITY MIGRATION REPORT"
echo "=================================================="
echo "Containers migrated to non-root: $SECURE_COUNT/$TOTAL_COUNT"
echo "Services functional: $FUNCTIONAL_COUNT/4 (Promtail has no health endpoint)"
echo ""

if [ $SECURE_COUNT -eq $TOTAL_COUNT ]; then
    log_success "SUCCESS: All monitoring services now running as non-root users!"
    echo ""
    echo "ACHIEVEMENT UNLOCKED: 100% Non-Root Monitoring Stack"
    echo "Security compliance increased from 83% to 100% for monitoring services"
else
    log_warning "Migration partially successful. Some services may need additional configuration."
    echo "Please check the logs for failed services:"
    echo "  docker compose logs promtail"
    echo "  docker compose logs cadvisor"
    echo "  docker compose logs blackbox-exporter"
    echo "  docker compose logs consul"
    echo "  docker compose logs redis-exporter"
fi

echo ""
echo "Next steps:"
echo "1. Monitor services for 5 minutes to ensure stability"
echo "2. Check Prometheus targets at http://localhost:10200/targets"
echo "3. Verify metrics collection in Grafana at http://localhost:10201"
echo ""
log_info "Migration complete!"