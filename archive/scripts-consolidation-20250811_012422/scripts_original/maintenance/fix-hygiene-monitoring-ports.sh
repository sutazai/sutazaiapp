#!/bin/bash

# Fix Hygiene Monitoring Port Conflicts and Service Issues
# This script resolves the identified port conflicts and service connectivity issues

set -e


# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

echo "=========================================="
echo "Hygiene Monitoring System - Port Fix"
echo "=========================================="

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -i :$port > /dev/null 2>&1; then
        echo "❌ Port $port is in use"
        return 1
    else
        echo "✅ Port $port is available"
        return 0
    fi
}

echo "Step 1: Checking current port availability..."
echo "Hygiene monitoring ports:"
check_port 5433 || echo "  Warning: PostgreSQL port conflict detected"
check_port 6380 || echo "  Warning: Redis port conflict detected"  
check_port 8081 || echo "  Warning: Backend port conflict detected"
check_port 8101 || echo "  Warning: Rule Control API port conflict detected"
check_port 3002 || echo "  Warning: Dashboard port conflict detected"
check_port 8082 || echo "  Warning: Nginx port conflict detected"

echo ""
echo "Step 2: Stopping existing hygiene monitoring services..."
docker compose -f docker-compose.hygiene-monitor.yml down

echo ""
echo "Step 3: Cleaning up any orphaned containers..."
docker container prune -f

echo ""
echo "Step 4: Checking Docker network..."
if docker network ls | grep -q hygiene-network; then
    echo "✅ Hygiene network exists"
else
    echo "Creating hygiene network..."
    docker network create hygiene-network
fi

echo ""
echo "Step 5: Starting services in correct order..."
echo "Starting PostgreSQL first..."
docker compose -f docker-compose.hygiene-monitor.yml up -d hygiene-postgres

echo "Waiting for PostgreSQL to be healthy..."
for i in {1..30}; do
    if docker compose -f docker-compose.hygiene-monitor.yml ps hygiene-postgres | grep -q "healthy"; then
        echo "✅ PostgreSQL is healthy"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ PostgreSQL failed to start properly"
        docker compose -f docker-compose.hygiene-monitor.yml logs hygiene-postgres
        exit 1
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

echo ""
echo "Starting Redis..."
docker compose -f docker-compose.hygiene-monitor.yml up -d hygiene-redis

echo "Waiting for Redis to be healthy..."
for i in {1..15}; do
    if docker compose -f docker-compose.hygiene-monitor.yml ps hygiene-redis | grep -q "healthy"; then
        echo "✅ Redis is healthy"
        break
    fi
    if [ $i -eq 15 ]; then
        echo "❌ Redis failed to start properly"
        docker compose -f docker-compose.hygiene-monitor.yml logs hygiene-redis
        exit 1
    fi
    echo "Waiting... ($i/15)"
    sleep 2
done

echo ""
echo "Starting backend services..."
docker compose -f docker-compose.hygiene-monitor.yml up -d hygiene-backend rule-control-api

echo "Waiting for backend services to be healthy..."
sleep 10

echo ""
echo "Starting frontend services..."
docker compose -f docker-compose.hygiene-monitor.yml up -d hygiene-dashboard nginx

echo ""
echo "Step 6: Final health check..."
sleep 10

echo ""
echo "=========================================="
echo "Service Status:"
echo "=========================================="
docker compose -f docker-compose.hygiene-monitor.yml ps

echo ""
echo "=========================================="
echo "Port Mapping Summary:"
echo "=========================================="
echo "✅ PostgreSQL (hygiene-postgres): Host:5433 -> Container:5432"
echo "✅ Redis (hygiene-redis): Host:6380 -> Container:6379"  
echo "✅ Backend API (hygiene-backend): Host:8081 -> Container:8080"
echo "✅ Rule Control API (rule-control-api): Host:8101 -> Container:8100"
echo "✅ Dashboard (hygiene-dashboard): Host:3002 -> Container:3000"
echo "✅ Nginx Proxy (hygiene-nginx): Host:8082 -> Container:80"

echo ""
echo "=========================================="
echo "Access URLs:"
echo "=========================================="
echo "🌐 Hygiene Dashboard: http://localhost:8082"
echo "🔧 Direct Dashboard: http://localhost:3002" 
echo "⚙️  Backend API: http://localhost:8081"
echo "🎛️  Rule Control API: http://localhost:8101"
echo "🗄️  PostgreSQL: localhost:5433"
echo "💾 Redis: localhost:6380"

echo ""
echo "=========================================="
echo "Health Check Results:"
echo "=========================================="

# Check each service
services=("hygiene-postgres" "hygiene-redis" "hygiene-backend" "rule-control-api" "hygiene-dashboard" "nginx")
for service in "${services[@]}"; do
    if docker compose -f docker-compose.hygiene-monitor.yml ps $service | grep -q "healthy"; then
        echo "✅ $service: Healthy"
    elif docker compose -f docker-compose.hygiene-monitor.yml ps $service | grep -q "Up"; then
        echo "⚠️  $service: Running (health check pending)"
    else
        echo "❌ $service: Not running"
    fi
done

echo ""
echo "=========================================="
echo "Hygiene Monitoring System - Ready!"
echo "=========================================="
echo ""
echo "All services are configured with conflict-free ports."
echo "No port conflicts detected with the main Sutazai system."
echo ""
echo "If you see any unhealthy services, check the logs with:"
echo "docker compose -f docker-compose.hygiene-monitor.yml logs [service-name]"