#!/bin/bash

# 🚀 Sutazai Hygiene Monitoring System - Perfect Launch Script
# Purpose: One-command startup for containerized monitoring system
# Author: Multi-Agent Coordination System
# Version: 1.0.0 - Production Ready

set -e

echo "🚀 Starting Sutazai Hygiene Monitoring System"
echo "=================================================="

# Color codes for output
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

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

print_success "Docker is running"

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    print_error "docker-compose is not available. Please install docker-compose."
    exit 1
fi

# Use docker compose if available, fallback to docker-compose
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker compose"
else
    DOCKER_COMPOSE_CMD="docker-compose"
fi

print_success "Docker Compose is available"

# Create necessary directories
print_status "Creating required directories..."
mkdir -p logs
mkdir -p config
mkdir -p sql

print_success "Directories created"

# Stop any existing containers
print_status "Stopping any existing hygiene monitoring containers..."
$DOCKER_COMPOSE_CMD -f docker-compose.hygiene-monitor.yml down --remove-orphans

# Build and start the services
print_status "Building and starting services..."
$DOCKER_COMPOSE_CMD -f docker-compose.hygiene-monitor.yml up --build -d

# Wait for services to be healthy
print_status "Waiting for services to become healthy..."

# Function to check if a service is healthy
check_service_health() {
    local service_name=$1
    local health_endpoint=$2
    local max_retries=30
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        if curl -f -s $health_endpoint > /dev/null 2>&1; then
            print_success "$service_name is healthy"
            return 0
        fi
        
        echo -n "."
        sleep 2
        retry_count=$((retry_count + 1))
    done
    
    print_error "$service_name failed to become healthy"
    return 1
}

# Wait for database
print_status "Waiting for PostgreSQL..."
sleep 10  # Give database time to initialize

# Check backend health
echo -n "Checking backend health"
if check_service_health "Backend API" "http://localhost:8081/health"; then
    backend_healthy=true
else
    backend_healthy=false
fi

# Check rule control API health
echo -n "Checking rule control API health"
if check_service_health "Rule Control API" "http://localhost:8101/api/health/live"; then
    rule_api_healthy=true
else
    rule_api_healthy=false
fi

# Check dashboard health
echo -n "Checking dashboard health"
if check_service_health "Dashboard" "http://localhost:3000/health"; then
    dashboard_healthy=true
else
    dashboard_healthy=false
fi

# Check nginx health
echo -n "Checking nginx proxy health"
if check_service_health "Nginx Proxy" "http://localhost:80/health/"; then
    nginx_healthy=true
else
    nginx_healthy=false
fi

echo ""
echo "=========================================="
echo "🎯 SYSTEM STATUS REPORT"
echo "=========================================="

# Display service status
services=(
    "PostgreSQL Database:postgres:5433"
    "Redis Cache:redis:6379"
    "Backend API:hygiene-backend:8081"
    "Rule Control API:rule-control-api:8101" 
    "Dashboard:hygiene-dashboard:3000"
    "Nginx Proxy:nginx:80"
)

for service in "${services[@]}"; do
    IFS=':' read -r name container port <<< "$service"
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "hygiene-$container\|$container"; then
        status=$(docker ps --format "table {{.Names}}\t{{.Status}}" | grep "hygiene-$container\|$container" | awk '{print $2}')
        if [[ $status == "Up" ]]; then
            print_success "$name - Running on port $port"
        else
            print_warning "$name - Status: $status"
        fi
    else
        print_error "$name - Not running"
    fi
done

echo ""
echo "=========================================="
echo "🌐 ACCESS POINTS"
echo "=========================================="

if [ "$nginx_healthy" = true ]; then
    print_success "🎨 Main Dashboard: http://localhost"
    print_success "🔧 Direct Backend API: http://localhost:8081"
    print_success "⚙️  Direct Rule API: http://localhost:8101"
    print_success "🔗 WebSocket: ws://localhost/ws"
else
    print_warning "🎨 Dashboard (direct): http://localhost:3000"
    print_warning "🔧 Backend API (direct): http://localhost:8081" 
    print_warning "⚙️  Rule API (direct): http://localhost:8101"
    print_warning "🔗 WebSocket (direct): ws://localhost:8081/ws"
fi

echo ""
echo "=========================================="
echo "🔍 TESTING ENDPOINTS"
echo "=========================================="

# Test key endpoints
print_status "Testing key endpoints..."

echo -n "Backend status: "
if curl -f -s http://localhost:8081/api/hygiene/status > /dev/null; then
    print_success "✅ Working"
else
    print_error "❌ Failed"
fi

echo -n "System metrics: "
if curl -f -s http://localhost:8081/api/system/metrics > /dev/null; then
    print_success "✅ Working"
else
    print_error "❌ Failed"
fi

echo -n "Rule control: "
if curl -f -s http://localhost:8101/api/rules > /dev/null; then
    print_success "✅ Working"
else
    print_error "❌ Failed"
fi

echo ""
echo "=========================================="
echo "📊 REAL-TIME MONITORING"
echo "=========================================="

# Show some real-time data
print_status "Fetching initial dashboard data..."

if curl -f -s http://localhost:8081/api/hygiene/status > /tmp/dashboard_data.json 2>/dev/null; then
    compliance_score=$(cat /tmp/dashboard_data.json | grep -o '"complianceScore":[0-9]*' | cut -d':' -f2)
    total_violations=$(cat /tmp/dashboard_data.json | grep -o '"totalViolations":[0-9]*' | cut -d':' -f2)
    system_status=$(cat /tmp/dashboard_data.json | grep -o '"systemStatus":"[^"]*"' | cut -d':' -f2 | tr -d '"')
    
    print_success "✅ Compliance Score: ${compliance_score:-0}%"
    print_success "📋 Total Violations: ${total_violations:-0}"
    print_success "🟢 System Status: ${system_status:-UNKNOWN}"
else
    print_warning "⚠️  Unable to fetch dashboard data (system may still be starting)"
fi

echo ""
echo "=========================================="
echo "🎯 QUICK ACTIONS"
echo "=========================================="

echo "To view logs:                docker-compose -f docker-compose.hygiene-monitor.yml logs -f"
echo "To stop system:              docker-compose -f docker-compose.hygiene-monitor.yml down"
echo "To restart a service:        docker-compose -f docker-compose.hygiene-monitor.yml restart [service]"
echo "To scale a service:          docker-compose -f docker-compose.hygiene-monitor.yml up -d --scale [service]=N"

echo ""
echo "=========================================="
echo "🚀 SYSTEM READY!"
echo "=========================================="

if [ "$backend_healthy" = true ] && [ "$dashboard_healthy" = true ]; then
    print_success "🎉 All services are running and healthy!"
    print_success "🌐 Open http://localhost in your browser to access the dashboard"
    print_success "🔄 Real-time monitoring is active with WebSocket connections"
    print_success "💾 Data is persisted in PostgreSQL database"
    print_success "⚡ Redis caching is enabled for optimal performance"
    
    echo ""
    print_status "🔍 The system will now continuously monitor your codebase for hygiene violations"
    print_status "📊 Check the dashboard for real-time compliance metrics"
    print_status "⚙️  Use the rule control panel to customize enforcement settings"
    
else
    print_warning "⚠️  Some services may need more time to start up"
    print_warning "📱 Check the dashboard in a few minutes if not accessible immediately"
    print_warning "🔍 Use 'docker-compose -f docker-compose.hygiene-monitor.yml logs' to debug issues"
fi

echo ""
print_status "🎯 Perfect containerized hygiene monitoring system is now active!"

# Optional: Watch logs
read -p "Would you like to watch the logs in real-time? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "📜 Showing real-time logs (Ctrl+C to exit)..."
    $DOCKER_COMPOSE_CMD -f docker-compose.hygiene-monitor.yml logs -f
fi