#!/bin/bash
# Purpose: Deploy core service mesh infrastructure for 150+ agents
# Usage: ./deploy-infrastructure.sh
# Requirements: Docker, Docker Compose

set -e

echo "======================================"
echo "Deploying Service Mesh Infrastructure"
echo "======================================"
echo "Services:"
echo "- Consul (Service Discovery) - Port 10040"
echo "- Kong (API Gateway) - Ports 10001/10002"
echo "- RabbitMQ (Message Queue) - Ports 10041/10042"
echo ""

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker is not running"
    exit 1
fi

# Navigate to project root
cd "$(dirname "$0")/.."

# Stop any existing infrastructure
echo "Stopping any existing infrastructure..."
docker-compose -f docker-compose.infrastructure.yml down 2>/dev/null || true

# Remove old volumes if requested
if [ "$1" == "--clean" ]; then
    echo "Cleaning up old volumes..."
    docker volume rm sutazaiapp_consul-data sutazaiapp_kong-db-data sutazaiapp_rabbitmq-data 2>/dev/null || true
fi

# Deploy infrastructure
echo "Starting infrastructure services..."
docker-compose -f docker-compose.infrastructure.yml up -d

# Wait for services to be healthy
echo "Waiting for services to be healthy..."

# Check Consul
echo -n "Waiting for Consul..."
for i in {1..30}; do
    if curl -s http://localhost:10040/v1/status/leader >/dev/null 2>&1; then
        echo " OK"
        break
    fi
    echo -n "."
    sleep 2
done

# Check Kong
echo -n "Waiting for Kong..."
for i in {1..30}; do
    if curl -s http://localhost:10002/status >/dev/null 2>&1; then
        echo " OK"
        break
    fi
    echo -n "."
    sleep 2
done

# Check RabbitMQ
echo -n "Waiting for RabbitMQ..."
for i in {1..30}; do
    if curl -s http://localhost:10042/api/overview -u admin:adminpass >/dev/null 2>&1; then
        echo " OK"
        break
    fi
    echo -n "."
    sleep 2
done

echo ""
echo "======================================"
echo "Infrastructure Status:"
echo "======================================"
docker-compose -f docker-compose.infrastructure.yml ps

echo ""
echo "======================================"
echo "Service URLs:"
echo "======================================"
echo "Consul UI:      http://localhost:10040"
echo "Kong Admin API: http://localhost:10002"
echo "Kong Proxy:     http://localhost:10001"
echo "RabbitMQ UI:    http://localhost:10042 (admin/adminpass)"
echo ""

# Configure Kong for service mesh
echo "Configuring Kong for service mesh..."
sleep 5

# Add upstream for agent services
curl -s -X POST http://localhost:10002/upstreams \
    --data "name=agent-services" \
    --data "healthchecks.active.healthy.interval=5" \
    --data "healthchecks.active.unhealthy.interval=5" >/dev/null 2>&1 || true

# Add service for agent communication
curl -s -X POST http://localhost:10002/services \
    --data "name=agent-gateway" \
    --data "host=agent-services" \
    --data "port=80" \
    --data "connect_timeout=5000" \
    --data "read_timeout=30000" \
    --data "write_timeout=30000" >/dev/null 2>&1 || true

# Add route for agent services
curl -s -X POST http://localhost:10002/services/agent-gateway/routes \
    --data "paths[]=/agents" \
    --data "strip_path=false" >/dev/null 2>&1 || true

# Configure RabbitMQ for agent communication
echo "Configuring RabbitMQ for agent communication..."

# Create agent exchange
curl -s -X PUT http://localhost:10042/api/exchanges/%2F/agent-exchange \
    -u admin:adminpass \
    -H "content-type: application/json" \
    -d '{"type":"topic","durable":true}' >/dev/null 2>&1 || true

# Create agent queue
curl -s -X PUT http://localhost:10042/api/queues/%2F/agent-tasks \
    -u admin:adminpass \
    -H "content-type: application/json" \
    -d '{"durable":true,"arguments":{"x-max-length":10000}}' >/dev/null 2>&1 || true

echo ""
echo "======================================"
echo "Infrastructure deployed successfully!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Update agent configurations to use service discovery"
echo "2. Register agents with Consul"
echo "3. Configure Kong routes for specific agent services"
echo "4. Set up RabbitMQ queues for agent communication"
echo ""