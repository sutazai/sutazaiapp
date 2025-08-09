#!/bin/bash

# SutazAI Missing Services Validation Script
# Tests all services to ensure they're running correctly

set -euo pipefail

echo "🔍 Validating SutazAI Missing Services deployment..."

# Service endpoints to test
declare -A services=(
    ["Neo4j HTTP"]="http://localhost:10002"
    ["Kong Gateway"]="http://localhost:10005"
    ["Kong Admin"]="http://localhost:10044"
    ["Consul UI"]="http://localhost:10006"
    ["RabbitMQ Management"]="http://localhost:10008"
    ["Resource Manager"]="http://localhost:10009/health"
    ["Backend API"]="http://localhost:10010/health"
    ["Frontend UI"]="http://localhost:10011"
    ["FAISS Vector"]="http://localhost:10103/health"
    ["Loki"]="http://localhost:10202/ready"
    ["Alertmanager"]="http://localhost:10203/-/healthy"
    ["AI Metrics"]="http://localhost:10204/metrics"
)

# Container health checks
containers=(
    "sutazai-neo4j"
    "sutazai-kong-gateway"
    "sutazai-consul-discovery"
    "sutazai-rabbitmq-queue"
    "sutazai-resource-manager"
    "sutazai-backend-api"
    "sutazai-frontend-ui"
    "sutazai-faiss-vector"
    "sutazai-loki"
    "sutazai-alertmanager"
    "sutazai-ai-metrics"
)

failed_services=()
passed_services=()

echo "🐳 Checking container health..."
for container in "${containers[@]}"; do
    if docker ps --filter "name=$container" --filter "status=running" --format "{{.Names}}" | grep -q "$container"; then
        health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "no-healthcheck")
        if [ "$health_status" = "healthy" ] || [ "$health_status" = "no-healthcheck" ]; then
            echo "  ✅ $container: running ($health_status)"
            passed_services+=("$container")
        else
            echo "  ❌ $container: unhealthy ($health_status)"
            failed_services+=("$container")
        fi
    else
        echo "  ❌ $container: not running"
        failed_services+=("$container")
    fi
done

echo ""
echo "🌐 Testing service endpoints..."
for service_name in "${!services[@]}"; do
    endpoint="${services[$service_name]}"
    echo "  Testing $service_name at $endpoint..."
    
    if curl -sf --max-time 10 "$endpoint" > /dev/null 2>&1; then
        echo "    ✅ $service_name: accessible"
    else
        echo "    ❌ $service_name: not accessible"
        failed_services+=("$service_name")
    fi
done

echo ""
echo "📊 Deployment Summary:"
echo "  ✅ Passed: ${#passed_services[@]} services"
echo "  ❌ Failed: ${#failed_services[@]} services"

if [ ${#failed_services[@]} -ne 0 ]; then
    echo ""
    echo "❌ Failed services/containers:"
    printf '  - %s\n' "${failed_services[@]}"
    echo ""
    echo "🔧 Troubleshooting tips:"
    echo "  1. Check logs: docker-compose -f docker-compose.missing-services.yml logs [service-name]"
    echo "  2. Check container status: docker ps -a"
    echo "  3. Restart services: docker-compose -f docker-compose.missing-services.yml restart [service-name]"
    exit 1
else
    echo ""
    echo "🎉 All services are running successfully!"
    echo ""
    echo "📋 Service Access URLs:"
    echo "  • Neo4j Browser: http://localhost:10002"
    echo "  • Kong Gateway: http://localhost:10005"
    echo "  • Kong Admin API: http://localhost:10044"
    echo "  • Consul UI: http://localhost:10006"
    echo "  • RabbitMQ Management: http://localhost:10008"
    echo "  • Backend API: http://localhost:10010"
    echo "  • Frontend UI: http://localhost:10011"
    echo "  • Loki: http://localhost:10202"
    echo "  • Alertmanager: http://localhost:10203"
    echo ""
    echo "🔐 Default Credentials:"
    echo "  • Neo4j: neo4j/sutazai_neo4j"
    echo "  • RabbitMQ: sutazai/sutazai_rmq"
    exit 0
fi