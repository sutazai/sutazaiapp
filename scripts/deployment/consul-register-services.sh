#!/bin/bash

# Strict error handling
set -euo pipefail


# Consul Service Registration Script
# Registers all SutazAI services with health checks


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

echo "======================================"
echo "SutazAI Consul Service Registration"
echo "======================================"

# Check if Consul is reachable
if ! curl -s http://localhost:10006/v1/agent/self > /dev/null; then
    echo "❌ Consul is not reachable on port 10006"
    exit 1
fi

echo "✅ Consul is reachable"

# Function to register a service
register_service() {
    local service_name=$1
    local service_id=$2
    local service_address=$3
    local service_port=$4
    local health_endpoint=$5
    local health_interval=${6:-"10s"}
    local health_timeout=${7:-"2s"}
    
    # Create JSON payload
    cat > "$(mktemp /tmp/consul-${service_id}.json.XXXXXX)" <<EOF
{
  "ID": "${service_id}",
  "Name": "${service_name}",
  "Tags": ["sutazai", "production"],
  "Address": "${service_address}",
  "Port": ${service_port},
  "Check": {
    "HTTP": "http://${service_address}:${service_port}${health_endpoint}",
    "Method": "GET",
    "Interval": "${health_interval}",
    "Timeout": "${health_timeout}",
    "DeregisterCriticalServiceAfter": "90s"
  }
}
EOF
    
    # Register the service
    curl -s -X PUT \
        http://localhost:10006/v1/agent/service/register \
        -H "Content-Type: application/json" \
        -d @/tmp/consul-${service_id}.json
    
    if [ $? -eq 0 ]; then
        echo "✅ Registered: ${service_name} (${service_id})"
    else
        echo "❌ Failed to register: ${service_name}"
    fi
    
    rm -f /tmp/consul-${service_id}.json
}

echo ""
echo "Registering core services..."
echo "----------------------------"

# Backend Service
register_service \
    "sutazai-backend" \
    "backend-1" \
    "sutazai-backend" \
    "8000" \
    "/health" \
    "10s" \
    "2s"

# Frontend Service  
register_service \
    "sutazai-frontend" \
    "frontend-1" \
    "sutazai-frontend" \
    "8501" \
    "/health" \
    "10s" \
    "2s"

echo ""
echo "Registering agent services..."
echo "-----------------------------"

# AI Agent Orchestrator
register_service \
    "ai-agent-orchestrator" \
    "agent-orchestrator-1" \
    "sutazai-ai-agent-orchestrator" \
    "8080" \
    "/health" \
    "10s" \
    "2s"

# Task Assignment Coordinator
register_service \
    "task-assignment-coordinator" \
    "agent-coordinator-1" \
    "sutazai-task-assignment-coordinator" \
    "8080" \
    "/health" \
    "10s" \
    "2s"

# Resource Arbitration Agent
register_service \
    "resource-arbitration-agent" \
    "agent-arbitration-1" \
    "sutazai-resource-arbitration-agent" \
    "8080" \
    "/health" \
    "10s" \
    "2s"

# Multi-Agent Coordinator
register_service \
    "multi-agent-coordinator" \
    "agent-multi-1" \
    "sutazai-multi-agent-coordinator" \
    "8080" \
    "/health" \
    "10s" \
    "2s"

# Hardware Resource Optimizer
register_service \
    "hardware-resource-optimizer" \
    "agent-hardware-1" \
    "sutazai-hardware-resource-optimizer" \
    "8080" \
    "/health" \
    "10s" \
    "2s"

echo ""
echo "Registering database services..."
echo "--------------------------------"

# PostgreSQL
register_service \
    "postgresql" \
    "postgres-1" \
    "sutazai-postgres" \
    "5432" \
    "/" \
    "30s" \
    "5s"

# Redis
register_service \
    "redis" \
    "redis-1" \
    "sutazai-redis" \
    "6379" \
    "/" \
    "10s" \
    "2s"

# Qdrant Vector DB
register_service \
    "qdrant" \
    "qdrant-1" \
    "sutazai-qdrant" \
    "6333" \
    "/health" \
    "10s" \
    "2s"

echo ""
echo "Registering infrastructure services..."
echo "--------------------------------------"

# RabbitMQ
register_service \
    "rabbitmq" \
    "rabbitmq-1" \
    "sutazaiapp-rabbitmq" \
    "5672" \
    "/" \
    "10s" \
    "2s"

# Ollama
register_service \
    "ollama" \
    "ollama-1" \
    "sutazai-ollama" \
    "11434" \
    "/api/tags" \
    "30s" \
    "5s"

# Kong API Gateway
register_service \
    "kong" \
    "kong-1" \
    "sutazaiapp-kong" \
    "8000" \
    "/" \
    "10s" \
    "2s"

echo ""
echo "======================================"
echo "Service Registration Complete"
echo "======================================"
echo ""

# List all registered services
echo "Verifying registered services:"
echo "------------------------------"
curl -s http://localhost:10006/v1/agent/services | python3 -c "
import json, sys
data = json.load(sys.stdin)
for service_id, service in data.items():
    print(f'  ✓ {service[\"Service\"]} ({service_id}) - Port {service.get(\"Port\", \"N/A\")}')"

echo ""
echo "To view in Consul UI: http://localhost:10006/ui/dc1/services"
echo ""