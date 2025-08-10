#!/bin/bash

# Strict error handling
set -euo pipefail


# Working Consul Service Registration Script
# Uses actual Docker IPs and registers with health checks


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
echo "Consul Service Registration - Working"
echo "======================================"

# Function to register service
register_service() {
    local name=$1
    local id=$2
    local address=$3
    local port=$4
    local check_url=$5
    
    curl -s -X PUT http://localhost:10006/v1/agent/service/register \
    -H "Content-Type: application/json" \
    -d "{
        \"ID\": \"${id}\",
        \"Name\": \"${name}\",
        \"Address\": \"${address}\",
        \"Port\": ${port},
        \"Check\": {
            \"HTTP\": \"${check_url}\",
            \"Interval\": \"10s\",
            \"Timeout\": \"3s\"
        }
    }"
    
    echo "✅ Registered: ${name} (${address}:${port})"
}

# Clean up old registrations
echo "Cleaning up old registrations..."
for sid in $(curl -s http://localhost:10006/v1/agent/services | python3 -c "import json,sys; [print(k) for k in json.load(sys.stdin).keys()]"); do
    curl -s -X PUT http://localhost:10006/v1/agent/service/deregister/${sid}
done

echo ""
echo "Registering services..."
echo "----------------------"

# Core Services
register_service "backend-api" "backend-api" "172.20.0.6" 8000 "http://172.20.0.6:8000/health"
register_service "frontend-ui" "frontend-ui" "172.20.0.2" 8501 "http://172.20.0.2:8501/health"

# Agent Services  
register_service "ai-orchestrator" "ai-orchestrator" "172.29.0.4" 8080 "http://172.29.0.4:8080/health"
register_service "task-coordinator" "task-coordinator" "172.29.0.8" 8080 "http://172.29.0.8:8080/health"
register_service "resource-arbitrator" "resource-arbitrator" "172.29.0.5" 8080 "http://172.29.0.5:8080/health"
register_service "multi-coordinator" "multi-coordinator" "172.29.0.7" 8080 "http://172.29.0.7:8080/health"
register_service "hardware-optimizer" "hardware-optimizer" "172.29.0.3" 8080 "http://172.29.0.3:8080/health"

# Database Services (TCP checks)
curl -s -X PUT http://localhost:10006/v1/agent/service/register \
-H "Content-Type: application/json" \
-d '{
    "ID": "postgresql",
    "Name": "postgresql",
    "Address": "172.20.0.15",
    "Port": 5432,
    "Check": {
        "TCP": "172.20.0.15:5432",
        "Interval": "30s",
        "Timeout": "5s"
    }
}'
echo "✅ Registered: postgresql (172.20.0.15:5432)"

curl -s -X PUT http://localhost:10006/v1/agent/service/register \
-H "Content-Type: application/json" \
-d '{
    "ID": "redis",
    "Name": "redis",
    "Address": "172.20.0.17",
    "Port": 6379,
    "Check": {
        "TCP": "172.20.0.17:6379",
        "Interval": "10s",
        "Timeout": "2s"
    }
}'
echo "✅ Registered: redis (172.20.0.17:6379)"

# Vector DB
register_service "qdrant" "qdrant" "172.20.0.16" 6333 "http://172.20.0.16:6333/health"

# Message Queue
curl -s -X PUT http://localhost:10006/v1/agent/service/register \
-H "Content-Type: application/json" \
-d '{
    "ID": "rabbitmq",
    "Name": "rabbitmq",
    "Address": "172.29.0.6",
    "Port": 5672,
    "Check": {
        "TCP": "172.29.0.6:5672",
        "Interval": "10s",
        "Timeout": "2s"
    }
}'
echo "✅ Registered: rabbitmq (172.29.0.6:5672)"

# Ollama LLM
register_service "ollama" "ollama" "172.20.0.7" 11434 "http://172.20.0.7:11434/api/tags"

# Monitoring
register_service "prometheus" "prometheus" "172.20.0.4" 9090 "http://172.20.0.4:9090/-/ready"
register_service "grafana" "grafana" "172.20.0.3" 3000 "http://172.20.0.3:3000/api/health"

echo ""
echo "======================================"
echo "Registration Complete"
echo "======================================"

# Wait for health checks
echo ""
echo "Waiting for health checks..."
sleep 5

# Check service status
echo ""
echo "Service Status:"
echo "---------------"
curl -s http://localhost:10006/v1/health/state/any | python3 -c "
import json, sys
data = json.load(sys.stdin)
services = {}
for check in data:
    service = check.get('ServiceName', '')
    status = check.get('Status', '')
    if service and service not in services:
        services[service] = status

for service in sorted(services.keys()):
    status = services[service]
    icon = '✅' if status == 'passing' else '⚠️' if status == 'warning' else '❌'
    print(f'{icon} {service}: {status.upper()}')
"

echo ""
echo "Consul UI: http://localhost:10006/ui/dc1/services"