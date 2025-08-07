#!/bin/bash

# Enhanced Consul Service Registration with Docker Network Discovery
# Uses actual Docker container IPs for proper health checking

echo "======================================"
echo "SutazAI Consul Service Registration v2"
echo "======================================"

# Get Docker network name
NETWORK_NAME="sutazai-network"

# Function to get container IP
get_container_ip() {
    local container_name=$1
    docker inspect "$container_name" --format "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}" 2>/dev/null | head -1
}

# Function to register service with Docker IP
register_docker_service() {
    local service_name=$1
    local container_name=$2
    local container_port=$3
    local health_path=$4
    local tags=${5:-"sutazai,production"}
    
    # Get container IP
    local container_ip=$(get_container_ip "$container_name")
    
    if [ -z "$container_ip" ]; then
        echo "⚠️  Skipping $service_name - container not found or no IP"
        return 1
    fi
    
    # Generate unique service ID
    local service_id="${service_name}-$(echo $container_ip | tr '.' '-')"
    
    # Create the service definition
    local json_payload=$(cat <<EOF
{
  "ID": "${service_id}",
  "Name": "${service_name}",
  "Tags": [$(echo "$tags" | sed 's/,/","/g' | sed 's/^/"/;s/$/"/') ],
  "Address": "${container_ip}",
  "Port": ${container_port},
  "Check": {
    "HTTP": "http://${container_ip}:${container_port}${health_path}",
    "Method": "GET",
    "Interval": "10s",
    "Timeout": "3s",
    "DeregisterCriticalServiceAfter": "60s"
  },
  "Meta": {
    "container": "${container_name}",
    "environment": "production"
  }
}
EOF
)
    
    # Register the service
    response=$(curl -s -X PUT \
        http://localhost:10006/v1/agent/service/register \
        -H "Content-Type: application/json" \
        -d "$json_payload")
    
    if [ $? -eq 0 ]; then
        echo "✅ Registered: ${service_name} @ ${container_ip}:${container_port}"
    else
        echo "❌ Failed: ${service_name}"
    fi
}

# Deregister old services first
echo "Cleaning up old registrations..."
for service_id in $(curl -s http://localhost:10006/v1/agent/services | python3 -c "import json,sys; [print(k) for k in json.load(sys.stdin).keys() if 'test-service' not in k]"); do
    curl -s -X PUT http://localhost:10006/v1/agent/service/deregister/${service_id}
    echo "  Deregistered: ${service_id}"
done

echo ""
echo "Registering services with Docker IPs..."
echo "---------------------------------------"

# Core Services
register_docker_service "backend-api" "sutazai-backend" 8000 "/health" "api,backend,core"
register_docker_service "frontend-ui" "sutazai-frontend" 8501 "/health" "ui,frontend,streamlit"

# Agent Services
register_docker_service "ai-orchestrator" "sutazai-ai-agent-orchestrator" 8080 "/health" "agent,orchestrator"
register_docker_service "task-coordinator" "sutazai-task-assignment-coordinator" 8080 "/health" "agent,coordinator"
register_docker_service "resource-arbitrator" "sutazai-resource-arbitration-agent" 8080 "/health" "agent,arbitrator"
register_docker_service "multi-coordinator" "sutazai-multi-agent-coordinator" 8080 "/health" "agent,multi"
register_docker_service "hardware-optimizer" "sutazai-hardware-resource-optimizer" 8080 "/health" "agent,hardware"

# Database Services
register_docker_service "postgresql-db" "sutazai-postgres" 5432 "/" "database,postgres"
register_docker_service "redis-cache" "sutazai-redis" 6379 "/" "cache,redis"
register_docker_service "qdrant-vector" "sutazai-qdrant" 6333 "/health" "vectordb,qdrant"

# Infrastructure Services
register_docker_service "rabbitmq-broker" "sutazaiapp-rabbitmq" 15672 "/api/health/checks/virtual-hosts" "messaging,rabbitmq"
register_docker_service "ollama-llm" "sutazai-ollama" 11434 "/api/tags" "llm,ollama,tinyllama"
register_docker_service "kong-gateway" "sutazaiapp-kong" 8001 "/status" "gateway,kong"
register_docker_service "consul-server" "sutazaiapp-consul" 8500 "/v1/agent/self" "consul,discovery"

# Monitoring Services
register_docker_service "prometheus-metrics" "sutazai-prometheus" 9090 "/-/ready" "monitoring,prometheus"
register_docker_service "grafana-dashboards" "sutazai-grafana" 3000 "/api/health" "monitoring,grafana"

echo ""
echo "======================================"
echo "Registration Complete"
echo "======================================"

# Wait for health checks to run
echo ""
echo "Waiting for health checks to execute..."
sleep 5

# Show service health summary
echo ""
echo "Service Health Summary:"
echo "-----------------------"
curl -s http://localhost:10006/v1/health/state/passing | python3 -c "
import json, sys
data = json.load(sys.stdin)
services = {}
for check in data:
    service = check.get('ServiceName', '')
    if service and service not in services:
        services[service] = 'passing'
        
for service in sorted(services.keys()):
    print(f'  ✅ {service}: HEALTHY')
"

curl -s http://localhost:10006/v1/health/state/critical | python3 -c "
import json, sys
data = json.load(sys.stdin)
services = {}
for check in data:
    service = check.get('ServiceName', '')
    if service and service not in services:
        services[service] = check.get('Output', 'No output')[:40]
        
for service in sorted(services.keys()):
    print(f'  ❌ {service}: CRITICAL')
"

echo ""
echo "View full details at: http://localhost:10006/ui/dc1/services"
echo ""

# Generate JSON config for persistence
echo "Generating persistent configuration..."
cat > /opt/sutazaiapp/config/consul-services.json <<EOF
{
  "services": [
$(curl -s http://localhost:10006/v1/agent/services | python3 -c "
import json, sys
services = json.load(sys.stdin)
configs = []
for sid, svc in services.items():
    if 'test-service' not in sid:
        config = {
            'name': svc.get('Service', ''),
            'id': sid,
            'address': svc.get('Address', ''),
            'port': svc.get('Port', 0),
            'tags': svc.get('Tags', [])
        }
        configs.append(json.dumps(config, indent=4))
print(',\\n'.join(configs))
")
  ]
}
EOF

echo "Configuration saved to: /opt/sutazaiapp/config/consul-services.json"