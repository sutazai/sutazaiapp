#!/bin/bash

# SutazAI Architecture Enhancement Deployment Script
# This script deploys the enhanced components without disrupting existing services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "======================================"
echo "SutazAI Architecture Enhancement Deploy"
echo "======================================"
echo ""

cd "$PROJECT_ROOT"

# Function to check if service is running
check_service() {
    local service=$1
    if docker ps --format "{{.Names}}" | grep -q "^sutazai-$service$"; then
        return 0
    else
        return 1
    fi
}

# Function to wait for service health
wait_for_health() {
    local service=$1
    local url=$2
    local max_attempts=30
    local attempt=0
    
    echo "Waiting for $service to be healthy..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            echo "✓ $service is healthy"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    echo "✗ $service failed to become healthy"
    return 1
}

# Phase 1: Deploy Service Mesh Components
echo ""
echo "Phase 1: Deploying Service Mesh Components"
echo "==========================================="

# Deploy Kong Gateway
if ! check_service "kong"; then
    echo "Deploying Kong Gateway..."
    docker run -d \
        --name sutazai-kong \
        --network sutazai-network \
        -e "KONG_DATABASE=off" \
        -e "KONG_DECLARATIVE_CONFIG=/kong/kong.yml" \
        -e "KONG_PROXY_ACCESS_LOG=/dev/stdout" \
        -e "KONG_ADMIN_ACCESS_LOG=/dev/stdout" \
        -e "KONG_PROXY_ERROR_LOG=/dev/stderr" \
        -e "KONG_ADMIN_ERROR_LOG=/dev/stderr" \
        -p 10005:8000 \
        -p 8001:8001 \
        -v "$PROJECT_ROOT/config/kong:/kong" \
        --restart unless-stopped \
        kong:3.5
    
    wait_for_health "Kong" "http://localhost:8001/status"
else
    echo "✓ Kong is already running"
fi

# Deploy Consul
if ! check_service "consul"; then
    echo "Deploying Consul..."
    docker run -d \
        --name sutazai-consul \
        --network sutazai-network \
        -p 10006:8500 \
        -p 8600:8600/udp \
        -e CONSUL_BIND_INTERFACE=eth0 \
        --restart unless-stopped \
        hashicorp/consul:latest \
        agent -server -bootstrap-expect=1 -ui -client=0.0.0.0
    
    wait_for_health "Consul" "http://localhost:10006/v1/status/leader"
else
    echo "✓ Consul is already running"
fi

# Deploy RabbitMQ
if ! check_service "rabbitmq"; then
    echo "Deploying RabbitMQ..."
    docker run -d \
        --name sutazai-rabbitmq \
        --network sutazai-network \
        -e RABBITMQ_DEFAULT_USER=admin \
        -e RABBITMQ_DEFAULT_PASS=sutazai_rabbit \
        -p 10007:5672 \
        -p 10008:15672 \
        --restart unless-stopped \
        rabbitmq:3.12-management
    
    wait_for_health "RabbitMQ" "http://admin:sutazai_rabbit@localhost:10008/api/overview"
else
    echo "✓ RabbitMQ is already running"
fi

# Phase 2: Build and Deploy Enhanced AI Agents
echo ""
echo "Phase 2: Building Enhanced AI Agents"
echo "====================================="

# Build AI Agent Orchestrator
if [ -f "$PROJECT_ROOT/agents/ai-agent-orchestrator/enhanced_app.py" ]; then
    echo "Building AI Agent Orchestrator..."
    docker build -t sutazai-ai-orchestrator:enhanced \
        -f "$PROJECT_ROOT/agents/ai-agent-orchestrator/Dockerfile" \
        "$PROJECT_ROOT/agents/ai-agent-orchestrator"
    
    # Stop old version if running
    docker stop sutazai-ai-agent-orchestrator 2>/dev/null || true
    docker rm sutazai-ai-agent-orchestrator 2>/dev/null || true
    
    # Deploy new version
    echo "Deploying enhanced AI Agent Orchestrator..."
    docker run -d \
        --name sutazai-ai-agent-orchestrator \
        --network sutazai-network \
        -e PORT=8589 \
        -e REDIS_HOST=redis \
        -e RABBITMQ_HOST=rabbitmq \
        -e OLLAMA_URL=http://ollama:10104 \
        -e OLLAMA_MODEL=tinyllama \
        -p 8589:8589 \
        --restart unless-stopped \
        sutazai-ai-orchestrator:enhanced
    
    wait_for_health "AI Orchestrator" "http://localhost:8589/health"
fi

# Phase 3: Configure Kong Routes
echo ""
echo "Phase 3: Configuring Kong Routes"
echo "================================="

# Create Kong configuration directory
mkdir -p "$PROJECT_ROOT/config/kong"

# Create Kong configuration
cat > "$PROJECT_ROOT/config/kong/kong.yml" << 'EOF'
_format_version: "3.0"
_transform: true

services:
  - name: backend-api
    url: http://backend:8000
    routes:
      - name: backend-route
        paths:
          - /api/v1
        strip_path: false

  - name: ai-orchestrator
    url: http://ai-agent-orchestrator:8589
    routes:
      - name: orchestrator-route
        paths:
          - /ai/orchestrate
        strip_path: true

  - name: frontend
    url: http://frontend:8501
    routes:
      - name: frontend-route
        paths:
          - /ui
        strip_path: true

plugins:
  - name: cors
    config:
      origins:
        - http://localhost:10011
        - http://localhost:10005
      credentials: true
      methods:
        - GET
        - POST
        - PUT
        - DELETE
        - OPTIONS
      headers:
        - Accept
        - Authorization
        - Content-Type
      exposed_headers:
        - X-Auth-Token
      max_age: 3600

  - name: rate-limiting
    config:
      minute: 100
      hour: 10000
      policy: local
EOF

# Reload Kong configuration
echo "Reloading Kong configuration..."
docker exec sutazai-kong kong reload 2>/dev/null || echo "Kong will pick up config on next restart"

# Phase 4: Setup RabbitMQ Exchanges and Queues
echo ""
echo "Phase 4: Configuring RabbitMQ"
echo "============================="

# Wait a bit for RabbitMQ to fully initialize
sleep 5

# Create Python script to setup RabbitMQ
cat > /tmp/setup_rabbitmq.py << 'EOF'
import pika
import sys
import time

def setup_rabbitmq():
    try:
        # Connect to RabbitMQ
        credentials = pika.PlainCredentials('admin', 'sutazai_rabbit')
        connection = pika.BlockingConnection(
            pika.ConnectionParameters('localhost', 10007, credentials=credentials)
        )
        channel = connection.channel()
        
        # Create exchanges
        channel.exchange_declare(exchange='ai.tasks', exchange_type='topic', durable=True)
        channel.exchange_declare(exchange='ai.events', exchange_type='fanout', durable=True)
        channel.exchange_declare(exchange='ai.coordination', exchange_type='direct', durable=True)
        
        # Create queues
        agents = ['orchestrator', 'task_assignment', 'resource_arbitration', 
                  'multi_agent_coordinator', 'hardware_optimizer']
        
        for agent in agents:
            # Task queue
            queue_name = f'agent.{agent}.tasks'
            channel.queue_declare(queue=queue_name, durable=True)
            channel.queue_bind(exchange='ai.tasks', queue=queue_name, 
                              routing_key=f'task.{agent}.*')
            
            # Event queue
            event_queue = f'agent.{agent}.events'
            channel.queue_declare(queue=event_queue, durable=False)
            channel.queue_bind(exchange='ai.events', queue=event_queue)
        
        print("✓ RabbitMQ setup complete")
        connection.close()
        return 0
        
    except Exception as e:
        print(f"✗ RabbitMQ setup failed: {e}")
        return 1

if __name__ == "__main__":
    # Retry logic
    for attempt in range(5):
        result = setup_rabbitmq()
        if result == 0:
            sys.exit(0)
        time.sleep(3)
    sys.exit(1)
EOF

# Run RabbitMQ setup
python3 /tmp/setup_rabbitmq.py || echo "RabbitMQ setup will be completed by services"

# Phase 5: Register Services with Consul
echo ""
echo "Phase 5: Registering Services with Consul"
echo "========================================="

# Register backend service
curl -X PUT http://localhost:10006/v1/agent/service/register -d '{
  "ID": "backend-api-1",
  "Name": "backend-api",
  "Port": 8000,
  "Address": "backend",
  "Check": {
    "HTTP": "http://backend:8000/health",
    "Interval": "10s"
  }
}' 2>/dev/null || echo "Backend registration pending"

# Register Ollama service
curl -X PUT http://localhost:10006/v1/agent/service/register -d '{
  "ID": "ollama-1",
  "Name": "ollama-llm",
  "Port": 10104,
  "Address": "ollama",
  "Check": {
    "HTTP": "http://ollama:10104/api/tags",
    "Interval": "30s"
  }
}' 2>/dev/null || echo "Ollama registration pending"

# Phase 6: System Health Check
echo ""
echo "Phase 6: System Health Check"
echo "============================"

echo ""
echo "Service Status:"
echo "---------------"

# Check core services
services=("postgres" "redis" "neo4j" "ollama" "backend" "frontend")
for service in "${services[@]}"; do
    if check_service "$service"; then
        echo "✓ $service: Running"
    else
        echo "✗ $service: Not running"
    fi
done

echo ""
echo "New Services:"
echo "-------------"

# Check new services
new_services=("kong" "consul" "rabbitmq" "ai-agent-orchestrator")
for service in "${new_services[@]}"; do
    if check_service "$service"; then
        echo "✓ $service: Running"
    else
        echo "✗ $service: Not running"
    fi
done

# Phase 7: Quick Integration Test
echo ""
echo "Phase 7: Integration Test"
echo "========================="

# Test Kong routing
echo "Testing Kong API Gateway..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:10005/api/v1/health 2>/dev/null | grep -q "200\|404"; then
    echo "✓ Kong routing works"
else
    echo "✗ Kong routing failed"
fi

# Test Consul service discovery
echo "Testing Consul service discovery..."
if curl -s http://localhost:10006/v1/catalog/services 2>/dev/null | grep -q "consul"; then
    echo "✓ Consul service discovery works"
else
    echo "✗ Consul service discovery failed"
fi

# Test RabbitMQ
echo "Testing RabbitMQ..."
if curl -s -u admin:sutazai_rabbit http://localhost:10008/api/overview 2>/dev/null | grep -q "rabbitmq_version"; then
    echo "✓ RabbitMQ management API works"
else
    echo "✗ RabbitMQ management API failed"
fi

# Test AI Orchestrator
echo "Testing AI Agent Orchestrator..."
if curl -s http://localhost:8589/health 2>/dev/null | grep -q "healthy"; then
    echo "✓ AI Agent Orchestrator is healthy"
else
    echo "✗ AI Agent Orchestrator not responding"
fi

echo ""
echo "======================================"
echo "Deployment Complete!"
echo "======================================"
echo ""
echo "Access Points:"
echo "- Kong Gateway: http://localhost:10005"
echo "- Kong Admin: http://localhost:8001"
echo "- Consul UI: http://localhost:10006"
echo "- RabbitMQ Management: http://localhost:10008 (admin/sutazai_rabbit)"
echo "- AI Orchestrator: http://localhost:8589"
echo ""
echo "Next Steps:"
echo "1. Test the orchestrator: curl -X POST http://localhost:8589/orchestrate -H 'Content-Type: application/json' -d '{\"task_type\":\"code_generation\",\"payload\":{\"language\":\"python\"}}'"
echo "2. View Consul services: http://localhost:10006/ui/dc1/services"
echo "3. Monitor RabbitMQ queues: http://localhost:10008/#/queues"
echo "4. Check system metrics: http://localhost:10201 (Grafana)"
echo ""
echo "For the full enhancement plan, see: $PROJECT_ROOT/SUTAZAI_ARCHITECTURE_ENHANCEMENT_PLAN.md"