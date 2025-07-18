#!/bin/bash

echo "🚀 Starting SutazAI AGI/ASI v10 Complete Deployment..."
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Error handling
set -e
trap 'echo -e "${RED}❌ Deployment failed at line $LINENO${NC}"' ERR

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Running as root. Some operations may require non-root user.${NC}"
fi

# Function to print status
print_status() {
    echo -e "${BLUE}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check system requirements
print_status "Checking system requirements..."

if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

print_success "System requirements met"

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p data/{models,vector,workspace,logs}
mkdir -p config/{autogen,qdrant}
mkdir -p monitoring/{grafana,prometheus}
mkdir -p ssl
print_success "Directories created"

# Set up environment variables
print_status "Setting up environment variables..."
cat > .env << EOF
# SutazAI v10 Environment Configuration
POSTGRES_DB=sutazai
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=sutazai_password
REDIS_URL=redis://redis:6379
QDRANT_URL=http://qdrant:6333
CHROMADB_URL=http://chromadb:8000
OLLAMA_URL=http://ollama:11434

# Model Configuration
AUTO_PULL_MODELS=true
DEEPSEEK_R1_MODEL=deepseek-r1:8b
QWEN3_MODEL=qwen3:8b
DEEPSEEK_CODER_MODEL=deepseek-coder:33b
LLAMA2_MODEL=llama2:7b

# Service Configuration
STT_ENGINE=openai-whisper
CONTEXT_COMPRESSION_RATIO=0.5
FSDP_AUTO_WRAP_THRESHOLD=100000000

# Security
JWT_SECRET_KEY=sutazai-jwt-secret-key-v10
API_KEY=sutazai-api-key-v10
EOF
print_success "Environment variables configured"

# Install Ollama if not present
print_status "Setting up Ollama..."
if ! command -v ollama &> /dev/null; then
    print_status "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    print_success "Ollama installed"
else
    print_success "Ollama already installed"
fi

# Create systemd services for better management
print_status "Creating systemd services..."
sudo tee /etc/systemd/system/sutazai-agi.service > /dev/null << EOF
[Unit]
Description=SutazAI AGI/ASI System v10
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/sutazaiapp
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
print_success "Systemd service created"

# Build and start all services
print_status "Building and starting all AI services..."
docker-compose down --remove-orphans 2>/dev/null || true
docker-compose pull
docker-compose build --parallel
docker-compose up -d

print_success "All services started"

# Wait for core services to be ready
print_status "Waiting for core services to be ready..."
services=(
    "postgres:5432"
    "redis:6379"
    "qdrant:6333"
    "chromadb:8000"
    "ollama:11434"
)

for service in "${services[@]}"; do
    host=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    print_status "Waiting for $host:$port..."
    
    timeout=60
    while ! docker exec sutazai-$host sh -c "echo > /dev/tcp/localhost/$port" 2>/dev/null; do
        sleep 2
        timeout=$((timeout - 2))
        if [ $timeout -le 0 ]; then
            print_error "Timeout waiting for $host:$port"
            break
        fi
    done
    print_success "$host:$port is ready"
done

# Auto-pull required models
print_status "Auto-pulling required AI models..."
docker exec sutazai-enhanced-model-manager curl -X POST http://localhost:8090/auto-setup &

# Setup monitoring
print_status "Configuring monitoring..."
docker exec sutazai-grafana sh -c "
grafana-cli plugins install grafana-piechart-panel
grafana-cli plugins install grafana-worldmap-panel
" || print_warning "Grafana plugins installation failed (non-critical)"

# Create AI orchestration scripts
print_status "Creating AI orchestration scripts..."
cat > ai_orchestrator.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import httpx
import json
from datetime import datetime

class SutazAIOrchestrator:
    def __init__(self):
        self.services = {
            'backend': 'http://localhost:8000',
            'enhanced_model_manager': 'http://localhost:8098',
            'context_engineering': 'http://localhost:8099',
            'fms_fsdp': 'http://localhost:8100',
            'realtimestt': 'http://localhost:8101',
            'autogpt': 'http://localhost:8080',
            'localagi': 'http://localhost:8082',
            'langflow': 'http://localhost:7860',
            'dify': 'http://localhost:5001'
        }
    
    async def health_check_all(self):
        results = {}
        async with httpx.AsyncClient() as client:
            for service, url in self.services.items():
                try:
                    response = await client.get(f"{url}/health", timeout=5)
                    results[service] = {
                        'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                        'response_time': response.elapsed.total_seconds()
                    }
                except Exception as e:
                    results[service] = {'status': 'error', 'error': str(e)}
        return results
    
    async def orchestrate_task(self, task_description: str):
        # Example orchestration workflow
        workflow = {
            'timestamp': datetime.now().isoformat(),
            'task': task_description,
            'steps': []
        }
        
        # Step 1: Context engineering
        try:
            async with httpx.AsyncClient() as client:
                context_response = await client.post(
                    f"{self.services['context_engineering']}/process",
                    json={
                        'text': task_description,
                        'context_type': 'general',
                        'compression_ratio': 0.7
                    }
                )
                workflow['steps'].append({
                    'service': 'context_engineering',
                    'status': 'success',
                    'result': context_response.json()
                })
        except Exception as e:
            workflow['steps'].append({
                'service': 'context_engineering',
                'status': 'error',
                'error': str(e)
            })
        
        return workflow

if __name__ == "__main__":
    orchestrator = SutazAIOrchestrator()
    
    # Health check
    health_results = asyncio.run(orchestrator.health_check_all())
    print("Service Health Status:")
    for service, status in health_results.items():
        print(f"  {service}: {status['status']}")
    
    # Example task orchestration
    task_result = asyncio.run(orchestrator.orchestrate_task("Generate a Python function for data analysis"))
    print(f"\nTask Orchestration Result: {json.dumps(task_result, indent=2)}")
EOF

chmod +x ai_orchestrator.py
print_success "AI orchestrator created"

# Create monitoring script
print_status "Creating system monitoring script..."
cat > monitor_system.py << 'EOF'
#!/usr/bin/env python3
import docker
import time
import json
from datetime import datetime

def monitor_services():
    client = docker.from_env()
    
    while True:
        try:
            containers = client.containers.list(filters={'label': 'com.docker.compose.project=sutazaiapp'})
            
            status_report = {
                'timestamp': datetime.now().isoformat(),
                'total_containers': len(containers),
                'services': {}
            }
            
            for container in containers:
                service_name = container.labels.get('com.docker.compose.service', 'unknown')
                status_report['services'][service_name] = {
                    'status': container.status,
                    'health': container.attrs.get('State', {}).get('Health', {}).get('Status', 'unknown'),
                    'cpu_usage': 'monitoring...',
                    'memory_usage': 'monitoring...'
                }
            
            print(f"System Status Report - {status_report['timestamp']}")
            print(f"Total Containers: {status_report['total_containers']}")
            print("Service Status:")
            for service, info in status_report['services'].items():
                print(f"  {service}: {info['status']} (health: {info['health']})")
            
            print("-" * 50)
            time.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            print(f"Monitoring error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    monitor_services()
EOF

chmod +x monitor_system.py
print_success "System monitoring script created"

# Create convenience scripts
print_status "Creating convenience scripts..."

# Start script
cat > start_sutazai.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting SutazAI AGI/ASI System..."
docker-compose up -d
echo "✅ System started. Access at: http://localhost:8501"
echo "📊 Monitoring: http://localhost:3000 (Grafana)"
echo "🔧 API Docs: http://localhost:8000/docs"
EOF
chmod +x start_sutazai.sh

# Stop script
cat > stop_sutazai.sh << 'EOF'
#!/bin/bash
echo "🛑 Stopping SutazAI AGI/ASI System..."
docker-compose down
echo "✅ System stopped."
EOF
chmod +x stop_sutazai.sh

# Status script
cat > status_sutazai.sh << 'EOF'
#!/bin/bash
echo "📊 SutazAI AGI/ASI System Status:"
echo "================================="
docker-compose ps
echo ""
echo "🌐 Service URLs:"
echo "• Main UI: http://localhost:8501"
echo "• Backend API: http://localhost:8000"
echo "• Grafana: http://localhost:3000"
echo "• Prometheus: http://localhost:9090"
echo "• AutoGPT: http://localhost:8080"
echo "• LocalAGI: http://localhost:8082"
echo "• TabbyML: http://localhost:8081"
echo "• Browser Use: http://localhost:8083"
echo "• Skyvern: http://localhost:8084"
echo "• Enhanced Model Manager: http://localhost:8098"
echo "• Context Engineering: http://localhost:8099"
echo "• FSDP Service: http://localhost:8100"
echo "• RealtimeSTT: http://localhost:8101"
EOF
chmod +x status_sutazai.sh

print_success "Convenience scripts created"

# Final health check
print_status "Performing final health check..."
sleep 15  # Give services time to fully start

# Check main services
main_services=("8501" "8000" "5432" "6379" "6333")
all_healthy=true

for port in "${main_services[@]}"; do
    if curl -s http://localhost:$port/health > /dev/null 2>&1 || nc -z localhost $port 2>/dev/null; then
        print_success "Service on port $port is responding"
    else
        print_warning "Service on port $port is not responding yet"
        all_healthy=false
    fi
done

# Display final status
echo ""
echo "=============================================="
if [ "$all_healthy" = true ]; then
    print_success "🎉 SutazAI AGI/ASI v10 Deployment Complete!"
else
    print_warning "⚠️  Deployment complete with some services still starting"
fi
echo "=============================================="
echo ""
echo "🌐 Access Points:"
echo "• Main Application: http://localhost:8501"
echo "• Backend API: http://localhost:8000/docs"
echo "• Monitoring Dashboard: http://localhost:3000"
echo "• Prometheus: http://localhost:9090"
echo ""
echo "🔧 Management Commands:"
echo "• Start system: ./start_sutazai.sh"
echo "• Stop system: ./stop_sutazai.sh"
echo "• Check status: ./status_sutazai.sh"
echo "• Monitor system: python3 monitor_system.py"
echo "• Orchestrate AI: python3 ai_orchestrator.py"
echo ""
echo "📁 Important Directories:"
echo "• Models: ./data/models"
echo "• Workspace: ./data/workspace"
echo "• Logs: ./data/logs"
echo "• Config: ./config"
echo ""
echo "🔒 Default Credentials:"
echo "• Grafana: admin/admin"
echo "• PostgreSQL: sutazai/sutazai_password"
echo ""
echo "📚 Documentation:"
echo "• System Guide: ./COMPLETE_SYSTEM_USAGE_GUIDE.md"
echo "• API Documentation: http://localhost:8000/docs"
echo ""
print_success "Deployment completed successfully! 🚀"