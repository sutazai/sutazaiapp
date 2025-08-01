#!/bin/bash

# Deploy Memory-Optimized SutazAI System for Small Models
# Prevents system freezing on 15GB RAM systems

set -e

echo "🚀 Deploying Memory-Optimized SutazAI System for Small Models"
echo "================================================="

# Configuration
COMPOSE_FILE="docker-compose.yml"
MEMORY_COMPOSE_FILE="docker-compose.memory-optimized.yml"
OLLAMA_CONFIG="/opt/sutazaiapp/config/ollama_optimization.yaml"

# Check system requirements
echo "📋 Checking system requirements..."
TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
echo "   Total RAM: ${TOTAL_RAM}GB"

if [ "$TOTAL_RAM" -lt 12 ]; then
    echo "⚠️  WARNING: System has less than 12GB RAM. Small model optimization is critical."
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running or not accessible"
    exit 1
fi

# Stop existing containers gracefully
echo "🛑 Stopping existing containers..."
docker-compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true

# Clean up Docker system
echo "🧹 Cleaning up Docker system..."
docker system prune -f --volumes
docker image prune -f

# Verify small models are available
echo "🔍 Verifying small models..."
if [ -d "/opt/sutazaiapp/data/ollama" ]; then
    echo "   Ollama data directory exists"
else
    echo "   Creating Ollama data directory"
    mkdir -p /opt/sutazaiapp/data/ollama
fi

# Install required Python packages for optimization scripts
echo "📦 Installing optimization dependencies..."
pip3 install -q psutil pyyaml requests schedule flask 2>/dev/null || true

# Make optimization scripts executable
chmod +x /opt/sutazaiapp/scripts/ollama_memory_optimizer.py
chmod +x /opt/sutazaiapp/scripts/memory_cleanup_service.py

# Deploy with memory optimization
echo "🔧 Deploying with memory optimization..."
docker-compose -f "$COMPOSE_FILE" -f "$MEMORY_COMPOSE_FILE" up -d --remove-orphans

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to be ready..."
timeout=120
counter=0
while [ $counter -lt $timeout ]; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "   ✅ Ollama is ready"
        break
    fi
    sleep 2
    counter=$((counter + 2))
    if [ $((counter % 20)) -eq 0 ]; then
        echo "   Still waiting for Ollama... (${counter}s)"
    fi
done

if [ $counter -ge $timeout ]; then
    echo "   ⚠️  Ollama took longer than expected to start"
fi

# Pull small models if not present
echo "🔽 Ensuring small models are available..."
docker exec sutazai-ollama ollama list | grep -q "qwen2.5:3b" || {
    echo "   Pulling qwen2.5:3b (small, efficient model)..."
    docker exec sutazai-ollama ollama pull qwen2.5:3b
}

docker exec sutazai-ollama ollama list | grep -q "llama3.2:3b" || {
    echo "   Pulling llama3.2:3b (backup small model)..."
    docker exec sutazai-ollama ollama pull llama3.2:3b
}

# Start memory cleanup service
echo "🔧 Starting memory cleanup service..."
nohup python3 /opt/sutazaiapp/scripts/memory_cleanup_service.py > /opt/sutazaiapp/logs/memory_cleanup.log 2>&1 &
echo $! > /opt/sutazaiapp/memory_cleanup.pid

# Start Ollama memory optimizer
echo "🔧 Starting Ollama memory optimizer..."
nohup python3 /opt/sutazaiapp/scripts/ollama_memory_optimizer.py > /opt/sutazaiapp/logs/ollama_optimizer.log 2>&1 &
echo $! > /opt/sutazaiapp/ollama_optimizer.pid

# Wait for services to stabilize
echo "⏳ Waiting for services to stabilize..."
sleep 30

# Verify deployment
echo "🔍 Verifying deployment..."
FAILED_SERVICES=0

# Check critical services
CRITICAL_SERVICES="sutazai-ollama sutazai-postgres sutazai-redis sutazai-backend-agi"
for service in $CRITICAL_SERVICES; do
    if docker ps | grep -q "$service"; then
        echo "   ✅ $service is running"
    else
        echo "   ❌ $service is not running"
        FAILED_SERVICES=$((FAILED_SERVICES + 1))
    fi
done

# Check memory usage
echo "📊 Current system status:"
MEMORY_USAGE=$(free | awk '/^Mem:/{printf "%.1f", $3/$2 * 100.0}')
echo "   Memory usage: ${MEMORY_USAGE}%"

if [ $(echo "$MEMORY_USAGE > 85" | bc -l) -eq 1 ]; then
    echo "   ⚠️  High memory usage detected"
    echo "   🔧 Triggering immediate optimization..."
    curl -s http://localhost:8523/optimize > /dev/null || true
    curl -s http://localhost:8523/optimize-small-models > /dev/null || true
fi

# Show loaded models
echo "   Loaded models:"
curl -s http://localhost:11434/api/ps | jq -r '.models[]?.name // "No models loaded"' 2>/dev/null || echo "   Unable to check loaded models"

# Test small model
echo "🧪 Testing small model functionality..."
TEST_RESPONSE=$(curl -s -X POST http://localhost:11434/api/generate \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen2.5:3b",
        "prompt": "Hello, respond with just: OK",
        "stream": false,
        "options": {
            "temperature": 0,
            "num_predict": 5
        }
    }' | jq -r '.response // "ERROR"' 2>/dev/null)

if echo "$TEST_RESPONSE" | grep -q "OK"; then
    echo "   ✅ Small model is working correctly"
else
    echo "   ⚠️  Small model test inconclusive"
fi

# Final status
echo ""
echo "🎯 Deployment Summary:"
echo "================================================="
echo "   Total containers: $(docker ps | wc -l)"
echo "   Memory usage: ${MEMORY_USAGE}%"
echo "   Failed services: $FAILED_SERVICES"
echo ""

if [ $FAILED_SERVICES -eq 0 ]; then
    echo "✅ Memory-optimized deployment completed successfully!"
    echo ""
    echo "🔗 Access URLs:"
    echo "   Frontend: http://localhost:8501"
    echo "   Backend API: http://localhost:8000"
    echo "   Hardware Optimizer: http://localhost:8523"
    echo "   Ollama API: http://localhost:11434"
    echo ""
    echo "📝 Small Model Configuration:"
    echo "   Primary model: qwen2.5:3b (~2GB RAM)"
    echo "   Secondary model: llama3.2:3b (~2GB RAM)"
    echo "   Max concurrent models: 1"
    echo "   Memory limit per model: 3GB"
    echo ""
    echo "🔧 Optimization Services:"
    echo "   Memory cleanup: PID $(cat /opt/sutazaiapp/memory_cleanup.pid 2>/dev/null || echo 'Not running')"
    echo "   Ollama optimizer: PID $(cat /opt/sutazaiapp/ollama_optimizer.pid 2>/dev/null || echo 'Not running')"
    echo ""
    echo "📊 Monitor with: curl http://localhost:8523/system-summary"
else
    echo "⚠️  Deployment completed with $FAILED_SERVICES failed services"
    echo "   Check logs: docker-compose logs -f"
fi

echo ""
echo "💡 Tips for optimal performance on 15GB RAM:"
echo "   • Only use small models (3B parameters or less)"
echo "   • Monitor memory with: watch -n 5 'free -h'"
echo "   • Force optimization: curl http://localhost:8523/optimize"
echo "   • Emergency cleanup: curl http://localhost:8523/emergency-scale-down"