#!/bin/bash
# Check health of all SutazAI services

echo "🏥 Checking SutazAI Services Health..."

# Function to check service
check_service() {
    local name=$1
    local url=$2
    
    if curl -s -f "$url" > /dev/null 2>&1; then
        echo "✅ $name: HEALTHY"
    else
        echo "❌ $name: UNREACHABLE"
    fi
}

# Core services
echo -e "\n📦 Core Services:"
check_service "Backend API" "http://localhost:8000/health"
check_service "Frontend" "http://localhost:8501"
check_service "Ollama" "http://localhost:11434/api/tags"

# Vector databases
echo -e "\n🗄️ Vector Databases:"
check_service "ChromaDB" "http://localhost:8100/api/v1/heartbeat"
check_service "Qdrant" "http://localhost:6333/health"

# Workflow engines
echo -e "\n🔄 Workflow Engines:"
check_service "n8n" "http://localhost:5678"
check_service "LangFlow" "http://localhost:7860"
check_service "Flowise" "http://localhost:3030"
check_service "Dify" "http://localhost:3031"

# Monitoring
echo -e "\n📊 Monitoring:"
check_service "Grafana" "http://localhost:3000/api/health"
check_service "Prometheus" "http://localhost:9090/-/healthy"

# Additional services
echo -e "\n🔧 Additional Services:"
check_service "Jupyter" "http://localhost:8888"
check_service "Ray Dashboard" "http://localhost:8265"
check_service "Chainlit" "http://localhost:8001"

# Count running containers
echo -e "\n📈 Container Statistics:"
TOTAL=$(docker ps --filter "name=sutazai-" | wc -l)
AGENTS=$(docker ps --filter "name=sutazai-" | grep -E "agent|developer|engineer|specialist" | wc -l)
echo "Total Containers: $((TOTAL-1))"
echo "AI Agents Running: $AGENTS"

# Show resource usage
echo -e "\n💻 Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -10