#!/bin/bash

# SutazAI TinyLlama Startup Script
# 100% Local LLM - No External APIs

set -e

echo "🚀 Starting SutazAI with TinyLlama (100% Local Mode)"
echo "=================================================="

# Check system resources
echo "📊 System Resource Check:"
free -h | grep -E "Mem:|Swap:"
echo ""

# Use TinyLlama environment
export $(cat .env.tinyllama | grep -v '^#' | xargs)

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose.tinyllama.yml down 2>/dev/null || true

# Clean up old data if requested
if [ "$1" = "--clean" ]; then
    echo "🧹 Cleaning up old data..."
    docker volume rm sutazaiapp_ollama_data sutazaiapp_postgres_data sutazaiapp_redis_data 2>/dev/null || true
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p agents/task-assignment-coordinator
mkdir -p agents/infrastructure-devops-manager
mkdir -p agents/ollama-integration-specialist
mkdir -p workspace
mkdir -p logs

# Create simple Dockerfiles for agents
echo "🐳 Creating agent Dockerfiles..."

# Task Coordinator Dockerfile
cat > agents/task-assignment-coordinator/Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
RUN pip install httpx asyncio
COPY /opt/sutazaiapp/agents/base_agent.py /app/
COPY /opt/sutazaiapp/.claude/agents /agents/
ENV PYTHONPATH=/app
CMD ["python", "-c", "from base_agent import TaskCoordinatorAgent; import asyncio; agent = TaskCoordinatorAgent(); asyncio.run(agent.execute_task('Initialize task coordinator'))"]
EOF

# Infrastructure Manager Dockerfile
cat > agents/infrastructure-devops-manager/Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
RUN pip install httpx asyncio docker
COPY /opt/sutazaiapp/agents/base_agent.py /app/
COPY /opt/sutazaiapp/.claude/agents /agents/
ENV PYTHONPATH=/app
CMD ["python", "-c", "from base_agent import OllamaLocalAgent; import asyncio; agent = OllamaLocalAgent('infrastructure-devops-manager'); asyncio.run(agent.execute_task('Initialize infrastructure manager'))"]
EOF

# Ollama Specialist Dockerfile
cat > agents/ollama-integration-specialist/Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
RUN pip install httpx asyncio
COPY /opt/sutazaiapp/agents/base_agent.py /app/
COPY /opt/sutazaiapp/.claude/agents /agents/
ENV PYTHONPATH=/app
CMD ["python", "-c", "from base_agent import OllamaLocalAgent; import asyncio; agent = OllamaLocalAgent('ollama-integration-specialist'); asyncio.run(agent.execute_task('Initialize Ollama specialist'))"]
EOF

# Start core services
echo "🐳 Starting core services..."
docker-compose -f docker-compose.tinyllama.yml up -d ollama postgres redis

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to be ready..."
sleep 10

# Initialize TinyLlama
echo "🤖 Initializing TinyLlama model..."
docker-compose -f docker-compose.tinyllama.yml run --rm init-tinyllama

# Start core agents
echo "🧠 Starting core agents..."
docker-compose -f docker-compose.tinyllama.yml up -d task-coordinator infra-manager ollama-specialist

# Show status
echo ""
echo "✅ SutazAI TinyLlama is starting up!"
echo ""
echo "📊 Resource Usage:"
echo "  - Model: TinyLlama (637MB)"
echo "  - RAM Usage: ~2-3GB total"
echo "  - CPU: 50% maximum"
echo "  - 100% Local - No external APIs"
echo ""
echo "🔗 Service URLs:"
echo "  - Ollama Native API: http://localhost:11434"
echo "  - PostgreSQL: localhost:5432"
echo "  - Redis: localhost:6379"
echo ""
echo "📝 Test Commands:"
echo "  - Test Ollama: curl http://localhost:11434/api/tags"
echo "  - Generate text: curl -X POST http://localhost:11434/api/generate -d '{\"model\": \"tinyllama:latest\", \"prompt\": \"Hello!\"}'"
echo "  - Chat with TinyLlama: ollama run tinyllama:latest"
echo ""
echo "🎯 Using Native Ollama API:"
echo "  - All agents communicate directly with Ollama"
echo "  - No OpenAI compatibility layer needed"
echo "  - 100% local, 100% private"
echo ""
echo "📖 Logs:"
echo "  docker-compose -f docker-compose.tinyllama.yml logs -f"