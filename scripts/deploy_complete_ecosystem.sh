#!/bin/bash
# Complete SutazAI Ecosystem Deployment Script

set -e

echo "🚀 Deploying Complete SutazAI Ecosystem..."

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/{models,jupyter,ray,langflow,flowise,dify,llamaindex,chainlit,comfyui,pytorch,tensorflow}
mkdir -p data/agents
mkdir -p data/workflow_reports
mkdir -p logs

# Ensure network exists
echo "🌐 Creating Docker network..."
docker network create sutazai-network 2>/dev/null || true

# Deploy core infrastructure
echo "🏗️ Deploying core infrastructure..."
docker-compose -f docker-compose.minimal.yml up -d postgres redis ollama

# Wait for core services
echo "⏳ Waiting for core services..."
sleep 10

# Deploy backend and frontend
echo "🎯 Deploying backend and frontend..."
docker-compose -f docker-compose.minimal.yml up -d backend frontend

# Deploy vector databases
echo "🗄️ Deploying vector databases..."
docker-compose -f docker-compose.minimal.yml up -d chromadb qdrant neo4j

# Deploy monitoring stack
echo "📊 Deploying monitoring stack..."
docker-compose -f docker-compose.yml up -d prometheus grafana loki promtail

# Deploy workflow engines
echo "🔄 Deploying workflow engines..."
docker-compose -f docker-compose.workflow-engines.yml up -d n8n langflow flowise dify-api dify-worker dify-web

# Deploy ML/DL frameworks (if GPU available)
if command -v nvidia-smi &> /dev/null; then
    echo "🤖 Deploying ML/DL frameworks with GPU support..."
    docker-compose -f docker-compose.workflow-engines.yml up -d pytorch tensorflow
else
    echo "ℹ️ No GPU detected, skipping GPU-dependent services"
fi

# Deploy all agents
echo "🤖 Deploying ALL 85+ agents..."

# Original agents
docker-compose -f docker-compose.agents.yml up -d

# Extended agents (external frameworks)
docker-compose -f docker-compose.agents-extended.yml up -d

# Remaining agents
docker-compose -f docker-compose.agents-remaining.yml up -d

# Deploy additional services
echo "🔧 Deploying additional services..."
docker-compose -f docker-compose.workflow-engines.yml up -d llamaindex chainlit jupyter ray-head

# Wait for all services to start
echo "⏳ Waiting for all services to initialize..."
sleep 30

# Health check
echo "🏥 Running health checks..."
./scripts/check_all_services.sh

# Count deployed services
TOTAL_CONTAINERS=$(docker ps --filter "name=sutazai-" | wc -l)
AGENT_COUNT=$(docker ps --filter "name=sutazai-" | grep -E "agent|developer|engineer|specialist|coordinator|manager|optimizer" | wc -l)

echo "
✅ DEPLOYMENT COMPLETE!

📊 Deployment Summary:
- Total Containers: $TOTAL_CONTAINERS
- AI Agents: $AGENT_COUNT / 85+
- Status: FULLY OPERATIONAL

🌐 Access Points:
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000/docs
- Grafana: http://localhost:3000
- n8n: http://localhost:5678
- LangFlow: http://localhost:7860
- Flowise: http://localhost:3030
- Dify: http://localhost:3031
- Jupyter: http://localhost:8888 (token: sutazai-jupyter-token)
- Ray Dashboard: http://localhost:8265
- Chainlit: http://localhost:8001

📝 Next Steps:
1. Access the frontend at http://localhost:8501
2. Check API documentation at http://localhost:8000/docs
3. Monitor system health at http://localhost:3000 (Grafana)
4. Create workflows using n8n, LangFlow, or Flowise

🔧 Management Commands:
- View logs: docker-compose logs -f [service]
- Stop all: docker-compose down
- Restart service: docker-compose restart [service]

🎉 The complete SutazAI Multi-Agent Ecosystem is now running!
"