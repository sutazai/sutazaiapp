#!/bin/bash
# Deploy new agents without dependencies issues

echo "🚀 Deploying new agents..."

# First, remove depends_on from compose files since services are already running
echo "📝 Preparing agent deployments..."

# Deploy extended agents (external frameworks)
echo "🤖 Building and deploying extended agents..."
docker-compose -f docker-compose.agents-extended.yml build
docker-compose -f docker-compose.agents-extended.yml up -d --no-deps

# Deploy remaining agents  
echo "🤖 Building and deploying remaining agents..."
docker-compose -f docker-compose.agents-remaining.yml build
docker-compose -f docker-compose.agents-remaining.yml up -d --no-deps

# Deploy workflow engines
echo "🔄 Deploying workflow engines..."
docker-compose -f docker-compose.workflow-engines.yml up -d langflow flowise dify-api dify-worker dify-web chainlit llamaindex

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 10

# Check deployment status
echo "📊 Checking deployment status..."
TOTAL_AGENTS=$(docker ps --filter "name=sutazai-" | grep -E "agent|developer|engineer|specialist|coordinator|manager|optimizer" | wc -l)
TOTAL_CONTAINERS=$(docker ps --filter "name=sutazai-" | wc -l)

echo "
✅ Deployment Status:
- Total Containers: $((TOTAL_CONTAINERS-1))
- AI Agents Running: $TOTAL_AGENTS
"

# Show new agents
echo "🆕 Newly deployed agents:"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "autogpt|crewai|aider|gpt-engineer|letta|babyagi|shellgpt|pentestgpt|langflow|flowise|dify|chainlit"

echo "✅ New agents deployment complete!"