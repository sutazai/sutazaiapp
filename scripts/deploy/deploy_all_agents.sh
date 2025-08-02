#!/bin/bash
# Deploy ALL agents

echo "🚀 Deploying ALL agents..."

# Deploy existing agents
docker-compose -f docker-compose.agents.yml up -d

# Deploy extended agents
docker-compose -f docker-compose.agents-extended.yml up -d

# Deploy remaining agents
docker-compose -f docker-compose.agents-remaining.yml up -d

# Show count
AGENT_COUNT=$(docker ps --filter "name=sutazai-" | grep -E "agent|developer|engineer|specialist|coordinator|manager|optimizer" | wc -l)
echo "✅ Total agents deployed: $AGENT_COUNT"

# List all agents
docker ps --filter "name=sutazai-" --format "table {.Names}\t{.Status}\t{.Ports}"
