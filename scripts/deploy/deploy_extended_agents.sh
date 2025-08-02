#!/bin/bash
# Deploy extended agents

echo "🚀 Deploying extended agents..."

# Build all agent images
docker-compose -f docker-compose.agents-extended.yml build

# Deploy agents
docker-compose -f docker-compose.agents-extended.yml up -d

# Show status
docker-compose -f docker-compose.agents-extended.yml ps

echo "✅ Extended agents deployed!"
