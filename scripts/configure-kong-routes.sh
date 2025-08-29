#!/bin/bash

# SutazAI Platform - Kong API Gateway Configuration
# Sets up routing for all services through the API gateway

KONG_ADMIN_URL="http://localhost:10009"

echo "========================================="
echo "Configuring Kong API Gateway Routes"
echo "========================================="

# Function to create service and route
create_service_route() {
    local name=$1
    local url=$2
    local path=$3
    local tags=$4
    
    echo "Creating service: $name"
    
    # Create service
    curl -i -X POST "$KONG_ADMIN_URL/services" \
        --data "name=$name" \
        --data "url=$url" \
        --data "tags=$tags" 2>/dev/null | grep -q "201 Created"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Service created"
    else
        echo "  ⚠ Service may already exist"
    fi
    
    # Create route
    curl -i -X POST "$KONG_ADMIN_URL/services/$name/routes" \
        --data "paths[]=$path" \
        --data "strip_path=false" \
        --data "preserve_host=true" 2>/dev/null | grep -q "201 Created"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Route created: $path"
    else
        echo "  ⚠ Route may already exist"
    fi
    echo ""
}

# Backend API Services
create_service_route "backend-api" "http://localhost:10200" "/api/v1" "backend,api"
create_service_route "mcp-bridge" "http://localhost:11100" "/mcp" "mcp,bridge,ai"

# Frontend Service
create_service_route "frontend" "http://localhost:11000" "/" "frontend,ui"

# Vector Database Services
create_service_route "chromadb" "http://localhost:10100" "/vector/chroma" "vector,chromadb"
create_service_route "qdrant" "http://localhost:10101" "/vector/qdrant" "vector,qdrant"
create_service_route "faiss" "http://localhost:10103" "/vector/faiss" "vector,faiss"

# AI Model Service
create_service_route "ollama" "http://localhost:11434" "/llm" "llm,ollama,ai"

# AI Agent Services - Phase 1
create_service_route "letta" "http://localhost:11400" "/agents/letta" "agent,letta"
create_service_route "crewai" "http://localhost:11401" "/agents/crewai" "agent,crewai"
create_service_route "autogpt" "http://localhost:11402" "/agents/autogpt" "agent,autogpt"
create_service_route "aider" "http://localhost:11403" "/agents/aider" "agent,aider"
create_service_route "private-gpt" "http://localhost:11404" "/agents/private-gpt" "agent,private-gpt"

# AI Agent Services - Phase 2
create_service_route "localagi" "http://localhost:11406" "/agents/localagi" "agent,localagi"
create_service_route "bigagi" "http://localhost:11407" "/agents/bigagi" "agent,bigagi"
create_service_route "agentzero" "http://localhost:11408" "/agents/agentzero" "agent,agentzero"
create_service_route "skyvern" "http://localhost:11409" "/agents/skyvern" "agent,skyvern"
create_service_route "autogen" "http://localhost:11415" "/agents/autogen" "agent,autogen"

# Testing & Security Services
create_service_route "semgrep" "http://localhost:11801" "/security/semgrep" "security,semgrep"
create_service_route "browseruse" "http://localhost:11703" "/testing/browseruse" "testing,browseruse"

# Health Check Aggregator Route
echo "Creating health check aggregator..."
curl -i -X POST "$KONG_ADMIN_URL/services" \
    --data "name=health-check" \
    --data "url=http://localhost:11100/status" \
    --data "tags=health,monitoring" 2>/dev/null | grep -q "201 Created"

curl -i -X POST "$KONG_ADMIN_URL/services/health-check/routes" \
    --data "paths[]=/health" \
    --data "strip_path=true" 2>/dev/null | grep -q "201 Created"

echo ""
echo "========================================="
echo "Kong Configuration Complete"
echo "========================================="

# List all configured services
echo ""
echo "Configured Services:"
curl -s "$KONG_ADMIN_URL/services" | jq -r '.data[] | "\(.name): \(.url)"'

echo ""
echo "Configured Routes:"
curl -s "$KONG_ADMIN_URL/routes" | jq -r '.data[] | "\(.paths[0]) -> \(.service.name)"'

# Add rate limiting plugin globally
echo ""
echo "Adding global rate limiting..."
curl -i -X POST "$KONG_ADMIN_URL/plugins" \
    --data "name=rate-limiting" \
    --data "config.minute=100" \
    --data "config.policy=local" 2>/dev/null | grep -q "201 Created"

if [ $? -eq 0 ]; then
    echo "✓ Rate limiting enabled (100 requests/minute)"
else
    echo "⚠ Rate limiting may already be configured"
fi

# Add CORS plugin globally
echo "Adding global CORS support..."
curl -i -X POST "$KONG_ADMIN_URL/plugins" \
    --data "name=cors" \
    --data "config.origins=*" \
    --data "config.methods=GET,POST,PUT,DELETE,OPTIONS" \
    --data "config.headers=Accept,Accept-Version,Content-Length,Content-MD5,Content-Type,Date,X-Auth-Token,Authorization" \
    --data "config.exposed_headers=X-Auth-Token" \
    --data "config.credentials=true" \
    --data "config.max_age=3600" 2>/dev/null | grep -q "201 Created"

if [ $? -eq 0 ]; then
    echo "✓ CORS enabled globally"
else
    echo "⚠ CORS may already be configured"
fi

echo ""
echo "Kong API Gateway is ready at: http://localhost:10008"
echo "Kong Admin API is available at: http://localhost:10009"