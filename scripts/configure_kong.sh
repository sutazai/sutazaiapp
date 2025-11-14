#!/bin/bash
# Kong Gateway Configuration Script
# Configures services and routes for SutazAI Platform

set -e

KONG_ADMIN="http://localhost:10009"

echo "=== Kong Gateway Configuration ==="
echo ""

# Function to create service
create_service() {
    local name=$1
    local url=$2
    echo "Creating service: $name"
    curl -s -X POST "$KONG_ADMIN/services" \
        -d "name=$name" \
        -d "url=$url" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"  ✓ Service ID: {d.get('id', 'ERROR')[:16]}\")" 2>/dev/null || echo "  ✗ Failed"
}

# Function to create route
create_route() {
    local service=$1
    local path=$2
    local name=$3
    echo "Creating route: $name"
    curl -s -X POST "$KONG_ADMIN/services/$service/routes" \
        -d "paths[]=$path" \
        -d "name=$name" \
        -d "strip_path=false" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"  ✓ Route ID: {d.get('id', 'ERROR')[:16]}\")" 2>/dev/null || echo "  ✗ Failed"
}

# Function to add rate limiting plugin
add_rate_limit() {
    local service=$1
    local limit=$2
    echo "Adding rate limiting to $service ($limit req/min)"
    curl -s -X POST "$KONG_ADMIN/services/$service/plugins" \
        -d "name=rate-limiting" \
        -d "config.minute=$limit" \
        -d "config.policy=local" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"  ✓ Plugin ID: {d.get('id', 'ERROR')[:16]}\")" 2>/dev/null || echo "  ✗ Failed"
}

# Function to add CORS plugin
add_cors() {
    local service=$1
    echo "Adding CORS to $service"
    curl -s -X POST "$KONG_ADMIN/services/$service/plugins" \
        -d "name=cors" \
        -d "config.origins=*" \
        -d "config.methods=GET,POST,PUT,DELETE,OPTIONS" \
        -d "config.headers=Accept,Content-Type,Authorization" \
        -d "config.exposed_headers=X-Auth-Token" \
        -d "config.credentials=true" \
        -d "config.max_age=3600" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"  ✓ Plugin ID: {d.get('id', 'ERROR')[:16]}\")" 2>/dev/null || echo "  ✗ Failed"
}

echo "1. Creating Backend API Service"
create_service "backend-api" "http://sutazai-backend:8000"
create_route "backend-api" "/api" "backend-route"
add_rate_limit "backend-api" 1000
add_cors "backend-api"
echo ""

echo "2. Creating MCP Bridge Service"
create_service "mcp-bridge" "http://sutazai-mcp-bridge:11100"
create_route "mcp-bridge" "/mcp" "mcp-route"
add_rate_limit "mcp-bridge" 500
add_cors "mcp-bridge"
echo ""

echo "3. Creating AI Agents Service (Proxy)"
create_service "ai-agents-proxy" "http://sutazai-backend:8000"
create_route "ai-agents-proxy" "/agents" "agents-route"
add_rate_limit "ai-agents-proxy" 200
add_cors "ai-agents-proxy"
echo ""

echo "4. Creating Vector DB Service (Proxy)"
create_service "vector-db-proxy" "http://sutazai-chromadb:8000"
create_route "vector-db-proxy" "/vectors" "vectors-route"
add_rate_limit "vector-db-proxy" 500
echo ""

echo ""
echo "=== Configuration Summary ==="
curl -s "$KONG_ADMIN/services" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"Total Services: {len(d['data'])}\"); [print(f\"  - {s['name']}\") for s in d['data']]"
echo ""
curl -s "$KONG_ADMIN/routes" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"Total Routes: {len(d['data'])}\"); [print(f\"  - {r['name']}: {r['paths']}\") for r in d['data']]"
echo ""

echo "=== Kong Gateway Ready ==="
echo "Gateway URL: http://localhost:10008"
echo ""
echo "Available Routes:"
echo "  - http://localhost:10008/api/*     → Backend API"
echo "  - http://localhost:10008/mcp/*     → MCP Bridge"
echo "  - http://localhost:10008/agents/*  → AI Agents"
echo "  - http://localhost:10008/vectors/* → Vector Databases"
echo ""
echo "Test command:"
echo "  curl http://localhost:10008/api/health"
