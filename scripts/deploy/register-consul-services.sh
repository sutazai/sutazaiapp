#!/bin/bash

# SutazAI Platform - Consul Service Registration Script
# Registers all services with Consul for service discovery

CONSUL_URL="http://localhost:10006"

echo "========================================="
echo "Registering Services with Consul"
echo "========================================="

# Function to register a service
register_service() {
    local name=$1
    local id=$2
    local address=$3
    local port=$4
    local tags=$5
    local health_endpoint=$6
    
    echo "Registering service: $name"
    
    cat <<EOF | curl -X PUT "$CONSUL_URL/v1/agent/service/register" -H "Content-Type: application/json" -d @-
{
  "ID": "$id",
  "Name": "$name",
  "Tags": $tags,
  "Address": "$address",
  "Port": $port,
  "Check": {
    "HTTP": "$health_endpoint",
    "Interval": "30s",
    "Timeout": "10s"
  }
}
EOF
    echo " ✓"
}

# Core Infrastructure Services
register_service "postgres" "postgres-1" "localhost" 10000 '["database","sql","core"]' "http://localhost:10000/health"
register_service "redis" "redis-1" "localhost" 10001 '["cache","session","core"]' "http://localhost:10001/health"
register_service "neo4j" "neo4j-1" "localhost" 10002 '["graph","database","core"]' "http://localhost:10002/db/data/"
register_service "rabbitmq" "rabbitmq-1" "localhost" 10004 '["queue","messaging","core"]' "http://localhost:10005/api/health/checks/alarms"
register_service "kong" "kong-1" "localhost" 10008 '["gateway","api","proxy"]' "http://localhost:10009/status"

# Vector Databases
register_service "chromadb" "chromadb-1" "localhost" 10100 '["vector","database","ai"]' "http://localhost:10100/api/v1/heartbeat"
register_service "qdrant" "qdrant-1" "localhost" 10101 '["vector","database","ai"]' "http://localhost:10101/readiness"
register_service "faiss" "faiss-1" "localhost" 10103 '["vector","database","ai"]' "http://localhost:10103/health"

# Backend Services
register_service "backend-api" "backend-1" "localhost" 10200 '["api","backend","core"]' "http://localhost:10200/health"
register_service "mcp-bridge" "mcp-bridge-1" "localhost" 11100 '["bridge","mcp","ai"]' "http://localhost:11100/health"

# Frontend Services
register_service "frontend" "frontend-1" "localhost" 11000 '["ui","frontend","streamlit"]' "http://localhost:11000/health"

# AI Services
register_service "ollama" "ollama-1" "localhost" 11434 '["llm","ai","model"]' "http://localhost:11434/api/tags"

# AI Agents (Phase 1)
register_service "letta" "letta-1" "localhost" 11400 '["agent","ai","memory"]' "http://localhost:11400/health"
register_service "crewai" "crewai-1" "localhost" 11401 '["agent","ai","orchestration"]' "http://localhost:11401/health"
register_service "autogpt" "autogpt-1" "localhost" 11402 '["agent","ai","autonomous"]' "http://localhost:11402/health"
register_service "aider" "aider-1" "localhost" 11403 '["agent","ai","code"]' "http://localhost:11403/health"
register_service "private-gpt" "private-gpt-1" "localhost" 11404 '["agent","ai","documents"]' "http://localhost:11404/health"

# AI Agents (Phase 2)
register_service "localagi" "localagi-1" "localhost" 11406 '["agent","ai","local"]' "http://localhost:11406/health"
register_service "bigagi" "bigagi-1" "localhost" 11407 '["agent","ai","chat"]' "http://localhost:11407/health"
register_service "agentzero" "agentzero-1" "localhost" 11408 '["agent","ai","framework"]' "http://localhost:11408/health"
register_service "skyvern" "skyvern-1" "localhost" 11409 '["agent","ai","browser"]' "http://localhost:11409/health"
register_service "autogen" "autogen-1" "localhost" 11415 '["agent","ai","config"]' "http://localhost:11415/health"

# Testing & Security
register_service "semgrep" "semgrep-1" "localhost" 11801 '["security","testing","code"]' "http://localhost:11801/health"
register_service "browseruse" "browseruse-1" "localhost" 11703 '["testing","browser","automation"]' "http://localhost:11703/health"

echo ""
echo "========================================="
echo "Service Registration Complete"
echo "========================================="

# List all registered services
echo ""
echo "Registered Services:"
curl -s "$CONSUL_URL/v1/agent/services" | jq -r 'to_entries[] | "\(.key): \(.value.Service) on port \(.value.Port)"'