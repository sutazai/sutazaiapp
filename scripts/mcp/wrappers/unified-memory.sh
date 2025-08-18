#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Unified Memory MCP Wrapper
# Consolidates extended-memory and memory-bank-mcp functionality
# Routes requests to unified memory service running on localhost:3009

if [ "${1:-}" = "--selfcheck" ]; then
  echo "unified-memory MCP selfcheck $(date)"
  
  # Check if unified memory service is running
  if curl -sf http://localhost:3009/health >/dev/null 2>&1; then
    echo "✅ unified-memory service is healthy"
    exit 0
  else
    echo "❌ unified-memory service not responding"
    exit 127
  fi
fi

if [ "${1:-}" = "health" ]; then
  curl -sf http://localhost:3009/health
  exit $?
fi

if [ "${1:-}" = "export-all" ]; then
  curl -sf http://localhost:3009/memory/stats
  exit $?
fi

# MCP Protocol Implementation
# Reads JSON-RPC requests from stdin and routes to HTTP API

while IFS= read -r line; do
  if [ -z "$line" ]; then
    continue
  fi
  
  # Parse JSON-RPC request
  method=$(echo "$line" | jq -r '.method // empty')
  params=$(echo "$line" | jq -r '.params // {}')
  id=$(echo "$line" | jq -r '.id // "1"')
  
  case "$method" in
    "store"|"save_context")
      # Store memory
      response=$(curl -s -X POST http://localhost:3009/memory/store \
        -H "Content-Type: application/json" \
        -d "$params")
      ;;
    "retrieve"|"load_contexts")
      # Retrieve memory
      key=$(echo "$params" | jq -r '.key // .context_id // empty')
      namespace=$(echo "$params" | jq -r '.namespace // "default"')
      response=$(curl -s "http://localhost:3009/memory/retrieve/$key?namespace=$namespace")
      ;;
    "search")
      # Search memory
      query=$(echo "$params" | jq -r '.query // .pattern // empty')
      namespace=$(echo "$params" | jq -r '.namespace // "default"')
      limit=$(echo "$params" | jq -r '.limit // 10')
      response=$(curl -s "http://localhost:3009/memory/search?query=$query&namespace=$namespace&limit=$limit")
      ;;
    "delete"|"forget_context")
      # Delete memory
      key=$(echo "$params" | jq -r '.key // .context_id // empty')
      namespace=$(echo "$params" | jq -r '.namespace // "default"')
      response=$(curl -s -X DELETE "http://localhost:3009/memory/delete/$key?namespace=$namespace")
      ;;
    "stats"|"list_all_projects"|"get_popular_tags")
      # Get statistics
      response=$(curl -s http://localhost:3009/memory/stats)
      ;;
    "health")
      # Health check
      response=$(curl -s http://localhost:3009/health)
      ;;
    *)
      # Unknown method
      response='{"success": false, "error": "Unknown method: '"$method"'"}'
      ;;
  esac
  
  # Return JSON-RPC response
  echo "{\"jsonrpc\": \"2.0\", \"id\": $id, \"result\": $response}"
done