#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/_common.sh"

echo "=== Debug MCP Container Cleanup ==="
echo

echo "All postgres-mcp containers:"
docker ps -a --filter ancestor=crystaldba/postgres-mcp --format "{{.Names}}\t{{.Status}}"
echo

echo "Checking labels for each container:"
docker ps -a --filter ancestor=crystaldba/postgres-mcp --format "{{.Names}}" | while read -r container_name; do
  if [ -n "$container_name" ]; then
    echo "Container: $container_name"
    mcp_service=$(docker inspect "$container_name" --format "{{index .Config.Labels \"mcp-service\"}}" 2>/dev/null || echo "")
    echo "  mcp-service label: '$mcp_service'"
    
    if [ "$mcp_service" != "postgres" ]; then
      echo "  -> This is a LEGACY container (should be cleaned up with --force)"
      
      # Check age
      created_time=$(docker inspect "$container_name" --format "{{.Created}}" 2>/dev/null || echo "")
      if [ -n "$created_time" ]; then
        created_epoch=$(date -d "$created_time" +%s 2>/dev/null || echo "0")
        current_time=$(date +%s)
        age=$((current_time - created_epoch))
        echo "  -> Age: ${age} seconds (max allowed: 3600)"
        if [ $age -gt 3600 ]; then
          echo "  -> SHOULD BE CLEANED (aged out)"
        else
          echo "  -> Too new for regular cleanup"
        fi
      fi
    else
      echo "  -> This has proper MCP labeling"
    fi
    echo
  fi
done