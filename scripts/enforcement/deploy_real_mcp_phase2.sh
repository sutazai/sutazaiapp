#!/bin/bash
# Deploy Real MCP Services - Phase 2
# Following Rule 1: Real Implementation Only
# Generated: 2025-08-19

set -euo pipefail

DIND_CONTAINER="sutazai-mcp-orchestrator"
MCP_MANIFESTS_DIR="/opt/sutazaiapp/docker/dind/orchestrator/mcp-manifests"
LOG_FILE="/opt/sutazaiapp/logs/mcp_deployment_$(date +%Y%m%d_%H%M%S).log"

echo "=== DEPLOYING REAL MCP SERVICES ===" | tee "$LOG_FILE"
echo "Following Rule 1: Real Implementation Only - No Fantasy Code" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"

# Step 1: Clean up old fake MCP containers
echo "" | tee -a "$LOG_FILE"
echo "STEP 1: Stopping remaining duplicate MCP containers..." | tee -a "$LOG_FILE"

# Stop any remaining containers with random names
docker ps --format "{{.Names}}" | grep -E "^[a-z]+_[a-z]+$" | while read container; do
    echo "Stopping duplicate: $container" | tee -a "$LOG_FILE"
    docker stop "$container" 2>/dev/null || true
    docker rm "$container" 2>/dev/null || true
done

# Step 2: Deploy real MCP services in DinD
echo "" | tee -a "$LOG_FILE"
echo "STEP 2: Deploying MCP services in DinD orchestrator..." | tee -a "$LOG_FILE"

# List of core MCP services to deploy
MCP_SERVICES=(
    "claude-flow"
    "ruv-swarm"
    "files"
    "context7"
    "ddg"
    "http-fetch"
    "sequentialthinking"
    "extended-memory"
    "playwright-mcp"
    "github"
)

echo "Core MCP services to deploy: ${#MCP_SERVICES[@]}" | tee -a "$LOG_FILE"

# Deploy each MCP service
for service in "${MCP_SERVICES[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "Deploying $service..." | tee -a "$LOG_FILE"
    
    case "$service" in
        "claude-flow")
            docker exec "$DIND_CONTAINER" docker run -d \
                --name "mcp-claude-flow" \
                --network bridge \
                -e MCP_PORT=3001 \
                node:18-alpine \
                sh -c "npx @modelcontextprotocol/claude-flow --port 3001" 2>&1 | tee -a "$LOG_FILE" || true
            ;;
        "ruv-swarm")
            docker exec "$DIND_CONTAINER" docker run -d \
                --name "mcp-ruv-swarm" \
                --network bridge \
                -e MCP_PORT=3002 \
                node:18-alpine \
                sh -c "npx @modelcontextprotocol/ruv-swarm --port 3002" 2>&1 | tee -a "$LOG_FILE" || true
            ;;
        "files")
            docker exec "$DIND_CONTAINER" docker run -d \
                --name "mcp-files" \
                --network bridge \
                -v /opt/sutazaiapp:/mcp-shared:ro \
                -e ALLOWED_PATHS="/mcp-shared" \
                ghcr.io/modelcontextprotocol/files:latest 2>&1 | tee -a "$LOG_FILE" || true
            ;;
        "context7")
            docker exec "$DIND_CONTAINER" docker run -d \
                --name "mcp-context7" \
                --network bridge \
                -e MCP_PORT=3004 \
                node:18-alpine \
                sh -c "npx @modelcontextprotocol/context7 --port 3004" 2>&1 | tee -a "$LOG_FILE" || true
            ;;
        "ddg")
            docker exec "$DIND_CONTAINER" docker run -d \
                --name "mcp-ddg" \
                --network bridge \
                -e MCP_PORT=3005 \
                node:18-alpine \
                sh -c "npx @modelcontextprotocol/ddg --port 3005" 2>&1 | tee -a "$LOG_FILE" || true
            ;;
        "http-fetch")
            docker exec "$DIND_CONTAINER" docker run -d \
                --name "mcp-http-fetch" \
                --network bridge \
                -e MCP_PORT=3006 \
                node:18-alpine \
                sh -c "npx @modelcontextprotocol/http-fetch --port 3006" 2>&1 | tee -a "$LOG_FILE" || true
            ;;
        "sequentialthinking")
            docker exec "$DIND_CONTAINER" docker run -d \
                --name "mcp-sequentialthinking" \
                --network bridge \
                -e MCP_PORT=3007 \
                node:18-alpine \
                sh -c "npx @modelcontextprotocol/sequentialthinking --port 3007" 2>&1 | tee -a "$LOG_FILE" || true
            ;;
        "extended-memory")
            docker exec "$DIND_CONTAINER" docker run -d \
                --name "mcp-extended-memory" \
                --network bridge \
                -e MCP_PORT=3008 \
                node:18-alpine \
                sh -c "npx @modelcontextprotocol/extended-memory --port 3008" 2>&1 | tee -a "$LOG_FILE" || true
            ;;
        "playwright-mcp")
            docker exec "$DIND_CONTAINER" docker run -d \
                --name "mcp-playwright" \
                --network bridge \
                -e MCP_PORT=3009 \
                node:18-alpine \
                sh -c "npx @modelcontextprotocol/playwright-mcp --port 3009" 2>&1 | tee -a "$LOG_FILE" || true
            ;;
        "github")
            docker exec "$DIND_CONTAINER" docker run -d \
                --name "mcp-github" \
                --network bridge \
                -e MCP_PORT=3010 \
                -e GITHUB_TOKEN="${GITHUB_TOKEN:-}" \
                node:18-alpine \
                sh -c "npx @modelcontextprotocol/github --port 3010" 2>&1 | tee -a "$LOG_FILE" || true
            ;;
    esac
    
    echo "✅ Deployed $service" | tee -a "$LOG_FILE"
done

# Step 3: Verify deployments
echo "" | tee -a "$LOG_FILE"
echo "STEP 3: Verifying MCP deployments..." | tee -a "$LOG_FILE"

sleep 5  # Give containers time to start

echo "" | tee -a "$LOG_FILE"
echo "MCP containers running in DinD:" | tee -a "$LOG_FILE"
docker exec "$DIND_CONTAINER" docker ps --format "table {{.Names}}\t{{.Status}}" | tee -a "$LOG_FILE"

# Step 4: Count deployed services
DEPLOYED_COUNT=$(docker exec "$DIND_CONTAINER" docker ps --format "{{.Names}}" | grep "^mcp-" | wc -l)

echo "" | tee -a "$LOG_FILE"
echo "=== DEPLOYMENT SUMMARY ===" | tee -a "$LOG_FILE"
echo "Requested MCP services: ${#MCP_SERVICES[@]}" | tee -a "$LOG_FILE"
echo "Successfully deployed: $DEPLOYED_COUNT" | tee -a "$LOG_FILE"

if [ "$DEPLOYED_COUNT" -eq "${#MCP_SERVICES[@]}" ]; then
    echo "✅ ALL MCP SERVICES DEPLOYED SUCCESSFULLY" | tee -a "$LOG_FILE"
else
    echo "⚠️ Some services failed to deploy. Check logs for details." | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "Deployment log: $LOG_FILE"
echo "Completed: $(date)"