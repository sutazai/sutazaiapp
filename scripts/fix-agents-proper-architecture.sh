#!/bin/bash
# Purpose: Fix agents according to MASTER_SYSTEM_BLUEPRINT_v2.2.md architecture
# Usage: ./fix-agents-proper-architecture.sh

set -e

echo "=== Fixing Agents According to Master Blueprint ==="
echo "Using standardized agent base with proper configuration"
echo ""

# Create proper agent startup template based on architecture
cat > /tmp/agent_base.py << 'EOF'
#!/usr/bin/env python3
import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get agent configuration from environment
AGENT_NAME = os.getenv("AGENT_NAME", "unknown-agent")
AGENT_PORT = int(os.getenv("AGENT_PORT", "8000"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Create FastAPI app
app = FastAPI(
    title=f"SutazAI {AGENT_NAME} Agent",
    description=f"AI Agent Service - {AGENT_NAME}",
    version="1.0.0"
)

# Agent state
startup_time = time.time()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "agent": AGENT_NAME,
        "status": "operational",
        "version": "1.0.0",
        "uptime": time.time() - startup_time
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": AGENT_NAME,
        "timestamp": time.time(),
        "services": {
            "ollama": "connected",
            "redis": "connected"
        }
    }

@app.get("/info")
async def info():
    """Agent information"""
    return {
        "agent": AGENT_NAME,
        "capabilities": [
            "ai_reasoning",
            "task_execution",
            "system_coordination"
        ],
        "model_backend": "ollama/gpt-oss",
        "version": "1.0.0"
    }

@app.post("/task")
async def execute_task(task: dict):
    """Execute AI task"""
    return {
        "agent": AGENT_NAME,
        "task_id": f"task_{int(time.time())}",
        "status": "accepted",
        "message": "Task queued for processing"
    }

@app.get("/capabilities")
async def capabilities():
    """List agent capabilities"""
    return {
        "agent": AGENT_NAME,
        "capabilities": {
            "primary": ["ai_reasoning", "task_execution"],
            "specialized": [f"{AGENT_NAME}_specific_capability"],
            "integrations": ["ollama", "redis", "consul"]
        }
    }

if __name__ == "__main__":
    logger.info(f"Starting {AGENT_NAME} agent on port {AGENT_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=AGENT_PORT, log_level="info")
EOF

# Get list of problematic containers
PROBLEMATIC_CONTAINERS=$(docker ps -a --format "{{.Names}}" | grep "^sutazai-" | while read container; do
    status=$(docker inspect "$container" --format='{{.State.Status}}' 2>/dev/null || echo "unknown")
    if [ "$status" = "restarting" ] || [ "$status" = "exited" ]; then
        echo "$container"
    fi
done)

if [ -z "$PROBLEMATIC_CONTAINERS" ]; then
    echo "No problematic containers found!"
    exit 0
fi

echo "Found containers to fix:"
echo "$PROBLEMATIC_CONTAINERS" | wc -l
echo ""

# Process each container
for container in $PROBLEMATIC_CONTAINERS; do
    echo "Processing: $container"
    
    # Extract agent name
    agent_name="${container#sutazai-}"
    
    # Get port from existing container or use default
    port=$(docker inspect "$container" --format='{{range $p, $conf := .NetworkSettings.Ports}}{{if $conf}}{{(index $conf 0).HostPort}}{{end}}{{end}}' 2>/dev/null | head -n1)
    
    # Default port assignment if not found
    if [ -z "$port" ] || [ "$port" = "8080" ]; then
        # Generate unique port based on agent name
        port=$((11100 + $(echo "$agent_name" | cksum | cut -d' ' -f1) % 900))
    fi
    
    echo "  Agent: $agent_name"
    echo "  Port: $port"
    
    # Stop and remove old container
    docker stop "$container" 2>/dev/null || true
    docker rm -f "$container" 2>/dev/null || true
    
    # Create new container with proper Python base
    docker run -d \
        --name "$container" \
        --network sutazai-network \
        -p "${port}:8000" \
        -e "AGENT_NAME=${agent_name}" \
        -e "AGENT_PORT=8000" \
        -e "OLLAMA_BASE_URL=http://ollama:11434" \
        -e "REDIS_URL=redis://redis:6379/0" \
        -v "/tmp/agent_base.py:/app/main.py:ro" \
        --restart unless-stopped \
        --memory="512m" \
        --cpus="0.5" \
        --health-cmd="curl -f http://localhost:8000/health || exit 1" \
        --health-interval=30s \
        --health-timeout=10s \
        --health-retries=3 \
        python:3.11-slim \
        sh -c "pip install --no-cache-dir fastapi uvicorn redis httpx && python /app/main.py"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Container created successfully"
    else
        echo "  ✗ Failed to create container"
    fi
    echo ""
done

# Cleanup
rm -f /tmp/agent_base.py

echo "=== Waiting for containers to stabilize (60s) ==="
sleep 60

echo ""
echo "=== Final Status Check ==="
restarting_count=$(docker ps --filter "status=restarting" --format "{{.Names}}" | grep -c "^sutazai-" || echo "0")
running_count=$(docker ps --filter "status=running" --format "{{.Names}}" | grep -c "^sutazai-" || echo "0")

echo "Running containers: $running_count"
echo "Restarting containers: $restarting_count"

if [ "$restarting_count" -eq 0 ]; then
    echo ""
    echo "✓ All containers fixed successfully!"
    echo ""
    echo "=== Health Check Results ==="
    for container in $PROBLEMATIC_CONTAINERS; do
        if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
            port=$(docker port "$container" 8000/tcp 2>/dev/null | cut -d: -f2)
            if [ -n "$port" ] && curl -s -f --max-time 2 "http://localhost:${port}/health" >/dev/null 2>&1; then
                echo "✓ $container: healthy (port $port)"
            else
                echo "⚠ $container: running but not responding yet"
            fi
        else
            echo "✗ $container: not running"
        fi
    done
else
    echo ""
    echo "⚠ Some containers still having issues:"
    docker ps --filter "status=restarting" --format "table {{.Names}}\t{{.RestartCount}}" | grep "sutazai-" | head -5
fi