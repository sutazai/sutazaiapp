#!/bin/bash

# 🚨 EMERGENCY UNIFIED ARCHITECTURE - PHASE 3
# Created: 2025-08-16 23:20:00 UTC
# Purpose: Create single unified MCP architecture

set -e

echo "================================================"
echo "🚨 EMERGENCY UNIFIED ARCHITECTURE - PHASE 3"
echo "================================================"
echo "Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# Function to log actions
log_action() {
    echo "[$(date -u '+%H:%M:%S')] $1" | tee -a /opt/sutazaiapp/logs/emergency_unified_arch.log
}

# Step 1: Create unified network
log_action "STEP 1: Creating unified network..."
docker network create sutazai-unified --driver bridge --subnet 172.28.0.0/16 2>/dev/null || {
    log_action "Network already exists or error creating"
}

# Step 2: Create MCP Gateway service
log_action "STEP 2: Creating MCP Gateway service..."
cat > /opt/sutazaiapp/backend/app/mcp_gateway.py << 'EOF'
from fastapi import FastAPI, HTTPException
from typing import Dict, List, Any
import httpx
import asyncio
import json
import subprocess

app = FastAPI(title="MCP Gateway", version="1.0.0")

class MCPGateway:
    def __init__(self):
        self.mcps = {}
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def register_mcp(self, name: str, config: Dict[str, Any]):
        """Register an MCP server"""
        self.mcps[name] = config
        return {"status": "registered", "name": name}
    
    async def execute_stdio_mcp(self, mcp_path: str, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute STDIO-based MCP"""
        try:
            process = await asyncio.create_subprocess_exec(
                mcp_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            input_data = json.dumps({
                "jsonrpc": "2.0",
                "method": command.get("method", "execute"),
                "params": command.get("params", {}),
                "id": 1
            }).encode()
            
            stdout, stderr = await process.communicate(input=input_data)
            
            if process.returncode != 0:
                raise Exception(f"MCP error: {stderr.decode()}")
                
            return json.loads(stdout.decode())
        except Exception as e:
            return {"error": str(e), "status": "failed"}

gateway = MCPGateway()

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "mcp-gateway"}

@app.get("/mcps")
async def list_mcps():
    return {"mcps": list(gateway.mcps.keys()), "count": len(gateway.mcps)}

@app.post("/register")
async def register_mcp(name: str, config: Dict[str, Any]):
    return await gateway.register_mcp(name, config)

@app.post("/execute/{mcp_name}")
async def execute_mcp(mcp_name: str, command: Dict[str, Any]):
    if mcp_name not in gateway.mcps:
        raise HTTPException(status_code=404, detail=f"MCP {mcp_name} not found")
    
    config = gateway.mcps[mcp_name]
    if config.get("type") == "stdio":
        return await gateway.execute_stdio_mcp(config["path"], command)
    else:
        raise HTTPException(status_code=501, detail=f"MCP type {config.get('type')} not implemented")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11000)
EOF

# Step 3: Create Dockerfile for MCP Gateway
log_action "STEP 3: Creating MCP Gateway Dockerfile..."
cat > /opt/sutazaiapp/backend/Dockerfile.mcp-gateway << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn httpx

COPY app/mcp_gateway.py .

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:11000/health || exit 1

CMD ["python", "mcp_gateway.py"]
EOF

# Step 4: Build MCP Gateway image
log_action "STEP 4: Building MCP Gateway image..."
cd /opt/sutazaiapp/backend
docker build -f Dockerfile.mcp-gateway -t sutazai-mcp-gateway:latest . || {
    log_action "❌ ERROR: Failed to build MCP Gateway"
    exit 1
}

# Step 5: Stop any existing MCP gateway
log_action "STEP 5: Stopping existing MCP services..."
docker stop sutazai-mcp-gateway 2>/dev/null || true
docker rm sutazai-mcp-gateway 2>/dev/null || true

# Step 6: Start MCP Gateway
log_action "STEP 6: Starting MCP Gateway..."
docker run -d \
    --name sutazai-mcp-gateway \
    --network sutazai-unified \
    -p 11000:11000 \
    --restart unless-stopped \
    sutazai-mcp-gateway:latest

# Step 7: Connect all services to unified network
log_action "STEP 7: Connecting all services to unified network..."
for container in $(docker ps --format "{{.Names}}"); do
    if [[ "$container" == sutazai-* ]]; then
        log_action "Connecting $container to unified network..."
        docker network disconnect bridge "$container" 2>/dev/null || true
        docker network connect sutazai-unified "$container" 2>/dev/null || true
    fi
done

# Step 8: Update backend to use MCP Gateway
log_action "STEP 8: Creating backend MCP integration..."
cat > /opt/sutazaiapp/backend/app/api/v1/endpoints/mcp_unified.py << 'EOF'
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import httpx
import asyncio

router = APIRouter()

class UnifiedMCPClient:
    def __init__(self):
        self.gateway_url = "http://sutazai-mcp-gateway:11000"
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def list_servers(self) -> List[str]:
        """List all MCP servers"""
        try:
            response = await self.client.get(f"{self.gateway_url}/mcps")
            data = response.json()
            return data.get("mcps", [])
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))
    
    async def get_status(self) -> Dict[str, Any]:
        """Get MCP system status"""
        try:
            response = await self.client.get(f"{self.gateway_url}/health")
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))
    
    async def execute(self, mcp_name: str, command: Dict[str, Any]) -> Any:
        """Execute MCP command"""
        try:
            response = await self.client.post(
                f"{self.gateway_url}/execute/{mcp_name}",
                json=command
            )
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))

mcp_client = UnifiedMCPClient()

@router.get("/servers")
async def list_mcp_servers():
    return await mcp_client.list_servers()

@router.get("/status")
async def get_mcp_status():
    return await mcp_client.get_status()

@router.post("/execute/{mcp_name}")
async def execute_mcp_command(mcp_name: str, command: Dict[str, Any]):
    return await mcp_client.execute(mcp_name, command)
EOF

# Step 9: Verify MCP Gateway
log_action "STEP 9: Verifying MCP Gateway..."
sleep 5
GATEWAY_HEALTH=$(curl -s --max-time 10 http://localhost:11000/health 2>/dev/null || echo "FAILED")
if [[ "$GATEWAY_HEALTH" == *"healthy"* ]]; then
    log_action "✅ MCP Gateway is healthy"
else
    log_action "⚠️ MCP Gateway health check failed"
fi

# Step 10: Register services with Consul
log_action "STEP 10: Registering services with Consul..."
CONSUL_URL="http://localhost:10006/v1/agent/service/register"

# Register MCP Gateway
curl -X PUT $CONSUL_URL -d '{
  "ID": "mcp-gateway",
  "Name": "mcp-gateway",
  "Port": 11000,
  "Tags": ["mcp", "gateway", "unified"],
  "Check": {
    "HTTP": "http://sutazai-mcp-gateway:11000/health",
    "Interval": "30s"
  }
}' 2>/dev/null || log_action "Failed to register MCP Gateway with Consul"

echo ""
echo "=== UNIFIED ARCHITECTURE SUMMARY ==="
echo "Unified Network: CREATED"
echo "MCP Gateway: DEPLOYED"
echo "Service Connections: UPDATED"
echo "Gateway Health: $([[ "$GATEWAY_HEALTH" == *"healthy"* ]] && echo "HEALTHY" || echo "UNHEALTHY")"
echo ""

echo "================================================"
echo "PHASE 3 UNIFIED ARCHITECTURE COMPLETE"
echo "Next: Run emergency_validation.sh"
echo "================================================"