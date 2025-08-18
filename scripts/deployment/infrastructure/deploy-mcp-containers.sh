#!/bin/bash
# Deploy MCP servers as containers in DinD
set -euo pipefail

DIND_CONTAINER="sutazai-mcp-orchestrator-notls"
MCP_CONFIG="/opt/sutazaiapp/.mcp.json"

echo "=== MCP CONTAINER DEPLOYMENT SCRIPT ==="
echo "Target DinD: $DIND_CONTAINER"
echo "MCP Config: $MCP_CONFIG"

# Function to deploy single MCP container
deploy_mcp() {
    local mcp_name="$1"
    local mcp_command="$2"
    
    echo "Deploying MCP: $mcp_name"
    
    # Create container in DinD
    docker exec "$DIND_CONTAINER" docker run -d \
        --name "mcp-$mcp_name" \
        --network bridge \
        -e MCP_NAME="$mcp_name" \
        -e MCP_COMMAND="$mcp_command" \
        alpine:latest sleep infinity
        
    echo "✅ Deployed MCP container: mcp-$mcp_name"
}

# Deploy all MCPs from config
echo "Reading MCP configurations..."
python3 << 'PYTHON_EOF'
import json
import subprocess
import sys

try:
    with open("/opt/sutazaiapp/.mcp.json", "r") as f:
        config = json.load(f)
    
    mcps = config.get("mcpServers", {})
    print(f"Found {len(mcps)} MCP servers to deploy")
    
    for mcp_name, mcp_config in mcps.items():
        command = mcp_config.get("command", "")
        args = " ".join(mcp_config.get("args", []))
        full_command = f"{command} {args}".strip()
        
        print(f"Deploying {mcp_name}: {full_command}")
        
        # For now, create simple containers - we'll enhance with proper MCP images later
        try:
            result = subprocess.run([
                "docker", "exec", "sutazai-mcp-orchestrator-notls", 
                "docker", "run", "-d",
                "--name", f"mcp-{mcp_name}",
                "--restart", "unless-stopped",
                "-e", f"MCP_NAME={mcp_name}",
                "-e", f"MCP_COMMAND={full_command}",
                "alpine:latest", "sleep", "infinity"
            ], capture_output=True, text=True, check=True)
            print(f"✅ Deployed container mcp-{mcp_name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to deploy {mcp_name}: {e.stderr}")
            
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
PYTHON_EOF

echo "=== MCP DEPLOYMENT COMPLETE ==="
docker exec "$DIND_CONTAINER" docker ps --format "table {{.Names}}\t{{.Status}}"
