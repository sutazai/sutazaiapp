#!/usr/bin/env bash
# Install and setup the MCP Manager system

set -e

echo "════════════════════════════════════════════════════════════════"
echo "           MCP Manager Installation"  
echo "════════════════════════════════════════════════════════════════"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Install Python dependencies
echo -e "\n${YELLOW}Installing Python dependencies...${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "/opt/sutazaiapp/mcp-manager/venv" ]; then
    python3 -m venv /opt/sutazaiapp/mcp-manager/venv
fi

# Activate and install dependencies
source /opt/sutazaiapp/mcp-manager/venv/bin/activate

# Install the official MCP SDK
pip install --quiet --upgrade pip
pip install --quiet mcp

# Check if MCP SDK is installed
if python -c "import mcp" 2>/dev/null; then
    echo -e "${GREEN}✓ MCP SDK installed successfully${NC}"
else
    echo -e "${RED}✗ Failed to install MCP SDK${NC}"
    exit 1
fi

# Fix the claude-task-runner
echo -e "\n${YELLOW}Fixing claude-task-runner...${NC}"

# Install dependencies in claude-task-runner venv
if [ -d "/opt/sutazaiapp/mcp-servers/claude-task-runner/venv" ]; then
    # Ensure the venv has proper permissions
    chmod +x /opt/sutazaiapp/mcp-servers/claude-task-runner/venv/bin/python 2>/dev/null || true
    chmod +x /opt/sutazaiapp/mcp-servers/claude-task-runner/venv/bin/pip 2>/dev/null || true
    /opt/sutazaiapp/mcp-servers/claude-task-runner/venv/bin/pip install --quiet mcp 2>/dev/null || echo -e "${YELLOW}⚠ Could not update claude-task-runner venv${NC}"
    echo -e "${GREEN}✓ Processed claude-task-runner dependencies${NC}"
fi

# Update the wrapper to use our fixed version
cat > /opt/sutazaiapp/scripts/mcp/wrappers/claude-task-runner-fixed.sh << 'EOF'
#!/usr/bin/env bash
# Fixed wrapper for claude-task-runner using official MCP SDK

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCP_MANAGER_DIR="/opt/sutazaiapp/mcp-manager"

# Selfcheck function
selfcheck() {
    echo "Performing selfcheck for claude-task-runner (fixed)..."
    
    if [ -f "$MCP_MANAGER_DIR/fixed_claude_task_runner.py" ]; then
        if "$MCP_MANAGER_DIR/venv/bin/python" "$MCP_MANAGER_DIR/fixed_claude_task_runner.py" health >/dev/null 2>&1; then
            echo "✓ claude-task-runner (fixed) selfcheck passed"
            return 0
        fi
    fi
    
    echo "✗ claude-task-runner (fixed) selfcheck failed"
    return 1
}

# Start function
start_mcp() {
    exec "$MCP_MANAGER_DIR/venv/bin/python" "$MCP_MANAGER_DIR/fixed_claude_task_runner.py" start
}

# Main command handling
case "${1:-start}" in
    start)
        start_mcp
        ;;
    selfcheck|--selfcheck)
        selfcheck
        ;;
    health)
        "$MCP_MANAGER_DIR/venv/bin/python" "$MCP_MANAGER_DIR/fixed_claude_task_runner.py" health
        ;;
    *)
        echo "Usage: $0 {start|selfcheck|health}"
        exit 1
        ;;
esac
EOF

chmod +x /opt/sutazaiapp/scripts/mcp/wrappers/claude-task-runner-fixed.sh

# Create a simple CLI wrapper
cat > /opt/sutazaiapp/mcp-manager/mcp << 'EOF'
#!/usr/bin/env bash
# MCP Manager CLI

exec /opt/sutazaiapp/mcp-manager/venv/bin/python /opt/sutazaiapp/mcp-manager/manager.py "$@"
EOF

chmod +x /opt/sutazaiapp/mcp-manager/mcp

# Add to PATH
if ! grep -q "/opt/sutazaiapp/mcp-manager" ~/.bashrc; then
    echo "export PATH=\$PATH:/opt/sutazaiapp/mcp-manager" >> ~/.bashrc
fi

echo -e "\n${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}           Installation Complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"

echo -e "\nUsage:"
echo "  mcp status     - Check status of all MCP servers"
echo "  mcp health     - Run health checks"
echo "  mcp list       - List all available servers"
echo "  mcp fix        - Apply fixes to failing servers"

echo -e "\nTo use the fixed claude-task-runner, update .mcp.json to use:"
echo "  /opt/sutazaiapp/scripts/mcp/wrappers/claude-task-runner-fixed.sh"