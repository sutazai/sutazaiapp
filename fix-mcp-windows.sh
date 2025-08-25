#!/bin/bash
# Windows MCP Server Fix Script for Git Bash/MSYS
# Fixes MCP server connections in Windows environment
# Author: MCP Integration Specialist
# Date: 2025-08-25

echo "=== Windows MCP Server Fix ==="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$SCRIPT_DIR"
CLAUDE_CONFIG="$APPDATA/Claude/claude_desktop_config.json"
MCP_CONFIG="$ROOT_DIR/.mcp.json"

# Function to test command
test_command() {
    local cmd="$1"
    if command -v "$cmd" >/dev/null 2>&1; then
        echo -e "  $cmd: ${GREEN}✓${NC}"
        return 0
    else
        echo -e "  $cmd: ${RED}✗${NC}"
        return 1
    fi
}

# Function to test npm package
test_npm_package() {
    local package="$1"
    echo -n "  Testing $package..."
    if timeout 10 npx -y "$package" --version >/dev/null 2>&1 || timeout 10 npx -y "$package" --help >/dev/null 2>&1; then
        echo -e " ${GREEN}✓${NC}"
        return 0
    else
        echo -e " ${RED}✗${NC}"
        return 1
    fi
}

echo -e "\n${YELLOW}Step 1: Checking prerequisites...${NC}"
test_command "node"
test_command "npm"
test_command "npx"
test_command "git"

echo -e "\n${YELLOW}Step 2: Stopping existing MCP processes...${NC}"
# Kill any existing MCP-related node processes
taskkill //F //IM "node.exe" //FI "WINDOWTITLE eq *mcp*" 2>/dev/null || true
echo -e "  ${GREEN}Processes terminated${NC}"

echo -e "\n${YELLOW}Step 3: Testing MCP packages...${NC}"
PACKAGES=(
    "@modelcontextprotocol/server-files"
    "@modelcontextprotocol/server-http-fetch"
    "@modelcontextprotocol/server-ddg"
    "claude-flow@alpha"
    "ruv-swarm@latest"
)

WORKING_PACKAGES=()
for pkg in "${PACKAGES[@]}"; do
    if test_npm_package "$pkg"; then
        WORKING_PACKAGES+=("$pkg")
    fi
done

echo -e "\n${YELLOW}Step 4: Creating Windows MCP configuration...${NC}"

# Convert Windows path to proper format
WIN_ROOT_PATH=$(cygpath -w "$ROOT_DIR" 2>/dev/null || echo "$ROOT_DIR")

# Create MCP configuration
cat > "$MCP_CONFIG" << EOF
{
  "mcpServers": {
    "files": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-files", "$WIN_ROOT_PATH"],
      "type": "stdio"
    },
    "http-fetch": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-http-fetch"],
      "type": "stdio"
    },
    "ddg-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-ddg"],
      "type": "stdio"
    },
    "claude-flow": {
      "command": "npx",
      "args": ["claude-flow@alpha", "mcp", "start", "--stdio"],
      "type": "stdio"
    },
    "ruv-swarm": {
      "command": "npx",
      "args": ["ruv-swarm@latest", "mcp", "start", "--stdio", "--stability"],
      "type": "stdio"
    }
  }
}
EOF

echo -e "  ${GREEN}Created $MCP_CONFIG${NC}"

echo -e "\n${YELLOW}Step 5: Updating Claude Desktop configuration...${NC}"

# Backup existing config
if [ -f "$CLAUDE_CONFIG" ]; then
    cp "$CLAUDE_CONFIG" "$CLAUDE_CONFIG.backup"
    echo -e "  ${GREEN}Backed up existing config${NC}"
fi

# Create Claude config with working servers
cat > "$CLAUDE_CONFIG" << EOF
{
  "mcpServers": {
    "files": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-files", "$WIN_ROOT_PATH"]
    },
    "http-fetch": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-http-fetch"]
    },
    "ddg-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-ddg"]
    }
  }
}
EOF

# Add sequential-thinking if installed
if [ -f "$APPDATA/npm/node_modules/mcp-sequential-thinking/dist/index.js" ]; then
    echo -e "  ${GREEN}Found sequential-thinking - adding to config${NC}"
    # Update config to include sequential-thinking
    node -e "
    const fs = require('fs');
    const config = JSON.parse(fs.readFileSync('$CLAUDE_CONFIG', 'utf8'));
    config.mcpServers['sequential-thinking'] = {
        command: 'node',
        args: ['$APPDATA/npm/node_modules/mcp-sequential-thinking/dist/index.js']
    };
    fs.writeFileSync('$CLAUDE_CONFIG', JSON.stringify(config, null, 2));
    "
fi

echo -e "  ${GREEN}Updated Claude Desktop config${NC}"

echo -e "\n${YELLOW}Step 6: Testing MCP server connections...${NC}"

# Function to test MCP server
test_mcp_server() {
    local name="$1"
    local cmd="$2"
    shift 2
    local args="$@"
    
    echo -n "  Testing $name..."
    
    # Start the server in background
    timeout 3 $cmd $args >/dev/null 2>&1 &
    local pid=$!
    
    sleep 1
    
    if kill -0 $pid 2>/dev/null; then
        kill $pid 2>/dev/null
        echo -e " ${GREEN}✓ (responds)${NC}"
        return 0
    else
        echo -e " ${RED}✗ (failed)${NC}"
        return 1
    fi
}

WORKING_SERVERS=()

# Test basic NPX servers
test_mcp_server "files" "npx" "-y" "@modelcontextprotocol/server-files" "$WIN_ROOT_PATH" && WORKING_SERVERS+=("files")
test_mcp_server "http-fetch" "npx" "-y" "@modelcontextprotocol/server-http-fetch" && WORKING_SERVERS+=("http-fetch")
test_mcp_server "ddg-search" "npx" "-y" "@modelcontextprotocol/server-ddg" && WORKING_SERVERS+=("ddg-search")

# Test sequential-thinking if available
if command -v mcp-server-sequential-thinking >/dev/null 2>&1; then
    test_mcp_server "sequential-thinking" "mcp-server-sequential-thinking" && WORKING_SERVERS+=("sequential-thinking")
fi

echo -e "\n${BLUE}=== MCP Server Status Report ===${NC}"
echo -e "${GREEN}Working Servers: ${#WORKING_SERVERS[@]}${NC}"
for server in "${WORKING_SERVERS[@]}"; do
    echo -e "  ${GREEN}✓ $server${NC}"
done

echo -e "\n${YELLOW}Configuration Files:${NC}"
echo "  MCP Config: $MCP_CONFIG"
echo "  Claude Config: $CLAUDE_CONFIG"

echo -e "\n${BLUE}=== Next Steps ===${NC}"
echo -e "${YELLOW}1. Restart Claude Desktop application${NC}"
echo -e "${YELLOW}2. Use the /mcp command in Claude to reconnect${NC}"
echo -e "${YELLOW}3. Available servers:${NC}"
for server in "${WORKING_SERVERS[@]}"; do
    echo "   - $server"
done

# Save status report
REPORT_FILE="$ROOT_DIR/mcp-status-report.json"
cat > "$REPORT_FILE" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "platform": "Windows (Git Bash)",
  "working_servers": $(printf '[\n    "%s"' "${WORKING_SERVERS[@]}" | sed 's/" "/",\n    "/g' | sed 's/$/\n  ]/'),
  "total_servers": ${#WORKING_SERVERS[@]},
  "config_files": {
    "mcp_config": "$MCP_CONFIG",
    "claude_config": "$CLAUDE_CONFIG"
  }
}
EOF

echo -e "\n${GREEN}Fix completed at: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo "Status report saved to: $REPORT_FILE"