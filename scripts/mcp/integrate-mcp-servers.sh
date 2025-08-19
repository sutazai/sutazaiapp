#!/bin/bash
# MCP Integration Script for Claude Code and Codex
# Properly configures MCP servers for STDIO communication

set -e

echo "🚀 Integrating MCP Servers for Claude Code and Codex"

# Create MCP configuration directory
MCP_CONFIG_DIR="/opt/sutazaiapp/.mcp"
mkdir -p "$MCP_CONFIG_DIR/configs"

# Create claude_desktop_config.json for MCP integration with ALL servers
cat > "$MCP_CONFIG_DIR/claude_desktop_config.json" << 'EOF'
{
  "mcpServers": {
    "claude-flow": {
      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/claude-flow.sh"
    },
    "ruv-swarm": {
      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/ruv-swarm.sh"
    },
    "files": {
      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/files.sh"
    },
    "context7": {
      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/context7.sh"
    },
    "http_fetch": {
      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/http_fetch.sh"
    },
    "ddg": {
      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/ddg.sh"
    },
    "sequentialthinking": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "mcp/sequentialthinking"]
    },
    "extended-memory": {
      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/extended-memory.sh"
    },
    "ultimatecoder": {
      "command": "python",
      "args": ["/opt/sutazaiapp/.mcp/UltimateCoderMCP/main.py"]
    },
    "playwright-mcp": {
      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/playwright-mcp.sh"
    },
    "knowledge-graph-mcp": {
      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/knowledge-graph-mcp.sh"
    },
    "compass-mcp": {
      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/compass-mcp.sh"
    },
    "github": {
      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/github.sh",
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "language-server": {
      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/language-server.sh"
    },
    "memory-bank-mcp": {
      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/memory-bank-mcp.sh"
    },
    "nx-mcp": {
      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/nx-mcp.sh"
    },
    "mcp_ssh": {
      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/mcp_ssh.sh"
    },
    "claude-task-runner": {
      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/claude-task-runner.sh"
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/opt/sutazaiapp"]
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://sutazai:${DB_PASSWORD}@localhost:10000/sutazai_db"]
    },
    "duckduckgo": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "mcp/duckduckgo"]
    },
    "fetch": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "mcp/fetch"]
    },
    "sutazai-custom": {
      "command": "node",
      "args": ["/opt/sutazaiapp/docker/mcp-services/real-mcp-server/dist/server.js"]
    }
  }
}
EOF

# Create MCP server wrapper for Claude integration
cat > "$MCP_CONFIG_DIR/mcp-stdio-wrapper.js" << 'EOF'
#!/usr/bin/env node
/**
 * MCP STDIO Wrapper for Claude Integration
 * Provides proper STDIO communication for MCP servers
 */

const { spawn } = require('child_process');
const readline = require('readline');

const serverMap = {
  'filesystem': ['npx', '-y', '@modelcontextprotocol/server-filesystem', '/opt/sutazaiapp'],
  'github': ['npx', '-y', '@modelcontextprotocol/server-github'],
  'postgres': ['npx', '-y', '@modelcontextprotocol/server-postgres', process.env.DATABASE_URL],
  'sutazai': ['node', '/opt/sutazaiapp/docker/mcp-services/real-mcp-server/dist/server.js']
};

const serverName = process.argv[2] || 'sutazai';
const serverCmd = serverMap[serverName];

if (!serverCmd) {
  console.error(`Unknown server: ${serverName}`);
  process.exit(1);
}

const server = spawn(serverCmd[0], serverCmd.slice(1), {
  stdio: ['pipe', 'pipe', 'pipe']
});

// Forward stdin to server
process.stdin.pipe(server.stdin);

// Forward server stdout to our stdout
server.stdout.pipe(process.stdout);

// Forward server stderr to our stderr
server.stderr.pipe(process.stderr);

// Handle server exit
server.on('exit', (code) => {
  process.exit(code);
});

// Handle errors
server.on('error', (err) => {
  console.error('Server error:', err);
  process.exit(1);
});
EOF

chmod +x "$MCP_CONFIG_DIR/mcp-stdio-wrapper.js"

# Install required MCP packages globally
echo "📦 Installing MCP packages..."
npm install -g \
  @modelcontextprotocol/sdk \
  @modelcontextprotocol/server-filesystem \
  @modelcontextprotocol/server-github \
  @modelcontextprotocol/server-postgres \
  @modelcontextprotocol/server-toolkit || true

# Build our custom MCP server
echo "🔨 Building custom MCP server..."
cd /opt/sutazaiapp/docker/mcp-services/real-mcp-server
npm install
npm run build

# Create systemd service for MCP registry (optional)
cat > "$MCP_CONFIG_DIR/mcp-registry.service" << 'EOF'
[Unit]
Description=MCP Registry Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/sutazaiapp/.mcp
ExecStart=/usr/bin/node /opt/sutazaiapp/.mcp/mcp-stdio-wrapper.js sutazai
Restart=always
StandardInput=tty
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Update Claude settings to include MCP servers
echo "📝 Updating Claude settings..."
CLAUDE_SETTINGS="/opt/sutazaiapp/.claude/settings.json"
if [ -f "$CLAUDE_SETTINGS" ]; then
  # Backup existing settings
  cp "$CLAUDE_SETTINGS" "${CLAUDE_SETTINGS}.backup.$(date +%Y%m%d_%H%M%S)"
  
  # Add MCP configuration using jq if available
  if command -v jq &> /dev/null; then
    jq '.mcpServers = {
      "filesystem": {
        "enabled": true,
        "path": "/opt/sutazaiapp"
      },
      "sutazai": {
        "enabled": true,
        "command": "/opt/sutazaiapp/.mcp/mcp-stdio-wrapper.js sutazai"
      }
    }' "$CLAUDE_SETTINGS" > "${CLAUDE_SETTINGS}.tmp" && mv "${CLAUDE_SETTINGS}.tmp" "$CLAUDE_SETTINGS"
  fi
fi

echo "✅ MCP servers integrated successfully!"
echo ""
echo "Available MCP servers:"
echo "  - filesystem: File system access to /opt/sutazaiapp"
echo "  - github: GitHub integration (requires GITHUB_TOKEN env var)"
echo "  - postgres: PostgreSQL database access"
echo "  - sutazai: Custom SutazAI MCP server"
echo ""
echo "To test MCP integration:"
echo "  echo '{\"jsonrpc\":\"2.0\",\"method\":\"initialize\",\"params\":{\"capabilities\":{}},\"id\":1}' | node $MCP_CONFIG_DIR/mcp-stdio-wrapper.js sutazai"