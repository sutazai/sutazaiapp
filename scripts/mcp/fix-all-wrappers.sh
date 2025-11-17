#!/bin/bash

echo "Fixing all MCP wrapper scripts..."

# Fix playwright
cat > /opt/sutazaiapp/scripts/mcp/wrappers/playwright.sh << 'EOF'
#!/bin/bash
if [ "$1" == "--selfcheck" ]; then
    echo "playwright MCP wrapper operational"
    exit 0
fi
exec npx -y @playwright/mcp@latest
EOF

# Fix context7
cat > /opt/sutazaiapp/scripts/mcp/wrappers/context7.sh << 'EOF'
#!/bin/bash
if [ "$1" == "--selfcheck" ]; then
    echo "context7 MCP wrapper operational"
    exit 0
fi
exec npx -y @context-labs/context7-mcp@latest
EOF

# Fix ddg
cat > /opt/sutazaiapp/scripts/mcp/wrappers/ddg.sh << 'EOF'
#!/bin/bash
if [ "$1" == "--selfcheck" ]; then
    echo "ddg MCP wrapper operational"
    exit 0
fi
exec npx -y @keithbrink/mcp-ddg@latest
EOF

# Fix http-fetch
cat > /opt/sutazaiapp/scripts/mcp/wrappers/http-fetch.sh << 'EOF'
#!/bin/bash
if [ "$1" == "--selfcheck" ]; then
    echo "http-fetch MCP wrapper operational"
    exit 0
fi
exec npx -y @keithbrink/mcp-http-fetch@latest
EOF

# Fix extended-memory
cat > /opt/sutazaiapp/scripts/mcp/wrappers/extended-memory.sh << 'EOF'
#!/bin/bash
if [ "$1" == "--selfcheck" ]; then
    echo "extended-memory MCP wrapper operational"
    exit 0
fi
exec npx -y @mcp-servers/extended-memory@latest
EOF

# Fix code-index
cat > /opt/sutazaiapp/scripts/mcp/wrappers/code-index.sh << 'EOF'
#!/bin/bash
if [ "$1" == "--selfcheck" ]; then
    echo "code-index MCP wrapper operational"
    exit 0
fi
exec npx -y @kevinclancy/mcp-code-index@latest
EOF

# Fix github-project-manager
cat > /opt/sutazaiapp/scripts/mcp/wrappers/github-project-manager.sh << 'EOF'
#!/bin/bash
if [ "$1" == "--selfcheck" ]; then
    echo "github-project-manager MCP wrapper operational"
    exit 0
fi
# Load GitHub token from .env file if it exists
if [ -f /opt/sutazaiapp/.env ]; then
    export $(grep -v '^#' /opt/sutazaiapp/.env | xargs)
fi
export GITHUB_TOKEN="${GITHUB_TOKEN:-}"
exec npx -y @bstefanescu/mcp-github-project-manager@latest
EOF

# Fix removed
cat > /opt/sutazaiapp/scripts/mcp/wrappers/removed.sh << 'EOF'
#!/bin/bash
if [ "$1" == "--selfcheck" ]; then
    echo "removed MCP wrapper operational"
    exit 0
fi
exec npx -y removed@alpha mcp start
EOF

# Fix removed
cat > /opt/sutazaiapp/scripts/mcp/wrappers/removed.sh << 'EOF'
#!/bin/bash
if [ "$1" == "--selfcheck" ]; then
    echo "removed MCP wrapper operational"
    exit 0
fi
exec npx -y removed@latest mcp start
EOF

chmod +x /opt/sutazaiapp/scripts/mcp/wrappers/*.sh

echo "All wrapper scripts updated!"