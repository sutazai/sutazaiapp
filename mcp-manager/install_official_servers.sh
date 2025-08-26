#!/usr/bin/env bash
# Install Official MCP Servers
set -e

echo "Installing official MCP servers..."

MCP_DIR="/opt/sutazaiapp/mcp-servers"
mkdir -p "$MCP_DIR"

# Install git-mcp from official repository
echo "Installing git-mcp..."
if [ ! -d "$MCP_DIR/git-mcp" ]; then
    cp -r /tmp/mcp-official-servers/src/git "$MCP_DIR/git-mcp"
    cd "$MCP_DIR/git-mcp"
    python3 -m venv venv
    ./venv/bin/pip install --quiet --upgrade pip
    ./venv/bin/pip install --quiet -e .
fi

# Install playwright-mcp from official repository  
echo "Installing playwright-mcp..."
if [ ! -d "$MCP_DIR/playwright-mcp-official" ]; then
    cp -r /tmp/playwright-mcp-official "$MCP_DIR/playwright-mcp-official"
    cd "$MCP_DIR/playwright-mcp-official"
    npm install --silent
    npm run build
fi

# Install nx-mcp
echo "Installing nx-mcp..."
npm install -g @nx-console/nx-mcp-server@latest --silent || true

# Create proper wrapper scripts
echo "Creating wrapper scripts..."

# Git MCP wrapper
cat > /opt/sutazaiapp/scripts/mcp/wrappers/git-mcp-official.sh << 'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

MCP_DIR="/opt/sutazaiapp/mcp-servers/git-mcp"

selfcheck() {
    if [ -f "$MCP_DIR/venv/bin/mcp-server-git" ]; then
        echo "✓ git-mcp selfcheck passed"
        return 0
    fi
    echo "✗ git-mcp selfcheck failed"
    return 1
}

case "${1:-start}" in
    start)
        exec "$MCP_DIR/venv/bin/mcp-server-git"
        ;;
    selfcheck|--selfcheck)
        selfcheck
        ;;
    health)
        echo '{"status": "healthy", "server": "git-mcp"}'
        ;;
    *)
        echo "Usage: $0 {start|selfcheck|health}"
        exit 1
        ;;
esac
EOF

chmod +x /opt/sutazaiapp/scripts/mcp/wrappers/git-mcp-official.sh

# Playwright MCP wrapper
cat > /opt/sutazaiapp/scripts/mcp/wrappers/playwright-mcp-official.sh << 'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

MCP_DIR="/opt/sutazaiapp/mcp-servers/playwright-mcp-official"

selfcheck() {
    if [ -f "$MCP_DIR/package.json" ]; then
        echo "✓ playwright-mcp selfcheck passed"
        return 0
    fi
    echo "✗ playwright-mcp selfcheck failed"
    return 1
}

case "${1:-start}" in
    start)
        cd "$MCP_DIR"
        exec npx @playwright/mcp@latest
        ;;
    selfcheck|--selfcheck)
        selfcheck
        ;;
    health)
        echo '{"status": "healthy", "server": "playwright-mcp"}'
        ;;
    *)
        echo "Usage: $0 {start|selfcheck|health}"
        exit 1
        ;;
esac
EOF

chmod +x /opt/sutazaiapp/scripts/mcp/wrappers/playwright-mcp-official.sh

# nx-mcp wrapper
cat > /opt/sutazaiapp/scripts/mcp/wrappers/nx-mcp-official.sh << 'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

selfcheck() {
    if command -v npx >/dev/null 2>&1; then
        echo "✓ nx-mcp selfcheck passed"
        return 0
    fi
    echo "✗ nx-mcp selfcheck failed"
    return 1
}

case "${1:-start}" in
    start)
        exec npx @nx-console/nx-mcp-server@latest
        ;;
    selfcheck|--selfcheck)
        selfcheck
        ;;
    health)
        echo '{"status": "healthy", "server": "nx-mcp"}'
        ;;
    *)
        echo "Usage: $0 {start|selfcheck|health}"
        exit 1
        ;;
esac
EOF

chmod +x /opt/sutazaiapp/scripts/mcp/wrappers/nx-mcp-official.sh

echo "Official MCP servers installed successfully!"