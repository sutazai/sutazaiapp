#!/bin/bash

echo "MCP Troubleshooting and Issue Detection"
echo "======================================="
echo ""

# Check for connection issues
echo "1. Testing all MCP server connections..."
echo ""

FAILED_SERVERS=()
SLOW_SERVERS=()

# Test each server
for server in github-mcp gitmcp-anthropic gitmcp-docs gitmcp-sutazai sequential-thinking context7 playwright; do
    echo -n "Testing $server... "
    START=$(date +%s%N)
    
    case $server in
        github-mcp)
            timeout 3 npx -y @modelcontextprotocol/server-github --version &>/dev/null && STATUS="OK" || STATUS="FAIL"
            ;;
        gitmcp-*)
            URL="https://gitmcp.io/${server#gitmcp-}"
            [ "$server" = "gitmcp-anthropic" ] && URL="https://gitmcp.io/anthropics/claude-code"
            [ "$server" = "gitmcp-sutazai" ] && URL="https://gitmcp.io/sutazai/sutazaiapp"
            timeout 3 npx -y mcp-remote "$URL" &>/dev/null && STATUS="OK" || STATUS="FAIL"
            ;;
        sequential-thinking)
            timeout 3 npx -y @modelcontextprotocol/server-sequential-thinking --version &>/dev/null && STATUS="OK" || STATUS="FAIL"
            ;;
        context7)
            timeout 3 npx -y @upstash/context7-mcp@latest --help &>/dev/null && STATUS="OK" || STATUS="FAIL"
            ;;
        playwright)
            timeout 3 npx -y @playwright/mcp@latest --version &>/dev/null && STATUS="OK" || STATUS="FAIL"
            ;;
    esac
    
    END=$(date +%s%N)
    DURATION=$((($END - $START) / 1000000))
    
    if [ "$STATUS" = "OK" ]; then
        if [ $DURATION -gt 2000 ]; then
            echo "✓ SLOW (${DURATION}ms)"
            SLOW_SERVERS+=("$server: ${DURATION}ms")
        else
            echo "✓ OK (${DURATION}ms)"
        fi
    else
        echo "✗ FAILED"
        FAILED_SERVERS+=("$server")
    fi
done

echo ""
echo "2. Checking for common issues..."
echo ""

# Check Node.js version
NODE_VERSION=$(node --version)
echo -n "Node.js version: $NODE_VERSION "
if [[ "$NODE_VERSION" =~ ^v1[68] ]] || [[ "$NODE_VERSION" =~ ^v2[0-9] ]]; then
    echo "✓"
else
    echo "⚠️ (recommended: v18+)"
fi

# Check npm version
NPM_VERSION=$(npm --version)
echo -n "npm version: $NPM_VERSION "
if [[ "${NPM_VERSION%%.*}" -ge 8 ]]; then
    echo "✓"
else
    echo "⚠️ (recommended: v8+)"
fi

# Check network connectivity
echo -n "Network connectivity: "
timeout 2 curl -s https://api.github.com > /dev/null 2>&1 && echo "✓" || echo "✗ (check internet)"

# Check disk space
DISK_USAGE=$(df -h / | tail -1 | awk '{print $5}' | sed 's/%//')
echo -n "Disk usage: ${DISK_USAGE}% "
[ $DISK_USAGE -lt 80 ] && echo "✓" || echo "⚠️ (high disk usage)"

# Check memory
MEM_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
echo -n "Memory usage: ${MEM_USAGE}% "
[ $MEM_USAGE -lt 80 ] && echo "✓" || echo "⚠️ (high memory usage)"

echo ""
echo "3. Issues Summary:"
echo "------------------"

if [ ${#FAILED_SERVERS[@]} -gt 0 ]; then
    echo "❌ Failed servers:"
    for server in "${FAILED_SERVERS[@]}"; do
        echo "   - $server"
    done
else
    echo "✅ All servers connecting"
fi

if [ ${#SLOW_SERVERS[@]} -gt 0 ]; then
    echo "⚠️ Slow servers (>2s):"
    for server in "${SLOW_SERVERS[@]}"; do
        echo "   - $server"
    done
fi

echo ""
echo "4. Recommended Actions:"
echo "-----------------------"

if [ ${#FAILED_SERVERS[@]} -gt 0 ]; then
    echo "• For failed servers, try:"
    echo "  - Check internet connectivity"
    echo "  - Clear npm cache: npm cache clean --force"
    echo "  - Update npm packages: npm update -g"
fi

if [ ${#SLOW_SERVERS[@]} -gt 0 ]; then
    echo "• For slow servers, try:"
    echo "  - Kill old processes: pkill -f 'npx.*mcp'"
    echo "  - Set NODE_OPTIONS='--max-old-space-size=512'"
    echo "  - Use offline mode: npm config set prefer-offline true"
fi

if [ $MEM_USAGE -gt 80 ]; then
    echo "• High memory usage detected:"
    echo "  - Run cleanup: /tmp/cleanup-mcp.sh"
    echo "  - Restart Claude Code"
fi
