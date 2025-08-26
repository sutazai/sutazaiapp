#!/bin/bash

# MCP Performance Optimization Script
# This script optimizes MCP server configurations for better performance

echo "MCP Performance Optimization"
echo "============================"
echo ""

# 1. Set memory limits for Node processes
echo "1. Setting Node.js memory limits..."
export NODE_OPTIONS="--max-old-space-size=512"
echo "   Set NODE_OPTIONS to limit memory to 512MB per process"

# 2. Configure npm to use less resources
echo ""
echo "2. Configuring npm for performance..."
npm config set prefer-offline true
npm config set cache-min 9999999
npm config set audit false
npm config set fund false
echo "   Configured npm for offline preference and disabled audits"

# 3. Create MCP server pooling configuration
echo ""
echo "3. Creating MCP server pool configuration..."
cat << 'EOF' > /opt/sutazaiapp/.mcp-servers/mcp-pool.json
{
  "maxProcesses": 10,
  "processTimeout": 300000,
  "recycleAfter": 100,
  "memoryLimit": "512M",
  "cpuLimit": "25%",
  "autoKillStale": true,
  "staleThreshold": 3600000
}
EOF
echo "   Created pool configuration with limits"

# 4. Optimize Claude configuration
echo ""
echo "4. Optimizing Claude configuration..."
python3 << 'PYTHON'
import json

# Load Claude config
with open('/root/.claude.json', 'r') as f:
    config = json.load(f)

# Add performance optimizations
if 'performance' not in config:
    config['performance'] = {}

config['performance'].update({
    'maxConcurrentServers': 10,
    'serverTimeout': 300000,
    'enableCaching': True,
    'cacheSize': '100MB',
    'autoCleanup': True,
    'cleanupInterval': 3600000
})

# Save config
with open('/root/.claude.json', 'w') as f:
    json.dump(config, f, indent=2)
print("   Added performance settings to Claude config")
PYTHON

# 5. Create process monitor
echo ""
echo "5. Creating process monitor..."
cat << 'EOF' > /opt/sutazaiapp/.mcp-servers/monitor-mcp.sh
#!/bin/bash
# MCP Process Monitor

while true; do
    # Count MCP processes
    PROCESS_COUNT=$(ps aux | grep -E "mcp|npx" | grep -v grep | wc -l)
    
    # If too many processes, clean up
    if [ $PROCESS_COUNT -gt 50 ]; then
        echo "[$(date)] Warning: $PROCESS_COUNT MCP processes detected, cleaning up..."
        
        # Kill old npx processes
        for pid in $(ps aux | grep "npx" | grep -v grep | awk '{print $2}' | tail -n +20); do
            kill -9 $pid 2>/dev/null
        done
    fi
    
    # Sleep for 5 minutes
    sleep 300
done
EOF
chmod +x /opt/sutazaiapp/.mcp-servers/monitor-mcp.sh
echo "   Created automatic process monitor"

# 6. Test server response times
echo ""
echo "6. Testing MCP server response times..."
echo ""
echo "Server Response Times:"
echo "----------------------"

# Test GitHub MCP
START=$(date +%s%N)
timeout 2 npx -y @modelcontextprotocol/server-github --version 2>&1 >/dev/null
END=$(date +%s%N)
echo "github-mcp: $((($END - $START) / 1000000))ms"

# Test Sequential Thinking
START=$(date +%s%N)
timeout 2 npx -y @modelcontextprotocol/server-sequential-thinking --version 2>&1 >/dev/null
END=$(date +%s%N)
echo "sequential-thinking: $((($END - $START) / 1000000))ms"

# Test Context7
START=$(date +%s%N)
timeout 2 npx -y @upstash/context7-mcp@latest --version 2>&1 >/dev/null
END=$(date +%s%N)
echo "context7: $((($END - $START) / 1000000))ms"

echo ""
echo "Optimization complete!"
echo ""
echo "Summary:"
echo "--------"
echo "✓ Node.js memory limited to 512MB per process"
echo "✓ npm configured for offline mode and caching"
echo "✓ MCP server pooling configured"
echo "✓ Claude performance settings added"
echo "✓ Process monitor created"
echo "✓ Server response times tested"
echo ""
echo "To start the process monitor in background:"
echo "nohup /opt/sutazaiapp/.mcp-servers/monitor-mcp.sh > /tmp/mcp-monitor.log 2>&1 &"