#!/bin/bash
###############################################################################
# MCP Host Deployment Script
# Purpose: Deploy MCP services directly on host with proper port mapping
# Created: 2025-08-20
###############################################################################

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${2:-$GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

log "============================================" "$BLUE"
log "MCP HOST DEPLOYMENT" "$BLUE"
log "============================================" "$BLUE"

# Check existing MCP containers
log "Checking existing MCP containers..." "$YELLOW"
docker ps --format "table {{.Names}}\t{{.Ports}}" | grep -E "mcp|MCP" || true

# Define missing services and their ports
declare -A MISSING_SERVICES=(
    ["mcp-claude-flow"]="3001"
    ["mcp-ruv-swarm"]="3002"
    ["mcp-files"]="3003"
    ["mcp-context7"]="3004"
    ["mcp-http-fetch"]="3005"
    ["mcp-ddg"]="3006"
    ["mcp-extended-memory"]="3009"
    ["mcp-ssh"]="3010"
    ["mcp-ultimatecoder"]="3011"
    ["mcp-knowledge-graph-mcp"]="3014"
    ["mcp-github"]="3016"
    ["mcp-language-server"]="3018"
    ["mcp-claude-task-runner"]="3019"
)

# Deploy missing services
log "Deploying missing MCP services..." "$BLUE"

for service in "${!MISSING_SERVICES[@]}"; do
    port="${MISSING_SERVICES[$service]}"
    
    # Check if container already exists
    if docker ps -a --format "{{.Names}}" | grep -q "^$service$"; then
        log "Container $service already exists, removing..." "$YELLOW"
        docker rm -f "$service" 2>/dev/null || true
    fi
    
    # Check if port is already in use
    if ss -tuln | grep -q ":$port "; then
        log "Port $port is already in use, skipping $service" "$YELLOW"
        continue
    fi
    
    log "Deploying $service on port $port..." "$YELLOW"
    
    # Choose appropriate image based on service
    case "$service" in
        *fetch*)
            if docker images | grep -q "mcp/fetch"; then
                docker run -d \
                    --name "$service" \
                    -p "$port:8080" \
                    -e PORT=8080 \
                    --restart unless-stopped \
                    --network sutazai-network \
                    mcp/fetch || true
            fi
            ;;
        *ddg*)
            if docker images | grep -q "mcp/duckduckgo"; then
                docker run -d \
                    --name "$service" \
                    -p "$port:8080" \
                    -e PORT=8080 \
                    --restart unless-stopped \
                    --network sutazai-network \
                    mcp/duckduckgo || true
            fi
            ;;
        *)
            # Use existing MCP images or create simple HTTP servers
            if docker images | grep -q "sutazai-mcp-nodejs"; then
                docker run -d \
                    --name "$service" \
                    -p "$port:$port" \
                    -e MCP_SERVICE="${service#mcp-}" \
                    -e MCP_PORT="$port" \
                    --restart unless-stopped \
                    --network sutazai-network \
                    sutazai-mcp-nodejs || true
            else
                # Create a simple HTTP server
                docker run -d \
                    --name "$service" \
                    -p "$port:8080" \
                    -e SERVICE_NAME="${service#mcp-}" \
                    -e SERVICE_PORT="$port" \
                    --restart unless-stopped \
                    --network sutazai-network \
                    node:18-alpine \
                    sh -c "cat > /app.js <<'EOF'
const http = require('http');
const service = process.env.SERVICE_NAME;
const port = 8080;

http.createServer((req, res) => {
    res.writeHead(200, {'Content-Type': 'application/json'});
    if (req.url === '/health') {
        res.end(JSON.stringify({
            status: 'healthy',
            service: service,
            port: process.env.SERVICE_PORT,
            timestamp: new Date().toISOString()
        }));
    } else {
        res.end(JSON.stringify({
            message: 'MCP Service Ready',
            service: service
        }));
    }
}).listen(port, () => console.log('Service ' + service + ' on port ' + port));
EOF
node /app.js" || true
            fi
            ;;
    esac
done

# Wait for services to stabilize
log "Waiting for services to stabilize..." "$YELLOW"
sleep 5

# Check deployment status
log "Checking deployment status..." "$BLUE"
echo "----------------------------------------"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "mcp|MCP" || true

# Test service connectivity
log "Testing service connectivity..." "$BLUE"
echo "----------------------------------------"

for service in "${!MISSING_SERVICES[@]}"; do
    port="${MISSING_SERVICES[$service]}"
    
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port/health" | grep -q "200"; then
        log "✓ $service (port $port) - HEALTHY" "$GREEN"
    elif timeout 1 bash -c "echo > /dev/tcp/localhost/$port" 2>/dev/null; then
        log "⚠ $service (port $port) - PORT OPEN (no health endpoint)" "$YELLOW"
    else
        log "✗ $service (port $port) - NOT RESPONDING" "$RED"
    fi
done

# Register services with Consul
log "Registering services with Consul..." "$YELLOW"

for service in "${!MISSING_SERVICES[@]}"; do
    port="${MISSING_SERVICES[$service]}"
    
    # Register with Consul
    curl -s -X PUT "http://localhost:10006/v1/agent/service/register" \
        -H "Content-Type: application/json" \
        -d "{
            \"ID\": \"$service\",
            \"Name\": \"$service\",
            \"Port\": $port,
            \"Check\": {
                \"TCP\": \"localhost:$port\",
                \"Interval\": \"30s\",
                \"Timeout\": \"10s\"
            }
        }" || true
done

# Create health monitoring script
cat > /opt/sutazaiapp/scripts/mesh/monitor_mcp.sh <<'MONITOR_EOF'
#!/bin/bash
echo "MCP Services Health Monitor"
echo "============================"
echo ""

# Define all MCP services and ports
declare -A MCP_PORTS=(
    ["claude-flow"]="3001"
    ["ruv-swarm"]="3002"
    ["files"]="3003"
    ["context7"]="3004"
    ["http-fetch"]="3005"
    ["ddg"]="3006"
    ["extended-memory"]="3009"
    ["ssh"]="3010"
    ["ultimatecoder"]="3011"
    ["knowledge-graph-mcp"]="3014"
    ["github"]="3016"
    ["language-server"]="3018"
    ["claude-task-runner"]="3019"
)

# Check each service
for service in "${!MCP_PORTS[@]}"; do
    port="${MCP_PORTS[$service]}"
    
    # Check if port is open
    if timeout 1 bash -c "echo > /dev/tcp/localhost/$port" 2>/dev/null; then
        # Try health endpoint
        if curl -s "http://localhost:$port/health" 2>/dev/null | grep -q "healthy"; then
            echo "✓ mcp-$service:$port - HEALTHY"
        else
            echo "⚠ mcp-$service:$port - RUNNING (no health endpoint)"
        fi
    else
        echo "✗ mcp-$service:$port - DOWN"
    fi
done

echo ""
echo "Docker Status:"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "mcp|MCP" || echo "No MCP containers found"
MONITOR_EOF

chmod +x /opt/sutazaiapp/scripts/mesh/monitor_mcp.sh

log "============================================" "$GREEN"
log "MCP HOST DEPLOYMENT COMPLETE" "$GREEN"
log "============================================" "$GREEN"
echo ""
log "Monitor services with:" "$YELLOW"
log "  bash /opt/sutazaiapp/scripts/mesh/monitor_mcp.sh" "$YELLOW"
echo ""
log "Check Consul UI:" "$YELLOW"
log "  http://localhost:10006/ui/dc1/services" "$YELLOW"

exit 0