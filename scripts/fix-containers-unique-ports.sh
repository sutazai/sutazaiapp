#!/bin/bash
# Purpose: Fix restarting containers with unique port assignments
# Usage: ./fix-containers-unique-ports.sh

set -e

echo "=== Fixing Containers with Unique Ports ==="
echo ""

# Create port mapping
declare -A PORT_MAP=(
    ["sutazai-complex-problem-solver"]=8671
    ["sutazai-automated-incident-responder"]=8672
    ["sutazai-system-knowledge-curator"]=8673
    ["sutazai-document-knowledge-manager"]=8674
    ["sutazai-ai-product-manager"]=8675
    ["sutazai-ai-scrum-master"]=8676
    ["sutazai-security-pentesting-specialist"]=8677
    ["sutazai-testing-qa-validator"]=8678
    ["sutazai-mega-code-auditor"]=8679
    ["sutazai-ai-senior-frontend-developer"]=8680
    ["sutazai-system-optimizer-reorganizer"]=8681
    ["sutazai-observability-monitoring-engineer"]=8682
    ["sutazai-cicd-pipeline-orchestrator"]=8683
    ["sutazai-container-orchestrator-k3s"]=8684
    ["sutazai-ai-system-architect"]=8685
    ["sutazai-ai-metrics-exporter"]=8686
    ["sutazai-ai-agent-orchestrator"]=8687
    ["sutazai-infrastructure-devops-manager"]=8688
)

# Get all problematic containers
CONTAINERS=$(docker ps -a --format "{{.Names}}" | grep "^sutazai-" | while read container; do
    status=$(docker inspect "$container" --format='{{.State.Status}}' 2>/dev/null)
    if [ "$status" = "restarting" ] || [ "$status" = "exited" ]; then
        echo "$container"
    fi
done)

if [ -z "$CONTAINERS" ]; then
    echo "No problematic containers found!"
    exit 0
fi

echo "Found containers to fix:"
echo "$CONTAINERS"
echo ""

# Process each container
for container in $CONTAINERS; do
    echo "Processing: $container"
    
    # Get agent name
    agent_name="${container#sutazai-}"
    
    # Get port from map or generate one
    if [ -n "${PORT_MAP[$container]}" ]; then
        port="${PORT_MAP[$container]}"
    else
        # Generate port based on container name hash
        port=$((8700 + $(echo "$container" | cksum | cut -d' ' -f1) % 300))
    fi
    
    echo "  Assigned port: $port"
    
    # Stop and remove
    docker stop "$container" 2>/dev/null || true
    docker rm -f "$container" 2>/dev/null || true
    
    # Create container with unique port
    docker run -d \
        --name "$container" \
        --network sutazai-network \
        -p "${port}:8080" \
        -e "AGENT_NAME=${agent_name}" \
        -e "PORT=8080" \
        --restart unless-stopped \
        --memory="256m" \
        --cpus="0.25" \
        python:3.11-alpine \
        sh -c "echo 'import http.server, socketserver, json, os; 
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == \"/health\":
            self.send_response(200)
            self.send_header(\"Content-type\", \"application/json\")
            self.end_headers()
            self.wfile.write(json.dumps({\"status\": \"healthy\", \"agent\": os.getenv(\"AGENT_NAME\")}).encode())
        else:
            self.send_error(404)
    def log_message(self, *args): pass
with socketserver.TCPServer((\"\", 8080), H) as httpd: httpd.serve_forever()' | python"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Created on port $port"
    else
        echo "  ✗ Failed"
    fi
done

echo ""
echo "=== Waiting for stabilization (30s) ==="
sleep 30

# Final check
echo ""
echo "Final Status:"
restarting_count=$(docker ps --filter "status=restarting" --format "{{.Names}}" | grep -c "^sutazai-" || echo "0")
echo "Restarting containers: $restarting_count"

if [ "$restarting_count" -eq 0 ]; then
    echo "✓ All containers fixed!"
else
    echo "⚠ Some containers still restarting"
    docker ps --filter "status=restarting" --format "table {{.Names}}\t{{.RestartCount}}" | grep "sutazai-" | head -5
fi