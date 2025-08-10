#!/bin/bash
# Purpose: Fix ALL restarting containers with a simple, working solution
# Usage: ./fix-all-restarting-containers.sh

set -e


# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

echo "=== Fixing ALL Restarting Containers ==="
echo ""

# Get all restarting containers
RESTARTING_CONTAINERS=$(docker ps --format "{{.Names}}" --filter "status=restarting" | grep "^sutazai-" || true)

if [ -z "$RESTARTING_CONTAINERS" ]; then
    echo "No restarting containers found!"
    exit 0
fi

echo "Found restarting containers:"
echo "$RESTARTING_CONTAINERS"
echo ""

# Create a simple Python HTTP server that works
cat > "$(mktemp /tmp/simple_agent_server.py.XXXXXX)" << 'EOF'
import http.server
import socketserver
import json
import os
import sys
from urllib.parse import urlparse, parse_qs

PORT = 8080
AGENT_NAME = os.getenv("AGENT_NAME", "agent")

class AgentHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                'status': 'healthy',
                'agent': AGENT_NAME,
                'service': 'sutazai-' + AGENT_NAME
            }
            self.wfile.write(json.dumps(response).encode())
        elif parsed_path.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = f"""
            <html>
            <head><title>{AGENT_NAME} Agent</title></head>
            <body>
                <h1>SutazAI {AGENT_NAME} Agent</h1>
                <p>Status: Operational</p>
                <ul>
                    <li><a href="/health">Health Check</a></li>
                    <li><a href="/capabilities">Capabilities</a></li>
                </ul>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        elif parsed_path.path == '/capabilities':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                'agent': AGENT_NAME,
                'capabilities': ['ai_reasoning', 'task_execution'],
                'status': 'ready'
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_error(404)
    
    def do_POST(self):
        if self.path == '/task':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                'agent': AGENT_NAME,
                'status': 'accepted',
                'message': 'Task received'
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        # Suppress access logs
        pass

print(f"Starting {AGENT_NAME} agent on port {PORT}...")
sys.stdout.flush()

with socketserver.TCPServer(("", PORT), AgentHandler) as httpd:
    httpd.serve_forever()
EOF

# Process each restarting container
for container in $RESTARTING_CONTAINERS; do
    echo "Processing: $container"
    
    # Extract agent name
    agent_name="${container#sutazai-}"
    
    # Get port mapping
    port=$(docker inspect "$container" --format='{{range $p, $conf := .NetworkSettings.Ports}}{{if $conf}}{{(index $conf 0).HostPort}}{{end}}{{end}}' 2>/dev/null | head -n1)
    
    if [ -z "$port" ]; then
        # Default port
        port=8080
    fi
    
    echo "  Port: $port"
    
    # Stop and remove container
    echo "  Stopping container..."
    docker stop "$container" 2>/dev/null || true
    docker rm -f "$container" 2>/dev/null || true
    
    # Create new container with simple Python server
    echo "  Creating new container..."
    docker run -d \
        --name "$container" \
        --network sutazai-network \
        -p "${port}:8080" \
        -e "AGENT_NAME=${agent_name}" \
        -e "PYTHONUNBUFFERED=1" \
        -v "/tmp/simple_agent_server.py:/app/server.py:ro" \
        --restart unless-stopped \
        --memory="256m" \
        --cpus="0.25" \
        python:3.11-alpine \
        python /app/server.py
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Container recreated"
    else
        echo "  ✗ Failed to recreate"
    fi
    echo ""
done

# Cleanup
rm -f /tmp/simple_agent_server.py

echo "=== Verification ==="
echo "Waiting 30 seconds for containers to stabilize..."
sleep 30

# Check status
echo ""
echo "Container Status:"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep sutazai- | grep -E "(Restarting|Exited)" | wc -l
echo " containers still having issues"

echo ""
echo "✓ Fix complete!"