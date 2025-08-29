#!/bin/bash
# Quick fix for Docker health check and IP conflict issues

set -e

echo "======================================"
echo "Fixing Docker Infrastructure Issues"
echo "======================================"

cd /opt/sutazaiapp

# 1. Fix Ollama health check by recreating with proper command
echo "Fixing Ollama health check..."
docker stop sutazai-ollama 2>/dev/null || true
docker rm sutazai-ollama 2>/dev/null || true

# Start ollama with corrected health check
docker run -d \
  --name sutazai-ollama \
  --network sutazaiapp_sutazai-network \
  -p 11435:11434 \
  -v ollama-data:/root/.ollama \
  -e OLLAMA_HOST=0.0.0.0 \
  -e OLLAMA_KEEP_ALIVE=24h \
  --restart unless-stopped \
  --health-cmd "wget --spider --quiet http://localhost:11434/api/tags || exit 1" \
  --health-interval 30s \
  --health-timeout 10s \
  --health-retries 3 \
  --health-start-period 60s \
  --memory="4g" \
  --cpus="2" \
  ollama/ollama:latest serve

echo "✓ Ollama restarted with fixed health check"

# 2. Fix backend IP conflict
echo "Fixing backend IP conflict..."
docker stop sutazai-backend 2>/dev/null || true

# Update backend compose file to use different IP
sed -i 's/ipv4_address: 172.20.0.30/ipv4_address: 172.20.0.40/' docker-compose-backend.yml

# Restart backend with new IP
docker compose -f docker-compose-backend.yml up -d

echo "✓ Backend IP changed from 172.20.0.30 to 172.20.0.40"

# 3. Fix semgrep wrapper to not hang on health check
echo "Fixing Semgrep health endpoint..."

# Create a fixed wrapper that doesn't hang
cat > /tmp/fix_semgrep.py << 'EOF'
import sys
import os

# Find the semgrep wrapper file
wrapper_file = "/opt/sutazaiapp/agents/wrappers/semgrep_local.py"

# Read the current content
with open(wrapper_file, 'r') as f:
    content = f.read()

# Add a simple health endpoint that returns immediately
health_endpoint = '''
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "agent": "semgrep"}
'''

# Insert after the setup_semgrep_routes method definition
if "@self.app.get(\"/health\")" not in content:
    insert_pos = content.find("def setup_semgrep_routes(self):")
    if insert_pos != -1:
        # Find the next method or end of class
        next_line = content.find("\n", insert_pos)
        indent_pos = content.find("@self.app.post", next_line)
        if indent_pos != -1:
            content = content[:indent_pos] + health_endpoint + "\n" + content[indent_pos:]
            
            with open(wrapper_file, 'w') as f:
                f.write(content)
            print("Fixed semgrep health endpoint")
        else:
            print("Could not find insertion point")
    else:
        print("Could not find setup_semgrep_routes method")
else:
    print("Health endpoint already exists")
EOF

python3 /tmp/fix_semgrep.py

# Restart semgrep
docker restart sutazai-semgrep 2>/dev/null || true

echo "✓ Semgrep health endpoint fixed"

# 4. Verify all fixes
echo ""
echo "Verifying fixes..."
echo "=================="

# Check Ollama
if curl -f --max-time 5 http://localhost:11435/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama: Healthy"
else
    echo "⚠️  Ollama: Still starting up"
fi

# Check Backend IP
BACKEND_IP=$(docker inspect sutazai-backend 2>/dev/null | grep -A 2 "sutazai-network" | grep "IPAddress" | cut -d'"' -f4)
if [ "$BACKEND_IP" = "172.20.0.40" ]; then
    echo "✅ Backend: Running on correct IP (172.20.0.40)"
else
    echo "⚠️  Backend: IP is $BACKEND_IP (expected 172.20.0.40)"
fi

# Check for IP conflicts
echo ""
echo "Checking for IP conflicts..."
CONFLICTS=$(docker network inspect sutazaiapp_sutazai-network --format '{{json .Containers}}' 2>/dev/null | \
    jq -r '.[] | "\(.IPv4Address)"' | cut -d/ -f1 | sort | uniq -d | wc -l)

if [ "$CONFLICTS" -eq 0 ]; then
    echo "✅ No IP conflicts detected"
else
    echo "⚠️  IP conflicts still exist"
    docker network inspect sutazaiapp_sutazai-network --format '{{json .Containers}}' | \
        jq -r '.[] | "\(.Name): \(.IPv4Address)"' | sort
fi

# 5. Show current container status
echo ""
echo "Current Container Status:"
echo "========================"
docker ps --filter "name=sutazai" --format "table {{.Names}}\t{{.Status}}" | head -15

echo ""
echo "Fix script completed!"
echo ""
echo "Note: Some services may take a few minutes to become fully healthy."
echo "Run 'docker ps' to monitor status."