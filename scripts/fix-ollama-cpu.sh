#!/bin/bash
# Fix Ollama CPU overload issue (185% -> <50%)
# Implements Rule 22 from IMPROVED_CODEBASE_RULES_v2.0.md

set -euo pipefail

echo "🚀 Starting Ollama CPU optimization fix..."
echo "Current target: Reduce CPU from 185% to <50%"
echo "============================================"

# Step 1: Check current Ollama status
echo -e "\n📊 Step 1: Checking current Ollama status..."
if docker ps | grep -q sutazai-ollama; then
    echo "✓ Ollama container is running"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" sutazai-ollama || true
else
    echo "⚠️  Ollama container not found running"
fi

# Step 2: Stop current Ollama instance
echo -e "\n🛑 Step 2: Stopping current Ollama instance..."
docker-compose -f docker-compose.yml stop ollama 2>/dev/null || true
docker stop sutazai-ollama 2>/dev/null || true
sleep 5

# Step 3: Apply the optimized configuration
echo -e "\n🔧 Step 3: Applying optimized Ollama configuration..."
docker-compose -f docker-compose.ollama-fix.yml up -d

# Step 4: Wait for Ollama to be healthy
echo -e "\n⏳ Step 4: Waiting for Ollama to become healthy..."
for i in {1..30}; do
    if curl -f http://localhost:10104/api/tags >/dev/null 2>&1; then
        echo "✓ Ollama is healthy!"
        break
    fi
    echo -n "."
    sleep 2
done

# Step 5: Pull and configure tinyllama
echo -e "\n🤖 Step 5: Configuring tinyllama model..."
docker exec sutazai-ollama ollama pull tinyllama:latest || {
    echo "⚠️  Failed to pull tinyllama, but continuing..."
}

# Step 6: Create Ollama connection pooling script
echo -e "\n🔌 Step 6: Creating connection pooling manager..."
cat > /opt/sutazaiapp/scripts/ollama-pool-manager.py << 'EOF'
#!/usr/bin/env python3
"""
Ollama Connection Pool Manager
Implements connection pooling and request queuing for Ollama
"""
import asyncio
import aiohttp
import redis
from typing import Dict, Any
import json
import time
from datetime import datetime

class OllamaPoolManager:
    def __init__(self, ollama_host="http://localhost:10104", max_connections=10):
        self.ollama_host = ollama_host
        self.max_connections = max_connections
        self.queue = redis.Redis(host='localhost', port=10105, decode_responses=True)
        self.active_connections = 0
        
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request with connection pooling"""
        while self.active_connections >= self.max_connections:
            await asyncio.sleep(0.1)
            
        self.active_connections += 1
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_host}/api/generate",
                    json=request,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    result = await response.json()
                    return result
        finally:
            self.active_connections -= 1
            
    async def queue_processor(self):
        """Process queued requests"""
        while True:
            try:
                # Get request from queue
                request_json = self.queue.lpop('ollama:requests')
                if request_json:
                    request = json.loads(request_json)
                    result = await self.process_request(request['payload'])
                    
                    # Store result
                    self.queue.setex(
                        f"ollama:result:{request['id']}", 
                        300,  # 5 minute TTL
                        json.dumps(result)
                    )
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error processing request: {e}")
                await asyncio.sleep(1)

if __name__ == "__main__":
    manager = OllamaPoolManager()
    asyncio.run(manager.queue_processor())
EOF

chmod +x /opt/sutazaiapp/scripts/ollama-pool-manager.py

# Step 7: Create systemd service for pool manager
echo -e "\n🔧 Step 7: Creating systemd service for pool manager..."
cat > /tmp/ollama-pool-manager.service << 'EOF'
[Unit]
Description=Ollama Connection Pool Manager
After=docker.service
Requires=docker.service

[Service]
Type=simple
ExecStart=/usr/bin/python3 /opt/sutazaiapp/scripts/ollama-pool-manager.py
Restart=always
RestartSec=10
User=root
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
EOF

# Note: Would need sudo to install systemd service
echo "ℹ️  Systemd service file created at /tmp/ollama-pool-manager.service"
echo "   To install: sudo cp /tmp/ollama-pool-manager.service /etc/systemd/system/"
echo "   Then: sudo systemctl daemon-reload && sudo systemctl enable --now ollama-pool-manager"

# Step 8: Update agent configurations
echo -e "\n📝 Step 8: Creating agent configuration update script..."
cat > /opt/sutazaiapp/scripts/update-agent-ollama-config.sh << 'EOF'
#!/bin/bash
# Update all agents to use optimized Ollama settings

echo "Updating agent configurations for Ollama optimization..."

# Find all agent docker-compose files
for compose_file in $(find /opt/sutazaiapp -name "docker-compose*.yml" -type f); do
    if grep -q "OLLAMA_" "$compose_file"; then
        echo "Updating: $compose_file"
        # Backup original
        cp "$compose_file" "${compose_file}.backup"
        
        # Update Ollama environment variables
        sed -i 's/OLLAMA_NUM_PARALLEL:.*/OLLAMA_NUM_PARALLEL: 1/g' "$compose_file"
        sed -i 's/OLLAMA_NUM_THREADS:.*/OLLAMA_NUM_THREADS: 4/g' "$compose_file"
    fi
done

echo "✓ Agent configurations updated"
EOF

chmod +x /opt/sutazaiapp/scripts/update-agent-ollama-config.sh

# Step 9: Monitor CPU usage
echo -e "\n📊 Step 9: Monitoring new CPU usage..."
sleep 10
echo "Current Ollama resource usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" sutazai-ollama sutazai-ollama-queue

# Step 10: Create monitoring script
echo -e "\n📈 Step 10: Creating continuous monitoring script..."
cat > /opt/sutazaiapp/scripts/monitor-ollama-cpu.sh << 'EOF'
#!/bin/bash
# Monitor Ollama CPU usage continuously

echo "Monitoring Ollama CPU usage (Ctrl+C to stop)..."
echo "Target: <50% CPU usage"
echo "================================"

while true; do
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    cpu_usage=$(docker stats --no-stream --format "{{.CPUPerc}}" sutazai-ollama 2>/dev/null | sed 's/%//')
    
    if [ -n "$cpu_usage" ]; then
        echo -n "[$timestamp] Ollama CPU: ${cpu_usage}% "
        
        # Check if CPU is within target
        if (( $(echo "$cpu_usage < 50" | bc -l) )); then
            echo "✓ [OK]"
        else
            echo "⚠️  [HIGH]"
        fi
    else
        echo "[$timestamp] Ollama container not running"
    fi
    
    sleep 5
done
EOF

chmod +x /opt/sutazaiapp/scripts/monitor-ollama-cpu.sh

echo -e "\n✅ Ollama CPU optimization fix completed!"
echo "============================================"
echo "📋 Summary of changes:"
echo "  - OLLAMA_NUM_PARALLEL: 2 → 1"
echo "  - OLLAMA_NUM_THREADS: 8 → 4"
echo "  - OLLAMA_MAX_LOADED_MODELS: unlimited → 1"
echo "  - OLLAMA_KEEP_ALIVE: 2m → 30s"
echo "  - CPU limit: unlimited → 4 cores"
echo "  - CPU affinity: none → cores 4-7"
echo "  - Added Redis queue for request management"
echo ""
echo "🔧 Next steps:"
echo "  1. Run: ./scripts/monitor-ollama-cpu.sh"
echo "  2. Update all agents: ./scripts/update-agent-ollama-config.sh"
echo "  3. Install pool manager service (see instructions above)"
echo ""
echo "🎯 Expected outcome: CPU usage should drop from 185% to <50%"