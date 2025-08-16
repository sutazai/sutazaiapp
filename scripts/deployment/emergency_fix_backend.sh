#!/bin/bash

# 🚨 EMERGENCY BACKEND FIX - PHASE 2
# Created: 2025-08-16 23:20:00 UTC  
# Purpose: Fix backend networkx dependency and restart

set -e

echo "================================================"
echo "🚨 EMERGENCY BACKEND FIX - PHASE 2"
echo "================================================"
echo "Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# Function to log actions
log_action() {
    echo "[$(date -u '+%H:%M:%S')] $1" | tee -a /opt/sutazaiapp/logs/emergency_backend_fix.log
}

# Step 1: Check current backend status
log_action "STEP 1: Checking current backend status..."
if curl -s --max-time 5 http://localhost:10010/health > /dev/null 2>&1; then
    log_action "Backend is responding (but may be unhealthy)"
else
    log_action "Backend is not responding"
fi

# Step 2: Add networkx to requirements.txt
log_action "STEP 2: Adding networkx to requirements.txt..."
cd /opt/sutazaiapp/backend

if ! grep -q "^networkx" requirements.txt; then
    echo "networkx==3.2.1" >> requirements.txt
    log_action "Added networkx==3.2.1 to requirements.txt"
else
    log_action "networkx already in requirements.txt"
fi

# Step 3: Create fixed Dockerfile
log_action "STEP 3: Creating emergency Dockerfile..."
cat > Dockerfile.emergency << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
EOF

# Step 4: Build new backend image
log_action "STEP 4: Building new backend image..."
docker build -f Dockerfile.emergency -t sutazai-backend:emergency . || {
    log_action "❌ ERROR: Docker build failed"
    exit 1
}

# Step 5: Stop current backend
log_action "STEP 5: Stopping current backend container..."
docker stop sutazai-backend 2>/dev/null || true
docker rm sutazai-backend 2>/dev/null || true

# Step 6: Start new backend
log_action "STEP 6: Starting new backend container..."
docker run -d \
    --name sutazai-backend \
    --network sutazai-network \
    -p 10010:8000 \
    -e DATABASE_URL="postgresql://postgres:postgres@sutazai-postgres:5432/sutazai" \
    -e REDIS_URL="redis://sutazai-redis:6379/0" \
    -e SERVICE_NAME="sutazai-backend" \
    -e CONSUL_HOST="sutazai-consul" \
    -e CONSUL_PORT="8500" \
    --restart unless-stopped \
    sutazai-backend:emergency

# Step 7: Wait for backend to start
log_action "STEP 7: Waiting for backend to start..."
MAX_WAIT=60
WAIT_COUNT=0

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s --max-time 5 http://localhost:10010/health > /dev/null 2>&1; then
        log_action "✅ Backend is responding!"
        break
    fi
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
    echo -n "."
done
echo ""

# Step 8: Verify backend health
log_action "STEP 8: Verifying backend health..."
HEALTH_STATUS=$(curl -s --max-time 10 http://localhost:10010/health 2>/dev/null || echo '{"status":"error"}')
echo "Health response: $HEALTH_STATUS"

# Step 9: Check container logs
log_action "STEP 9: Checking container logs for errors..."
docker logs --tail 20 sutazai-backend 2>&1 | grep -E "(ERROR|CRITICAL|networkx)" || true

# Step 10: Verify MCP endpoints
log_action "STEP 10: Testing MCP endpoints..."
MCP_STATUS=$(curl -s --max-time 10 http://localhost:10010/api/v1/mcp/status 2>/dev/null || echo "TIMEOUT")
if [ "$MCP_STATUS" != "TIMEOUT" ]; then
    log_action "✅ MCP endpoints responding"
else
    log_action "⚠️ MCP endpoints still timing out (may need Phase 3)"
fi

echo ""
echo "=== BACKEND FIX SUMMARY ==="
echo "networkx dependency: ADDED"
echo "Docker image: REBUILT"
echo "Container: RESTARTED"
echo "Health check: $([[ "$HEALTH_STATUS" == *"healthy"* ]] && echo "PASSED" || echo "FAILED")"
echo "MCP endpoints: $([[ "$MCP_STATUS" != "TIMEOUT" ]] && echo "RESPONDING" || echo "TIMEOUT")"
echo ""

echo "================================================"
echo "PHASE 2 BACKEND FIX COMPLETE"
echo "Next: Run emergency_unified_architecture.sh"
echo "================================================"