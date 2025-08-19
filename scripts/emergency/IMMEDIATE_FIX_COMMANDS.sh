#!/bin/bash
# IMMEDIATE FIX COMMANDS - EXECUTE NOW!
# Generated: 2025-08-19 by ULTRATHINK System Reorganizer
# Purpose: Restore critical infrastructure in 5 minutes

set -e

echo "=================================="
echo "EMERGENCY INFRASTRUCTURE RESTORE"
echo "=================================="

# 1. Stop random containers
echo "[1/6] Stopping non-sutazai containers..."
docker stop nice_curie adoring_poincare jolly_volhard great_sutherland wizardly_panini suspicious_khayyam 2>/dev/null || true

# 2. Create network
echo "[2/6] Creating sutazai-network..."
docker network create sutazai-network 2>/dev/null || echo "Network already exists"

# 3. Backup and fix docker-compose
echo "[3/6] Fixing docker-compose configuration..."
if [ -f "/opt/sutazaiapp/docker-compose.yml" ]; then
    mv /opt/sutazaiapp/docker-compose.yml /opt/sutazaiapp/docker-compose.yml.broken.$(date +%Y%m%d_%H%M%S)
fi
cp /opt/sutazaiapp/docker/docker-compose.consolidated.yml /opt/sutazaiapp/docker-compose.yml

# 4. Start critical services
echo "[4/6] Starting critical infrastructure..."
cd /opt/sutazaiapp
docker-compose up -d postgres redis neo4j

# 5. Wait for databases
echo "[5/6] Waiting for databases to initialize (30s)..."
sleep 30

# 6. Start all services
echo "[6/6] Starting all sutazai services..."
docker-compose up -d

echo ""
echo "=================================="
echo "VERIFICATION"
echo "=================================="
docker ps --filter "name=sutazai-" --format "table {{.Names}}\t{{.Status}}" | head -20

echo ""
echo "=================================="
echo "QUICK HEALTH CHECK"
echo "=================================="
echo -n "Backend API: "
curl -s -o /dev/null -w "%{http_code}" http://localhost:10010/health || echo "FAILED"
echo ""
echo -n "Frontend UI: "
curl -s -o /dev/null -w "%{http_code}" http://localhost:10011/ || echo "FAILED"
echo ""

echo "=================================="
echo "INFRASTRUCTURE RESTORE COMPLETE"
echo "=================================="
echo "Run full cleanup: /opt/sutazaiapp/scripts/emergency/ULTRATHINK_CRITICAL_CLEANUP.sh"