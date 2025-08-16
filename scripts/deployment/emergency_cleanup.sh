#!/bin/bash

# 🚨 EMERGENCY CLEANUP SCRIPT - PHASE 1
# Created: 2025-08-16 23:20:00 UTC
# Purpose: Kill host MCPs and clean Docker chaos

set -e

echo "================================================"
echo "🚨 EMERGENCY SYSTEM CLEANUP - PHASE 1"
echo "================================================"
echo "Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# Function to log actions
log_action() {
    echo "[$(date -u '+%H:%M:%S')] $1" | tee -a /opt/sutazaiapp/logs/emergency_cleanup.log
}

# Step 1: Kill all host MCP processes
log_action "STEP 1: Killing all host MCP processes..."
MCP_COUNT_BEFORE=$(ps aux | grep -E "(mcp|claude)" | grep -v grep | wc -l)
log_action "Found $MCP_COUNT_BEFORE MCP processes running on host"

pkill -f "mcp" || true
pkill -f "claude" || true
pkill -f "npm exec" || true
pkill -f "node.*mcp" || true

# Kill zombie processes
ZOMBIES=$(ps aux | grep defunct | awk '{print $2}' | wc -l)
if [ "$ZOMBIES" -gt 0 ]; then
    log_action "Killing $ZOMBIES zombie processes..."
    ps aux | grep defunct | awk '{print $2}' | xargs -r kill -9
fi

MCP_COUNT_AFTER=$(ps aux | grep -E "(mcp|claude)" | grep -v grep | wc -l)
log_action "Remaining MCP processes: $MCP_COUNT_AFTER"

# Step 2: Stop and remove orphaned containers
log_action "STEP 2: Removing orphaned containers..."
ORPHANED="0c8d27e88cf7 6a20fbb0d87f 0df1eb6b5c89 3315cf444b73"
for container in $ORPHANED; do
    if docker ps -q | grep -q "$container"; then
        log_action "Stopping orphaned container: $container"
        docker stop "$container" 2>/dev/null || true
        docker rm "$container" 2>/dev/null || true
    fi
done

# Step 3: Clean up Docker resources
log_action "STEP 3: Cleaning Docker resources..."
VOLUMES_BEFORE=$(docker volume ls -q | wc -l)
IMAGES_BEFORE=$(docker images -f "dangling=true" -q | wc -l)

docker volume prune -f
docker network prune -f
docker image prune -f

VOLUMES_AFTER=$(docker volume ls -q | wc -l)
IMAGES_AFTER=$(docker images -f "dangling=true" -q | wc -l)

log_action "Volumes cleaned: $((VOLUMES_BEFORE - VOLUMES_AFTER))"
log_action "Images cleaned: $((IMAGES_BEFORE - IMAGES_AFTER))"

# Step 4: Clean Python cache
log_action "STEP 4: Cleaning Python cache files..."
CACHE_SIZE_BEFORE=$(find /opt/sutazaiapp -type d -name "__pycache__" 2>/dev/null | wc -l)
find /opt/sutazaiapp -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
log_action "Removed $CACHE_SIZE_BEFORE __pycache__ directories"

# Step 5: Remove binary packages from docs
log_action "STEP 5: Removing binary packages from documentation..."
rm -f /opt/sutazaiapp/docs/*.deb 2>/dev/null || true
rm -f /opt/sutazaiapp/docs/*.tgz 2>/dev/null || true
rm -f /opt/sutazaiapp/docs/*.tar.gz 2>/dev/null || true

# Step 6: Stop monitoring scripts
log_action "STEP 6: Stopping monitoring scripts..."
pkill -f "cleanup_containers.sh" || true
pkill -f "mcp_conflict_monitoring.sh" || true

# Step 7: Disable host MCP wrappers
log_action "STEP 7: Disabling host MCP wrapper scripts..."
if [ -d "/opt/sutazaiapp/scripts/mcp/wrappers" ]; then
    for script in /opt/sutazaiapp/scripts/mcp/wrappers/*; do
        if [ -f "$script" ] && [ -x "$script" ]; then
            chmod -x "$script"
            mv "$script" "$script.disabled"
            log_action "Disabled: $(basename $script)"
        fi
    done
fi

# Step 8: Final verification
log_action "STEP 8: Final verification..."
echo ""
echo "=== CLEANUP SUMMARY ==="
echo "MCP Processes: $MCP_COUNT_BEFORE → $MCP_COUNT_AFTER"
echo "Docker Volumes: $VOLUMES_BEFORE → $VOLUMES_AFTER"
echo "Dangling Images: $IMAGES_BEFORE → $IMAGES_AFTER"
echo "Zombie Processes Killed: $ZOMBIES"
echo ""

# Check if cleanup was successful
if [ "$MCP_COUNT_AFTER" -eq 0 ]; then
    log_action "✅ SUCCESS: All host MCP processes terminated"
else
    log_action "⚠️ WARNING: $MCP_COUNT_AFTER MCP processes still running"
    ps aux | grep -E "(mcp|claude)" | grep -v grep
fi

echo "================================================"
echo "PHASE 1 CLEANUP COMPLETE"
echo "Next: Run emergency_fix_backend.sh"
echo "================================================"