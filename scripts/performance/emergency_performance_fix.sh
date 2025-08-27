#!/bin/bash
# Emergency Performance Fix Script
# Date: 2025-08-17
# Purpose: Fix critical performance issues - CPU spike, container sprawl, MCP failures

set -e

echo "================================================"
echo "SutazAI Emergency Performance Fix"
echo "Started: $(date)"
echo "================================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 1. Fix Backend CPU Spike
log_info "Step 1: Fixing backend CPU spike..."

# Check if backend is running with reload
if docker exec sutazai-backend ps aux | grep -q -- "--reload"; then
    log_warn "Backend running with --reload flag detected"
    
    # Create fixed entrypoint
    cat > /tmp/backend_entrypoint.sh << 'EOF'
#!/bin/sh
# Production entrypoint without reload
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2 --loop uvloop
EOF
    
    # Copy to container and restart
    docker cp /tmp/backend_entrypoint.sh sutazai-backend:/entrypoint_fixed.sh
    docker exec sutazai-backend chmod +x /entrypoint_fixed.sh
    
    log_info "Restarting backend without reload..."
    docker restart sutazai-backend
    sleep 5
    
    # Verify fix
    if docker exec sutazai-backend ps aux | grep -q -- "--reload"; then
        log_error "Failed to remove reload flag"
    else
        log_info "✓ Backend reload fixed"
    fi
else
    log_info "✓ Backend not using reload flag"
fi

# 2. Clean Orphaned Containers
log_info "Step 2: Cleaning orphaned containers..."

# Get list of orphaned containers
ORPHANED=$(docker ps --format "{{.Names}}" | grep -E "^(charming_|bold_|happy_|goofy_|youthful_|amazing_|fervent_|infallible_|kind_)" || true)

if [ -n "$ORPHANED" ]; then
    log_warn "Found orphaned containers: $(echo $ORPHANED | wc -w)"
    for container in $ORPHANED; do
        log_info "  Removing: $container"
        docker rm -f $container 2>/dev/null || true
    done
else
    log_info "✓ No orphaned containers found"
fi

fi

# Clean exited containers
EXITED_COUNT=$(docker ps -a -f status=exited -q | wc -l)
if [ $EXITED_COUNT -gt 0 ]; then
    log_info "Cleaning $EXITED_COUNT exited containers..."
    docker container prune -f
fi

# 3. Verify Container Count
log_info "Step 3: Verifying container count..."

CURRENT_COUNT=$(docker ps --format "{{.Names}}" | wc -l)
EXPECTED_COUNT=23
TOLERANCE=3

if [ $CURRENT_COUNT -gt $((EXPECTED_COUNT + TOLERANCE)) ]; then
    log_warn "Container count high: $CURRENT_COUNT (expected: ~$EXPECTED_COUNT)"
    
    # List unexpected containers
    log_info "Listing all containers for review:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" | head -40
elif [ $CURRENT_COUNT -lt $((EXPECTED_COUNT - TOLERANCE)) ]; then
    log_warn "Container count low: $CURRENT_COUNT (expected: ~$EXPECTED_COUNT)"
    log_info "Some services may not be running"
else
    log_info "✓ Container count normal: $CURRENT_COUNT"
fi

# 4. Check Resource Usage
log_info "Step 4: Checking resource usage..."

# Get backend CPU usage
BACKEND_CPU=$(docker stats --no-stream --format "{{.Container}}: {{.CPUPerc}}" | grep backend | cut -d: -f2 | tr -d ' %' || echo "0")

if (( $(echo "$BACKEND_CPU > 80" | bc -l) )); then
    log_error "Backend CPU still high: ${BACKEND_CPU}%"
    log_info "Manual intervention may be required"
else
    log_info "✓ Backend CPU normal: ${BACKEND_CPU}%"
fi

# 5. Test Backend Health
log_info "Step 5: Testing backend health..."

if curl -sf http://localhost:10010/health > /dev/null 2>&1; then
    log_info "✓ Backend health check passed"
else
    log_error "Backend health check failed"
    log_info "Checking backend logs..."
    docker logs sutazai-backend --tail 20 2>&1 | grep -E "(ERROR|CRITICAL)" || true
fi

# 6. Summary
echo ""
echo "================================================"
echo "Performance Fix Summary"
echo "================================================"

FINAL_COUNT=$(docker ps --format "{{.Names}}" | wc -l)
FINAL_CPU=$(docker stats --no-stream --format "{{.Container}}: {{.CPUPerc}}" | grep backend | cut -d: -f2 | tr -d ' %' || echo "0")

log_info "Container count: $FINAL_COUNT"
log_info "Backend CPU: ${FINAL_CPU}%"

# Check if fixes were successful
if [ $FINAL_COUNT -le $((EXPECTED_COUNT + TOLERANCE)) ] && (( $(echo "$FINAL_CPU < 80" | bc -l) )); then
    log_info "✓ Performance issues resolved!"
else
    log_warn "⚠ Some issues remain - manual review required"
    log_info "Run 'docker ps' and 'docker stats' for details"
fi

echo "================================================"
echo "Completed: $(date)"
echo "================================================"