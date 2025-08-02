#!/bin/bash
# Test the deployment with enhanced Brain and Docker handling

echo "🧠 Testing Fixed Deployment with Smart Docker Handling"
echo "==================================================="
echo ""

cd /opt/sutazaiapp

# Ensure Docker is running before test
if ! docker info >/dev/null 2>&1; then
    echo "⚠️  Docker not running - starting it first..."
    sudo dockerd > /tmp/dockerd_pretest.log 2>&1 &
    sleep 10
fi

# Run deployment
LOG_FILE="fixed_deployment_$(date +%Y%m%d_%H%M%S).log"

echo "Starting deployment with enhanced Brain system..."
echo "Log file: $LOG_FILE"
echo ""

# Run deployment with timeout
timeout 180 bash scripts/deploy_complete_system.sh 2>&1 | tee "$LOG_FILE" &
PID=$!

# Monitor progress
sleep 10

echo "Monitoring deployment progress..."
echo "================================"

# Check Brain activity
if grep -q "Brain: Analyzing system state" "$LOG_FILE"; then
    echo "✅ Brain is active and analyzing system"
fi

# Check Docker handling
if grep -q "Brain: Detected WSL2 with running Docker - avoiding risky restart" "$LOG_FILE"; then
    echo "✅ Brain correctly detected WSL2 Docker and avoided restart!"
elif grep -q "Brain: WSL2 detected - using specialized recovery strategy" "$LOG_FILE"; then
    echo "✅ Brain used WSL2-specific Docker recovery!"
fi

# Monitor for errors
sleep 30
if grep -q "Advanced Docker recovery failed" "$LOG_FILE"; then
    echo "❌ Docker recovery failed"
    
    # Show last error
    echo ""
    echo "Last error:"
    grep -A5 "Advanced Docker recovery failed" "$LOG_FILE" | tail -10
else
    echo "✅ No Docker recovery failures detected"
fi

# Check deployment progress
if grep -q "Phase 3:" "$LOG_FILE"; then
    echo "✅ Deployment progressed past initial phases"
fi

# Show Brain decisions
echo ""
echo "Brain Decisions Made:"
echo "===================="
grep "Brain Decision:" "$LOG_FILE" | head -10

# Kill deployment after monitoring
sleep 10
kill $PID 2>/dev/null

echo ""
echo "Test complete. Check log: $LOG_FILE"