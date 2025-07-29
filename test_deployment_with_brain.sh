#!/bin/bash
# Test deployment with Brain monitoring

echo "🧠 Testing Deployment with Brain System..."
echo "========================================"
echo ""

cd /opt/sutazaiapp

# Run deployment and capture output
LOG_FILE="brain_deployment_$(date +%Y%m%d_%H%M%S).log"

echo "Starting deployment..."
echo "Log file: $LOG_FILE"
echo ""

# Run deployment with timeout
timeout 60 bash scripts/deploy_complete_system.sh 2>&1 | tee "$LOG_FILE" &
PID=$!

# Monitor for Brain activity
echo "Monitoring for Brain activity..."
sleep 5

# Check Brain initialization
if grep -q "Initializing Super Intelligent Brain Core System" "$LOG_FILE"; then
    echo "✅ Brain initialized!"
    
    # Check for Brain decisions
    sleep 10
    if grep -q "Brain decided on deployment strategy" "$LOG_FILE"; then
        echo "✅ Brain is making deployment decisions!"
        grep "Brain decided on deployment strategy" "$LOG_FILE"
    fi
    
    # Check Brain monitoring
    if grep -q "Brain Status Dashboard" "$LOG_FILE"; then
        echo "✅ Brain monitoring is active!"
    fi
    
    # Show Brain-related log entries
    echo ""
    echo "Brain Activity Log:"
    echo "==================="
    grep -E "🧠|Brain:" "$LOG_FILE" | head -20
else
    echo "❌ Brain not initialized - checking deployment flow..."
    echo ""
    echo "Current deployment phase:"
    grep -E "Phase [0-9]:|Starting SutazAI" "$LOG_FILE" | tail -5
fi

# Kill deployment
kill $PID 2>/dev/null

echo ""
echo "Test complete. Full log: $LOG_FILE"