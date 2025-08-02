#!/bin/bash
# Test deployment with Coordinator monitoring

echo "🧠 Testing Deployment with Coordinator System..."
echo "========================================"
echo ""

cd /opt/sutazaiapp

# Run deployment and capture output
LOG_FILE="coordinator_deployment_$(date +%Y%m%d_%H%M%S).log"

echo "Starting deployment..."
echo "Log file: $LOG_FILE"
echo ""

# Run deployment with timeout
timeout 60 bash scripts/deploy_complete_system.sh 2>&1 | tee "$LOG_FILE" &
PID=$!

# Monitor for Coordinator activity
echo "Monitoring for Coordinator activity..."
sleep 5

# Check Coordinator initialization
if grep -q "Initializing Super Intelligent Coordinator Core System" "$LOG_FILE"; then
    echo "✅ Coordinator initialized!"
    
    # Check for Coordinator decisions
    sleep 10
    if grep -q "Coordinator decided on deployment strategy" "$LOG_FILE"; then
        echo "✅ Coordinator is making deployment decisions!"
        grep "Coordinator decided on deployment strategy" "$LOG_FILE"
    fi
    
    # Check Coordinator monitoring
    if grep -q "Coordinator Status Dashboard" "$LOG_FILE"; then
        echo "✅ Coordinator monitoring is active!"
    fi
    
    # Show Coordinator-related log entries
    echo ""
    echo "Coordinator Activity Log:"
    echo "==================="
    grep -E "🧠|Coordinator:" "$LOG_FILE" | head -20
else
    echo "❌ Coordinator not initialized - checking deployment flow..."
    echo ""
    echo "Current deployment phase:"
    grep -E "Phase [0-9]:|Starting SutazAI" "$LOG_FILE" | tail -5
fi

# Kill deployment
kill $PID 2>/dev/null

echo ""
echo "Test complete. Full log: $LOG_FILE"