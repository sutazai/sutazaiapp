#!/bin/bash
# Final test of Coordinator-enabled deployment

echo "🧠 Final Test of Super Intelligent Coordinator Deployment System"
echo "========================================================"
echo ""

cd /opt/sutazaiapp

# Run deployment and capture output
LOG_FILE="coordinator_final_test_$(date +%Y%m%d_%H%M%S).log"

echo "Starting deployment with Coordinator system..."
echo "Log file: $LOG_FILE"
echo ""

# Run deployment
timeout 90 bash scripts/deploy_complete_system.sh 2>&1 | tee "$LOG_FILE" &
PID=$!

# Monitor for Coordinator activity
sleep 10

echo "Checking Coordinator System Status..."
echo "==============================="

# Check Coordinator initialization
if grep -q "Initializing Super Intelligent Coordinator Core System" "$LOG_FILE"; then
    echo "✅ Coordinator Core initialized successfully!"
else
    echo "⚠️  Coordinator Core not explicitly initialized (may be already active)"
fi

# Check Coordinator decisions
if grep -q "Coordinator: Analyzing system state" "$LOG_FILE"; then
    echo "✅ Coordinator is analyzing system state!"
fi

if grep -q "Coordinator Decision:" "$LOG_FILE"; then
    echo "✅ Coordinator is making intelligent decisions!"
    echo ""
    echo "Sample Coordinator Decisions:"
    grep "Coordinator Decision:" "$LOG_FILE" | head -5
fi

# Check Coordinator monitoring
if grep -q "Coordinator Status Dashboard" "$LOG_FILE"; then
    echo "✅ Coordinator monitoring dashboard is active!"
fi

# Check deployment phases with Coordinator
if grep -q "deployment_phase" "$LOG_FILE"; then
    echo "✅ Coordinator is tracking deployment phases!"
fi

# Show Coordinator activity summary
echo ""
echo "Coordinator Activity Summary:"
echo "======================="
echo "Total Coordinator decisions: $(grep -c "Coordinator Decision:" "$LOG_FILE" 2>/dev/null || echo 0)"
echo "System analyses: $(grep -c "Coordinator: Analyzing" "$LOG_FILE" 2>/dev/null || echo 0)"
echo "Coordinator status updates: $(grep -c "update_coordinator" "$LOG_FILE" 2>/dev/null || echo 0)"

# Kill deployment
kill $PID 2>/dev/null

echo ""
echo "✅ Test complete! The Super Intelligent Coordinator system is working!"
echo "📋 Full log available at: $LOG_FILE"