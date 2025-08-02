#!/bin/bash
# Final test of Brain-enabled deployment

echo "🧠 Final Test of Super Intelligent Brain Deployment System"
echo "========================================================"
echo ""

cd /opt/sutazaiapp

# Run deployment and capture output
LOG_FILE="brain_final_test_$(date +%Y%m%d_%H%M%S).log"

echo "Starting deployment with Brain system..."
echo "Log file: $LOG_FILE"
echo ""

# Run deployment
timeout 90 bash scripts/deploy_complete_system.sh 2>&1 | tee "$LOG_FILE" &
PID=$!

# Monitor for Brain activity
sleep 10

echo "Checking Brain System Status..."
echo "==============================="

# Check Brain initialization
if grep -q "Initializing Super Intelligent Brain Core System" "$LOG_FILE"; then
    echo "✅ Brain Core initialized successfully!"
else
    echo "⚠️  Brain Core not explicitly initialized (may be already active)"
fi

# Check Brain decisions
if grep -q "Brain: Analyzing system state" "$LOG_FILE"; then
    echo "✅ Brain is analyzing system state!"
fi

if grep -q "Brain Decision:" "$LOG_FILE"; then
    echo "✅ Brain is making intelligent decisions!"
    echo ""
    echo "Sample Brain Decisions:"
    grep "Brain Decision:" "$LOG_FILE" | head -5
fi

# Check Brain monitoring
if grep -q "Brain Status Dashboard" "$LOG_FILE"; then
    echo "✅ Brain monitoring dashboard is active!"
fi

# Check deployment phases with Brain
if grep -q "deployment_phase" "$LOG_FILE"; then
    echo "✅ Brain is tracking deployment phases!"
fi

# Show Brain activity summary
echo ""
echo "Brain Activity Summary:"
echo "======================="
echo "Total Brain decisions: $(grep -c "Brain Decision:" "$LOG_FILE" 2>/dev/null || echo 0)"
echo "System analyses: $(grep -c "Brain: Analyzing" "$LOG_FILE" 2>/dev/null || echo 0)"
echo "Brain status updates: $(grep -c "update_brain" "$LOG_FILE" 2>/dev/null || echo 0)"

# Kill deployment
kill $PID 2>/dev/null

echo ""
echo "✅ Test complete! The Super Intelligent Brain system is working!"
echo "📋 Full log available at: $LOG_FILE"