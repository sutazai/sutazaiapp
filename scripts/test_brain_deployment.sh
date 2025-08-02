#!/bin/bash
# Test the Coordinator-enhanced deployment system

echo "🧠 Testing Super Intelligent Coordinator Deployment System..."
echo "=================================================="
echo ""

# Run deployment with Coordinator system
cd /opt/sutazaiapp

# Create test log file
LOG_FILE="coordinator_deployment_test_$(date +%Y%m%d_%H%M%S).log"

echo "📝 Running deployment with Coordinator system enabled..."
echo "📋 Log file: $LOG_FILE"
echo ""

# Run with timeout to prevent hanging
timeout 120 bash scripts/deploy_complete_system.sh 2>&1 | tee "$LOG_FILE" &
PID=$!

# Monitor for Coordinator initialization
echo "🔍 Monitoring for Coordinator initialization..."
sleep 5

# Check if Coordinator was initialized
if grep -q "Initializing Super Intelligent Coordinator Core System" "$LOG_FILE"; then
    echo "✅ Coordinator system successfully initialized!"
    
    # Check for Coordinator decisions
    if grep -q "Coordinator decided on deployment strategy" "$LOG_FILE"; then
        echo "✅ Coordinator is making deployment decisions!"
    else
        echo "⚠️  Coordinator initialized but not making decisions yet"
    fi
    
    # Check Coordinator status
    if grep -q "Coordinator Status Dashboard" "$LOG_FILE"; then
        echo "✅ Coordinator status monitoring is active!"
    fi
else
    echo "❌ Coordinator system was NOT initialized - checking why..."
    
    # Check where it got stuck
    echo ""
    echo "Last 10 lines of output:"
    tail -10 "$LOG_FILE"
fi

# Kill the deployment after test
kill $PID 2>/dev/null

echo ""
echo "📊 Test Summary:"
echo "- Log file: $LOG_FILE"
echo "- Check the log for detailed Coordinator activity"