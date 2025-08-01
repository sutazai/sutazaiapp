#!/bin/bash
# Test the Brain-enhanced deployment system

echo "🧠 Testing Super Intelligent Brain Deployment System..."
echo "=================================================="
echo ""

# Run deployment with Brain system
cd /opt/sutazaiapp

# Create test log file
LOG_FILE="brain_deployment_test_$(date +%Y%m%d_%H%M%S).log"

echo "📝 Running deployment with Brain system enabled..."
echo "📋 Log file: $LOG_FILE"
echo ""

# Run with timeout to prevent hanging
timeout 120 bash scripts/deploy_complete_system.sh 2>&1 | tee "$LOG_FILE" &
PID=$!

# Monitor for Brain initialization
echo "🔍 Monitoring for Brain initialization..."
sleep 5

# Check if Brain was initialized
if grep -q "Initializing Super Intelligent Brain Core System" "$LOG_FILE"; then
    echo "✅ Brain system successfully initialized!"
    
    # Check for Brain decisions
    if grep -q "Brain decided on deployment strategy" "$LOG_FILE"; then
        echo "✅ Brain is making deployment decisions!"
    else
        echo "⚠️  Brain initialized but not making decisions yet"
    fi
    
    # Check Brain status
    if grep -q "Brain Status Dashboard" "$LOG_FILE"; then
        echo "✅ Brain status monitoring is active!"
    fi
else
    echo "❌ Brain system was NOT initialized - checking why..."
    
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
echo "- Check the log for detailed Brain activity"