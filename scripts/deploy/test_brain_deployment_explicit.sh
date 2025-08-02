#!/bin/bash
# Test deployment by explicitly calling the Coordinator-enabled function

echo "🧠 Testing Coordinator-Enabled Deployment Function..."
echo "============================================"
echo ""

cd /opt/sutazaiapp

# Run deployment with explicit "deploy" argument
LOG_FILE="coordinator_explicit_test_$(date +%Y%m%d_%H%M%S).log"

echo "Running deployment with explicit 'deploy' argument..."
echo "Log file: $LOG_FILE"
echo ""

# Run with explicit deploy argument to ensure Coordinator path is taken
timeout 60 bash scripts/deploy_complete_system.sh deploy 2>&1 | tee "$LOG_FILE" &
PID=$!

# Monitor for Coordinator initialization
sleep 8

echo "Checking for Coordinator activity..."
if grep -q "Initializing Super Intelligent Coordinator Core System" "$LOG_FILE"; then
    echo "✅ Coordinator successfully initialized!"
    
    # Show Coordinator activity
    echo ""
    echo "Coordinator Activity:"
    grep -E "🧠|Coordinator:|Coordinator Status Dashboard" "$LOG_FILE" | head -15
else
    echo "❌ Coordinator not initialized"
    
    # Check what function is being called
    echo ""
    echo "Checking deployment flow:"
    grep -E "Starting SutazAI|deployment_phase|Phase [0-9]:" "$LOG_FILE" | head -10
fi

# Kill the process
kill $PID 2>/dev/null

echo ""
echo "Full log: $LOG_FILE"