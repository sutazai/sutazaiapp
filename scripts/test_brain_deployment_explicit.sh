#!/bin/bash
# Test deployment by explicitly calling the Brain-enabled function

echo "🧠 Testing Brain-Enabled Deployment Function..."
echo "============================================"
echo ""

cd /opt/sutazaiapp

# Run deployment with explicit "deploy" argument
LOG_FILE="brain_explicit_test_$(date +%Y%m%d_%H%M%S).log"

echo "Running deployment with explicit 'deploy' argument..."
echo "Log file: $LOG_FILE"
echo ""

# Run with explicit deploy argument to ensure Brain path is taken
timeout 60 bash scripts/deploy_complete_system.sh deploy 2>&1 | tee "$LOG_FILE" &
PID=$!

# Monitor for Brain initialization
sleep 8

echo "Checking for Brain activity..."
if grep -q "Initializing Super Intelligent Brain Core System" "$LOG_FILE"; then
    echo "✅ Brain successfully initialized!"
    
    # Show Brain activity
    echo ""
    echo "Brain Activity:"
    grep -E "🧠|Brain:|Brain Status Dashboard" "$LOG_FILE" | head -15
else
    echo "❌ Brain not initialized"
    
    # Check what function is being called
    echo ""
    echo "Checking deployment flow:"
    grep -E "Starting SutazAI|deployment_phase|Phase [0-9]:" "$LOG_FILE" | head -10
fi

# Kill the process
kill $PID 2>/dev/null

echo ""
echo "Full log: $LOG_FILE"