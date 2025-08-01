#!/bin/bash
echo "Testing SutazAI deployment automation..."
echo "Starting test at $(date)"
cd /opt/sutazaiapp
export AUTOMATED=true
export SKIP_CLEANUP=true  
export SKIP_MODEL_DOWNLOADS=true
timeout 120 sudo -E ./scripts/deploy_complete_system.sh > automation_test.log 2>&1
echo "Test completed at $(date)"
echo "Exit code: $?"
echo ""
echo "=== FIRST 20 LINES OF OUTPUT ==="
head -20 automation_test.log
echo ""
echo "=== LAST 20 LINES OF OUTPUT ==="
tail -20 automation_test.log
