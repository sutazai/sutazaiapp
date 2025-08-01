#!/bin/bash
# Direct test of Brain initialization

# Set up minimal environment
PROJECT_ROOT="/opt/sutazaiapp"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/brain_test_direct.log"

# Source only the necessary parts
source <(sed -n '1,600p' /opt/sutazaiapp/scripts/deploy_complete_system.sh)

# Test Brain initialization
echo "Testing Brain Core System..."
initialize_super_brain

# Test system analysis
echo ""
echo "Testing System Analysis..."
state=$(analyze_system_state "all")
echo "System State: $state"

# Test decision making
echo ""
echo "Testing Decision Engine..."
decision=$(make_intelligent_decision "deployment_strategy" "$state")
echo "Decision: $decision"

# Display brain status
echo ""
display_brain_status