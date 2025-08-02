#!/bin/bash
# Direct test of Coordinator initialization

# Set up minimal environment
PROJECT_ROOT="/opt/sutazaiapp"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/coordinator_test_direct.log"

# Source only the necessary parts
source <(sed -n '1,600p' /opt/sutazaiapp/scripts/deploy_complete_system.sh)

# Test Coordinator initialization
echo "Testing Coordinator Core System..."
initialize_super_coordinator

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

# Display coordinator status
echo ""
display_coordinator_status