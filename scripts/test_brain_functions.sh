#!/bin/bash
# Direct test of Brain functions

echo "🧠 Testing Brain Functions Directly..."
echo "===================================="

# Source the Brain functions (lines 75-700 to include all Brain functions)
source <(sed -n '75,700p' /opt/sutazaiapp/scripts/deploy_complete_system.sh)

# Also source the logging functions
source <(sed -n '1,74p' /opt/sutazaiapp/scripts/deploy_complete_system.sh)

# Initialize minimal environment
PROJECT_ROOT="/opt/sutazaiapp"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/brain_functions_test.log"

# Define missing logging functions if needed
log_info() { echo "[INFO] $1"; }
log_success() { echo "[SUCCESS] $1"; }
log_error() { echo "[ERROR] $1"; }
log_warning() { echo "[WARNING] $1"; }

# Test Brain initialization
echo ""
echo "1. Testing Brain Initialization..."
initialize_super_brain

# Test system analysis
echo ""
echo "2. Testing System Analysis..."
state=$(analyze_system_state "all")
echo "System State: $state"

# Test decision making
echo ""
echo "3. Testing Decision Engine..."
decision=$(make_intelligent_decision "deployment_strategy" "$state")
echo "Decision: $decision"

# Test component state update
echo ""
echo "4. Testing Component State Update..."
update_brain_component_state "test_component" "testing" "Running test"

# Display Brain status
echo ""
echo "5. Testing Brain Status Display..."
display_brain_status

echo ""
echo "✅ Brain function tests completed!"