#!/bin/bash
# Direct test of Coordinator functions

echo "🧠 Testing Coordinator Functions Directly..."
echo "===================================="

# Source the Coordinator functions (lines 75-700 to include all Coordinator functions)
source <(sed -n '75,700p' /opt/sutazaiapp/scripts/deploy_complete_system.sh)

# Also source the logging functions
source <(sed -n '1,74p' /opt/sutazaiapp/scripts/deploy_complete_system.sh)

# Initialize minimal environment
PROJECT_ROOT="/opt/sutazaiapp"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/coordinator_functions_test.log"

# Define missing logging functions if needed
log_info() { echo "[INFO] $1"; }
log_success() { echo "[SUCCESS] $1"; }
log_error() { echo "[ERROR] $1"; }
log_warning() { echo "[WARNING] $1"; }

# Test Coordinator initialization
echo ""
echo "1. Testing Coordinator Initialization..."
initialize_super_coordinator

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
update_coordinator_component_state "test_component" "testing" "Running test"

# Display Coordinator status
echo ""
echo "5. Testing Coordinator Status Display..."
display_coordinator_status

echo ""
echo "✅ Coordinator function tests completed!"