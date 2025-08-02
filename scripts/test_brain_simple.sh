#!/bin/bash
# Simple test to check if Coordinator is working

# Change to project directory
cd /opt/sutazaiapp

# Run the deployment script but skip the initial detection phases
# and go directly to our main function
bash -c '
source scripts/deploy_complete_system.sh
echo "🧠 Testing Coordinator System Directly..."
echo "================================="
echo ""

# Initialize Coordinator
initialize_super_coordinator

echo ""
echo "🧠 Testing System Analysis..."
state=$(analyze_system_state "all")
echo "State: $state"

echo ""
echo "🧠 Testing Decision Making..."
decision=$(make_intelligent_decision "deployment_strategy" "$state")
echo "Decision: $decision"

echo ""
echo "🧠 Testing Docker Coordinator..."
ensure_docker_running_perfectly
'