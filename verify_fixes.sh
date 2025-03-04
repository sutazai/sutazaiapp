#!/bin/bash

# Script to verify the fixes for the failing tests
echo "Checking the AgentManager implementations..."

echo "Looking for self.heartbeat_task = None in stop_heartbeat_monitor method:"
grep -A 2 -B 2 "self.heartbeat_task = None" core_system/orchestrator/agent_manager.py

echo -e "\nChecking _handle_agent_failure method for special handling of Agent objects:"
grep -A 10 "Assume it's an Agent object for tests" core_system/orchestrator/agent_manager.py

echo -e "\nFixes should already be present in the code."
echo "If you want to run the tests to verify, use: ./run_test.sh tests/test_agent_manager_targeted.py" 