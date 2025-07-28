#!/bin/bash
# Test script to verify function timeout approach works

# Source the main script to get function definitions
source /opt/sutazaiapp/scripts/deploy_complete_system.sh

# Test 1: Check if functions are defined
echo "Testing function definitions..."
if declare -F optimize_network_downloads >/dev/null; then
    echo "✅ optimize_network_downloads function is defined"
else
    echo "❌ optimize_network_downloads function is NOT defined"
fi

if declare -F install_all_system_dependencies >/dev/null; then
    echo "✅ install_all_system_dependencies function is defined"
else
    echo "❌ install_all_system_dependencies function is NOT defined"
fi

# Test 2: Test the declare -f approach
echo -e "\nTesting declare -f approach..."
test_function() {
    echo "Test function executed successfully"
    return 0
}

# Test with timeout and declare -f
if timeout 5 bash -c "$(declare -f test_function); test_function" 2>/dev/null; then
    echo "✅ declare -f approach works"
else
    echo "❌ declare -f approach failed"
fi

# Test 3: Alternative approach - export functions
echo -e "\nTesting export function approach..."
export -f test_function
if timeout 5 bash -c 'test_function' 2>/dev/null; then
    echo "✅ export -f approach works"
else
    echo "❌ export -f approach failed"
fi

# Test 4: Check dependencies of the problematic functions
echo -e "\nChecking function dependencies..."
echo "optimize_network_downloads dependencies:"
grep -n "log_info\|log_success\|log_warn\|log_error" /opt/sutazaiapp/scripts/deploy_complete_system.sh | grep -A5 -B5 optimize_network_downloads | head -10

echo -e "\ninstall_all_system_dependencies dependencies:"
grep -n "log_info\|log_success\|log_warn\|log_error" /opt/sutazaiapp/scripts/deploy_complete_system.sh | grep -A5 -B5 install_all_system_dependencies | head -10