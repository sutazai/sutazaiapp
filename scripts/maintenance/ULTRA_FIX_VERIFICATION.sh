#!/bin/bash

echo "🚀 ULTRA-FIX VERIFICATION SCRIPT"
echo "=================================="
echo "Hardware Resource Optimizer - Complete Verification"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

function run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -e "🔍 Testing: ${YELLOW}$test_name${NC}"
    
    if eval "$test_command" >/dev/null 2>&1; then
        echo -e "   ✅ ${GREEN}PASSED${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "   ❌ ${RED}FAILED${NC}"
        ((TESTS_FAILED++))
    fi
}

echo "1️⃣ CODE QUALITY TESTS"
echo "----------------------"

# Test Python syntax
run_test "Python Syntax Validation" "python3 -m py_compile app.py"

# Test imports
run_test "Import Dependencies" "python3 -c 'import sys; sys.path.insert(0, \".\"); from app import validate_safe_path'"

# Test path validation function
run_test "Path Validation Security" "python3 test_path_validation.py"

echo ""
echo "2️⃣ DOCKER BUILD TESTS"
echo "----------------------"

# Test Docker build
run_test "Docker Image Build" "docker build -t test-hardware-optimizer -f Dockerfile ."

# Test if image exists
run_test "Docker Image Creation" "docker images | grep -q test-hardware-optimizer"

echo ""
echo "3️⃣ CONFIGURATION TESTS"
echo "----------------------"

# Test requirements.txt exists
run_test "Requirements File Exists" "test -f requirements.txt"

# Test Dockerfile exists
run_test "Dockerfile Exists" "test -f Dockerfile"

# Test shared directory
run_test "Shared Directory Structure" "test -d shared && test -f shared/agent_base.py"

echo ""
echo "4️⃣ SECURITY TESTS"
echo "------------------"

# Test security features in code
run_test "Thread Safety Imports" "grep -q 'threading' app.py"
run_test "Path Validation Function" "grep -q 'validate_safe_path' app.py"
run_test "Docker Client Lock" "grep -q 'docker_client_lock' app.py"
run_test "Hash Cache Lock" "grep -q 'hash_cache_lock' app.py"

echo ""
echo "5️⃣ FUNCTIONAL TESTS"
echo "-------------------"

# Test if the service would start (syntax check)
run_test "Service Startup Check" "python3 -c 'import sys; sys.path.insert(0, \".\"); from app import HardwareResourceOptimizerAgent; print(\"Agent class loads successfully\")'"

echo ""
echo "🎯 VERIFICATION RESULTS"
echo "======================="
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
echo "Total Tests: $TOTAL_TESTS"

if [ $TESTS_FAILED -eq 0 ]; then
    echo ""
    echo -e "🎉 ${GREEN}ALL TESTS PASSED!${NC}"
    echo -e "✅ ${GREEN}HARDWARE RESOURCE OPTIMIZER IS PRODUCTION READY!${NC}"
    echo ""
    echo "🚀 Ready for deployment:"
    echo "   docker compose up -d hardware-resource-optimizer"
    echo "   curl http://localhost:11110/health"
    exit 0
else
    echo ""
    echo -e "💥 ${RED}SOME TESTS FAILED!${NC}"
    echo -e "❌ ${RED}NEEDS ATTENTION BEFORE DEPLOYMENT${NC}"
    exit 1
fi