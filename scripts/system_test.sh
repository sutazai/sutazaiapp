#!/bin/bash

# System functionality test script
echo "🧠 SutazAI System Functionality Test"
echo "====================================="
echo "Timestamp: $(date)"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test results
TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    local test_name="$1"
    local command="$2"
    local expected_code="${3:-0}"
    
    echo -n "Testing $test_name... "
    
    if eval "$command" >/dev/null 2>&1; then
        if [ $? -eq $expected_code ]; then
            echo -e "${GREEN}✓ PASS${NC}"
            ((TESTS_PASSED++))
            return 0
        fi
    fi
    
    echo -e "${RED}✗ FAIL${NC}"
    ((TESTS_FAILED++))
    return 1
}

run_api_test() {
    local endpoint="$1"
    local test_name="$2"
    local expected_key="$3"
    
    echo -n "Testing $test_name... "
    
    response=$(curl -s "http://172.31.77.193:8000$endpoint")
    if echo "$response" | python3 -c "import sys, json; data=json.load(sys.stdin); sys.exit(0 if '$expected_key' in data else 1)" 2>/dev/null; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

echo "🔧 Container Health Tests"
echo "========================"

# Test container health
CONTAINERS=("sutazai-postgres" "sutazai-redis" "sutazai-ollama" "sutazai-chromadb" "sutazai-qdrant" "sutazai-backend-agi" "sutazai-frontend-agi")

for container in "${CONTAINERS[@]}"; do
    run_test "$container health" "docker ps | grep -q '$container.*healthy'"
done

echo ""
echo "🌐 API Endpoint Tests"
echo "===================="

# Test API endpoints
run_api_test "/health" "Backend health endpoint" "status"
run_api_test "/agents" "Agents endpoint" "agents"
run_api_test "/models" "Models endpoint" "models"
run_api_test "/metrics" "Metrics endpoint" "system"

echo ""
echo "🤖 AI Model Tests"
echo "================="

# Test Ollama
run_test "Ollama API" "curl -s http://172.31.77.193:11434/api/tags | grep -q 'models'"

# Test model inference
echo -n "Testing AI model inference... "
response=$(curl -s -X POST http://172.31.77.193:8000/simple-chat \
    -H "Content-Type: application/json" \
    -d '{"message": "Say hello", "model": "llama3.2:1b"}')

if echo "$response" | python3 -c "import sys, json; data=json.load(sys.stdin); sys.exit(0 if 'response' in data and len(data['response']) > 0 else 1)" 2>/dev/null; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAIL${NC}"
    ((TESTS_FAILED++))
fi

echo ""
echo "🎯 Frontend Tests"
echo "================"

run_test "Frontend accessibility" "curl -s http://172.31.77.193:8501 | grep -q 'SutazAI'"
run_test "Frontend health" "docker logs sutazai-frontend-agi --tail 10 | grep -v TypeError"

echo ""
echo "📊 System Resource Tests"
echo "========================"

# Check system resources
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')

echo "System Resources:"
echo "  CPU Usage: ${CPU_USAGE}%"
echo "  Memory Usage: ${MEMORY_USAGE}%"
echo "  Disk Usage: ${DISK_USAGE}%"

# Resource tests
run_test "CPU usage acceptable" "[ $(echo '$CPU_USAGE < 80' | bc -l) -eq 1 ]"
run_test "Memory usage acceptable" "[ $(echo '$MEMORY_USAGE < 80' | bc -l) -eq 1 ]"
run_test "Disk usage acceptable" "[ $DISK_USAGE -lt 80 ]"

echo ""
echo "🛠️ Utility Scripts Tests"
echo "========================"

run_test "Live logs script" "timeout 2 /opt/sutazaiapp/scripts/live_logs.sh status"
run_test "Ollama health check" "/opt/sutazaiapp/scripts/ollama_health_check.sh | grep -q 'healthy'"
run_test "Performance test script" "[ -x /opt/sutazaiapp/scripts/test_performance.sh ]"
run_test "Monitor dashboard script" "[ -x /opt/sutazaiapp/scripts/monitor_dashboard.sh ]"

echo ""
echo "📋 Test Summary"
echo "==============="
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
echo -e "Total Tests: $((TESTS_PASSED + TESTS_FAILED))"

if [ $TESTS_FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 ALL TESTS PASSED! SutazAI system is fully operational.${NC}"
    exit 0
else
    echo ""
    echo -e "${YELLOW}⚠️  Some tests failed. Please check the issues above.${NC}"
    exit 1
fi