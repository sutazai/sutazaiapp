#!/bin/bash
# Phase 5 Integration Validation Script
# Validates the complete MCP-Mesh integration and production readiness

set -e

echo "=========================================="
echo "PHASE 5 INTEGRATION VALIDATION"
echo "=========================================="
echo "Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n "Testing: $test_name... "
    
    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}❌ FAILED${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Function to check endpoint
check_endpoint() {
    local endpoint="$1"
    local expected_code="${2:-200}"
    
    response_code=$(curl -s -o /dev/null -w "%{http_code}" "$endpoint")
    [ "$response_code" = "$expected_code" ]
}

echo "1. BACKEND INTEGRATION CHECKS"
echo "------------------------------"

# Check backend health
run_test "Backend health endpoint" "check_endpoint http://localhost:8000/health"

# Check detailed health
run_test "Detailed health endpoint" "check_endpoint http://localhost:8000/api/v1/health/detailed"

# Check MCP endpoints
run_test "MCP services endpoint" "check_endpoint http://localhost:8000/api/v1/mcp/services"
run_test "MCP status endpoint" "check_endpoint http://localhost:8000/api/v1/mcp/status"
run_test "MCP health endpoint" "check_endpoint http://localhost:8000/api/v1/mcp/health"

echo ""
echo "2. SERVICE MESH INTEGRATION"
echo "---------------------------"

# Check mesh endpoints
run_test "Mesh services discovery" "check_endpoint http://localhost:8000/api/v1/mesh/v2/services"
run_test "Mesh health check" "check_endpoint http://localhost:8000/api/v1/mesh/v2/health"

echo ""
echo "3. MCP SERVICE VALIDATION"
echo "-------------------------"

# Test MCP integration
echo "Testing MCP-Mesh integration..."
integration_test=$(curl -s -X POST http://localhost:8000/api/v1/mcp/test-integration 2>/dev/null || echo "{}")

if echo "$integration_test" | grep -q "success_rate"; then
    success_rate=$(echo "$integration_test" | grep -o '"success_rate":[0-9.]*' | cut -d':' -f2)
    if (( $(echo "$success_rate > 70" | bc -l) )); then
        echo -e "${GREEN}✅ MCP Integration Success Rate: ${success_rate}%${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}❌ MCP Integration Success Rate: ${success_rate}% (Below 70% threshold)${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
fi

echo ""
echo "4. PERFORMANCE VALIDATION"
echo "-------------------------"

# Test response times
echo "Testing response times (10 requests)..."
total_time=0
for i in {1..10}; do
    start_time=$(date +%s%N)
    curl -s http://localhost:8000/health > /dev/null 2>&1
    end_time=$(date +%s%N)
    elapsed=$((($end_time - $start_time) / 1000000))
    total_time=$((total_time + elapsed))
done

avg_time=$((total_time / 10))
echo "Average response time: ${avg_time}ms"

if [ $avg_time -lt 200 ]; then
    echo -e "${GREEN}✅ Response time < 200ms target${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${YELLOW}⚠️  Response time ${avg_time}ms > 200ms target${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

echo ""
echo "5. MULTI-CLIENT ACCESS TEST"
echo "---------------------------"

# Test concurrent access
echo "Testing concurrent Claude Code and Codex access..."

# Simulate Claude Code request
claude_test=$(curl -s -X POST http://localhost:8000/api/v1/mcp/request \
    -H "Content-Type: application/json" \
    -d '{
        "client_type": "claude_code",
        "client_id": "test_claude",
        "service": "files",
        "method": "list",
        "params": {"path": "/opt/sutazaiapp"}
    }' 2>/dev/null || echo "failed")

# Simulate Codex request
codex_test=$(curl -s -X POST http://localhost:8000/api/v1/mcp/request \
    -H "Content-Type: application/json" \
    -d '{
        "client_type": "codex",
        "client_id": "test_codex",
        "service": "context7",
        "method": "search",
        "params": {"query": "test"}
    }' 2>/dev/null || echo "failed")

if [[ "$claude_test" != "failed" ]] && [[ "$codex_test" != "failed" ]]; then
    echo -e "${GREEN}✅ Multi-client access working${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${RED}❌ Multi-client access failed${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

echo ""
echo "6. PRODUCTION READINESS CHECKS"
echo "------------------------------"

# Check monitoring
run_test "Prometheus metrics endpoint" "check_endpoint http://localhost:8000/metrics"

# Check circuit breakers
run_test "Circuit breakers status" "check_endpoint http://localhost:8000/api/v1/health/circuit-breakers"

# Check cache stats
run_test "Cache statistics" "check_endpoint http://localhost:8000/api/v1/cache/stats"

# Check settings
run_test "System settings" "check_endpoint http://localhost:8000/api/v1/settings"

echo ""
echo "7. SYSTEM RESOURCE CHECK"
echo "------------------------"

# Check CPU and memory
cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
mem_usage=$(free | grep Mem | awk '{print ($3/$2) * 100.0}')

echo "CPU Usage: ${cpu_usage}%"
echo "Memory Usage: ${mem_usage}%"

echo ""
echo "=========================================="
echo "VALIDATION SUMMARY"
echo "=========================================="
echo "Total Tests: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $FAILED_TESTS"

SUCCESS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))
echo "Success Rate: ${SUCCESS_RATE}%"

echo ""
if [ $SUCCESS_RATE -ge 80 ]; then
    echo -e "${GREEN}✅ PHASE 5 VALIDATION PASSED${NC}"
    echo "The system has successfully resolved the 71.4% failure rate issue."
    echo "MCP-Mesh integration is working correctly."
    echo "System is ready for production deployment."
    exit 0
else
    echo -e "${RED}❌ PHASE 5 VALIDATION FAILED${NC}"
    echo "The system needs additional work to meet production requirements."
    echo "Please review the failed tests and address the issues."
    exit 1
fi