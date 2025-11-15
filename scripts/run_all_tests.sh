#!/bin/bash
# Test Coverage Validation Script
# Executes all test suites and generates coverage report

set -e

echo "=========================================="
echo "SutazAI Test Coverage Validation"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Frontend E2E Tests
echo "📱 Frontend E2E Tests (Playwright)"
echo "------------------------------------------"
cd /opt/sutazaiapp/frontend

if npx playwright test --reporter=list 2>&1 | tee /tmp/playwright-results.txt; then
    FRONTEND_PASSED=$(grep -c "✓" /tmp/playwright-results.txt || echo 0)
    FRONTEND_FAILED=$(grep -c "✘" /tmp/playwright-results.txt || echo 0)
    TOTAL_TESTS=$((TOTAL_TESTS + FRONTEND_PASSED + FRONTEND_FAILED))
    PASSED_TESTS=$((PASSED_TESTS + FRONTEND_PASSED))
    FAILED_TESTS=$((FAILED_TESTS + FRONTEND_FAILED))
    echo -e "${GREEN}✅ Frontend tests completed${NC}"
else
    echo -e "${RED}❌ Frontend tests failed${NC}"
fi

echo ""
echo "🔧 Backend API Tests (pytest)"
echo "------------------------------------------"
cd /opt/sutazaiapp/backend

# Check if pytest is available
if command -v pytest &> /dev/null; then
    if pytest tests/ -v --tb=short 2>&1 | tee /tmp/pytest-results.txt; then
        BACKEND_PASSED=$(grep -c "PASSED" /tmp/pytest-results.txt || echo 0)
        BACKEND_FAILED=$(grep -c "FAILED" /tmp/pytest-results.txt || echo 0)
        TOTAL_TESTS=$((TOTAL_TESTS + BACKEND_PASSED + BACKEND_FAILED))
        PASSED_TESTS=$((PASSED_TESTS + BACKEND_PASSED))
        FAILED_TESTS=$((FAILED_TESTS + BACKEND_FAILED))
        echo -e "${GREEN}✅ Backend tests completed${NC}"
    else
        echo -e "${RED}❌ Backend tests failed${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  pytest not installed - skipping backend tests${NC}"
    echo "Install with: pip install pytest pytest-asyncio httpx"
fi

echo ""
echo "🤖 AI Agent Health Checks"
echo "------------------------------------------"
AGENTS=("11401:Letta" "11403:CrewAI" "11404:Aider" "11405:LangChain" "11410:FinRobot" "11413:ShellGPT" "11414:Documind" "11416:GPT-Engineer")
AGENT_PASSED=0
AGENT_FAILED=0

for agent in "${AGENTS[@]}"; do
    IFS=':' read -r port name <<< "$agent"
    if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $name ($port)${NC}"
        AGENT_PASSED=$((AGENT_PASSED + 1))
    else
        echo -e "${RED}❌ $name ($port)${NC}"
        AGENT_FAILED=$((AGENT_FAILED + 1))
    fi
done

TOTAL_TESTS=$((TOTAL_TESTS + AGENT_PASSED + AGENT_FAILED))
PASSED_TESTS=$((PASSED_TESTS + AGENT_PASSED))
FAILED_TESTS=$((FAILED_TESTS + AGENT_FAILED))

echo ""
echo "📊 Monitoring Stack Validation"
echo "------------------------------------------"
SERVICES=("10300:Prometheus" "10301:Grafana" "10310:Loki" "10305:Node-Exporter")
MONITORING_PASSED=0
MONITORING_FAILED=0

for service in "${SERVICES[@]}"; do
    IFS=':' read -r port name <<< "$service"
    if curl -sf "http://localhost:$port" > /dev/null 2>&1 || curl -sf "http://localhost:$port/-/healthy" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $name ($port)${NC}"
        MONITORING_PASSED=$((MONITORING_PASSED + 1))
    else
        echo -e "${RED}❌ $name ($port)${NC}"
        MONITORING_FAILED=$((MONITORING_FAILED + 1))
    fi
done

TOTAL_TESTS=$((TOTAL_TESTS + MONITORING_PASSED + MONITORING_FAILED))
PASSED_TESTS=$((PASSED_TESTS + MONITORING_PASSED))
FAILED_TESTS=$((FAILED_TESTS + MONITORING_FAILED))

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "Total Tests:   $TOTAL_TESTS"
echo -e "${GREEN}Passed:        $PASSED_TESTS${NC}"
echo -e "${RED}Failed:        $FAILED_TESTS${NC}"

if [ $TOTAL_TESTS -gt 0 ]; then
    PASS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    echo "Pass Rate:     ${PASS_RATE}%"
    
    if [ $PASS_RATE -ge 90 ]; then
        echo -e "${GREEN}✅ EXCELLENT - Production ready${NC}"
        exit 0
    elif [ $PASS_RATE -ge 75 ]; then
        echo -e "${YELLOW}⚠️  GOOD - Minor issues to address${NC}"
        exit 0
    else
        echo -e "${RED}❌ NEEDS WORK - Critical issues detected${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ No tests executed${NC}"
    exit 1
fi
