#!/bin/bash

# Comprehensive Test Execution Script
# Runs all backend and frontend tests with proper reporting

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  SUTAZAI COMPREHENSIVE TEST SUITE EXECUTION"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Create results directory
mkdir -p test-results
RESULTS_DIR="test-results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}Results will be saved to: $RESULTS_DIR${NC}"
echo ""

# Function to print section header
print_section() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${YELLOW}$1${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Function to run test and capture results
run_test_suite() {
    local name=$1
    local command=$2
    local output_file="$RESULTS_DIR/${name// /_}.log"
    
    echo -e "${BLUE}Running: $name${NC}"
    echo "Command: $command"
    echo ""
    
    if eval "$command" > "$output_file" 2>&1; then
        echo -e "${GREEN}✓ $name PASSED${NC}"
        ((PASSED_TESTS++))
    else
        echo -e "${RED}✗ $name FAILED${NC}"
        echo "  See: $output_file"
        ((FAILED_TESTS++))
    fi
    
    ((TOTAL_TESTS++))
}

# ═══════════════════════════════════════════════════════════════
# PHASE 1: BACKEND TESTS
# ═══════════════════════════════════════════════════════════════

print_section "PHASE 1: Backend Python Tests"

echo "Checking Python environment..."
cd /opt/sutazaiapp/backend

if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing test dependencies..."
pip install -q pytest pytest-asyncio httpx 2>&1 | grep -v "already satisfied" || true

echo ""
echo "Backend test files:"
ls -1 tests/test_*.py

echo ""

# Run each backend test suite
print_section "Backend API Tests"
run_test_suite "API Endpoints" "pytest tests/test_api_endpoints.py -v --tb=short"

print_section "Authentication Tests"
run_test_suite "Auth & JWT" "pytest tests/test_auth.py -v --tb=short"

print_section "MCP Bridge Tests"
run_test_suite "MCP Bridge Integration" "pytest tests/test_mcp_bridge.py -v --tb=short"

print_section "AI Agents Tests"
run_test_suite "AI Agents Health" "pytest tests/test_ai_agents.py -v --tb=short"

print_section "Database Tests"
run_test_suite "Database Integration" "pytest tests/test_databases.py -v --tb=short"

print_section "Monitoring Tests"
run_test_suite "Monitoring Stack" "pytest tests/test_monitoring.py -v --tb=short"

print_section "Security Tests"
run_test_suite "Security Suite" "pytest tests/test_security.py -v --tb=short"

print_section "Performance Tests"
run_test_suite "Performance & Load" "pytest tests/test_performance.py -v --tb=short"

print_section "Infrastructure Tests"
run_test_suite "RabbitMQ/Consul/Kong" "pytest tests/test_rabbitmq_consul_kong.py -v --tb=short"

print_section "Infrastructure Container Tests"
run_test_suite "Container Health" "pytest tests/test_infrastructure.py -v --tb=short"

print_section "E2E Workflow Tests"
run_test_suite "End-to-End Workflows" "pytest tests/test_e2e_workflows.py -v --tb=short"

deactivate

# ═══════════════════════════════════════════════════════════════
# PHASE 2: FRONTEND TESTS
# ═══════════════════════════════════════════════════════════════

print_section "PHASE 2: Frontend Playwright Tests"

cd /opt/sutazaiapp/frontend

echo "Checking Node.js environment..."
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing Node.js dependencies...${NC}"
    npm install
fi

echo ""
echo "Frontend test files:"
ls -1 tests/e2e/*.spec.ts

echo ""

# Run Playwright tests
print_section "Playwright E2E Tests"

run_test_suite "Basic UI Tests" "npx playwright test tests/e2e/jarvis-basic.spec.ts"
run_test_suite "Chat Interface Tests" "npx playwright test tests/e2e/jarvis-chat.spec.ts"
run_test_suite "WebSocket Tests" "npx playwright test tests/e2e/jarvis-websocket.spec.ts"
run_test_suite "Model Selection Tests" "npx playwright test tests/e2e/jarvis-models.spec.ts"
run_test_suite "Voice Interface Tests" "npx playwright test tests/e2e/jarvis-voice.spec.ts"
run_test_suite "Advanced UI Tests" "npx playwright test tests/e2e/jarvis-ui.spec.ts"
run_test_suite "Integration Tests" "npx playwright test tests/e2e/jarvis-integration.spec.ts"
run_test_suite "Advanced Features Tests" "npx playwright test tests/e2e/jarvis-enhanced-features.spec.ts"
run_test_suite "Security & Performance Tests" "npx playwright test tests/e2e/jarvis-advanced.spec.ts"

# ═══════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════

print_section "TEST EXECUTION SUMMARY"

echo ""
echo "Total Test Suites: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "${RED}Failed: $FAILED_TESTS${NC}"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ✓ ALL TESTS PASSED!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
else
    echo -e "${RED}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}  ✗ SOME TESTS FAILED - CHECK LOGS${NC}"
    echo -e "${RED}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Failed test logs in: $RESULTS_DIR"
fi

echo ""
echo "Detailed results saved to: $RESULTS_DIR"
echo ""

# Generate summary report
cat > "$RESULTS_DIR/summary.txt" << EOF
SUTAZAI COMPREHENSIVE TEST SUITE RESULTS
========================================
Execution Time: $(date)

SUMMARY:
- Total Test Suites: $TOTAL_TESTS
- Passed: $PASSED_TESTS
- Failed: $FAILED_TESTS
- Success Rate: $(echo "scale=2; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc)%

BACKEND TESTS:
- API Endpoints
- Authentication & JWT
- MCP Bridge Integration  
- AI Agents Health
- Database Integration
- Monitoring Stack
- Security Suite
- Performance & Load
- RabbitMQ/Consul/Kong
- Container Health
- E2E Workflows

FRONTEND TESTS:
- Basic UI
- Chat Interface
- WebSocket Communication
- Model Selection
- Voice Interface
- Advanced UI Features
- Integration Tests
- Enhanced Features
- Security & Performance

See individual log files for detailed results.
EOF

echo "Summary report: $RESULTS_DIR/summary.txt"
cat "$RESULTS_DIR/summary.txt"

exit $FAILED_TESTS
