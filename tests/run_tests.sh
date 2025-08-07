#!/bin/bash
"""
Test runner script for agent implementations
"""

echo "========================================="
echo "Running Agent Implementation Tests"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Change to project root
cd /opt/sutazaiapp

# Install test dependencies
echo -e "${YELLOW}Installing test dependencies...${NC}"
pip install -q -r tests/requirements-test.txt

# Run unit tests with coverage
echo -e "\n${YELLOW}Running unit tests...${NC}"
pytest tests/test_ai_agent_orchestrator.py \
       tests/test_task_assignment_coordinator.py \
       tests/test_resource_arbitration_agent.py \
       -v \
       --cov=agents \
       --cov-report=term-missing \
       --cov-report=html:tests/coverage_html \
       -m "not integration"

UNIT_TEST_RESULT=$?

# Run integration tests (if requested)
if [ "$1" == "--integration" ]; then
    echo -e "\n${YELLOW}Running integration tests...${NC}"
    pytest tests/test_ai_agent_orchestrator.py \
           tests/test_task_assignment_coordinator.py \
           tests/test_resource_arbitration_agent.py \
           -v \
           -m "integration"
    
    INTEGRATION_TEST_RESULT=$?
else
    echo -e "\n${YELLOW}Skipping integration tests (use --integration to run)${NC}"
    INTEGRATION_TEST_RESULT=0
fi

# Summary
echo -e "\n========================================="
echo "Test Results Summary"
echo "========================================="

if [ $UNIT_TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ Unit tests passed${NC}"
else
    echo -e "${RED}✗ Unit tests failed${NC}"
fi

if [ "$1" == "--integration" ]; then
    if [ $INTEGRATION_TEST_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ Integration tests passed${NC}"
    else
        echo -e "${RED}✗ Integration tests failed${NC}"
    fi
fi

echo -e "\nCoverage report available at: tests/coverage_html/index.html"

# Exit with appropriate code
if [ $UNIT_TEST_RESULT -ne 0 ] || [ $INTEGRATION_TEST_RESULT -ne 0 ]; then
    exit 1
fi

exit 0