#!/bin/bash

# Test script for live_logs.sh - tests all 15 options
# Usage: ./test_live_logs_all_options.sh

SCRIPT="/opt/sutazaiapp/scripts/monitoring/live_logs.sh"
RESULTS_FILE="/tmp/live_logs_test_results.txt"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Testing SutazAI Live Logs Script - All Options"
echo "==============================================="
echo "" | tee $RESULTS_FILE

# Function to test an option
test_option() {
    local option=$1
    local description=$2
    local timeout_val=${3:-5}
    
    echo -n "Testing Option $option ($description)... "
    echo "Testing Option $option ($description)" >> $RESULTS_FILE
    
    # Run the option with timeout
    if echo "$option" | timeout $timeout_val $SCRIPT 2>&1 | grep -q -E "(╔|║|╚|SUTAZAI|Monitoring|containers|services|Docker)"; then
        echo -e "${GREEN}✓ WORKING${NC}"
        echo "  ✓ WORKING" >> $RESULTS_FILE
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo "  ✗ FAILED" >> $RESULTS_FILE
        return 1
    fi
}

# Test each option
test_option "1" "System Overview" 3
test_option "2" "Live Logs (All Services)" 3
test_option "3" "Test API Endpoints" 5
test_option "4" "Container Statistics" 3
test_option "5" "Log Management" 3
test_option "6" "Debug Controls" 3
test_option "7" "Database Repair" 3
test_option "8" "System Repair" 3
test_option "9" "Restart All Services" 3
test_option "10" "Unified Live Logs" 3
test_option "11" "Docker Troubleshooting" 3
test_option "12" "Redeploy All Containers" 3
test_option "13" "Smart Health Check" 3
test_option "14" "Container Health Status" 3
test_option "15" "Selective Service Deploy" 3

echo ""
echo "================================"
echo "Test Results Summary:"
echo "================================"

# Count results
working=$(grep -c "✓ WORKING" $RESULTS_FILE)
failed=$(grep -c "✗ FAILED" $RESULTS_FILE)
total=$((working + failed))

echo -e "Working: ${GREEN}$working/$total${NC}"
echo -e "Failed: ${RED}$failed/$total${NC}"

# Show detailed results
echo ""
echo "Detailed Results:"
cat $RESULTS_FILE

# Final verdict
echo ""
if [ $failed -eq 0 ]; then
    echo -e "${GREEN}✓ ALL OPTIONS WORKING!${NC}"
else
    echo -e "${YELLOW}⚠ $failed options need fixing${NC}"
fi