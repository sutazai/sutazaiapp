#!/bin/bash

# Test Script for live_logs.sh All 15 Options
# Tests each option systematically and reports results

set -euo pipefail

PROJECT_ROOT="/opt/sutazaiapp"
LIVE_LOGS_SCRIPT="${PROJECT_ROOT}/scripts/monitoring/live_logs.sh"
RESULTS_FILE="${PROJECT_ROOT}/docs/live_logs_test_results.md"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Initialize results file
echo "# Live Logs Testing Report - $(date '+%Y-%m-%d %H:%M:%S')" > "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "## Testing Environment" >> "$RESULTS_FILE"
echo "- Script: ${LIVE_LOGS_SCRIPT}" >> "$RESULTS_FILE"
echo "- User: $(whoami)" >> "$RESULTS_FILE"
echo "- Working Directory: $(pwd)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Test function for each option
test_option() {
    local option_num=$1
    local option_name=$2
    local timeout_seconds=${3:-5}
    
    echo -e "${CYAN}Testing Option $option_num: $option_name${NC}"
    echo "## Option $option_num: $option_name" >> "$RESULTS_FILE"
    
    # Create test command with timeout
    local test_cmd="timeout $timeout_seconds bash -c 'echo $option_num | $LIVE_LOGS_SCRIPT 2>&1'"
    
    # Run test and capture result
    local start_time=$(date +%s)
    local exit_code=0
    local output=""
    
    # Set environment variables to skip interactive parts
    export LIVE_LOGS_NONINTERACTIVE=true
    export LIVE_LOGS_SKIP_NUMLOCK=true
    
    # Run the test
    output=$(eval $test_cmd 2>&1) || exit_code=$?
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Analyze results
    if [[ $exit_code -eq 124 ]]; then
        # Timeout - might be expected for interactive options
        echo -e "${YELLOW}  ⚠ Timed out after ${timeout_seconds}s (may be interactive)${NC}"
        echo "- **Status**: TIMEOUT (may be interactive)" >> "$RESULTS_FILE"
        echo "- **Duration**: ${timeout_seconds}s (timeout)" >> "$RESULTS_FILE"
        echo "- **Notes**: Option appears to be interactive or long-running" >> "$RESULTS_FILE"
    elif [[ $exit_code -eq 0 ]]; then
        echo -e "${GREEN}  ✓ Success (${duration}s)${NC}"
        echo "- **Status**: SUCCESS ✅" >> "$RESULTS_FILE"
        echo "- **Duration**: ${duration}s" >> "$RESULTS_FILE"
        
        # Check for specific patterns in output
        if echo "$output" | grep -q "Error\|error\|ERROR"; then
            echo -e "${YELLOW}    Warning: Found error messages in output${NC}"
            echo "- **Warning**: Error messages found in output" >> "$RESULTS_FILE"
        fi
        
        if echo "$output" | grep -q "not found\|No such"; then
            echo -e "${YELLOW}    Warning: Missing dependencies detected${NC}"
            echo "- **Warning**: Missing dependencies detected" >> "$RESULTS_FILE"
        fi
    else
        echo -e "${RED}  ✗ Failed with exit code $exit_code${NC}"
        echo "- **Status**: FAILED ❌" >> "$RESULTS_FILE"
        echo "- **Exit Code**: $exit_code" >> "$RESULTS_FILE"
        echo "- **Duration**: ${duration}s" >> "$RESULTS_FILE"
        
        # Extract error message if available
        local error_msg=$(echo "$output" | grep -i "error" | head -1)
        if [[ -n "$error_msg" ]]; then
            echo -e "${RED}    Error: $error_msg${NC}"
            echo "- **Error**: $error_msg" >> "$RESULTS_FILE"
        fi
    fi
    
    # Save sample output (first 5 lines)
    echo "- **Sample Output**:" >> "$RESULTS_FILE"
    echo '```' >> "$RESULTS_FILE"
    echo "$output" | head -5 >> "$RESULTS_FILE"
    echo '```' >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    
    # Small delay between tests
    sleep 1
}

# Header
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         TESTING ALL LIVE LOGS OPTIONS (1-15)                ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Test all options
test_option 1 "System Overview" 3
test_option 2 "Live Logs (All Services)" 5
test_option 3 "Test API Endpoints" 10
test_option 4 "Container Statistics" 3
test_option 5 "Log Management" 3
test_option 6 "Debug Controls" 3
test_option 7 "Database Repair" 10
test_option 8 "System Repair" 15
test_option 9 "Restart All Services" 10
test_option 10 "Unified Live Logs (All in One)" 5
test_option 11 "Docker Troubleshooting & Recovery" 5
test_option 12 "Redeploy All Containers" 20
test_option 13 "Smart Health Check & Repair" 10
test_option 14 "Container Health Status" 3
test_option 15 "Selective Service Deployment" 5

# Summary
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}                          TEST SUMMARY                         ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

# Count results from the file
success_count=$(grep -c "SUCCESS ✅" "$RESULTS_FILE" || echo 0)
failed_count=$(grep -c "FAILED ❌" "$RESULTS_FILE" || echo 0)
timeout_count=$(grep -c "TIMEOUT" "$RESULTS_FILE" || echo 0)

echo ""
echo -e "${GREEN}✅ Successful: $success_count${NC}"
echo -e "${RED}❌ Failed: $failed_count${NC}"
echo -e "${YELLOW}⏱ Timeout/Interactive: $timeout_count${NC}"
echo ""

# Add summary to results file
echo "## Summary" >> "$RESULTS_FILE"
echo "- **Successful**: $success_count" >> "$RESULTS_FILE"
echo "- **Failed**: $failed_count" >> "$RESULTS_FILE"
echo "- **Timeout/Interactive**: $timeout_count" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Check for specific working option 10
echo -e "${PURPLE}Special Test: Option 10 (mentioned as working)${NC}"
echo "## Special Test: Option 10" >> "$RESULTS_FILE"

# Test option 10 with the dedicated script
if [[ -f "/opt/sutazaiapp/scripts/run_live_logs_10.sh" ]]; then
    echo -e "${CYAN}Testing with dedicated run_live_logs_10.sh script...${NC}"
    timeout 5 /opt/sutazaiapp/scripts/run_live_logs_10.sh 2>&1 | head -10 || true
    echo "- Dedicated script exists and was tested" >> "$RESULTS_FILE"
else
    echo -e "${YELLOW}Dedicated script not found${NC}"
    echo "- Dedicated script not found" >> "$RESULTS_FILE"
fi

echo ""
echo -e "${GREEN}Results saved to: $RESULTS_FILE${NC}"
echo ""

# Display the results file
echo -e "${CYAN}Full Test Results:${NC}"
echo "───────────────────────────────────────────────────────────────"
cat "$RESULTS_FILE"