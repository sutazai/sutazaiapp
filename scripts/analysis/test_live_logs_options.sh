#!/bin/bash
# Test all 15 live logs options systematically

SCRIPT_PATH="/opt/sutazaiapp/scripts/monitoring/live_logs.sh"
RESULTS_FILE="/opt/sutazaiapp/docs/index/live_logs_audit.json"

# Initialize JSON results
echo '{' > "$RESULTS_FILE"
echo '  "test_date": "'$(date -Iseconds)'",' >> "$RESULTS_FILE"
echo '  "script_path": "'$SCRIPT_PATH'",' >> "$RESULTS_FILE"
echo '  "options": [' >> "$RESULTS_FILE"

# Function to test an option
test_option() {
    local option_num=$1
    local option_name=$2
    local test_input=$3
    
    echo "Testing Option $option_num: $option_name"
    
    # Capture output and error
    OUTPUT=$(echo "$test_input" | timeout 3 "$SCRIPT_PATH" 2>&1 | head -200)
    EXIT_CODE=$?
    
    # Check if option produces expected output
    if echo "$OUTPUT" | grep -q "╔═"; then
        STATUS="working"
        # Check for specific indicators
        if echo "$OUTPUT" | grep -q "Error\|Failed\|not found\|Cannot"; then
            STATUS="partial"
        fi
    else
        STATUS="broken"
    fi
    
    # Extract error messages if any
    ERROR_MSG=$(echo "$OUTPUT" | grep -i "error\|failed\|not found\|cannot" | head -3 | tr '\n' ' ' | sed 's/"/\\"/g')
    
    # Write JSON entry
    if [ $option_num -gt 1 ]; then
        echo ',' >> "$RESULTS_FILE"
    fi
    
    cat >> "$RESULTS_FILE" << EOF
    {
      "option": $option_num,
      "name": "$option_name",
      "status": "$STATUS",
      "exit_code": $EXIT_CODE,
      "error_messages": "$ERROR_MSG",
      "has_output": $([ -n "$OUTPUT" ] && echo "true" || echo "false")
    }
EOF
}

# Test each option
test_option 1 "System Overview" "1"
test_option 2 "Live Logs (All Services)" "2"
test_option 3 "Test API Endpoints" "3"
test_option 4 "Container Statistics" "4"
test_option 5 "Log Management" "5"
test_option 6 "Debug Controls" "6"
test_option 7 "Database Repair" "7"
test_option 8 "System Repair" "8"
test_option 9 "Restart All Services" "9"
test_option 10 "Unified Live Logs" "10"
test_option 11 "Docker Troubleshooting" "11"
test_option 12 "Redeploy All Containers" "12"
test_option 13 "Smart Health Check" "13"
test_option 14 "Container Health Status" "14"
test_option 15 "Selective Service Deployment" "15"

# Close JSON
echo '' >> "$RESULTS_FILE"
echo '  ]' >> "$RESULTS_FILE"
echo '}' >> "$RESULTS_FILE"

echo "Test complete. Results saved to $RESULTS_FILE"