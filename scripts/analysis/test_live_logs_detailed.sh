#!/bin/bash
# Detailed test of each live logs option

SCRIPT_PATH="/opt/sutazaiapp/scripts/monitoring/live_logs.sh"
OUTPUT_DIR="/opt/sutazaiapp/docs/index/live_logs_tests"

mkdir -p "$OUTPUT_DIR"

echo "===== DETAILED LIVE LOGS TESTING ====="
echo ""

# Test Option 1: System Overview
echo "Testing Option 1: System Overview..."
echo "1" | timeout 3 "$SCRIPT_PATH" 2>&1 > "$OUTPUT_DIR/option_1_output.txt"
if grep -q "SYSTEM OVERVIEW" "$OUTPUT_DIR/option_1_output.txt"; then
    echo "✓ Option 1: Working - Shows system overview"
else
    echo "✗ Option 1: BROKEN - No system overview output"
fi

# Test Option 2: Live Logs
echo "Testing Option 2: Live Logs..."
echo -e "2\n0" | timeout 3 "$SCRIPT_PATH" 2>&1 > "$OUTPUT_DIR/option_2_output.txt"
if grep -q "LIVE LOGS" "$OUTPUT_DIR/option_2_output.txt"; then
    echo "✓ Option 2: Working - Shows live logs menu"
else
    echo "✗ Option 2: BROKEN - No live logs output"
fi

# Test Option 3: API Endpoints
echo "Testing Option 3: Test API Endpoints..."
echo "3" | timeout 3 "$SCRIPT_PATH" 2>&1 > "$OUTPUT_DIR/option_3_output.txt"
if grep -q "API ENDPOINT TESTING" "$OUTPUT_DIR/option_3_output.txt"; then
    echo "✓ Option 3: Working - Tests API endpoints"
    grep -E "✓|✗|Testing:|Health" "$OUTPUT_DIR/option_3_output.txt" | head -10
else
    echo "✗ Option 3: BROKEN - No API testing output"
fi

# Test Option 4: Container Statistics
echo "Testing Option 4: Container Statistics..."
echo "4" | timeout 3 "$SCRIPT_PATH" 2>&1 > "$OUTPUT_DIR/option_4_output.txt"
if grep -q "CONTAINER STATISTICS" "$OUTPUT_DIR/option_4_output.txt"; then
    echo "✓ Option 4: Working - Shows container stats"
else
    echo "✗ Option 4: BROKEN - No container stats"
fi

# Test Option 5: Log Management
echo "Testing Option 5: Log Management..."
echo -e "5\n8" | timeout 3 "$SCRIPT_PATH" 2>&1 > "$OUTPUT_DIR/option_5_output.txt"
if grep -q "LOG MANAGEMENT" "$OUTPUT_DIR/option_5_output.txt"; then
    echo "✓ Option 5: Working - Shows log management menu"
else
    echo "✗ Option 5: BROKEN - No log management menu"
fi

# Test Option 6: Debug Controls
echo "Testing Option 6: Debug Controls..."
echo -e "6\n9" | timeout 3 "$SCRIPT_PATH" 2>&1 > "$OUTPUT_DIR/option_6_output.txt"
if grep -q "DEBUG CONTROLS" "$OUTPUT_DIR/option_6_output.txt"; then
    echo "✓ Option 6: Working - Shows debug controls"
else
    echo "✗ Option 6: BROKEN - No debug controls"
fi

# Test Option 7: Database Repair
echo "Testing Option 7: Database Repair..."
echo "7" | timeout 3 "$SCRIPT_PATH" 2>&1 > "$OUTPUT_DIR/option_7_output.txt"
if grep -q "DATABASE INITIALIZATION" "$OUTPUT_DIR/option_7_output.txt"; then
    echo "✓ Option 7: Working - Initializes database"
else
    echo "✗ Option 7: BROKEN - No database initialization"
fi

# Test Option 8: System Repair
echo "Testing Option 8: System Repair..."
echo "8" | timeout 3 "$SCRIPT_PATH" 2>&1 > "$OUTPUT_DIR/option_8_output.txt"
if grep -q "SYSTEM REPAIR" "$OUTPUT_DIR/option_8_output.txt"; then
    echo "✓ Option 8: Working - Runs system repair"
else
    echo "✗ Option 8: BROKEN - No system repair"
fi

# Test Option 9: Restart Services
echo "Testing Option 9: Restart All Services..."
echo "9" | timeout 3 "$SCRIPT_PATH" 2>&1 > "$OUTPUT_DIR/option_9_output.txt"
if grep -q "Restarting all SutazAI services" "$OUTPUT_DIR/option_9_output.txt"; then
    echo "✓ Option 9: Working - Restarts services"
else
    echo "✗ Option 9: BROKEN - No service restart"
fi

# Test Option 10: Unified Live Logs
echo "Testing Option 10: Unified Live Logs..."
echo "10" | timeout 3 "$SCRIPT_PATH" 2>&1 > "$OUTPUT_DIR/option_10_output.txt"
if grep -q "UNIFIED LIVE LOGS" "$OUTPUT_DIR/option_10_output.txt"; then
    echo "✓ Option 10: Working - Shows unified logs"
    # Check for errors
    if grep -q "Error\|Failed" "$OUTPUT_DIR/option_10_output.txt"; then
        echo "  Warning: Some log errors detected (normal for certain exporters)"
    fi
else
    echo "✗ Option 10: BROKEN - No unified logs"
fi

# Test Option 11: Docker Troubleshooting
echo "Testing Option 11: Docker Troubleshooting..."
echo -e "11\n9" | timeout 3 "$SCRIPT_PATH" 2>&1 > "$OUTPUT_DIR/option_11_output.txt"
if grep -q "DOCKER TROUBLESHOOTING" "$OUTPUT_DIR/option_11_output.txt"; then
    echo "✓ Option 11: Working - Shows Docker troubleshooting menu"
else
    echo "✗ Option 11: BROKEN - No Docker troubleshooting"
fi

# Test Option 12: Redeploy Containers
echo "Testing Option 12: Redeploy All Containers..."
echo "12" | timeout 3 "$SCRIPT_PATH" 2>&1 > "$OUTPUT_DIR/option_12_output.txt"
if grep -q "CONTAINER REDEPLOYMENT" "$OUTPUT_DIR/option_12_output.txt"; then
    echo "✓ Option 12: Working - Shows redeployment process"
else
    echo "✗ Option 12: BROKEN - No redeployment"
fi

# Test Option 13: Smart Health Check
echo "Testing Option 13: Smart Health Check & Repair..."
echo "13" | timeout 3 "$SCRIPT_PATH" 2>&1 > "$OUTPUT_DIR/option_13_output.txt"
if grep -q "SMART HEALTH CHECK" "$OUTPUT_DIR/option_13_output.txt"; then
    echo "✓ Option 13: Working - Performs health check"
else
    echo "✗ Option 13: BROKEN - No health check"
fi

# Test Option 14: Container Health Status
echo "Testing Option 14: Container Health Status..."
echo "14" | timeout 3 "$SCRIPT_PATH" 2>&1 > "$OUTPUT_DIR/option_14_output.txt"
if grep -q "CONTAINER HEALTH STATUS" "$OUTPUT_DIR/option_14_output.txt"; then
    echo "✓ Option 14: Working - Shows container health"
else
    echo "✗ Option 14: BROKEN - No health status"
fi

# Test Option 15: Selective Deployment
echo "Testing Option 15: Selective Service Deployment..."
echo -e "15\nq" | timeout 3 "$SCRIPT_PATH" 2>&1 > "$OUTPUT_DIR/option_15_output.txt"
if grep -q "SELECTIVE SERVICE DEPLOYMENT" "$OUTPUT_DIR/option_15_output.txt"; then
    echo "✓ Option 15: Working - Shows service deployment menu"
    # Show available services
    grep -E "\[Running\]|\[Stopped\]" "$OUTPUT_DIR/option_15_output.txt" | head -5
else
    echo "✗ Option 15: BROKEN - No deployment menu"
fi

echo ""
echo "===== TEST COMPLETE ====="
echo "Output files saved in: $OUTPUT_DIR"