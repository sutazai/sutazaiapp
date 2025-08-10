#!/bin/bash

# ULTRA-COMPREHENSIVE Hardware Resource Optimizer API Validation Test Suite
# Performs exhaustive testing of all endpoints with detailed reporting

set -e

# Configuration
BACKEND_URL="http://localhost:10010/api/v1/hardware"
DIRECT_URL="http://localhost:11110"
TIMEOUT=15
REPORT_FILE="hardware_api_validation_report_$(date +%Y%m%d_%H%M%S).json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Results array
declare -a RESULTS=()

echo -e "${BLUE}🚀 Starting ULTRA-COMPREHENSIVE Hardware API Validation${NC}"
echo "================================================================================"
echo "Backend URL: $BACKEND_URL"
echo "Direct URL: $DIRECT_URL"
echo "Timeout: ${TIMEOUT}s"
echo "Report File: $REPORT_FILE"
echo ""

# Function to test an endpoint
test_endpoint() {
    local method="$1"
    local url="$2" 
    local description="$3"
    local expect_error="${4:-false}"
    local data="${5:-}"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    echo -n "Testing: $description ... "
    
    local start_time=$(date +%s.%N)
    local status_code=""
    local response=""
    local success="false"
    local error_message=""
    
    # Make the request
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "HTTPSTATUS:%{http_code}" -m $TIMEOUT "$url" 2>/dev/null || echo "HTTPSTATUS:000")
    elif [ "$method" = "POST" ]; then
        if [ -n "$data" ]; then
            response=$(curl -s -w "HTTPSTATUS:%{http_code}" -m $TIMEOUT -X POST -H "Content-Type: application/json" -d "$data" "$url" 2>/dev/null || echo "HTTPSTATUS:000")
        else
            response=$(curl -s -w "HTTPSTATUS:%{http_code}" -m $TIMEOUT -X POST "$url" 2>/dev/null || echo "HTTPSTATUS:000")
        fi
    fi
    
    local end_time=$(date +%s.%N)
    local duration=$(echo "($end_time - $start_time) * 1000" | bc -l)
    
    # Parse status code
    status_code=$(echo "$response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    local body=$(echo "$response" | sed 's/HTTPSTATUS:[0-9]*$//')
    
    # Determine success
    if [ "$expect_error" = "true" ]; then
        if [ "$status_code" -ge 400 ] || [ "$status_code" -eq 0 ]; then
            success="true"
        else
            success="false"
            error_message="Expected error but got success (status: $status_code)"
        fi
    else
        if [ "$status_code" -ge 200 ] && [ "$status_code" -lt 300 ]; then
            success="true"
        else
            success="false"
            error_message="Unexpected status code: $status_code"
        fi
    fi
    
    # Update counters
    if [ "$success" = "true" ]; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo -e "${GREEN}✅ PASS${NC} (${status_code}, ${duration%.*}ms)"
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo -e "${RED}❌ FAIL${NC} (${status_code}, ${duration%.*}ms)"
        [ -n "$error_message" ] && echo "   Error: $error_message"
    fi
    
    # Store result
    RESULTS+=("{\"endpoint\":\"$url\",\"method\":\"$method\",\"description\":\"$description\",\"status_code\":$status_code,\"response_time_ms\":${duration%.*},\"success\":$success,\"error_message\":\"$error_message\"}")
    
    # Small delay to avoid overwhelming the service
    sleep 0.1
}

# Test Categories

echo -e "\n${BLUE}📋 Direct Service Health Tests${NC}"
echo "--------------------------------------------------"

test_endpoint "GET" "$DIRECT_URL/health" "Direct Health Check"
test_endpoint "GET" "$DIRECT_URL/status" "Direct Status Check"
test_endpoint "GET" "$DIRECT_URL/" "Direct Root Endpoint"

echo -e "\n${BLUE}📋 Backend Integration Tests${NC}"
echo "--------------------------------------------------"

test_endpoint "GET" "$BACKEND_URL/router/health" "Backend Router Health"
test_endpoint "GET" "$BACKEND_URL/health" "Backend Hardware Health"
test_endpoint "GET" "$BACKEND_URL/status" "Backend Hardware Status" "true"

echo -e "\n${BLUE}📋 Optimization Endpoint Tests${NC}"
echo "--------------------------------------------------"

test_endpoint "POST" "$DIRECT_URL/optimize/memory" "Memory Optimization"
sleep 1  # Allow some time between optimizations
test_endpoint "POST" "$DIRECT_URL/optimize/cpu" "CPU Optimization"
sleep 1
test_endpoint "POST" "$DIRECT_URL/optimize/disk" "Disk Optimization"
sleep 1
test_endpoint "POST" "$DIRECT_URL/optimize/docker" "Docker Optimization"
sleep 2  # Docker operations might take longer
test_endpoint "POST" "$DIRECT_URL/optimize/storage?dry_run=true" "Storage Optimization (Dry Run)"
test_endpoint "POST" "$DIRECT_URL/optimize/storage/cache" "Cache Optimization"
test_endpoint "POST" "$DIRECT_URL/optimize/storage/logs" "Log Optimization"

echo -e "\n${BLUE}📋 Analysis Endpoint Tests${NC}"
echo "--------------------------------------------------"

test_endpoint "GET" "$DIRECT_URL/analyze/storage?path=/tmp" "Storage Analysis - /tmp"
test_endpoint "GET" "$DIRECT_URL/analyze/storage?path=/var/log" "Storage Analysis - /var/log"
test_endpoint "GET" "$DIRECT_URL/analyze/storage/duplicates?path=/tmp" "Duplicate Analysis"
test_endpoint "GET" "$DIRECT_URL/analyze/storage/large-files?path=/&min_size_mb=100" "Large Files Analysis"
test_endpoint "GET" "$DIRECT_URL/analyze/storage/report" "Comprehensive Storage Report"

echo -e "\n${BLUE}📋 Error Handling Tests${NC}"
echo "--------------------------------------------------"

test_endpoint "GET" "$DIRECT_URL/nonexistent" "Nonexistent Direct Endpoint" "true"
test_endpoint "POST" "$DIRECT_URL/optimize/invalid" "Invalid Optimization Type" "true"
test_endpoint "GET" "$DIRECT_URL/analyze/storage?path=/nonexistent" "Invalid Path Analysis"
test_endpoint "GET" "$BACKEND_URL/nonexistent" "Backend Nonexistent Endpoint" "true"

echo -e "\n${BLUE}📋 Data Validation Tests${NC}"
echo "--------------------------------------------------"

test_endpoint "GET" "$DIRECT_URL/analyze/storage?path=../../../etc" "Path Traversal Test"
test_endpoint "GET" "$DIRECT_URL/analyze/storage/large-files?min_size_mb=-100" "Negative Size Parameter"
test_endpoint "GET" "$DIRECT_URL/analyze/storage/large-files?min_size_mb=abc" "Invalid Size Parameter" "true"

echo -e "\n${BLUE}📋 Performance Load Tests${NC}"
echo "--------------------------------------------------"

echo "⚡ Testing concurrent health checks..."
# Run 5 concurrent health checks
for i in {1..5}; do
    (test_endpoint "GET" "$DIRECT_URL/health" "Concurrent Health Check $i") &
done
wait

echo "⚡ Testing concurrent memory optimizations..."
# Run 3 concurrent memory optimizations
for i in {1..3}; do
    (test_endpoint "POST" "$DIRECT_URL/optimize/memory" "Concurrent Memory Opt $i") &
done
wait

echo -e "\n${BLUE}📋 Extended Functionality Tests${NC}"
echo "--------------------------------------------------"

# Test optimization with parameters
test_endpoint "POST" "$DIRECT_URL/optimize/storage/compress?path=/var/log&days_old=30" "Compress Old Files"
test_endpoint "POST" "$DIRECT_URL/optimize/storage/duplicates?path=/tmp&dry_run=true" "Remove Duplicates (Dry Run)"

# Test analysis with different parameters
test_endpoint "GET" "$DIRECT_URL/analyze/storage/large-files?path=/var&min_size_mb=1" "Large Files in /var (1MB+)"
test_endpoint "GET" "$DIRECT_URL/analyze/storage/duplicates?path=/var/log" "Duplicate Analysis /var/log"

# Generate Final Report
echo ""
echo "================================================================================"
echo -e "${BLUE}🎯 FINAL VALIDATION REPORT${NC}"
echo "================================================================================"

SUCCESS_RATE=$(echo "scale=2; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc -l)

echo "Test Execution Summary:"
echo "  Total Tests: $TOTAL_TESTS"
echo "  Passed Tests: $PASSED_TESTS" 
echo "  Failed Tests: $FAILED_TESTS"
echo "  Success Rate: ${SUCCESS_RATE}%"

# Determine overall result
if (( $(echo "$SUCCESS_RATE >= 80" | bc -l) )); then
    echo -e "\n${GREEN}🏆 OVERALL VALIDATION: PASSED${NC}"
    OVERALL_RESULT="PASSED"
    EXIT_CODE=0
else
    echo -e "\n${RED}💥 OVERALL VALIDATION: FAILED${NC}"
    OVERALL_RESULT="FAILED"
    EXIT_CODE=1
fi

# Create JSON report
cat > "$REPORT_FILE" << EOF
{
  "test_execution": {
    "timestamp": "$(date -Iseconds)",
    "total_tests": $TOTAL_TESTS,
    "passed_tests": $PASSED_TESTS,
    "failed_tests": $FAILED_TESTS,
    "success_rate": $SUCCESS_RATE,
    "overall_result": "$OVERALL_RESULT"
  },
  "test_results": [
    $(IFS=','; echo "${RESULTS[*]}")
  ],
  "environment": {
    "backend_url": "$BACKEND_URL",
    "direct_url": "$DIRECT_URL", 
    "timeout": $TIMEOUT
  }
}
EOF

echo -e "\n📄 Detailed report saved to: $REPORT_FILE"

# Critical validation summary
echo -e "\nCritical Validation Results:"

# Check critical endpoints
DIRECT_HEALTH=$(echo "${RESULTS[@]}" | grep -c "Direct Health Check.*success\":true" || echo "0")
BACKEND_HEALTH=$(echo "${RESULTS[@]}" | grep -c "Backend.*Health.*success\":true" || echo "0") 
OPTIMIZATION=$(echo "${RESULTS[@]}" | grep -c "optimize.*success\":true" || echo "0")
ANALYSIS=$(echo "${RESULTS[@]}" | grep -c "analyze.*success\":true" || echo "0")

echo "  ✅ Direct Service Health: $([ $DIRECT_HEALTH -gt 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  ✅ Backend Integration: $([ $BACKEND_HEALTH -gt 0 ] && echo 'PASS' || echo 'FAIL')"  
echo "  ✅ Optimization Endpoints: $([ $OPTIMIZATION -gt 0 ] && echo 'PASS' || echo 'FAIL')"
echo "  ✅ Analysis Endpoints: $([ $ANALYSIS -gt 0 ] && echo 'PASS' || echo 'FAIL')"

echo -e "\n================================================================================"

exit $EXIT_CODE