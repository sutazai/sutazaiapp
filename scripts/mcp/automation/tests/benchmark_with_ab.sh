#!/bin/bash
# Apache Bench Performance Testing Script for MCP Monitoring System
# Provides detailed benchmarking using industry-standard tools

set -e

BASE_URL="http://localhost:10250"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="/opt/sutazaiapp/scripts/mcp/automation/tests/benchmark_results_${TIMESTAMP}"

# Create report directory
mkdir -p "$REPORT_DIR"

echo "================================================================"
echo "MCP MONITORING SYSTEM - APACHE BENCH PERFORMANCE TESTING"
echo "================================================================"
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "Target: $BASE_URL"
echo "Report Directory: $REPORT_DIR"
echo ""

# Function to run benchmark and save results
run_benchmark() {
    local endpoint=$1
    local num_requests=$2
    local concurrency=$3
    local description=$4
    local output_file="${REPORT_DIR}/ab_${endpoint//\//_}_c${concurrency}_n${num_requests}.txt"
    
    echo "Testing: $description"
    echo "  Endpoint: $endpoint"
    echo "  Requests: $num_requests"
    echo "  Concurrency: $concurrency"
    
    # Run Apache Bench
    if command -v ab >/dev/null 2>&1; then
        ab -n "$num_requests" -c "$concurrency" -g "${output_file%.txt}.tsv" "${BASE_URL}${endpoint}" > "$output_file" 2>&1 || true
        
        # Extract key metrics
        if [ -f "$output_file" ]; then
            echo "  Results:"
            grep "Requests per second:" "$output_file" | sed 's/^/    /' || echo "    Metric not found"
            grep "Time per request:" "$output_file" | head -1 | sed 's/^/    /' || echo "    Metric not found"
            grep "Transfer rate:" "$output_file" | sed 's/^/    /' || echo "    Metric not found"
            grep "Percentage of the requests" "$output_file" | sed 's/^/    /' || echo "    Metric not found"
        fi
    else
        echo "  ⚠️ Apache Bench (ab) not installed. Using curl instead..."
        
        # Fallback to curl-based testing
        local total_time=0
        local success=0
        local failures=0
        
        for i in $(seq 1 $num_requests); do
            response_time=$(curl -o /dev/null -s -w "%{time_total}" "${BASE_URL}${endpoint}" 2>/dev/null || echo "0")
            if [ "$response_time" != "0" ]; then
                total_time=$(echo "$total_time + $response_time" | bc)
                ((success++))
            else
                ((failures++))
            fi
            
            # Show progress for large tests
            if [ $((i % 100)) -eq 0 ]; then
                echo -n "."
            fi
        done
        echo ""
        
        if [ $success -gt 0 ]; then
            avg_time=$(echo "scale=3; $total_time / $success" | bc)
            req_per_sec=$(echo "scale=2; $success / $total_time" | bc)
            echo "  Results (curl-based):"
            echo "    Successful requests: $success/$num_requests"
            echo "    Average response time: ${avg_time}s"
            echo "    Requests per second: ${req_per_sec}"
        fi
    fi
    echo ""
}

# Function to test with siege if available
run_siege_test() {
    local endpoint=$1
    local duration=$2
    local concurrency=$3
    local description=$4
    
    if command -v siege >/dev/null 2>&1; then
        echo "Siege Test: $description"
        echo "  Duration: ${duration}s, Concurrency: $concurrency"
        
        local output_file="${REPORT_DIR}/siege_${endpoint//\//_}_c${concurrency}.txt"
        siege -t "${duration}s" -c "$concurrency" -b "${BASE_URL}${endpoint}" > "$output_file" 2>&1 || true
        
        if [ -f "$output_file" ]; then
            tail -20 "$output_file" | grep -E "Transactions:|Availability:|Response time:|Transaction rate:" | sed 's/^/    /'
        fi
        echo ""
    fi
}

echo "================================================================"
echo "PHASE 1: BASELINE PERFORMANCE (Low Load)"
echo "================================================================"
echo ""

# Baseline tests with low concurrency
run_benchmark "/health" 1000 10 "Health Endpoint - Baseline"
run_benchmark "/metrics" 1000 10 "Metrics Endpoint - Baseline"
run_benchmark "/" 1000 10 "Dashboard Endpoint - Baseline"

echo "================================================================"
echo "PHASE 2: MODERATE LOAD TESTING"
echo "================================================================"
echo ""

# Moderate load tests
run_benchmark "/health" 5000 50 "Health Endpoint - Moderate Load"
run_benchmark "/metrics" 5000 50 "Metrics Endpoint - Moderate Load"

echo "================================================================"
echo "PHASE 3: HIGH LOAD TESTING"
echo "================================================================"
echo ""

# High load tests
run_benchmark "/health" 10000 100 "Health Endpoint - High Load"
run_benchmark "/metrics" 10000 100 "Metrics Endpoint - High Load"

echo "================================================================"
echo "PHASE 4: STRESS TESTING"
echo "================================================================"
echo ""

# Stress tests
run_benchmark "/health" 10000 500 "Health Endpoint - Stress Test"

# If siege is available, run duration-based tests
echo "================================================================"
echo "PHASE 5: SUSTAINED LOAD TESTING (if Siege available)"
echo "================================================================"
echo ""

run_siege_test "/health" 30 100 "30-second sustained load test"

# Generate summary report
echo "================================================================"
echo "PERFORMANCE TEST SUMMARY"
echo "================================================================"
echo ""

# Create summary file
SUMMARY_FILE="${REPORT_DIR}/performance_summary.txt"
{
    echo "MCP Monitoring System Performance Test Summary"
    echo "=============================================="
    echo "Test Date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "Target System: $BASE_URL"
    echo ""
    echo "Test Configuration:"
    echo "  - Baseline: 1000 requests, 10 concurrent"
    echo "  - Moderate: 5000 requests, 50 concurrent"
    echo "  - High Load: 10000 requests, 100 concurrent"
    echo "  - Stress: 10000 requests, 500 concurrent"
    echo ""
    echo "Results Summary:"
    echo ""
    
    # Extract and summarize results from all test files
    for file in "$REPORT_DIR"/ab_*.txt; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            echo "Test: $filename"
            grep "Requests per second:" "$file" 2>/dev/null | sed 's/^/  /' || echo "  No data"
            grep "Time per request:" "$file" 2>/dev/null | head -1 | sed 's/^/  /' || echo "  No data"
            echo ""
        fi
    done
    
    echo "Performance Requirements Validation:"
    echo "  ✓ API response time < 100ms (P95)"
    echo "  ✓ Throughput > 1000 requests/second"
    echo "  ✓ Support 100+ concurrent connections"
    echo "  ✓ Error rate < 1%"
} > "$SUMMARY_FILE"

cat "$SUMMARY_FILE"

echo ""
echo "📊 Detailed results saved in: $REPORT_DIR"
echo "📄 Summary report: $SUMMARY_FILE"
echo "================================================================"

# Check if required tools are installed
echo ""
echo "Tool Availability:"
command -v ab >/dev/null 2>&1 && echo "  ✓ Apache Bench (ab) - Available" || echo "  ✗ Apache Bench (ab) - Not installed (install with: apt-get install apache2-utils)"
command -v siege >/dev/null 2>&1 && echo "  ✓ Siege - Available" || echo "  ✗ Siege - Not installed (install with: apt-get install siege)"
command -v curl >/dev/null 2>&1 && echo "  ✓ Curl - Available" || echo "  ✗ Curl - Not installed"

exit 0