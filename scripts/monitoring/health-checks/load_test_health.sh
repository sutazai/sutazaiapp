#!/bin/bash
"""
ULTRADEEP Load Test for Backend Health Endpoint
Tests health endpoint under extreme load to identify timeout issues
"""

echo "🚀 ULTRADEEP Backend Health Endpoint Load Test"
echo "=" | tr ' ' '=' | head -c 60; echo

# Function to run load test
run_load_test() {
    local concurrent=$1
    local total=$2
    local description=$3
    
    echo "🧪 $description ($concurrent concurrent, $total total)"
    echo "-" | tr ' ' '-' | head -c 60; echo
    
    local success=0
    local timeout=0
    local error=0
    local times=()
    
    start_time=$(date +%s%3N)
    
    # Create temporary files for results
    local temp_dir=$(mktemp -d)
    
    # Run requests in batches
    local batch_size=$concurrent
    local batch_count=$((total / batch_size))
    
    for ((batch=1; batch<=batch_count; batch++)); do
        echo "🚀 Running batch $batch..."
        
        # Start batch of concurrent requests
        for ((i=1; i<=batch_size; i++)); do
            {
                local request_start=$(date +%s%3N)
                local response=$(curl -s -w "HTTPCODE:%{http_code};TIME:%{time_total}s" -m 10 http://localhost:10010/health 2>/dev/null)
                local request_end=$(date +%s%3N)
                local request_time=$((request_end - request_start))
                
                echo "$response;REALTIME:${request_time}ms" >> "$temp_dir/batch_$batch"
            } &
        done
        
        # Wait for batch to complete
        wait
        
        # Process batch results
        local batch_success=0
        local batch_timeout=0
        local batch_error=0
        
        if [[ -f "$temp_dir/batch_$batch" ]]; then
            while IFS= read -r line; do
                if [[ $line == *"HTTPCODE:200"* ]]; then
                    ((batch_success++))
                    # Extract time
                    if [[ $line =~ REALTIME:([0-9]+)ms ]]; then
                        times+=(${BASH_REMATCH[1]})
                    fi
                elif [[ $line == *"HTTPCODE:000"* ]] || [[ $line == "" ]]; then
                    ((batch_timeout++))
                else
                    ((batch_error++))
                fi
            done < "$temp_dir/batch_$batch"
        fi
        
        ((success += batch_success))
        ((timeout += batch_timeout))
        ((error += batch_error))
        
        echo "   ✅ Success: $batch_success, ⏰ Timeouts: $batch_timeout, ❌ Errors: $batch_error"
        
        # Brief pause between batches
        if [[ $batch -lt $batch_count ]]; then
            sleep 0.1
        fi
    done
    
    end_time=$(date +%s%3N)
    total_time=$((end_time - start_time))
    
    # Calculate statistics
    local avg_time=0
    local min_time=999999
    local max_time=0
    
    if [[ ${#times[@]} -gt 0 ]]; then
        local sum=0
        for time in "${times[@]}"; do
            ((sum += time))
            if [[ $time -lt $min_time ]]; then
                min_time=$time
            fi
            if [[ $time -gt $max_time ]]; then
                max_time=$time
            fi
        done
        avg_time=$((sum / ${#times[@]}))
    fi
    
    # Clean up temp files
    rm -rf "$temp_dir"
    
    # Results
    echo "=" | tr ' ' '=' | head -c 60; echo
    echo "🎯 ULTRADEEP LOAD TEST RESULTS"
    echo "=" | tr ' ' '=' | head -c 60; echo
    echo "Total Requests: $total"
    echo "✅ Successful: $success ($(( success * 100 / total ))%)"
    echo "⏰ Timeouts: $timeout ($(( timeout * 100 / total ))%)"
    echo "❌ Errors: $error ($(( error * 100 / total ))%)"
    echo "🕒 Total Test Time: ${total_time}ms"
    echo "🚀 Requests/Second: $(( total * 1000 / total_time ))"
    
    if [[ ${#times[@]} -gt 0 ]]; then
        echo "-" | tr ' ' '-' | head -c 40; echo
        echo "📈 RESPONSE TIME ANALYSIS"
        echo "Average: ${avg_time}ms"
        echo "Max:     ${max_time}ms"
        echo "Min:     ${min_time}ms"
        
        echo "-" | tr ' ' '-' | head -c 40; echo
        echo "🎯 PERFORMANCE ASSESSMENT"
        if [[ $success -eq $total && $avg_time -lt 50 ]]; then
            echo "🟢 PERFECT: 100% success rate with <50ms average response"
        elif [[ $success -eq $total && $avg_time -lt 100 ]]; then
            echo "🟡 GOOD: 100% success rate with <100ms average response"
        elif [[ $success -ge $((total * 99 / 100)) && $avg_time -lt 200 ]]; then
            echo "🟡 ACCEPTABLE: >99% success rate with <200ms average response"
        else
            echo "🔴 NEEDS IMPROVEMENT: Below target performance"
        fi
        
        if [[ $timeout -gt 0 ]]; then
            echo "⚠️  TIMEOUT PATTERN DETECTED: $timeout requests timed out"
            echo "🔧 RECOMMENDATION: Optimize health endpoint for better concurrency"
        fi
        
        if [[ $error -gt 0 ]]; then
            echo "⚠️  ERROR PATTERN DETECTED: $error requests failed"
            echo "🔧 RECOMMENDATION: Add better error handling and resilience"
        fi
    fi
    
    echo
}

# Run progressive load tests
run_load_test 10 50 "Test 1: Light Load"
sleep 2

run_load_test 25 100 "Test 2: Medium Load"
sleep 2

run_load_test 50 200 "Test 3: Heavy Load"
sleep 2

run_load_test 100 300 "Test 4: Extreme Load"

echo "🏁 All load tests completed!"