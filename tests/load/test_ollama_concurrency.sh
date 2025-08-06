#!/bin/bash

# Purpose: Test Ollama GPT-OSS concurrent capacity
# Usage: ./test_ollama_concurrency.sh [num_requests]
# Requirements: curl, jq

NUM_REQUESTS=${1:-20}
OLLAMA_URL="http://localhost:11434"
MODEL="gpt-oss"

echo "Testing Ollama concurrency with $NUM_REQUESTS concurrent requests..."
echo "Model: $MODEL"
echo "Started at: $(date)"

# Function to make a request
make_request() {
    local id=$1
    local start_time=$(date +%s.%N)
    
    response=$(timeout 60s curl -s -X POST "$OLLAMA_URL/api/generate" \
        -d "{\"model\": \"$MODEL\", \"prompt\": \"Request $id: What is AI?\", \"stream\": false}" \
        2>/dev/null | jq -r '.response' 2>/dev/null)
    
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc -l)
    
    if [ -n "$response" ] && [ "$response" != "null" ]; then
        echo "Request $id: SUCCESS (${duration}s) - ${response:0:50}..."
        return 0
    else
        echo "Request $id: FAILED (${duration}s)"
        return 1
    fi
}

# Run concurrent requests
echo "Starting $NUM_REQUESTS concurrent requests..."
success_count=0
failed_count=0

for i in $(seq 1 $NUM_REQUESTS); do
    make_request $i &
done

# Wait for all background jobs and count results
for job in $(jobs -p); do
    wait $job
    if [ $? -eq 0 ]; then
        ((success_count++))
    else
        ((failed_count++))
    fi
done

echo ""
echo "Results:"
echo "Total requests: $NUM_REQUESTS"
echo "Successful: $success_count"
echo "Failed: $failed_count"
echo "Success rate: $(echo "scale=2; $success_count * 100 / $NUM_REQUESTS" | bc -l)%"
echo "Completed at: $(date)"