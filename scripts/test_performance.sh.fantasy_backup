#!/bin/bash

# Performance testing script for SutazAI
echo "SutazAI Performance Test Results"
echo "================================"
echo "Timestamp: $(date)"
echo ""

# Test different message types
test_messages=(
    '{"message": "Hi", "model": "qwen2.5:3b"}'
    '{"message": "What is 2+2?", "model": "qwen2.5:3b"}'
    '{"message": "Explain quantum computing in one sentence.", "model": "qwen2.5:3b"}'
    '{"message": "Hi", "model": "qwen2.5-coder:3b"}'
    '{"message": "What is 2+2?", "model": "qwen2.5-coder:3b"}'
    '{"message": "Write a simple Python function.", "model": "qwen2.5-coder:3b"}'
)

test_names=(
    "Short response (1B)"
    "Simple math (1B)"
    "Complex topic (1B)"
    "Short response (7B)"
    "Simple math (7B)"
    "Code generation (7B)"
)

echo "Testing API response times:"
echo ""

for i in "${!test_messages[@]}"; do
    echo "Test $((i+1)): ${test_names[i]}"
    
    start_time=$(date +%s.%N)
    response=$(curl -s -X POST http://172.31.77.193:8000/simple-chat \
        -H "Content-Type: application/json" \
        -d "${test_messages[i]}")
    end_time=$(date +%s.%N)
    
    # Calculate total time
    total_time=$(echo "$end_time - $start_time" | bc -l)
    
    # Extract processing time from response
    processing_time=$(echo "$response" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('processing_time', 'N/A'))" 2>/dev/null || echo "N/A")
    
    # Get response length
    response_length=$(echo "$response" | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('response', '')))" 2>/dev/null || echo "N/A")
    
    printf "  Total time: %.2fs\n" "$total_time"
    printf "  Model processing: %ss\n" "$processing_time"
    printf "  Response length: %s chars\n" "$response_length"
    echo ""
    
    sleep 2
done

echo "System Resources:"
echo "================="

# Memory usage
echo "Memory usage:"
free -h | grep -E "Mem:|Swap:"

echo ""
echo "Container resource usage:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | grep sutazai

echo ""
echo "Ollama model status:"
docker exec sutazai-ollama ollama list

echo ""
echo "Performance test complete!"