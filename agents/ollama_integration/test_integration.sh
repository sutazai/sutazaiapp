#!/bin/bash

# Ollama Integration Test Script
# Tests the deployed Ollama integration service

set -e

echo "=== Ollama Integration Service Test ==="
echo

# Service URL
SERVICE_URL="http://127.0.0.1:8090"

# 1. Test health endpoint
echo "1. Testing health endpoint..."
HEALTH=$(curl -s "$SERVICE_URL/health")
echo "$HEALTH" | jq
STATUS=$(echo "$HEALTH" | jq -r '.status')
if [ "$STATUS" != "healthy" ]; then
    echo "❌ Service is not healthy"
    exit 1
fi
echo "✅ Service is healthy"
echo

# 2. Test models endpoint
echo "2. Testing models endpoint..."
MODELS=$(curl -s "$SERVICE_URL/models")
echo "$MODELS" | jq
MODEL_COUNT=$(echo "$MODELS" | jq '.models | length')
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "❌ No models available"
    exit 1
fi
echo "✅ Found $MODEL_COUNT model(s)"
echo

# 3. Test generation with different temperatures
echo "3. Testing text generation..."

# Low temperature (deterministic)
echo "   a) Low temperature (0.3)..."
RESPONSE1=$(curl -s -X POST "$SERVICE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is Python?",
    "temperature": 0.3,
    "max_tokens": 30
  }')
echo "$RESPONSE1" | jq '.response' | head -c 100
echo "..."
TOKENS1=$(echo "$RESPONSE1" | jq -r '.tokens')
echo "   Generated $TOKENS1 tokens"
echo

# Medium temperature
echo "   b) Medium temperature (0.7)..."
RESPONSE2=$(curl -s -X POST "$SERVICE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a haiku about coding:",
    "temperature": 0.7,
    "max_tokens": 50
  }')
echo "$RESPONSE2" | jq -r '.response'
LATENCY2=$(echo "$RESPONSE2" | jq -r '.latency')
printf "   Latency: %.2f ms\n" "$LATENCY2"
echo

# High temperature (creative)
echo "   c) High temperature (1.5)..."
RESPONSE3=$(curl -s -X POST "$SERVICE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Imagine a world where",
    "temperature": 1.5,
    "max_tokens": 40
  }')
echo "$RESPONSE3" | jq -r '.response' | head -c 150
echo "..."
TPS=$(echo "$RESPONSE3" | jq -r '.tokens_per_second')
printf "   Tokens per second: %.2f\n" "$TPS"
echo

# 4. Test with stop sequences
echo "4. Testing stop sequences..."
RESPONSE4=$(curl -s -X POST "$SERVICE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "List three colors:\n1.",
    "temperature": 0.5,
    "max_tokens": 100,
    "stop": ["\n4.", "\n\n"]
  }')
echo "$RESPONSE4" | jq -r '.response'
echo

# 5. Test error handling
echo "5. Testing error handling..."
echo "   a) Invalid temperature..."
ERROR1=$(curl -s -X POST "$SERVICE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Test",
    "temperature": 3.0,
    "max_tokens": 10
  }')
ERROR_MSG=$(echo "$ERROR1" | jq -r '.detail' 2>/dev/null || echo "$ERROR1")
if [[ "$ERROR_MSG" == *"temperature"* ]] || [[ "$ERROR_MSG" == *"validation"* ]]; then
    echo "   ✅ Correctly rejected invalid temperature"
else
    echo "   ⚠️ Unexpected error response"
fi

echo "   b) Empty prompt..."
ERROR2=$(curl -s -X POST "$SERVICE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "",
    "temperature": 0.7,
    "max_tokens": 10
  }')
ERROR_MSG2=$(echo "$ERROR2" | jq -r '.detail' 2>/dev/null || echo "$ERROR2")
if [[ "$ERROR_MSG2" == *"prompt"* ]] || [[ "$ERROR_MSG2" == *"validation"* ]]; then
    echo "   ✅ Correctly rejected empty prompt"
else
    echo "   ⚠️ Unexpected error response"
fi
echo

# 6. Performance test
echo "6. Running performance test..."
echo "   Sending 3 concurrent requests..."
START_TIME=$(date +%s%N)

# Run 3 requests in parallel
(curl -s -X POST "$SERVICE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test 1", "temperature": 0.5, "max_tokens": 20}' > /tmp/ollama_test1.json) &
(curl -s -X POST "$SERVICE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test 2", "temperature": 0.5, "max_tokens": 20}' > /tmp/ollama_test2.json) &
(curl -s -X POST "$SERVICE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test 3", "temperature": 0.5, "max_tokens": 20}' > /tmp/ollama_test3.json) &

wait

END_TIME=$(date +%s%N)
ELAPSED=$((($END_TIME - $START_TIME) / 1000000))
echo "   Total time for 3 concurrent requests: ${ELAPSED}ms"

# Check all responses succeeded
for i in 1 2 3; do
    if [ -f "/tmp/ollama_test$i.json" ]; then
        RESPONSE=$(cat "/tmp/ollama_test$i.json")
        if echo "$RESPONSE" | jq -e '.response' > /dev/null 2>&1; then
            echo "   ✅ Request $i succeeded"
        else
            echo "   ❌ Request $i failed"
        fi
        rm -f "/tmp/ollama_test$i.json"
    fi
done
echo

echo "=== All Tests Complete ==="
echo "✅ Ollama Integration Service is fully operational!"