#!/bin/bash

###############################################################################
# SutazAI System Test Script
# Tests all components of the deployed system
###############################################################################

set -euo pipefail

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test results
PASSED=0
FAILED=0

log() {
    echo -e "${GREEN}[TEST]${NC} $1"
}

pass() {
    echo -e "${GREEN}✅ PASS${NC} $1"
    ((PASSED++))
}

fail() {
    echo -e "${RED}❌ FAIL${NC} $1"
    ((FAILED++))
}

info() {
    echo -e "${BLUE}ℹ️  INFO${NC} $1"
}

# Test function
test_endpoint() {
    local name=$1
    local url=$2
    local expected_code=${3:-200}
    
    log "Testing $name..."
    
    response_code=$(curl -s -o /dev/null -w "%{http_code}" "$url" || echo "000")
    
    if [ "$response_code" = "$expected_code" ]; then
        pass "$name - Response: $response_code"
    else
        fail "$name - Expected: $expected_code, Got: $response_code"
    fi
}

# Test WebSocket
test_websocket() {
    local name=$1
    local url=$2
    
    log "Testing WebSocket $name..."
    
    # Use Python to test WebSocket
    python3 -c "
import websocket
import json
import sys

try:
    ws = websocket.WebSocket()
    ws.connect('$url')
    ws.send(json.dumps({'message': 'test', 'model': 'llama3.2:1b'}))
    result = ws.recv()
    ws.close()
    print('PASS')
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
" && pass "$name - WebSocket connected" || fail "$name - WebSocket connection failed"
}

# Test Docker containers
test_containers() {
    log "Testing Docker containers..."
    
    containers=(
        "sutazai-postgres"
        "sutazai-redis"
        "sutazai-qdrant"
        "sutazai-chromadb"
        "sutazai-ollama"
        "sutazai-backend"
        "sutazai-streamlit"
    )
    
    for container in "${containers[@]}"; do
        if docker ps --format "{{.Names}}" | grep -q "^$container$"; then
            pass "Container $container is running"
        else
            fail "Container $container is not running"
        fi
    done
}

# Test API endpoints
test_api() {
    log "Testing API endpoints..."
    
    # Health check
    test_endpoint "Health Check" "http://localhost:8000/health"
    
    # Chat endpoint
    log "Testing Chat API..."
    response=$(curl -s -X POST http://localhost:8000/api/chat \
        -H "Content-Type: application/json" \
        -d '{"message": "Hello", "model": "llama3.2:1b"}' || echo "{}")
    
    if echo "$response" | grep -q "response"; then
        pass "Chat API - Response received"
    else
        fail "Chat API - No response"
    fi
    
    # Agent status
    test_endpoint "Agent Status" "http://localhost:8000/api/agents/status"
    
    # Metrics
    test_endpoint "Metrics" "http://localhost:8000/api/metrics"
}

# Test Streamlit UI
test_streamlit() {
    log "Testing Streamlit UI..."
    
    test_endpoint "Streamlit UI" "http://localhost:8501"
}

# Test monitoring
test_monitoring() {
    log "Testing monitoring services..."
    
    test_endpoint "Prometheus" "http://localhost:9090"
    test_endpoint "Grafana" "http://localhost:3000"
}

# Test AI agents
test_agents() {
    log "Testing AI agents..."
    
    agents=(
        "AutoGPT:8080"
        "BigAGI:8090"
        "LangFlow:7860"
        "Dify:5001"
    )
    
    for agent in "${agents[@]}"; do
        name="${agent%%:*}"
        port="${agent##*:}"
        test_endpoint "$name" "http://localhost:$port" 200
    done
}

# Test vector databases
test_vector_dbs() {
    log "Testing vector databases..."
    
    test_endpoint "Qdrant" "http://localhost:6333/health"
    test_endpoint "ChromaDB" "http://localhost:8001/api/v1/heartbeat"
}

# Test Ollama models
test_models() {
    log "Testing Ollama models..."
    
    test_endpoint "Ollama API" "http://localhost:11434/api/health"
    
    # List models
    models=$(curl -s http://localhost:11434/api/tags | jq -r '.models[].name' 2>/dev/null || echo "")
    
    if [ -n "$models" ]; then
        info "Available models:"
        echo "$models" | while read -r model; do
            echo "  - $model"
        done
        pass "Models loaded successfully"
    else
        fail "No models found"
    fi
}

# Performance test
test_performance() {
    log "Running basic performance test..."
    
    start_time=$(date +%s.%N)
    
    # Make 10 concurrent requests
    for i in {1..10}; do
        curl -s -X POST http://localhost:8000/api/chat \
            -H "Content-Type: application/json" \
            -d '{"message": "Hi", "model": "llama3.2:1b"}' > /dev/null &
    done
    
    wait
    
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc)
    
    info "10 concurrent requests completed in ${duration}s"
    
    if (( $(echo "$duration < 30" | bc -l) )); then
        pass "Performance test - Acceptable response time"
    else
        fail "Performance test - Slow response time"
    fi
}

# Main test execution
main() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}🧪 SutazAI System Test Suite${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    
    # Run tests
    test_containers
    echo
    test_api
    echo
    test_streamlit
    echo
    test_monitoring
    echo
    test_agents
    echo
    test_vector_dbs
    echo
    test_models
    echo
    test_websocket "Chat WebSocket" "ws://localhost:8000/ws/chat"
    echo
    test_performance
    
    # Summary
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}📊 Test Summary${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Passed: $PASSED${NC}"
    echo -e "${RED}Failed: $FAILED${NC}"
    
    if [ $FAILED -eq 0 ]; then
        echo -e "\n${GREEN}🎉 All tests passed! System is fully operational.${NC}"
        exit 0
    else
        echo -e "\n${RED}⚠️  Some tests failed. Please check the logs.${NC}"
        exit 1
    fi
}

# Check dependencies
check_deps() {
    deps=(curl jq bc python3)
    missing=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing+=("$dep")
        fi
    done
    
    if [ ${#missing[@]} -gt 0 ]; then
        echo -e "${RED}Missing dependencies: ${missing[*]}${NC}"
        echo "Please install them first:"
        echo "  sudo apt-get install ${missing[*]}"
        exit 1
    fi
}

# Run checks and tests
check_deps
main "$@"