#!/bin/bash
# SutazAI Enhanced Brain Testing Script
# Comprehensive validation of all components

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
test_endpoint() {
    local name="$1"
    local url="$2"
    local expected_code="${3:-200}"
    
    echo -n "Testing $name... "
    
    response_code=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
    
    if [ "$response_code" = "$expected_code" ]; then
        echo -e "${GREEN}✓ PASSED${NC} (HTTP $response_code)"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC} (Expected: $expected_code, Got: $response_code)"
        ((TESTS_FAILED++))
        return 1
    fi
}

test_post_endpoint() {
    local name="$1"
    local url="$2"
    local data="$3"
    local expected_field="$4"
    
    echo -n "Testing $name... "
    
    response=$(curl -s -X POST "$url" \
        -H "Content-Type: application/json" \
        -d "$data" 2>/dev/null || echo "{}")
    
    if echo "$response" | grep -q "$expected_field"; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo "Response: $response" | head -1
        ((TESTS_FAILED++))
        return 1
    fi
}

test_docker_container() {
    local name="$1"
    local container="$2"
    
    echo -n "Testing container $name... "
    
    if docker ps --format "{{.Names}}" | grep -q "^$container$"; then
        status=$(docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null || echo "unknown")
        if [ "$status" = "running" ]; then
            echo -e "${GREEN}✓ RUNNING${NC}"
            ((TESTS_PASSED++))
            return 0
        else
            echo -e "${YELLOW}⚠ STATUS: $status${NC}"
            ((TESTS_FAILED++))
            return 1
        fi
    else
        echo -e "${RED}✗ NOT FOUND${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Main test sequence
echo "═══════════════════════════════════════════════════════════════"
echo "         🧪 SutazAI Enhanced Brain System Tests 🧪"
echo "═══════════════════════════════════════════════════════════════"
echo

# Test 1: Core Services
echo -e "\n${BLUE}[1/7] Testing Core Services${NC}"
echo "───────────────────────────────────────"
test_docker_container "Ollama" "sutazai-ollama"
test_docker_container "Redis" "sutazai-redis"
test_docker_container "PostgreSQL" "sutazai-postgresql"
test_docker_container "Qdrant" "sutazai-qdrant"
test_docker_container "ChromaDB" "sutazai-chromadb"
test_docker_container "Neo4j" "sutazai-neo4j"

# Test 2: Brain Core
echo -e "\n${BLUE}[2/7] Testing Brain Core${NC}"
echo "───────────────────────────────────────"
test_docker_container "Brain Core" "sutazai-brain-core"
test_endpoint "Brain Health" "http://localhost:8888/health"
test_endpoint "Brain Status" "http://localhost:8888/status"
test_endpoint "Brain Agents List" "http://localhost:8888/agents"

# Test 3: Enhanced Agents
echo -e "\n${BLUE}[3/7] Testing Enhanced Agents${NC}"
echo "───────────────────────────────────────"
test_docker_container "JARVIS" "sutazai-jarvis"
test_endpoint "JARVIS Health" "http://localhost:8026/health"
test_docker_container "AutoGen v2" "sutazai-autogen-v2"
test_endpoint "AutoGen Health" "http://localhost:8001/health"
test_docker_container "CrewAI v2" "sutazai-crewai-v2"
test_endpoint "CrewAI Health" "http://localhost:8002/health"

# Test 4: ML Frameworks
echo -e "\n${BLUE}[4/7] Testing ML Frameworks${NC}"
echo "───────────────────────────────────────"
test_docker_container "PyTorch" "sutazai-pytorch"
test_endpoint "PyTorch Service" "http://localhost:8888/health"
test_docker_container "TensorFlow" "sutazai-tensorflow"
test_docker_container "JAX" "sutazai-jax"

# Test 5: Brain API Functionality
echo -e "\n${BLUE}[5/7] Testing Brain API Functionality${NC}"
echo "───────────────────────────────────────"
test_post_endpoint "Brain Process" \
    "http://localhost:8888/process" \
    '{"input": "Test request"}' \
    "output"

test_post_endpoint "JARVIS Execution" \
    "http://localhost:8026/execute" \
    '{"input": "Hello JARVIS", "mode": "text"}' \
    "output"

# Test 6: Monitoring Stack
echo -e "\n${BLUE}[6/7] Testing Monitoring Stack${NC}"
echo "───────────────────────────────────────"
test_docker_container "Brain Prometheus" "sutazai-brain-prometheus"
test_endpoint "Prometheus" "http://localhost:9091"
test_docker_container "Brain Grafana" "sutazai-brain-grafana"
test_endpoint "Grafana" "http://localhost:3001" "302"

# Test 7: Advanced Features
echo -e "\n${BLUE}[7/7] Testing Advanced Features${NC}"
echo "───────────────────────────────────────"

# Test Universal Learning Machine
echo -n "Testing ULM integration... "
ulm_response=$(curl -s -X POST "http://localhost:8888/process" \
    -H "Content-Type: application/json" \
    -d '{"input": "Demonstrate learning capabilities"}' 2>/dev/null || echo "{}")

if echo "$ulm_response" | grep -q "learning_progress"; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}"
    ((TESTS_FAILED++))
fi

# Test Memory System
echo -n "Testing Memory System... "
memory_response=$(curl -s "http://localhost:8888/memory/stats" 2>/dev/null || echo "{}")

if echo "$memory_response" | grep -q "redis_keys\|qdrant_points\|chroma_documents"; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}"
    ((TESTS_FAILED++))
fi

# Test Self-Improvement
echo -n "Testing Self-Improvement capability... "
if curl -s "http://localhost:8888/status" | grep -q "auto_improve.*true"; then
    echo -e "${GREEN}✓ ENABLED${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}⚠ DISABLED${NC}"
    ((TESTS_FAILED++))
fi

# Performance Benchmark
echo -e "\n${BLUE}Performance Benchmark${NC}"
echo "───────────────────────────────────────"

echo -n "Testing response time... "
start_time=$(date +%s%N)
curl -s -X POST "http://localhost:8888/process" \
    -H "Content-Type: application/json" \
    -d '{"input": "Quick test"}' >/dev/null 2>&1
end_time=$(date +%s%N)
response_time=$(( (end_time - start_time) / 1000000 ))

if [ "$response_time" -lt 5000 ]; then
    echo -e "${GREEN}✓ FAST${NC} (${response_time}ms)"
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}⚠ SLOW${NC} (${response_time}ms)"
    ((TESTS_FAILED++))
fi

# Model Availability
echo -e "\n${BLUE}Model Availability${NC}"
echo "───────────────────────────────────────"

models=("deepseek-r1:8b" "codellama:7b" "qwen2.5:7b")
for model in "${models[@]}"; do
    echo -n "Checking $model... "
    if docker exec sutazai-ollama ollama list 2>/dev/null | grep -q "$model"; then
        echo -e "${GREEN}✓ AVAILABLE${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ NOT FOUND${NC}"
        ((TESTS_FAILED++))
    fi
done

# Summary
echo
echo "═══════════════════════════════════════════════════════════════"
echo "                        TEST SUMMARY"
echo "═══════════════════════════════════════════════════════════════"
echo -e "Total Tests: $((TESTS_PASSED + TESTS_FAILED))"
echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
echo

if [ "$TESTS_FAILED" -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED! The Enhanced Brain is fully operational.${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Please check the logs for details.${NC}"
    echo
    echo "Troubleshooting tips:"
    echo "1. Check container logs: docker logs <container-name>"
    echo "2. Verify all services are running: docker ps"
    echo "3. Check Brain logs: tail -f /workspace/brain/logs/*.log"
    echo "4. Ensure models are downloaded: docker exec sutazai-ollama ollama list"
    exit 1
fi