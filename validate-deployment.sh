#!/bin/bash

# SutazAI Deployment Validation Script
# Version: 1.0
# Date: 2025-08-05

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

COMPOSE_FILE="docker-compose.consolidated.yml"
VALIDATION_LOG="validation-$(date +%Y%m%d_%H%M%S).log"

print_status() {
    echo -e "${GREEN}[PASS]${NC} $1" | tee -a "$VALIDATION_LOG"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$VALIDATION_LOG"
}

print_error() {
    echo -e "${RED}[FAIL]${NC} $1" | tee -a "$VALIDATION_LOG"
}

print_header() {
    echo -e "${BLUE}===================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================================${NC}"
}

# Test HTTP endpoint
test_http_endpoint() {
    local name="$1"
    local url="$2"
    local expected_code="${3:-200}"
    local timeout="${4:-10}"
    
    if curl -s -f --max-time "$timeout" --write-out "%{http_code}" --output /dev/null "$url" | grep -q "$expected_code"; then
        print_status "$name endpoint is responding ($url)"
        return 0
    else
        print_error "$name endpoint is not responding ($url)"
        return 1
    fi
}

# Test TCP port
test_tcp_port() {
    local name="$1"
    local host="$2"
    local port="$3"
    local timeout="${4:-5}"
    
    if timeout "$timeout" bash -c "</dev/tcp/$host/$port"; then
        print_status "$name port is open ($host:$port)"
        return 0
    else
        print_error "$name port is not accessible ($host:$port)"
        return 1
    fi
}

# Test Docker service health
test_service_health() {
    local service="$1"
    
    if docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up\|healthy"; then
        print_status "$service container is running"
        return 0
    else
        print_error "$service container is not running properly"
        return 1
    fi
}

# Test database connection
test_database_connection() {
    local db_name="$1"
    local container="$2"
    local test_command="$3"
    
    if docker exec "$container" $test_command &>/dev/null; then
        print_status "$db_name database connection successful"
        return 0
    else
        print_error "$db_name database connection failed"
        return 1
    fi
}

# Main validation function
validate_deployment() {
    print_header "SUTAZAI DEPLOYMENT VALIDATION"
    local total_tests=0
    local passed_tests=0
    
    # Test 1: Docker network exists
    print_header "TESTING DOCKER NETWORK"
    total_tests=$((total_tests + 1))
    if docker network ls | grep -q "sutazai-network"; then
        print_status "Docker network 'sutazai-network' exists"
        passed_tests=$((passed_tests + 1))
    else
        print_error "Docker network 'sutazai-network' does not exist"
    fi
    
    # Test 2: Service containers are running
    print_header "TESTING CONTAINER STATUS"
    services=("postgres" "redis" "neo4j" "chromadb" "qdrant" "ollama" "backend" "frontend" "prometheus" "grafana" "loki" "health-monitor")
    
    for service in "${services[@]}"; do
        total_tests=$((total_tests + 1))
        if test_service_health "$service"; then
            passed_tests=$((passed_tests + 1))
        fi
    done
    
    # Test 3: Database connections
    print_header "TESTING DATABASE CONNECTIONS"
    
    # PostgreSQL
    total_tests=$((total_tests + 1))
    if test_database_connection "PostgreSQL" "sutazai-postgres" "pg_isready -U sutazai"; then
        passed_tests=$((passed_tests + 1))
    fi
    
    # Redis
    total_tests=$((total_tests + 1))
    if test_database_connection "Redis" "sutazai-redis" "redis-cli ping"; then
        passed_tests=$((passed_tests + 1))
    fi
    
    # Test 4: TCP Port accessibility
    print_header "TESTING PORT ACCESSIBILITY"
    ports=(
        "PostgreSQL:localhost:10000"
        "Redis:localhost:10001"
        "Neo4j-HTTP:localhost:10002"
        "Neo4j-Bolt:localhost:10003"
        "Backend:localhost:10010"
        "Frontend:localhost:10011"
        "ChromaDB:localhost:10100"
        "Qdrant-HTTP:localhost:10101"
        "Qdrant-gRPC:localhost:10102"
        "Ollama:localhost:10104"
        "Prometheus:localhost:10200"
        "Grafana:localhost:10201"
        "Loki:localhost:10202"
        "Health-Monitor:localhost:10210"
    )
    
    for port_info in "${ports[@]}"; do
        IFS=':' read -r name host port <<< "$port_info"
        total_tests=$((total_tests + 1))
        if test_tcp_port "$name" "$host" "$port"; then
            passed_tests=$((passed_tests + 1))
        fi
    done
    
    # Test 5: HTTP Endpoints
    print_header "TESTING HTTP ENDPOINTS"
    
    # Wait for services to be fully ready
    echo "Waiting 30 seconds for services to be fully ready..."
    sleep 30
    
    endpoints=(
        "Backend-Health:http://localhost:10010/health"
        "Grafana:http://localhost:10201"
        "Prometheus:http://localhost:10200"
        "ChromaDB-Heartbeat:http://localhost:10100/api/v1/heartbeat"
        "Qdrant-Health:http://localhost:10101"
    )
    
    for endpoint_info in "${endpoints[@]}"; do
        IFS=':' read -r name url <<< "$endpoint_info"
        total_tests=$((total_tests + 1))
        if test_http_endpoint "$name" "$url"; then
            passed_tests=$((passed_tests + 1))
        fi
    done
    
    # Test 6: Ollama model availability
    print_header "TESTING OLLAMA MODEL"
    total_tests=$((total_tests + 1))
    if docker exec sutazai-ollama ollama list | grep -q "gpt-oss\|gpt-oss"; then
        print_status "Ollama has models available"
        passed_tests=$((passed_tests + 1))
    else
        print_warning "No models found in Ollama (this is normal on first run)"
    fi
    
    # Test 7: Docker resource usage
    print_header "TESTING RESOURCE USAGE"
    total_tests=$((total_tests + 1))
    memory_usage=$(docker stats --no-stream --format "table {{.MemUsage}}" | tail -n +2 | awk -F'/' '{sum += $1} END {print sum}' | head -1)
    if [ -n "$memory_usage" ]; then
        print_status "Docker containers are using memory (resource monitoring active)"
        passed_tests=$((passed_tests + 1))
    else
        print_warning "Could not determine memory usage"
    fi
    
    # Test 8: Log accessibility
    print_header "TESTING LOG ACCESSIBILITY"
    total_tests=$((total_tests + 1))
    if docker-compose -f "$COMPOSE_FILE" logs --tail=1 backend &>/dev/null; then
        print_status "Container logs are accessible"
        passed_tests=$((passed_tests + 1))
    else
        print_error "Container logs are not accessible"
    fi
    
    # Final summary
    print_header "VALIDATION SUMMARY"
    echo -e "${BLUE}Total Tests: $total_tests${NC}"
    echo -e "${GREEN}Passed: $passed_tests${NC}"
    echo -e "${RED}Failed: $((total_tests - passed_tests))${NC}"
    
    success_rate=$((passed_tests * 100 / total_tests))
    echo -e "${BLUE}Success Rate: $success_rate%${NC}"
    
    if [ $success_rate -ge 90 ]; then
        echo -e "${GREEN}✅ DEPLOYMENT VALIDATION SUCCESSFUL${NC}"
        echo -e "${GREEN}Your SutazAI deployment is working correctly!${NC}"
        return 0
    elif [ $success_rate -ge 70 ]; then
        echo -e "${YELLOW}⚠️  DEPLOYMENT PARTIALLY WORKING${NC}"
        echo -e "${YELLOW}Some services may need attention, but core functionality is available.${NC}"
        return 1
    else
        echo -e "${RED}❌ DEPLOYMENT VALIDATION FAILED${NC}"
        echo -e "${RED}Critical issues detected. Please check the logs and restart services.${NC}"
        return 2
    fi
}

# Quick health check function
quick_health_check() {
    print_header "QUICK HEALTH CHECK"
    
    echo "Service Status:"
    docker-compose -f "$COMPOSE_FILE" ps
    
    echo -e "\nCritical Services:"
    critical_services=("postgres" "redis" "backend" "frontend" "ollama")
    for service in "${critical_services[@]}"; do
        if docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up\|healthy"; then
            echo -e "  ${GREEN}✓${NC} $service"
        else
            echo -e "  ${RED}✗${NC} $service"
        fi
    done
    
    echo -e "\nResource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -10
}

# Command line interface
case "${1:-validate}" in
    "validate"|"full")
        validate_deployment
        ;;
    "quick"|"status")
        quick_health_check
        ;;
    "endpoints")
        print_header "TESTING ENDPOINTS ONLY"
        test_http_endpoint "Backend" "http://localhost:10010/health" 200 10
        test_http_endpoint "Frontend" "http://localhost:10011" 200 10
        test_http_endpoint "Grafana" "http://localhost:10201" 200 10
        test_http_endpoint "Prometheus" "http://localhost:10200" 200 10
        ;;
    "ports")
        print_header "TESTING PORTS ONLY"
        test_tcp_port "Backend" "localhost" "10010"
        test_tcp_port "Frontend" "localhost" "10011"
        test_tcp_port "PostgreSQL" "localhost" "10000"
        test_tcp_port "Redis" "localhost" "10001"
        test_tcp_port "Ollama" "localhost" "10104"
        ;;
    "help"|"-h"|"--help")
        echo "SutazAI Deployment Validation Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  validate    Full validation suite (default)"
        echo "  quick       Quick health check"
        echo "  endpoints   Test HTTP endpoints only"
        echo "  ports       Test TCP ports only"
        echo "  help        Show this help message"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac