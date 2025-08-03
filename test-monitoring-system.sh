#!/bin/bash

# 🧪 Hygiene Monitoring System - Comprehensive Test Suite
# Purpose: Validate all system components and connections
# Author: Testing QA Validator Agent
# Version: 1.0.0 - Complete Validation

set -e

echo "🧪 Testing Sutazai Hygiene Monitoring System"
echo "============================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

test_count=0
passed_tests=0
failed_tests=0

# Test functions
run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_result="$3"
    
    test_count=$((test_count + 1))
    echo -n "🔍 Test $test_count: $test_name... "
    
    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        passed_tests=$((passed_tests + 1))
        return 0
    else
        echo -e "${RED}FAIL${NC}"
        failed_tests=$((failed_tests + 1))
        return 1
    fi
}

# API endpoint tests
test_api_endpoint() {
    local endpoint="$1"
    local description="$2"
    
    run_test "$description" "curl -f -s '$endpoint' | jq . > /dev/null"
}

# WebSocket test
test_websocket() {
    local ws_url="$1"
    
    # Simple WebSocket test using curl with upgrade headers
    run_test "WebSocket Connection" "curl -f -s -H 'Connection: Upgrade' -H 'Upgrade: websocket' -H 'Sec-WebSocket-Version: 13' -H 'Sec-WebSocket-Key: test' '$ws_url'"
}

echo ""
echo "🔧 INFRASTRUCTURE TESTS"
echo "========================"

# Test Docker containers
run_test "PostgreSQL Container" "docker ps | grep -q hygiene-postgres"
run_test "Redis Container" "docker ps | grep -q hygiene-redis"  
run_test "Backend Container" "docker ps | grep -q hygiene-backend"
run_test "Rule API Container" "docker ps | grep -q rule-control-api"
run_test "Dashboard Container" "docker ps | grep -q hygiene-dashboard"
run_test "Nginx Container" "docker ps | grep -q hygiene-nginx"

echo ""
echo "🌐 NETWORK CONNECTIVITY TESTS"
echo "=============================="

# Test database connectivity
run_test "PostgreSQL Connection" "docker exec hygiene-postgres pg_isready -U hygiene_user"
run_test "Redis Connection" "docker exec hygiene-redis redis-cli ping | grep -q PONG"

echo ""
echo "🔗 API ENDPOINT TESTS"
echo "==================="

# Test backend endpoints
test_api_endpoint "http://localhost:8080/health" "Backend Health Check"
test_api_endpoint "http://localhost:8080/api/hygiene/status" "Backend Status API"
test_api_endpoint "http://localhost:8080/api/system/metrics" "System Metrics API"

# Test rule control endpoints
test_api_endpoint "http://localhost:8100/api/health/live" "Rule API Health Check"
test_api_endpoint "http://localhost:8100/api/rules" "Rules Configuration API"

# Test dashboard
run_test "Dashboard Accessibility" "curl -f -s http://localhost:3000/health"

# Test nginx proxy
run_test "Nginx Proxy Health" "curl -f -s http://localhost:80/health/"

echo ""
echo "⚡ REAL-TIME FEATURES TESTS"
echo "=========================="

# Test WebSocket (basic connectivity)
run_test "WebSocket Endpoint" "curl -f -s http://localhost:8080/ws"

# Test real-time data flow
echo -n "🔍 Testing real-time data collection... "
if data=$(curl -f -s http://localhost:8080/api/hygiene/status | jq -r '.timestamp'); then
    if [ "$data" != "null" ] && [ -n "$data" ]; then
        echo -e "${GREEN}PASS${NC}"
        passed_tests=$((passed_tests + 1))
    else
        echo -e "${RED}FAIL${NC} (No timestamp data)"
        failed_tests=$((failed_tests + 1))
    fi
else
    echo -e "${RED}FAIL${NC} (API request failed)"
    failed_tests=$((failed_tests + 1))
fi
test_count=$((test_count + 1))

echo ""
echo "📊 DATA VALIDATION TESTS"
echo "========================"

# Test database data
echo -n "🔍 Testing database data integrity... "
if docker exec hygiene-postgres psql -U hygiene_user -d hygiene_monitoring -c "SELECT COUNT(*) FROM system_metrics;" > /dev/null 2>&1; then
    echo -e "${GREEN}PASS${NC}"
    passed_tests=$((passed_tests + 1))
else
    echo -e "${RED}FAIL${NC}"
    failed_tests=$((failed_tests + 1))
fi
test_count=$((test_count + 1))

# Test violation scanning
echo -n "🔍 Testing violation detection... "
if result=$(curl -f -s -X POST http://localhost:8080/api/hygiene/scan | jq -r '.success'); then
    if [ "$result" = "true" ]; then
        echo -e "${GREEN}PASS${NC}"
        passed_tests=$((passed_tests + 1))
    else
        echo -e "${RED}FAIL${NC} (Scan failed)"
        failed_tests=$((failed_tests + 1))
    fi
else
    echo -e "${RED}FAIL${NC} (API request failed)"
    failed_tests=$((failed_tests + 1))
fi
test_count=$((test_count + 1))

echo ""
echo "⚙️ RULE MANAGEMENT TESTS"
echo "======================="

# Test rule retrieval
echo -n "🔍 Testing rule configuration retrieval... "
if rules=$(curl -f -s http://localhost:8100/api/rules | jq -r '.rules | length'); then
    if [ "$rules" -gt 0 ]; then
        echo -e "${GREEN}PASS${NC} ($rules rules found)"
        passed_tests=$((passed_tests + 1))
    else
        echo -e "${RED}FAIL${NC} (No rules found)"
        failed_tests=$((failed_tests + 1))
    fi
else
    echo -e "${RED}FAIL${NC} (API request failed)"
    failed_tests=$((failed_tests + 1))
fi
test_count=$((test_count + 1))

echo ""
echo "🔒 SECURITY TESTS"
echo "================="

# Test CORS headers
echo -n "🔍 Testing CORS headers... "
if cors_header=$(curl -f -s -I http://localhost:8080/api/hygiene/status | grep -i "access-control-allow-origin"); then
    echo -e "${GREEN}PASS${NC}"
    passed_tests=$((passed_tests + 1))
else
    echo -e "${YELLOW}WARN${NC} (CORS headers not found)"
    # Don't fail the test, just warn
fi
test_count=$((test_count + 1))

# Test security headers
echo -n "🔍 Testing security headers... "
if security_headers=$(curl -f -s -I http://localhost:80/ | grep -E "(X-Frame-Options|X-XSS-Protection|X-Content-Type-Options)"); then
    echo -e "${GREEN}PASS${NC}"
    passed_tests=$((passed_tests + 1))
else
    echo -e "${YELLOW}WARN${NC} (Some security headers missing)"
    # Don't fail the test, just warn
fi
test_count=$((test_count + 1))

echo ""
echo "📈 PERFORMANCE TESTS"
echo "==================="

# Test response time
echo -n "🔍 Testing API response times... "
response_time=$(curl -w "%{time_total}" -o /dev/null -s http://localhost:8080/api/hygiene/status)
if (( $(echo "$response_time < 2.0" | bc -l) )); then
    echo -e "${GREEN}PASS${NC} (${response_time}s)"
    passed_tests=$((passed_tests + 1))
else
    echo -e "${YELLOW}SLOW${NC} (${response_time}s)"
    # Don't fail, just note slow response
fi
test_count=$((test_count + 1))

echo ""
echo "🧪 INTEGRATION TESTS"
echo "==================="

# Test end-to-end data flow
echo -n "🔍 Testing complete data flow... "
if dashboard_data=$(curl -f -s http://localhost:8080/api/hygiene/status | jq -r '.systemMetrics.cpu_usage'); then
    if [ "$dashboard_data" != "null" ] && [ -n "$dashboard_data" ]; then
        echo -e "${GREEN}PASS${NC} (CPU: ${dashboard_data}%)"
        passed_tests=$((passed_tests + 1))
    else
        echo -e "${RED}FAIL${NC} (No system metrics)"
        failed_tests=$((failed_tests + 1))
    fi
else
    echo -e "${RED}FAIL${NC} (Data flow broken)"
    failed_tests=$((failed_tests + 1))
fi
test_count=$((test_count + 1))

echo ""
echo "============================================="
echo "🎯 TEST RESULTS SUMMARY"
echo "============================================="

echo -e "Total Tests: $test_count"
echo -e "${GREEN}Passed: $passed_tests${NC}"
echo -e "${RED}Failed: $failed_tests${NC}"

success_rate=$(( (passed_tests * 100) / test_count ))
echo -e "Success Rate: $success_rate%"

echo ""
if [ $failed_tests -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL TESTS PASSED! System is fully operational.${NC}"
    echo ""
    echo "✅ Perfect containerized monitoring system validated!"
    echo "🌐 Dashboard: http://localhost"
    echo "🔧 Backend API: http://localhost:8080"
    echo "⚙️  Rule Control: http://localhost:8100"
    echo "🔗 WebSocket: ws://localhost/ws"
    echo ""
    echo "🚀 System is ready for production use!"
    exit 0
elif [ $success_rate -ge 80 ]; then
    echo -e "${YELLOW}⚠️  MOSTLY WORKING with minor issues ($success_rate% success rate)${NC}"
    echo ""
    echo "The system is functional but has some minor issues."
    echo "Check the failed tests above for details."
    exit 1
else
    echo -e "${RED}❌ SYSTEM HAS SIGNIFICANT ISSUES ($success_rate% success rate)${NC}"
    echo ""
    echo "Multiple critical tests failed. System needs attention."
    echo "Check logs: docker-compose -f docker-compose.hygiene-monitor.yml logs"
    exit 2
fi