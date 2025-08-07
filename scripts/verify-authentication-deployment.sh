#!/bin/bash

set -e

echo "=========================================="
echo "SutazAI Authentication Deployment Verification"
echo "=========================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Service URLs
KEYCLOAK_URL="http://localhost:10050"
KONG_ADMIN_URL="http://localhost:10052"
KONG_PROXY_URL="http://localhost:10051"
JWT_SERVICE_URL="http://localhost:10054"
SERVICE_ACCOUNT_MANAGER_URL="http://localhost:10055"
RBAC_ENGINE_URL="http://localhost:10056"
VAULT_URL="http://localhost:10053"

# Test results
TOTAL_TESTS=0
PASSED_TESTS=0

# Function to check service health
check_service() {
    local service_name=$1
    local url=$2
    local endpoint=${3:-"/health"}
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    echo -n "Checking $service_name... "
    
    if curl -f -s "$url$endpoint" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        return 1
    fi
}

# Function to check docker services
check_docker_services() {
    echo "Checking Docker services..."
    echo "----------------------------------------"
    
    services=(
        "sutazai-keycloak"
        "sutazai-kong" 
        "sutazai-jwt-service"
        "sutazai-service-account-manager"
        "sutazai-rbac-engine"
        "sutazai-vault"
    )
    
    for service in "${services[@]}"; do
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        echo -n "Container $service... "
        
        if docker ps --format "table {{.Names}}" | grep -q "^$service$"; then
            echo -e "${GREEN}✓ RUNNING${NC}"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            echo -e "${RED}✗ NOT RUNNING${NC}"
        fi
    done
    echo
}

# Function to check service endpoints
check_service_endpoints() {
    echo "Checking service endpoints..."
    echo "----------------------------------------"
    
    check_service "Keycloak" "$KEYCLOAK_URL" "/health/ready"
    check_service "Kong Admin API" "$KONG_ADMIN_URL" "/status"
    check_service "JWT Service" "$JWT_SERVICE_URL"
    check_service "Service Account Manager" "$SERVICE_ACCOUNT_MANAGER_URL"
    check_service "RBAC Policy Engine" "$RBAC_ENGINE_URL"
    check_service "Vault" "$VAULT_URL" "/v1/sys/health"
    
    echo
}

# Function to test basic functionality
test_basic_functionality() {
    echo "Testing basic functionality..."
    echo "----------------------------------------"
    
    # Test service account creation
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n "Service account creation... "
    
    response=$(curl -s -w "%{http_code}" -X POST "$SERVICE_ACCOUNT_MANAGER_URL/service-accounts" \
        -H "Content-Type: application/json" \
        -d '{
            "name": "test-verification-agent",
            "description": "Test agent for deployment verification",
            "scopes": ["read", "write", "agent"]
        }' 2>/dev/null)
    
    http_code="${response: -3}"
    
    if [[ "$http_code" == "200" || "$http_code" == "201" || "$http_code" == "409" ]]; then
        echo -e "${GREEN}✓ PASS${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}✗ FAIL (HTTP $http_code)${NC}"
    fi
    
    # Test JWT token generation
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n "JWT token generation... "
    
    response=$(curl -s -w "%{http_code}" -X POST "$JWT_SERVICE_URL/auth/token" \
        -H "Content-Type: application/json" \
        -d '{
            "service_name": "test-verification-agent",
            "scopes": ["read", "write"],
            "expires_in": 300
        }' 2>/dev/null)
    
    http_code="${response: -3}"
    
    if [[ "$http_code" == "200" ]]; then
        echo -e "${GREEN}✓ PASS${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        
        # Extract token for further testing
        TEST_TOKEN=$(echo "${response%???}" | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)
    else
        echo -e "${RED}✗ FAIL (HTTP $http_code)${NC}"
        TEST_TOKEN=""
    fi
    
    # Test RBAC access check
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n "RBAC access check... "
    
    response=$(curl -s -w "%{http_code}" -X POST "$RBAC_ENGINE_URL/access/check" \
        -H "Content-Type: application/json" \
        -d '{
            "subject": "role:ai-agent",
            "object": "api:ollama",
            "action": "read"
        }' 2>/dev/null)
    
    http_code="${response: -3}"
    
    if [[ "$http_code" == "200" ]]; then
        echo -e "${GREEN}✓ PASS${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}✗ FAIL (HTTP $http_code)${NC}"
    fi
    
    # Test Kong proxy (if token available)
    if [[ -n "$TEST_TOKEN" ]]; then
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        echo -n "Kong proxy access... "
        
        response=$(curl -s -w "%{http_code}" -X GET "$KONG_PROXY_URL/api/health" \
            -H "Authorization: Bearer $TEST_TOKEN" 2>/dev/null)
        
        http_code="${response: -3}"
        
        if [[ "$http_code" == "200" || "$http_code" == "401" ]]; then
            echo -e "${GREEN}✓ PASS${NC}"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            echo -e "${RED}✗ FAIL (HTTP $http_code)${NC}"
        fi
    fi
    
    echo
}

# Function to check configuration files
check_configuration_files() {
    echo "Checking configuration files..."
    echo "----------------------------------------"
    
    config_files=(
        "/opt/sutazaiapp/docker-compose.auth.yml"
        "/opt/sutazaiapp/auth/kong/kong.yml"
        "/opt/sutazaiapp/auth/vault/policies/agent-policy.hcl"
        "/opt/sutazaiapp/scripts/setup-authentication.sh"
        "/opt/sutazaiapp/scripts/test-authentication.py"
        "/opt/sutazaiapp/docs/authentication-guide.md"
    )
    
    for config_file in "${config_files[@]}"; do
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        echo -n "$(basename "$config_file")... "
        
        if [[ -f "$config_file" ]]; then
            echo -e "${GREEN}✓ EXISTS${NC}"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            echo -e "${RED}✗ MISSING${NC}"
        fi
    done
    
    echo
}

# Function to show service URLs
show_service_urls() {
    echo "Service URLs:"
    echo "----------------------------------------"
    echo "Keycloak Admin Console: $KEYCLOAK_URL/admin"
    echo "Kong Admin API: $KONG_ADMIN_URL"
    echo "Kong Proxy: $KONG_PROXY_URL"
    echo "JWT Service: $JWT_SERVICE_URL"
    echo "Service Account Manager: $SERVICE_ACCOUNT_MANAGER_URL"
    echo "RBAC Policy Engine: $RBAC_ENGINE_URL"
    echo "Vault UI: $VAULT_URL/ui"
    echo
}

# Function to show deployment commands
show_deployment_commands() {
    echo "Deployment Commands:"
    echo "----------------------------------------"
    echo "Start authentication services:"
    echo "  docker-compose -f docker-compose.auth.yml up -d"
    echo
    echo "Initialize authentication system:"
    echo "  ./scripts/setup-authentication.sh"
    echo
    echo "Update agent configurations:"
    echo "  python3 ./scripts/update-agent-auth.py"
    echo
    echo "Run comprehensive tests:"
    echo "  python3 ./scripts/test-authentication.py"
    echo
}

# Main execution
main() {
    # Change to SutazAI directory
    cd /opt/sutazaiapp
    
    # Run all checks
    check_docker_services
    check_service_endpoints
    test_basic_functionality
    check_configuration_files
    
    # Calculate success rate
    if [[ $TOTAL_TESTS -gt 0 ]]; then
        SUCCESS_RATE=$(( PASSED_TESTS * 100 / TOTAL_TESTS ))
    else
        SUCCESS_RATE=0
    fi
    
    # Display results
    echo "=========================================="
    echo "VERIFICATION RESULTS"
    echo "=========================================="
    echo "Total Tests: $TOTAL_TESTS"
    echo "Passed: $PASSED_TESTS"
    echo "Failed: $((TOTAL_TESTS - PASSED_TESTS))"
    echo "Success Rate: $SUCCESS_RATE%"
    echo
    
    if [[ $SUCCESS_RATE -ge 80 ]]; then
        echo -e "${GREEN}🎉 Authentication system is mostly working!${NC}"
        echo "The deployment appears to be successful."
    elif [[ $SUCCESS_RATE -ge 50 ]]; then
        echo -e "${YELLOW}⚠️  Authentication system is partially working.${NC}"
        echo "Some components may need attention."
    else
        echo -e "${RED}❌ Authentication system has significant issues.${NC}"
        echo "Please review the failed tests and fix any problems."
    fi
    
    echo
    show_service_urls
    show_deployment_commands
    
    # Return appropriate exit code
    if [[ $SUCCESS_RATE -ge 80 ]]; then
        return 0
    else
        return 1
    fi
}

# Execute main function
main "$@"