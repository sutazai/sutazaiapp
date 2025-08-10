#!/bin/bash
# Purpose: Validate container security configuration and compliance
# Usage: ./validate-container-security.sh [--compose-file docker-compose.secure.yml]
# Requirements: Docker, Trivy, jq

set -euo pipefail


# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

COMPOSE_FILE="${1:-docker-compose.secure.yml}"
VALIDATION_DATE=$(date +"%Y%m%d_%H%M%S")
REPORT_DIR="/opt/sutazaiapp/security-reports"
VALIDATION_REPORT="${REPORT_DIR}/security_validation_${VALIDATION_DATE}.md"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TESTS_PASSED=0
TESTS_FAILED=0

mkdir -p "${REPORT_DIR}"

echo -e "${BLUE}🔒 Container Security Validation Report${NC}"
echo "Date: $(date)"
echo "Compose File: ${COMPOSE_FILE}"
echo ""

# Initialize report
cat > "${VALIDATION_REPORT}" << EOF
# Container Security Validation Report
**Date**: $(date)
**Compose File**: ${COMPOSE_FILE}
**Validation Script**: validate-container-security.sh

## Executive Summary
This report validates the security configuration of containerized services.

## Security Checks Results

EOF

# Function to log test results
log_test() {
    local test_name="$1"
    local status="$2"
    local details="$3"
    
    if [[ "$status" == "PASS" ]]; then
        echo -e "${GREEN}✅ PASS${NC}: $test_name"
        echo "- **✅ PASS**: $test_name - $details" >> "${VALIDATION_REPORT}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌ FAIL${NC}: $test_name"
        echo "  Details: $details"
        echo "- **❌ FAIL**: $test_name - $details" >> "${VALIDATION_REPORT}"
        ((TESTS_FAILED++))
    fi
}

echo "## 1. Docker Compose Configuration Validation"
echo ""

# Test 1: Check if secure compose file exists
if [[ -f "$COMPOSE_FILE" ]]; then
    log_test "Secure compose file exists" "PASS" "Found $COMPOSE_FILE"
else
    log_test "Secure compose file exists" "FAIL" "$COMPOSE_FILE not found"
    exit 1
fi

# Test 2: Check for non-root users in compose
echo "Checking non-root user configurations..."
if grep -q "user:" "$COMPOSE_FILE"; then
    log_test "Non-root users configured" "PASS" "Found user specifications in compose file"
else
    log_test "Non-root users configured" "FAIL" "No user specifications found in compose file"
fi

# Test 3: Check for security contexts
echo "Checking security contexts..."
if grep -q "security_opt:" "$COMPOSE_FILE" && grep -q "no-new-privileges:true" "$COMPOSE_FILE"; then
    log_test "Security contexts configured" "PASS" "Found security_opt with no-new-privileges"
else
    log_test "Security contexts configured" "FAIL" "Missing or incomplete security contexts"
fi

# Test 4: Check for capability dropping
if grep -q "cap_drop:" "$COMPOSE_FILE" && grep -q "ALL" "$COMPOSE_FILE"; then
    log_test "Linux capabilities dropped" "PASS" "Found cap_drop: ALL configuration"
else
    log_test "Linux capabilities dropped" "FAIL" "Missing capability dropping configuration"
fi

# Test 5: Check for read-only filesystems
if grep -q "read_only: true" "$COMPOSE_FILE"; then
    log_test "Read-only filesystems enabled" "PASS" "Found read_only: true configurations"
else
    log_test "Read-only filesystems enabled" "FAIL" "Missing read-only filesystem configurations"
fi

# Test 6: Check for resource limits
if grep -q "deploy:" "$COMPOSE_FILE" && grep -q "resources:" "$COMPOSE_FILE"; then
    log_test "Resource limits configured" "PASS" "Found resource limit configurations"
else
    log_test "Resource limits configured" "FAIL" "Missing resource limit configurations"
fi

# Test 7: Check for health checks
if grep -q "healthcheck:" "$COMPOSE_FILE"; then
    log_test "Health checks configured" "PASS" "Found healthcheck configurations"
else
    log_test "Health checks configured" "FAIL" "Missing health check configurations"
fi

echo ""
echo "## 2. Dockerfile Security Validation"
echo ""

# Test 8: Check for secure Dockerfiles
if [[ -f "backend/Dockerfile.secure" ]] && [[ -f "frontend/Dockerfile.secure" ]]; then
    log_test "Secure Dockerfiles exist" "PASS" "Found secure Dockerfile variants"
else
    log_test "Secure Dockerfiles exist" "FAIL" "Missing secure Dockerfile variants"
fi

# Test 9: Check for non-root users in Dockerfiles
if grep -q "USER appuser" backend/Dockerfile.secure 2>/dev/null; then
    log_test "Dockerfile non-root users" "PASS" "Found USER appuser in backend Dockerfile"
else
    log_test "Dockerfile non-root users" "FAIL" "Missing non-root user in backend Dockerfile"
fi

# Test 10: Check for multi-stage builds
if grep -q "FROM.*AS builder" backend/Dockerfile.secure 2>/dev/null; then
    log_test "Multi-stage builds used" "PASS" "Found multi-stage build configuration"
else
    log_test "Multi-stage builds used" "FAIL" "Missing multi-stage build configuration"
fi

echo ""
echo "## 3. Network Security Validation"
echo ""

# Test 11: Check for custom networks
if grep -q "networks:" "$COMPOSE_FILE" && grep -q "sutazai-network:" "$COMPOSE_FILE"; then
    log_test "Custom networks configured" "PASS" "Found custom network configuration"
else
    log_test "Custom networks configured" "FAIL" "Missing custom network configuration"
fi

# Test 12: Check for network isolation
if grep -q "enable_icc.*false" "$COMPOSE_FILE"; then
    log_test "Network isolation enabled" "PASS" "Found ICC disabled in network configuration"
else
    log_test "Network isolation enabled" "FAIL" "Missing network isolation configuration"
fi

echo ""
echo "## 4. Runtime Security Validation"
echo ""

# Test 13: Check if containers are running
if docker-compose -f "$COMPOSE_FILE" ps --services 2>/dev/null | head -1 > /dev/null 2>&1; then
    echo "Checking running containers for security compliance..."
    
    # Test running containers for non-root processes
    CONTAINERS=$(docker-compose -f "$COMPOSE_FILE" ps -q 2>/dev/null || true)
    
    if [[ -n "$CONTAINERS" ]]; then
        NON_ROOT_COUNT=0
        TOTAL_CONTAINERS=0
        
        for container in $CONTAINERS; do
            if [[ -n "$container" ]]; then
                ((TOTAL_CONTAINERS++))
                # Check if processes are running as non-root
                if docker exec "$container" sh -c "id -u" 2>/dev/null | grep -qv "^0$"; then
                    ((NON_ROOT_COUNT++))
                fi
            fi
        done
        
        if [[ $NON_ROOT_COUNT -gt 0 ]]; then
            log_test "Runtime non-root processes" "PASS" "$NON_ROOT_COUNT/$TOTAL_CONTAINERS containers running as non-root"
        else
            log_test "Runtime non-root processes" "FAIL" "All containers running as root"
        fi
    else
        log_test "Runtime security validation" "FAIL" "No running containers found"
    fi
else
    log_test "Runtime security validation" "FAIL" "Cannot check running containers - services not running"
fi

echo ""
echo "## 5. Image Security Validation"
echo ""

# Test 14: Scan secure images with Trivy (if they exist)
SECURE_IMAGES=("sutazai-backend-secure" "sutazai-frontend-secure")
IMAGES_SCANNED=0
CLEAN_IMAGES=0

for image in "${SECURE_IMAGES[@]}"; do
    if docker images | grep -q "$image"; then
        ((IMAGES_SCANNED++))
        echo "Scanning $image for vulnerabilities..."
        
        VULN_COUNT=$(trivy image "$image:latest" --severity HIGH,CRITICAL --format json --quiet 2>/dev/null | jq '[.Results[]?.Vulnerabilities // empty] | length' 2>/dev/null || echo "unknown")
        
        if [[ "$VULN_COUNT" == "0" ]]; then
            ((CLEAN_IMAGES++))
            log_test "Image $image vulnerability scan" "PASS" "No HIGH/CRITICAL vulnerabilities found"
        elif [[ "$VULN_COUNT" != "unknown" ]]; then
            log_test "Image $image vulnerability scan" "FAIL" "$VULN_COUNT HIGH/CRITICAL vulnerabilities found"
        else
            log_test "Image $image vulnerability scan" "FAIL" "Unable to scan image"
        fi
    fi
done

if [[ $IMAGES_SCANNED -eq 0 ]]; then
    log_test "Secure image validation" "FAIL" "No secure images found to scan"
fi

echo ""
echo "## 6. Configuration Security Validation"
echo ""

# Test 15: Check for secrets in environment variables
if grep -E "PASSWORD|SECRET|KEY" "$COMPOSE_FILE" | grep -qv "\${"; then
    log_test "Secrets management" "FAIL" "Found hardcoded secrets in compose file"
else
    log_test "Secrets management" "PASS" "No hardcoded secrets found in compose file"
fi

# Test 16: Check for privileged containers
if grep -q "privileged: true" "$COMPOSE_FILE"; then
    log_test "Privileged containers" "FAIL" "Found privileged containers in configuration"
else
    log_test "Privileged containers" "PASS" "No privileged containers found"
fi

# Test 17: Check for host network mode
if grep -q "network_mode: host" "$COMPOSE_FILE"; then
    log_test "Host network isolation" "FAIL" "Found host network mode usage"
else
    log_test "Host network isolation" "PASS" "Proper network isolation configured"
fi

echo ""

# Generate summary
cat >> "${VALIDATION_REPORT}" << EOF

## Validation Summary

### Test Results
- **Total Tests**: $((TESTS_PASSED + TESTS_FAILED))
- **Tests Passed**: $TESTS_PASSED
- **Tests Failed**: $TESTS_FAILED
- **Success Rate**: $(( (TESTS_PASSED * 100) / (TESTS_PASSED + TESTS_FAILED) ))%

### Security Score Calculation
EOF

# Calculate security score (0-10 scale)
TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
if [[ $TOTAL_TESTS -gt 0 ]]; then
    SECURITY_SCORE=$(( (TESTS_PASSED * 10) / TOTAL_TESTS ))
else
    SECURITY_SCORE=0
fi

echo "📊 Security Validation Summary:"
echo "   Total Tests: $TOTAL_TESTS"
echo "   Tests Passed: $TESTS_PASSED"
echo "   Tests Failed: $TESTS_FAILED"
echo "   Security Score: $SECURITY_SCORE/10"

cat >> "${VALIDATION_REPORT}" << EOF
- **Security Score**: ${SECURITY_SCORE}/10

### Recommendations

EOF

# Add recommendations based on failures
if [[ $TESTS_FAILED -gt 0 ]]; then
    cat >> "${VALIDATION_REPORT}" << EOF
#### Priority Fixes Required
1. Review failed tests above and implement missing security controls
2. Ensure all containers run as non-root users
3. Configure proper security contexts and capability dropping
4. Enable read-only filesystems where possible
5. Implement comprehensive health checks
6. Scan and update base images regularly

EOF
fi

cat >> "${VALIDATION_REPORT}" << EOF
#### Best Practices
1. Regular security scans with Trivy
2. Automated security testing in CI/CD pipeline
3. Network segmentation and isolation
4. Secrets management with external vaults
5. Regular security updates and patches
6. Container image signing and verification

**Report Generated**: $(date)
**Validation Script**: validate-container-security.sh
EOF

echo ""
if [[ $SECURITY_SCORE -ge 8 ]]; then
    echo -e "${GREEN}🎉 Excellent! Security score: $SECURITY_SCORE/10${NC}"
elif [[ $SECURITY_SCORE -ge 6 ]]; then
    echo -e "${YELLOW}⚠️  Good, but needs improvement. Security score: $SECURITY_SCORE/10${NC}"
else
    echo -e "${RED}🚨 Critical security issues found. Security score: $SECURITY_SCORE/10${NC}"
fi

echo ""
echo "📁 Detailed report saved to: ${VALIDATION_REPORT}"
echo ""

# Exit with appropriate code
if [[ $TESTS_FAILED -eq 0 ]]; then
    exit 0
else
    exit 1
fi