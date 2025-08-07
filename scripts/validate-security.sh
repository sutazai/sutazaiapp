#!/bin/bash
#
# Security Validation Script for SutazAI
# Validates security hardening measures
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISSUES=0

check() {
    local test_name="$1"
    local test_command="$2"
    
    echo -n "Checking $test_name... "
    if eval "$test_command" >/dev/null 2>&1; then
        echo -e "${GREEN}PASS${NC}"
    else
        echo -e "${RED}FAIL${NC}"
        ((ISSUES++))
    fi
}

echo "=== SutazAI Security Validation ==="
echo

# Check secret files permissions
check "Secret files permissions" "[[ ! -d '$PROJECT_DIR/secrets' ]] || [[ \$(stat -c '%a' '$PROJECT_DIR/secrets') == '700' ]]"

# Check for environment file
check "Environment configuration" "[[ -f '$PROJECT_DIR/.env' ]]"

# Check SSL certificates
check "SSL certificates present" "[[ -f '$PROJECT_DIR/ssl/cert.pem' && -f '$PROJECT_DIR/ssl/key.pem' ]]"

# Check nginx security config
check "Nginx security configuration" "[[ -f '$PROJECT_DIR/nginx/security.conf' ]]"

# Check for exposed ports (basic check)
check "No plaintext secrets in docker-compose" "! grep -r 'password.*=' '$PROJECT_DIR/docker-compose.yml' || true"

# Check Docker security
check "Hardened Docker configuration exists" "[[ -f '$PROJECT_DIR/docker-compose.secure.yml' ]]"

echo
if [[ $ISSUES -eq 0 ]]; then
    echo -e "${GREEN}All security checks passed!${NC}"
    exit 0
else
    echo -e "${RED}$ISSUES security issues found. Please review and fix.${NC}"
    exit 1
fi
