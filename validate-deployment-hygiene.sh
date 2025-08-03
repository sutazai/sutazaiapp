#!/bin/bash
#
# SutazAI Deployment Hygiene Validator
# 
# PURPOSE:
#   Validates that deployment configurations follow CLAUDE.md codebase hygiene standards
#
# USAGE:
#   ./validate-deployment-hygiene.sh
#

set -euo pipefail

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Validation results
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNINGS=0

log_pass() {
    echo -e "${GREEN}✅ PASS: $1${NC}"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
}

log_fail() {
    echo -e "${RED}❌ FAIL: $1${NC}"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
}

log_warn() {
    echo -e "${YELLOW}⚠️  WARN: $1${NC}"
    WARNINGS=$((WARNINGS + 1))
}

log_info() {
    echo -e "${BLUE}ℹ️  INFO: $1${NC}"
}

echo -e "${BOLD}SutazAI Deployment Hygiene Validation${NC}"
echo "======================================"
echo

# Rule 1: Single canonical deployment script
log_info "Checking Rule 1: Single canonical deployment script"
if [[ -f "deploy.sh" ]]; then
    log_pass "Canonical deploy.sh exists"
else
    log_fail "Missing canonical deploy.sh"
fi

# Check for duplicate deployment scripts
DUPLICATE_SCRIPTS=$(find . -name "deploy*.sh" -not -path "./deploy.sh" -not -path "./validate-deployment-hygiene.sh" | wc -l)
if [[ $DUPLICATE_SCRIPTS -eq 0 ]]; then
    log_pass "No duplicate deployment scripts found"
else
    log_fail "Found $DUPLICATE_SCRIPTS duplicate deployment scripts"
    find . -name "deploy*.sh" -not -path "./deploy.sh" -not -path "./validate-deployment-hygiene.sh"
fi

# Rule 2: Canonical docker-compose.yml
log_info "Checking Rule 2: Canonical docker-compose configuration"
if [[ -f "docker-compose.yml" ]]; then
    log_pass "Canonical docker-compose.yml exists"
    
    # Validate syntax
    if docker compose config >/dev/null 2>&1; then
        log_pass "docker-compose.yml has valid syntax"
    else
        log_fail "docker-compose.yml has syntax errors"
    fi
else
    log_fail "Missing canonical docker-compose.yml"
fi

# Check for unnecessary docker-compose duplicates in main directory
MAIN_COMPOSE_FILES=$(find . -maxdepth 1 -name "docker-compose*.yml" -not -name "docker-compose.yml" -not -name "docker-compose.monitoring.yml" -not -name "docker-compose.gpu.yml" -not -name "docker-compose.cpu-only.yml" | wc -l)
if [[ $MAIN_COMPOSE_FILES -eq 0 ]]; then
    log_pass "No unnecessary docker-compose files in main directory"
else
    log_warn "Found $MAIN_COMPOSE_FILES potentially unnecessary docker-compose files"
    find . -maxdepth 1 -name "docker-compose*.yml" -not -name "docker-compose.yml" -not -name "docker-compose.monitoring.yml" -not -name "docker-compose.gpu.yml" -not -name "docker-compose.cpu-only.yml"
fi

# Rule 3: No hardcoded secrets
log_info "Checking Rule 3: No hardcoded secrets"
# Check for literal password values (not variables or file reads)
if ! grep -r "password.*=['\"]" deploy.sh | grep -v "\${" >/dev/null 2>&1; then
    log_pass "No hardcoded passwords found in deploy.sh"
else
    log_fail "Found hardcoded passwords in deploy.sh"
    grep -r "password.*=['\"]" deploy.sh | grep -v "\${" || true
fi

# Check for hardcoded secrets in docker-compose.yml
if ! grep -i "password.*:" docker-compose.yml | grep -v "\${" >/dev/null 2>&1; then
    log_pass "No hardcoded passwords in docker-compose.yml"
else
    log_warn "Check passwords in docker-compose.yml for proper environment variable usage"
fi

# Rule 4: Environment variable configuration
log_info "Checking Rule 4: Environment variable configuration"
if grep -q "POSTGRES_PASSWORD:-" deploy.sh; then
    log_pass "PostgreSQL password uses environment variables with fallback"
else
    log_fail "PostgreSQL password not properly configured with environment variables"
fi

if grep -q "REDIS_PASSWORD:-" deploy.sh; then
    log_pass "Redis password uses environment variables with fallback"
else
    log_fail "Redis password not properly configured with environment variables"
fi

# Rule 5: Proper documentation
log_info "Checking Rule 5: Proper documentation"
if [[ -f "DEPLOYMENT.md" ]]; then
    log_pass "Deployment documentation exists"
else
    log_fail "Missing DEPLOYMENT.md documentation"
fi

# Check deploy.sh has proper header documentation
if head -50 deploy.sh | grep -q "DESCRIPTION:"; then
    log_pass "deploy.sh has proper header documentation"
else
    log_fail "deploy.sh missing proper header documentation"
fi

# Rule 6: Executable permissions
log_info "Checking Rule 6: Executable permissions"
if [[ -x "deploy.sh" ]]; then
    log_pass "deploy.sh has executable permissions"
else
    log_fail "deploy.sh missing executable permissions"
    chmod +x deploy.sh
    log_info "Fixed: Made deploy.sh executable"
fi

# Rule 7: Error handling
log_info "Checking Rule 7: Error handling"
if grep -q "set -euo pipefail" deploy.sh; then
    log_pass "deploy.sh has proper error handling (set -euo pipefail)"
else
    log_fail "deploy.sh missing proper error handling"
fi

if grep -q "trap.*cleanup" deploy.sh; then
    log_pass "deploy.sh has cleanup trap handling"
else
    log_warn "deploy.sh could benefit from cleanup trap handling"
fi

# Rule 8: Idempotent operations
log_info "Checking Rule 8: Idempotent operations"
if grep -q "docker compose.*up.*-d" deploy.sh; then
    log_pass "Uses docker compose for idempotent deployments"
else
    log_warn "Consider using docker compose for idempotent operations"
fi

# Rule 9: Security best practices
log_info "Checking Rule 9: Security best practices"
if [[ -d "secrets" ]]; then
    log_pass "Secrets directory exists"
    
    # Check secrets directory permissions
    if [[ "$(stat -c %a secrets)" == "700" ]]; then
        log_pass "Secrets directory has proper permissions (700)"
    else
        log_warn "Secrets directory should have 700 permissions"
    fi
else
    log_warn "No secrets directory found"
fi

# Rule 10: Clean structure
log_info "Checking Rule 10: Clean structure"
if [[ -d "logs" ]]; then
    log_pass "Logs directory exists"
else
    log_info "Logs directory will be created on first run"
fi

# Check for proper .gitignore
if [[ -f ".gitignore" ]] && grep -q "secrets/" .gitignore; then
    log_pass "Secrets properly ignored in .gitignore"
else
    log_warn "Ensure secrets/ is in .gitignore"
fi

# Summary
echo
echo -e "${BOLD}Validation Summary${NC}"
echo "=================="
echo -e "${GREEN}Passed: $PASSED_CHECKS${NC}"
echo -e "${RED}Failed: $FAILED_CHECKS${NC}"
echo -e "${YELLOW}Warnings: $WARNINGS${NC}"

if [[ $FAILED_CHECKS -eq 0 ]]; then
    echo
    echo -e "${GREEN}${BOLD}🎉 All critical hygiene checks passed!${NC}"
    echo -e "${GREEN}Deployment configuration follows CLAUDE.md standards.${NC}"
    exit 0
else
    echo
    echo -e "${RED}${BOLD}❌ $FAILED_CHECKS critical issues found.${NC}"
    echo -e "${RED}Please fix the issues above before deployment.${NC}"
    exit 1
fi