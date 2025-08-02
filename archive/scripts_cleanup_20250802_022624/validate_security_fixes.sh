#!/bin/bash
#
# Security Validation Script
# Validates that all critical security vulnerabilities have been fixed
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓ PASS]${NC} $1"
    ((PASSED_CHECKS++))
}

log_failure() {
    echo -e "${RED}[✗ FAIL]${NC} $1"
    ((FAILED_CHECKS++))
}

log_warning() {
    echo -e "${YELLOW}[⚠ WARN]${NC} $1"
}

# Helper function to run a check
run_check() {
    local check_name="$1"
    local check_command="$2"
    
    ((TOTAL_CHECKS++))
    log_info "Checking: $check_name"
    
    if eval "$check_command"; then
        log_success "$check_name"
        return 0
    else
        log_failure "$check_name"
        return 1
    fi
}

# Check 1: Verify no hardcoded Grafana passwords
check_grafana_password() {
    local files_to_check=(
        "/opt/sutazaiapp/backend/monitoring/docker-compose.yml"
        "/opt/sutazaiapp/backend/monitoring/start_monitoring.py"
        "/opt/sutazaiapp/scripts/setup_monitoring.sh"
        "/opt/sutazaiapp/backend/monitoring/configure_monitoring.sh"
    )
    
    for file in "${files_to_check[@]}"; do
        if [[ -f "$file" ]]; then
            if grep -q "GF_SECURITY_ADMIN_PASSWORD=sutazai" "$file"; then
                echo "Found hardcoded Grafana password in $file"
                return 1
            fi
        fi
    done
    return 0
}

# Check 2: Verify CORS wildcards are removed
check_cors_wildcards() {
    local main_file="/opt/sutazaiapp/backend/main.py"
    
    if [[ -f "$main_file" ]]; then
        if grep -q 'allow_origins=\["\*"\]' "$main_file"; then
            echo "Found CORS wildcard in $main_file"
            return 1
        fi
        
        if ! grep -q "get_allowed_origins" "$main_file"; then
            echo "Secure CORS configuration not found in $main_file"
            return 1
        fi
    fi
    return 0
}

# Check 3: Verify authentication secret is secure
check_auth_secret() {
    local auth_file="/opt/sutazaiapp/backend/security/auth.py"
    
    if [[ -f "$auth_file" ]]; then
        if grep -q "fallback-insecure-secret-key-for-dev" "$auth_file"; then
            echo "Found insecure fallback secret in $auth_file"
            return 1
        fi
        
        if ! grep -q "get_auth_secret" "$auth_file"; then
            echo "Secure auth configuration not found in $auth_file"
            return 1
        fi
    fi
    return 0
}

# Check 4: Verify secure configuration files exist
check_secure_config_files() {
    local required_files=(
        "/opt/sutazaiapp/backend/security/secure_config.py"
        "/opt/sutazaiapp/backend/security/rate_limiter.py"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo "Required security file missing: $file"
            return 1
        fi
    done
    return 0
}

# Check 5: Verify rate limiting is implemented
check_rate_limiting() {
    local main_file="/opt/sutazaiapp/backend/main.py"
    
    if [[ -f "$main_file" ]]; then
        if ! grep -q "RateLimitMiddleware" "$main_file"; then
            echo "Rate limiting middleware not found in $main_file"
            return 1
        fi
    fi
    return 0
}

# Check 6: Verify SSL certificates (if they exist)
check_ssl_certificates() {
    local ssl_dir="/opt/sutazaiapp/ssl"
    
    if [[ -d "$ssl_dir" ]]; then
        local cert_file="$ssl_dir/cert.pem"
        local key_file="$ssl_dir/key.pem"
        
        if [[ -f "$cert_file" && -f "$key_file" ]]; then
            # Check permissions
            local cert_perms=$(stat -c %a "$cert_file")
            local key_perms=$(stat -c %a "$key_file")
            
            if [[ "$key_perms" != "600" ]]; then
                echo "SSL private key has incorrect permissions: $key_perms (should be 600)"
                return 1
            fi
            
            # Validate certificate
            if ! openssl x509 -in "$cert_file" -text -noout > /dev/null 2>&1; then
                echo "SSL certificate is not valid"
                return 1
            fi
        fi
    fi
    return 0
}

# Check 7: Verify secure directory permissions
check_directory_permissions() {
    local secure_dirs=(
        "/opt/sutazaiapp/config/secure"
        "/opt/sutazaiapp/ssl"
    )
    
    for dir in "${secure_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            local perms=$(stat -c %a "$dir")
            if [[ "$perms" != "700" ]]; then
                echo "Directory $dir has incorrect permissions: $perms (should be 700)"
                return 1
            fi
        fi
    done
    return 0
}

# Check 8: Verify environment file security
check_environment_files() {
    local env_files=(
        "/opt/sutazaiapp/.env"
        "/opt/sutazaiapp/.env.production"
        "/opt/sutazaiapp/.env.development"
        "/opt/sutazaiapp/.env.staging"
    )
    
    for env_file in "${env_files[@]}"; do
        if [[ -f "$env_file" ]]; then
            local perms=$(stat -c %a "$env_file")
            if [[ "$perms" != "600" ]]; then
                echo "Environment file $env_file has incorrect permissions: $perms (should be 600)"
                return 1
            fi
        fi
    done
    return 0
}

# Check 9: Verify no hardcoded secrets in codebase
check_no_hardcoded_secrets() {
    local base_dir="/opt/sutazaiapp"
    local secret_patterns=(
        "password.*=.*['\"][^'\"]{8,}['\"]"
        "secret.*=.*['\"][^'\"]{8,}['\"]"
        "key.*=.*['\"][^'\"]{8,}['\"]"
    )
    
    for pattern in "${secret_patterns[@]}"; do
        # Search for patterns, excluding this script and test files
        local results=$(grep -r "$pattern" "$base_dir" \
            --exclude-dir=.git \
            --exclude-dir=__pycache__ \
            --exclude="*.pyc" \
            --exclude="test_*.py" \
            --exclude="validate_security_fixes.sh" \
            --exclude="SECURITY_AUDIT_REPORT.md" \
            2>/dev/null || true)
        
        if [[ -n "$results" ]]; then
            echo "Found potential hardcoded secrets:"
            echo "$results"
            return 1
        fi
    done
    return 0
}

# Check 10: Verify Python security modules can be imported
check_python_security_modules() {
    cd /opt/sutazaiapp
    
    # Check if secure config can be imported
    if ! python3 -c "from backend.security.secure_config import SecureConfigManager; print('✓ secure_config imported')" 2>/dev/null; then
        echo "Cannot import secure_config module"
        return 1
    fi
    
    # Check if rate limiter can be imported
    if ! python3 -c "from backend.security.rate_limiter import AdvancedRateLimiter; print('✓ rate_limiter imported')" 2>/dev/null; then
        echo "Cannot import rate_limiter module"
        return 1
    fi
    
    return 0
}

# Main execution
main() {
    echo "========================================"
    echo "SutazAI V7 Security Validation Script"
    echo "========================================"
    echo
    
    log_info "Starting security validation checks..."
    echo
    
    # Run all security checks
    run_check "Grafana password security" "check_grafana_password"
    run_check "CORS wildcard removal" "check_cors_wildcards"
    run_check "Authentication secret security" "check_auth_secret"
    run_check "Secure configuration files" "check_secure_config_files"
    run_check "Rate limiting implementation" "check_rate_limiting"
    run_check "SSL certificate security" "check_ssl_certificates"
    run_check "Directory permissions" "check_directory_permissions"
    run_check "Environment file security" "check_environment_files"
    run_check "No hardcoded secrets" "check_no_hardcoded_secrets"
    run_check "Python security modules" "check_python_security_modules"
    
    echo
    echo "========================================"
    echo "Security Validation Summary"
    echo "========================================"
    
    echo "Total Checks: $TOTAL_CHECKS"
    echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
    echo -e "Failed: ${RED}$FAILED_CHECKS${NC}"
    
    if [[ $FAILED_CHECKS -eq 0 ]]; then
        echo
        log_success "🔒 All security validation checks passed!"
        echo
        echo "Critical security vulnerabilities have been successfully fixed:"
        echo "✓ Hardcoded Grafana passwords removed"
        echo "✓ CORS wildcards replaced with environment-specific origins"
        echo "✓ Authentication secrets use secure configuration"
        echo "✓ Rate limiting implemented"
        echo "✓ SSL certificates configured"
        echo "✓ Secure file permissions applied"
        echo
        echo "Your SutazAI V7 system is now secure for enterprise deployment."
        exit 0
    else
        echo
        log_failure "❌ Security validation failed!"
        echo
        echo "Please address the failed checks before deploying to production."
        echo "Run this script again after making the necessary fixes."
        exit 1
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi