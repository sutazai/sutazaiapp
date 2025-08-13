#!/bin/bash
# ============================================================================
# ULTRAFIX SECURITY VALIDATION SCRIPT
# ============================================================================
# Purpose: Comprehensive security validation for consolidated Dockerfiles
# Author: DevOps Infrastructure Manager - ULTRAFIX Operation
# Date: August 10, 2025
# Version: v1.0.0 - Production Security Validation
# ============================================================================

set -euo pipefail

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs/security-validation"
REPORT_FILE="$LOG_DIR/security-validation-report-$(date +%Y%m%d_%H%M%S).json"

# Color coding
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Security metrics
TOTAL_FILES=0
SECURE_FILES=0
ROOT_VIOLATIONS=0
MISSING_HEALTH_CHECKS=0
MISSING_USER_DIRECTIVE=0
HARDCODED_SECRETS=0
INSECURE_PRACTICES=0

setup_logging() {
    mkdir -p "$LOG_DIR"
    exec 1> >(tee -a "$LOG_DIR/security-validation-$(date +%Y%m%d_%H%M%S).log")
    exec 2>&1
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

# Security validation functions
check_non_root_user() {
    local dockerfile="$1"
    local relative_path="$2"
    
    # Check for USER directive
    if ! grep -q "^USER " "$dockerfile"; then
        # Check if it inherits from secure base
        if grep -q "FROM sutazai-.*:v[0-9]" "$dockerfile"; then
            log_success "✅ $relative_path: Inherits non-root user from secure base"
            return 0
        else
            log_error "❌ $relative_path: Missing USER directive and not using secure base"
            ((MISSING_USER_DIRECTIVE++))
            return 1
        fi
    fi
    
    # Check for root user
    if grep -q "USER root\|USER 0" "$dockerfile"; then
        log_error "❌ $relative_path: Explicitly runs as root"
        ((ROOT_VIOLATIONS++))
        return 1
    fi
    
    # Check for proper non-root user
    if grep -q "USER appuser\|USER [1-9][0-9][0-9]" "$dockerfile"; then
        log_success "✅ $relative_path: Uses non-root user"
        return 0
    fi
    
    log_warning "⚠️  $relative_path: User directive present but unclear if non-root"
    return 1
}

check_health_check() {
    local dockerfile="$1"
    local relative_path="$2"
    
    if grep -q "HEALTHCHECK" "$dockerfile"; then
        log_success "✅ $relative_path: Health check configured"
        return 0
    else
        log_warning "⚠️  $relative_path: Missing health check"
        ((MISSING_HEALTH_CHECKS++))
        return 1
    fi
}

check_secrets() {
    local dockerfile="$1"
    local relative_path="$2"
    
    # Common patterns for hardcoded secrets
    SECRET_PATTERNS=(
        "password.*="
        "secret.*="
        "key.*="
        "token.*="
        "apikey.*="
        "api_key.*="
    )
    
    for pattern in "${SECRET_PATTERNS[@]}"; do
        if grep -i "$pattern" "$dockerfile" | grep -v "_FILE\|/config/\|/secrets/" | grep -q "="; then
            log_error "❌ $relative_path: Potential hardcoded secret: $(grep -i "$pattern" "$dockerfile" | head -1)"
            ((HARDCODED_SECRETS++))
            return 1
        fi
    done
    
    log_success "✅ $relative_path: No hardcoded secrets detected"
    return 0
}

check_insecure_practices() {
    local dockerfile="$1"
    local relative_path="$2"
    local issues=0
    
    # Check for curl without SSL verification
    if grep -q "curl.*-k\|curl.*--insecure" "$dockerfile"; then
        log_error "❌ $relative_path: Insecure curl usage (disabled SSL verification)"
        ((issues++))
    fi
    
    # Check for wget without SSL verification
    if grep -q "wget.*--no-check-certificate" "$dockerfile"; then
        log_error "❌ $relative_path: Insecure wget usage (disabled SSL verification)"
        ((issues++))
    fi
    
    # Check for ADD with URLs (prefer COPY)
    if grep -q "^ADD http" "$dockerfile"; then
        log_warning "⚠️  $relative_path: Using ADD with URL (consider COPY for local files)"
        ((issues++))
    fi
    
    # Check for running as root in RUN commands
    if grep -q "sudo\|su -\|su root" "$dockerfile"; then
        log_warning "⚠️  $relative_path: Contains sudo/su commands"
        ((issues++))
    fi
    
    # Check for apt-get without cleanup
    if grep -q "apt-get install" "$dockerfile" && ! grep -q "rm -rf /var/lib/apt/lists" "$dockerfile"; then
        log_warning "⚠️  $relative_path: apt-get without cleanup"
        ((issues++))
    fi
    
    if [ $issues -eq 0 ]; then
        log_success "✅ $relative_path: No insecure practices detected"
        return 0
    else
        ((INSECURE_PRACTICES+=issues))
        return 1
    fi
}

check_base_image_security() {
    local dockerfile="$1"
    local relative_path="$2"
    
    # Extract base image
    base_image=$(grep "^FROM " "$dockerfile" | head -1 | awk '{print $2}')
    
    # Check for secure base images
    case $base_image in
        sutazai-python-agent-master:v2|sutazai-ai-ml-cuda:v1|sutazai-database-secure:v1)
            log_success "✅ $relative_path: Uses secure consolidated base image ($base_image)"
            return 0
            ;;
        *alpine*|*slim*)
            log_success "✅ $relative_path: Uses   base image ($base_image)"
            return 0
            ;;
        *latest)
            log_warning "⚠️  $relative_path: Uses 'latest' tag ($base_image) - consider pinning version"
            return 1
            ;;
        ubuntu:*|debian:*|centos:*)
            log_warning "⚠️  $relative_path: Uses full OS base image ($base_image) - consider slimmer alternative"
            return 1
            ;;
        *)
            log_info "ℹ️  $relative_path: Uses specialized base image ($base_image)"
            return 0
            ;;
    esac
}

validate_dockerfile() {
    local dockerfile="$1"
    local relative_path="${dockerfile#$PROJECT_ROOT/}"
    
    log_info "Validating: $relative_path"
    
    local security_score=0
    local max_score=5
    
    # Run security checks
    check_non_root_user "$dockerfile" "$relative_path" && ((security_score++))
    check_health_check "$dockerfile" "$relative_path" && ((security_score++))
    check_secrets "$dockerfile" "$relative_path" && ((security_score++))
    check_insecure_practices "$dockerfile" "$relative_path" && ((security_score++))
    check_base_image_security "$dockerfile" "$relative_path" && ((security_score++))
    
    # Calculate security percentage
    security_percentage=$((security_score * 100 / max_score))
    
    if [ $security_score -eq $max_score ]; then
        log_success "✅ $relative_path: SECURE ($security_percentage%)"
        ((SECURE_FILES++))
    elif [ $security_percentage -ge 80 ]; then
        log_warning "⚠️  $relative_path: MOSTLY SECURE ($security_percentage%)"
    else
        log_error "❌ $relative_path: NEEDS ATTENTION ($security_percentage%)"
    fi
    
    ((TOTAL_FILES++))
    
    return 0
}

generate_security_report() {
    local overall_security_percentage=$((SECURE_FILES * 100 / TOTAL_FILES))
    
    # Generate JSON report
    cat > "$REPORT_FILE" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "scan_type": "dockerfile_security_validation",
  "project_root": "$PROJECT_ROOT",
  "summary": {
    "total_files": $TOTAL_FILES,
    "secure_files": $SECURE_FILES,
    "security_percentage": $overall_security_percentage,
    "violations": {
      "root_violations": $ROOT_VIOLATIONS,
      "missing_health_checks": $MISSING_HEALTH_CHECKS,
      "missing_user_directive": $MISSING_USER_DIRECTIVE,
      "hardcoded_secrets": $HARDCODED_SECRETS,
      "insecure_practices": $INSECURE_PRACTICES
    }
  },
  "recommendations": [
    "Ensure all containers run as non-root users",
    "Add health checks to all services",
    "Use environment variables or mounted secrets instead of hardcoded values",
    "Prefer   base images (Alpine, slim variants)",
    "Pin base image versions instead of using 'latest'"
  ],
  "compliance_status": "$([ $overall_security_percentage -ge 90 ] && echo "COMPLIANT" || echo "NEEDS_IMPROVEMENT")"
}
EOF
    
    log_info "Security report saved to: $REPORT_FILE"
}

show_final_summary() {
    local overall_security_percentage=$((SECURE_FILES * 100 / TOTAL_FILES))
    
    echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                        SECURITY VALIDATION SUMMARY                           ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    
    echo -e "\n📊 ${BLUE}SECURITY METRICS:${NC}"
    echo -e "   Total Dockerfiles Scanned: ${BLUE}$TOTAL_FILES${NC}"
    echo -e "   Fully Secure Files: ${GREEN}$SECURE_FILES${NC}"
    echo -e "   Overall Security Score: ${GREEN}$overall_security_percentage%${NC}"
    
    echo -e "\n🚨 ${BLUE}SECURITY ISSUES FOUND:${NC}"
    echo -e "   Root User Violations: ${RED}$ROOT_VIOLATIONS${NC}"
    echo -e "   Missing User Directives: ${YELLOW}$MISSING_USER_DIRECTIVE${NC}"
    echo -e "   Missing Health Checks: ${YELLOW}$MISSING_HEALTH_CHECKS${NC}"
    echo -e "   Hardcoded Secrets: ${RED}$HARDCODED_SECRETS${NC}"
    echo -e "   Insecure Practices: ${YELLOW}$INSECURE_PRACTICES${NC}"
    
    echo -e "\n🎯 ${BLUE}COMPLIANCE STATUS:${NC}"
    if [ $overall_security_percentage -ge 95 ]; then
        echo -e "   ${GREEN}✅ EXCELLENT SECURITY - Production Ready${NC}"
    elif [ $overall_security_percentage -ge 90 ]; then
        echo -e "   ${GREEN}✅ GOOD SECURITY - Minor improvements needed${NC}"
    elif [ $overall_security_percentage -ge 80 ]; then
        echo -e "   ${YELLOW}⚠️  ACCEPTABLE SECURITY - Some issues need attention${NC}"
    else
        echo -e "   ${RED}❌ SECURITY IMPROVEMENTS REQUIRED${NC}"
    fi
    
    echo -e "\n📋 ${BLUE}RECOMMENDED ACTIONS:${NC}"
    
    if [ $ROOT_VIOLATIONS -gt 0 ]; then
        echo -e "   ${RED}•${NC} Fix $ROOT_VIOLATIONS containers running as root"
    fi
    
    if [ $HARDCODED_SECRETS -gt 0 ]; then
        echo -e "   ${RED}•${NC} Remove $HARDCODED_SECRETS hardcoded secrets"
    fi
    
    if [ $MISSING_HEALTH_CHECKS -gt 0 ]; then
        echo -e "   ${YELLOW}•${NC} Add health checks to $MISSING_HEALTH_CHECKS services"
    fi
    
    if [ $INSECURE_PRACTICES -gt 0 ]; then
        echo -e "   ${YELLOW}•${NC} Address $INSECURE_PRACTICES insecure practices"
    fi
    
    if [ $overall_security_percentage -ge 90 ]; then
        echo -e "   ${GREEN}•${NC} Security validation passed! Ready for production deployment"
    fi
    
    echo -e "\n📄 Detailed report: ${BLUE}$REPORT_FILE${NC}"
    echo ""
}

main() {
    setup_logging
    
    log_info "Starting ULTRAFIX security validation..."
    log_info "Project root: $PROJECT_ROOT"
    
    # Find all Dockerfiles (excluding archives and backups)
    find "$PROJECT_ROOT" -name "Dockerfile*" -not -path "*/node_modules/*" -not -path "*/archive/*" -not -path "*/backups/*" -not -name "*.backup" | sort | while read -r dockerfile; do
        validate_dockerfile "$dockerfile"
    done
    
    # Generate report and summary
    generate_security_report
    show_final_summary
    
    # Exit with appropriate code
    overall_security_percentage=$((SECURE_FILES * 100 / TOTAL_FILES))
    if [ $overall_security_percentage -ge 90 ]; then
        exit 0  # Success
    else
        exit 1  # Needs improvement
    fi
}

# Execute main function
main "$@"