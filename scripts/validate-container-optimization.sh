#!/bin/bash

# Container Optimization Validation Script
# Validates security, resource limits, health checks, and networking
# Created: August 10, 2025

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="${1:-docker-compose.yml}"
REPORT_FILE="container-optimization-report.json"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo -e "${BLUE}Container Optimization Validation Tool${NC}"
echo -e "${BLUE}=====================================>${NC}"
echo "Analyzing: $COMPOSE_FILE"
echo "Timestamp: $TIMESTAMP"
echo ""

# Initialize counters
TOTAL_SERVICES=0
SECURE_SERVICES=0
RESOURCE_LIMITED=0
HEALTH_CHECKED=0
NETWORK_ISOLATED=0
ISSUES_FOUND=0

# Initialize report
cat > "$REPORT_FILE" << EOF
{
  "timestamp": "$TIMESTAMP",
  "compose_file": "$COMPOSE_FILE",
  "validation_results": {
EOF

# Function to check if service runs as non-root
check_non_root() {
    local service=$1
    local user=$(docker-compose -f "$COMPOSE_FILE" config | yq eval ".services.$service.user" -)
    
    if [[ "$user" != "null" && "$user" != "0:0" && "$user" != "root" ]]; then
        echo -e "  ${GREEN}✓${NC} Non-root user: $user"
        return 0
    else
        echo -e "  ${RED}✗${NC} Running as root or no user specified"
        return 1
    fi
}

# Function to check resource limits
check_resource_limits() {
    local service=$1
    local memory_limit=$(docker-compose -f "$COMPOSE_FILE" config | yq eval ".services.$service.deploy.resources.limits.memory" -)
    local cpu_limit=$(docker-compose -f "$COMPOSE_FILE" config | yq eval ".services.$service.deploy.resources.limits.cpus" -)
    
    if [[ "$memory_limit" != "null" && "$cpu_limit" != "null" ]]; then
        echo -e "  ${GREEN}✓${NC} Resource limits: Memory=$memory_limit, CPU=$cpu_limit"
        return 0
    else
        echo -e "  ${YELLOW}⚠${NC} Missing resource limits"
        return 1
    fi
}

# Function to check health checks
check_health_check() {
    local service=$1
    local healthcheck=$(docker-compose -f "$COMPOSE_FILE" config | yq eval ".services.$service.healthcheck" -)
    
    if [[ "$healthcheck" != "null" ]]; then
        local interval=$(echo "$healthcheck" | yq eval ".interval" -)
        local timeout=$(echo "$healthcheck" | yq eval ".timeout" -)
        local retries=$(echo "$healthcheck" | yq eval ".retries" -)
        echo -e "  ${GREEN}✓${NC} Health check configured (interval=$interval, timeout=$timeout, retries=$retries)"
        return 0
    else
        echo -e "  ${YELLOW}⚠${NC} No health check configured"
        return 1
    fi
}

# Function to check security options
check_security_options() {
    local service=$1
    local security_opt=$(docker-compose -f "$COMPOSE_FILE" config | yq eval ".services.$service.security_opt" -)
    local cap_drop=$(docker-compose -f "$COMPOSE_FILE" config | yq eval ".services.$service.cap_drop" -)
    
    if [[ "$security_opt" != "null" ]] && [[ "$cap_drop" != "null" ]]; then
        echo -e "  ${GREEN}✓${NC} Security hardening applied"
        return 0
    else
        echo -e "  ${YELLOW}⚠${NC} Missing security hardening"
        return 1
    fi
}

# Function to check network configuration
check_network_config() {
    local service=$1
    local networks=$(docker-compose -f "$COMPOSE_FILE" config | yq eval ".services.$service.networks" -)
    
    if [[ "$networks" != "null" ]]; then
        echo -e "  ${GREEN}✓${NC} Network isolation configured"
        return 0
    else
        echo -e "  ${YELLOW}⚠${NC} No network isolation"
        return 1
    fi
}

# Check if required tools are installed
if ! command -v yq &> /dev/null; then
    echo -e "${RED}Error: yq is not installed. Please install it first.${NC}"
    echo "Install with: brew install yq (macOS) or snap install yq (Linux)"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: docker-compose is not installed.${NC}"
    exit 1
fi

# Validate compose file exists
if [[ ! -f "$COMPOSE_FILE" ]]; then
    echo -e "${RED}Error: Compose file not found: $COMPOSE_FILE${NC}"
    exit 1
fi

# Get list of services
SERVICES=$(docker-compose -f "$COMPOSE_FILE" config | yq eval '.services | keys | .[]' -)

echo -e "${BLUE}Validating Services:${NC}"
echo "===================="

# Validate each service
for service in $SERVICES; do
    TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
    echo -e "\n${BLUE}Service: $service${NC}"
    
    service_issues=0
    
    # Check non-root user
    if check_non_root "$service"; then
        SECURE_SERVICES=$((SECURE_SERVICES + 1))
    else
        service_issues=$((service_issues + 1))
    fi
    
    # Check resource limits
    if check_resource_limits "$service"; then
        RESOURCE_LIMITED=$((RESOURCE_LIMITED + 1))
    else
        service_issues=$((service_issues + 1))
    fi
    
    # Check health checks
    if check_health_check "$service"; then
        HEALTH_CHECKED=$((HEALTH_CHECKED + 1))
    else
        service_issues=$((service_issues + 1))
    fi
    
    # Check security options
    if ! check_security_options "$service"; then
        service_issues=$((service_issues + 1))
    fi
    
    # Check network configuration
    if check_network_config "$service"; then
        NETWORK_ISOLATED=$((NETWORK_ISOLATED + 1))
    else
        service_issues=$((service_issues + 1))
    fi
    
    if [[ $service_issues -eq 0 ]]; then
        echo -e "  ${GREEN}✓ Service fully optimized${NC}"
    else
        echo -e "  ${YELLOW}⚠ $service_issues optimization opportunities found${NC}"
        ISSUES_FOUND=$((ISSUES_FOUND + service_issues))
    fi
done

# Calculate scores
SECURITY_SCORE=$((SECURE_SERVICES * 100 / TOTAL_SERVICES))
RESOURCE_SCORE=$((RESOURCE_LIMITED * 100 / TOTAL_SERVICES))
HEALTH_SCORE=$((HEALTH_CHECKED * 100 / TOTAL_SERVICES))
NETWORK_SCORE=$((NETWORK_ISOLATED * 100 / TOTAL_SERVICES))
OVERALL_SCORE=$(((SECURITY_SCORE + RESOURCE_SCORE + HEALTH_SCORE + NETWORK_SCORE) / 4))

# Display summary
echo -e "\n${BLUE}=====================================>${NC}"
echo -e "${BLUE}Optimization Summary${NC}"
echo -e "${BLUE}=====================================>${NC}"
echo ""
echo -e "Total Services Analyzed: ${TOTAL_SERVICES}"
echo -e "Security Score: ${SECURITY_SCORE}% (${SECURE_SERVICES}/${TOTAL_SERVICES} non-root)"
echo -e "Resource Score: ${RESOURCE_SCORE}% (${RESOURCE_LIMITED}/${TOTAL_SERVICES} limited)"
echo -e "Health Score: ${HEALTH_SCORE}% (${HEALTH_CHECKED}/${TOTAL_SERVICES} monitored)"
echo -e "Network Score: ${NETWORK_SCORE}% (${NETWORK_ISOLATED}/${TOTAL_SERVICES} isolated)"
echo -e "Overall Score: ${OVERALL_SCORE}%"
echo ""

# Provide recommendations based on score
if [[ $OVERALL_SCORE -ge 90 ]]; then
    echo -e "${GREEN}✓ Excellent! Containers are production-ready.${NC}"
elif [[ $OVERALL_SCORE -ge 70 ]]; then
    echo -e "${YELLOW}⚠ Good progress, but some optimizations needed.${NC}"
else
    echo -e "${RED}✗ Significant improvements required for production.${NC}"
fi

# Complete JSON report
cat >> "$REPORT_FILE" << EOF
    "summary": {
      "total_services": $TOTAL_SERVICES,
      "secure_services": $SECURE_SERVICES,
      "resource_limited": $RESOURCE_LIMITED,
      "health_checked": $HEALTH_CHECKED,
      "network_isolated": $NETWORK_ISOLATED,
      "issues_found": $ISSUES_FOUND
    },
    "scores": {
      "security": $SECURITY_SCORE,
      "resources": $RESOURCE_SCORE,
      "health": $HEALTH_SCORE,
      "network": $NETWORK_SCORE,
      "overall": $OVERALL_SCORE
    },
    "recommendations": []
  }
}
EOF

# Generate recommendations
echo -e "\n${BLUE}Recommendations:${NC}"
echo "================"

if [[ $SECURITY_SCORE -lt 100 ]]; then
    echo -e "${YELLOW}Security:${NC}"
    echo "  - Convert remaining root containers to non-root users"
    echo "  - Add security_opt: [no-new-privileges:true] to all services"
    echo "  - Implement cap_drop: [ALL] with minimal cap_add"
fi

if [[ $RESOURCE_SCORE -lt 100 ]]; then
    echo -e "${YELLOW}Resources:${NC}"
    echo "  - Add memory and CPU limits to all services"
    echo "  - Configure PID limits to prevent fork bombs"
    echo "  - Set appropriate reservations for guaranteed resources"
fi

if [[ $HEALTH_SCORE -lt 100 ]]; then
    echo -e "${YELLOW}Health Monitoring:${NC}"
    echo "  - Add health checks to all services"
    echo "  - Use consistent intervals (30s recommended)"
    echo "  - Configure appropriate timeouts and retries"
fi

if [[ $NETWORK_SCORE -lt 100 ]]; then
    echo -e "${YELLOW}Networking:${NC}"
    echo "  - Implement network segmentation (internal/external)"
    echo "  - Use internal networks for databases"
    echo "  - Minimize port exposures"
fi

echo -e "\n${GREEN}Report saved to: $REPORT_FILE${NC}"

# K3s readiness check
echo -e "\n${BLUE}K3s Readiness Check:${NC}"
echo "===================="

if [[ -f "k3s-deployment.yaml" ]]; then
    echo -e "${GREEN}✓${NC} K3s deployment manifests found"
    echo -e "${GREEN}✓${NC} Ready for Kubernetes migration"
else
    echo -e "${YELLOW}⚠${NC} K3s deployment manifests not found"
    echo "  Run: kubectl apply -f k3s-deployment.yaml to deploy"
fi

# Performance recommendations
echo -e "\n${BLUE}Performance Optimization Tips:${NC}"
echo "=============================="
echo "1. Use Alpine-based images where possible (smaller size)"
echo "2. Implement multi-stage builds to reduce image size"
echo "3. Enable BuildKit for faster builds: export DOCKER_BUILDKIT=1"
echo "4. Use .dockerignore to exclude unnecessary files"
echo "5. Implement caching strategies for dependencies"
echo "6. Consider using distroless images for production"

# Exit with appropriate code
if [[ $OVERALL_SCORE -ge 90 ]]; then
    exit 0
elif [[ $OVERALL_SCORE -ge 70 ]]; then
    exit 1
else
    exit 2
fi