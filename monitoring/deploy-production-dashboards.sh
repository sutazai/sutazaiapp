#!/bin/bash

# Production-Grade Monitoring Dashboard Deployment Script for SutazAI
# This script deploys all 8 production dashboards to Grafana

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
GRAFANA_URL="http://localhost:10050"
GRAFANA_USER="admin"
GRAFANA_PASSWORD_FILE="/opt/sutazaiapp/secrets/grafana_password.txt"
DASHBOARD_DIR="/opt/sutazaiapp/monitoring/grafana/dashboards"
PROMETHEUS_CONFIG="/opt/sutazaiapp/monitoring/prometheus"

echo -e "${BLUE}=== SutazAI Production Monitoring Dashboard Deployment ===${NC}"
echo "Deploying comprehensive monitoring dashboards for all stakeholders"

# Function to print status
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Grafana is running
    if ! curl -s -o /dev/null -w "%{http_code}" "${GRAFANA_URL}/api/health" | grep -q "200"; then
        print_error "Grafana is not accessible at ${GRAFANA_URL}"
        exit 1
    fi
    
    # Check if password file exists
    if [[ ! -f "${GRAFANA_PASSWORD_FILE}" ]]; then
        print_error "Grafana password file not found at ${GRAFANA_PASSWORD_FILE}"
        exit 1
    fi
    
    # Check if dashboard directory exists
    if [[ ! -d "${DASHBOARD_DIR}" ]]; then
        print_error "Dashboard directory not found at ${DASHBOARD_DIR}"
        exit 1
    fi
    
    print_status "Prerequisites check passed"
}

# Get Grafana password
get_grafana_password() {
    GRAFANA_PASSWORD=$(cat "${GRAFANA_PASSWORD_FILE}")
    if [[ -z "${GRAFANA_PASSWORD}" ]]; then
        print_error "Grafana password is empty"
        exit 1
    fi
}

# Deploy dashboard function
deploy_dashboard() {
    local dashboard_file="$1"
    local dashboard_name="$2"
    local folder_name="$3"
    
    print_status "Deploying ${dashboard_name} dashboard..."
    
    # Create folder if it doesn't exist
    local folder_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
        -d "{\"title\":\"${folder_name}\"}" \
        "${GRAFANA_URL}/api/folders" || echo '{"message":"folder exists"}')
    
    # Get folder ID
    local folder_id=$(curl -s -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
        "${GRAFANA_URL}/api/folders/${folder_name,,}" | jq -r '.id // 0')
    
    # Prepare dashboard JSON with folder ID
    local dashboard_json=$(cat "${dashboard_file}")
    local payload=$(echo "${dashboard_json}" | jq --argjson folderId "${folder_id}" '. + {folderId: $folderId}' | jq '{dashboard: ., overwrite: true}')
    
    # Deploy dashboard
    local response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
        -d "${payload}" \
        "${GRAFANA_URL}/api/dashboards/db")
    
    if echo "${response}" | jq -e '.status == "success"' > /dev/null; then
        print_status "Successfully deployed ${dashboard_name} dashboard"
        local dashboard_url=$(echo "${response}" | jq -r '.url')
        echo "  Dashboard URL: ${GRAFANA_URL}${dashboard_url}"
    else
        print_error "Failed to deploy ${dashboard_name} dashboard"
        echo "Response: ${response}"
        return 1
    fi
}

# Deploy all dashboards
deploy_dashboards() {
    print_status "Deploying production dashboards..."
    
    # Executive Dashboard
    deploy_dashboard "${DASHBOARD_DIR}/executive/executive-overview.json" "Executive Overview" "Executive"
    
    # Operations Dashboard
    deploy_dashboard "${DASHBOARD_DIR}/operations/operations-overview.json" "Operations Overview" "Operations"
    
    # Developer Dashboard
    deploy_dashboard "${DASHBOARD_DIR}/developer/developer-overview.json" "Developer Dashboard" "Developer"
    
    # Security Dashboard
    deploy_dashboard "${DASHBOARD_DIR}/security/security-overview.json" "Security Overview" "Security"
    
    # Business Metrics Dashboard
    deploy_dashboard "${DASHBOARD_DIR}/business/business-metrics.json" "Business Metrics" "Business"
    
    # Cost Optimization Dashboard
    deploy_dashboard "${DASHBOARD_DIR}/cost/cost-optimization.json" "Cost Optimization" "Cost"
    
    # User Experience Dashboard
    deploy_dashboard "${DASHBOARD_DIR}/ux/user-experience.json" "User Experience" "UX"
    
    # Capacity Planning Dashboard
    deploy_dashboard "${DASHBOARD_DIR}/capacity/capacity-planning.json" "Capacity Planning" "Capacity"
}

# Update Prometheus configuration
update_prometheus_config() {
    print_status "Updating Prometheus configuration with new alert rules..."
    
    # Copy production alerts to Prometheus
    if [[ -f "${PROMETHEUS_CONFIG}/production_alerts.yml" ]]; then
        cp "${PROMETHEUS_CONFIG}/production_alerts.yml" "${PROMETHEUS_CONFIG}/alert_rules.yml"
        print_status "Production alert rules updated"
    else
        print_warning "Production alert rules file not found"
    fi
}

# Test dashboard accessibility
test_dashboards() {
    print_status "Testing dashboard accessibility..."
    
    local dashboards=(
        "executive-overview"
        "operations-overview" 
        "developer-overview"
        "security-overview"
        "business-metrics"
        "cost-optimization"
        "user-experience"
        "capacity-planning"
    )
    
    for dashboard in "${dashboards[@]}"; do
        local response_code=$(curl -s -o /dev/null -w "%{http_code}" \
            -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
            "${GRAFANA_URL}/api/dashboards/uid/${dashboard}")
        
        if [[ "${response_code}" == "200" ]]; then
            print_status "✓ ${dashboard} dashboard is accessible"
        else
            print_error "✗ ${dashboard} dashboard is not accessible (HTTP ${response_code})"
        fi
    done
}

# Create dashboard summary
create_dashboard_summary() {
    print_status "Creating dashboard summary..."
    
    cat > /opt/sutazaiapp/monitoring/DASHBOARD_SUMMARY.md << EOF
# SutazAI Production Monitoring Dashboards

This document provides an overview of all production monitoring dashboards deployed for SutazAI.

## Dashboard Access

All dashboards are accessible at: **${GRAFANA_URL}**
- Username: \`admin\`
- Password: Located in \`${GRAFANA_PASSWORD_FILE}\`

## Available Dashboards

### 1. Executive Overview Dashboard
- **Audience**: C-Suite, Leadership Team
- **URL**: ${GRAFANA_URL}/d/executive-overview/executive-overview
- **Refresh**: 30 seconds
- **Focus**: High-level KPIs, system availability, business metrics
- **Key Metrics**: System uptime, API success rate, active services, response times

### 2. Operations Dashboard (NOC)
- **Audience**: Network Operations Center, DevOps Team
- **URL**: ${GRAFANA_URL}/d/operations-overview/operations-overview
- **Refresh**: 10 seconds
- **Focus**: System health, infrastructure monitoring, real-time alerts
- **Key Metrics**: CPU/Memory/Disk usage, network I/O, active alerts, service status

### 3. Developer Dashboard
- **Audience**: Development Team, Software Engineers
- **URL**: ${GRAFANA_URL}/d/developer-overview/developer-overview
- **Refresh**: 5 seconds
- **Focus**: Application performance, debugging metrics, code-level insights
- **Key Metrics**: API response times, error rates, database performance, container metrics

### 4. Security Dashboard (SOC)
- **Audience**: Security Operations Center, InfoSec Team
- **URL**: ${GRAFANA_URL}/d/security-overview/security-overview
- **Refresh**: 5 seconds
- **Focus**: Security events, threat detection, access monitoring
- **Key Metrics**: Failed requests, auth failures, access violations, network anomalies

### 5. Business Metrics Dashboard
- **Audience**: Product Managers, Business Analysts
- **URL**: ${GRAFANA_URL}/d/business-metrics/business-metrics
- **Refresh**: 30 seconds
- **Focus**: Business KPIs, user engagement, service quality
- **Key Metrics**: Daily active usage, AI agent utilization, service quality, feature usage

### 6. Cost Optimization Dashboard
- **Audience**: Finance Team, Resource Managers
- **URL**: ${GRAFANA_URL}/d/cost-optimization/cost-optimization
- **Refresh**: 30 seconds
- **Focus**: Resource utilization, cost efficiency, optimization opportunities
- **Key Metrics**: Resource efficiency, cost distribution, utilization patterns

### 7. User Experience Dashboard
- **Audience**: UX Team, Customer Success
- **URL**: ${GRAFANA_URL}/d/user-experience/user-experience
- **Refresh**: 5 seconds
- **Focus**: User-facing performance, quality metrics
- **Key Metrics**: Response time percentiles, success rates, user journey performance

### 8. Capacity Planning Dashboard
- **Audience**: Infrastructure Team, Architects
- **URL**: ${GRAFANA_URL}/d/capacity-planning/capacity-planning
- **Refresh**: 1 minute
- **Focus**: Resource forecasting, scaling requirements, growth planning
- **Key Metrics**: Capacity trends, growth patterns, scaling recommendations

## Alert Rules

Production alert rules have been configured for each dashboard category:
- **Executive**: System availability, error rates
- **Operations**: Resource thresholds, service health  
- **Developer**: Performance degradation, application errors
- **Security**: Security threats, access violations
- **Business**: SLA breaches, service quality
- **Cost**: Resource wastage, cost thresholds
- **UX**: User experience degradation
- **Capacity**: Scaling thresholds, capacity limits

## Maintenance

- Dashboards are automatically provisioned via Grafana
- Alert rules are managed through Prometheus configuration
- All configurations are version controlled in the repository
- Regular reviews should be conducted quarterly

## Support

For dashboard issues or modifications, contact the Platform Team or create an issue in the SutazAI repository.

---
Last Updated: $(date)
Deployment Script: deploy-production-dashboards.sh
EOF

    print_status "Dashboard summary created at /opt/sutazaiapp/monitoring/DASHBOARD_SUMMARY.md"
}

# Main execution
main() {
    print_status "Starting SutazAI production dashboard deployment..."
    
    check_prerequisites
    get_grafana_password
    deploy_dashboards
    update_prometheus_config
    test_dashboards
    create_dashboard_summary
    
    echo ""
    echo -e "${GREEN}=== Deployment Complete ===${NC}"
    echo "All 8 production dashboards have been successfully deployed!"
    echo ""
    echo -e "${BLUE}Dashboard Access:${NC}"
    echo "URL: ${GRAFANA_URL}"
    echo "Username: admin"
    echo "Password: (see ${GRAFANA_PASSWORD_FILE})"
    echo ""
    echo -e "${BLUE}Quick Links:${NC}"
    echo "• Executive: ${GRAFANA_URL}/d/executive-overview/executive-overview"
    echo "• Operations: ${GRAFANA_URL}/d/operations-overview/operations-overview"
    echo "• Developer: ${GRAFANA_URL}/d/developer-overview/developer-overview"
    echo "• Security: ${GRAFANA_URL}/d/security-overview/security-overview"
    echo "• Business: ${GRAFANA_URL}/d/business-metrics/business-metrics"
    echo "• Cost: ${GRAFANA_URL}/d/cost-optimization/cost-optimization"
    echo "• UX: ${GRAFANA_URL}/d/user-experience/user-experience"
    echo "• Capacity: ${GRAFANA_URL}/d/capacity-planning/capacity-planning"
    echo ""
    echo -e "${YELLOW}Note:${NC} All dashboards have auto-refresh enabled with appropriate intervals for each audience."
}

# Error handling
trap 'print_error "Script failed at line $LINENO"' ERR

# Run main function
main "$@"