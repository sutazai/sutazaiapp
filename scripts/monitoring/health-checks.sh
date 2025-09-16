#!/bin/bash
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Script: health-checks.sh
# Purpose: Comprehensive system health verification for Sutazai infrastructure
# Author: Sutazai System
# Date: 2025-09-03
# Version: 1.0.0
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Usage:
#   ./health-checks.sh [options]
#
# Options:
#   -h, --help          Show this help message
#   -v, --verbose       Enable verbose output
#   -s, --service SVC   Check specific service only
#   -j, --json          Output results as JSON
#   -a, --alert         Send alerts for failures
#   -r, --repair        Attempt to repair unhealthy services
#
# Requirements:
#   - Docker 20.10+
#   - curl
#   - jq (for JSON output)
#
# Examples:
#   ./health-checks.sh --verbose
#   ./health-checks.sh --service backend --repair
#   ./health-checks.sh --json > health-report.json
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -euo pipefail
IFS=$'\n\t'

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly LOG_FILE="${PROJECT_ROOT}/logs/health-checks_${TIMESTAMP}.log"
readonly REPORT_FILE="${PROJECT_ROOT}/reports/health_${TIMESTAMP}.json"

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

# Options
VERBOSE=false
JSON_OUTPUT=false
SEND_ALERTS=false
REPAIR_MODE=false
SPECIFIC_SERVICE=""

# Health status tracking
declare -A SERVICE_STATUS
declare -A SERVICE_DETAILS
TOTAL_SERVICES=0
HEALTHY_SERVICES=0
UNHEALTHY_SERVICES=0

# Service configurations
declare -A SERVICES=(
    ["backend"]="http://localhost:10200/health"
    ["frontend"]="http://localhost:11000/_stcore/health"
    ["postgres"]="pg_isready -h localhost -p 10000"
    ["redis"]="redis-cli -h localhost -p 10001 ping"
    ["neo4j"]="http://localhost:10002"
    ["rabbitmq"]="http://localhost:10005/api/health/checks/alarms"
    ["chromadb"]="http://localhost:10100/api/v1/heartbeat"
    ["qdrant"]="http://localhost:10101/health"
    ["mcp-bridge"]="http://localhost:11100/health"
)

# Logging
log() {
    local level="$1"
    local message="$2"
    local color="${3:-$NC}"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    
    echo "[${timestamp}] [${level}] ${message}" >> "${LOG_FILE}"
    
    if [[ "$VERBOSE" == true ]] || [[ "$level" == "ERROR" ]] || [[ "$level" == "CRITICAL" ]]; then
        echo -e "${color}[${level}]${NC} ${message}"
    fi
}

# Check Docker service
check_docker_service() {
    local service="$1"
    local container_name="sutazai-${service}"
    
    if docker ps --format "{{.Names}}" | grep -q "^${container_name}$"; then
        local status=$(docker inspect -f '{{.State.Health.Status}}' "${container_name}" 2>/dev/null || echo "unknown")
        
        case "$status" in
            healthy)
                return 0
                ;;
            starting)
                log "WARN" "${service}: Docker health check is starting" "${YELLOW}"
                return 1
                ;;
            unhealthy|unknown)
                log "ERROR" "${service}: Docker health check failed (${status})" "${RED}"
                return 1
                ;;
        esac
    else
        log "ERROR" "${service}: Container not running" "${RED}"
        return 1
    fi
}

# Check HTTP endpoint
check_http_endpoint() {
    local service="$1"
    local endpoint="$2"
    local timeout=5
    
    if curl -sf -o /dev/null -w "%{http_code}" --max-time ${timeout} "${endpoint}" > /dev/null; then
        return 0
    else
        log "ERROR" "${service}: HTTP endpoint check failed" "${RED}"
        return 1
    fi
}

# Check PostgreSQL
check_postgres() {
    if docker exec sutazai-postgres pg_isready -h localhost -p 5432 &>/dev/null; then
        return 0
    else
        log "ERROR" "PostgreSQL: Connection check failed" "${RED}"
        return 1
    fi
}

# Check Redis
check_redis() {
    if docker exec sutazai-redis redis-cli ping &>/dev/null; then
        return 0
    else
        log "ERROR" "Redis: Connection check failed" "${RED}"
        return 1
    fi
}

# Check service health
check_service_health() {
    local service="$1"
    local check_command="${SERVICES[$service]:-}"
    
    log "INFO" "Checking ${service}..." "${BLUE}"
    
    local status="unhealthy"
    local details=""
    local start_time=$(date +%s%N)
    
    # Perform health check based on service type
    case "$service" in
        postgres)
            if check_postgres; then
                status="healthy"
                details="PostgreSQL is accepting connections"
            else
                details="PostgreSQL connection failed"
            fi
            ;;
        redis)
            if check_redis; then
                status="healthy"
                details="Redis is responding to PING"
            else
                details="Redis connection failed"
            fi
            ;;
        backend|frontend|neo4j|rabbitmq|chromadb|qdrant|mcp-bridge)
            if [[ "$check_command" =~ ^http ]]; then
                if check_http_endpoint "$service" "$check_command"; then
                    status="healthy"
                    details="HTTP endpoint responsive"
                else
                    details="HTTP endpoint not responsive"
                fi
            fi
            ;;
        *)
            # Check Docker container status
            if check_docker_service "$service"; then
                status="healthy"
                details="Docker container healthy"
            else
                details="Docker container unhealthy or not running"
            fi
            ;;
    esac
    
    local end_time=$(date +%s%N)
    local response_time=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds
    
    # Store results
    SERVICE_STATUS[$service]="$status"
    SERVICE_DETAILS[$service]="$details (${response_time}ms)"
    
    # Update counters
    ((TOTAL_SERVICES++))
    
    if [[ "$status" == "healthy" ]]; then
        ((HEALTHY_SERVICES++))
        log "INFO" "${service}: ✓ Healthy" "${GREEN}"
    else
        ((UNHEALTHY_SERVICES++))
        log "ERROR" "${service}: ✗ Unhealthy - ${details}" "${RED}"
        
        # Attempt repair if requested
        if [[ "$REPAIR_MODE" == true ]]; then
            repair_service "$service"
        fi
    fi
}

# Repair unhealthy service
repair_service() {
    local service="$1"
    
    log "INFO" "Attempting to repair ${service}..." "${YELLOW}"
    
    case "$service" in
        backend|frontend)
            docker restart "sutazai-${service}" &>/dev/null
            sleep 5
            if check_service_health "$service"; then
                log "INFO" "${service}: Successfully repaired" "${GREEN}"
            else
                log "ERROR" "${service}: Repair failed" "${RED}"
            fi
            ;;
        postgres|redis|neo4j)
            docker restart "sutazai-${service}" &>/dev/null
            sleep 10
            ;;
        *)
            log "WARN" "${service}: No repair procedure available" "${YELLOW}"
            ;;
    esac
}

# Generate JSON report
generate_json_report() {
    local report="{
        \"timestamp\": \"$(date -Iseconds)\",
        \"summary\": {
            \"total\": ${TOTAL_SERVICES},
            \"healthy\": ${HEALTHY_SERVICES},
            \"unhealthy\": ${UNHEALTHY_SERVICES},
            \"health_percentage\": $(( HEALTHY_SERVICES * 100 / TOTAL_SERVICES ))
        },
        \"services\": {"
    
    local first=true
    for service in "${!SERVICE_STATUS[@]}"; do
        if [[ "$first" == false ]]; then
            report+=","
        fi
        report+="
            \"${service}\": {
                \"status\": \"${SERVICE_STATUS[$service]}\",
                \"details\": \"${SERVICE_DETAILS[$service]}\"
            }"
        first=false
    done
    
    report+="
        }
    }"
    
    if [[ "$JSON_OUTPUT" == true ]]; then
        echo "$report" | jq '.'
    else
        mkdir -p "$(dirname "${REPORT_FILE}")"
        echo "$report" | jq '.' > "${REPORT_FILE}"
        log "INFO" "Report saved to: ${REPORT_FILE}" "${GREEN}"
    fi
}

# Send alerts for failures
send_alerts() {
    if [[ "$SEND_ALERTS" != true ]] || [[ $UNHEALTHY_SERVICES -eq 0 ]]; then
        return 0
    fi
    
    log "WARN" "Sending alerts for ${UNHEALTHY_SERVICES} unhealthy services" "${YELLOW}"
    
    # TODO: Implement actual alerting mechanism (email, Slack, etc.)
    # For now, just log critical status
    for service in "${!SERVICE_STATUS[@]}"; do
        if [[ "${SERVICE_STATUS[$service]}" == "unhealthy" ]]; then
            log "CRITICAL" "ALERT: ${service} is unhealthy - ${SERVICE_DETAILS[$service]}" "${RED}"
        fi
    done
}

# Display summary
display_summary() {
    echo
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}                 HEALTH CHECK SUMMARY                   ${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    
    local health_percent=$(( HEALTHY_SERVICES * 100 / TOTAL_SERVICES ))
    local status_color="${GREEN}"
    
    if [[ $health_percent -lt 50 ]]; then
        status_color="${RED}"
    elif [[ $health_percent -lt 80 ]]; then
        status_color="${YELLOW}"
    fi
    
    echo -e "Total Services:    ${TOTAL_SERVICES}"
    echo -e "Healthy Services:  ${GREEN}${HEALTHY_SERVICES}${NC}"
    echo -e "Unhealthy Services: ${RED}${UNHEALTHY_SERVICES}${NC}"
    echo -e "Health Score:      ${status_color}${health_percent}%${NC}"
    echo
    
    # List unhealthy services
    if [[ $UNHEALTHY_SERVICES -gt 0 ]]; then
        echo -e "${RED}Unhealthy Services:${NC}"
        for service in "${!SERVICE_STATUS[@]}"; do
            if [[ "${SERVICE_STATUS[$service]}" == "unhealthy" ]]; then
                echo -e "  - ${service}: ${SERVICE_DETAILS[$service]}"
            fi
        done
        echo
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            head -n 35 "${BASH_SOURCE[0]}" | grep -E '^#( |$)' | sed 's/^#//'
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -s|--service)
            SPECIFIC_SERVICE="${2:-}"
            shift 2
            ;;
        -j|--json)
            JSON_OUTPUT=true
            shift
            ;;
        -a|--alert)
            SEND_ALERTS=true
            shift
            ;;
        -r|--repair)
            REPAIR_MODE=true
            shift
            ;;
        *)
            log "ERROR" "Unknown option: $1" "${RED}"
            exit 2
            ;;
    esac
done

# Main execution
main() {
    mkdir -p "$(dirname "${LOG_FILE}")"
    
    if [[ "$JSON_OUTPUT" != true ]]; then
        log "INFO" "Starting health checks..." "${BLUE}"
    fi
    
    # Check specific service or all services
    if [[ -n "$SPECIFIC_SERVICE" ]]; then
        if [[ -n "${SERVICES[$SPECIFIC_SERVICE]:-}" ]]; then
            check_service_health "$SPECIFIC_SERVICE"
        else
            log "ERROR" "Unknown service: $SPECIFIC_SERVICE" "${RED}"
            exit 1
        fi
    else
        # Check all services
        for service in "${!SERVICES[@]}"; do
            check_service_health "$service"
        done
    fi
    
    # Generate report
    generate_json_report
    
    # Send alerts if needed
    send_alerts
    
    # Display summary (unless JSON output)
    if [[ "$JSON_OUTPUT" != true ]]; then
        display_summary
        
        # Exit with error if any service is unhealthy
        if [[ $UNHEALTHY_SERVICES -gt 0 ]]; then
            exit 1
        fi
    fi
}

main