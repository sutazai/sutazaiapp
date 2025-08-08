#!/bin/bash

# Perfect JARVIS Post-Go-Live Monitoring Script
# Implements comprehensive monitoring following Prompt 7.4.1
# Tracks system health, performance metrics, and user activity

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/monitoring"
METRICS_DIR="${PROJECT_ROOT}/metrics"
ALERT_CONFIG="${PROJECT_ROOT}/config/monitoring/alerts.yaml"
REPORT_DIR="${PROJECT_ROOT}/reports/post-golive"

# Create directories
mkdir -p "$LOG_DIR" "$METRICS_DIR" "$REPORT_DIR"

# Logging
LOG_FILE="${LOG_DIR}/post-golive-$(date +%Y%m%d).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Timestamp function
timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# Log function
log() {
    echo "[$(timestamp)] $1"
}

# Check function with colored output
check() {
    local name=$1
    local command=$2
    local expected=$3
    
    echo -n "Checking $name... "
    if eval "$command" 2>/dev/null | grep -q "$expected"; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${RED}✗${NC}"
        return 1
    fi
}

# =============================================================================
# System Health Monitoring
# =============================================================================

monitor_system_health() {
    log "Starting system health monitoring..."
    
    local health_report="${REPORT_DIR}/health-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "$health_report" <<EOF
{
  "timestamp": "$(timestamp)",
  "system": {
    "uptime": "$(uptime -p)",
    "load_average": $(uptime | awk -F'load average:' '{print $2}' | sed 's/,/,/g' | awk '{printf "[%s, %s, %s]", $1, $2, $3}'),
    "cpu_usage": $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1),
    "memory": {
      "total_mb": $(free -m | awk 'NR==2{print $2}'),
      "used_mb": $(free -m | awk 'NR==2{print $3}'),
      "free_mb": $(free -m | awk 'NR==2{print $4}'),
      "usage_percent": $(free | awk 'NR==2{printf "%.2f", $3/$2*100}')
    },
    "disk": {
      "total_gb": $(df -h / | awk 'NR==2{print $2}' | sed 's/G//'),
      "used_gb": $(df -h / | awk 'NR==2{print $3}' | sed 's/G//'),
      "available_gb": $(df -h / | awk 'NR==2{print $4}' | sed 's/G//'),
      "usage_percent": $(df / | awk 'NR==2{print $5}' | sed 's/%//')
    }
  },
  "containers": {
    "total": $(docker ps -a | tail -n +2 | wc -l),
    "running": $(docker ps | tail -n +2 | wc -l),
    "stopped": $(docker ps -a --filter "status=exited" | tail -n +2 | wc -l),
    "unhealthy": $(docker ps --filter "health=unhealthy" | tail -n +2 | wc -l)
  }
}
EOF
    
    log "Health report saved to: $health_report"
}

# =============================================================================
# Service Availability Monitoring
# =============================================================================

monitor_services() {
    log "Monitoring service availability..."
    
    local services=(
        "backend:10010:/health"
        "frontend:10011:/"
        "ollama:10104:/api/tags"
        "postgres:10000:"
        "redis:10001:"
        "neo4j:10002:"
        "prometheus:10200:/metrics"
        "grafana:10201:/api/health"
    )
    
    local status_report="${REPORT_DIR}/services-$(date +%Y%m%d-%H%M%S).json"
    echo "{" > "$status_report"
    echo '  "timestamp": "'$(timestamp)'",' >> "$status_report"
    echo '  "services": {' >> "$status_report"
    
    local first=true
    for service_info in "${services[@]}"; do
        IFS=':' read -r name port endpoint <<< "$service_info"
        
        if [ "$first" = false ]; then
            echo "," >> "$status_report"
        fi
        first=false
        
        echo -n "    \"$name\": {" >> "$status_report"
        echo -n "\"port\": $port, " >> "$status_report"
        
        if [ -n "$endpoint" ]; then
            response_time=$(curl -o /dev/null -s -w '%{time_total}' "http://localhost:${port}${endpoint}" 2>/dev/null || echo "null")
            http_code=$(curl -o /dev/null -s -w '%{http_code}' "http://localhost:${port}${endpoint}" 2>/dev/null || echo "0")
            
            if [ "$http_code" = "200" ] || [ "$http_code" = "204" ]; then
                echo -n "\"status\": \"healthy\", " >> "$status_report"
                echo -n "\"response_time_ms\": $(echo "$response_time * 1000" | bc), " >> "$status_report"
            else
                echo -n "\"status\": \"unhealthy\", " >> "$status_report"
                echo -n "\"response_time_ms\": null, " >> "$status_report"
            fi
            echo -n "\"http_code\": $http_code" >> "$status_report"
        else
            # For services without HTTP endpoints, check if port is listening
            if nc -z localhost "$port" 2>/dev/null; then
                echo -n "\"status\": \"healthy\", \"response_time_ms\": null, \"http_code\": null" >> "$status_report"
            else
                echo -n "\"status\": \"unhealthy\", \"response_time_ms\": null, \"http_code\": null" >> "$status_report"
            fi
        fi
        echo -n "}" >> "$status_report"
    done
    
    echo "" >> "$status_report"
    echo '  }' >> "$status_report"
    echo '}' >> "$status_report"
    
    log "Service status report saved to: $status_report"
}

# =============================================================================
# Performance Metrics Collection
# =============================================================================

collect_performance_metrics() {
    log "Collecting performance metrics..."
    
    local metrics_file="${METRICS_DIR}/performance-$(date +%Y%m%d-%H%M%S).json"
    
    # Query Prometheus for key metrics
    local prom_url="http://localhost:10200"
    
    # Request rate
    request_rate=$(curl -s "${prom_url}/api/v1/query?query=sum(rate(jarvis_requests_total[5m]))" | \
        jq -r '.data.result[0].value[1] // "0"' 2>/dev/null || echo "0")
    
    # Error rate
    error_rate=$(curl -s "${prom_url}/api/v1/query?query=sum(rate(jarvis_errors_total[5m]))" | \
        jq -r '.data.result[0].value[1] // "0"' 2>/dev/null || echo "0")
    
    # P95 latency
    p95_latency=$(curl -s "${prom_url}/api/v1/query?query=histogram_quantile(0.95,sum(rate(jarvis_latency_seconds_bucket[5m]))by(le))" | \
        jq -r '.data.result[0].value[1] // "0"' 2>/dev/null || echo "0")
    
    cat > "$metrics_file" <<EOF
{
  "timestamp": "$(timestamp)",
  "performance": {
    "request_rate_per_second": $request_rate,
    "error_rate_per_second": $error_rate,
    "error_percentage": $(echo "scale=2; if ($request_rate > 0) $error_rate / $request_rate * 100 else 0" | bc),
    "p95_latency_seconds": $p95_latency,
    "active_connections": $(netstat -an | grep ESTABLISHED | wc -l),
    "database_connections": $(docker exec sutazai-postgres psql -U sutazai -t -c "SELECT count(*) FROM pg_stat_activity;" 2>/dev/null || echo "0")
  },
  "resource_usage": {
    "containers": $(docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | tail -n +2 | wc -l),
    "total_cpu_percent": $(docker stats --no-stream --format "{{.CPUPerc}}" | sed 's/%//' | awk '{sum+=$1} END {print sum}'),
    "total_memory_mb": $(docker stats --no-stream --format "{{.MemUsage}}" | awk '{print $1}' | sed 's/MiB//' | sed 's/GiB/*1024/' | bc | awk '{sum+=$1} END {print sum}')
  }
}
EOF
    
    log "Performance metrics saved to: $metrics_file"
}

# =============================================================================
# Alert Checking
# =============================================================================

check_alerts() {
    log "Checking for alerts..."
    
    local alert_file="${REPORT_DIR}/alerts-$(date +%Y%m%d-%H%M%S).json"
    local has_alerts=false
    
    echo '{"timestamp": "'$(timestamp)'", "alerts": [' > "$alert_file"
    
    # Check high error rate
    error_rate=$(curl -s "http://localhost:10200/api/v1/query?query=sum(rate(jarvis_errors_total[5m]))" | \
        jq -r '.data.result[0].value[1] // "0"' 2>/dev/null || echo "0")
    
    if (( $(echo "$error_rate > 0.05" | bc -l) )); then
        if [ "$has_alerts" = true ]; then echo "," >> "$alert_file"; fi
        echo '  {"type": "high_error_rate", "severity": "warning", "value": '$error_rate', "threshold": 0.05}' >> "$alert_file"
        has_alerts=true
        log "⚠️  ALERT: High error rate detected: $error_rate errors/sec"
    fi
    
    # Check high latency
    p95_latency=$(curl -s "http://localhost:10200/api/v1/query?query=histogram_quantile(0.95,sum(rate(jarvis_latency_seconds_bucket[5m]))by(le))" | \
        jq -r '.data.result[0].value[1] // "0"' 2>/dev/null || echo "0")
    
    if (( $(echo "$p95_latency > 2" | bc -l) )); then
        if [ "$has_alerts" = true ]; then echo "," >> "$alert_file"; fi
        echo '  {"type": "high_latency", "severity": "warning", "value": '$p95_latency', "threshold": 2}' >> "$alert_file"
        has_alerts=true
        log "⚠️  ALERT: High latency detected: ${p95_latency}s (P95)"
    fi
    
    # Check disk usage
    disk_usage=$(df / | awk 'NR==2{print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt 85 ]; then
        if [ "$has_alerts" = true ]; then echo "," >> "$alert_file"; fi
        echo '  {"type": "high_disk_usage", "severity": "critical", "value": '$disk_usage', "threshold": 85}' >> "$alert_file"
        has_alerts=true
        log "🔴 ALERT: High disk usage: ${disk_usage}%"
    fi
    
    # Check memory usage
    mem_usage=$(free | awk 'NR==2{printf "%.0f", $3/$2*100}')
    if [ "$mem_usage" -gt 85 ]; then
        if [ "$has_alerts" = true ]; then echo "," >> "$alert_file"; fi
        echo '  {"type": "high_memory_usage", "severity": "warning", "value": '$mem_usage', "threshold": 85}' >> "$alert_file"
        has_alerts=true
        log "⚠️  ALERT: High memory usage: ${mem_usage}%"
    fi
    
    echo ']}' >> "$alert_file"
    
    if [ "$has_alerts" = false ]; then
        log "✅ No alerts triggered"
    fi
    
    log "Alert report saved to: $alert_file"
}

# =============================================================================
# Generate Daily Report
# =============================================================================

generate_daily_report() {
    log "Generating daily report..."
    
    local report_file="${REPORT_DIR}/daily-report-$(date +%Y%m%d).md"
    
    cat > "$report_file" <<EOF
# JARVIS Post-Go-Live Daily Report
**Date:** $(date '+%Y-%m-%d')
**Generated:** $(timestamp)

## Executive Summary

- **System Status:** $(docker ps --filter "health=healthy" | tail -n +2 | wc -l) healthy services
- **Uptime:** $(uptime -p)
- **Total Requests (24h):** $(curl -s "http://localhost:10200/api/v1/query?query=increase(jarvis_requests_total[24h])" | jq -r '.data.result[0].value[1] // "0"' 2>/dev/null || echo "0")
- **Error Rate:** $(curl -s "http://localhost:10200/api/v1/query?query=sum(rate(jarvis_errors_total[5m]))" | jq -r '.data.result[0].value[1] // "0"' 2>/dev/null || echo "0") errors/sec

## System Health

### Resource Usage
- **CPU Usage:** $(top -bn1 | grep "Cpu(s)" | awk '{print $2}')%
- **Memory Usage:** $(free | awk 'NR==2{printf "%.1f", $3/$2*100}')%
- **Disk Usage:** $(df / | awk 'NR==2{print $5}')

### Container Status
- **Running:** $(docker ps | tail -n +2 | wc -l)
- **Stopped:** $(docker ps -a --filter "status=exited" | tail -n +2 | wc -l)
- **Unhealthy:** $(docker ps --filter "health=unhealthy" | tail -n +2 | wc -l)

## Performance Metrics

### Response Times (P95)
- **Backend API:** $(curl -o /dev/null -s -w '%{time_total}s\n' http://localhost:10010/health 2>/dev/null || echo "N/A")
- **Frontend:** $(curl -o /dev/null -s -w '%{time_total}s\n' http://localhost:10011 2>/dev/null || echo "N/A")

### Database Performance
- **Active Connections:** $(docker exec sutazai-postgres psql -U sutazai -t -c "SELECT count(*) FROM pg_stat_activity;" 2>/dev/null || echo "0")
- **Database Size:** $(docker exec sutazai-postgres psql -U sutazai -t -c "SELECT pg_database_size('sutazai')/1024/1024 || ' MB';" 2>/dev/null || echo "N/A")

## Alerts Summary

$(if [ -f "${REPORT_DIR}/alerts-$(date +%Y%m%d)"*.json ]; then
    echo "Recent alerts:"
    jq -r '.alerts[] | "- \(.type): \(.value) (threshold: \(.threshold))"' "${REPORT_DIR}/alerts-$(date +%Y%m%d)"*.json 2>/dev/null | tail -5
else
    echo "No alerts triggered today"
fi)

## Recommendations

$(if [ "$(df / | awk 'NR==2{print $5}' | sed 's/%//')" -gt 80 ]; then
    echo "- ⚠️ Disk usage is high. Consider cleaning up logs and unused Docker images."
fi)

$(if [ "$(free | awk 'NR==2{printf "%.0f", $3/$2*100}')" -gt 80 ]; then
    echo "- ⚠️ Memory usage is high. Consider restarting non-critical services."
fi)

$(if [ "$(docker ps --filter "health=unhealthy" | tail -n +2 | wc -l)" -gt 0 ]; then
    echo "- 🔴 Unhealthy containers detected. Investigate and restart affected services."
fi)

## Log Files

- System logs: \`${LOG_DIR}\`
- Metrics data: \`${METRICS_DIR}\`
- Reports: \`${REPORT_DIR}\`

---
*This report was automatically generated by the JARVIS monitoring system.*
EOF
    
    log "Daily report generated: $report_file"
    
    # Send notification if configured
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        curl -X POST "$SLACK_WEBHOOK_URL" \
            -H 'Content-Type: application/json' \
            -d '{"text": "JARVIS Daily Report generated: '$(date +%Y-%m-%d)'"}' \
            2>/dev/null || true
    fi
}

# =============================================================================
# Continuous Monitoring Loop
# =============================================================================

continuous_monitoring() {
    local interval=${1:-300}  # Default 5 minutes
    
    log "Starting continuous monitoring (interval: ${interval}s)..."
    
    while true; do
        echo -e "\n${BLUE}[$(timestamp)] Running monitoring checks...${NC}"
        
        monitor_system_health
        monitor_services
        collect_performance_metrics
        check_alerts
        
        # Generate daily report at midnight
        if [ "$(date +%H:%M)" = "00:00" ]; then
            generate_daily_report
        fi
        
        echo -e "${GREEN}[$(timestamp)] Monitoring cycle complete${NC}"
        echo "Next check in ${interval} seconds..."
        sleep "$interval"
    done
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
    echo -e "${BLUE}   JARVIS Post-Go-Live Monitoring System${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════${NC}\n"
    
    # Parse arguments
    case "${1:-}" in
        --once)
            log "Running single monitoring check..."
            monitor_system_health
            monitor_services
            collect_performance_metrics
            check_alerts
            ;;
        --report)
            generate_daily_report
            ;;
        --continuous)
            continuous_monitoring "${2:-300}"
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --once          Run monitoring checks once"
            echo "  --report        Generate daily report"
            echo "  --continuous [interval]  Run continuous monitoring (default: 300s)"
            echo "  --help          Show this help message"
            ;;
        *)
            # Default: run once
            log "Running monitoring checks..."
            monitor_system_health
            monitor_services
            collect_performance_metrics
            check_alerts
            generate_daily_report
            ;;
    esac
    
    echo -e "\n${GREEN}✓ Monitoring complete${NC}"
}

# Run main function
main "$@"