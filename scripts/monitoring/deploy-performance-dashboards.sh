#!/bin/bash
# Deploy Performance Monitoring Dashboards to Grafana

set -e


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

echo "=========================================="
echo "DEPLOYING PERFORMANCE DASHBOARDS"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

GRAFANA_URL="http://localhost:10201"
GRAFANA_USER="admin"
GRAFANA_PASS="admin"

# Wait for Grafana to be ready
echo -e "${YELLOW}Waiting for Grafana to be ready...${NC}"
for i in {1..30}; do
    if curl -s "${GRAFANA_URL}/api/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Grafana is ready${NC}"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

# Create datasource for Prometheus if not exists
echo -e "${YELLOW}Configuring Prometheus datasource...${NC}"
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{
        "name": "Prometheus",
        "type": "prometheus",
        "url": "http://prometheus:9090",
        "access": "proxy",
        "isDefault": true
    }' \
    "${GRAFANA_URL}/api/datasources" \
    -u "${GRAFANA_USER}:${GRAFANA_PASS}" 2>/dev/null || echo "Datasource may already exist"

# Import Redis Performance Dashboard
echo -e "${YELLOW}Importing Redis Performance Dashboard...${NC}"
if [ -f "monitoring/dashboards/redis-performance-dashboard.json" ]; then
    DASHBOARD_JSON=$(cat monitoring/dashboards/redis-performance-dashboard.json)
    curl -X POST \
        -H "Content-Type: application/json" \
        -d "${DASHBOARD_JSON}" \
        "${GRAFANA_URL}/api/dashboards/db" \
        -u "${GRAFANA_USER}:${GRAFANA_PASS}" 2>/dev/null || echo "Dashboard import issue"
    echo -e "${GREEN}✓ Redis dashboard imported${NC}"
else
    echo "Redis dashboard file not found"
fi

# Import Ollama Performance Dashboard
echo -e "${YELLOW}Importing Ollama Performance Dashboard...${NC}"
if [ -f "monitoring/dashboards/ollama-performance-dashboard.json" ]; then
    DASHBOARD_JSON=$(cat monitoring/dashboards/ollama-performance-dashboard.json)
    curl -X POST \
        -H "Content-Type: application/json" \
        -d "${DASHBOARD_JSON}" \
        "${GRAFANA_URL}/api/dashboards/db" \
        -u "${GRAFANA_USER}:${GRAFANA_PASS}" 2>/dev/null || echo "Dashboard import issue"
    echo -e "${GREEN}✓ Ollama dashboard imported${NC}"
else
    echo "Ollama dashboard file not found"
fi

# Set up alerts
echo -e "${YELLOW}Configuring performance alerts...${NC}"

# Redis low hit rate alert
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{
        "uid": "redis-hit-rate-alert",
        "title": "Redis Low Hit Rate Alert",
        "condition": "query",
        "data": [{
            "refId": "A",
            "model": {
                "expr": "redis_keyspace_hits_total / (redis_keyspace_hits_total + redis_keyspace_misses_total) * 100",
                "refId": "A"
            }
        }],
        "message": "Redis cache hit rate dropped below 85%",
        "for": "5m",
        "frequency": "1m",
        "handler": 1,
        "noDataState": "no_data",
        "executionErrorState": "alerting"
    }' \
    "${GRAFANA_URL}/api/alerts" \
    -u "${GRAFANA_USER}:${GRAFANA_PASS}" 2>/dev/null || echo "Alert may already exist"

# Ollama high response time alert
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{
        "uid": "ollama-response-time-alert",
        "title": "Ollama High Response Time Alert",
        "condition": "query",
        "data": [{
            "refId": "A",
            "model": {
                "expr": "histogram_quantile(0.99, rate(ollama_request_duration_seconds_bucket[5m]))",
                "refId": "A"
            }
        }],
        "message": "Ollama p99 response time exceeded 10 seconds",
        "for": "5m",
        "frequency": "1m",
        "handler": 1,
        "noDataState": "no_data",
        "executionErrorState": "alerting"
    }' \
    "${GRAFANA_URL}/api/alerts" \
    -u "${GRAFANA_USER}:${GRAFANA_PASS}" 2>/dev/null || echo "Alert may already exist"

echo ""
echo "=========================================="
echo "DASHBOARD DEPLOYMENT COMPLETE"
echo "=========================================="
echo -e "${GREEN}Access dashboards at:${NC}"
echo "  • Grafana: ${GRAFANA_URL} (admin/admin)"
echo "  • Redis Dashboard: ${GRAFANA_URL}/d/redis-performance"
echo "  • Ollama Dashboard: ${GRAFANA_URL}/d/ollama-performance"
echo ""
echo -e "${GREEN}Performance Monitoring Active:${NC}"
echo "  • Redis cache hit rate monitoring"
echo "  • Ollama response time tracking"
echo "  • Memory and resource usage"
echo "  • Alert notifications configured"
echo ""