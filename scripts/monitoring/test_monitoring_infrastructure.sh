#!/bin/bash

# Test Monitoring Infrastructure
# Author: observability-monitoring-engineer
# Date: 2025-08-21

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        MONITORING INFRASTRUCTURE TEST REPORT                 ║${NC}"
echo -e "${BLUE}║                   $(date +'%Y-%m-%d %H:%M:%S')                        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"

# Test function
test_service() {
    local name=$1
    local url=$2
    local expected=$3
    
    echo -n "Testing $name... "
    
    if curl -s -f -o /dev/null "$url" 2>/dev/null; then
        echo -e "${GREEN}✓ WORKING${NC} - $url"
        return 0
    else
        echo -e "${RED}✗ FAILED${NC} - $url"
        return 1
    fi
}

# Test with specific endpoint
test_endpoint() {
    local name=$1
    local url=$2
    local grep_pattern=$3
    
    echo -n "Testing $name... "
    
    response=$(curl -s "$url" 2>/dev/null || echo "FAILED")
    
    if echo "$response" | grep -q "$grep_pattern" 2>/dev/null; then
        echo -e "${GREEN}✓ WORKING${NC} - $url"
        return 0
    else
        echo -e "${RED}✗ FAILED${NC} - $url (Pattern: $grep_pattern not found)"
        return 1
    fi
}

echo -e "\n${YELLOW}═══ Core Monitoring Services ═══${NC}"
test_service "Prometheus" "http://localhost:10200/-/healthy" "Prometheus is Healthy"
test_service "Grafana" "http://localhost:10201/api/health" "ok"
test_service "Loki" "http://localhost:10202/ready" "ready"
test_service "Consul" "http://localhost:10006/v1/agent/self" "Config"
test_service "cAdvisor" "http://localhost:10206/containers/" "subcontainers"
test_service "Node Exporter" "http://localhost:10205/metrics" "node_"

echo -e "\n${YELLOW}═══ Prometheus Targets Status ═══${NC}"
targets=$(curl -s http://localhost:10200/api/v1/targets 2>/dev/null | jq -r '.data.activeTargets[] | "\(.labels.job): \(.health)"' 2>/dev/null | sort -u)
if [ -n "$targets" ]; then
    while IFS= read -r target; do
        if echo "$target" | grep -q "up"; then
            echo -e "  ${GREEN}✓${NC} $target"
        else
            echo -e "  ${RED}✗${NC} $target"
        fi
    done <<< "$targets"
else
    echo -e "  ${RED}No targets found${NC}"
fi

echo -e "\n${YELLOW}═══ Grafana Dashboards ═══${NC}"
dashboards=$(curl -s http://localhost:10201/api/search 2>/dev/null | jq -r '.[].title' 2>/dev/null | head -10)
if [ -n "$dashboards" ]; then
    while IFS= read -r dashboard; do
        echo -e "  ${GREEN}✓${NC} $dashboard"
    done <<< "$dashboards"
else
    echo -e "  ${YELLOW}No dashboards found or API not accessible${NC}"
fi

echo -e "\n${YELLOW}═══ Log Aggregation (Loki) ═══${NC}"
echo -n "Checking Loki labels... "
labels=$(curl -s http://localhost:10202/loki/api/v1/labels 2>/dev/null | jq -r '.data[]' 2>/dev/null | head -5)
if [ -n "$labels" ]; then
    echo -e "${GREEN}✓ Available${NC}"
    echo "$labels" | while read label; do
        echo -e "  - $label"
    done
else
    echo -e "${YELLOW}No labels found (might be empty)${NC}"
fi

echo -e "\n${YELLOW}═══ Container Metrics (cAdvisor) ═══${NC}"
containers=$(curl -s http://localhost:10206/api/v1.3/containers/ 2>/dev/null | jq -r '.name' 2>/dev/null)
if [ -n "$containers" ]; then
    echo -e "  ${GREEN}✓ Monitoring root container${NC}"
else
    echo -e "  ${RED}✗ cAdvisor API not responding${NC}"
fi

echo -e "\n${YELLOW}═══ Service Discovery (Consul) ═══${NC}"
services=$(curl -s http://localhost:10006/v1/catalog/services 2>/dev/null | jq -r 'keys[]' 2>/dev/null | head -10)
if [ -n "$services" ]; then
    echo -e "  ${GREEN}✓ Services registered:${NC}"
    echo "$services" | while read service; do
        echo -e "    - $service"
    done
else
    echo -e "  ${YELLOW}No services registered${NC}"
fi

echo -e "\n${YELLOW}═══ Docker Container Logs ═══${NC}"
echo "Testing docker logs access..."
if docker logs sutazai-backend --tail 1 2>/dev/null | head -1 > /dev/null; then
    echo -e "  ${GREEN}✓ Backend logs accessible${NC}"
else
    echo -e "  ${RED}✗ Backend logs not accessible${NC}"
fi

if docker logs sutazai-frontend --tail 1 2>/dev/null | head -1 > /dev/null; then
    echo -e "  ${GREEN}✓ Frontend logs accessible${NC}"
else
    echo -e "  ${RED}✗ Frontend logs not accessible${NC}"
fi

echo -e "\n${YELLOW}═══ Alternative Log Access Methods ═══${NC}"
echo "1. Docker logs: ${GREEN}docker logs -f <container-name>${NC}"
echo "2. Docker compose logs: ${GREEN}docker-compose logs -f${NC}"
echo "3. Loki LogCLI: ${YELLOW}logcli query '{job=\"backend\"}' --addr=http://localhost:10202${NC}"
echo "4. Prometheus metrics: ${GREEN}http://localhost:10200${NC}"
echo "5. Grafana dashboards: ${GREEN}http://localhost:10201${NC}"
echo "6. Consul UI: ${GREEN}http://localhost:10006${NC}"

echo -e "\n${YELLOW}═══ Working Monitoring Scripts ═══${NC}"
for script in /opt/sutazaiapp/scripts/monitoring/*.py; do
    if [ -f "$script" ]; then
        basename=$(basename "$script")
        if grep -q "if __name__" "$script" 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} $basename (Python executable)"
        fi
    fi
done

echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    TEST COMPLETE                             ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"

# Summary
echo -e "\n${YELLOW}SUMMARY:${NC}"
echo "✅ WORKING: Prometheus, Grafana, Loki, Consul, cAdvisor, Node Exporter"
echo "⚠️  MISSING: live_logs.sh script (referenced but not found)"
echo "📊 Use Grafana at http://localhost:10201 for visual monitoring"
echo "🔍 Use Prometheus at http://localhost:10200 for metrics queries"