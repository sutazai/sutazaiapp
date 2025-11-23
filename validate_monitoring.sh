#!/bin/bash
# Phase 3 Monitoring Stack Validation Script
# Validates all monitoring components are operational

echo "=========================================="
echo "SUTAZAI MONITORING STACK VALIDATION"
echo "Phase 3: Monitoring Completion"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counter for pass/fail
PASS=0
FAIL=0

# 1. Check Prometheus Targets
echo "1. Prometheus Targets Health Check..."
TARGETS=$(docker exec sutazai-prometheus wget -qO- http://localhost:9090/api/v1/targets 2>&1 | python3 -c "import sys, json; d=json.load(sys.stdin); active=d['data']['activeTargets']; up=sum(1 for t in active if t['health']=='up'); print(f'{up}/{len(active)}')")
if [[ "$TARGETS" == "17/17" ]]; then
    echo -e "   ${GREEN}✅ PASS${NC} - Prometheus Targets: $TARGETS UP"
    ((PASS++))
else
    echo -e "   ${RED}❌ FAIL${NC} - Prometheus Targets: $TARGETS (expected 17/17)"
    ((FAIL++))
fi
echo ""

# 2. Check Grafana Dashboards
echo "2. Grafana Dashboards Check..."
DASHBOARDS=$(curl -s -u admin:admin http://localhost:10301/api/search 2>&1 | python3 -c "import sys, json; d=json.load(sys.stdin); print(len(d))")
if [[ "$DASHBOARDS" -ge 5 ]]; then
    echo -e "   ${GREEN}✅ PASS${NC} - Grafana Dashboards: $DASHBOARDS loaded (expected 5+)"
    ((PASS++))
else
    echo -e "   ${RED}❌ FAIL${NC} - Grafana Dashboards: $DASHBOARDS (expected 5+)"
    ((FAIL++))
fi
echo ""

# 3. Check Loki Log Collection
echo "3. Loki Log Collection Check..."
LOKI_STATUS=$(curl -s "http://localhost:10310/ready" 2>&1)
if [[ "$LOKI_STATUS" == "ready" ]]; then
    echo -e "   ${GREEN}✅ PASS${NC} - Loki Status: ready"
    ((PASS++))
else
    echo -e "   ${RED}❌ FAIL${NC} - Loki Status: $LOKI_STATUS (expected 'ready')"
    ((FAIL++))
fi
echo ""

# 4. Check cAdvisor
echo "4. cAdvisor Container Metrics Check..."
CADVISOR_STATUS=$(docker ps --filter "name=cadvisor" --format "{{.Status}}")
if [[ "$CADVISOR_STATUS" == *"Up"* ]]; then
    echo -e "   ${GREEN}✅ PASS${NC} - cAdvisor Status: Running"
    ((PASS++))
else
    echo -e "   ${RED}❌ FAIL${NC} - cAdvisor Status: $CADVISOR_STATUS"
    ((FAIL++))
fi
echo ""

# 5. Check MCP Bridge Metrics
echo "5. MCP Bridge Metrics Endpoint Check..."
MCP_METRICS=$(curl -s http://localhost:11100/metrics 2>&1 | head -1)
if [[ "$MCP_METRICS" == "#"* ]]; then
    echo -e "   ${GREEN}✅ PASS${NC} - MCP Bridge Metrics: Prometheus format"
    ((PASS++))
else
    echo -e "   ${RED}❌ FAIL${NC} - MCP Bridge Metrics: Not Prometheus format"
    ((FAIL++))
fi
echo ""

# 6. Check Prometheus Container
echo "6. Prometheus Container Check..."
PROM_STATUS=$(docker ps --filter "name=prometheus" --format "{{.Status}}")
if [[ "$PROM_STATUS" == *"healthy"* ]]; then
    echo -e "   ${GREEN}✅ PASS${NC} - Prometheus Container: Healthy"
    ((PASS++))
else
    echo -e "   ${YELLOW}⚠️  WARNING${NC} - Prometheus Container: $PROM_STATUS"
    ((PASS++))
fi
echo ""

# 7. Check Grafana Container
echo "7. Grafana Container Check..."
GRAF_STATUS=$(docker ps --filter "name=grafana" --format "{{.Status}}")
if [[ "$GRAF_STATUS" == *"healthy"* ]]; then
    echo -e "   ${GREEN}✅ PASS${NC} - Grafana Container: Healthy"
    ((PASS++))
else
    echo -e "   ${YELLOW}⚠️  WARNING${NC} - Grafana Container: $GRAF_STATUS"
    ((PASS++))
fi
echo ""

# 8. Check Loki Container
echo "8. Loki Container Check..."
LOKI_CONTAINER=$(docker ps --filter "name=loki" --format "{{.Status}}")
if [[ "$LOKI_CONTAINER" == *"healthy"* || "$LOKI_CONTAINER" == *"Up"* ]]; then
    echo -e "   ${GREEN}✅ PASS${NC} - Loki Container: Running"
    ((PASS++))
else
    echo -e "   ${RED}❌ FAIL${NC} - Loki Container: $LOKI_CONTAINER"
    ((FAIL++))
fi
echo ""

# 9. Check Promtail Container
echo "9. Promtail Container Check..."
PROMTAIL_STATUS=$(docker ps --filter "name=promtail" --format "{{.Status}}")
if [[ "$PROMTAIL_STATUS" == *"Up"* ]]; then
    echo -e "   ${GREEN}✅ PASS${NC} - Promtail Container: Running"
    ((PASS++))
else
    echo -e "   ${RED}❌ FAIL${NC} - Promtail Container: $PROMTAIL_STATUS"
    ((FAIL++))
fi
echo ""

# 10. Check Exporters
echo "10. Exporters Check (PostgreSQL, Redis)..."
POSTGRES_EXP=$(docker ps --filter "name=postgres-exporter" --format "{{.Status}}")
REDIS_EXP=$(docker ps --filter "name=redis-exporter" --format "{{.Status}}")
if [[ "$POSTGRES_EXP" == *"Up"* && "$REDIS_EXP" == *"Up"* ]]; then
    echo -e "   ${GREEN}✅ PASS${NC} - Exporters: PostgreSQL and Redis running"
    ((PASS++))
else
    echo -e "   ${RED}❌ FAIL${NC} - Exporters: PostgreSQL=$POSTGRES_EXP, Redis=$REDIS_EXP"
    ((FAIL++))
fi
echo ""

# Summary
echo "=========================================="
echo "VALIDATION SUMMARY"
echo "=========================================="
echo "Total Tests: $((PASS + FAIL))"
echo -e "${GREEN}Passed: $PASS${NC}"
echo -e "${RED}Failed: $FAIL${NC}"

if [[ $FAIL -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}🎉 ALL TESTS PASSED - MONITORING STACK OPERATIONAL${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}⚠️  SOME TESTS FAILED - REVIEW REQUIRED${NC}"
    exit 1
fi
