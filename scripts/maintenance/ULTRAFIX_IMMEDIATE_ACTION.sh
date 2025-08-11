#!/bin/bash
# ULTRAFIX: Immediate Actions for System Perfection
# Generated: August 11, 2025
# Target: Fix critical issues blocking perfection

set -euo pipefail

echo "================================================"
echo "ULTRAFIX: IMMEDIATE PERFECTION ACTIONS"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

echo ""
echo "Phase 1: Security Fixes"
echo "------------------------"

# 1. Check for root containers
echo "Checking for containers running as root..."
ROOT_CONTAINERS=$(docker ps --format "{{.Names}}" | xargs -I {} sh -c 'echo -n "{}: "; docker exec {} id -u 2>/dev/null || echo "error"' | grep ": 0$" | cut -d: -f1)

if [ ! -z "$ROOT_CONTAINERS" ]; then
    print_warning "Found containers running as root:"
    echo "$ROOT_CONTAINERS"
    echo ""
    echo "To fix, update their Dockerfiles with:"
    echo "USER 1000  # or appropriate non-root user"
else
    print_status "All containers running as non-root users"
fi

# 2. Check for hardcoded credentials
echo ""
echo "Scanning for hardcoded credentials..."
CREDS_FOUND=$(grep -r "password\|secret\|key\|token" /opt/sutazaiapp --include="*.yml" --include="*.yaml" --include="*.json" 2>/dev/null | grep -v "#" | grep -v ".env" | wc -l)

if [ "$CREDS_FOUND" -gt 0 ]; then
    print_warning "Found $CREDS_FOUND potential hardcoded credentials"
    echo "Review: grep -r 'password\|secret\|key\|token' /opt/sutazaiapp --include='*.yml' --include='*.yaml'"
else
    print_status "No hardcoded credentials found"
fi

echo ""
echo "Phase 2: Performance Quick Wins"
echo "--------------------------------"

# 3. Check memory usage
echo "Analyzing memory efficiency..."
TOTAL_MEM=$(docker stats --no-stream --format "{{.MemUsage}}" | awk '{print $1}' | sed 's/MiB//' | sed 's/GiB/*1024/' | bc | awk '{sum+=$1} END {print sum}')
echo "Total memory usage: ${TOTAL_MEM}MB"

if (( $(echo "$TOTAL_MEM > 2000" | bc -l) )); then
    print_warning "Memory usage exceeds 2GB target"
    echo "Recommendation: Right-size container limits in docker-compose.yml"
else
    print_status "Memory usage within targets"
fi

# 4. Check CPU efficiency
echo ""
echo "Analyzing CPU efficiency..."
LOW_CPU=$(docker stats --no-stream --format "{{.Name}}\t{{.CPUPerc}}" | awk '$2 < 0.1' | wc -l)

if [ "$LOW_CPU" -gt 5 ]; then
    print_warning "Found $LOW_CPU containers with <0.1% CPU usage"
    echo "Consider consolidating or removing idle services"
else
    print_status "CPU usage looks reasonable"
fi

echo ""
echo "Phase 3: Quick Health Checks"
echo "-----------------------------"

# 5. Service health status
SERVICES=("10010:Backend" "10011:Frontend" "10104:Ollama" "10200:Prometheus" "10201:Grafana")

for service in "${SERVICES[@]}"; do
    IFS=':' read -r port name <<< "$service"
    if curl -s "http://localhost:$port/health" >/dev/null 2>&1 || curl -s "http://localhost:$port/" >/dev/null 2>&1; then
        print_status "$name is healthy on port $port"
    else
        print_error "$name might have issues on port $port"
    fi
done

echo ""
echo "Phase 4: Code Quality Scan"
echo "---------------------------"

# 6. Count technical debt
TODO_COUNT=$(grep -r "TODO\|FIXME\|HACK\|XXX" /opt/sutazaiapp --include="*.py" 2>/dev/null | wc -l)
print_warning "Found $TODO_COUNT TODO/FIXME comments in Python files"

# 7. Count duplicate requirements files
REQ_COUNT=$(find /opt/sutazaiapp -name "requirements*.txt" -type f | wc -l)
print_warning "Found $REQ_COUNT requirements files (should be 3 max)"

echo ""
echo "Phase 5: Database Optimization Check"
echo "-------------------------------------"

# 8. Check database connections
DB_CONN=$(docker exec sutazai-postgres psql -U sutazai -d sutazai -t -c "SELECT count(*) FROM pg_stat_activity;" 2>/dev/null | tr -d ' ')
echo "Active database connections: $DB_CONN"

if [ "$DB_CONN" -gt 50 ]; then
    print_warning "High number of database connections - implement pooling"
else
    print_status "Database connections within normal range"
fi

echo ""
echo "================================================"
echo "ULTRAFIX SUMMARY"
echo "================================================"

# Calculate perfection score
SCORE=92
ISSUES=0

[ ! -z "$ROOT_CONTAINERS" ] && ((ISSUES++))
[ "$CREDS_FOUND" -gt 0 ] && ((ISSUES++))
[ "$TOTAL_MEM" -gt 2000 ] && ((ISSUES++))
[ "$LOW_CPU" -gt 5 ] && ((ISSUES++))
[ "$TODO_COUNT" -gt 10 ] && ((ISSUES++))
[ "$REQ_COUNT" -gt 3 ] && ((ISSUES++))

FINAL_SCORE=$((100 - ISSUES * 2))

echo ""
echo "Current Perfection Score: $FINAL_SCORE/100"
echo "Issues to Fix: $ISSUES"
echo ""

if [ "$ISSUES" -eq 0 ]; then
    print_status "SYSTEM IS PERFECT! 100/100"
else
    print_warning "Fix the $ISSUES issues above to achieve perfection"
    echo ""
    echo "Next Steps:"
    echo "1. Fix root containers in Dockerfiles"
    echo "2. Move credentials to environment variables"
    echo "3. Right-size container memory limits"
    echo "4. Consolidate idle services"
    echo "5. Clean up technical debt"
    echo "6. Unify requirements files"
fi

echo ""
echo "For detailed analysis, see: ULTRATHINK_SYSTEM_PERFECTION_ANALYSIS.md"
echo "================================================"