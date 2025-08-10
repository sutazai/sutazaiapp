#!/bin/bash

# ULTRA-PERFORMANCE VALIDATOR
# Quick performance validation script for SutazAI system
# Usage: ./performance-validator.sh

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

echo "========================================"
echo "   SUTAZAI PERFORMANCE VALIDATOR v1.0  "
echo "========================================"
echo "Started: $(date)"
echo

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored status
print_status() {
    if [ "$1" == "PASS" ]; then
        echo -e "${GREEN}✅ $2${NC}"
    elif [ "$1" == "FAIL" ]; then
        echo -e "${RED}❌ $2${NC}"
    else
        echo -e "${YELLOW}⚠️ $2${NC}"
    fi
}

# 1. Check Redis Cache Performance
echo "1. REDIS CACHE PERFORMANCE"
echo "--------------------------"
REDIS_STATS=$(docker exec sutazai-redis redis-cli INFO stats 2>/dev/null | grep -E "keyspace_hits|keyspace_misses" | tr '\r' ' ')
HITS=$(echo "$REDIS_STATS" | grep -o "keyspace_hits:[0-9]*" | cut -d: -f2)
MISSES=$(echo "$REDIS_STATS" | grep -o "keyspace_misses:[0-9]*" | cut -d: -f2)

if [ -n "$HITS" ] && [ -n "$MISSES" ] && [ "$MISSES" -gt 0 ]; then
    HIT_RATE=$(echo "scale=2; $HITS * 100 / ($HITS + $MISSES)" | bc)
    echo "Cache Hit Rate: ${HIT_RATE}% (Target: >85%)"
    
    if (( $(echo "$HIT_RATE > 85" | bc -l) )); then
        print_status "PASS" "Cache hit rate meets target"
    else
        print_status "FAIL" "Cache hit rate below target"
    fi
else
    print_status "WARN" "Unable to calculate cache hit rate"
fi
echo

# 2. Check API Response Times
echo "2. API ENDPOINT PERFORMANCE"
echo "---------------------------"
TOTAL_TIME=0
SUCCESS_COUNT=0

for i in {1..5}; do
    START=$(date +%s%N)
    if curl -s http://localhost:10010/health > /dev/null 2>&1; then
        END=$(date +%s%N)
        DURATION=$((($END - $START) / 1000000))
        TOTAL_TIME=$((TOTAL_TIME + DURATION))
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    fi
done

if [ $SUCCESS_COUNT -gt 0 ]; then
    AVG_TIME=$((TOTAL_TIME / SUCCESS_COUNT))
    echo "Average Response Time: ${AVG_TIME}ms (Target: <100ms)"
    
    if [ $AVG_TIME -lt 100 ]; then
        print_status "PASS" "API response time meets target"
    else
        print_status "FAIL" "API response time exceeds target"
    fi
else
    print_status "FAIL" "API health check failed"
fi
echo

# 3. Check Container Resources
echo "3. CONTAINER RESOURCE USAGE"
echo "---------------------------"
MEMORY_USAGE=$(docker stats --no-stream --format "{{.MemUsage}}" | \
    awk '{split($1,a,"MiB|GiB"); if (index($1,"GiB")) mem+=a[1]*1024; else mem+=a[1]} END {printf "%.2f", mem/1024}')

echo "Total Memory Usage: ${MEMORY_USAGE}GB (Target: <15GB)"

if (( $(echo "$MEMORY_USAGE < 15" | bc -l) )); then
    print_status "PASS" "Memory usage within target"
else
    print_status "FAIL" "Memory usage exceeds target"
fi

# CPU check
AVG_CPU=$(docker stats --no-stream --format "{{.CPUPerc}}" | \
    awk '{gsub("%","",$1); sum+=$1; count++} END {printf "%.2f", sum/count}')

echo "Average CPU Usage: ${AVG_CPU}% (Target: <50%)"

if (( $(echo "$AVG_CPU < 50" | bc -l) )); then
    print_status "PASS" "CPU usage within target"
else
    print_status "FAIL" "CPU usage exceeds target"
fi
echo

# 4. Check Database Performance
echo "4. DATABASE QUERY PERFORMANCE"
echo "-----------------------------"
DB_TIME=$(docker exec sutazai-postgres psql -U sutazai -d sutazai -tAc \
    "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT COUNT(*) FROM users;" 2>/dev/null | \
    jq -r '.[0]."Execution Time"' 2>/dev/null || echo "N/A")

if [ "$DB_TIME" != "N/A" ]; then
    echo "Query Execution Time: ${DB_TIME}ms (Target: <50ms)"
    
    # Extract numeric value
    DB_TIME_NUM=$(echo "$DB_TIME" | grep -o "[0-9.]*" | head -1)
    if (( $(echo "$DB_TIME_NUM < 50" | bc -l) )); then
        print_status "PASS" "Database query time meets target"
    else
        print_status "FAIL" "Database query time exceeds target"
    fi
else
    print_status "WARN" "Unable to measure database performance"
fi
echo

# 5. Service Health Summary
echo "5. SERVICE HEALTH CHECK"
echo "-----------------------"
HEALTHY_COUNT=0
TOTAL_COUNT=0

# Check critical services
SERVICES=("localhost:10010:Backend" "localhost:10104:Ollama" "localhost:10001:Redis" "localhost:10000:PostgreSQL")

for SERVICE in "${SERVICES[@]}"; do
    IFS=':' read -r HOST PORT NAME <<< "$SERVICE"
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    
    if nc -z -w1 $HOST $PORT 2>/dev/null; then
        print_status "PASS" "$NAME is responsive"
        HEALTHY_COUNT=$((HEALTHY_COUNT + 1))
    else
        print_status "FAIL" "$NAME is not responding"
    fi
done

echo
echo "========================================"
echo "         VALIDATION SUMMARY             "
echo "========================================"
echo "Services Healthy: $HEALTHY_COUNT/$TOTAL_COUNT"
echo "Completed: $(date)"
echo

# Exit with appropriate code
if [ $HEALTHY_COUNT -eq $TOTAL_COUNT ]; then
    echo -e "${GREEN}VALIDATION PASSED${NC}"
    exit 0
else
    echo -e "${RED}VALIDATION FAILED${NC}"
    exit 1
fi