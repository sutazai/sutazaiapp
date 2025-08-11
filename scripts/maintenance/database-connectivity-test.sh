#!/bin/bash
# Database Connectivity and Health Test Suite
# Tests all 6 databases for 100% connectivity

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
LOG_FILE="/opt/sutazaiapp/logs/database_connectivity_test_$(date +%Y%m%d_%H%M%S).log"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo -e "[$TIMESTAMP] $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "[$TIMESTAMP] ${GREEN}✓${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "[$TIMESTAMP] ${YELLOW}⚠${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "[$TIMESTAMP] ${RED}✗${NC} $1" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "[$TIMESTAMP] ${BLUE}ℹ${NC} $1" | tee -a "$LOG_FILE"
}

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

test_database() {
    local db_name="$1"
    local test_command="$2"
    local description="$3"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    log_info "Testing $db_name: $description"
    
    if eval "$test_command" >/dev/null 2>&1; then
        log_success "$db_name: $description - PASSED"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        log_error "$db_name: $description - FAILED"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Start testing
log "=================================="
log "Database Connectivity Test Suite"
log "=================================="
log "Starting comprehensive database connectivity tests..."

# Test 1: PostgreSQL
log_info "Testing PostgreSQL (Port 10000)..."
test_database "PostgreSQL" \
    "docker exec sutazai-postgres psql -U sutazai -d sutazai -c 'SELECT 1' -t" \
    "Basic connectivity and query execution"

test_database "PostgreSQL" \
    "docker exec sutazai-postgres psql -U sutazai -d sutazai -c 'SELECT count(*) FROM information_schema.tables' -t" \
    "Database schema accessibility"

test_database "PostgreSQL" \
    "docker exec sutazai-postgres psql -U sutazai -d sutazai -c 'SELECT current_user, current_database()' -t" \
    "User authentication and database selection"

# Test 2: Redis
log_info "Testing Redis (Port 10001)..."
test_database "Redis" \
    "docker exec sutazai-redis redis-cli ping" \
    "Basic connectivity"

test_database "Redis" \
    "docker exec sutazai-redis redis-cli set test_key 'test_value'" \
    "Write operation"

test_database "Redis" \
    "docker exec sutazai-redis redis-cli get test_key" \
    "Read operation"

test_database "Redis" \
    "docker exec sutazai-redis redis-cli del test_key" \
    "Delete operation"

# Test 3: Neo4j
log_info "Testing Neo4j (Port 10002/10003)..."
test_database "Neo4j" \
    "curl -sf http://localhost:10002/db/data/" \
    "HTTP interface connectivity"

NEO4J_PASSWORD=$(grep NEO4J_PASSWORD /opt/sutazaiapp/.env | cut -d'=' -f2)
test_database "Neo4j" \
    "docker exec sutazai-neo4j cypher-shell -u neo4j -p '$NEO4J_PASSWORD' 'RETURN 1 as test' --format plain" \
    "Cypher query execution with authentication"

test_database "Neo4j" \
    "docker exec sutazai-neo4j cypher-shell -u neo4j -p '$NEO4J_PASSWORD' 'MATCH (n) RETURN count(n) as node_count' --format plain" \
    "Graph database query"

# Test 4: Qdrant Vector Database
log_info "Testing Qdrant (Port 10101/10102)..."
test_database "Qdrant" \
    "curl -sf http://localhost:10101/collections" \
    "HTTP API connectivity"

test_database "Qdrant" \
    "curl -sf http://localhost:10101/" \
    "Health endpoint check"

test_database "Qdrant" \
    "curl -sf http://localhost:10101/collections -H 'Content-Type: application/json'" \
    "Collections listing"

# Test 5: ChromaDB Vector Database
log_info "Testing ChromaDB (Port 10100)..."
test_database "ChromaDB" \
    "curl -sf http://localhost:10100/api/v1/heartbeat" \
    "Heartbeat endpoint"

test_database "ChromaDB" \
    "curl -sf http://localhost:10100/api/v1/version" \
    "Version information"

test_database "ChromaDB" \
    "curl -sf http://localhost:10100/api/v1/collections" \
    "Collections API"

# Test 6: FAISS Vector Database
log_info "Testing FAISS (Port 10103)..."
test_database "FAISS" \
    "curl -sf http://localhost:10103/health" \
    "Health endpoint"

test_database "FAISS" \
    "curl -sf http://localhost:10103/" \
    "Root endpoint connectivity"

# Performance Tests
log_info "Running performance tests..."

# PostgreSQL performance test
PG_START_TIME=$(date +%s%N)
if docker exec sutazai-postgres psql -U sutazai -d sutazai -c "SELECT count(*) FROM pg_stat_activity" -t >/dev/null 2>&1; then
    PG_END_TIME=$(date +%s%N)
    PG_RESPONSE_TIME=$(( (PG_END_TIME - PG_START_TIME) / 1000000 )) # Convert to milliseconds
    if [ $PG_RESPONSE_TIME -lt 100 ]; then
        log_success "PostgreSQL response time: ${PG_RESPONSE_TIME}ms (excellent)"
    elif [ $PG_RESPONSE_TIME -lt 500 ]; then
        log_warning "PostgreSQL response time: ${PG_RESPONSE_TIME}ms (acceptable)"
    else
        log_error "PostgreSQL response time: ${PG_RESPONSE_TIME}ms (slow)"
    fi
else
    log_error "PostgreSQL performance test failed"
fi

# Redis performance test
REDIS_START_TIME=$(date +%s%N)
if docker exec sutazai-redis redis-cli ping >/dev/null 2>&1; then
    REDIS_END_TIME=$(date +%s%N)
    REDIS_RESPONSE_TIME=$(( (REDIS_END_TIME - REDIS_START_TIME) / 1000000 ))
    if [ $REDIS_RESPONSE_TIME -lt 10 ]; then
        log_success "Redis response time: ${REDIS_RESPONSE_TIME}ms (excellent)"
    elif [ $REDIS_RESPONSE_TIME -lt 50 ]; then
        log_warning "Redis response time: ${REDIS_RESPONSE_TIME}ms (acceptable)"
    else
        log_error "Redis response time: ${REDIS_RESPONSE_TIME}ms (slow)"
    fi
else
    log_error "Redis performance test failed"
fi

# Connection Pool Tests
log_info "Testing connection pooling..."

# PostgreSQL max connections check
MAX_CONN=$(docker exec sutazai-postgres psql -U sutazai -d sutazai -c "SELECT setting FROM pg_settings WHERE name='max_connections'" -t | tr -d ' ')
CURRENT_CONN=$(docker exec sutazai-postgres psql -U sutazai -d sutazai -c "SELECT count(*) FROM pg_stat_activity" -t | tr -d ' ')
CONN_UTILIZATION=$(( CURRENT_CONN * 100 / MAX_CONN ))

log_info "PostgreSQL connections: $CURRENT_CONN/$MAX_CONN (${CONN_UTILIZATION}%)"

if [ $CONN_UTILIZATION -lt 50 ]; then
    log_success "PostgreSQL connection utilization: ${CONN_UTILIZATION}% (healthy)"
elif [ $CONN_UTILIZATION -lt 80 ]; then
    log_warning "PostgreSQL connection utilization: ${CONN_UTILIZATION}% (monitor)"
else
    log_error "PostgreSQL connection utilization: ${CONN_UTILIZATION}% (critical)"
fi

# Backup Status Check
log_info "Checking backup status..."

check_backup_age() {
    local db_type="$1"
    local backup_dir="/opt/sutazaiapp/backups/$db_type"
    
    if [ ! -d "$backup_dir" ]; then
        log_warning "$db_type: No backup directory found"
        return
    fi
    
    # Find most recent backup
    local most_recent_backup=$(find "$backup_dir" -name "*.gz" -o -name "*.sql" -o -name "*.tar.gz" 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
    
    if [ -z "$most_recent_backup" ]; then
        log_warning "$db_type: No backup files found"
        return
    fi
    
    # Calculate backup age in hours
    local backup_timestamp=$(stat -c %Y "$most_recent_backup" 2>/dev/null || echo 0)
    local current_timestamp=$(date +%s)
    local age_hours=$(( (current_timestamp - backup_timestamp) / 3600 ))
    
    if [ $age_hours -lt 25 ]; then
        log_success "$db_type backup: ${age_hours} hours old (recent)"
    elif [ $age_hours -lt 48 ]; then
        log_warning "$db_type backup: ${age_hours} hours old (aging)"
    else
        log_error "$db_type backup: ${age_hours} hours old (stale)"
    fi
}

# Check backup ages
check_backup_age "postgres"
check_backup_age "redis"  
check_backup_age "neo4j"
check_backup_age "vector-databases"

# Container Health Status
log_info "Checking container health status..."

DATABASE_CONTAINERS=(
    "sutazai-postgres:PostgreSQL"
    "sutazai-redis:Redis"  
    "sutazai-neo4j:Neo4j"
    "sutazai-qdrant:Qdrant"
    "sutazai-chromadb:ChromaDB"
    "sutazai-faiss:FAISS"
)

HEALTHY_CONTAINERS=0
TOTAL_CONTAINERS=${#DATABASE_CONTAINERS[@]}

for container_info in "${DATABASE_CONTAINERS[@]}"; do
    container_name="${container_info%%:*}"
    db_name="${container_info##*:}"
    
    if docker ps --format '{{.Names}} {{.Status}}' | grep -q "^${container_name}.*healthy"; then
        log_success "$db_name container: HEALTHY"
        HEALTHY_CONTAINERS=$((HEALTHY_CONTAINERS + 1))
    elif docker ps --format '{{.Names}}' | grep -q "^${container_name}$"; then
        log_warning "$db_name container: RUNNING (no health check)"
        HEALTHY_CONTAINERS=$((HEALTHY_CONTAINERS + 1))
    else
        log_error "$db_name container: NOT RUNNING"
    fi
done

# Final Results
log "=================================="
log "DATABASE CONNECTIVITY TEST RESULTS"
log "=================================="

CONNECTIVITY_PERCENTAGE=$(( HEALTHY_CONTAINERS * 100 / TOTAL_CONTAINERS ))
TESTS_PERCENTAGE=$(( PASSED_TESTS * 100 / TOTAL_TESTS ))

log "Container Health: $HEALTHY_CONTAINERS/$TOTAL_CONTAINERS (${CONNECTIVITY_PERCENTAGE}%)"
log "Connectivity Tests: $PASSED_TESTS/$TOTAL_TESTS (${TESTS_PERCENTAGE}%)"

if [ $CONNECTIVITY_PERCENTAGE -eq 100 ] && [ $TESTS_PERCENTAGE -ge 90 ]; then
    log_success "OVERALL STATUS: EXCELLENT - All databases connected and operational"
    EXIT_CODE=0
elif [ $CONNECTIVITY_PERCENTAGE -ge 83 ] && [ $TESTS_PERCENTAGE -ge 80 ]; then  # 5/6 databases = 83%
    log_warning "OVERALL STATUS: GOOD - Most databases operational with minor issues"
    EXIT_CODE=0
else
    log_error "OVERALL STATUS: CRITICAL - Significant database connectivity issues"
    EXIT_CODE=1
fi

log "Detailed test log saved to: $LOG_FILE"

# Performance Summary
log "=================================="
log "PERFORMANCE SUMMARY"
log "=================================="

if [ ! -z "${PG_RESPONSE_TIME:-}" ]; then
    log "PostgreSQL response time: ${PG_RESPONSE_TIME}ms"
fi

if [ ! -z "${REDIS_RESPONSE_TIME:-}" ]; then
    log "Redis response time: ${REDIS_RESPONSE_TIME}ms"
fi

log "PostgreSQL connection utilization: ${CONN_UTILIZATION}%"
log "=================================="

# Recommendations
if [ $CONNECTIVITY_PERCENTAGE -ne 100 ]; then
    log "RECOMMENDATIONS:"
    log "1. Check failed database containers with: docker logs [container_name]"
    log "2. Restart failing containers with: docker restart [container_name]"
    log "3. Verify environment variables and configurations"
fi

if [ ${CONN_UTILIZATION:-0} -gt 75 ]; then
    log "4. Consider increasing PostgreSQL max_connections or implementing connection pooling"
fi

if [ ${PG_RESPONSE_TIME:-0} -gt 100 ]; then
    log "5. Optimize PostgreSQL performance with proper indexing and configuration tuning"
fi

log "Test completed at $(date)"

exit $EXIT_CODE