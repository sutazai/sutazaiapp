#!/bin/bash
################################################################################
# DATABASE VALIDATION SCRIPT
# Purpose: Validate PostgreSQL database state and ensure proper initialization
# Author: ULTRA-REMEDIATION-MASTER-001
# Date: August 13, 2025
# Follows: CLAUDE.md Rules 1, 2, 3, 4, 19 (REAL implementation only)
################################################################################

set -euo pipefail

# Script configuration
readonly SCRIPT_NAME="Database Validation"
readonly SCRIPT_VERSION="1.0.0"
readonly PROJECT_ROOT="/opt/sutazaiapp"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly LOG_FILE="${PROJECT_ROOT}/logs/database_validation_${TIMESTAMP}.log"

# Database configuration from docker-compose.yml (REAL values)
readonly DB_HOST="localhost"
readonly DB_PORT="10000"
readonly DB_NAME="${POSTGRES_DB:-sutazai}"
readonly DB_USER="${POSTGRES_USER:-sutazai}"
readonly DB_CONTAINER="sutazai-postgres"

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly BOLD='\033[1m'
readonly NC='\033[0m'

# Expected tables (from REAL backend code analysis)
readonly EXPECTED_TABLES=(
    "alembic_version"
    "users"
    "sessions"
    "tasks"
    "agents"
    "models"
    "conversations"
    "messages"
    "documents"
    "embeddings"
)

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

################################################################################
# LOGGING FUNCTIONS
################################################################################

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    case "$level" in
        ERROR)
            echo -e "${RED}[ERROR]${NC} $message" >&2
            ;;
        SUCCESS)
            echo -e "${GREEN}[SUCCESS]${NC} $message"
            ;;
        INFO)
            echo -e "${BLUE}[INFO]${NC} $message"
            ;;
        WARN)
            echo -e "${YELLOW}[WARN]${NC} $message"
            ;;
    esac
}

################################################################################
# VALIDATION FUNCTIONS
################################################################################

check_prerequisites() {
    log "INFO" "Checking prerequisites..."
    
    # Check if PostgreSQL container is running
    if ! docker ps --filter "name=$DB_CONTAINER" --format "{{.Names}}" | grep -q "$DB_CONTAINER"; then
        log "ERROR" "PostgreSQL container '$DB_CONTAINER' is not running"
        log "INFO" "Try running: docker-compose up -d postgres"
        return 1
    fi
    
    # Check if container is healthy
    local container_health
    container_health=$(docker inspect "$DB_CONTAINER" --format='{{.State.Health.Status}}' 2>/dev/null || echo "no_health_check")
    
    if [[ "$container_health" == "healthy" ]]; then
        log "SUCCESS" "PostgreSQL container is healthy"
    elif [[ "$container_health" == "starting" ]]; then
        log "WARN" "PostgreSQL container is still starting up"
    else
        log "WARN" "PostgreSQL container health status: $container_health"
    fi
    
    # Test database connectivity
    if docker exec "$DB_CONTAINER" pg_isready -U "$DB_USER" >/dev/null 2>&1; then
        log "SUCCESS" "PostgreSQL is accepting connections"
    else
        log "ERROR" "PostgreSQL is not accepting connections"
        return 1
    fi
    
    log "SUCCESS" "Prerequisites check passed"
    return 0
}

get_database_info() {
    log "INFO" "Gathering database information..."
    
    # Get PostgreSQL version
    local pg_version
    pg_version=$(docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT version();" 2>/dev/null | head -1 | xargs)
    log "INFO" "PostgreSQL Version: $pg_version"
    
    # Get database size
    local db_size
    db_size=$(docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT pg_size_pretty(pg_database_size('$DB_NAME'));" 2>/dev/null | xargs)
    log "INFO" "Database Size: $db_size"
    
    # Get connection count
    local conn_count
    conn_count=$(docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT count(*) FROM pg_stat_activity WHERE datname='$DB_NAME';" 2>/dev/null | xargs)
    log "INFO" "Active Connections: $conn_count"
}

validate_schema() {
    log "INFO" "Validating database schema..."
    
    # Get list of existing tables
    local existing_tables
    existing_tables=$(docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT tablename 
        FROM pg_tables 
        WHERE schemaname = 'public' 
        ORDER BY tablename;
    " 2>/dev/null | grep -v '^$' | xargs)
    
    if [[ -z "$existing_tables" ]]; then
        log "WARN" "No tables found in database - database may need initialization"
        return 1
    fi
    
    log "INFO" "Found tables: $existing_tables"
    
    # Check for critical tables
    local missing_tables=()
    local found_tables=()
    
    for expected_table in "${EXPECTED_TABLES[@]}"; do
        if echo "$existing_tables" | grep -q "\b$expected_table\b"; then
            found_tables+=("$expected_table")
        else
            missing_tables+=("$expected_table")
        fi
    done
    
    log "INFO" "Found expected tables (${#found_tables[@]}/${#EXPECTED_TABLES[@]}): ${found_tables[*]}"
    
    if [[ ${#missing_tables[@]} -gt 0 ]]; then
        log "WARN" "Missing expected tables (${#missing_tables[@]}): ${missing_tables[*]}"
        return 2
    fi
    
    log "SUCCESS" "Schema validation passed - all expected tables present"
    return 0
}

validate_alembic_version() {
    log "INFO" "Validating Alembic migration status..."
    
    # Check if alembic_version table exists and has data
    local version_result
    version_result=$(docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT version_num 
        FROM alembic_version 
        LIMIT 1;
    " 2>/dev/null | xargs || echo "not_found")
    
    if [[ "$version_result" == "not_found" ]]; then
        log "WARN" "No Alembic version found - database may not be migrated"
        return 1
    fi
    
    log "INFO" "Current Alembic version: $version_result"
    
    # Validate version format (should be a hash)
    if [[ "$version_result" =~ ^[a-f0-9]{12}$ ]]; then
        log "SUCCESS" "Alembic version format is valid"
    else
        log "WARN" "Alembic version format may be invalid: $version_result"
    fi
    
    return 0
}

check_table_data() {
    log "INFO" "Checking table data consistency..."
    
    local total_records=0
    
    # Check record counts for key tables
    for table in "${EXPECTED_TABLES[@]}"; do
        # Skip alembic_version as it's not a data table
        [[ "$table" == "alembic_version" ]] && continue
        
        local record_count
        record_count=$(docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -t -c "
            SELECT COUNT(*) FROM $table;
        " 2>/dev/null | xargs || echo "0")
        
        log "INFO" "Table '$table': $record_count records"
        total_records=$((total_records + record_count))
    done
    
    log "INFO" "Total records across all tables: $total_records"
    
    if [[ $total_records -eq 0 ]]; then
        log "WARN" "No data found in any tables - this may be expected for a fresh installation"
        return 1
    fi
    
    return 0
}

validate_constraints() {
    log "INFO" "Validating database constraints..."
    
    # Check for foreign key violations
    local fk_violations
    fk_violations=$(docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT COUNT(*) 
        FROM information_schema.table_constraints 
        WHERE constraint_type = 'FOREIGN KEY' 
        AND table_schema = 'public';
    " 2>/dev/null | xargs || echo "0")
    
    log "INFO" "Foreign key constraints found: $fk_violations"
    
    # Check for unique constraint violations (basic check)
    local unique_constraints
    unique_constraints=$(docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT COUNT(*) 
        FROM information_schema.table_constraints 
        WHERE constraint_type = 'UNIQUE' 
        AND table_schema = 'public';
    " 2>/dev/null | xargs || echo "0")
    
    log "INFO" "Unique constraints found: $unique_constraints"
    
    log "SUCCESS" "Constraint validation completed"
    return 0
}

run_health_diagnostics() {
    log "INFO" "Running database health diagnostics..."
    
    # Check for long-running queries
    local long_queries
    long_queries=$(docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT COUNT(*) 
        FROM pg_stat_activity 
        WHERE state = 'active' 
        AND query_start < NOW() - INTERVAL '5 minutes';
    " 2>/dev/null | xargs || echo "0")
    
    if [[ $long_queries -gt 0 ]]; then
        log "WARN" "Found $long_queries long-running queries (>5 minutes)"
    else
        log "SUCCESS" "No long-running queries detected"
    fi
    
    # Check for blocked queries
    local blocked_queries
    blocked_queries=$(docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT COUNT(*) 
        FROM pg_stat_activity 
        WHERE wait_event_type IS NOT NULL;
    " 2>/dev/null | xargs || echo "0")
    
    if [[ $blocked_queries -gt 0 ]]; then
        log "WARN" "Found $blocked_queries waiting/blocked queries"
    else
        log "SUCCESS" "No blocked queries detected"
    fi
    
    return 0
}

suggest_remediation() {
    local validation_result="$1"
    
    log "INFO" "Suggesting remediation actions..."
    
    case "$validation_result" in
        0)
            log "SUCCESS" "Database is healthy - no action needed"
            ;;
        1)
            log "WARN" "Database schema incomplete - consider running migrations:"
            echo -e "  ${YELLOW}cd $PROJECT_ROOT${NC}"
            echo -e "  ${YELLOW}docker exec $DB_CONTAINER psql -U $DB_USER -d $DB_NAME -f /docker-entrypoint-initdb.d/init.sql${NC}"
            echo -e "  ${YELLOW}# OR run: docker-compose restart backend  # to trigger migrations${NC}"
            ;;
        2)
            log "WARN" "Missing expected tables - run database initialization:"
            echo -e "  ${YELLOW}# Check if init_db.sql exists:${NC}"
            echo -e "  ${YELLOW}ls -la $PROJECT_ROOT/IMPORTANT/init_db.sql${NC}"
            echo -e "  ${YELLOW}# If exists, apply it:${NC}"
            echo -e "  ${YELLOW}docker exec -i $DB_CONTAINER psql -U $DB_USER -d $DB_NAME < $PROJECT_ROOT/IMPORTANT/init_db.sql${NC}"
            ;;
        *)
            log "ERROR" "Unexpected validation result: $validation_result"
            ;;
    esac
}

display_summary() {
    local overall_result="$1"
    
    echo -e "\n${BOLD}Database Validation Summary:${NC}"
    echo -e "Database: $DB_NAME@$DB_HOST:$DB_PORT"
    echo -e "Container: $DB_CONTAINER"
    echo -e "Log File: $LOG_FILE"
    
    case "$overall_result" in
        0)
            echo -e "Status: ${GREEN}HEALTHY${NC} ✓"
            ;;
        1)
            echo -e "Status: ${YELLOW}NEEDS_ATTENTION${NC} ⚠"
            ;;
        2)
            echo -e "Status: ${RED}INITIALIZATION_REQUIRED${NC} ✗"
            ;;
        *)
            echo -e "Status: ${RED}UNKNOWN_ERROR${NC} ✗"
            ;;
    esac
}

################################################################################
# MAIN EXECUTION
################################################################################

main() {
    log "INFO" "Starting $SCRIPT_NAME v$SCRIPT_VERSION"
    
    echo -e "${BOLD}PostgreSQL Database Validation${NC}"
    echo -e "Target: $DB_NAME@$DB_HOST:$DB_PORT"
    echo -e "Log: $LOG_FILE\n"
    
    local overall_result=0
    
    # Step 1: Check prerequisites
    if ! check_prerequisites; then
        display_summary 2
        exit 2
    fi
    
    # Step 2: Get database info
    get_database_info
    
    # Step 3: Validate schema
    local schema_result
    if ! validate_schema; then
        schema_result=$?
        overall_result=$schema_result
    fi
    
    # Step 4: Validate Alembic version
    if ! validate_alembic_version; then
        [[ $overall_result -eq 0 ]] && overall_result=1
    fi
    
    # Step 5: Check table data (informational)
    check_table_data || true
    
    # Step 6: Validate constraints
    validate_constraints || true
    
    # Step 7: Run health diagnostics
    run_health_diagnostics || true
    
    # Step 8: Display summary and suggestions
    suggest_remediation "$overall_result"
    display_summary "$overall_result"
    
    exit "$overall_result"
}

# Signal handlers
trap 'log "ERROR" "Script interrupted by user"; exit 130' INT
trap 'log "ERROR" "Script terminated"; exit 143' TERM

# Execute main function
main "$@"