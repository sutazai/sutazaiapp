#!/bin/bash
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Script: reset-db.sh
# Purpose: Reset database to clean state for development
# Author: Sutazai System
# Date: 2025-09-03
# Version: 1.0.0
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Usage:
#   ./reset-db.sh [options]
#
# Options:
#   -h, --help          Show this help message
#   -v, --verbose       Enable verbose output
#   -d, --dry-run       Run in simulation mode
#   -f, --force         Skip confirmation prompt
#   -s, --seed          Load seed data after reset
#   -b, --backup        Create backup before reset
#
# Requirements:
#   - Docker with PostgreSQL container running
#   - psql client
#   - Valid database credentials in .env file
#
# Examples:
#   ./reset-db.sh --backup --seed
#   ./reset-db.sh --force --dry-run
#━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -euo pipefail
IFS=$'\n\t'

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
readonly ENV_FILE="${PROJECT_ROOT}/.env"
readonly BACKUP_DIR="${PROJECT_ROOT}/backups/database"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly LOG_FILE="${PROJECT_ROOT}/logs/reset-db_${TIMESTAMP}.log"

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Default options
VERBOSE=false
DRY_RUN=false
FORCE=false
SEED_DATA=false
CREATE_BACKUP=false

# Database configuration (loaded from .env)
DB_HOST=""
DB_PORT=""
DB_NAME=""
DB_USER=""
DB_PASSWORD=""

# Logging
log() {
    local level="$1"
    local message="$2"
    local color="${3:-$NC}"
    local timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    
    echo "[${timestamp}] [${level}] ${message}" >> "${LOG_FILE}"
    
    if [[ "$VERBOSE" == true ]] || [[ "$level" == "ERROR" ]] || [[ "$level" == "WARN" ]]; then
        echo -e "${color}[${level}]${NC} ${message}"
    fi
}

# Error handler
error_handler() {
    local line_no=$1
    log "ERROR" "Error occurred at line ${line_no}" "${RED}"
    exit 1
}
trap 'error_handler ${LINENO}' ERR

# Cleanup
cleanup() {
    local exit_code=$?
    if [[ -n "${TEMP_DIR:-}" ]] && [[ -d "${TEMP_DIR}" ]]; then
        rm -rf "${TEMP_DIR}"
    fi
    exit ${exit_code}
}
trap cleanup EXIT INT TERM

# Load environment variables
load_env() {
    if [[ ! -f "${ENV_FILE}" ]]; then
        log "ERROR" "Environment file not found: ${ENV_FILE}" "${RED}"
        exit 5
    fi
    
    # Load database configuration from .env (securely)
    DB_HOST=$(grep -E "^DB_HOST=" "${ENV_FILE}" | cut -d'=' -f2- | tr -d '"' || echo "localhost")
    DB_PORT=$(grep -E "^DB_PORT=" "${ENV_FILE}" | cut -d'=' -f2- | tr -d '"' || echo "10000")
    DB_NAME=$(grep -E "^DB_NAME=" "${ENV_FILE}" | cut -d'=' -f2- | tr -d '"' || echo "jarvis_ai")
    DB_USER=$(grep -E "^DB_USER=" "${ENV_FILE}" | cut -d'=' -f2- | tr -d '"' || echo "jarvis")
    
    # Get password from secrets file (not .env directly)
    local secrets_file="${PROJECT_ROOT}/.secrets/database.secret"
    if [[ -f "${secrets_file}" ]]; then
        DB_PASSWORD=$(grep -E "^POSTGRES_PASSWORD=" "${secrets_file}" | cut -d'=' -f2- | tr -d '"')
    else
        # Fallback to .env (but warn about security)
        DB_PASSWORD=$(grep -E "^DB_PASSWORD=" "${ENV_FILE}" | cut -d'=' -f2- | tr -d '"')
        log "WARN" "Using password from .env file - consider using .secrets/database.secret" "${YELLOW}"
    fi
    
    # Validate we have all required values
    if [[ -z "${DB_PASSWORD}" ]]; then
        log "ERROR" "Database password not found in configuration" "${RED}"
        exit 5
    fi
    
    log "INFO" "Database configuration loaded" "${GREEN}"
}

# Check database connection
check_connection() {
    log "INFO" "Checking database connection..." "${BLUE}"
    
    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "[DRY-RUN] Would check database connection" "${YELLOW}"
        return 0
    fi
    
    # Use environment variable for password (secure)
    export PGPASSWORD="${DB_PASSWORD}"
    
    if psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d postgres -c '\l' &>/dev/null; then
        log "INFO" "Database connection successful" "${GREEN}"
    else
        log "ERROR" "Cannot connect to database" "${RED}"
        exit 5
    fi
    
    unset PGPASSWORD
}

# Create backup
create_backup() {
    if [[ "$CREATE_BACKUP" != true ]]; then
        return 0
    fi
    
    log "INFO" "Creating database backup..." "${BLUE}"
    
    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "[DRY-RUN] Would create backup at ${BACKUP_DIR}/backup_${TIMESTAMP}.sql" "${YELLOW}"
        return 0
    fi
    
    mkdir -p "${BACKUP_DIR}"
    
    export PGPASSWORD="${DB_PASSWORD}"
    local backup_file="${BACKUP_DIR}/backup_${TIMESTAMP}.sql"
    
    if pg_dump -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" > "${backup_file}"; then
        log "INFO" "Backup created: ${backup_file}" "${GREEN}"
        
        # Compress backup
        gzip "${backup_file}"
        log "INFO" "Backup compressed: ${backup_file}.gz" "${GREEN}"
    else
        log "ERROR" "Backup failed" "${RED}"
        exit 1
    fi
    
    unset PGPASSWORD
}

# Reset database
reset_database() {
    log "INFO" "Resetting database..." "${YELLOW}"
    
    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "[DRY-RUN] Would drop and recreate database: ${DB_NAME}" "${YELLOW}"
        return 0
    fi
    
    export PGPASSWORD="${DB_PASSWORD}"
    
    # Drop existing database
    log "INFO" "Dropping database ${DB_NAME}..." "${YELLOW}"
    psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d postgres -c "DROP DATABASE IF EXISTS ${DB_NAME};"
    
    # Create new database
    log "INFO" "Creating database ${DB_NAME}..." "${BLUE}"
    psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d postgres -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"
    
    # Run migrations
    log "INFO" "Running database migrations..." "${BLUE}"
    cd "${PROJECT_ROOT}/backend"
    if [[ -f "alembic.ini" ]]; then
        ./venv/bin/alembic upgrade head
        log "INFO" "Migrations completed" "${GREEN}"
    else
        log "WARN" "No Alembic configuration found, skipping migrations" "${YELLOW}"
    fi
    
    unset PGPASSWORD
}

# Load seed data
load_seed_data() {
    if [[ "$SEED_DATA" != true ]]; then
        return 0
    fi
    
    log "INFO" "Loading seed data..." "${BLUE}"
    
    if [[ "$DRY_RUN" == true ]]; then
        log "INFO" "[DRY-RUN] Would load seed data" "${YELLOW}"
        return 0
    fi
    
    local seed_file="${PROJECT_ROOT}/data/seed/seed_data.sql"
    
    if [[ -f "${seed_file}" ]]; then
        export PGPASSWORD="${DB_PASSWORD}"
        psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" < "${seed_file}"
        log "INFO" "Seed data loaded" "${GREEN}"
        unset PGPASSWORD
    else
        log "WARN" "Seed data file not found: ${seed_file}" "${YELLOW}"
    fi
}

# Confirmation prompt
confirm_reset() {
    if [[ "$FORCE" == true ]] || [[ "$DRY_RUN" == true ]]; then
        return 0
    fi
    
    echo -e "${YELLOW}⚠ WARNING: This will DELETE all data in database ${DB_NAME}${NC}"
    echo -en "${YELLOW}Are you sure you want to continue? [y/N]: ${NC}"
    read -r response
    
    case "$response" in
        [yY][eE][sS]|[yY])
            return 0
            ;;
        *)
            log "INFO" "Operation cancelled by user" "${YELLOW}"
            exit 0
            ;;
    esac
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
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -s|--seed)
            SEED_DATA=true
            shift
            ;;
        -b|--backup)
            CREATE_BACKUP=true
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
    
    log "INFO" "Starting database reset process" "${BLUE}"
    
    # Load configuration
    load_env
    
    # Check connection
    check_connection
    
    # Confirm action
    confirm_reset
    
    # Create backup if requested
    create_backup
    
    # Reset database
    reset_database
    
    # Load seed data if requested
    load_seed_data
    
    log "INFO" "Database reset completed successfully!" "${GREEN}"
    echo -e "\n${GREEN}✓${NC} Database has been reset to clean state"
    
    if [[ "$SEED_DATA" == true ]]; then
        echo -e "${GREEN}✓${NC} Seed data has been loaded"
    fi
}

main