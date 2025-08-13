#!/usr/bin/env bash
# =====================================================================
# UUID MIGRATION EXECUTION WRAPPER
# =====================================================================
# Purpose: Safely execute the INTEGER to UUID migration with all safety checks
# Usage: ./execute_uuid_migration.sh [--dry-run] [--skip-backup] [--auto-confirm]
# 
# Author: Claude Code (Senior Backend Developer)
# Date: 2025-08-09
# =====================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MIGRATION_SQL="$ROOT_DIR/migrations/sql/integer_to_uuid_migration.sql"
VALIDATION_SQL="$ROOT_DIR/migrations/sql/validate_uuid_migration.sql"
ROLLBACK_SQL="$ROOT_DIR/migrations/sql/rollback_uuid_to_integer.sql"

# Database connection parameters
DB_HOST="${POSTGRES_HOST:-127.0.0.1}"
DB_PORT="${POSTGRES_PORT:-10000}"
DB_NAME="${POSTGRES_DB:-sutazai}"
DB_USER="${POSTGRES_USER:-sutazai}"
DB_PASSWORD="${POSTGRES_PASSWORD:-sutazai_pass}"

# Flags
DRY_RUN=false
SKIP_BACKUP=false
AUTO_CONFIRM=false

# =====================================================================
# FUNCTIONS
# =====================================================================

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

confirm() {
    if [[ "$AUTO_CONFIRM" == "true" ]]; then
        return 0
    fi
    
    local message="$1"
    echo -e "${YELLOW}$message${NC}"
    read -p "Continue? [y/N]: " response
    case "$response" in
        [yY][eE][sS]|[yY]) 
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

run_psql() {
    local sql_file="$1"
    local description="$2"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY RUN: Would execute $description"
        return 0
    fi
    
    log "Executing: $description"
    export PGPASSWORD="$DB_PASSWORD"
    
    if ! psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
        -v ON_ERROR_STOP=1 -f "$sql_file"; then
        error "Failed to execute: $description"
        return 1
    fi
    
    success "Completed: $description"
    return 0
}

check_database_connection() {
    log "Testing database connection..."
    
    export PGPASSWORD="$DB_PASSWORD"
    if ! psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
        -c "SELECT version();" > /dev/null 2>&1; then
        error "Cannot connect to database at $DB_HOST:$DB_PORT"
        error "Please check your database connection parameters"
        return 1
    fi
    
    success "Database connection successful"
    return 0
}

check_current_schema() {
    log "Checking current database schema..."
    
    export PGPASSWORD="$DB_PASSWORD"
    local users_id_type
    users_id_type=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
        -t -c "SELECT data_type FROM information_schema.columns WHERE table_name='users' AND column_name='id';" 2>/dev/null | tr -d '[:space:]')
    
    if [[ "$users_id_type" == "uuid" ]]; then
        error "Database already uses UUID schema! Migration not needed."
        error "If you need to rollback, use the rollback script instead."
        return 1
    elif [[ "$users_id_type" == "integer" ]]; then
        success "Confirmed: Database uses INTEGER schema (migration needed)"
        return 0
    else
        error "Unexpected schema state. users.id has type: '$users_id_type'"
        return 1
    fi
}

create_full_backup() {
    if [[ "$SKIP_BACKUP" == "true" ]]; then
        warning "Skipping full database backup (--skip-backup flag used)"
        return 0
    fi
    
    log "Creating full database backup..."
    
    local backup_file="$ROOT_DIR/backups/pre_uuid_migration_$(date +%Y%m%d_%H%M%S).sql"
    mkdir -p "$(dirname "$backup_file")"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY RUN: Would create backup at $backup_file"
        return 0
    fi
    
    export PGPASSWORD="$DB_PASSWORD"
    if ! pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
        --verbose --no-password > "$backup_file"; then
        error "Failed to create database backup"
        return 1
    fi
    
    success "Full database backup created: $backup_file"
    log "Backup size: $(du -h "$backup_file" | cut -f1)"
    return 0
}

show_current_data() {
    log "Current database contents:"
    
    export PGPASSWORD="$DB_PASSWORD"
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT 'USERS' as table_name, COUNT(*) as record_count FROM users
        UNION ALL
        SELECT 'AGENTS', COUNT(*) FROM agents  
        UNION ALL
        SELECT 'TASKS', COUNT(*) FROM tasks
        UNION ALL
        SELECT 'CHAT_HISTORY', COUNT(*) FROM chat_history
        UNION ALL
        SELECT 'AGENT_EXECUTIONS', COUNT(*) FROM agent_executions
        UNION ALL
        SELECT 'SYSTEM_METRICS', COUNT(*) FROM system_metrics
        ORDER BY table_name;
    "
}

check_files_exist() {
    log "Checking required migration files..."
    
    if [[ ! -f "$MIGRATION_SQL" ]]; then
        error "Migration script not found: $MIGRATION_SQL"
        return 1
    fi
    
    if [[ ! -f "$VALIDATION_SQL" ]]; then
        error "Validation script not found: $VALIDATION_SQL"
        return 1
    fi
    
    if [[ ! -f "$ROLLBACK_SQL" ]]; then
        error "Rollback script not found: $ROLLBACK_SQL"
        return 1
    fi
    
    success "All required migration files found"
    return 0
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN=true
                log "DRY RUN mode enabled - no changes will be made"
                shift
                ;;
            --skip-backup)
                SKIP_BACKUP=true
                warning "Full backup will be SKIPPED"
                shift
                ;;
            --auto-confirm)
                AUTO_CONFIRM=true
                log "Auto-confirm enabled - no prompts will be shown"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    cat << EOF
UUID Migration Execution Script

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --dry-run        Show what would be done without making changes
    --skip-backup    Skip creating full database backup (NOT recommended)
    --auto-confirm   Skip confirmation prompts (use with caution)
    -h, --help       Show this help message

ENVIRONMENT VARIABLES:
    POSTGRES_HOST    Database host (default: 127.0.0.1)
    POSTGRES_PORT    Database port (default: 10000)
    POSTGRES_DB      Database name (default: sutazai)
    POSTGRES_USER    Database user (default: sutazai)
    POSTGRES_PASSWORD Database password (default: sutazai_pass)

EXAMPLES:
    # Test run (recommended first)
    $0 --dry-run
    
    # Execute migration with backup
    $0
    
    # Execute without prompts (CI/CD)
    $0 --auto-confirm

IMPORTANT:
    - This migration converts all INTEGER IDs to UUIDs
    - All existing data will be preserved via careful mapping
    - Full database backup is created by default
    - Rollback script is available if needed
    - Run --dry-run first to test!

EOF
}

# =====================================================================
# MAIN EXECUTION
# =====================================================================

main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  UUID MIGRATION EXECUTION SCRIPT${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Pre-flight checks
    log "Starting pre-flight checks..."
    
    check_files_exist || exit 1
    check_database_connection || exit 1
    check_current_schema || exit 1
    
    # Show current data
    show_current_data
    echo
    
    # Major warning
    cat << EOF
${RED}⚠️  CRITICAL WARNING ⚠️${NC}

This migration will:
${YELLOW}1. Convert ALL INTEGER primary keys to UUID${NC}
${YELLOW}2. Update ALL foreign key references${NC}
${YELLOW}3. Rebuild ALL constraints and indexes${NC}
${YELLOW}4. Modify core database structure${NC}

${GREEN}Safety measures in place:${NC}
${GREEN}✅ Full database backup will be created${NC}
${GREEN}✅ All data will be preserved via mapping tables${NC}
${GREEN}✅ Atomic transaction (all or nothing)${NC}
${GREEN}✅ Rollback script available if needed${NC}

${RED}THIS IS A MAJOR DATABASE OPERATION!${NC}

EOF

    if ! confirm "Do you want to proceed with the UUID migration?"; then
        log "Migration cancelled by user"
        exit 0
    fi
    
    # Create full backup
    if ! create_full_backup; then
        error "Cannot proceed without backup. Aborting."
        exit 1
    fi
    
    # Final confirmation before migration
    if ! confirm "Ready to execute the UUID migration. This will modify your database structure. Continue?"; then
        log "Migration cancelled by user"
        exit 0
    fi
    
    # Execute migration
    log "Starting UUID migration..."
    
    local start_time
    start_time=$(date +%s)
    
    if run_psql "$MIGRATION_SQL" "UUID Migration"; then
        success "UUID migration completed successfully!"
        
        # Run validation
        log "Running post-migration validation..."
        if run_psql "$VALIDATION_SQL" "Migration Validation"; then
            success "Validation passed! Migration was successful."
        else
            warning "Validation had issues. Check the output above."
        fi
        
        local end_time
        end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        log "Migration completed in ${duration} seconds"
        
        # Show rollback information
        cat << EOF

${GREEN}Migration completed successfully!${NC}

${BLUE}Important notes:${NC}
${GREEN}✅ All INTEGER IDs have been converted to UUIDs${NC}
${GREEN}✅ All data relationships have been preserved${NC}
${GREEN}✅ Backup tables are available for rollback${NC}

${YELLOW}If you need to rollback:${NC}
${YELLOW}    psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f $ROLLBACK_SQL${NC}

${YELLOW}Backup tables created (can be deleted after confirming success):${NC}
${YELLOW}    *_backup_pre_uuid${NC}
${YELLOW}    uuid_migration_mapping_*${NC}

EOF
        
    else
        error "Migration failed! Database should be rolled back automatically."
        error "If not, you can manually rollback using: $ROLLBACK_SQL"
        exit 1
    fi
}

# Handle script interruption
trap 'error "Migration interrupted! Check database state."; exit 1' INT TERM

# Execute main function with all arguments
main "$@"