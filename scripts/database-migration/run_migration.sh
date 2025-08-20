#!/bin/bash
# Master orchestration script for database consolidation
# Runs all migration steps in sequence with safety checks

set -e

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"

echo "================================================"
echo "    DATABASE CONSOLIDATION ORCHESTRATOR"
echo "================================================"
echo ""
echo "This script will consolidate all SQLite databases"
echo "into a unified PostgreSQL database."
echo ""

# Function to check if PostgreSQL is running
check_postgres() {
    echo -n "Checking PostgreSQL connectivity... "
    if python3 -c "import psycopg2; psycopg2.connect(host='localhost', port=10000, user='sutazai', password='sutazai_password', database='sutazai')" 2>/dev/null; then
        echo "✓ Connected"
        return 0
    else
        echo "✗ Failed"
        echo "ERROR: Cannot connect to PostgreSQL on port 10000"
        echo "Please ensure PostgreSQL is running with correct credentials"
        return 1
    fi
}

# Function to run a step
run_step() {
    local step_name="$1"
    local script="$2"
    shift 2
    local args="$@"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "STEP: $step_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    if [ -f "$script" ]; then
        chmod +x "$script"
        if $script $args; then
            echo ""
            echo "✓ $step_name completed successfully"
            return 0
        else
            echo ""
            echo "✗ $step_name failed"
            return 1
        fi
    else
        echo "✗ Script not found: $script"
        return 1
    fi
}

# Parse command line arguments
DRY_RUN=""
SKIP_BACKUP=false
SKIP_VALIDATION=false
AUTO_CLEANUP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            echo "🔍 DRY RUN MODE: No actual data will be migrated"
            shift
            ;;
        --skip-backup)
            SKIP_BACKUP=true
            echo "⚠️  WARNING: Skipping backup step"
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            echo "⚠️  WARNING: Skipping validation step"
            shift
            ;;
        --auto-cleanup)
            AUTO_CLEANUP=true
            echo "🗑️  Auto-cleanup enabled after successful migration"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run         Simulate migration without actual data transfer"
            echo "  --skip-backup     Skip the backup step (dangerous!)"
            echo "  --skip-validation Skip the validation step"
            echo "  --auto-cleanup    Automatically cleanup old databases after migration"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

# Check prerequisites
echo "Checking prerequisites..."

# Check if Python packages are installed
echo -n "Checking Python packages... "
if python3 -c "import psycopg2, sqlite3, json" 2>/dev/null; then
    echo "✓ Required packages installed"
else
    echo "✗ Missing packages"
    echo "Installing required packages..."
    pip3 install psycopg2-binary
fi

# Check PostgreSQL connectivity
if ! check_postgres; then
    exit 1
fi

echo ""
echo "Prerequisites check complete ✓"
echo ""

# Confirm before proceeding
if [ -z "$DRY_RUN" ]; then
    echo "This will migrate all SQLite databases to PostgreSQL."
    echo -n "Do you want to proceed? (yes/no): "
    read -r response
    if [ "$response" != "yes" ]; then
        echo "Migration cancelled"
        exit 0
    fi
fi

# Step 1: Backup
if [ "$SKIP_BACKUP" = false ]; then
    if ! run_step "Database Backup" "./00_backup_all_databases.sh"; then
        echo "Backup failed! Aborting migration."
        exit 1
    fi
else
    echo ""
    echo "⚠️  Skipping backup as requested"
fi

# Step 2: Create PostgreSQL schema
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP: Create PostgreSQL Schema"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -z "$DRY_RUN" ]; then
    echo "Creating database schema..."
    if PGPASSWORD=sutazai_password psql -h localhost -p 10000 -U sutazai -d sutazai -f 01_create_postgres_schema.sql; then
        echo "✓ Schema created successfully"
    else
        echo "✗ Schema creation failed"
        exit 1
    fi
else
    echo "🔍 Dry run: Would create PostgreSQL schema"
fi

# Step 3: Run migration
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP: Data Migration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

chmod +x 02_migrate_sqlite_to_postgres.py
if python3 02_migrate_sqlite_to_postgres.py $DRY_RUN; then
    echo "✓ Migration completed successfully"
else
    echo "✗ Migration failed"
    exit 1
fi

# Step 4: Validation
if [ "$SKIP_VALIDATION" = false ] && [ -z "$DRY_RUN" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "STEP: Migration Validation"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    chmod +x 03_validate_migration.py
    if python3 03_validate_migration.py; then
        echo "✓ Validation completed"
        
        # Check validation results
        if [ -f "validation_report.json" ]; then
            SUCCESS_RATE=$(python3 -c "import json; data=json.load(open('validation_report.json')); print(data['validation_summary']['success_rate'])")
            echo ""
            echo "Validation success rate: $SUCCESS_RATE%"
            
            if (( $(echo "$SUCCESS_RATE < 100" | bc -l) )); then
                echo "⚠️  WARNING: Validation did not pass 100%"
                echo "Review validation_report.json for details"
                AUTO_CLEANUP=false
            fi
        fi
    else
        echo "✗ Validation failed"
        AUTO_CLEANUP=false
    fi
else
    if [ -n "$DRY_RUN" ]; then
        echo ""
        echo "🔍 Dry run: Skipping validation"
    else
        echo ""
        echo "⚠️  Skipping validation as requested"
    fi
fi

# Step 5: Cleanup (if requested)
if [ "$AUTO_CLEANUP" = true ] && [ -z "$DRY_RUN" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "STEP: Database Cleanup"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    if run_step "Database Cleanup" "./04_cleanup_old_databases.sh" "--confirm-cleanup"; then
        echo "✓ Old databases cleaned up"
    else
        echo "✗ Cleanup failed (databases remain in place)"
    fi
else
    if [ -z "$DRY_RUN" ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Next Steps"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "1. Test your application with the new PostgreSQL database"
        echo "2. Monitor for any issues"
        echo "3. Once confirmed working, run cleanup:"
        echo "   ./04_cleanup_old_databases.sh --confirm-cleanup"
    fi
fi

echo ""
echo "================================================"
echo "    MIGRATION PROCESS COMPLETE"
echo "================================================"
echo ""

if [ -n "$DRY_RUN" ]; then
    echo "This was a dry run. No data was actually migrated."
    echo "To perform the actual migration, run without --dry-run"
else
    echo "✓ Database consolidation complete!"
    echo ""
    echo "PostgreSQL unified database is ready at:"
    echo "  Host: localhost"
    echo "  Port: 10000"
    echo "  Database: sutazai"
    echo "  Table: unified_memory"
    echo ""
    echo "Total records migrated: Check validation_report.json for details"
fi