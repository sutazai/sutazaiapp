#!/bin/bash
# Purpose: Automated backup verification for SutazAI system
# Usage: ./backup-verification.sh [--dry-run] [--verify-all]
# Requires: Standard Unix tools, PostgreSQL client

set -euo pipefail


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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/opt/sutazaiapp"
BACKUP_DIR="$BASE_DIR/backups"
LOG_DIR="$BASE_DIR/logs"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Configuration
DRY_RUN=false
VERIFY_ALL=false
MAX_BACKUP_AGE_DAYS=7      # Warn if no recent backups
MAX_VERIFICATION_TIME=300  # Maximum time for each verification (seconds)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verify-all)
            VERIFY_ALL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--verify-all]"
            exit 1
            ;;
    esac
done

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local log_file="$LOG_DIR/backup_verification_$TIMESTAMP.log"
    
    echo "[$timestamp] $level: $message" >> "$log_file"
    
    case $level in
        ERROR) echo -e "${RED}[$timestamp] ERROR: $message${NC}" ;;
        WARN) echo -e "${YELLOW}[$timestamp] WARN: $message${NC}" ;;
        INFO) echo -e "${BLUE}[$timestamp] INFO: $message${NC}" ;;
        SUCCESS) echo -e "${GREEN}[$timestamp] SUCCESS: $message${NC}" ;;
    esac
}

# Setup verification directories
setup_verification_directories() {
    log "INFO" "Setting up backup verification directories..."
    mkdir -p "$LOG_DIR"
}

# Find all backup files
find_backup_files() {
    log "INFO" "Scanning for backup files..."
    
    local backup_files="{}"
    
    # Database backups
    local db_backups=()
    if [[ -d "$BACKUP_DIR/database" ]]; then
        while IFS= read -r -d '' backup; do
            if [[ -n "$backup" ]]; then
                db_backups+=("$backup")
            fi
        done < <(find "$BACKUP_DIR/database" -name "*.sql.gz" -type f -print0 2>/dev/null)
    fi
    
    # Configuration backups
    local config_backups=()
    if [[ -d "$BACKUP_DIR/config" ]]; then
        while IFS= read -r -d '' backup; do
            if [[ -n "$backup" ]]; then
                config_backups+=("$backup")
            fi
        done < <(find "$BACKUP_DIR/config" -name "*.tar.gz" -type f -print0 2>/dev/null)
    fi
    
    # Certificate backups
    local cert_backups=()
    if [[ -d "$BACKUP_DIR/certificates" ]]; then
        while IFS= read -r -d '' backup; do
            if [[ -n "$backup" ]]; then
                cert_backups+=("$backup")
            fi
        done < <(find "$BACKUP_DIR/certificates" -name "*.pem" -type f -print0 2>/dev/null)
    fi
    
    # System backups
    local system_backups=()
    if [[ -d "$BACKUP_DIR/system" ]]; then
        while IFS= read -r -d '' backup; do
            if [[ -n "$backup" ]]; then
                system_backups+=("$backup")
            fi
        done < <(find "$BACKUP_DIR/system" -name "*.tar.gz" -type f -print0 2>/dev/null)
    fi
    
    backup_files=$(jq -n \
        --argjson db_backups "$(printf '%s\n' "${db_backups[@]}" | jq -R . | jq -s . 2>/dev/null || echo '[]')" \
        --argjson config_backups "$(printf '%s\n' "${config_backups[@]}" | jq -R . | jq -s . 2>/dev/null || echo '[]')" \
        --argjson cert_backups "$(printf '%s\n' "${cert_backups[@]}" | jq -R . | jq -s . 2>/dev/null || echo '[]')" \
        --argjson system_backups "$(printf '%s\n' "${system_backups[@]}" | jq -R . | jq -s . 2>/dev/null || echo '[]')" \
        '{
            "database": $db_backups,
            "config": $config_backups,
            "certificates": $cert_backups,
            "system": $system_backups
        }')
    
    local total_backups=$((${#db_backups[@]} + ${#config_backups[@]} + ${#cert_backups[@]} + ${#system_backups[@]}))
    log "INFO" "Found $total_backups backup files: ${#db_backups[@]} database, ${#config_backups[@]} config, ${#cert_backups[@]} certificates, ${#system_backups[@]} system"
    
    echo "$backup_files"
}

# Check backup file integrity
check_file_integrity() {
    local file_path="$1"
    local file_type="$2"
    
    log "INFO" "Checking integrity of $file_type backup: $(basename "$file_path")"
    
    if [[ ! -f "$file_path" ]]; then
        log "ERROR" "Backup file not found: $file_path"
        return 1
    fi
    
    # Check file size (should be > 0)
    local file_size=$(stat -c%s "$file_path" 2>/dev/null || echo 0)
    if [[ $file_size -eq 0 ]]; then
        log "ERROR" "Backup file is empty: $file_path"
        return 1
    fi
    
    # Check file age
    local file_age_days=$(( ($(date +%s) - $(stat -c%Y "$file_path" 2>/dev/null || echo 0)) / 86400 ))
    if [[ $file_age_days -gt $MAX_BACKUP_AGE_DAYS ]]; then
        log "WARN" "Backup file is $file_age_days days old (threshold: $MAX_BACKUP_AGE_DAYS days): $file_path"
    fi
    
    # Type-specific integrity checks
    case $file_type in
        "database")
            if [[ "$file_path" == *.gz ]]; then
                if ! gzip -t "$file_path" 2>/dev/null; then
                    log "ERROR" "Database backup is corrupted (gzip test failed): $file_path"
                    return 1
                fi
            fi
            ;;
        "config"|"system")
            if [[ "$file_path" == *.tar.gz ]]; then
                if ! tar -tzf "$file_path" >/dev/null 2>&1; then
                    log "ERROR" "Archive backup is corrupted (tar test failed): $file_path"
                    return 1
                fi
            fi
            ;;
        "certificates")
            if [[ "$file_path" == *.pem ]]; then
                if ! openssl x509 -in "$file_path" -noout -text >/dev/null 2>&1; then
                    log "ERROR" "Certificate backup is corrupted: $file_path"
                    return 1
                fi
            fi
            ;;
    esac
    
    log "SUCCESS" "Backup file integrity check passed: $(basename "$file_path") (${file_size} bytes, ${file_age_days} days old)"
    return 0
}

# Verify database backup content
verify_database_backup() {
    local backup_file="$1"
    
    log "INFO" "Verifying database backup content: $(basename "$backup_file")"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would verify database backup: $backup_file"
        return 0
    fi
    
    # Create temporary test database
    local test_db_name="sutazai_backup_test_$(date +%s)"
    local temp_sql_file="/tmp/backup_test_$TIMESTAMP.sql"
    
    # Extract and examine backup content
    if [[ "$backup_file" == *.gz ]]; then
        if ! timeout $MAX_VERIFICATION_TIME gunzip -c "$backup_file" > "$temp_sql_file" 2>/dev/null; then
            log "ERROR" "Failed to extract database backup: $backup_file"
            rm -f "$temp_sql_file"
            return 1
        fi
    else
        cp "$backup_file" "$temp_sql_file"
    fi
    
    # Check if backup contains expected SQL content
    if ! grep -q "CREATE TABLE\|INSERT INTO\|COPY" "$temp_sql_file" 2>/dev/null; then
        log "ERROR" "Database backup does not contain expected SQL content: $backup_file"
        rm -f "$temp_sql_file"
        return 1
    fi
    
    # Count tables and data in backup
    local table_count=$(grep -c "CREATE TABLE" "$temp_sql_file" 2>/dev/null || echo 0)
    local insert_count=$(grep -c "INSERT INTO\|COPY" "$temp_sql_file" 2>/dev/null || echo 0)
    
    log "INFO" "Database backup contains $table_count tables and $insert_count data operations"
    
    # Try to restore to test database (if PostgreSQL is available)
    if docker ps --format "{{.Names}}" | grep -q "sutazai-postgres-minimal"; then
        log "INFO" "Testing database restoration..."
        
        # Create test database
        if docker exec sutazai-postgres-minimal createdb -U sutazai "$test_db_name" >/dev/null 2>&1; then
            # Try to restore backup
            if timeout $MAX_VERIFICATION_TIME docker exec -i sutazai-postgres-minimal psql -U sutazai -d "$test_db_name" < "$temp_sql_file" >/dev/null 2>&1; then
                log "SUCCESS" "Database backup restoration test passed"
                
                # Get some statistics from restored database
                local restored_tables=$(docker exec sutazai-postgres-minimal psql -U sutazai -d "$test_db_name" -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>/dev/null | xargs || echo 0)
                log "INFO" "Restored database contains $restored_tables tables"
            else
                log "ERROR" "Database backup restoration test failed: $backup_file"
                docker exec sutazai-postgres-minimal dropdb -U sutazai "$test_db_name" >/dev/null 2>&1 || true
                rm -f "$temp_sql_file"
                return 1
            fi
            
            # Clean up test database
            docker exec sutazai-postgres-minimal dropdb -U sutazai "$test_db_name" >/dev/null 2>&1 || true
        else
            log "WARN" "Could not create test database for restoration test"
        fi
    else
        log "WARN" "PostgreSQL not available for restoration test"
    fi
    
    # Clean up
    rm -f "$temp_sql_file"
    
    log "SUCCESS" "Database backup verification completed: $(basename "$backup_file")"
    return 0
}

# Verify configuration backup content
verify_config_backup() {
    local backup_file="$1"
    
    log "INFO" "Verifying configuration backup content: $(basename "$backup_file")"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would verify config backup: $backup_file"
        return 0
    fi
    
    # Test archive extraction
    local temp_dir="/tmp/config_test_$TIMESTAMP"
    mkdir -p "$temp_dir"
    
    if timeout $MAX_VERIFICATION_TIME tar -xzf "$backup_file" -C "$temp_dir" >/dev/null 2>&1; then
        local file_count=$(find "$temp_dir" -type f | wc -l)
        local total_size=$(du -sh "$temp_dir" | cut -f1)
        
        log "SUCCESS" "Configuration backup extraction test passed: $file_count files, $total_size total"
        
        # Check for expected configuration files
        local expected_configs=("docker-compose.yml" "config" ".env")
        local found_configs=0
        
        for config in "${expected_configs[@]}"; do
            if find "$temp_dir" -name "$config" -type f >/dev/null 2>&1; then
                ((found_configs++))
                log "INFO" "Found expected config: $config"
            fi
        done
        
        if [[ $found_configs -gt 0 ]]; then
            log "SUCCESS" "Configuration backup contains $found_configs expected configuration files"
        else
            log "WARN" "Configuration backup does not contain expected configuration files"
        fi
    else
        log "ERROR" "Configuration backup extraction test failed: $backup_file"
        rm -rf "$temp_dir"
        return 1
    fi
    
    # Clean up
    rm -rf "$temp_dir"
    
    log "SUCCESS" "Configuration backup verification completed: $(basename "$backup_file")"
    return 0
}

# Verify certificate backup content
verify_certificate_backup() {
    local backup_file="$1"
    
    log "INFO" "Verifying certificate backup: $(basename "$backup_file")"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would verify certificate backup: $backup_file"
        return 0
    fi
    
    # Check certificate validity
    if openssl x509 -in "$backup_file" -noout -text >/dev/null 2>&1; then
        local subject=$(openssl x509 -in "$backup_file" -noout -subject 2>/dev/null | cut -d= -f2-)
        local expiry=$(openssl x509 -in "$backup_file" -noout -enddate 2>/dev/null | cut -d= -f2)
        local expiry_epoch=$(date -d "$expiry" +%s 2>/dev/null || echo 0)
        local current_epoch=$(date +%s)
        local days_until_expiry=$(( (expiry_epoch - current_epoch) / 86400 ))
        
        log "SUCCESS" "Certificate backup is valid: $subject"
        log "INFO" "Certificate expires in $days_until_expiry days: $expiry"
        
        if [[ $days_until_expiry -lt 30 ]]; then
            log "WARN" "Certificate in backup expires soon ($days_until_expiry days)"
        fi
    else
        log "ERROR" "Certificate backup is invalid or corrupted: $backup_file"
        return 1
    fi
    
    log "SUCCESS" "Certificate backup verification completed: $(basename "$backup_file")"
    return 0
}

# Check backup completeness
check_backup_completeness() {
    local backup_files="$1"
    
    log "INFO" "Checking backup completeness..."
    
    local completeness_issues=0
    local recommendations=()
    
    # Check if we have recent database backups
    local db_backup_count=$(echo "$backup_files" | jq '.database | length')
    if [[ $db_backup_count -eq 0 ]]; then
        log "ERROR" "No database backups found"
        ((completeness_issues++))
        recommendations+=("Create database backups regularly")
    else
        # Check age of most recent database backup
        local most_recent_db=""
        local newest_timestamp=0
        
        while IFS= read -r backup; do
            if [[ -n "$backup" ]]; then
                local backup_timestamp=$(stat -c%Y "$backup" 2>/dev/null || echo 0)
                if [[ $backup_timestamp -gt $newest_timestamp ]]; then
                    newest_timestamp=$backup_timestamp
                    most_recent_db="$backup"
                fi
            fi
        done < <(echo "$backup_files" | jq -r '.database[]')
        
        if [[ -n "$most_recent_db" ]]; then
            local db_age_days=$(( ($(date +%s) - $newest_timestamp) / 86400 ))
            if [[ $db_age_days -gt $MAX_BACKUP_AGE_DAYS ]]; then
                log "WARN" "Most recent database backup is $db_age_days days old"
                ((completeness_issues++))
                recommendations+=("Create more frequent database backups")
            else
                log "SUCCESS" "Recent database backup found ($(basename "$most_recent_db"), $db_age_days days old)"
            fi
        fi
    fi
    
    # Check configuration backups
    local config_backup_count=$(echo "$backup_files" | jq '.config | length')
    if [[ $config_backup_count -eq 0 ]]; then
        log "WARN" "No configuration backups found"
        recommendations+=("Consider backing up configuration files")
    else
        log "SUCCESS" "Found $config_backup_count configuration backups"
    fi
    
    # Check certificate backups
    local cert_backup_count=$(echo "$backup_files" | jq '.certificates | length')
    if [[ $cert_backup_count -eq 0 ]]; then
        log "WARN" "No certificate backups found"
        recommendations+=("Consider backing up SSL certificates")
    else
        log "SUCCESS" "Found $cert_backup_count certificate backups"
    fi
    
    # Check system backups
    local system_backup_count=$(echo "$backup_files" | jq '.system | length')
    if [[ $system_backup_count -eq 0 ]]; then
        log "INFO" "No system backups found (optional)"
    else
        log "SUCCESS" "Found $system_backup_count system backups"
    fi
    
    # Check backup storage capacity
    if [[ -d "$BACKUP_DIR" ]]; then
        local backup_size=$(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1 || echo "0")
        local available_space=$(df -h "$BACKUP_DIR" | tail -1 | awk '{print $4}')
        log "INFO" "Backup storage usage: $backup_size used, $available_space available"
        
        # Check if backup directory is getting full
        local usage_percent=$(df "$BACKUP_DIR" | tail -1 | awk '{print $5}' | sed 's/%//')
        if [[ $usage_percent -gt 90 ]]; then
            log "ERROR" "Backup storage is $usage_percent% full"
            ((completeness_issues++))
            recommendations+=("Free up backup storage space")
        elif [[ $usage_percent -gt 80 ]]; then
            log "WARN" "Backup storage is $usage_percent% full"
            recommendations+=("Monitor backup storage space")
        fi
    fi
    
    # Create completeness summary
    local completeness_summary=$(jq -n \
        --arg issues "$completeness_issues" \
        --arg db_backups "$db_backup_count" \
        --arg config_backups "$config_backup_count" \
        --arg cert_backups "$cert_backup_count" \
        --arg system_backups "$system_backup_count" \
        --argjson recommendations "$(printf '%s\n' "${recommendations[@]}" | jq -R . | jq -s . 2>/dev/null || echo '[]')" \
        '{
            "completeness_issues": ($issues | tonumber),
            "backup_counts": {
                "database": ($db_backups | tonumber),
                "config": ($config_backups | tonumber),
                "certificates": ($cert_backups | tonumber),
                "system": ($system_backups | tonumber)
            },
            "recommendations": $recommendations
        }')
    
    if [[ $completeness_issues -eq 0 ]]; then
        log "SUCCESS" "Backup completeness check passed"
    else
        log "WARN" "Backup completeness check found $completeness_issues issues"
    fi
    
    echo "$completeness_summary"
}

# Generate backup verification report
generate_verification_report() {
    local backup_files="$1"
    local verification_results="$2"
    local completeness_summary="$3"
    
    local report_file="$LOG_DIR/backup_verification_report_$TIMESTAMP.json"
    
    log "INFO" "Generating backup verification report..."
    
    jq -n \
        --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --arg hostname "$(hostname)" \
        --arg verification_type "$([ "$VERIFY_ALL" == "true" ] && echo "comprehensive" || echo "standard")" \
        --argjson backup_files "$backup_files" \
        --argjson verification_results "$verification_results" \
        --argjson completeness "$completeness_summary" \
        '{
            "verification_info": {
                "timestamp": $timestamp,
                "hostname": $hostname,
                "verification_type": $verification_type,
                "dry_run": '"$DRY_RUN"'
            },
            "backup_files": $backup_files,
            "verification_results": $verification_results,
            "completeness_summary": $completeness,
            "next_verification": "'"$(date -d '+1 day' -u +%Y-%m-%dT%H:%M:%SZ)"'"
        }' > "$report_file"
    
    log "SUCCESS" "Backup verification report saved to: $report_file"
    
    # Create symlink to latest report
    if [[ "$DRY_RUN" == "false" ]]; then
        ln -sf "$report_file" "$LOG_DIR/latest_backup_verification_report.json"
    fi
    
    echo "$report_file"
}

# Main execution
main() {
    log "INFO" "Starting backup verification for SutazAI system"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "Running in DRY RUN mode - no actual verifications will be performed"
    fi
    
    if [[ "$VERIFY_ALL" == "true" ]]; then
        log "INFO" "Comprehensive verification mode enabled"
    fi
    
    # Setup directories
    setup_verification_directories
    
    # Find all backup files
    local backup_files=$(find_backup_files)
    
    # Initialize verification results
    local verification_results='{"verified_files": [], "failed_files": [], "total_verified": 0, "total_failed": 0}'
    
    # Verify database backups
    log "INFO" "Verifying database backups..."
    while IFS= read -r backup_file; do
        if [[ -n "$backup_file" ]]; then
            if check_file_integrity "$backup_file" "database"; then
                if [[ "$VERIFY_ALL" == "true" ]]; then
                    if verify_database_backup "$backup_file"; then
                        verification_results=$(echo "$verification_results" | jq --arg file "$backup_file" '.verified_files += [$file] | .total_verified += 1')
                    else
                        verification_results=$(echo "$verification_results" | jq --arg file "$backup_file" '.failed_files += [$file] | .total_failed += 1')
                    fi
                else
                    verification_results=$(echo "$verification_results" | jq --arg file "$backup_file" '.verified_files += [$file] | .total_verified += 1')
                fi
            else
                verification_results=$(echo "$verification_results" | jq --arg file "$backup_file" '.failed_files += [$file] | .total_failed += 1')
            fi
        fi
    done < <(echo "$backup_files" | jq -r '.database[]' 2>/dev/null)
    
    # Verify configuration backups
    log "INFO" "Verifying configuration backups..."
    while IFS= read -r backup_file; do
        if [[ -n "$backup_file" ]]; then
            if check_file_integrity "$backup_file" "config"; then
                if [[ "$VERIFY_ALL" == "true" ]]; then
                    if verify_config_backup "$backup_file"; then
                        verification_results=$(echo "$verification_results" | jq --arg file "$backup_file" '.verified_files += [$file] | .total_verified += 1')
                    else
                        verification_results=$(echo "$verification_results" | jq --arg file "$backup_file" '.failed_files += [$file] | .total_failed += 1')
                    fi
                else
                    verification_results=$(echo "$verification_results" | jq --arg file "$backup_file" '.verified_files += [$file] | .total_verified += 1')
                fi
            else
                verification_results=$(echo "$verification_results" | jq --arg file "$backup_file" '.failed_files += [$file] | .total_failed += 1')
            fi
        fi
    done < <(echo "$backup_files" | jq -r '.config[]' 2>/dev/null)
    
    # Verify certificate backups
    log "INFO" "Verifying certificate backups..."
    while IFS= read -r backup_file; do
        if [[ -n "$backup_file" ]]; then
            if check_file_integrity "$backup_file" "certificates"; then
                if verify_certificate_backup "$backup_file"; then
                    verification_results=$(echo "$verification_results" | jq --arg file "$backup_file" '.verified_files += [$file] | .total_verified += 1')
                else
                    verification_results=$(echo "$verification_results" | jq --arg file "$backup_file" '.failed_files += [$file] | .total_failed += 1')
                fi
            else
                verification_results=$(echo "$verification_results" | jq --arg file "$backup_file" '.failed_files += [$file] | .total_failed += 1')
            fi
        fi
    done < <(echo "$backup_files" | jq -r '.certificates[]' 2>/dev/null)
    
    # Verify system backups (basic integrity only)
    log "INFO" "Verifying system backups..."
    while IFS= read -r backup_file; do
        if [[ -n "$backup_file" ]]; then
            if check_file_integrity "$backup_file" "system"; then
                verification_results=$(echo "$verification_results" | jq --arg file "$backup_file" '.verified_files += [$file] | .total_verified += 1')
            else
                verification_results=$(echo "$verification_results" | jq --arg file "$backup_file" '.failed_files += [$file] | .total_failed += 1')
            fi
        fi
    done < <(echo "$backup_files" | jq -r '.system[]' 2>/dev/null)
    
    # Check backup completeness
    local completeness_summary=$(check_backup_completeness "$backup_files")
    
    # Generate verification report
    local report_file=$(generate_verification_report "$backup_files" "$verification_results" "$completeness_summary")
    
    log "SUCCESS" "Backup verification completed"
    
    # Show summary
    local total_verified=$(echo "$verification_results" | jq -r '.total_verified')
    local total_failed=$(echo "$verification_results" | jq -r '.total_failed')
    local completeness_issues=$(echo "$completeness_summary" | jq -r '.completeness_issues')
    
    echo
    echo "============================================"
    echo "       BACKUP VERIFICATION SUMMARY"
    echo "============================================"
    echo "Mode: $([ "$DRY_RUN" == "true" ] && echo "DRY RUN" || echo "ACTUAL VERIFICATION")"
    echo "Verification Type: $([ "$VERIFY_ALL" == "true" ] && echo "COMPREHENSIVE" || echo "STANDARD")"
    echo "Verified Backups: $total_verified"
    echo "Failed Verifications: $total_failed"
    echo "Completeness Issues: $completeness_issues"
    echo "Report: $report_file"
    echo "Timestamp: $(date)"
    echo "============================================"
    
    # Exit with appropriate code
    if [[ $total_failed -gt 0 ]]; then
        exit 2  # Verification failures
    elif [[ $completeness_issues -gt 0 ]]; then
        exit 1  # Completeness issues
    else
        exit 0  # All good
    fi
}

# Run main function
main "$@"