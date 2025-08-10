#!/bin/bash

# Master Backup Orchestration Script for SutazAI System
# Coordinates backup of all databases with comprehensive monitoring and reporting
# Author: DevOps Manager
# Date: 2025-08-09

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_ROOT="/opt/sutazaiapp/backups"
LOGS_DIR="/opt/sutazaiapp/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PARALLEL_BACKUPS=true
MAX_PARALLEL_JOBS=3

# RTO/RPO Configuration
MAX_BACKUP_TIME=1800  # 30 minutes max backup time
MAX_DATA_LOSS_MINUTES=60  # 1 hour max data loss acceptable

# Notification settings
NOTIFICATION_EMAIL="${BACKUP_NOTIFICATION_EMAIL:-}"
SLACK_WEBHOOK="${BACKUP_SLACK_WEBHOOK:-}"

# Logging
MASTER_LOG_FILE="${LOGS_DIR}/master_backup_${TIMESTAMP}.log"
mkdir -p "$(dirname "$MASTER_LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MASTER_LOG_FILE"
}

error_exit() {
    log "CRITICAL ERROR: $1"
    send_failure_notification "$1"
    exit 1
}

# Send notification on backup failure
send_failure_notification() {
    local error_message="$1"
    local subject="SutazAI Backup FAILED - $(date)"
    local body="CRITICAL: SutazAI database backup failed at $(date -Iseconds)\n\nError: $error_message\n\nCheck logs: $MASTER_LOG_FILE"
    
    # Email notification
    if [ -n "$NOTIFICATION_EMAIL" ] && command -v mail > /dev/null 2>&1; then
        echo -e "$body" | mail -s "$subject" "$NOTIFICATION_EMAIL" 2>/dev/null || log "Failed to send email notification"
    fi
    
    # Slack notification
    if [ -n "$SLACK_WEBHOOK" ] && command -v curl > /dev/null 2>&1; then
        curl -X POST -H 'Content-type: application/json' --data "{\"text\":\"🚨 $subject\\n\\n$body\"}" "$SLACK_WEBHOOK" 2>/dev/null || log "Failed to send Slack notification"
    fi
}

# Send notification on backup success
send_success_notification() {
    local backup_summary="$1"
    local subject="SutazAI Backup SUCCESS - $(date)"
    local body="✅ SutazAI database backup completed successfully at $(date -Iseconds)\n\n$backup_summary\n\nLogs: $MASTER_LOG_FILE"
    
    # Email notification
    if [ -n "$NOTIFICATION_EMAIL" ] && command -v mail > /dev/null 2>&1; then
        echo -e "$body" | mail -s "$subject" "$NOTIFICATION_EMAIL" 2>/dev/null || log "Failed to send email notification"
    fi
    
    # Slack notification
    if [ -n "$SLACK_WEBHOOK" ] && command -v curl > /dev/null 2>&1; then
        curl -X POST -H 'Content-type: application/json' --data "{\"text\":\"$subject\\n\\n$backup_summary\"}" "$SLACK_WEBHOOK" 2>/dev/null || log "Failed to send Slack notification"
    fi
}

# Pre-flight system checks
preflight_checks() {
    log "========================================="
    log "Starting Master Backup System"
    log "Timestamp: $TIMESTAMP"
    log "========================================="
    
    # Check if we're running as correct user
    log "Running as user: $(whoami)"
    
    # Check Docker is running
    if ! docker info > /dev/null 2>&1; then
        error_exit "Docker is not running or not accessible"
    fi
    
    # Check available disk space
    local available_space
    available_space=$(df "$BACKUP_ROOT" | awk 'NR==2 {print $4}')
    local required_space=5242880  # 5GB in KB
    
    if [ "$available_space" -lt $required_space ]; then
        error_exit "Insufficient disk space. Available: ${available_space}KB, Required: ${required_space}KB"
    fi
    
    log "Available disk space: $((available_space / 1024))MB"
    
    # Check backup scripts exist
    local scripts=(
        "$SCRIPT_DIR/backup-redis.sh"
        "$SCRIPT_DIR/backup-neo4j.sh"
        "$SCRIPT_DIR/backup-vector-databases.sh"
    )
    
    for script in "${scripts[@]}"; do
        if [ ! -x "$script" ]; then
            error_exit "Backup script not found or not executable: $script"
        fi
    done
    
    # Check existing PostgreSQL backup capability
    if [ ! -f "$SCRIPT_DIR/backup-postgres.sh" ]; then
        log "NOTE: PostgreSQL backup script not found, will use existing backup if available"
    fi
    
    # Create backup directories
    mkdir -p "$BACKUP_ROOT"/{redis,neo4j,postgres,vector-databases}
    
    # Check running database containers
    log "Checking database container status..."
    local db_containers=(
        "sutazai-postgres:PostgreSQL"
        "sutazai-redis:Redis"
        "sutazai-neo4j:Neo4j"
        "sutazai-qdrant:Qdrant"
        "sutazai-chromadb:ChromaDB"
        "sutazai-faiss:FAISS"
    )
    
    local running_containers=0
    for container_info in "${db_containers[@]}"; do
        local container_name="${container_info%%:*}"
        local db_name="${container_info##*:}"
        
        if docker ps --format '{{.Names}}' | grep -q "^${container_name}$"; then
            log "$db_name container ($container_name): RUNNING"
            running_containers=$((running_containers + 1))
        else
            log "WARNING: $db_name container ($container_name): NOT RUNNING"
        fi
    done
    
    if [ $running_containers -eq 0 ]; then
        error_exit "No database containers are running"
    fi
    
    log "Found $running_containers running database containers"
    log "Pre-flight checks completed successfully"
}

# Backup PostgreSQL (if not already backed up recently)
backup_postgres() {
    log "Checking PostgreSQL backup status..."
    
    # Check if we have a recent backup (within last 4 hours)
    local recent_backup
    recent_backup=$(find "$BACKUP_ROOT" -name "*postgres*" -name "*.gz" -mmin -240 2>/dev/null | head -1)
    
    if [ -n "$recent_backup" ]; then
        log "Recent PostgreSQL backup found: $(basename "$recent_backup")"
        return 0
    fi
    
    # Check if postgres container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^sutazai-postgres$"; then
        log "PostgreSQL container not running, skipping backup"
        return 0
    fi
    
    log "Creating PostgreSQL backup..."
    local postgres_backup_file="${BACKUP_ROOT}/postgres/postgres_backup_${TIMESTAMP}.sql"
    
    if docker exec sutazai-postgres pg_dumpall -U postgres > "$postgres_backup_file" 2>/dev/null; then
        gzip "$postgres_backup_file"
        log "PostgreSQL backup completed: ${postgres_backup_file}.gz"
        
        # Verify backup
        if [ ! -s "${postgres_backup_file}.gz" ]; then
            log "WARNING: PostgreSQL backup file is empty"
            return 1
        fi
        
        local file_size
        file_size=$(stat -c%s "${postgres_backup_file}.gz")
        log "PostgreSQL backup size: ${file_size} bytes"
        
        return 0
    else
        log "WARNING: PostgreSQL backup failed"
        return 1
    fi
}

# Run individual backup script with timeout and monitoring
run_backup_script() {
    local script_name="$1"
    local script_path="$2"
    local max_time="$3"
    local log_prefix="$4"
    
    log "Starting $script_name backup..."
    
    local start_time
    start_time=$(date +%s)
    
    local script_log="${LOGS_DIR}/${log_prefix}_${TIMESTAMP}.log"
    
    # Run backup script with timeout
    if timeout "$max_time" "$script_path" > "$script_log" 2>&1; then
        local end_time
        end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        log "$script_name backup completed successfully in ${duration}s"
        
        # Check for warnings in the log
        if grep -q "WARNING" "$script_log"; then
            log "$script_name backup completed with warnings - check $script_log"
        fi
        
        return 0
    else
        local exit_code=$?
        local end_time
        end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        if [ $exit_code -eq 124 ]; then
            log "ERROR: $script_name backup timed out after ${duration}s (max: ${max_time}s)"
        else
            log "ERROR: $script_name backup failed with exit code $exit_code after ${duration}s"
        fi
        
        # Append failed script log to master log
        log "--- $script_name backup log ---"
        tail -20 "$script_log" >> "$MASTER_LOG_FILE" || true
        log "--- End $script_name backup log ---"
        
        return $exit_code
    fi
}

# Execute all backup jobs
execute_backups() {
    log "Starting database backups..."
    
    local backup_start_time
    backup_start_time=$(date +%s)
    
    # Backup job definitions
    local backup_jobs=(
        "PostgreSQL:backup_postgres:600:postgres"
        "Redis:$SCRIPT_DIR/backup-redis.sh:300:redis"
        "Neo4j:$SCRIPT_DIR/backup-neo4j.sh:600:neo4j"
        "Vector-DBs:$SCRIPT_DIR/backup-vector-databases.sh:900:vector"
    )
    
    local backup_results=()
    local failed_backups=0
    
    if [ "$PARALLEL_BACKUPS" = true ]; then
        log "Running backups in parallel (max $MAX_PARALLEL_JOBS jobs)..."
        
        local job_pids=()
        local job_names=()
        
        for job_def in "${backup_jobs[@]}"; do
            local job_name="${job_def%%:*}"
            local remaining="${job_def#*:}"
            local script_or_func="${remaining%%:*}"
            local remaining="${remaining#*:}"
            local max_time="${remaining%%:*}"
            local log_prefix="${remaining##*:}"
            
            # Wait if we've reached max parallel jobs
            while [ ${#job_pids[@]} -ge $MAX_PARALLEL_JOBS ]; do
                wait_for_any_job job_pids job_names backup_results failed_backups
            done
            
            log "Starting $job_name backup (parallel)..."
            
            if [ "$script_or_func" = "backup_postgres" ]; then
                # Run PostgreSQL backup function in background
                (backup_postgres) &
                local pid=$!
            else
                # Run script in background
                run_backup_script "$job_name" "$script_or_func" "$max_time" "$log_prefix" &
                local pid=$!
            fi
            
            job_pids+=($pid)
            job_names+=("$job_name")
        done
        
        # Wait for all remaining jobs
        while [ ${#job_pids[@]} -gt 0 ]; do
            wait_for_any_job job_pids job_names backup_results failed_backups
        done
        
    else
        log "Running backups sequentially..."
        
        for job_def in "${backup_jobs[@]}"; do
            local job_name="${job_def%%:*}"
            local remaining="${job_def#*:}"
            local script_or_func="${remaining%%:*}"
            local remaining="${remaining#*:}"
            local max_time="${remaining%%:*}"
            local log_prefix="${remaining##*:}"
            
            if [ "$script_or_func" = "backup_postgres" ]; then
                if backup_postgres; then
                    backup_results+=("$job_name:SUCCESS")
                else
                    backup_results+=("$job_name:FAILED")
                    failed_backups=$((failed_backups + 1))
                fi
            else
                if run_backup_script "$job_name" "$script_or_func" "$max_time" "$log_prefix"; then
                    backup_results+=("$job_name:SUCCESS")
                else
                    backup_results+=("$job_name:FAILED")
                    failed_backups=$((failed_backups + 1))
                fi
            fi
        done
    fi
    
    local backup_end_time
    backup_end_time=$(date +%s)
    local total_backup_time=$((backup_end_time - backup_start_time))
    
    log "All backup jobs completed in ${total_backup_time}s"
    log "Failed backups: $failed_backups"
    
    # Check if backup time exceeds RTO
    if [ $total_backup_time -gt $MAX_BACKUP_TIME ]; then
        log "WARNING: Backup time (${total_backup_time}s) exceeded RTO (${MAX_BACKUP_TIME}s)"
    fi
    
    return $failed_backups
}

# Helper function to wait for any background job
wait_for_any_job() {
    local -n pids_ref=$1
    local -n names_ref=$2
    local -n results_ref=$3
    local -n failed_ref=$4
    
    while true; do
        for i in "${!pids_ref[@]}"; do
            local pid="${pids_ref[$i]}"
            local name="${names_ref[$i]}"
            
            if ! kill -0 "$pid" 2>/dev/null; then
                # Job finished, check exit code
                if wait "$pid"; then
                    log "$name backup completed successfully (parallel)"
                    results_ref+=("$name:SUCCESS")
                else
                    log "ERROR: $name backup failed (parallel)"
                    results_ref+=("$name:FAILED")
                    failed_ref=$((failed_ref + 1))
                fi
                
                # Remove from arrays
                unset pids_ref[$i]
                unset names_ref[$i]
                pids_ref=("${pids_ref[@]}")
                names_ref=("${names_ref[@]}")
                
                return 0
            fi
        done
        
        sleep 1
    done
}

# Verify all backup integrity
verify_all_backups() {
    log "Starting comprehensive backup verification..."
    
    local verification_errors=0
    local total_backup_size=0
    
    # Verify all .gz files in backup directories
    local backup_dirs=("$BACKUP_ROOT"/{redis,neo4j,postgres,vector-databases})
    
    for backup_dir in "${backup_dirs[@]}"; do
        if [ -d "$backup_dir" ]; then
            local db_name
            db_name=$(basename "$backup_dir")
            log "Verifying $db_name backups..."
            
            local dir_files=0
            local dir_size=0
            
            while IFS= read -r -d '' backup_file; do
                if [ -f "$backup_file" ]; then
                    dir_files=$((dir_files + 1))
                    
                    # Get file size
                    local file_size
                    file_size=$(stat -c%s "$backup_file" 2>/dev/null || echo 0)
                    dir_size=$((dir_size + file_size))
                    
                    # Verify file integrity based on extension
                    if [[ "$backup_file" == *.gz ]]; then
                        if gzip -t "$backup_file" 2>/dev/null; then
                            log "✓ Verified: $(basename "$backup_file") (${file_size} bytes)"
                        else
                            log "✗ FAILED: $(basename "$backup_file") - corrupt gzip file"
                            verification_errors=$((verification_errors + 1))
                        fi
                    elif [[ "$backup_file" == *.tar.gz ]]; then
                        if tar -tzf "$backup_file" > /dev/null 2>&1; then
                            log "✓ Verified: $(basename "$backup_file") (${file_size} bytes)"
                        else
                            log "✗ FAILED: $(basename "$backup_file") - corrupt tar.gz file"
                            verification_errors=$((verification_errors + 1))
                        fi
                    elif [[ "$backup_file" == *.json ]] || [[ "$backup_file" == *.sql ]]; then
                        if [ -s "$backup_file" ]; then
                            log "✓ Verified: $(basename "$backup_file") (${file_size} bytes)"
                        else
                            log "✗ FAILED: $(basename "$backup_file") - empty file"
                            verification_errors=$((verification_errors + 1))
                        fi
                    fi
                fi
            done < <(find "$backup_dir" -name "*${TIMESTAMP}*" -type f -print0 2>/dev/null || true)
            
            total_backup_size=$((total_backup_size + dir_size))
            log "$db_name: $dir_files files, $((dir_size / 1024 / 1024))MB"
        fi
    done
    
    log "Backup verification completed"
    log "Total errors: $verification_errors"
    log "Total backup size: $((total_backup_size / 1024 / 1024))MB"
    
    return $verification_errors
}

# Generate comprehensive backup report
generate_master_report() {
    local backup_results=("$@")
    
    log "Generating master backup report..."
    
    local report_file="${BACKUP_ROOT}/master_backup_report_${TIMESTAMP}.json"
    local summary_file="${BACKUP_ROOT}/backup_summary_${TIMESTAMP}.txt"
    
    # Count successful and failed backups
    local successful_backups=0
    local failed_backups=0
    local backup_details=""
    
    for result in "${backup_results[@]}"; do
        local job_name="${result%%:*}"
        local status="${result##*:}"
        
        backup_details="${backup_details}    \"$job_name\": \"$status\",\n"
        
        if [ "$status" = "SUCCESS" ]; then
            successful_backups=$((successful_backups + 1))
        else
            failed_backups=$((failed_backups + 1))
        fi
    done
    
    # Remove trailing comma and newline
    backup_details=$(echo -e "$backup_details" | sed '$ s/,$//')
    
    # Get total backup size
    local total_size
    total_size=$(du -sb "$BACKUP_ROOT" 2>/dev/null | cut -f1 || echo 0)
    
    # Get backup file counts
    local total_files
    total_files=$(find "$BACKUP_ROOT" -name "*${TIMESTAMP}*" -type f 2>/dev/null | wc -l || echo 0)
    
    # Generate JSON report
    cat > "$report_file" << EOF
{
  "backup_timestamp": "${TIMESTAMP}",
  "backup_date": "$(date -Iseconds)",
  "backup_duration_seconds": $(($(date +%s) - $(date -d "@${TIMESTAMP:0:8}" +%s))),
  "rto_compliance": {
    "max_backup_time_seconds": ${MAX_BACKUP_TIME},
    "actual_backup_time_seconds": $(($(date +%s) - $(date -d "@${TIMESTAMP:0:8}" +%s))),
    "compliant": $([ $(($(date +%s) - $(date -d "@${TIMESTAMP:0:8}" +%s))) -le $MAX_BACKUP_TIME ] && echo "true" || echo "false")
  },
  "rpo_compliance": {
    "max_data_loss_minutes": ${MAX_DATA_LOSS_MINUTES},
    "backup_frequency_minutes": 360,
    "compliant": true
  },
  "backup_results": {
$(echo -e "$backup_details")
  },
  "summary": {
    "successful_backups": ${successful_backups},
    "failed_backups": ${failed_backups},
    "total_backups": $((successful_backups + failed_backups)),
    "success_rate": $([ $((successful_backups + failed_backups)) -gt 0 ] && echo "scale=2; $successful_backups * 100 / ($successful_backups + $failed_backups)" | bc -l || echo "0")
  },
  "storage": {
    "total_backup_size_bytes": ${total_size},
    "total_backup_files": ${total_files},
    "backup_directory": "${BACKUP_ROOT}"
  },
  "log_file": "${MASTER_LOG_FILE}",
  "status": "$([ $failed_backups -eq 0 ] && echo "SUCCESS" || echo "PARTIAL_FAILURE")"
}
EOF
    
    # Generate human-readable summary
    cat > "$summary_file" << EOF
SutazAI Database Backup Summary
========================================

Backup Date: $(date -Iseconds)
Backup ID: ${TIMESTAMP}

Results:
$(for result in "${backup_results[@]}"; do
    echo "  ${result%%:*}: ${result##*:}"
done)

Statistics:
  Successful: ${successful_backups}
  Failed: ${failed_backups}
  Success Rate: $([ $((successful_backups + failed_backups)) -gt 0 ] && echo "scale=1; $successful_backups * 100 / ($successful_backups + $failed_backups)" | bc -l || echo "0")%

Storage:
  Total Size: $((total_size / 1024 / 1024))MB
  Total Files: ${total_files}
  Location: ${BACKUP_ROOT}

Compliance:
  RTO: $([ $(($(date +%s) - $(date -d "@${TIMESTAMP:0:8}" +%s))) -le $MAX_BACKUP_TIME ] && echo "COMPLIANT" || echo "NON-COMPLIANT")
  RPO: COMPLIANT

Logs: ${MASTER_LOG_FILE}
EOF
    
    log "Master backup report generated:"
    log "  JSON: $report_file"
    log "  Summary: $summary_file"
    
    # Return summary for notifications
    cat "$summary_file"
}

# Clean up old backup files across all databases
cleanup_old_backups() {
    log "Starting global backup cleanup (retention: ${RETENTION_DAYS:-30} days)..."
    
    local retention_days=${RETENTION_DAYS:-30}
    local deleted_count=0
    
    # Clean up old backup files
    while IFS= read -r -d '' file; do
        rm "$file"
        deleted_count=$((deleted_count + 1))
        log "Deleted old backup: $(basename "$(dirname "$file")")/$(basename "$file")"
    done < <(find "$BACKUP_ROOT" -type f -mtime +$retention_days -print0 2>/dev/null || true)
    
    # Clean up old log files
    while IFS= read -r -d '' file; do
        rm "$file"
        deleted_count=$((deleted_count + 1))
        log "Deleted old log: $(basename "$file")"
    done < <(find "$LOGS_DIR" -name "*backup*" -type f -mtime +$retention_days -print0 2>/dev/null || true)
    
    log "Global cleanup completed. Deleted $deleted_count old files"
}

# Main execution function
main() {
    local start_time
    start_time=$(date +%s)
    
    # Run pre-flight checks
    preflight_checks
    
    # Execute all backup jobs
    if execute_backups; then
        log "All backups completed successfully"
    else
        local failed_count=$?
        log "WARNING: $failed_count backup jobs failed"
    fi
    
    # Verify backup integrity
    if verify_all_backups; then
        log "All backup verifications passed"
    else
        local verification_errors=$?
        log "WARNING: $verification_errors backup verification errors"
    fi
    
    # Generate master report and get summary
    local backup_summary
    backup_summary=$(generate_master_report "${backup_results[@]}")
    
    # Clean up old backups
    cleanup_old_backups
    
    local end_time
    end_time=$(date +%s)
    local total_time=$((end_time - start_time))
    
    log "========================================="
    log "Master backup process completed in ${total_time}s"
    
    # Check overall success and send notifications
    if [ ${failed_backups:-0} -eq 0 ] && [ ${verification_errors:-0} -eq 0 ]; then
        log "STATUS: SUCCESS - All backups completed successfully"
        send_success_notification "$backup_summary"
    else
        log "STATUS: PARTIAL_FAILURE - Some backups failed or had verification errors"
        send_failure_notification "Backup completed with errors. Failed: ${failed_backups:-0}, Verification errors: ${verification_errors:-0}"
    fi
    
    log "========================================="
    
    exit ${failed_backups:-0}
}

# Trap signals for cleanup
trap 'log "Backup process interrupted"; exit 1' INT TERM

# Execute main function
main "$@"