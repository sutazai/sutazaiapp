#!/bin/bash
# Purpose: Automated log rotation and cleanup for SutazAI system
# Usage: ./log-rotation-cleanup.sh [--dry-run] [--force]
# Requires: Standard Unix tools (find, gzip, etc.)

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
LOG_DIR="$BASE_DIR/logs"
ARCHIVE_DIR="$BASE_DIR/archive/logs"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Configuration
DRY_RUN=false
FORCE=false
MAX_LOG_SIZE_MB=100          # Archive logs larger than this
MAX_LOG_AGE_DAYS=7           # Archive logs older than this
DELETE_ARCHIVES_DAYS=30      # Delete archived logs older than this
MAX_TOTAL_LOG_SIZE_GB=5      # Total log directory size limit

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--force]"
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
    
    case $level in
        ERROR) echo -e "${RED}[$timestamp] ERROR: $message${NC}" ;;
        WARN) echo -e "${YELLOW}[$timestamp] WARN: $message${NC}" ;;
        INFO) echo -e "${BLUE}[$timestamp] INFO: $message${NC}" ;;
        SUCCESS) echo -e "${GREEN}[$timestamp] SUCCESS: $message${NC}" ;;
    esac
}

# Create necessary directories
setup_directories() {
    log "INFO" "Setting up directories..."
    
    if [[ "$DRY_RUN" == "false" ]]; then
        mkdir -p "$ARCHIVE_DIR"
        mkdir -p "$LOG_DIR/rotated"
    else
        log "INFO" "[DRY RUN] Would create directories: $ARCHIVE_DIR, $LOG_DIR/rotated"
    fi
}

# Get file size in MB
get_file_size_mb() {
    local file="$1"
    local size_bytes=$(stat -c%s "$file" 2>/dev/null || echo 0)
    echo $((size_bytes / 1024 / 1024))
}

# Get directory size in MB
get_dir_size_mb() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        local size_kb=$(du -sk "$dir" 2>/dev/null | cut -f1)
        echo $((size_kb / 1024))
    else
        echo 0
    fi
}

# Rotate large log files
rotate_large_logs() {
    log "INFO" "Checking for large log files (>${MAX_LOG_SIZE_MB}MB)..."
    
    local rotated_count=0
    local total_saved_mb=0
    
    while IFS= read -r -d '' logfile; do
        local size_mb=$(get_file_size_mb "$logfile")
        
        if [[ $size_mb -gt $MAX_LOG_SIZE_MB ]]; then
            local basename=$(basename "$logfile")
            local rotated_name="${basename%.log}_${TIMESTAMP}.log"
            local rotated_path="$LOG_DIR/rotated/$rotated_name"
            
            log "WARN" "Large log file found: $basename (${size_mb}MB)"
            
            if [[ "$DRY_RUN" == "false" ]]; then
                # Move and compress the log file
                mv "$logfile" "$rotated_path"
                gzip "$rotated_path"
                
                # Create new empty log file
                touch "$logfile"
                
                # Set appropriate permissions
                chmod 644 "$logfile"
                
                log "SUCCESS" "Rotated and compressed: $basename -> ${rotated_name}.gz"
            else
                log "INFO" "[DRY RUN] Would rotate: $basename (${size_mb}MB) -> ${rotated_name}.gz"
            fi
            
            ((rotated_count++))
            total_saved_mb=$((total_saved_mb + size_mb))
        fi
    done < <(find "$LOG_DIR" -maxdepth 1 -name "*.log" -type f -print0 2>/dev/null)
    
    if [[ $rotated_count -gt 0 ]]; then
        log "SUCCESS" "Rotated $rotated_count large log files, saved ${total_saved_mb}MB"
    else
        log "INFO" "No large log files found"
    fi
}

# Archive old log files
archive_old_logs() {
    log "INFO" "Checking for old log files (>${MAX_LOG_AGE_DAYS} days)..."
    
    local archived_count=0
    local total_archived_mb=0
    
    while IFS= read -r -d '' logfile; do
        local basename=$(basename "$logfile")
        local size_mb=$(get_file_size_mb "$logfile")
        local archived_name="${basename%.log}_archived_${TIMESTAMP}.log.gz"
        local archived_path="$ARCHIVE_DIR/$archived_name"
        
        log "INFO" "Archiving old log file: $basename (${size_mb}MB)"
        
        if [[ "$DRY_RUN" == "false" ]]; then
            # Compress and move to archive
            gzip -c "$logfile" > "$archived_path"
            rm "$logfile"
            
            log "SUCCESS" "Archived: $basename -> $archived_name"
        else
            log "INFO" "[DRY RUN] Would archive: $basename -> $archived_name"
        fi
        
        ((archived_count++))
        total_archived_mb=$((total_archived_mb + size_mb))
    done < <(find "$LOG_DIR" -maxdepth 1 -name "*.log" -type f -mtime +$MAX_LOG_AGE_DAYS -print0 2>/dev/null)
    
    if [[ $archived_count -gt 0 ]]; then
        log "SUCCESS" "Archived $archived_count old log files, totaling ${total_archived_mb}MB"
    else
        log "INFO" "No old log files found"
    fi
}

# Clean up Docker container logs
cleanup_docker_logs() {
    log "INFO" "Cleaning up Docker container logs..."
    
    if ! command -v docker >/dev/null 2>&1; then
        log "WARN" "Docker not available, skipping container log cleanup"
        return
    fi
    
    local cleaned_count=0
    
    # Get all SutazAI containers
    while IFS= read -r container; do
        if [[ -n "$container" ]]; then
            local log_size=$(docker logs --details "$container" 2>/dev/null | wc -c || echo 0)
            local log_size_mb=$((log_size / 1024 / 1024))
            
            if [[ $log_size_mb -gt 50 ]]; then  # Clean if larger than 50MB
                log "INFO" "Cleaning logs for container: $container (${log_size_mb}MB)"
                
                if [[ "$DRY_RUN" == "false" ]]; then
                    # Truncate container logs
                    local log_path=$(docker inspect --format='{{.LogPath}}' "$container" 2>/dev/null || echo "")
                    if [[ -n "$log_path" && -f "$log_path" ]]; then
                        echo "" > "$log_path"
                        log "SUCCESS" "Truncated logs for container: $container"
                    fi
                else
                    log "INFO" "[DRY RUN] Would clean logs for container: $container (${log_size_mb}MB)"
                fi
                
                ((cleaned_count++))
            fi
        fi
    done < <(docker ps --format "{{.Names}}" | grep "^sutazai-" || true)
    
    if [[ $cleaned_count -gt 0 ]]; then
        log "SUCCESS" "Cleaned logs for $cleaned_count Docker containers"
    else
        log "INFO" "No Docker container logs needed cleaning"
    fi
}

# Delete old archived logs
delete_old_archives() {
    log "INFO" "Checking for old archived logs (>${DELETE_ARCHIVES_DAYS} days)..."
    
    if [[ ! -d "$ARCHIVE_DIR" ]]; then
        log "INFO" "Archive directory does not exist, skipping"
        return
    fi
    
    local deleted_count=0
    local total_deleted_mb=0
    
    while IFS= read -r -d '' archive; do
        local basename=$(basename "$archive")
        local size_mb=$(get_file_size_mb "$archive")
        
        log "INFO" "Deleting old archive: $basename (${size_mb}MB)"
        
        if [[ "$DRY_RUN" == "false" ]]; then
            rm "$archive"
            log "SUCCESS" "Deleted: $basename"
        else
            log "INFO" "[DRY RUN] Would delete: $basename (${size_mb}MB)"
        fi
        
        ((deleted_count++))
        total_deleted_mb=$((total_deleted_mb + size_mb))
    done < <(find "$ARCHIVE_DIR" -name "*.log.gz" -type f -mtime +$DELETE_ARCHIVES_DAYS -print0 2>/dev/null)
    
    if [[ $deleted_count -gt 0 ]]; then
        log "SUCCESS" "Deleted $deleted_count old archives, freed ${total_deleted_mb}MB"
    else
        log "INFO" "No old archives found for deletion"
    fi
}

# Check total log directory size
check_total_size() {
    log "INFO" "Checking total log directory size..."
    
    local total_size_mb=$(get_dir_size_mb "$LOG_DIR")
    local max_size_mb=$((MAX_TOTAL_LOG_SIZE_GB * 1024))
    
    log "INFO" "Total log directory size: ${total_size_mb}MB (limit: ${max_size_mb}MB)"
    
    if [[ $total_size_mb -gt $max_size_mb ]]; then
        log "WARN" "Log directory size exceeds limit (${total_size_mb}MB > ${max_size_mb}MB)"
        
        if [[ "$FORCE" == "true" ]]; then
            log "WARN" "Force cleanup enabled, removing oldest files..."
            
            # Remove oldest files until under limit
            while IFS= read -r -d '' oldfile; do
                local size_mb=$(get_file_size_mb "$oldfile")
                local basename=$(basename "$oldfile")
                
                if [[ "$DRY_RUN" == "false" ]]; then
                    rm "$oldfile"
                    log "WARN" "Force deleted: $basename (${size_mb}MB)"
                else
                    log "INFO" "[DRY RUN] Would force delete: $basename (${size_mb}MB)"
                fi
                
                total_size_mb=$((total_size_mb - size_mb))
                
                if [[ $total_size_mb -le $max_size_mb ]]; then
                    break
                fi
            done < <(find "$LOG_DIR" -type f -printf '%T@ %p\0' | sort -zn | cut -zd' ' -f2-)
        else
            log "ERROR" "Manual intervention required. Use --force to enable aggressive cleanup."
            return 1
        fi
    else
        log "SUCCESS" "Log directory size is within limits"
    fi
}

# Generate cleanup report
generate_report() {
    log "INFO" "Generating cleanup report..."
    
    local report_file="$LOG_DIR/log_cleanup_report_$TIMESTAMP.json"
    local current_size_mb=$(get_dir_size_mb "$LOG_DIR")
    local archive_size_mb=$(get_dir_size_mb "$ARCHIVE_DIR")
    
    cat > "$report_file" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "cleanup_type": "$([ "$DRY_RUN" == "true" ] && echo "dry_run" || echo "actual")",
    "configuration": {
        "max_log_size_mb": $MAX_LOG_SIZE_MB,
        "max_log_age_days": $MAX_LOG_AGE_DAYS,
        "delete_archives_days": $DELETE_ARCHIVES_DAYS,
        "max_total_log_size_gb": $MAX_TOTAL_LOG_SIZE_GB
    },
    "results": {
        "current_log_size_mb": $current_size_mb,
        "archive_size_mb": $archive_size_mb,
        "total_size_mb": $((current_size_mb + archive_size_mb))
    },
    "next_cleanup": "$(date -d '+1 day' -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    
    log "SUCCESS" "Cleanup report saved to: $report_file"
    
    # Create symlink to latest report
    if [[ "$DRY_RUN" == "false" ]]; then
        ln -sf "$report_file" "$LOG_DIR/latest_cleanup_report.json"
    fi
}

# Main execution
main() {
    log "INFO" "Starting log rotation and cleanup for SutazAI system"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "Running in DRY RUN mode - no changes will be made"
    fi
    
    # Check if log directory exists
    if [[ ! -d "$LOG_DIR" ]]; then
        log "ERROR" "Log directory does not exist: $LOG_DIR"
        exit 1
    fi
    
    # Set up directories
    setup_directories
    
    # Run cleanup operations
    rotate_large_logs
    archive_old_logs
    cleanup_docker_logs
    delete_old_archives
    check_total_size
    
    # Generate report
    generate_report
    
    log "SUCCESS" "Log rotation and cleanup completed successfully"
    
    # Show summary
    echo
    echo "============================================"
    echo "       LOG CLEANUP SUMMARY"
    echo "============================================"
    echo "Mode: $([ "$DRY_RUN" == "true" ] && echo "DRY RUN" || echo "ACTUAL CLEANUP")"
    echo "Current log size: $(get_dir_size_mb "$LOG_DIR")MB"
    echo "Archive size: $(get_dir_size_mb "$ARCHIVE_DIR")MB"
    echo "Timestamp: $(date)"
    echo "============================================"
}

# Run main function
main "$@"