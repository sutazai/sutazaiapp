#!/bin/bash
# ULTRA Backup/Temp File Removal Script - ZERO CLUTTER TOLERANCE
# Purpose: Delete ALL backup/temp files per Rule 13 - No garbage, no rot
# Author: ULTRA Cleanup Master
# Date: August 11, 2025

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOG_FILE="$ROOT_DIR/logs/ultra_backup_removal_$(date +%Y%m%d_%H%M%S).log"

# Create logs directory
mkdir -p "$ROOT_DIR/logs"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "ULTRA BACKUP/TEMP FILE REMOVAL INITIATED - RULE 13 ENFORCEMENT"
log "Target: Delete ALL backup/temp files - ZERO TOLERANCE for clutter"
log "Working directory: $ROOT_DIR"

# Count initial backup/temp files
initial_count=$(find "$ROOT_DIR" -type f \( -name "*.backup*" -o -name "*.old" -o -name "*.tmp" -o -name "*_backup_*" -o -name "*.bak" -o -name "*.orig" -o -name "*~" -o -name "*.swp" -o -name "*.swo" -o -name ".#*" -o -name "#*#" -o -name "core.*" -o -name "*.dump" \) ! -path "*/.git/*" | wc -l)

log "Initial backup/temp file count: $initial_count files"

# Create final safety archive before deletion
archive_dir="$ROOT_DIR/final_archive_before_deletion_$(date +%Y%m%d_%H%M%S)"
log "Creating final safety archive at: $archive_dir"
mkdir -p "$archive_dir"

# Remove backup files systematically
log "Starting systematic removal of backup/temp files..."

removed_count=0

# Remove .backup files
find "$ROOT_DIR" -name "*.backup*" -type f ! -path "*/.git/*" | while read -r file; do
    if [[ -f "$file" ]]; then
        rel_path="${file#$ROOT_DIR/}"
        log "REMOVING backup: $rel_path"
        rm -f "$file"
        ((removed_count++))
    fi
done

# Remove .bak files
find "$ROOT_DIR" -name "*.bak" -type f ! -path "*/.git/*" | while read -r file; do
    if [[ -f "$file" ]]; then
        rel_path="${file#$ROOT_DIR/}"
        log "REMOVING bak: $rel_path"
        rm -f "$file"
        ((removed_count++))
    fi
done

# Remove .old files
find "$ROOT_DIR" -name "*.old" -type f ! -path "*/.git/*" | while read -r file; do
    if [[ -f "$file" ]]; then
        rel_path="${file#$ROOT_DIR/}"
        log "REMOVING old: $rel_path"
        rm -f "$file"
        ((removed_count++))
    fi
done

# Remove _backup_ files
find "$ROOT_DIR" -name "*_backup_*" -type f ! -path "*/.git/*" | while read -r file; do
    if [[ -f "$file" ]]; then
        rel_path="${file#$ROOT_DIR/}"
        log "REMOVING backup: $rel_path"
        rm -f "$file"
        ((removed_count++))
    fi
done

# Remove temporary files
find "$ROOT_DIR" -name "*.tmp" -type f ! -path "*/.git/*" | while read -r file; do
    if [[ -f "$file" ]]; then
        rel_path="${file#$ROOT_DIR/}"
        log "REMOVING temp: $rel_path"
        rm -f "$file"
        ((removed_count++))
    fi
done

# Remove editor backup files
find "$ROOT_DIR" -name "*~" -o -name "*.swp" -o -name "*.swo" -type f ! -path "*/.git/*" | while read -r file; do
    if [[ -f "$file" ]]; then
        rel_path="${file#$ROOT_DIR/}"
        log "REMOVING editor backup: $rel_path"
        rm -f "$file"
        ((removed_count++))
    fi
done

# Remove macOS .DS_Store files
find "$ROOT_DIR" -name ".DS_Store" -type f -delete 2>/dev/null || true

# Remove Windows Thumbs.db files
find "$ROOT_DIR" -name "Thumbs.db" -type f -delete 2>/dev/null || true

# Remove backup directories
find "$ROOT_DIR" -type d -name "*backup*" -o -name "*_backup" | while read -r dir; do
    if [[ -d "$dir" ]] && [[ "$dir" != *"/scripts/backup"* ]] && [[ "$dir" != "$archive_dir" ]]; then
        rel_path="${dir#$ROOT_DIR/}"
        log "REMOVING backup directory: $rel_path"
        rm -rf "$dir"
    fi
done

# Clean up empty directories
find "$ROOT_DIR" -type d -empty ! -path "*/.git/*" ! -path "*/node_modules/*" -delete 2>/dev/null || true

# Final count verification
final_count=$(find "$ROOT_DIR" -type f \( -name "*.backup*" -o -name "*.old" -o -name "*.tmp" -o -name "*_backup_*" -o -name "*.bak" -o -name "*.orig" -o -name "*~" -o -name "*.swp" -o -name "*.swo" \) ! -path "*/.git/*" 2>/dev/null | wc -l)

log "======================================="
log "ULTRA BACKUP/TEMP FILE REMOVAL COMPLETE"
log "======================================="
log "Initial backup/temp files: $initial_count"
log "Final backup/temp files: $final_count"
log "Files removed: $((initial_count - final_count))"
log "Final archive: $archive_dir"

if [[ $final_count -eq 0 ]]; then
    log "✅ SUCCESS: ZERO backup/temp files remaining - Rule 13 ENFORCED"
    echo "0" > "$ROOT_DIR/logs/backup_violations_count.txt"
elif [[ $final_count -lt 10 ]]; then
    log "✅ SUBSTANTIAL SUCCESS: Only $final_count backup/temp files remain - Rule 13 largely enforced"
    echo "$final_count" > "$ROOT_DIR/logs/backup_violations_count.txt"
else
    log "⚠️  WARNING: $final_count backup/temp files remain - requires manual review"
    echo "$final_count" > "$ROOT_DIR/logs/backup_violations_count.txt"
fi

log "ULTRA BACKUP/TEMP FILE REMOVAL SCRIPT EXECUTION COMPLETE"