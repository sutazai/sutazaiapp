#!/bin/bash
# SutazAI Bulk File Renaming Script

# Logging
LOG_FILE="/var/log/sutazai/bulk_rename.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Rename patterns
RENAME_PATTERNS=(
    "Quantum:SutazAi"
    "quantum:sutazai"
    "QUANTUM:SUTAZAI"
    "q_:sutazai_"
    "Qubit:Sutaz"
    "qubit:sutaz"
    "QUBIT:SUTAZ"
)

# Directories to exclude
EXCLUDE_DIRS=(
    ".git"
    ".venv"
    "node_modules"
    "__pycache__"
)

# Generate exclude pattern for find
generate_exclude_pattern() {
    local exclude_pattern=""
    for dir in "${EXCLUDE_DIRS[@]}"; do
        exclude_pattern+="! -path '*/$dir*' "
    done
    echo "$exclude_pattern"
}

# Rename files and directories
rename_files_and_dirs() {
    log "Starting file and directory renaming..."
    
    local exclude_pattern=$(generate_exclude_pattern)
    
    # Rename directories first (depth-first to handle nested directories)
    find . -type d -depth $exclude_pattern | while read -r dir; do
        for pattern in "${RENAME_PATTERNS[@]}"; do
            old_name="$dir"
            new_name=$(echo "$dir" | sed "s/$(echo "$pattern" | cut -d: -f1)/$(echo "$pattern" | cut -d: -f2)/g")
            if [ "$old_name" != "$new_name" ]; then
                mv "$old_name" "$new_name"
                log "Renamed directory: $old_name -> $new_name"
            fi
        done
    done

    # Rename files
    find . -type f $exclude_pattern | while read -r file; do
        for pattern in "${RENAME_PATTERNS[@]}"; do
            old_name="$file"
            new_name=$(echo "$file" | sed "s/$(echo "$pattern" | cut -d: -f1)/$(echo "$pattern" | cut -d: -f2)/g")
            if [ "$old_name" != "$new_name" ]; then
                mv "$old_name" "$new_name"
                log "Renamed file: $old_name -> $new_name"
            fi
        done
    done
}

# Replace content in files
replace_content() {
    log "Starting content replacement..."
    find . -type f $exclude_pattern -print0 | xargs -0 sed -i \
        -e 's/Quantum/SutazAi/g' \
        -e 's/quantum/sutazai/g' \
        -e 's/QUANTUM/SUTAZAI/g' \
        -e 's/q_/sutazai_/g' \
        -e 's/Qubit/Sutaz/g' \
        -e 's/qubit/sutaz/g' \
        -e 's/QUBIT/SUTAZ/g'
}

# Verification function
verify_rename() {
    log "Verifying rename process..."
    if grep -rn "Quantum\|quantum\|QUANTUM" . $exclude_pattern; then
        log "ERROR: Quantum references still exist!"
        exit 1
    else
        log "Rename process completed successfully!"
    fi
}

# Main execution
main() {
    log "🔄 Starting SutazAI Bulk Renaming Process"
    
    rename_files_and_dirs
    replace_content
    verify_rename
    
    log "✅ Bulk Renaming Complete"
}

# Execute main function
main 