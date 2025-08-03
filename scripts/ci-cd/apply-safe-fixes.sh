#!/bin/bash
# Apply Safe Automated Hygiene Fixes
# Purpose: Automatically applies safe fixes for specific hygiene rules
# Usage: ./apply-safe-fixes.sh --rules "6,7,8,13,15" --safe-mode [--report output.json]
# Requirements: Python 3.8+, git, black, autopep8

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${PROJECT_ROOT}/logs/safe-fixes-${TIMESTAMP}.log"
REPORT_FILE=""
SAFE_MODE=true
RULES_TO_FIX=""
DRY_RUN=false

# Safe fixable rules
SAFE_RULES="6,7,8,13,15"

# Logging functions
log() {
    local level="$1"
    shift
    local message="$*"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${level}] ${message}" | tee -a "${LOG_FILE}"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }

# Show help
show_help() {
    cat << EOF
Apply Safe Automated Hygiene Fixes

Usage: ${0##*/} [OPTIONS]

Options:
    -r, --rules RULES       Comma-separated list of rules to fix (default: 6,7,8,13,15)
    -s, --safe-mode        Only apply guaranteed safe fixes (default: true)
    -o, --report FILE      Output report file (JSON)
    -d, --dry-run          Show what would be done without making changes
    -h, --help             Show this help message

Safe Rules:
    6  - Documentation formatting (markdown, comments)
    7  - Script organization (move to correct directories)
    8  - Python docstrings and headers
    13 - Junk file removal (*.bak, *.tmp, etc.)
    15 - Documentation deduplication

Example:
    ${0##*/} --rules "6,8,13" --report fixes-applied.json
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--rules)
            RULES_TO_FIX="$2"
            shift 2
            ;;
        -s|--safe-mode)
            SAFE_MODE=true
            shift
            ;;
        -o|--report)
            REPORT_FILE="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Default rules if not specified
if [[ -z "${RULES_TO_FIX}" ]]; then
    RULES_TO_FIX="${SAFE_RULES}"
fi

# Initialize report
FIXES_APPLIED=()
FIXES_SKIPPED=()
FILES_MODIFIED=0
FILES_REMOVED=0

# Create log directory
mkdir -p "$(dirname "${LOG_FILE}")"

log_info "Starting safe automated fixes"
log_info "Rules to fix: ${RULES_TO_FIX}"
log_info "Safe mode: ${SAFE_MODE}"
log_info "Dry run: ${DRY_RUN}"

# Function to check if a rule is safe
is_safe_rule() {
    local rule="$1"
    [[ ",${SAFE_RULES}," =~ ",${rule}," ]]
}

# Function to apply fix and track changes
apply_fix() {
    local description="$1"
    local command="$2"
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would apply: ${description}"
        FIXES_SKIPPED+=("${description} (dry run)")
    else
        log_info "Applying: ${description}"
        if eval "${command}"; then
            FIXES_APPLIED+=("${description}")
        else
            log_warn "Failed to apply: ${description}"
            FIXES_SKIPPED+=("${description} (failed)")
        fi
    fi
}

# Rule 6: Documentation formatting
fix_rule_6() {
    log_info "=== Fixing Rule 6: Documentation formatting ==="
    
    # Fix markdown files
    find "${PROJECT_ROOT}" -name "*.md" -type f | while read -r file; do
        # Skip vendor and node_modules
        if [[ "${file}" =~ (vendor|node_modules|venv|.git) ]]; then
            continue
        fi
        
        # Fix common markdown issues
        if [[ "${DRY_RUN}" != "true" ]]; then
            # Add newline at end if missing
            if [[ -n "$(tail -c 1 "${file}")" ]]; then
                echo >> "${file}"
                ((FILES_MODIFIED++))
            fi
            
            # Fix heading spacing
            sed -i 's/^#\([^# ]\)/# \1/g' "${file}"
            sed -i 's/^##\([^# ]\)/## \1/g' "${file}"
            sed -i 's/^###\([^# ]\)/### \1/g' "${file}"
            
            # Fix bullet point spacing
            sed -i 's/^\*\([^ ]\)/* \1/g' "${file}"
            sed -i 's/^-\([^ ]\)/- \1/g' "${file}"
        fi
        
        apply_fix "Format markdown: ${file}" "true"
    done
}

# Rule 7: Script organization
fix_rule_7() {
    log_info "=== Fixing Rule 7: Script organization ==="
    
    # Create script directories if needed
    mkdir -p "${PROJECT_ROOT}/scripts"/{dev,deploy,data,utils,test}
    
    # Move misplaced scripts
    local script_moves=(
        "deploy*.sh:scripts/deploy/"
        "test*.sh:scripts/test/"
        "*_test.py:scripts/test/"
        "setup*.sh:scripts/dev/"
        "clean*.sh:scripts/utils/"
    )
    
    for move in "${script_moves[@]}"; do
        IFS=':' read -r pattern destination <<< "${move}"
        
        find "${PROJECT_ROOT}" -maxdepth 2 -name "${pattern}" -type f | while read -r file; do
            # Skip if already in correct location
            if [[ "${file}" =~ ${destination} ]]; then
                continue
            fi
            
            local basename=$(basename "${file}")
            local target="${PROJECT_ROOT}/${destination}${basename}"
            
            apply_fix "Move script: ${file} -> ${target}" "mv '${file}' '${target}'"
        done
    done
}

# Rule 8: Python docstrings and headers
fix_rule_8() {
    log_info "=== Fixing Rule 8: Python docstrings and headers ==="
    
    # Install Python formatting tools if needed
    if ! command -v black &> /dev/null && [[ "${DRY_RUN}" != "true" ]]; then
        pip install --quiet black autopep8
    fi
    
    find "${PROJECT_ROOT}" -name "*.py" -type f | while read -r file; do
        # Skip vendor and test files
        if [[ "${file}" =~ (vendor|node_modules|venv|.git|__pycache__|test_) ]]; then
            continue
        fi
        
        # Check if file has proper header
        if ! head -n 1 "${file}" | grep -q "^#!/usr/bin/env python3"; then
            if [[ "${DRY_RUN}" != "true" ]]; then
                # Add shebang and docstring template
                local temp_file=$(mktemp)
                echo '#!/usr/bin/env python3' > "${temp_file}"
                echo '"""' >> "${temp_file}"
                echo "Purpose: TODO: Add description" >> "${temp_file}"
                echo "Usage: python $(basename "${file}") [options]" >> "${temp_file}"
                echo "Requirements: TODO: List requirements" >> "${temp_file}"
                echo '"""' >> "${temp_file}"
                echo >> "${temp_file}"
                cat "${file}" >> "${temp_file}"
                mv "${temp_file}" "${file}"
                ((FILES_MODIFIED++))
            fi
            
            apply_fix "Add Python header: ${file}" "true"
        fi
        
        # Format with black (safe mode)
        if command -v black &> /dev/null && [[ "${DRY_RUN}" != "true" ]]; then
            black --quiet --safe "${file}" 2>/dev/null || true
        fi
    done
}

# Rule 13: Junk file removal
fix_rule_13() {
    log_info "=== Fixing Rule 13: Junk file removal ==="
    
    # Define junk patterns (only truly safe ones)
    local junk_patterns=(
        "*.bak"
        "*.tmp"
        "*.swp"
        "*.swo"
        "*~"
        ".DS_Store"
        "Thumbs.db"
        "*.pyc"
        "__pycache__"
        "*.log.1"
        "*.log.2"
        "*.old"
        "*.orig"
    )
    
    # Create archive directory
    local archive_dir="${PROJECT_ROOT}/archive/junk-cleanup-${TIMESTAMP}"
    if [[ "${DRY_RUN}" != "true" ]]; then
        mkdir -p "${archive_dir}"
    fi
    
    for pattern in "${junk_patterns[@]}"; do
        find "${PROJECT_ROOT}" -name "${pattern}" -type f 2>/dev/null | while read -r file; do
            # Skip .git directory
            if [[ "${file}" =~ \.git/ ]]; then
                continue
            fi
            
            # Archive before removal
            if [[ "${DRY_RUN}" != "true" ]]; then
                local rel_path="${file#${PROJECT_ROOT}/}"
                local archive_path="${archive_dir}/${rel_path}"
                mkdir -p "$(dirname "${archive_path}")"
                cp "${file}" "${archive_path}"
                rm "${file}"
                ((FILES_REMOVED++))
            fi
            
            apply_fix "Remove junk file: ${file}" "true"
        done
    done
}

# Rule 15: Documentation deduplication
fix_rule_15() {
    log_info "=== Fixing Rule 15: Documentation deduplication ==="
    
    # Find duplicate README files
    find "${PROJECT_ROOT}" -name "README*.md" -type f | sort | while read -r file; do
        local dir=$(dirname "${file}")
        local basename=$(basename "${file}")
        
        # If there's both README.md and another README variant, consolidate
        if [[ "${basename}" != "README.md" ]] && [[ -f "${dir}/README.md" ]]; then
            if [[ "${DRY_RUN}" != "true" ]]; then
                # Append content to main README
                echo -e "\n\n---\n\n# Content from ${basename}\n" >> "${dir}/README.md"
                cat "${file}" >> "${dir}/README.md"
                rm "${file}"
                ((FILES_REMOVED++))
            fi
            
            apply_fix "Consolidate documentation: ${file} -> README.md" "true"
        fi
    done
}

# Main execution
main() {
    # Convert comma-separated rules to array
    IFS=',' read -ra RULES_ARRAY <<< "${RULES_TO_FIX}"
    
    for rule in "${RULES_ARRAY[@]}"; do
        # Check if rule is safe
        if [[ "${SAFE_MODE}" == "true" ]] && ! is_safe_rule "${rule}"; then
            log_warn "Skipping unsafe rule: ${rule}"
            FIXES_SKIPPED+=("Rule ${rule} (unsafe)")
            continue
        fi
        
        # Apply fixes for each rule
        case "${rule}" in
            6)
                fix_rule_6
                ;;
            7)
                fix_rule_7
                ;;
            8)
                fix_rule_8
                ;;
            13)
                fix_rule_13
                ;;
            15)
                fix_rule_15
                ;;
            *)
                log_warn "No automated fix available for rule: ${rule}"
                FIXES_SKIPPED+=("Rule ${rule} (not implemented)")
                ;;
        esac
    done
    
    # Generate report
    log_info "=== Fix Summary ==="
    log_info "Files modified: ${FILES_MODIFIED}"
    log_info "Files removed: ${FILES_REMOVED}"
    log_info "Fixes applied: ${#FIXES_APPLIED[@]}"
    log_info "Fixes skipped: ${#FIXES_SKIPPED[@]}"
    
    if [[ -n "${REPORT_FILE}" ]]; then
        cat > "${REPORT_FILE}" <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "dry_run": ${DRY_RUN},
  "safe_mode": ${SAFE_MODE},
  "rules_processed": [${RULES_TO_FIX}],
  "statistics": {
    "files_modified": ${FILES_MODIFIED},
    "files_removed": ${FILES_REMOVED},
    "fixes_applied": ${#FIXES_APPLIED[@]},
    "fixes_skipped": ${#FIXES_SKIPPED[@]}
  },
  "fixes_applied": [
$(printf '    "%s"' "${FIXES_APPLIED[@]}" | paste -sd',\n' -)
  ],
  "fixes_skipped": [
$(printf '    "%s"' "${FIXES_SKIPPED[@]}" | paste -sd',\n' -)
  ]
}
EOF
        log_info "Report written to: ${REPORT_FILE}"
    fi
    
    # Return success if any fixes were applied
    if [[ ${#FIXES_APPLIED[@]} -gt 0 ]] || [[ "${DRY_RUN}" == "true" ]]; then
        return 0
    else
        return 1
    fi
}

# Run main
main