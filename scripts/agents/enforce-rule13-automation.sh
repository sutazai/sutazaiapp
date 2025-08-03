#!/bin/bash
#
# Rule 13 Enforcement Automation Script
#
# Purpose: Automated enforcement of "No Garbage, No Rot" with safety checks
# Usage: ./enforce-rule13-automation.sh [mode] [options]
# Requirements: Python 3.8+, git, ripgrep
#

set -euo pipefail

# Colors and formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/opt/sutazaiapp"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENFORCER_SCRIPT="$SCRIPT_DIR/garbage-collection-enforcer.py"
LOG_FILE="/opt/sutazaiapp/logs/rule13-enforcement.log"
BACKUP_DIR="/opt/sutazaiapp/archive"
SESSION_ID=$(date +%Y%m%d_%H%M%S)

# Default settings
MODE="scan"
CONFIDENCE_THRESHOLD="0.7"
RISK_THRESHOLD="safe"
AUTO_COMMIT=false
CREATE_BRANCH=false
FORCE_MODE=false
VERBOSE=false
INTERACTIVE=true

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    case "$level" in
        "ERROR")
            echo -e "${RED}❌ $message${NC}" >&2
            ;;
        "WARN")
            echo -e "${YELLOW}⚠️  $message${NC}"
            ;;
        "INFO")
            echo -e "${BLUE}ℹ️  $message${NC}"
            ;;
        "SUCCESS")
            echo -e "${GREEN}✅ $message${NC}"
            ;;
        "DEBUG")
            if [[ "$VERBOSE" == "true" ]]; then
                echo -e "${PURPLE}🔍 $message${NC}"
            fi
            ;;
    esac
}

# Help function
show_help() {
    cat << EOF
${BOLD}Rule 13 Enforcement Automation Script${NC}

${CYAN}MODES:${NC}
    scan        Perform dry-run scan only (default)
    clean       Perform safe cleanup (safe risk threshold)
    aggressive  More aggressive cleanup (moderate risk threshold)
    custom      Custom cleanup with specified parameters

${CYAN}OPTIONS:${NC}
    -p, --project-root PATH     Project root directory (default: /opt/sutazaiapp)
    -c, --confidence FLOAT      Confidence threshold 0.0-1.0 (default: 0.7)
    -r, --risk LEVEL           Risk threshold: safe|moderate|risky (default: safe)
    -b, --create-branch        Create git branch for cleanup
    -a, --auto-commit          Auto-commit changes after cleanup
    -f, --force                Skip safety confirmations
    -v, --verbose              Enable verbose output
    -i, --non-interactive      Disable interactive confirmations
    -h, --help                 Show this help message

${CYAN}EXAMPLES:${NC}
    # Safe scan
    ./enforce-rule13-automation.sh scan
    
    # Safe cleanup with git branch
    ./enforce-rule13-automation.sh clean -b -a
    
    # Aggressive cleanup (requires confirmation)
    ./enforce-rule13-automation.sh aggressive -c 0.6
    
    # Custom cleanup
    ./enforce-rule13-automation.sh custom -c 0.8 -r moderate -b

${CYAN}SAFETY FEATURES:${NC}
    • Automatic git status checking
    • Pre-cleanup backup creation
    • Interactive confirmations
    • Rollback instructions
    • Comprehensive logging

EOF
}

# Validation functions
validate_environment() {
    log "DEBUG" "Validating environment..."
    
    # Check required commands
    local required_commands=("python3" "git" "rg")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log "ERROR" "Required command not found: $cmd"
            return 1
        fi
    done
    
    # Check project root
    if [[ ! -d "$PROJECT_ROOT" ]]; then
        log "ERROR" "Project root directory does not exist: $PROJECT_ROOT"
        return 1
    fi
    
    # Check enforcer script
    if [[ ! -f "$ENFORCER_SCRIPT" ]]; then
        log "ERROR" "Garbage collection enforcer script not found: $ENFORCER_SCRIPT"
        return 1
    fi
    
    # Check git repo
    if ! git -C "$PROJECT_ROOT" rev-parse --git-dir &> /dev/null; then
        log "ERROR" "Project root is not a git repository: $PROJECT_ROOT"
        return 1
    fi
    
    # Create log directory if needed
    mkdir -p "$(dirname "$LOG_FILE")"
    
    log "SUCCESS" "Environment validation passed"
    return 0
}

check_git_status() {
    log "DEBUG" "Checking git status..."
    
    cd "$PROJECT_ROOT"
    
    # Check for uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        log "WARN" "Uncommitted changes detected in working directory"
        if [[ "$INTERACTIVE" == "true" && "$FORCE_MODE" == "false" ]]; then
            echo -e "${YELLOW}You have uncommitted changes. Continue? [y/N]:${NC} "
            read -r response
            if [[ ! "$response" =~ ^[Yy]$ ]]; then
                log "INFO" "Operation cancelled by user"
                exit 0
            fi
        elif [[ "$FORCE_MODE" == "false" ]]; then
            log "ERROR" "Uncommitted changes detected. Use --force to override or commit changes first"
            return 1
        fi
    fi
    
    # Check current branch
    local current_branch=$(git rev-parse --abbrev-ref HEAD)
    log "INFO" "Current branch: $current_branch"
    
    return 0
}

create_backup() {
    log "DEBUG" "Creating pre-cleanup backup..."
    
    local backup_path="$BACKUP_DIR/pre_rule13_cleanup_$SESSION_ID"
    
    # Create backup directory
    mkdir -p "$backup_path"
    
    # Create git snapshot
    cd "$PROJECT_ROOT"
    git stash push -m "Pre-Rule13-cleanup snapshot $SESSION_ID" --include-untracked || true
    
    # Save current commit hash
    git rev-parse HEAD > "$backup_path/commit_hash.txt"
    git status --porcelain > "$backup_path/git_status.txt"
    
    log "SUCCESS" "Backup created at: $backup_path"
    echo "$backup_path" > "/tmp/rule13_backup_path_$SESSION_ID"
}

run_garbage_scan() {
    log "INFO" "Running garbage collection scan..."
    
    local scan_args=(
        "$ENFORCER_SCRIPT"
        --project-root "$PROJECT_ROOT"
        --dry-run
        --confidence-threshold "$CONFIDENCE_THRESHOLD"
        --risk-threshold "$RISK_THRESHOLD"
        --output "/tmp/rule13_scan_$SESSION_ID.json"
    )
    
    if [[ "$VERBOSE" == "true" ]]; then
        scan_args+=(--verbose)
    fi
    
    # Run scan
    if python3 "${scan_args[@]}"; then
        log "SUCCESS" "Garbage scan completed successfully"
        return 0
    else
        log "ERROR" "Garbage scan failed"
        return 1
    fi
}

analyze_scan_results() {
    local scan_file="/tmp/rule13_scan_$SESSION_ID.json"
    
    if [[ ! -f "$scan_file" ]]; then
        log "ERROR" "Scan results file not found: $scan_file"
        return 1
    fi
    
    log "DEBUG" "Analyzing scan results..."
    
    # Extract key metrics using jq if available, or basic grep
    local items_found actionable_items space_mb
    
    if command -v jq &> /dev/null; then
        items_found=$(jq -r '.analysis.total_garbage_items // 0' "$scan_file")
        actionable_items=$(jq -r '.analysis.actionable_items // 0' "$scan_file")
        space_mb=$(jq -r '.analysis.total_potential_space_mb // 0' "$scan_file")
    else
        items_found=$(grep -o '"total_garbage_items": [0-9]*' "$scan_file" | cut -d: -f2 | xargs || echo "0")
        actionable_items=$(grep -o '"actionable_items": [0-9]*' "$scan_file" | cut -d: -f2 | xargs || echo "0")
        space_mb=$(grep -o '"total_potential_space_mb": [0-9.]*' "$scan_file" | cut -d: -f2 | xargs || echo "0")
    fi
    
    log "INFO" "Scan Analysis:"
    log "INFO" "  Total garbage items: $items_found"
    log "INFO" "  Actionable items: $actionable_items"
    log "INFO" "  Potential space recovery: ${space_mb}MB"
    
    # Store results for later use
    echo "$items_found" > "/tmp/rule13_items_found_$SESSION_ID"
    echo "$actionable_items" > "/tmp/rule13_actionable_$SESSION_ID"
    echo "$space_mb" > "/tmp/rule13_space_mb_$SESSION_ID"
    
    return 0
}

confirm_cleanup() {
    local actionable_items=$(cat "/tmp/rule13_actionable_$SESSION_ID" 2>/dev/null || echo "0")
    local space_mb=$(cat "/tmp/rule13_space_mb_$SESSION_ID" 2>/dev/null || echo "0")
    
    if [[ "$actionable_items" -eq 0 ]]; then
        log "SUCCESS" "No cleanup needed - codebase is already clean!"
        return 1  # No cleanup needed
    fi
    
    if [[ "$INTERACTIVE" == "true" && "$FORCE_MODE" == "false" ]]; then
        echo ""
        echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${BOLD}${CYAN}║                     CLEANUP CONFIRMATION                    ║${NC}"
        echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${YELLOW}About to perform cleanup with the following settings:${NC}"
        echo -e "  Mode: ${BOLD}$MODE${NC}"
        echo -e "  Confidence Threshold: ${BOLD}$CONFIDENCE_THRESHOLD${NC}"
        echo -e "  Risk Threshold: ${BOLD}$RISK_THRESHOLD${NC}"
        echo -e "  Actionable Items: ${BOLD}$actionable_items${NC}"
        echo -e "  Potential Space Recovery: ${BOLD}${space_mb}MB${NC}"
        echo ""
        echo -e "${RED}Warning: This will permanently remove files (after archiving)${NC}"
        echo -e "${YELLOW}Continue with cleanup? [y/N]:${NC} "
        read -r response
        
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log "INFO" "Cleanup cancelled by user"
            return 1
        fi
    fi
    
    return 0
}

run_cleanup() {
    log "INFO" "Starting garbage cleanup..."
    
    local cleanup_args=(
        "$ENFORCER_SCRIPT"
        --project-root "$PROJECT_ROOT"
        --live
        --confidence-threshold "$CONFIDENCE_THRESHOLD"
        --risk-threshold "$RISK_THRESHOLD"
        --output "/tmp/rule13_cleanup_$SESSION_ID.json"
    )
    
    if [[ "$VERBOSE" == "true" ]]; then
        cleanup_args+=(--verbose)
    fi
    
    # Run cleanup
    if python3 "${cleanup_args[@]}"; then
        log "SUCCESS" "Garbage cleanup completed successfully"
        return 0
    else
        log "ERROR" "Garbage cleanup failed"
        return 1
    fi
}

create_git_branch() {
    if [[ "$CREATE_BRANCH" == "true" ]]; then
        local branch_name="rule13-cleanup-$SESSION_ID"
        log "INFO" "Creating git branch: $branch_name"
        
        cd "$PROJECT_ROOT"
        git checkout -b "$branch_name"
        
        log "SUCCESS" "Created and switched to branch: $branch_name"
    fi
}

commit_changes() {
    if [[ "$AUTO_COMMIT" == "true" ]]; then
        log "INFO" "Committing cleanup changes..."
        
        cd "$PROJECT_ROOT"
        
        # Check if there are changes to commit
        if git diff-index --quiet HEAD --; then
            log "INFO" "No changes to commit"
            return 0
        fi
        
        # Add all changes
        git add -A
        
        # Create commit message
        local items_removed=$(grep -o '"items_removed": [0-9]*' "/tmp/rule13_cleanup_$SESSION_ID.json" | cut -d: -f2 | xargs || echo "0")
        local space_recovered=$(grep -o '"space_recovered_bytes": [0-9]*' "/tmp/rule13_cleanup_$SESSION_ID.json" | cut -d: -f2 | xargs || echo "0")
        local space_mb=$((space_recovered / 1024 / 1024))
        
        local commit_msg="Clean up garbage files (Rule 13: No Garbage, No Rot)

- Removed $items_removed garbage items
- Recovered ${space_mb}MB of space
- Confidence threshold: $CONFIDENCE_THRESHOLD
- Risk threshold: $RISK_THRESHOLD
- Session: $SESSION_ID

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>"
        
        git commit -m "$commit_msg"
        
        log "SUCCESS" "Changes committed successfully"
    fi
}

generate_report() {
    log "DEBUG" "Generating enforcement report..."
    
    local report_file="$PROJECT_ROOT/rule13_enforcement_report_$SESSION_ID.md"
    
    cat > "$report_file" << EOF
# Rule 13 Enforcement Report

**Session ID:** $SESSION_ID  
**Timestamp:** $(date)  
**Mode:** $MODE  
**Project Root:** $PROJECT_ROOT  

## Configuration

- **Confidence Threshold:** $CONFIDENCE_THRESHOLD
- **Risk Threshold:** $RISK_THRESHOLD
- **Create Branch:** $CREATE_BRANCH
- **Auto Commit:** $AUTO_COMMIT
- **Interactive Mode:** $INTERACTIVE

## Results

EOF
    
    # Add scan results if available
    if [[ -f "/tmp/rule13_scan_$SESSION_ID.json" ]]; then
        local items_found=$(cat "/tmp/rule13_items_found_$SESSION_ID" 2>/dev/null || echo "0")
        local actionable_items=$(cat "/tmp/rule13_actionable_$SESSION_ID" 2>/dev/null || echo "0")
        local space_mb=$(cat "/tmp/rule13_space_mb_$SESSION_ID" 2>/dev/null || echo "0")
        
        cat >> "$report_file" << EOF
### Scan Results

- **Total Garbage Items Found:** $items_found
- **Actionable Items:** $actionable_items  
- **Potential Space Recovery:** ${space_mb}MB

EOF
    fi
    
    # Add cleanup results if available
    if [[ -f "/tmp/rule13_cleanup_$SESSION_ID.json" ]]; then
        local cleanup_data="/tmp/rule13_cleanup_$SESSION_ID.json"
        
        if command -v jq &> /dev/null; then
            local items_removed=$(jq -r '.statistics.items_removed // 0' "$cleanup_data")
            local space_recovered=$(jq -r '.statistics.space_recovered_bytes // 0' "$cleanup_data")
            local space_recovered_mb=$((space_recovered / 1024 / 1024))
            
            cat >> "$report_file" << EOF
### Cleanup Results

- **Items Removed:** $items_removed
- **Space Recovered:** ${space_recovered_mb}MB
- **Items Archived:** $(jq -r '.statistics.items_archived // 0' "$cleanup_data")
- **Items Skipped:** $(jq -r '.statistics.items_skipped // 0' "$cleanup_data")

EOF
        fi
    fi
    
    # Add rollback instructions
    cat >> "$report_file" << EOF
## Rollback Instructions

If you need to revert the changes:

1. **Git Rollback:**
   \`\`\`bash
   cd $PROJECT_ROOT
   git reset --hard HEAD~1  # If changes were committed
   \`\`\`

2. **Archive Recovery:**
   \`\`\`bash
   # Files are archived in:
   ls -la $BACKUP_DIR/garbage_cleanup_$SESSION_ID/
   
   # Restore specific files:
   cp $BACKUP_DIR/garbage_cleanup_$SESSION_ID/path/to/file.ext ./path/to/
   \`\`\`

3. **Git Stash Recovery:**
   \`\`\`bash
   git stash list
   git stash apply stash@{0}  # Apply the pre-cleanup snapshot
   \`\`\`

## Files

- **Detailed JSON Report:** \`/tmp/rule13_cleanup_$SESSION_ID.json\`
- **Scan Report:** \`/tmp/rule13_scan_$SESSION_ID.json\`
- **Log File:** \`$LOG_FILE\`
- **Backup Location:** \`$BACKUP_DIR/pre_rule13_cleanup_$SESSION_ID\`

---
*Generated by Rule 13 Enforcement Automation - $(date)*
EOF
    
    log "SUCCESS" "Report generated: $report_file"
}

cleanup_temp_files() {
    log "DEBUG" "Cleaning up temporary files..."
    
    rm -f "/tmp/rule13_scan_$SESSION_ID.json" \
          "/tmp/rule13_cleanup_$SESSION_ID.json" \
          "/tmp/rule13_items_found_$SESSION_ID" \
          "/tmp/rule13_actionable_$SESSION_ID" \
          "/tmp/rule13_space_mb_$SESSION_ID" \
          "/tmp/rule13_backup_path_$SESSION_ID"
}

# Main execution function
main() {
    # Parse arguments
    while (( "$#" )); do
        case "$1" in
            scan|clean|aggressive|custom)
                MODE="$1"
                shift
                ;;
            -p|--project-root)
                if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                    PROJECT_ROOT="$2"
                    shift 2
                else
                    log "ERROR" "Argument for $1 is missing"
                    exit 1
                fi
                ;;
            -c|--confidence)
                if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                    CONFIDENCE_THRESHOLD="$2"
                    shift 2
                else
                    log "ERROR" "Argument for $1 is missing"
                    exit 1
                fi
                ;;
            -r|--risk)
                if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                    RISK_THRESHOLD="$2"
                    shift 2
                else
                    log "ERROR" "Argument for $1 is missing"
                    exit 1
                fi
                ;;
            -b|--create-branch)
                CREATE_BRANCH=true
                shift
                ;;
            -a|--auto-commit)
                AUTO_COMMIT=true
                shift
                ;;
            -f|--force)
                FORCE_MODE=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -i|--non-interactive)
                INTERACTIVE=false
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            -*|--*=)
                log "ERROR" "Unsupported flag $1"
                exit 1
                ;;
            *)
                log "ERROR" "Unsupported argument $1"
                exit 1
                ;;
        esac
    done
    
    # Adjust settings based on mode
    case "$MODE" in
        "clean")
            RISK_THRESHOLD="safe"
            CONFIDENCE_THRESHOLD="0.7"
            ;;
        "aggressive")
            RISK_THRESHOLD="moderate"
            CONFIDENCE_THRESHOLD="0.6"
            ;;
        "scan")
            # Just scanning, settings already default
            ;;
        "custom")
            # Use provided settings
            ;;
    esac
    
    # Header
    echo -e "${BOLD}${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${BLUE}║                   RULE 13 ENFORCEMENT AUTOMATION              ║${NC}"
    echo -e "${BOLD}${BLUE}║                      No Garbage, No Rot                       ║${NC}"
    echo -e "${BOLD}${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    log "INFO" "Starting Rule 13 enforcement automation"
    log "INFO" "Session ID: $SESSION_ID"
    log "INFO" "Mode: $MODE"
    
    # Validation
    if ! validate_environment; then
        exit 1
    fi
    
    if ! check_git_status; then
        exit 1
    fi
    
    # Create backup for non-scan modes
    if [[ "$MODE" != "scan" ]]; then
        create_backup
    fi
    
    # Run garbage scan
    if ! run_garbage_scan; then
        exit 1
    fi
    
    # Analyze results
    if ! analyze_scan_results; then
        exit 1
    fi
    
    # For scan mode, just show results and exit
    if [[ "$MODE" == "scan" ]]; then
        log "SUCCESS" "Scan completed successfully"
        generate_report
        cleanup_temp_files
        exit 0
    fi
    
    # Confirm cleanup for non-scan modes
    if ! confirm_cleanup; then
        log "INFO" "No cleanup performed"
        cleanup_temp_files
        exit 0
    fi
    
    # Create branch if requested
    create_git_branch
    
    # Run cleanup
    if ! run_cleanup; then
        log "ERROR" "Cleanup failed"
        exit 1
    fi
    
    # Commit changes if requested
    commit_changes
    
    # Generate report
    generate_report
    
    # Cleanup
    cleanup_temp_files
    
    log "SUCCESS" "Rule 13 enforcement completed successfully"
    echo ""
    echo -e "${GREEN}✨ Rule 13 enforcement completed! ✨${NC}"
    echo -e "${CYAN}Check the generated report for details.${NC}"
}

# Trap for cleanup on exit
trap cleanup_temp_files EXIT

# Run main function
main "$@"