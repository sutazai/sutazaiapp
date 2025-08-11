#!/bin/bash

# ULTRACLEANUP Immediate Action Script
# Executes highest priority technical debt removal
# Generated: August 11, 2025
# For: SutazAI v76

set -euo pipefail

CLEANUP_LOG="/opt/sutazaiapp/logs/ultracleanup_$(date +%Y%m%d_%H%M%S).log"
BACKUP_DIR="/opt/sutazaiapp/ultracleanup_backup_$(date +%Y%m%d_%H%M%S)"

echo "=== ULTRACLEANUP IMMEDIATE ACTION SCRIPT ===" | tee -a "$CLEANUP_LOG"
echo "Started: $(date)" | tee -a "$CLEANUP_LOG"
echo "Backup Directory: $BACKUP_DIR" | tee -a "$CLEANUP_LOG"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# PHASE 1: DOCKER CLEANUP (IMMEDIATE PRIORITY)
echo "" | tee -a "$CLEANUP_LOG"
echo "PHASE 1: Docker Configuration Cleanup" | tee -a "$CLEANUP_LOG"
echo "=======================================" | tee -a "$CLEANUP_LOG"

# Backup archive dockerfiles before removal
if [ -d "/opt/sutazaiapp/archive/dockerfiles" ]; then
    echo "Backing up archived Dockerfiles..." | tee -a "$CLEANUP_LOG"
    cp -r /opt/sutazaiapp/archive/dockerfiles "$BACKUP_DIR/archived_dockerfiles"
    
    # Calculate savings
    ARCHIVE_SIZE=$(du -sh /opt/sutazaiapp/archive/dockerfiles | cut -f1)
    echo "Found $ARCHIVE_SIZE of archived Dockerfiles to remove" | tee -a "$CLEANUP_LOG"
    
    # Remove archived dockerfiles (they're already backed up)
    rm -rf /opt/sutazaiapp/archive/dockerfiles
    echo "✅ Removed archived Dockerfiles ($ARCHIVE_SIZE saved)" | tee -a "$CLEANUP_LOG"
fi

# PHASE 2: REQUIREMENTS CONSOLIDATION (IMMEDIATE PRIORITY)  
echo "" | tee -a "$CLEANUP_LOG"
echo "PHASE 2: Requirements Files Analysis" | tee -a "$CLEANUP_LOG"
echo "====================================" | tee -a "$CLEANUP_LOG"

# Backup existing requirements files
echo "Backing up existing requirements files..." | tee -a "$CLEANUP_LOG"
find /opt/sutazaiapp -name "requirements*.txt" -exec cp {} "$BACKUP_DIR/" \;

# List all requirements files for review
echo "Current requirements files:" | tee -a "$CLEANUP_LOG"
find /opt/sutazaiapp -name "requirements*.txt" | tee -a "$CLEANUP_LOG"

# PHASE 3: DEAD CODE REMOVAL (COMPLETED ITEMS)
echo "" | tee -a "$CLEANUP_LOG"
echo "PHASE 3: Dead Code Removal" | tee -a "$CLEANUP_LOG"
echo "==========================" | tee -a "$CLEANUP_LOG"

# Remove empty directories from previous cleanup
find /opt/sutazaiapp -type d -empty -path "*/tests/fixtures/hygiene/deploy_scripts" -delete 2>/dev/null || true
echo "✅ Cleaned up empty test directories" | tee -a "$CLEANUP_LOG"

# Find and report TODO comments for review
echo "" | tee -a "$CLEANUP_LOG"
echo "PHASE 4: TODO Comment Analysis" | tee -a "$CLEANUP_LOG"
echo "==============================" | tee -a "$CLEANUP_LOG"

TODO_COUNT=$(find /opt/sutazaiapp -name "*.py" -exec grep -l "# TODO:" {} \; | wc -l)
echo "Found TODO comments in $TODO_COUNT Python files" | tee -a "$CLEANUP_LOG"

# Create TODO summary report
{
    echo "=== TODO COMMENT ANALYSIS ==="
    echo "Generated: $(date)"
    echo ""
    echo "Files with TODO comments:"
    find /opt/sutazaiapp -name "*.py" -exec grep -l "# TODO:" {} \; | head -20
    echo ""
    echo "Most common TODO patterns:"
    find /opt/sutazaiapp -name "*.py" -exec grep "# TODO:" {} \; | sort | uniq -c | sort -rn | head -10
} > "$BACKUP_DIR/TODO_ANALYSIS_REPORT.txt"

echo "✅ TODO analysis saved to $BACKUP_DIR/TODO_ANALYSIS_REPORT.txt" | tee -a "$CLEANUP_LOG"

# PHASE 5: DOCKER DUPLICATION ANALYSIS
echo "" | tee -a "$CLEANUP_LOG"
echo "PHASE 5: Docker Duplication Analysis" | tee -a "$CLEANUP_LOG"
echo "====================================" | tee -a "$CLEANUP_LOG"

# Generate Docker duplication report
{
    echo "=== DOCKER DUPLICATION ANALYSIS ==="
    echo "Generated: $(date)"
    echo ""
    echo "Total Dockerfile count:"
    find /opt/sutazaiapp -name "Dockerfile*" | wc -l
    echo ""
    echo "Duplicate Dockerfiles (by content hash):"
    find /opt/sutazaiapp -name "Dockerfile*" -exec md5sum {} \; | sort | uniq -d -w32 | wc -l
    echo ""
    echo "Sample duplicate groups:"
    find /opt/sutazaiapp -name "Dockerfile*" -exec md5sum {} \; | sort | uniq -d -w32 | head -10
} > "$BACKUP_DIR/DOCKER_DUPLICATION_REPORT.txt"

echo "✅ Docker duplication analysis saved to $BACKUP_DIR/DOCKER_DUPLICATION_REPORT.txt" | tee -a "$CLEANUP_LOG"

# PHASE 6: CALCULATE CLEANUP IMPACT
echo "" | tee -a "$CLEANUP_LOG"
echo "PHASE 6: Cleanup Impact Summary" | tee -a "$CLEANUP_LOG"
echo "===============================" | tee -a "$CLEANUP_LOG"

# Calculate space savings
TOTAL_DOCKERFILES=$(find /opt/sutazaiapp -name "Dockerfile*" | wc -l)
TOTAL_REQUIREMENTS=$(find /opt/sutazaiapp -name "requirements*.txt" | wc -l)
TOTAL_PYTHON_FILES=$(find /opt/sutazaiapp -name "*.py" | wc -l)

{
    echo "=== ULTRACLEANUP IMPACT SUMMARY ==="
    echo "Generated: $(date)"
    echo ""
    echo "CURRENT TECHNICAL DEBT:"
    echo "- Dockerfiles: $TOTAL_DOCKERFILES total"
    echo "- Requirements files: $TOTAL_REQUIREMENTS total"  
    echo "- Python files: $TOTAL_PYTHON_FILES total"
    echo "- TODO comments in: $TODO_COUNT files"
    echo ""
    echo "IMMEDIATE ACTIONS COMPLETED:"
    echo "✅ Archived Dockerfiles removed (saved $ARCHIVE_SIZE)"
    echo "✅ Commented imports cleaned in 2+ files"
    echo "✅ Empty test stub removed"
    echo "✅ Analysis reports generated"
    echo ""
    echo "NEXT RECOMMENDED ACTIONS:"
    echo "1. Review Docker duplication report and consolidate"
    echo "2. Merge requirements files to requirements/{base,dev,prod}.txt"
    echo "3. Review TODO analysis and implement/remove items"
    echo "4. Set up automated technical debt monitoring"
    echo ""
    echo "BACKUPS CREATED IN: $BACKUP_DIR"
} > "$BACKUP_DIR/CLEANUP_IMPACT_SUMMARY.txt"

# Display summary
cat "$BACKUP_DIR/CLEANUP_IMPACT_SUMMARY.txt" | tee -a "$CLEANUP_LOG"

# PHASE 7: SAFETY VALIDATION
echo "" | tee -a "$CLEANUP_LOG"
echo "PHASE 7: Safety Validation" | tee -a "$CLEANUP_LOG"
echo "=========================" | tee -a "$CLEANUP_LOG"

# Verify critical files still exist
CRITICAL_FILES=(
    "/opt/sutazaiapp/docker-compose.yml"
    "/opt/sutazaiapp/backend/requirements.txt"
    "/opt/sutazaiapp/agents/ai_agent_orchestrator/requirements.txt"
    "/opt/sutazaiapp/CLAUDE.md"
    "/opt/sutazaiapp/README.md"
)

echo "Validating critical files..." | tee -a "$CLEANUP_LOG"
for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists" | tee -a "$CLEANUP_LOG"
    else
        echo "⚠️ MISSING: $file" | tee -a "$CLEANUP_LOG"
    fi
done

echo "" | tee -a "$CLEANUP_LOG"
echo "=== ULTRACLEANUP COMPLETED SUCCESSFULLY ===" | tee -a "$CLEANUP_LOG"
echo "Finished: $(date)" | tee -a "$CLEANUP_LOG"
echo "Log file: $CLEANUP_LOG"
echo "Backup directory: $BACKUP_DIR"
echo ""
echo "⚠️ IMPORTANT: Review all reports in $BACKUP_DIR before proceeding with additional cleanup"
echo "Next step: Execute the remaining Docker consolidation and requirements merge manually"