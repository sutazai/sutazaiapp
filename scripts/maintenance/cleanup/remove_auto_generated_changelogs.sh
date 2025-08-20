#!/bin/bash
# CHANGELOG Cleanup Script
# Purpose: Remove auto-generated template CHANGELOGs with exactly 37 lines
# Date: 2025-08-20
# Author: Documentation Architecture Team

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
TOTAL_FOUND=0
DELETED_COUNT=0
KEPT_COUNT=0
ERROR_COUNT=0

# Backup directory
BACKUP_DIR="/opt/sutazaiapp/backup/changelogs_$(date +%Y%m%d_%H%M%S)"

echo -e "${GREEN}=== CHANGELOG Cleanup Script ===${NC}"
echo "Purpose: Remove auto-generated template CHANGELOGs (37-line templates)"
echo "Backup location: $BACKUP_DIR"
echo ""

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Log file for operations
LOG_FILE="$BACKUP_DIR/cleanup.log"
KEPT_FILE="$BACKUP_DIR/kept_changelogs.txt"
DELETED_FILE="$BACKUP_DIR/deleted_changelogs.txt"

echo "Starting cleanup at $(date)" > "$LOG_FILE"

# Find all CHANGELOG.md files
echo -e "${YELLOW}Scanning for CHANGELOG.md files...${NC}"
while IFS= read -r changelog; do
    ((TOTAL_FOUND++))
    
    # Count lines in the file
    line_count=$(wc -l < "$changelog" 2>/dev/null || echo 0)
    
    if [ "$line_count" -eq 37 ]; then
        # Check if it's the auto-generated template
        if grep -q "rule-enforcement-system" "$changelog" 2>/dev/null && \
           grep -q "2025-08-20 14:16:12 UTC" "$changelog" 2>/dev/null; then
            
            # This is an auto-generated file - backup and delete
            echo "Deleting auto-generated: $changelog" >> "$LOG_FILE"
            echo "$changelog" >> "$DELETED_FILE"
            
            # Create backup
            backup_path="$BACKUP_DIR/$(echo "$changelog" | sed 's|/|_|g')"
            cp "$changelog" "$backup_path" 2>/dev/null || {
                echo "Failed to backup: $changelog" >> "$LOG_FILE"
                ((ERROR_COUNT++))
                continue
            }
            
            # Delete the file
            if rm "$changelog" 2>/dev/null; then
                ((DELETED_COUNT++))
                echo -e "  ${RED}[DELETED]${NC} $changelog"
            else
                echo "Failed to delete: $changelog" >> "$LOG_FILE"
                ((ERROR_COUNT++))
            fi
        else
            # 37 lines but not matching the template pattern
            echo "Keeping (37 lines, not template): $changelog" >> "$LOG_FILE"
            echo "$changelog" >> "$KEPT_FILE"
            ((KEPT_COUNT++))
            echo -e "  ${GREEN}[KEPT]${NC} $changelog (37 lines but not auto-generated)"
        fi
    else
        # Not 37 lines - keep it
        echo "Keeping (legitimate): $changelog" >> "$LOG_FILE"
        echo "$changelog" >> "$KEPT_FILE"
        ((KEPT_COUNT++))
        if [ "$line_count" -gt 100 ]; then
            echo -e "  ${GREEN}[KEPT]${NC} $changelog ($line_count lines - substantial content)"
        fi
    fi
done < <(find /opt/sutazaiapp -name "CHANGELOG.md" -type f 2>/dev/null)

echo ""
echo -e "${GREEN}=== Cleanup Summary ===${NC}"
echo "Total CHANGELOGs found: $TOTAL_FOUND"
echo -e "${RED}Deleted (auto-generated):${NC} $DELETED_COUNT"
echo -e "${GREEN}Kept (legitimate):${NC} $KEPT_COUNT"
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}Errors encountered:${NC} $ERROR_COUNT"
fi

echo ""
echo "Backup created at: $BACKUP_DIR"
echo "Deleted files list: $DELETED_FILE"
echo "Kept files list: $KEPT_FILE"
echo "Operation log: $LOG_FILE"

# Verify cleanup
echo ""
echo -e "${YELLOW}Verifying cleanup...${NC}"
remaining_37_line=$(find /opt/sutazaiapp -name "CHANGELOG.md" -type f -exec sh -c 'test $(wc -l < "$1") -eq 37' _ {} \; -print 2>/dev/null | wc -l)

if [ "$remaining_37_line" -eq 0 ]; then
    echo -e "${GREEN}✓ All auto-generated CHANGELOGs successfully removed${NC}"
else
    echo -e "${YELLOW}⚠ Warning: $remaining_37_line 37-line CHANGELOGs still remain${NC}"
    echo "  These may be legitimate 37-line files. Review manually if needed."
fi

echo ""
echo "Cleanup completed at $(date)" >> "$LOG_FILE"
echo -e "${GREEN}Cleanup complete!${NC}"

# Show top-level remaining CHANGELOGs
echo ""
echo -e "${GREEN}=== Key CHANGELOGs Remaining ===${NC}"
for important in \
    "/opt/sutazaiapp/CHANGELOG.md" \
    "/opt/sutazaiapp/backend/CHANGELOG.md" \
    "/opt/sutazaiapp/frontend/CHANGELOG.md" \
    "/opt/sutazaiapp/tests/CHANGELOG.md"; do
    if [ -f "$important" ]; then
        lines=$(wc -l < "$important")
        echo -e "  ${GREEN}✓${NC} $important ($lines lines)"
    else
        echo -e "  ${RED}✗${NC} $important (not found)"
    fi
done

exit 0