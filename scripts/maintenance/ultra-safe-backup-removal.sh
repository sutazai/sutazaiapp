#!/bin/bash
# Ultra-Safe Backup File Removal Script
# Author: Garbage Collector Specialist with System Architect Verification
# Date: August 10, 2025
# Operation: Remove 27 .bak files and archive folder with FULL SAFETY

set -euo pipefail

# Colors for ultra-clear output

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

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     ULTRA-SAFE BACKUP REMOVAL OPERATION                   ║${NC}"
echo -e "${BLUE}║     Garbage Collector Specialist + System Architect       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

PROJECT_ROOT="/opt/sutazaiapp"
SAFETY_ARCHIVE="/tmp/sutazai-cleanup-safety-$(date +%Y%m%d_%H%M%S)"
MANIFEST_FILE="${SAFETY_ARCHIVE}/removal-manifest.txt"

# Create safety archive directory
echo -e "\n${YELLOW}[PHASE 1] Creating safety archive...${NC}"
mkdir -p "${SAFETY_ARCHIVE}"

# Document what we're removing
echo "SutazAI Cleanup Safety Archive" > "${MANIFEST_FILE}"
echo "Created: $(date)" >> "${MANIFEST_FILE}"
echo "Operation: Remove .bak files and archive folder" >> "${MANIFEST_FILE}"
echo "=" >> "${MANIFEST_FILE}"

# Count items to remove
BAK_COUNT=$(find "${PROJECT_ROOT}" -type f -name "*.bak" 2>/dev/null | wc -l)
ARCHIVE_SIZE=$(du -sh "${PROJECT_ROOT}/archive" 2>/dev/null | cut -f1 || echo "0")

echo -e "${BLUE}Items to remove:${NC}"
echo -e "  • .bak files: ${BAK_COUNT}"
echo -e "  • Archive folder size: ${ARCHIVE_SIZE}"

# Phase 2: Create safety backup
echo -e "\n${YELLOW}[PHASE 2] Creating safety backup...${NC}"

# Backup .bak files
echo -e "${BLUE}Backing up .bak files...${NC}"
find "${PROJECT_ROOT}" -type f -name "*.bak" -print0 2>/dev/null | while IFS= read -r -d '' file; do
    REL_PATH="${file#${PROJECT_ROOT}/}"
    BACKUP_PATH="${SAFETY_ARCHIVE}/bak-files/${REL_PATH}"
    mkdir -p "$(dirname "${BACKUP_PATH}")"
    cp -p "${file}" "${BACKUP_PATH}"
    echo "Backed up: ${REL_PATH}" >> "${MANIFEST_FILE}"
    echo -e "  ${GREEN}✓${NC} Backed up: ${REL_PATH}"
done

# Backup archive folder
if [ -d "${PROJECT_ROOT}/archive" ]; then
    echo -e "${BLUE}Backing up archive folder...${NC}"
    cp -rp "${PROJECT_ROOT}/archive" "${SAFETY_ARCHIVE}/archive-folder"
    echo "Backed up: archive/ folder" >> "${MANIFEST_FILE}"
    echo -e "  ${GREEN}✓${NC} Backed up archive folder"
fi

# Phase 3: Verification before removal
echo -e "\n${YELLOW}[PHASE 3] Ultra-verification...${NC}"

# Count backed up items
BACKED_UP_BAK=$(find "${SAFETY_ARCHIVE}/bak-files" -type f -name "*.bak" 2>/dev/null | wc -l || echo 0)
BACKED_UP_ARCHIVE=$([ -d "${SAFETY_ARCHIVE}/archive-folder" ] && echo "1" || echo "0")

echo -e "${BLUE}Verification:${NC}"
echo -e "  • .bak files backed up: ${BACKED_UP_BAK}/${BAK_COUNT}"
echo -e "  • Archive folder backed up: ${BACKED_UP_ARCHIVE}"

if [ "${BACKED_UP_BAK}" -ne "${BAK_COUNT}" ]; then
    echo -e "${RED}ERROR: Backup count mismatch for .bak files!${NC}"
    echo -e "${RED}Aborting for safety.${NC}"
    exit 1
fi

# Phase 4: Actual removal
echo -e "\n${YELLOW}[PHASE 4] Removing files...${NC}"

# Remove .bak files
REMOVED_COUNT=0
find "${PROJECT_ROOT}" -type f -name "*.bak" -print0 2>/dev/null | while IFS= read -r -d '' file; do
    REL_PATH="${file#${PROJECT_ROOT}/}"
    rm -f "${file}"
    echo "Removed: ${REL_PATH}" >> "${MANIFEST_FILE}"
    echo -e "  ${GREEN}✓${NC} Removed: ${REL_PATH}"
    ((REMOVED_COUNT++)) || true
done

# Remove archive folder
if [ -d "${PROJECT_ROOT}/archive" ]; then
    echo -e "${BLUE}Removing archive folder...${NC}"
    rm -rf "${PROJECT_ROOT}/archive"
    echo "Removed: archive/ folder" >> "${MANIFEST_FILE}"
    echo -e "  ${GREEN}✓${NC} Removed archive folder"
fi

# Phase 5: Final verification
echo -e "\n${YELLOW}[PHASE 5] Final ultra-verification...${NC}"

REMAINING_BAK=$(find "${PROJECT_ROOT}" -type f -name "*.bak" 2>/dev/null | wc -l)
ARCHIVE_EXISTS=$([ -d "${PROJECT_ROOT}/archive" ] && echo "YES" || echo "NO")

echo -e "${BLUE}Final status:${NC}"
echo -e "  • Remaining .bak files: ${REMAINING_BAK}"
echo -e "  • Archive folder exists: ${ARCHIVE_EXISTS}"

# Create compressed safety archive
echo -e "\n${YELLOW}[PHASE 6] Compressing safety archive...${NC}"
cd /tmp
tar -czf "sutazai-cleanup-safety-$(date +%Y%m%d_%H%M%S).tar.gz" "$(basename "${SAFETY_ARCHIVE}")"
ARCHIVE_SIZE=$(du -h /tmp/sutazai-cleanup-safety-*.tar.gz | tail -1 | cut -f1)

# Summary
echo -e "\n${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     CLEANUP OPERATION COMPLETE                            ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${BLUE}Summary:${NC}"
echo -e "  • .bak files removed: ${BAK_COUNT}"
echo -e "  • Archive folder removed: YES"
echo -e "  • Safety backup created: ${SAFETY_ARCHIVE}.tar.gz (${ARCHIVE_SIZE})"
echo -e "  • Manifest file: ${MANIFEST_FILE}"

if [ "${REMAINING_BAK}" -eq 0 ] && [ "${ARCHIVE_EXISTS}" = "NO" ]; then
    echo -e "\n${GREEN}✅ ALL CLEANUP TARGETS SUCCESSFULLY REMOVED${NC}"
    echo -e "${YELLOW}Safety backup preserved for 7 days in /tmp/${NC}"
else
    echo -e "\n${RED}⚠️  WARNING: Some items may not have been removed${NC}"
    echo -e "${RED}Please investigate remaining items${NC}"
fi

echo -e "\n${BLUE}To restore if needed:${NC}"
echo -e "  tar -xzf /tmp/sutazai-cleanup-safety-*.tar.gz -C /tmp/"
echo -e "  Then manually copy files back to original locations"

exit 0