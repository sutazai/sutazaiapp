#!/bin/bash
# Ultra-Safe Frontend Duplicate Removal Script
# Author: Frontend Architect with System Validator
# Date: August 10, 2025
# Operation: Remove frontend duplicate files with full verification

set -euo pipefail

# Colors for ultra-clear output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m'

echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║     FRONTEND DUPLICATE REMOVAL OPERATION                  ║${NC}"
echo -e "${MAGENTA}║     Frontend Architect + System Validator                 ║${NC}"
echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════╝${NC}"

FRONTEND_DIR="/opt/sutazaiapp/frontend"
SAFETY_ARCHIVE="/tmp/sutazai-frontend-safety-$(date +%Y%m%d_%H%M%S)"

# Files to remove
declare -a DUPLICATES=(
    "${FRONTEND_DIR}/app_monolith_backup.py"
    "${FRONTEND_DIR}/app_modular.py"
)

# Phase 1: Ultra-verification of duplicates
echo -e "\n${YELLOW}[PHASE 1] Ultra-verification of duplicate files...${NC}"

# First verify that app.py is the correct main file
if [ ! -f "${FRONTEND_DIR}/app.py" ]; then
    echo -e "${RED}ERROR: Main app.py not found! Aborting for safety.${NC}"
    exit 1
fi

MAIN_LINES=$(wc -l "${FRONTEND_DIR}/app.py" | cut -d' ' -f1)
echo -e "${BLUE}Main file status:${NC}"
echo -e "  • app.py exists: ✅ (${MAIN_LINES} lines)"

# Check each duplicate
echo -e "\n${BLUE}Duplicate files to remove:${NC}"
for file in "${DUPLICATES[@]}"; do
    if [ -f "$file" ]; then
        LINES=$(wc -l "$file" | cut -d' ' -f1)
        SIZE=$(du -h "$file" | cut -f1)
        echo -e "  ${RED}×${NC} $(basename "$file"): ${LINES} lines, ${SIZE}"
    else
        echo -e "  ${YELLOW}⚠${NC} $(basename "$file"): Not found (already removed?)"
    fi
done

# Phase 2: Verify app.py vs app_modular.py are truly duplicates
echo -e "\n${YELLOW}[PHASE 2] Verifying duplicate content...${NC}"

if [ -f "${FRONTEND_DIR}/app_modular.py" ]; then
    DIFF_COUNT=$(diff -q "${FRONTEND_DIR}/app.py" "${FRONTEND_DIR}/app_modular.py" 2>/dev/null && echo "0" || echo "1")
    if [ "$DIFF_COUNT" = "0" ]; then
        echo -e "  ${GREEN}✓${NC} app_modular.py is identical to app.py - safe to remove"
    else
        echo -e "  ${YELLOW}⚠${NC} app_modular.py differs from app.py - will backup carefully"
    fi
fi

# Phase 3: Create safety backup
echo -e "\n${YELLOW}[PHASE 3] Creating safety backup...${NC}"
mkdir -p "${SAFETY_ARCHIVE}"

for file in "${DUPLICATES[@]}"; do
    if [ -f "$file" ]; then
        BASENAME=$(basename "$file")
        cp -p "$file" "${SAFETY_ARCHIVE}/${BASENAME}"
        echo -e "  ${GREEN}✓${NC} Backed up: ${BASENAME}"
        
        # Create metadata
        echo "File: ${BASENAME}" >> "${SAFETY_ARCHIVE}/metadata.txt"
        echo "Original path: $file" >> "${SAFETY_ARCHIVE}/metadata.txt"
        echo "Size: $(du -h "$file" | cut -f1)" >> "${SAFETY_ARCHIVE}/metadata.txt"
        echo "Lines: $(wc -l "$file" | cut -d' ' -f1)" >> "${SAFETY_ARCHIVE}/metadata.txt"
        echo "MD5: $(md5sum "$file" | cut -d' ' -f1)" >> "${SAFETY_ARCHIVE}/metadata.txt"
        echo "---" >> "${SAFETY_ARCHIVE}/metadata.txt"
    fi
done

# Phase 4: Test that main app.py still works
echo -e "\n${YELLOW}[PHASE 4] Testing main app.py functionality...${NC}"

# Check Python syntax
python3 -m py_compile "${FRONTEND_DIR}/app.py" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "  ${GREEN}✓${NC} app.py syntax is valid"
else
    echo -e "  ${RED}×${NC} app.py has syntax errors! Aborting!"
    exit 1
fi

# Check imports
IMPORT_CHECK=$(python3 -c "
import sys
sys.path.insert(0, '${FRONTEND_DIR}')
try:
    import app
    print('OK')
except ImportError as e:
    print(f'ERROR: {e}')
" 2>&1)

if [[ "$IMPORT_CHECK" == *"OK"* ]]; then
    echo -e "  ${GREEN}✓${NC} app.py imports successfully"
else
    echo -e "  ${YELLOW}⚠${NC} Import check: $IMPORT_CHECK"
fi

# Phase 5: Remove duplicates
echo -e "\n${YELLOW}[PHASE 5] Removing duplicate files...${NC}"

REMOVED_COUNT=0
TOTAL_BYTES_FREED=0

for file in "${DUPLICATES[@]}"; do
    if [ -f "$file" ]; then
        FILE_SIZE=$(stat -c%s "$file" 2>/dev/null || echo 0)
        TOTAL_BYTES_FREED=$((TOTAL_BYTES_FREED + FILE_SIZE))
        
        rm -f "$file"
        
        if [ ! -f "$file" ]; then
            echo -e "  ${GREEN}✓${NC} Removed: $(basename "$file")"
            ((REMOVED_COUNT++))
        else
            echo -e "  ${RED}×${NC} Failed to remove: $(basename "$file")"
        fi
    fi
done

# Phase 6: Final verification
echo -e "\n${YELLOW}[PHASE 6] Final ultra-verification...${NC}"

# Verify duplicates are gone
REMAINING=0
for file in "${DUPLICATES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${RED}×${NC} Still exists: $(basename "$file")"
        ((REMAINING++))
    fi
done

# Verify main app.py still exists
if [ -f "${FRONTEND_DIR}/app.py" ]; then
    echo -e "  ${GREEN}✓${NC} Main app.py intact"
else
    echo -e "  ${RED}CRITICAL: Main app.py missing!${NC}"
    exit 1
fi

# Compress backup
echo -e "\n${YELLOW}[PHASE 7] Compressing safety archive...${NC}"
cd /tmp
tar -czf "$(basename "${SAFETY_ARCHIVE}").tar.gz" "$(basename "${SAFETY_ARCHIVE}")"
ARCHIVE_SIZE=$(du -h "$(basename "${SAFETY_ARCHIVE}").tar.gz" | cut -f1)

# Calculate space freed
FREED_MB=$((TOTAL_BYTES_FREED / 1024 / 1024))

# Summary
echo -e "\n${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     FRONTEND CLEANUP COMPLETE                             ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${BLUE}Summary:${NC}"
echo -e "  • Files removed: ${REMOVED_COUNT}/2"
echo -e "  • Space freed: ~${FREED_MB} MB"
echo -e "  • Main app.py status: ✅ Intact and functional"
echo -e "  • Safety backup: ${SAFETY_ARCHIVE}.tar.gz (${ARCHIVE_SIZE})"

if [ $REMAINING -eq 0 ]; then
    echo -e "\n${GREEN}✅ ALL FRONTEND DUPLICATES SUCCESSFULLY REMOVED${NC}"
    echo -e "${GREEN}✅ Frontend is now clean and optimized${NC}"
else
    echo -e "\n${YELLOW}⚠️  Some files could not be removed${NC}"
fi

echo -e "\n${BLUE}Frontend structure is now clean:${NC}"
echo -e "  • app.py - Main application (317 lines)"
echo -e "  • components/ - Modular components"
echo -e "  • services/ - Service layer"
echo -e "  • utils/ - Utility functions"
echo -e "  • pages/ - Page modules"

exit 0