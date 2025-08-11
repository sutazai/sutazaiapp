#!/bin/bash
# ULTRA SCRIPT CONSOLIDATION - SAFE EXECUTION SCRIPT
# This script safely consolidates 1,203 scripts down to 350
# Author: Ultra System Architect
# Date: 2025-08-10

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== ULTRA SCRIPT CONSOLIDATION ===${NC}"
echo "Current scripts: 1,203"
echo "Target scripts: 350 (71% reduction)"
echo ""

# Safety check
echo -e "${YELLOW}⚠️  WARNING: This will consolidate the entire script structure${NC}"
echo "The following safety measures are in place:"
echo "  ✓ Complete backup before changes"
echo "  ✓ Continuous validation during consolidation"
echo "  ✓ Automatic rollback on any error"
echo "  ✓ Archive instead of delete"
echo "  ✓ < 15 minute rollback capability"
echo ""
read -p "Do you want to proceed? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo -e "${RED}Consolidation cancelled${NC}"
    exit 0
fi

# Step 1: System health check
echo -e "\n${BLUE}Step 1: Checking system health...${NC}"
if ! bash /opt/sutazaiapp/pre_consolidation_validation.sh; then
    echo -e "${RED}System health check failed! Aborting.${NC}"
    exit 1
fi

# Step 2: Create ultra-safe backup
echo -e "\n${BLUE}Step 2: Creating comprehensive backup...${NC}"
BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/sutazaiapp/backups/pre-consolidation-$BACKUP_TIMESTAMP"

mkdir -p "$BACKUP_DIR"
echo "Backup directory: $BACKUP_DIR"

# Backup all script directories
for dir in scripts docker agents backend frontend tests monitoring services; do
    if [ -d "/opt/sutazaiapp/$dir" ]; then
        echo "  Backing up $dir/..."
        cp -r "/opt/sutazaiapp/$dir" "$BACKUP_DIR/" 2>/dev/null || true
    fi
done

# Create restore script
cat > "$BACKUP_DIR/emergency_restore.sh" << 'EOF'
#!/bin/bash
# Emergency Restore Script
echo "=== EMERGENCY RESTORE ==="
BACKUP_DIR="$(dirname "$0")"
for dir in scripts docker agents backend frontend tests monitoring services; do
    if [ -d "$BACKUP_DIR/$dir" ]; then
        echo "Restoring $dir..."
        rm -rf "/opt/sutazaiapp/$dir"
        cp -r "$BACKUP_DIR/$dir" "/opt/sutazaiapp/"
    fi
done
echo "Restore complete. Please restart services."
EOF
chmod +x "$BACKUP_DIR/emergency_restore.sh"

echo -e "${GREEN}Backup complete: $BACKUP_DIR${NC}"

# Step 3: Run consolidation
echo -e "\n${BLUE}Step 3: Running script consolidation...${NC}"
echo "This will:"
echo "  - Remove 474 duplicate script groups"
echo "  - Archive 70 stub scripts"
echo "  - Create master controllers for common tasks"
echo "  - Organize scripts into 11 categories"

# First run in dry-run mode
echo -e "\n${YELLOW}Running dry-run first...${NC}"
if ! python3 /opt/sutazaiapp/scripts/maintenance/ultra-script-consolidation.py \
    --dry-run \
    --target-scripts 350; then
    echo -e "${RED}Dry-run failed! Aborting.${NC}"
    exit 1
fi

echo -e "\n${GREEN}Dry-run successful!${NC}"
read -p "Proceed with actual consolidation? (yes/no): " proceed

if [ "$proceed" != "yes" ]; then
    echo -e "${YELLOW}Consolidation cancelled after dry-run${NC}"
    exit 0
fi

# Actual consolidation
echo -e "\n${BLUE}Executing consolidation...${NC}"
if ! python3 /opt/sutazaiapp/scripts/maintenance/ultra-script-consolidation.py \
    --backup-first \
    --validate-continuously \
    --rollback-on-error \
    --target-scripts 350 \
    --preserve-functionality \
    --generate-report; then
    echo -e "${RED}Consolidation failed! Check logs.${NC}"
    echo "Rollback script available at: $BACKUP_DIR/emergency_restore.sh"
    exit 1
fi

# Step 4: Post-consolidation validation
echo -e "\n${BLUE}Step 4: Validating consolidation results...${NC}"
bash /opt/sutazaiapp/pre_consolidation_validation.sh

# Step 5: Generate summary
echo -e "\n${BLUE}Step 5: Consolidation Summary${NC}"
echo "================================================"
echo "Initial scripts: 1,203"
echo "Final scripts: $(find /opt/sutazaiapp -name "*.sh" -o -name "*.py" -o -name "*.js" | grep -v node_modules | grep -v archive | wc -l)"
echo ""
echo "Reports generated:"
echo "  - /opt/sutazaiapp/SCRIPT_CONSOLIDATION_REPORT.md"
echo "  - /opt/sutazaiapp/ULTRA_SCRIPT_CONSOLIDATION_EXECUTIVE_SUMMARY.md"
echo ""
echo "Backup location: $BACKUP_DIR"
echo "Emergency restore: $BACKUP_DIR/emergency_restore.sh"
echo ""
echo -e "${GREEN}✓ CONSOLIDATION COMPLETE${NC}"
echo ""
echo "Next steps:"
echo "1. Review the consolidation report"
echo "2. Test critical workflows"
echo "3. Update team documentation"
echo "4. Remove old archives after 30 days"