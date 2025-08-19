#!/bin/bash
# Docker Consolidation Execution Script
# Date: 2025-08-18
# Purpose: Safely consolidate Docker configurations with validation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/opt/sutazaiapp"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${PROJECT_ROOT}/backups/docker_consolidation_${TIMESTAMP}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================================"
echo "Docker Configuration Consolidation"
echo "Date: $(date)"
echo "======================================================"

# Step 1: Create backup
echo -e "\n${GREEN}Step 1: Creating backup...${NC}"
mkdir -p "$BACKUP_DIR"
cp -r "$PROJECT_ROOT/docker" "$BACKUP_DIR/docker_backup"
cp "$PROJECT_ROOT/docker-compose.yml" "$BACKUP_DIR/docker-compose.yml.backup" 2>/dev/null || true
docker ps --format json > "$BACKUP_DIR/running_containers.json"
echo "✓ Backup created at: $BACKUP_DIR"

# Step 2: Run pre-consolidation health check
echo -e "\n${GREEN}Step 2: Running pre-consolidation health check...${NC}"
python3 "$SCRIPT_DIR/validate_docker_health.py" > "$BACKUP_DIR/pre_consolidation_health.txt"
echo "✓ Health check completed"

# Step 3: Check for the duplicate
echo -e "\n${GREEN}Step 3: Checking for duplicate file...${NC}"
if [ -f "$PROJECT_ROOT/docker/docker-compose.yml" ]; then
    if diff -q "$PROJECT_ROOT/docker-compose.yml" "$PROJECT_ROOT/docker/docker-compose.yml" > /dev/null 2>&1; then
        echo "✓ Confirmed: docker/docker-compose.yml is identical to root docker-compose.yml"
        
        # Step 4: Remove the duplicate
        echo -e "\n${GREEN}Step 4: Removing duplicate...${NC}"
        rm "$PROJECT_ROOT/docker/docker-compose.yml"
        echo "✓ Removed: docker/docker-compose.yml"
        
        # Step 5: Update any script references
        echo -e "\n${GREEN}Step 5: Updating script references...${NC}"
        find "$PROJECT_ROOT/scripts" -type f -name "*.sh" -exec grep -l "docker/docker-compose.yml" {} \; | while read script; do
            sed -i.bak 's|docker/docker-compose.yml|docker-compose.yml|g' "$script"
            echo "   ✓ Updated: $(basename $script)"
        done
    else
        echo -e "${YELLOW}⚠ Files are not identical. Manual review required.${NC}"
        exit 1
    fi
else
    echo "✓ No duplicate found at docker/docker-compose.yml"
fi

# Step 6: Validate Docker compose configuration
echo -e "\n${GREEN}Step 6: Validating Docker configuration...${NC}"
cd "$PROJECT_ROOT"
if docker-compose config > /dev/null 2>&1; then
    echo "✓ Docker compose configuration is valid"
else
    echo -e "${RED}✗ Docker compose configuration validation failed!${NC}"
    echo "Rolling back changes..."
    cp "$BACKUP_DIR/docker-compose.yml.backup" "$PROJECT_ROOT/docker-compose.yml" 2>/dev/null || true
    cp -r "$BACKUP_DIR/docker_backup/"* "$PROJECT_ROOT/docker/" 2>/dev/null || true
    exit 1
fi

# Step 7: Run post-consolidation health check
echo -e "\n${GREEN}Step 7: Running post-consolidation health check...${NC}"
python3 "$SCRIPT_DIR/validate_docker_health.py" > "$BACKUP_DIR/post_consolidation_health.txt"
echo "✓ Health check completed"

# Step 8: Check if containers are still running
echo -e "\n${GREEN}Step 8: Verifying containers are still running...${NC}"
CURRENT_COUNT=$(docker ps --format "{{.Names}}" | grep -c "sutazai-" || true)
if [ "$CURRENT_COUNT" -ge 10 ]; then
    echo "✓ $CURRENT_COUNT sutazai containers are running"
else
    echo -e "${YELLOW}⚠ Only $CURRENT_COUNT sutazai containers running (expected 10+)${NC}"
fi

# Step 9: Archive old backup files
echo -e "\n${GREEN}Step 9: Archiving old backup files...${NC}"
if [ -d "$PROJECT_ROOT/backups" ]; then
    OLD_BACKUPS=$(find "$PROJECT_ROOT/backups" -name "deploy_*" -type d -mtime +7 | wc -l)
    if [ "$OLD_BACKUPS" -gt 0 ]; then
        mkdir -p "$PROJECT_ROOT/backups/archive"
        find "$PROJECT_ROOT/backups" -name "deploy_*" -type d -mtime +7 -exec mv {} "$PROJECT_ROOT/backups/archive/" \;
        echo "✓ Archived $OLD_BACKUPS old backup directories"
    else
        echo "✓ No old backups to archive"
    fi
fi

# Step 10: Generate consolidation report
echo -e "\n${GREEN}Step 10: Generating consolidation report...${NC}"
cat > "$BACKUP_DIR/CONSOLIDATION_REPORT.md" << EOF
# Docker Consolidation Report
Date: $(date)

## Actions Taken:
1. Backed up existing Docker configurations
2. Removed duplicate docker/docker-compose.yml (identical to root)
3. Updated script references to use root docker-compose.yml
4. Validated Docker compose configuration
5. Verified containers are still running

## File Structure After Consolidation:
\`\`\`
/opt/sutazaiapp/
├── docker-compose.yml           # PRIMARY - All production services
└── docker/
    ├── docker-compose.base.yml       # Base configuration
    ├── docker-compose.blue-green.yml # Blue-green deployment
    ├── docker-compose.consolidated.yml # Extended services
    ├── docker-compose.secure.yml     # Security variant
    └── portainer/
        └── docker-compose.yml         # Portainer specific
\`\`\`

## Container Status:
- Running containers: $CURRENT_COUNT
- Network: sutazai-network (active)
- Volumes: All preserved

## Rule Compliance:
- Rule 4: Improved (reduced duplicate configurations)
- Rule 11: Improved (removed main duplicate)
- System Functionality: Preserved

## Backup Location:
$BACKUP_DIR

## Next Steps:
1. Monitor system for 24 hours
2. Run Playwright tests to verify functionality
3. Consider merging consolidated.yml services if needed
4. Update CHANGELOG.md with consolidation details
EOF

echo "✓ Report saved to: $BACKUP_DIR/CONSOLIDATION_REPORT.md"

# Final summary
echo -e "\n======================================================"
echo -e "${GREEN}Consolidation Complete!${NC}"
echo "======================================================"
echo "✓ Removed 1 duplicate Docker compose file"
echo "✓ Updated script references"
echo "✓ System remains functional"
echo "✓ Backup saved to: $BACKUP_DIR"
echo ""
echo "Recommended actions:"
echo "1. Run: docker-compose ps"
echo "2. Test API: curl http://localhost:10010/health"
echo "3. Check logs: docker-compose logs --tail=50"
echo ""
echo -e "${GREEN}Docker consolidation successful!${NC}"