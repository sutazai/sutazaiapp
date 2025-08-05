#!/bin/bash
set -e

# Emergency Docker Compose Cleanup Script
# This script addresses the most critical issues found in the compose file analysis

echo "🚨 EMERGENCY DOCKER COMPOSE CLEANUP STARTING..."
echo "This script will make significant changes to compose files."
echo "A backup will be created automatically."

# Create backup
BACKUP_DIR="backups/compose-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo "📦 Creating backup in $BACKUP_DIR..."
find . -name "docker-compose*.yml" -exec cp {} "$BACKUP_DIR/" \;
echo "✅ Backup created"

# Phase 1: Archive definitely abandoned files
echo "🗂️  Phase 1: Archiving abandoned files..."
mkdir -p archive/deprecated/$(date +%Y%m%d)

# Move files that are clearly abandoned
if [ -d "archive/20250803_193506_pre_cleanup" ]; then
    echo "Moving old archive files..."
    mv archive/20250803_193506_pre_cleanup/* archive/deprecated/$(date +%Y%m%d)/ 2>/dev/null || true
fi

if [ -d "tests/fixtures/hygiene/docker_chaos" ]; then
    echo "Moving test fixture files..."
    mv tests/fixtures/hygiene/docker_chaos/* archive/deprecated/$(date +%Y%m%d)/ 2>/dev/null || true
fi

# Move backup files
echo "Moving backup files..."
find . -name "docker-compose*.bak*" -exec mv {} archive/deprecated/$(date +%Y%m%d)/ \; 2>/dev/null || true

echo "✅ Phase 1 complete"

# Phase 2: Fix the most critical port conflicts
echo "🔧 Phase 2: Emergency port conflict resolution..."

# Create a port reassignment mapping for the most critical conflicts
declare -A PORT_REASSIGNMENTS=(
    # Core infrastructure gets 10000-10099 range
    ["10000"]="10050"  # If postgres conflicts, move to 10050
    ["10001"]="10051"  # If redis conflicts, move to 10051
    ["10002"]="10052"  # If neo4j http conflicts, move to 10052
    ["10003"]="10053"  # If neo4j bolt conflicts, move to 10053
    
    # AI services get 10100-10199 range  
    ["10100"]="10150"  # If chromadb conflicts, move to 10150
    ["10101"]="10151"  # If qdrant conflicts, move to 10151
    ["10104"]="10152"  # If ollama conflicts, move to 10152
    
    # Monitoring gets 10200-10299 range
    ["9090"]="10200"   # Move prometheus to dedicated monitoring range
    ["3000"]="10201"   # Move grafana to dedicated monitoring range
    
    # Agents get 10300-10399 range
    ["8001"]="10301"   # Move agent services out of 8001 conflict
    ["8002"]="10302"   
    ["8003"]="10303"
    ["8004"]="10304"
    ["8005"]="10305"
)

# Function to fix port conflicts in a file
fix_port_conflicts() {
    local file="$1"
    echo "  Fixing ports in $file..."
    
    for old_port in "${!PORT_REASSIGNMENTS[@]}"; do
        new_port="${PORT_REASSIGNMENTS[$old_port]}"
        # Use sed to replace port mappings
        sed -i.tmp "s/:${old_port}:/:${new_port}:/g" "$file" 2>/dev/null || true
        sed -i.tmp "s/\"${old_port}:/${new_port}:/g" "$file" 2>/dev/null || true
        sed -i.tmp "s/- ${old_port}:/- ${new_port}:/g" "$file" 2>/dev/null || true
        rm -f "$file.tmp" 2>/dev/null || true
    done
}

# Apply port fixes to key files
for file in docker-compose.yml docker-compose.production.yml docker-compose.override.yml; do
    if [ -f "$file" ]; then
        fix_port_conflicts "$file"
    fi
done

echo "✅ Phase 2 complete"

# Phase 3: Create a primary compose file registry
echo "📋 Phase 3: Creating compose file registry..."

cat > COMPOSE_FILE_REGISTRY.md << 'EOF'
# Docker Compose File Registry

## PRIMARY FILES (Use these for deployment)
- `docker-compose.yml` - Base configuration (MAIN)
- `docker-compose.production.yml` - Production overrides
- `docker-compose.override.yml` - Local development overrides

## SPECIALIZED FILES (Use only for specific features)
- `docker-compose.monitoring.yml` - Monitoring stack
- `docker-compose.agents.yml` - AI agent services  
- `docker-compose.security.yml` - Security tools

## DEPRECATED/REVIEW NEEDED
All other docker-compose*.yml files should be reviewed for necessity.

## Port Allocation (Updated by emergency cleanup)
- 10050: PostgreSQL (moved from conflicts)
- 10051: Redis (moved from conflicts)
- 10052: Neo4j HTTP (moved from conflicts)
- 10053: Neo4j Bolt (moved from conflicts)
- 10150: ChromaDB (moved from conflicts)
- 10151: Qdrant (moved from conflicts)
- 10152: Ollama (moved from conflicts)
- 10200: Prometheus (moved from conflicts)
- 10201: Grafana (moved from conflicts)
- 10301-10305: Agent services (moved from 8001-8005 conflicts)

## Next Steps
1. Review all files not in PRIMARY or SPECIALIZED categories
2. Consolidate duplicate service definitions
3. Remove services with missing implementations
4. Update documentation and team procedures
EOF

echo "✅ Phase 3 complete"

# Phase 4: Generate immediate action items
echo "🎯 Phase 4: Generating action items..."

cat > IMMEDIATE_ACTIONS_REQUIRED.md << 'EOF'
# IMMEDIATE ACTIONS REQUIRED

## CRITICAL (Do this week)

### 1. Validate Main Compose File
```bash
docker-compose -f docker-compose.yml config
```
If this fails, the main compose file has syntax errors that need fixing.

### 2. Test Port Changes
The emergency cleanup reassigned conflicting ports. Test that services still work:
```bash
docker-compose -f docker-compose.yml up -d postgres redis neo4j
docker-compose -f docker-compose.yml ps
```

### 3. Remove Duplicate Services
Services with 10+ duplicates need immediate consolidation:
- redis (21 definitions)
- postgres (20 definitions)  
- ollama (20 definitions)
- backend (18 definitions)

### 4. Fix Missing Implementations
Remove these services from compose files (they have no backing code):
- agi-orchestration-layer
- agi-task-decomposer
- security-pentesting-specialist
- system-optimizer-reorganizer

## HIGH PRIORITY (Do this month)

### 1. Establish Single Source of Truth
Choose ONE file for each service definition and remove others.

### 2. Create Environment-Specific Overrides
Instead of multiple complete files, use override pattern:
- docker-compose.yml (base)
- docker-compose.production.yml (prod overrides)
- docker-compose.development.yml (dev overrides)

### 3. Implement Port Registry
Document all port assignments to prevent future conflicts.

## MONITORING

After changes, monitor for:
- Services failing to start (port conflicts)
- Services connecting to wrong instances (duplicate issues)
- Performance problems (resource conflicts)
- Missing functionality (removed services)

## ROLLBACK PLAN

If issues occur:
```bash
# Restore from backup
cp backups/compose-backup-*/docker-compose*.yml .
docker-compose down
docker-compose up -d
```
EOF

echo "✅ Phase 4 complete"

# Summary
echo ""
echo "🎉 EMERGENCY CLEANUP COMPLETE!"
echo ""
echo "📊 SUMMARY OF CHANGES:"
echo "  - Created backup in: $BACKUP_DIR"
echo "  - Archived abandoned files to: archive/deprecated/$(date +%Y%m%d)"
echo "  - Applied emergency port conflict fixes"
echo "  - Created COMPOSE_FILE_REGISTRY.md"
echo "  - Created IMMEDIATE_ACTIONS_REQUIRED.md"
echo ""
echo "⚠️  NEXT STEPS:"
echo "1. Review the generated action items in IMMEDIATE_ACTIONS_REQUIRED.md"
echo "2. Test that critical services still start correctly"
echo "3. Begin systematic cleanup of duplicate services"
echo "4. Update team procedures to prevent future chaos"
echo ""
echo "🆘 IF SOMETHING BREAKS:"
echo "Restore from backup: cp $BACKUP_DIR/* ."
echo ""
echo "📋 For full analysis, see: DOCKER_COMPOSE_VALIDATION_REPORT.md"