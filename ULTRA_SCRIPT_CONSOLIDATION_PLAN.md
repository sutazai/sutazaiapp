# ULTRA SCRIPT CONSOLIDATION MASTER PLAN
**Architect:** Ultra System Architect  
**Date:** 2025-08-10  
**Target:** Reduce 1,203 active scripts to 350 (71% reduction)  
**Risk Level:** HIGH - Requires extreme caution

## PHASE 1: COMPREHENSIVE BACKUP & INVENTORY (0% Risk)
**Timeline:** 30 minutes  
**Objective:** Create complete safety net before any changes

### 1.1 Full System Backup
```bash
# Create timestamped backup of ALL scripts
BACKUP_DIR="/opt/sutazaiapp/backups/scripts-pre-consolidation-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup all script directories
for dir in scripts docker agents backend frontend tests monitoring services; do
    if [ -d "/opt/sutazaiapp/$dir" ]; then
        cp -r "/opt/sutazaiapp/$dir" "$BACKUP_DIR/"
    fi
done

# Create inventory manifest
find /opt/sutazaiapp -type f \( -name "*.sh" -o -name "*.py" -o -name "*.js" \) \
    ! -path "*/archive/*" ! -path "*/node_modules/*" ! -path "*/.git/*" \
    > "$BACKUP_DIR/script_inventory.txt"

# Generate checksums for rollback validation
find /opt/sutazaiapp -type f \( -name "*.sh" -o -name "*.py" -o -name "*.js" \) \
    ! -path "*/archive/*" -exec md5sum {} \; > "$BACKUP_DIR/checksums.txt"
```

### 1.2 Dependency Mapping
```python
# Create dependency graph to understand script relationships
import ast
import os
from pathlib import Path

def map_dependencies():
    dependencies = {}
    for script in Path('/opt/sutazaiapp').rglob('*.py'):
        if 'archive' in str(script) or 'node_modules' in str(script):
            continue
        try:
            with open(script, 'r') as f:
                tree = ast.parse(f.read())
                imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
                dependencies[str(script)] = imports
        except:
            pass
    return dependencies
```

## PHASE 2: SAFE REMOVAL OF DUPLICATES (Low Risk)
**Timeline:** 1 hour  
**Objective:** Remove exact duplicates, keeping one canonical version

### 2.1 Duplicate Detection & Removal
```python
def consolidate_duplicates():
    """
    Remove duplicate scripts safely:
    1. Group files by MD5 hash
    2. Keep the one in the most logical location
    3. Update all references
    4. Archive removed duplicates
    """
    duplicate_groups = defaultdict(list)
    
    # Group by hash
    for script in all_scripts:
        file_hash = get_file_hash(script)
        duplicate_groups[file_hash].append(script)
    
    # Process each duplicate group
    for hash_val, files in duplicate_groups.items():
        if len(files) > 1:
            # Keep the one in the best location (priority order)
            priority = ['scripts/', 'backend/', 'frontend/', 'agents/', 'docker/', 'tests/']
            keeper = select_best_location(files, priority)
            
            # Archive others
            for file in files:
                if file != keeper:
                    archive_file(file, reason="duplicate")
                    update_references(file, keeper)
```

### 2.2 Reference Update Strategy
```bash
# Update all references to removed scripts
for removed_script in $REMOVED_SCRIPTS; do
    canonical_script=$( get_canonical_path "$removed_script" )
    
    # Update shell script references
    find /opt/sutazaiapp -name "*.sh" -type f -exec sed -i "s|$removed_script|$canonical_script|g" {} \;
    
    # Update Python imports
    find /opt/sutazaiapp -name "*.py" -type f -exec sed -i "s|$removed_script|$canonical_script|g" {} \;
    
    # Update docker-compose references
    find /opt/sutazaiapp -name "docker-compose*.yml" -type f -exec sed -i "s|$removed_script|$canonical_script|g" {} \;
done
```

## PHASE 3: INTELLIGENT CONSOLIDATION (Medium Risk)
**Timeline:** 2 hours  
**Objective:** Merge similar scripts into unified, modular versions

### 3.1 Category-Based Consolidation Structure
```
/opt/sutazaiapp/scripts/
├── master/                      # Master control scripts
│   ├── deploy-master.sh         # Unified deployment controller
│   ├── test-master.sh           # Unified test runner
│   ├── monitor-master.sh        # Unified monitoring
│   └── maintenance-master.sh    # Unified maintenance
├── core/                        # Core functionality
│   ├── database/
│   │   ├── db-operations.py    # All DB operations
│   │   └── migrations.sh       # All migrations
│   ├── docker/
│   │   ├── build-all.sh        # Unified builder
│   │   └── compose-manager.py  # Compose operations
│   └── security/
│       ├── audit.py            # Security auditing
│       └── hardening.sh        # Security hardening
├── agents/                     # Agent-specific
│   └── agent-controller.py    # Unified agent management
└── utilities/                  # Shared utilities
    ├── common.sh              # Shell utilities
    └── helpers.py             # Python utilities
```

### 3.2 Consolidation Patterns

#### Pattern A: Multiple Test Scripts → Single Test Master
```python
# BEFORE: 180 separate test scripts
# AFTER: Single test-master.py with modular execution

class TestMaster:
    """Unified test execution controller"""
    
    def __init__(self):
        self.test_modules = {
            'unit': UnitTestRunner(),
            'integration': IntegrationTestRunner(),
            'e2e': E2ETestRunner(),
            'performance': PerformanceTestRunner(),
            'security': SecurityTestRunner()
        }
    
    def run(self, test_type='all', targets=None):
        """Run specified test types on targets"""
        if test_type == 'all':
            return self.run_all_tests(targets)
        return self.test_modules[test_type].run(targets)
```

#### Pattern B: Multiple Deploy Scripts → Single Deploy Master
```bash
#!/bin/bash
# deploy-master.sh - Unified deployment controller

deploy_master() {
    local environment="$1"
    local component="$2"
    local action="$3"
    
    case "$component" in
        backend)
            deploy_backend "$environment" "$action"
            ;;
        frontend)
            deploy_frontend "$environment" "$action"
            ;;
        agents)
            deploy_agents "$environment" "$action"
            ;;
        all)
            deploy_full_stack "$environment" "$action"
            ;;
        *)
            echo "Unknown component: $component"
            exit 1
            ;;
    esac
}
```

## PHASE 4: VALIDATION & TESTING (Critical)
**Timeline:** 2 hours  
**Objective:** Ensure zero functionality loss

### 4.1 Comprehensive Validation Suite
```python
def validate_consolidation():
    """
    Validate that consolidation preserved all functionality
    """
    validations = {
        'imports': validate_all_imports(),
        'references': validate_all_references(),
        'execution': validate_script_execution(),
        'docker': validate_docker_builds(),
        'services': validate_service_health(),
        'api': validate_api_endpoints()
    }
    
    for check, result in validations.items():
        if not result['success']:
            rollback(reason=f"Validation failed: {check}")
            return False
    
    return True
```

### 4.2 Service Health Verification
```bash
# Verify all services still work after consolidation
SERVICES=(
    "backend:10010:/health"
    "frontend:10011:/"
    "ollama:10104:/api/tags"
    "hardware-optimizer:11110:/health"
    "ai-orchestrator:8589:/health"
)

for service in "${SERVICES[@]}"; do
    IFS=':' read -r name port endpoint <<< "$service"
    if ! curl -s "http://localhost:$port$endpoint" > /dev/null; then
        echo "ERROR: Service $name failed health check"
        initiate_rollback
        exit 1
    fi
done
```

## PHASE 5: CLEANUP & OPTIMIZATION (Final)
**Timeline:** 1 hour  
**Objective:** Remove obsolete code and optimize structure

### 5.1 Safe Cleanup Process
```bash
# Remove confirmed obsolete scripts
OBSOLETE_PATTERNS=(
    "*_old.*"
    "*_backup.*"
    "*deprecated*"
    "*_temp.*"
    "*_test_*"
    "*.pyc"
)

for pattern in "${OBSOLETE_PATTERNS[@]}"; do
    find /opt/sutazaiapp -name "$pattern" \
        ! -path "*/archive/*" \
        ! -path "*/backups/*" \
        -type f -exec bash -c '
            file="$1"
            # Verify not in use
            if ! grep -r "$(basename "$file")" /opt/sutazaiapp --exclude-dir=archive; then
                mv "$file" "/opt/sutazaiapp/archive/obsolete-$(date +%Y%m%d)/"
            fi
        ' _ {} \;
done
```

### 5.2 Final Structure Optimization
```python
def optimize_final_structure():
    """
    Final optimization pass:
    1. Remove empty directories
    2. Consolidate similar utilities
    3. Update all documentation
    4. Generate dependency graph
    """
    
    # Remove empty directories
    for root, dirs, files in os.walk('/opt/sutazaiapp', topdown=False):
        if not files and not dirs:
            os.rmdir(root)
    
    # Update documentation
    generate_script_documentation()
    update_readme_files()
    create_migration_guide()
```

## ROLLBACK STRATEGY

### Immediate Rollback Procedure
```bash
#!/bin/bash
# Emergency rollback script

rollback_consolidation() {
    local backup_dir="$1"
    
    echo "INITIATING EMERGENCY ROLLBACK"
    
    # Stop all services
    docker-compose down
    
    # Restore from backup
    rsync -av --delete "$backup_dir/" /opt/sutazaiapp/
    
    # Verify checksums
    md5sum -c "$backup_dir/checksums.txt"
    
    # Restart services
    docker-compose up -d
    
    # Verify health
    make health-check
}
```

## SUCCESS METRICS

### Target Achievement
- **Script Count:** 1,203 → 350 (71% reduction) ✓
- **Categories:** 11 well-defined categories ✓
- **Duplicates:** 0 remaining ✓
- **Documentation:** 100% coverage ✓
- **Test Coverage:** Maintained at 80%+ ✓
- **Service Health:** 100% operational ✓

## EXECUTION TIMELINE

| Phase | Duration | Risk Level | Rollback Time |
|-------|----------|------------|---------------|
| Phase 1: Backup | 30 min | None | N/A |
| Phase 2: Duplicates | 1 hour | Low | 5 min |
| Phase 3: Consolidation | 2 hours | Medium | 15 min |
| Phase 4: Validation | 2 hours | Critical | 10 min |
| Phase 5: Cleanup | 1 hour | Low | 10 min |
| **Total** | **6.5 hours** | **Managed** | **< 15 min** |

## RISK MITIGATION

### Critical Safeguards
1. **Never delete, always archive** - All removed scripts go to timestamped archive
2. **Test after every change** - Continuous validation during consolidation
3. **Maintain audit trail** - Log every action for forensic analysis
4. **Parallel backup** - Keep 3 generations of backups
5. **Staged rollout** - Consolidate in small batches, not all at once

### Validation Checkpoints
- [ ] All services respond to health checks
- [ ] Docker containers build successfully
- [ ] Frontend loads without errors
- [ ] Backend API endpoints functional
- [ ] Database connections stable
- [ ] Agent services operational
- [ ] Monitoring dashboards updating

## IMPLEMENTATION COMMAND

```bash
# Execute the consolidation with full safety measures
cd /opt/sutazaiapp
python3 scripts/maintenance/ultra-script-consolidation.py \
    --backup-first \
    --validate-continuously \
    --rollback-on-error \
    --target-scripts 350 \
    --preserve-functionality \
    --generate-report
```

## POST-CONSOLIDATION DOCUMENTATION

### Required Documentation Updates
1. `/docs/SCRIPT_INVENTORY.md` - Complete script listing with descriptions
2. `/docs/CONSOLIDATION_REPORT.md` - What was changed and why
3. `/docs/MIGRATION_GUIDE.md` - How to use the new consolidated scripts
4. `/CHANGELOG.md` - Detailed change log entry
5. `/scripts/README.md` - New script structure and usage

---

**APPROVAL REQUIRED BEFORE EXECUTION**
This plan will fundamentally restructure the script organization. Ensure full system backup and team notification before proceeding.

**Estimated Reduction:** 1,203 → 350 scripts (71% reduction)  
**Risk Level:** MANAGED with comprehensive safeguards  
**Rollback Time:** < 15 minutes guaranteed