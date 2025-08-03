# SutazAI Codebase Cleanup Plan
## Date: 2025-08-03 18:27:29

### Archive Location
All removed files are backed up in: `/opt/sutazaiapp/archive/cleanup_20250803_182729/`

### Phase 1: Mystery Version Files (PRIORITY: HIGH)
**Files Identified:**
- =0.21.1 (pip install log for PyJWT-2.10.1, prometheus-client-0.22.1)
- =0.29.0
- =2.0.36 (pip install log for bleach-6.2.0, sqlalchemy-2.0.42, etc.)
- =2.10.1
- =44.0.0
- =6.1.0

**Analysis:** These are accidentally saved pip installation logs from shell redirect errors.

**References Found:** 46 files reference these version numbers, but they're referencing the actual package versions, NOT these files.

**Risk Assessment:** SAFE - These are junk files with no functional purpose.

**Action:** Archive and remove

### Phase 2: Requirements Files Consolidation (PRIORITY: HIGH)
**Current Count:** 134+ requirements files (needs audit)
**Target:** Maximum 4 files
- requirements.txt (main)
- requirements-dev.txt (development)
- requirements-test.txt (testing)
- requirements-prod.txt (production-specific)

### Phase 3: Backup Files Cleanup (PRIORITY: MEDIUM)
**Pattern:** *.backup files throughout codebase
**Action:** Archive and remove after verification

### Phase 4: Temporary Files and Directories (PRIORITY: MEDIUM)
**Patterns:**
- temp/ directories
- *.tmp files
- build artifacts in wrong locations

### Phase 5: Commented-Out Code (PRIORITY: LOW)
**Scope:** Systematic removal of commented code blocks

## Verification Steps Before Each Removal:
1. Search entire codebase for file references
2. Check if file is imported/included anywhere
3. Verify no functional dependency
4. Create archive copy with original path structure
5. Document removal reason
6. Test that removal doesn't break functionality

## Rollback Instructions:
1. All original files preserved in archive with full path structure
2. To restore: `cp -r /opt/sutazaiapp/archive/cleanup_20250803_182729/* /opt/sutazaiapp/`
3. Git stash/commit before cleanup for additional safety

## Testing Protocol:
- Run basic health checks after each phase
- Verify container builds still work
- Check that no imports are broken
- Validate deployment scripts still function