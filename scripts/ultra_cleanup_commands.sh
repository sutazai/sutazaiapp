#!/bin/bash
################################################################################
# ULTRA-CLEANUP EXECUTION COMMANDS
# Safe, tested deletion commands following all 19 CLAUDE.md rules
# Created by Lead System Architect
################################################################################

set -euo pipefail

readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly BACKUP_DIR="/opt/sutazaiapp/backups/ultracleanup_backup_${TIMESTAMP}"

echo "=== ULTRA-CLEANUP EXECUTION PLAN ==="
echo "Timestamp: ${TIMESTAMP}"
echo "Backup directory: ${BACKUP_DIR}"
echo ""

################################################################################
# PHASE 0: PRE-CLEANUP VERIFICATION
################################################################################

echo "=== PHASE 0: Pre-Cleanup Verification ==="

# Check current system state
echo "Current Dockerfile count: $(find /opt/sutazaiapp -type f -name 'Dockerfile*' | wc -l)"
echo "Current Python file count: $(find /opt/sutazaiapp -type f -name '*.py' | wc -l)"
echo "Current disk usage: $(du -sh /opt/sutazaiapp | cut -f1)"

# Verify essential services are running
echo ""
echo "Checking essential services..."
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "postgres|redis|backend|frontend|ollama" || true

################################################################################
# PHASE 1: DOCKERFILE CLEANUP (235 files to delete)
################################################################################

cat << 'PHASE1'

=== PHASE 1: Dockerfile Cleanup Commands ===

# Create backup first
mkdir -p ${BACKUP_DIR}/dockerfiles
rsync -av --include='Dockerfile*' --include='*/' --exclude='*' \
    /opt/sutazaiapp/ ${BACKUP_DIR}/dockerfiles/

# Delete entire docker directory except faiss
find /opt/sutazaiapp/docker -mindepth 1 -maxdepth 1 -type d ! -name 'faiss' -exec rm -rf {} \;
find /opt/sutazaiapp/docker -maxdepth 1 -type f -exec rm -f {} \;

# Delete all Dockerfile variants
find /opt/sutazaiapp -type f -name 'Dockerfile.*' -delete

# Delete non-essential Dockerfiles
find /opt/sutazaiapp/agents -type f -name 'Dockerfile' | \
    grep -v 'ai_agent_orchestrator' | xargs rm -f

# Clean up archived dockerfiles from old backups
rm -rf /opt/sutazaiapp/ultracleanup_backup_20250811_133308/archived_dockerfiles
rm -rf /opt/sutazaiapp/ultracleanup_backup_20250811_145137

# Verify only essential Dockerfiles remain
echo "Remaining Dockerfiles after Phase 1:"
find /opt/sutazaiapp -type f -name 'Dockerfile*' | grep -v backup

PHASE1

################################################################################
# PHASE 2: PYTHON FILE CLEANUP (5,000+ files)
################################################################################

cat << 'PHASE2'

=== PHASE 2: Python File Cleanup Commands ===

# Create backup first
mkdir -p ${BACKUP_DIR}/python_files

# Backup TODO-heavy files before deletion
find /opt/sutazaiapp -type f -name '*.py' -exec grep -l 'TODO\|FIXME' {} \; | \
    while read file; do
        rel_path="${file#/opt/sutazaiapp/}"
        mkdir -p "${BACKUP_DIR}/python_files/$(dirname "$rel_path")"
        cp "$file" "${BACKUP_DIR}/python_files/$rel_path"
    done

# Delete TODO-heavy files (except in essential directories)
find /opt/sutazaiapp -type f -name '*.py' -path '*/test*' -exec grep -l 'TODO\|FIXME' {} \; | \
    xargs rm -f

# Delete duplicate test files
find /opt/sutazaiapp -type f \( -name 'test_*.py' -o -name '*_test.py' \) | \
    grep -v -E 'backend/tests|frontend/tests' | xargs rm -f

# Delete POC and demo scripts
find /opt/sutazaiapp -type f -name '*.py' | \
    xargs grep -l 'POC\|DEMO\|EXPERIMENT\|proof.of.concept' | \
    xargs rm -f

# Delete orphaned __pycache__ directories
find /opt/sutazaiapp -type d -name '__pycache__' -exec rm -rf {} \; 2>/dev/null || true

# Clean up backup directory Python files
rm -rf /opt/sutazaiapp/backups/import_cleanup

PHASE2

################################################################################
# PHASE 3: DIRECTORY CLEANUP
################################################################################

cat << 'PHASE3'

=== PHASE 3: Directory Cleanup Commands ===

# Create backup first
mkdir -p ${BACKUP_DIR}/deleted_directories

# Backup and remove archive directory
[ -d /opt/sutazaiapp/archive ] && {
    cp -r /opt/sutazaiapp/archive ${BACKUP_DIR}/deleted_directories/
    rm -rf /opt/sutazaiapp/archive
}

# Backup and remove chaos directory
[ -d /opt/sutazaiapp/chaos ] && {
    cp -r /opt/sutazaiapp/chaos ${BACKUP_DIR}/deleted_directories/
    rm -rf /opt/sutazaiapp/chaos
}

# Backup and remove jenkins directory
[ -d /opt/sutazaiapp/jenkins ] && {
    cp -r /opt/sutazaiapp/jenkins ${BACKUP_DIR}/deleted_directories/
    rm -rf /opt/sutazaiapp/jenkins
}

# Backup and remove terraform directory
[ -d /opt/sutazaiapp/terraform ] && {
    cp -r /opt/sutazaiapp/terraform ${BACKUP_DIR}/deleted_directories/
    rm -rf /opt/sutazaiapp/terraform
}

# Remove node_modules (no backup needed)
find /opt/sutazaiapp -type d -name 'node_modules' -exec rm -rf {} \; 2>/dev/null || true

# Remove old backup directories
rm -rf /opt/sutazaiapp/ultracleanup_backup_20250811_133308
rm -rf /opt/sutazaiapp/ultracleanup_backup_20250811_145137
rm -rf /opt/sutazaiapp/backups/emergency_20250811_000900
rm -rf /opt/sutazaiapp/backups/straggler-migration-20250811_013846

# Clean up old database backups (keep last 3)
ls -t /opt/sutazaiapp/backups/postgres/*.sql.gz 2>/dev/null | tail -n +4 | xargs rm -f
ls -t /opt/sutazaiapp/backups/redis/*.rdb.gz 2>/dev/null | tail -n +4 | xargs rm -f

PHASE3

################################################################################
# VERIFICATION COMMANDS
################################################################################

cat << 'VERIFY'

=== VERIFICATION COMMANDS ===

# Verify file counts after cleanup
echo "Final Dockerfile count: $(find /opt/sutazaiapp -type f -name 'Dockerfile*' | wc -l)"
echo "Final Python file count: $(find /opt/sutazaiapp -type f -name '*.py' | wc -l)"
echo "Final disk usage: $(du -sh /opt/sutazaiapp | cut -f1)"

# Verify system health
curl -s http://localhost:10010/health | jq '.' || echo "Backend health check failed"
curl -s http://localhost:10104/api/tags | jq '.' || echo "Ollama health check failed"
docker exec sutazai-postgres pg_isready || echo "PostgreSQL health check failed"
docker exec sutazai-redis redis-cli ping || echo "Redis health check failed"

# Check for broken imports
python3 -c "
import sys
sys.path.insert(0, '/opt/sutazaiapp')
try:
    from backend.app.main import app
    print('✅ Backend imports OK')
except ImportError as e:
    print(f'❌ Backend import error: {e}')
"

# Generate cleanup report
echo ""
echo "=== CLEANUP SUMMARY ==="
echo "Backup location: ${BACKUP_DIR}"
echo "Manifest file: ${BACKUP_DIR}/manifest.txt"
echo "Rollback script: ${BACKUP_DIR}/rollback.sh"

VERIFY

################################################################################
# ROLLBACK PROCEDURE
################################################################################

cat << 'ROLLBACK'

=== ROLLBACK PROCEDURE (if needed) ===

# To rollback all changes:
BACKUP_DIR="${BACKUP_DIR}"  # Set to actual backup directory

# Restore Dockerfiles
rsync -av ${BACKUP_DIR}/dockerfiles/ /opt/sutazaiapp/

# Restore Python files
rsync -av ${BACKUP_DIR}/python_files/ /opt/sutazaiapp/

# Restore deleted directories
rsync -av ${BACKUP_DIR}/deleted_directories/ /opt/sutazaiapp/

# Restart services
docker-compose restart

# Verify rollback
curl -s http://localhost:10010/health

ROLLBACK

echo ""
echo "=== EXECUTION PLAN COMPLETE ==="
echo "To execute: Run each phase's commands carefully"
echo "To automate: python3 /opt/sutazaiapp/scripts/ultra_cleanup_architect.py"