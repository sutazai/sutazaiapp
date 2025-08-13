#!/bin/bash
# Safe cleanup script for orphaned requirements
set -e

# Create backup directory

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

BACKUP_DIR='/opt/sutazaiapp/requirements_backup_$(date +%Y%m%d_%H%M%S)'
mkdir -p $BACKUP_DIR

echo 'Backing up all requirements files...'
find /opt/sutazaiapp -name 'requirements*.txt' -type f | while read f; do
    rel_path=${f#/opt/sutazaiapp/}
    mkdir -p "$BACKUP_DIR/$(dirname $rel_path)"
    cp "$f" "$BACKUP_DIR/$rel_path"
done

echo 'Removing safe orphaned files...'
rm -f '/opt/sutazaiapp/tests/requirements-test.txt'
rm -f '/opt/sutazaiapp/docs/requirements/deployments/requirements.txt'
rm -f '/opt/sutazaiapp/docs/requirements/system/requirements.txt'
rm -f '/opt/sutazaiapp/docs/requirements/agent-message-bus/requirements.txt'
rm -f '/opt/sutazaiapp/docs/requirements/requirements/requirements.txt'
rm -f '/opt/sutazaiapp/docs/requirements/agent-registry/requirements.txt'
rm -f '/opt/sutazaiapp/docs/requirements/infrastructure-devops/requirements.txt'
rm -f '/opt/sutazaiapp/docs/requirements/localagi/requirements.txt'
rm -f '/opt/sutazaiapp/docs/requirements/brain/requirements_ .txt'
rm -f '/opt/sutazaiapp/docs/requirements/web_learning/requirements.txt'
rm -f '/opt/sutazaiapp/docs/requirements/context-optimizer/requirements.txt'
rm -f '/opt/sutazaiapp/docs/requirements/tests/requirements-test.txt'
rm -f '/opt/sutazaiapp/docs/requirements/universal-agent/requirements.txt'
rm -f '/opt/sutazaiapp/docs/requirements/archive/requirements-test.txt'
rm -f '/opt/sutazaiapp/docs/requirements/archive/requirements-optimized.txt'
rm -f '/opt/sutazaiapp/docs/requirements/archive/requirements- .txt'
rm -f '/opt/sutazaiapp/docs/requirements/archive/requirements-agi.txt'
rm -f '/opt/sutazaiapp/docs/requirements/jarvis-agi/requirements_super.txt'
rm -f '/opt/sutazaiapp/docs/requirements/hardware-optimizer/requirements.txt'

echo 'Cleanup complete! Backup stored in: $BACKUP_DIR'