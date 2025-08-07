#!/bin/bash
# Safe cleanup script for orphaned requirements
set -e

# Create backup directory
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
rm -f '/opt/sutazaiapp/docs/requirements/brain/requirements_minimal.txt'
rm -f '/opt/sutazaiapp/docs/requirements/web_learning/requirements.txt'
rm -f '/opt/sutazaiapp/docs/requirements/context-optimizer/requirements.txt'
rm -f '/opt/sutazaiapp/docs/requirements/tests/requirements-test.txt'
rm -f '/opt/sutazaiapp/docs/requirements/universal-agent/requirements.txt'
rm -f '/opt/sutazaiapp/docs/requirements/archive/requirements-test.txt'
rm -f '/opt/sutazaiapp/docs/requirements/archive/requirements-optimized.txt'
rm -f '/opt/sutazaiapp/docs/requirements/archive/requirements-minimal.txt'
rm -f '/opt/sutazaiapp/docs/requirements/archive/requirements-agi.txt'
rm -f '/opt/sutazaiapp/docs/requirements/jarvis-agi/requirements_super.txt'
rm -f '/opt/sutazaiapp/docs/requirements/hardware-optimizer/requirements.txt'

echo 'Cleanup complete! Backup stored in: $BACKUP_DIR'