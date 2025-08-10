#!/bin/bash

# Script to remove large garbage directories and files

set -e


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

echo "Starting removal of large garbage files and directories..."

# Remove archive directories
echo "Removing archive directories (331MB+)..."
if [ -d "/opt/sutazaiapp/archive" ]; then
    echo "Removing /opt/sutazaiapp/archive..."
    rm -rf /opt/sutazaiapp/archive
fi

if [ -d "/opt/sutazaiapp/backend_archive" ]; then
    echo "Removing /opt/sutazaiapp/backend_archive..."
    rm -rf /opt/sutazaiapp/backend_archive
fi

# Remove virtual environments
echo "Removing virtual environments..."
find /opt/sutazaiapp -type d \( -name "venv" -o -name ".venv" -o -name "virtualenv" -o -name "*_env" \) -exec rm -rf {} + 2>/dev/null || true

# Remove security_audit_env
if [ -d "/opt/sutazaiapp/security_audit_env" ]; then
    echo "Removing security_audit_env..."
    rm -rf /opt/sutazaiapp/security_audit_env
fi

# Remove node_modules directories if any
echo "Removing node_modules directories..."
find /opt/sutazaiapp -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true

# Remove .git directories from subdirectories (keep main .git)
echo "Removing nested .git directories..."
find /opt/sutazaiapp -mindepth 2 -type d -name ".git" -exec rm -rf {} + 2>/dev/null || true

# Remove backup directories
echo "Removing backup directories..."
find /opt/sutazaiapp -type d -name "*backup*" -o -name "*bak*" | grep -v ".claude" | xargs rm -rf 2>/dev/null || true

# Remove cache directories
echo "Removing cache directories..."
find /opt/sutazaiapp -type d -name "cache" -o -name ".cache" | grep -v ".claude" | xargs rm -rf 2>/dev/null || true

# Remove tmp directories
echo "Removing tmp directories..."
if [ -d "/opt/sutazaiapp/tmp" ]; then
    rm -rf /opt/sutazaiapp/tmp
fi

# Remove workspace directory if empty or contains garbage
if [ -d "/opt/sutazaiapp/workspace" ]; then
    echo "Removing workspace directory..."
    rm -rf /opt/sutazaiapp/workspace
fi

# Remove the strange ~/ directory
if [ -d "/opt/sutazaiapp/~" ]; then
    echo "Removing ~/ directory..."
    rm -rf "/opt/sutazaiapp/~"
fi

# Remove duplicate docker compose files
echo "Removing duplicate docker-compose files..."
ls -1 /opt/sutazaiapp/docker-compose*.yml | grep -v "docker-compose.yml$" | xargs rm -f 2>/dev/null || true

# Remove old log files
echo "Removing old log files..."
find /opt/sutazaiapp/logs -type f -name "*.log" -mtime +3 -delete 2>/dev/null || true

# Remove report files
echo "Removing old report files..."
find /opt/sutazaiapp/reports -type f -mtime +7 -delete 2>/dev/null || true

# Summary
echo -e "\n=== Cleanup Summary ==="
echo "Archive directories removed"
echo "Virtual environments removed"
echo "Backup directories removed"
echo "Cache directories removed"
echo "Duplicate docker-compose files removed"
echo "Old logs and reports cleaned"

# Show disk usage after cleanup
echo -e "\n=== Disk Usage After Cleanup ==="
du -sh /opt/sutazaiapp/

echo -e "\nLarge garbage removal complete!"