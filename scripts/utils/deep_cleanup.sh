#!/bin/bash

# Deep cleanup script to remove garbage files from the system

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

echo "Starting deep cleanup of SUTAZAI system..."

# Remove Python cache files
echo "Removing Python cache files..."
find /opt/sutazaiapp -type f -name "*.pyc" -delete 2>/dev/null || true
find /opt/sutazaiapp -type f -name "*.pyo" -delete 2>/dev/null || true
find /opt/sutazaiapp -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Remove editor temporary files
echo "Removing editor temporary files..."
find /opt/sutazaiapp -type f -name "*.swp" -delete 2>/dev/null || true
find /opt/sutazaiapp -type f -name "*.swo" -delete 2>/dev/null || true
find /opt/sutazaiapp -type f -name "*~" -delete 2>/dev/null || true
find /opt/sutazaiapp -type f -name ".*.swp" -delete 2>/dev/null || true

# Remove OS-specific files
echo "Removing OS-specific garbage files..."
find /opt/sutazaiapp -type f -name ".DS_Store" -delete 2>/dev/null || true
find /opt/sutazaiapp -type f -name "Thumbs.db" -delete 2>/dev/null || true
find /opt/sutazaiapp -type f -name "desktop.ini" -delete 2>/dev/null || true

# Remove backup files
echo "Removing backup files..."
find /opt/sutazaiapp -type f -name "*.backup" -delete 2>/dev/null || true
find /opt/sutazaiapp -type f -name "*.bak" -delete 2>/dev/null || true
find /opt/sutazaiapp -type f -name "*.old" -delete 2>/dev/null || true
find /opt/sutazaiapp -type f -name "*.orig" -delete 2>/dev/null || true

# Remove log files (except in logs directory)
echo "Removing stray log files..."
find /opt/sutazaiapp -type f -name "*.log" ! -path "*/logs/*" -delete 2>/dev/null || true

# Remove PID files
echo "Removing PID files..."
find /opt/sutazaiapp -type f -name "*.pid" -delete 2>/dev/null || true

# Remove temporary files
echo "Removing temporary files..."
find /opt/sutazaiapp -type f -name "*.tmp" -delete 2>/dev/null || true
find /opt/sutazaiapp -type f -name "*.temp" -delete 2>/dev/null || true

# Remove pytest cache
echo "Removing pytest cache..."
find /opt/sutazaiapp -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

# Remove coverage files
echo "Removing coverage files..."
find /opt/sutazaiapp -type f -name ".coverage" -delete 2>/dev/null || true
find /opt/sutazaiapp -type f -name "coverage.xml" -delete 2>/dev/null || true
find /opt/sutazaiapp -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true

# Remove mypy cache
echo "Removing mypy cache..."
find /opt/sutazaiapp -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true

# Remove ruff cache
echo "Removing ruff cache..."
find /opt/sutazaiapp -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

# Remove egg-info directories
echo "Removing egg-info directories..."
find /opt/sutazaiapp -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Remove dist and build directories
echo "Removing build artifacts..."
find /opt/sutazaiapp -type d -name "dist" ! -path "*/node_modules/*" -exec rm -rf {} + 2>/dev/null || true
find /opt/sutazaiapp -type d -name "build" ! -path "*/node_modules/*" -exec rm -rf {} + 2>/dev/null || true

# Remove .idea directories (IDE files)
echo "Removing IDE directories..."
find /opt/sutazaiapp -type d -name ".idea" -exec rm -rf {} + 2>/dev/null || true
find /opt/sutazaiapp -type d -name ".vscode" ! -path "*/.claude/*" -exec rm -rf {} + 2>/dev/null || true

# Remove archive directories (optional - uncomment if you want to remove archives)
# echo "Removing archive directories..."
# rm -rf /opt/sutazaiapp/archive
# rm -rf /opt/sutazaiapp/backend_archive

# Remove empty directories
echo "Removing empty directories..."
find /opt/sutazaiapp -type d -empty ! -path "*/.git/*" ! -path "*/.claude/*" -delete 2>/dev/null || true

# Clean up logs directory (keep only recent logs)
echo "Cleaning up logs directory..."
if [ -d "/opt/sutazaiapp/logs" ]; then
    find /opt/sutazaiapp/logs -type f -name "*.log" -mtime +7 -delete 2>/dev/null || true
fi

# Remove duplicate requirements files
echo "Cleaning up duplicate requirements files..."
find /opt/sutazaiapp -name "requirements.txt.backup" -delete 2>/dev/null || true
find /opt/sutazaiapp -name "requirements.secure.txt" -delete 2>/dev/null || true

# Summary
echo -e "\n=== Cleanup Summary ==="
echo "Python cache files removed"
echo "Editor temporary files removed"
echo "OS-specific files removed"
echo "Backup files removed"
echo "Build artifacts removed"
echo "Empty directories removed"

# Show disk usage after cleanup
echo -e "\n=== Disk Usage After Cleanup ==="
du -sh /opt/sutazaiapp/

echo -e "\nDeep cleanup complete!"