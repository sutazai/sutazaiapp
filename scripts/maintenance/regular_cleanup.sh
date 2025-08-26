#!/bin/bash
# SutazAI Regular Cleanup Script
# Purpose: Maintain optimal disk usage by removing redundant files
# Schedule: Run monthly or as needed
# Author: Garbage Collector Agent
# Date: 2025-08-26

set -e

echo "========================================="
echo "SutazAI System Cleanup Starting..."
echo "========================================="
echo "Initial disk usage:"
du -sh /opt/sutazaiapp 2>/dev/null || du -sh .

# Create archive directory if needed
ARCHIVE_DIR="/tmp/sutazai_cleanup_archive_$(date +%Y%m%d)"
mkdir -p "$ARCHIVE_DIR"
echo "Archive directory: $ARCHIVE_DIR"

# Python cleanup
echo "Cleaning Python cache files..."
find /opt/sutazaiapp -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find /opt/sutazaiapp -name "*.pyc" -delete 2>/dev/null
find /opt/sutazaiapp -name "*.pyo" -delete 2>/dev/null

# Node modules cleanup (keep only those with adjacent package.json)
echo "Cleaning orphaned node_modules..."
for dir in $(find /opt/sutazaiapp -type d -name "node_modules" 2>/dev/null); do
    parent_dir=$(dirname "$dir")
    if [ ! -f "$parent_dir/package.json" ]; then
        echo "  Archiving orphaned: $dir"
        tar -czf "$ARCHIVE_DIR/$(echo $dir | tr '/' '_').tar.gz" "$dir" 2>/dev/null
        rm -rf "$dir"
    fi
done

# Clean nested node_modules
find /opt/sutazaiapp/node_modules -mindepth 2 -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null

# Temporary files
echo "Cleaning temporary files..."
find /opt/sutazaiapp -type f \( -name "*.tmp" -o -name "*.bak" -o -name "*.old" -o -name "*.swp" -o -name "*~" -o -name ".DS_Store" \) -delete 2>/dev/null

# Old log files
echo "Cleaning old log files..."
find /opt/sutazaiapp/logs -type f -name "*.log" -mtime +7 -delete 2>/dev/null
find /opt/sutazaiapp/logs -type f -empty -delete 2>/dev/null

# Jupyter checkpoints
find /opt/sutazaiapp -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null

# Empty directories
echo "Removing empty directories..."
find /opt/sutazaiapp -type d -empty -delete 2>/dev/null

# Build artifacts (archive first)
echo "Archiving build artifacts..."
for dir in $(find /opt/sutazaiapp -type d \( -name "build" -o -name "dist" \) -not -path "*/node_modules/*" 2>/dev/null); do
    if [ -d "$dir" ]; then
        tar -czf "$ARCHIVE_DIR/$(echo $dir | tr '/' '_').tar.gz" "$dir" 2>/dev/null
        rm -rf "$dir"
    fi
done

echo "========================================="
echo "Cleanup Complete!"
echo "========================================="
echo "Final disk usage:"
du -sh /opt/sutazaiapp 2>/dev/null || du -sh .
echo ""
echo "Archives saved to: $ARCHIVE_DIR"
du -sh "$ARCHIVE_DIR" 2>/dev/null || echo "No archives created"
echo ""
echo "To recover archived files:"
echo "  tar -xzf $ARCHIVE_DIR/[archive_name].tar.gz -C /"
echo "========================================="