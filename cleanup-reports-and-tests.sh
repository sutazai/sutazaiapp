#!/bin/bash
# Cleanup old reports and test files

echo "🧹 Starting cleanup of reports and test files..."

# Create archive directory with timestamp
ARCHIVE_DIR="/opt/sutazaiapp/archive/cleanup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ARCHIVE_DIR"

# Move old report files (older than 1 day) to archive
echo "📦 Archiving old report files..."
find /opt/sutazaiapp -maxdepth 1 -name "*report*.json" -o -name "*report*.md" -mtime +1 | while read file; do
    mv "$file" "$ARCHIVE_DIR/" 2>/dev/null && echo "  Archived: $(basename "$file")"
done

# Move test output files to archive
echo "📦 Archiving test output files..."
find /opt/sutazaiapp -maxdepth 1 -name "test-*.html" -o -name "test-*.log" | while read file; do
    mv "$file" "$ARCHIVE_DIR/" 2>/dev/null && echo "  Archived: $(basename "$file")"
done

# Clean up old garbage collection reports
echo "🗑️ Cleaning up garbage collection reports..."
find /opt/sutazaiapp -name "garbage_cleanup_report_*.json" -mtime +1 -delete

# Clean up temporary Python cache
echo "🐍 Cleaning Python cache..."
find /opt/sutazaiapp -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Clean up npm logs
echo "📋 Cleaning npm logs..."
rm -f /opt/sutazaiapp/npm-debug.log* 2>/dev/null

# Clean up temporary files in /tmp (older than 2 days)
echo "🗑️ Cleaning old temporary files..."
find /tmp -type f -mtime +2 -delete 2>/dev/null

# Show disk usage after cleanup
echo ""
echo "💾 Disk usage after cleanup:"
df -h /opt/sutazaiapp | tail -1

# Count remaining files
echo ""
echo "📊 Summary:"
echo "  Reports directory: $(find /opt/sutazaiapp/reports -type f 2>/dev/null | wc -l) files"
echo "  Archive directory: $(find /opt/sutazaiapp/archive -type f 2>/dev/null | wc -l) files"
echo "  Test files: $(find /opt/sutazaiapp -maxdepth 1 -name "test-*" 2>/dev/null | wc -l) files"

echo ""
echo "✅ Cleanup completed!"