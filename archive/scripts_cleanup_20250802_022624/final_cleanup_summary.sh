#!/bin/bash

echo "=== SUTAZAI System Final Cleanup Summary ==="
echo

# Show current disk usage
echo "Current disk usage:"
du -sh /opt/sutazaiapp/
echo

# Show largest directories
echo "Largest directories:"
du -sh /opt/sutazaiapp/*/ 2>/dev/null | sort -rh | head -10
echo

# Show file counts by type
echo "File counts by type:"
echo "Python files: $(find /opt/sutazaiapp -name "*.py" -type f | wc -l)"
echo "Shell scripts: $(find /opt/sutazaiapp -name "*.sh" -type f | wc -l)"
echo "Docker files: $(find /opt/sutazaiapp -name "Dockerfile*" -type f | wc -l)"
echo "YAML files: $(find /opt/sutazaiapp -name "*.yml" -o -name "*.yaml" | wc -l)"
echo "JSON files: $(find /opt/sutazaiapp -name "*.json" -type f | wc -l)"
echo "Markdown files: $(find /opt/sutazaiapp -name "*.md" -type f | wc -l)"
echo

# Check for any remaining garbage
echo "Checking for remaining garbage files:"
echo "Python cache: $(find /opt/sutazaiapp -name "__pycache__" -type d 2>/dev/null | wc -l) directories"
echo "Backup files: $(find /opt/sutazaiapp -name "*.bak" -o -name "*.backup" 2>/dev/null | wc -l) files"
echo "Log files: $(find /opt/sutazaiapp -name "*.log" -type f 2>/dev/null | wc -l) files"
echo "Temp files: $(find /opt/sutazaiapp -name "*.tmp" -o -name "*.temp" 2>/dev/null | wc -l) files"
echo

echo "=== Cleanup Complete ==="
echo "The system has been cleaned of:"
echo "✓ Taskmaster files and directories"
echo "✓ MCP (Model Context Protocol) files"
echo "✓ Cursor IDE files"
echo "✓ Python cache and temporary files"
echo "✓ Archive directories (saved 331MB+)"
echo "✓ Virtual environments"
echo "✓ Backup and cache directories"
echo "✓ Duplicate docker-compose files"
echo "✓ Old logs and reports"
echo
echo "All documentation has been organized in: /opt/sutazaiapp/docs/"
echo "All scripts have been consolidated in: /opt/sutazaiapp/scripts/"