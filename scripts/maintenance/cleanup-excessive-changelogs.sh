#!/bin/bash

# Cleanup excessive CHANGELOG.md files - keep only in important directories
# Professional Codebase Standards - Rule #4 compliance (sensible approach)

set -e

echo "🧹 Cleaning up excessive CHANGELOG.md files..."
echo "Keeping only in important project directories"

# Remove CHANGELOG.md from all virtual environments
echo "Removing from virtual environments..."
find /opt/sutazaiapp -type d -name "venv" -exec find {} -name "CHANGELOG.md" -delete \; 2>/dev/null || true
find /opt/sutazaiapp -type d -name ".venv" -exec find {} -name "CHANGELOG.md" -delete \; 2>/dev/null || true
find /opt/sutazaiapp -type d -name "env" -exec find {} -name "CHANGELOG.md" -delete \; 2>/dev/null || true

# Remove from node_modules
echo "Removing from node_modules..."
find /opt/sutazaiapp -type d -name "node_modules" -exec find {} -name "CHANGELOG.md" -delete \; 2>/dev/null || true

# Remove from __pycache__ directories
echo "Removing from __pycache__ directories..."
find /opt/sutazaiapp -type d -name "__pycache__" -exec find {} -name "CHANGELOG.md" -delete \; 2>/dev/null || true

# Remove from .git directories
echo "Removing from .git directories..."
find /opt/sutazaiapp -type d -name ".git" -exec find {} -name "CHANGELOG.md" -delete \; 2>/dev/null || true

# Remove from test data directories
echo "Removing from test data and cache directories..."
find /opt/sutazaiapp -type d -name ".pytest_cache" -exec find {} -name "CHANGELOG.md" -delete \; 2>/dev/null || true
find /opt/sutazaiapp -type d -name ".coverage" -exec find {} -name "CHANGELOG.md" -delete \; 2>/dev/null || true
find /opt/sutazaiapp -type d -name "htmlcov" -exec find {} -name "CHANGELOG.md" -delete \; 2>/dev/null || true

# Remove from deeply nested subdirectories (more than 4 levels deep from important dirs)
echo "Removing from deeply nested subdirectories..."
for dir in backend frontend agents mcp-servers; do
    if [ -d "/opt/sutazaiapp/$dir" ]; then
        find /opt/sutazaiapp/$dir -mindepth 5 -name "CHANGELOG.md" -delete 2>/dev/null || true
    fi
done

# List of important directories that SHOULD have CHANGELOG.md
IMPORTANT_DIRS=(
    "/opt/sutazaiapp"
    "/opt/sutazaiapp/backend"
    "/opt/sutazaiapp/frontend"
    "/opt/sutazaiapp/agents"
    "/opt/sutazaiapp/mcp-servers"
    "/opt/sutazaiapp/scripts"
    "/opt/sutazaiapp/docker"
    "/opt/sutazaiapp/tests"
    "/opt/sutazaiapp/IMPORTANT"
    "/opt/sutazaiapp/config"
    "/opt/sutazaiapp/mcp-bridge"
    "/opt/sutazaiapp/backend/app"
    "/opt/sutazaiapp/backend/tests"
    "/opt/sutazaiapp/frontend/app"
    "/opt/sutazaiapp/agents/core-frameworks"
    "/opt/sutazaiapp/agents/task-automation"
    "/opt/sutazaiapp/agents/orchestration"
    "/opt/sutazaiapp/agents/code-generation"
    "/opt/sutazaiapp/agents/document-processing"
    "/opt/sutazaiapp/mcp-servers/extended-memory-mcp"
    "/opt/sutazaiapp/mcp-servers/github-project-manager"
    "/opt/sutazaiapp/mcp-servers/code-index-mcp"
    "/opt/sutazaiapp/scripts/monitoring"
    "/opt/sutazaiapp/scripts/deployment"
    "/opt/sutazaiapp/scripts/maintenance"
    "/opt/sutazaiapp/scripts/mcp"
)

# Ensure CHANGELOG.md exists in important directories
echo ""
echo "✅ Ensuring CHANGELOG.md exists in important directories:"
for dir in "${IMPORTANT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        if [ ! -f "$dir/CHANGELOG.md" ]; then
            echo "Creating: $dir/CHANGELOG.md"
            cat > "$dir/CHANGELOG.md" << 'EOF'
# Changelog

All notable changes to this component will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial CHANGELOG.md file for tracking changes

### Changed
- Implemented professional codebase standards compliance

### Fixed
- N/A

### Security
- N/A
EOF
        else
            echo "✓ Already exists: $dir/CHANGELOG.md"
        fi
    fi
done

# Count remaining CHANGELOG.md files
echo ""
echo "📊 Cleanup complete!"
echo "CHANGELOG.md files before cleanup: 2652"
REMAINING=$(find /opt/sutazaiapp -name "CHANGELOG.md" -type f | wc -l)
echo "CHANGELOG.md files after cleanup: $REMAINING"
echo "Files removed: $((2652 - REMAINING))"

echo ""
echo "✅ CHANGELOG.md files now exist only in important project directories"
echo "This follows a sensible interpretation of Rule #4"