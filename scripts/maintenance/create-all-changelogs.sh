#!/bin/bash
# Create CHANGELOG.md files in all directories

set -e

BASE_DIR="/opt/sutazaiapp"
CREATED=0
SKIPPED=0

echo "📝 Creating CHANGELOG.md files in all directories..."

# Template for CHANGELOG.md
create_changelog() {
    local dir="$1"
    local dirname=$(basename "$dir")
    
    cat > "$dir/CHANGELOG.md" << EOF
# Changelog - $dirname

All notable changes to this directory will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - $(date +%Y-%m-%d)

### Added
- Initial directory structure
- Basic configuration files

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A
EOF
}

# Find all directories and create CHANGELOG.md if missing
find "$BASE_DIR" -type d -not -path "*/.*" -not -path "*/node_modules/*" -not -path "*/venv/*" -not -path "*/__pycache__/*" -not -path "*/dist/*" -not -path "*/build/*" | while read -r dir; do
    if [ ! -f "$dir/CHANGELOG.md" ]; then
        create_changelog "$dir"
        ((CREATED++))
        echo "✅ Created: $dir/CHANGELOG.md"
    else
        ((SKIPPED++))
    fi
done

echo ""
echo "📊 Summary:"
echo "  ✅ Created: $CREATED CHANGELOG.md files"
echo "  ⏭️  Skipped: $SKIPPED (already exist)"
echo ""
echo "✅ All directories now have CHANGELOG.md files!"