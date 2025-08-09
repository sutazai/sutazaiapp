#!/bin/bash
# Script to remove duplicate CHANGELOG.md files (Rule 6 & 15 enforcement)
# Preserves only /opt/sutazaiapp/docs/CHANGELOG.md

set -euo pipefail

MAIN_CHANGELOG="/opt/sutazaiapp/docs/CHANGELOG.md"
REPO_ROOT="/opt/sutazaiapp"
COUNT=0

echo "🚨 ENFORCING RULE 6 & 15: Removing duplicate CHANGELOG.md files..."
echo "Preserving only: $MAIN_CHANGELOG"

# Find and remove all CHANGELOG.md files except the main one
while IFS= read -r file; do
    if [[ "$file" != "$MAIN_CHANGELOG" ]]; then
        echo "  ❌ Removing: $file"
        rm -f "$file"
        ((COUNT++))
    fi
done < <(find "$REPO_ROOT" -name "CHANGELOG.md" -type f)

echo ""
echo "✅ CLEANUP COMPLETE: Removed $COUNT duplicate CHANGELOG.md files"
echo "📝 Single source of truth preserved at: $MAIN_CHANGELOG"