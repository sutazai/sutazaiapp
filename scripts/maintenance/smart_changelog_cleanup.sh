#!/bin/bash
# Smart CHANGELOG cleanup - preserves key locations, removes auto-generated templates
# Respects Rule 6/15 while maintaining useful change tracking

set -euo pipefail

REPO_ROOT="/opt/sutazaiapp"
REMOVED_COUNT=0
PRESERVED_COUNT=0

echo "🔍 Smart CHANGELOG Cleanup - Preserving key locations only"
echo "================================================="

# Define key CHANGELOGs to preserve
declare -a PRESERVE_PATHS=(
    "/opt/sutazaiapp/docs/CHANGELOG.md"           # Main canonical
    "/opt/sutazaiapp/backend/CHANGELOG.md"        # Backend changes
    "/opt/sutazaiapp/frontend/CHANGELOG.md"       # Frontend changes
    "/opt/sutazaiapp/agents/CHANGELOG.md"         # Agent changes
    "/opt/sutazaiapp/IMPORTANT/CHANGELOG.md"      # Critical docs
    "/opt/sutazaiapp/scripts/CHANGELOG.md"        # Scripts changes
    "/opt/sutazaiapp/configs/CHANGELOG.md"        # Config changes
)

echo "Preserving these key CHANGELOGs:"
for path in "${PRESERVE_PATHS[@]}"; do
    if [[ -f "$path" ]]; then
        echo "  ✅ $path"
        ((PRESERVED_COUNT++))
    else
        echo "  ⚠️  $path (doesn't exist, will be created if needed)"
    fi
done

echo ""
echo "Analyzing and removing auto-generated template CHANGELOGs..."

# Find all CHANGELOG.md files
while IFS= read -r file; do
    # Check if this file should be preserved
    should_preserve=false
    for preserve_path in "${PRESERVE_PATHS[@]}"; do
        if [[ "$file" == "$preserve_path" ]]; then
            should_preserve=true
            break
        fi
    done
    
    if $should_preserve; then
        continue
    fi
    
    # Check if it's an auto-generated template (has the template marker)
    if grep -q "generated_by: scripts/utils/ensure_changelogs.py" "$file" 2>/dev/null; then
        # Check if it has any real content beyond the template
        # Count non-template lines
        content_lines=$(grep -v "^#\|^>\|^-\|^$\|title:\|generated_by:\|purpose:\|Conventions:\|Template entry:\|Follow Conventional\|Keep entries\|Include date\|This folder maintains\|authoritative\|quick, path-scoped" "$file" 2>/dev/null | wc -l)
        
        if [[ $content_lines -le 2 ]]; then
            # It's just a template with no real content - remove it
            echo "  ❌ Removing template: $file"
            rm -f "$file"
            ((REMOVED_COUNT++))
        else
            # Has actual content - preserve it
            echo "  ⚠️  Keeping (has content): $file"
            ((PRESERVED_COUNT++))
        fi
    else
        # Not auto-generated, might be legitimate
        # Check if it's in node_modules (shouldn't be tracked anyway)
        if [[ "$file" == *"/node_modules/"* ]]; then
            continue  # Skip node_modules
        fi
        
        # Check file size - if very small, likely empty
        if [[ $(wc -l < "$file") -le 5 ]]; then
            echo "  ❌ Removing empty: $file"
            rm -f "$file"
            ((REMOVED_COUNT++))
        else
            echo "  ✅ Keeping (manual): $file"
            ((PRESERVED_COUNT++))
        fi
    fi
done < <(find "$REPO_ROOT" -name "CHANGELOG.md" -type f -not -path "*/node_modules/*" -not -path "*/.git/*")

echo ""
echo "================================================="
echo "✅ CLEANUP COMPLETE"
echo "  Removed: $REMOVED_COUNT auto-generated/empty CHANGELOGs"
echo "  Preserved: $PRESERVED_COUNT CHANGELOGs with actual content"
echo ""
echo "📝 Next steps:"
echo "  1. Update docs/CHANGELOG.md with this cleanup"
echo "  2. Modify ensure_changelogs.py to only maintain key locations"
echo "  3. Establish clear guidelines for when to use folder-specific CHANGELOGs"