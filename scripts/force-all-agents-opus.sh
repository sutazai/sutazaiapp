#!/bin/bash

# Force ALL agents to use Opus model - comprehensive update
# This script ensures the model field is set correctly for Claude to recognize

AGENTS_DIR="/opt/sutazaiapp/.claude/agents"
UPDATED_COUNT=0
FAILED_COUNT=0
SONNET_COUNT=0

echo "========================================="
echo "FORCING ALL AGENTS TO USE OPUS MODEL"
echo "========================================="
echo ""

# First, let's find which agents still have sonnet
echo "Finding agents that still use sonnet..."
SONNET_AGENTS=$(find "$AGENTS_DIR" -name "*.md" -type f 2>/dev/null | while read -r file; do
    if grep -q "^model: sonnet" "$file" 2>/dev/null || ! grep -q "^model:" "$file" 2>/dev/null; then
        echo "$file"
    fi
done)

echo "Found $(echo "$SONNET_AGENTS" | wc -l) agents needing update"
echo ""

# Function to force update a single agent file
force_update_agent() {
    local file="$1"
    local relative_path="${file#$AGENTS_DIR/}"
    
    echo -n "Force updating $relative_path... "
    
    # Check if file exists and is readable
    if [ ! -r "$file" ]; then
        echo "❌ Cannot read file"
        ((FAILED_COUNT++))
        return 1
    fi
    
    # Use sed to update in place with different patterns
    # First, update any existing model field
    sed -i 's/^model: .*/model: opus/' "$file" 2>/dev/null
    
    # Also update modelId field
    sed -i 's/^modelId: .*/modelId: claude-opus-4-1-20250805/' "$file" 2>/dev/null
    
    # If model field doesn't exist, add it after the name field
    if ! grep -q "^model:" "$file" 2>/dev/null; then
        # Add model field after name field in frontmatter
        sed -i '/^name:/a\model: opus' "$file" 2>/dev/null
    fi
    
    # If modelId field doesn't exist, add it after model field
    if ! grep -q "^modelId:" "$file" 2>/dev/null; then
        sed -i '/^model:/a\modelId: claude-opus-4-1-20250805' "$file" 2>/dev/null
    fi
    
    # Verify the update worked
    if grep -q "^model: opus" "$file" 2>/dev/null; then
        echo "✅"
        ((UPDATED_COUNT++))
    else
        echo "❌ Update failed"
        ((FAILED_COUNT++))
    fi
}

# Process specifically the agents that still show sonnet
echo "Processing agents that need updating..."
echo ""

# Update all markdown files
find "$AGENTS_DIR" -name "*.md" -type f 2>/dev/null | while read -r file; do
    force_update_agent "$file"
done

echo ""
echo "========================================="
echo "Final Summary:"
echo "✅ Successfully updated: $UPDATED_COUNT agents"
if [ $FAILED_COUNT -gt 0 ]; then
    echo "❌ Failed to update: $FAILED_COUNT agents"
fi
echo "========================================="

# Final verification - count how many still have sonnet
echo ""
echo "Final verification check:"
echo "-----------------------------------------"

OPUS_COUNT=$(grep -r "^model: opus" "$AGENTS_DIR" 2>/dev/null | wc -l)
REMAINING_SONNET=$(grep -r "^model: sonnet" "$AGENTS_DIR" 2>/dev/null | wc -l)
NO_MODEL=$(find "$AGENTS_DIR" -name "*.md" -exec grep -L "^model:" {} \; 2>/dev/null | wc -l)

echo "✅ Agents with 'model: opus': $OPUS_COUNT"
echo "❌ Agents with 'model: sonnet': $REMAINING_SONNET"
echo "⚠️  Agents without model field: $NO_MODEL"

if [ $REMAINING_SONNET -gt 0 ]; then
    echo ""
    echo "Agents STILL showing sonnet:"
    grep -r "^model: sonnet" "$AGENTS_DIR" 2>/dev/null | cut -d: -f1 | while read file; do
        echo "  - ${file#$AGENTS_DIR/}"
    done
fi

echo ""
echo "✨ Script completed!"
echo ""
echo "IMPORTANT: Close and reopen the agents panel or restart Claude for changes to take effect."