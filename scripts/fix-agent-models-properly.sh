#!/bin/bash

# Script to properly update all Claude agents to use Opus model
# Adds both 'model' and 'modelId' fields for compatibility

AGENTS_DIR="/opt/sutazaiapp/.claude/agents"
UPDATED_COUNT=0
FAILED_COUNT=0

echo "========================================="
echo "Fixing all agents to use Opus model"
echo "========================================="
echo ""

# Function to update a single agent file
update_agent() {
    local file="$1"
    local relative_path="${file#$AGENTS_DIR/}"
    
    echo -n "Fixing $relative_path... "
    
    # Check if file exists and is readable
    if [ ! -r "$file" ]; then
        echo "❌ Cannot read file"
        ((FAILED_COUNT++))
        return 1
    fi
    
    # Create a temporary file for the updated content
    local temp_file="${file}.tmp"
    
    # Process the file line by line
    local in_frontmatter=false
    local model_added=false
    local modelid_found=false
    
    while IFS= read -r line; do
        if [[ "$line" == "---" ]]; then
            echo "$line" >> "$temp_file"
            if [ "$in_frontmatter" = false ]; then
                in_frontmatter=true
            else
                # End of frontmatter - add model field if not already added
                if [ "$in_frontmatter" = true ] && [ "$model_added" = false ]; then
                    # Insert model field before closing ---
                    sed -i '$ d' "$temp_file"  # Remove the last ---
                    echo "model: opus" >> "$temp_file"
                    echo "---" >> "$temp_file"
                    model_added=true
                fi
                in_frontmatter=false
            fi
        elif [ "$in_frontmatter" = true ]; then
            # Inside frontmatter
            if [[ "$line" =~ ^model: ]]; then
                # Replace existing model field
                echo "model: opus" >> "$temp_file"
                model_added=true
            elif [[ "$line" =~ ^modelId: ]]; then
                # Replace existing modelId field
                echo "modelId: claude-opus-4-1-20250805" >> "$temp_file"
                modelid_found=true
            else
                echo "$line" >> "$temp_file"
            fi
        else
            echo "$line" >> "$temp_file"
        fi
    done < "$file"
    
    # Move temp file to original
    if mv "$temp_file" "$file" 2>/dev/null; then
        echo "✅"
        ((UPDATED_COUNT++))
    else
        echo "❌ Failed to update"
        rm -f "$temp_file"
        ((FAILED_COUNT++))
    fi
}

# Find all agent markdown files
echo "Scanning for agent files in $AGENTS_DIR..."
echo ""

# Process all .md files recursively
while IFS= read -r -d '' file; do
    update_agent "$file"
done < <(find "$AGENTS_DIR" -type f -name "*.md" -print0 2>/dev/null)

echo ""
echo "========================================="
echo "Summary:"
echo "✅ Successfully updated: $UPDATED_COUNT agents"
if [ $FAILED_COUNT -gt 0 ]; then
    echo "❌ Failed to update: $FAILED_COUNT agents"
fi
echo "========================================="

# Verify changes
echo ""
echo "Verifying changes (showing first 10 agents):"
echo "-----------------------------------------"
find "$AGENTS_DIR" -type f -name "*.md" -exec grep -H "^model:" {} \; 2>/dev/null | head -10

echo ""
echo "✨ Script completed!"
echo ""
echo "Note: Restart Claude or reload agents for changes to take effect."