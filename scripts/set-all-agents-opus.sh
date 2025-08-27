#!/bin/bash

echo "Setting all agents to use Opus model..."

# Counter for tracking changes
updated=0
total=0

# Find all agent markdown files
for agent_file in $(find /opt/sutazaiapp/.claude/agents -name "*.md" -type f); do
    total=$((total + 1))
    
    # Check if file has modelId field
    if grep -q "^modelId:" "$agent_file"; then
        # Replace existing modelId with opus
        sed -i 's/^modelId:.*/modelId: claude-opus-4-1-20250805/' "$agent_file"
        echo "✓ Updated: $(basename "$agent_file")"
        updated=$((updated + 1))
    else
        # Add modelId field after name field if it doesn't exist
        if grep -q "^name:" "$agent_file"; then
            sed -i '/^name:/a modelId: claude-opus-4-1-20250805' "$agent_file"
            echo "✓ Added modelId to: $(basename "$agent_file")"
            updated=$((updated + 1))
        fi
    fi
done

echo ""
echo "========================================="
echo "Agent Model Update Complete!"
echo "========================================="
echo "Total agents found: $total"
echo "Agents updated: $updated"
echo ""
echo "All agents are now configured to use Opus model (claude-opus-4-1-20250805)"