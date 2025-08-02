#!/bin/bash
# Agent Startup Wrapper with CLAUDE.md Rules Enforcement

set -e

echo "==========================================="
echo "Starting Agent with CLAUDE.md Rules"
echo "==========================================="

# Export CLAUDE.md path
export CLAUDE_RULES_PATH="/opt/sutazaiapp/CLAUDE.md"
export ENFORCE_CLAUDE_RULES="true"

# Check if CLAUDE.md exists
if [ -f "$CLAUDE_RULES_PATH" ]; then
    echo "✓ CLAUDE.md rules found at: $CLAUDE_RULES_PATH"
else
    echo "⚠ Warning: CLAUDE.md not found at: $CLAUDE_RULES_PATH"
fi

# Display key rules
echo ""
echo "Key Rules to Follow:"
echo "1. No Fantasy Elements (BLOCKING)"
echo "2. Do Not Break Existing Functionality (BLOCKING)"
echo "3. Analyze Everything Before Proceeding (WARNING)"
echo "4. Reuse Before Creating (WARNING)"
echo "5. Treat as Professional Project (WARNING)"
echo ""

# Run the actual agent
if [ $# -eq 0 ]; then
    echo "Error: No agent command provided"
    exit 1
fi

echo "Starting agent: $@"
exec "$@"
