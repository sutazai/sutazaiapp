#!/bin/bash
# Enhanced Agent Startup Script with CLAUDE.md Rules Enforcement

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}SutazAI Agent Startup with CLAUDE.md Rules${NC}"
echo -e "${GREEN}================================================${NC}"

# Check for CLAUDE.md
CLAUDE_MD_PATH="/opt/sutazaiapp/CLAUDE.md"
if [ -f "$CLAUDE_MD_PATH" ]; then
    echo -e "${GREEN}✓${NC} CLAUDE.md rules found at: $CLAUDE_MD_PATH"
else
    echo -e "${RED}✗${NC} ERROR: CLAUDE.md not found at: $CLAUDE_MD_PATH"
    exit 1
fi

# Display enforcement rules
echo -e "\n${YELLOW}MANDATORY RULES TO FOLLOW:${NC}"
echo -e "${RED}BLOCKING RULES (Never Violate):${NC}"
echo "  1. No Fantasy Elements - No specific implementation name (e.g., emailSender, dataProcessor), wizards, or mythical references"
echo "  2. Do Not Break Existing Functionality - Always preserve what works"

echo -e "\n${YELLOW}WARNING RULES (Require Careful Consideration):${NC}"
echo "  3. Analyze Everything - Thorough investigation before changes"
echo "  4. Reuse Before Creating - Check for existing solutions first"
echo "  5. Professional Project - Not a playground, maintain standards"

echo -e "\n${GREEN}GUIDANCE RULES (Best Practices):${NC}"
echo "  • Clean, consistent, well-organized code"
echo "  • Proper documentation and testing"
echo "  • Security and performance considerations"

# Export environment variables
export CLAUDE_RULES_PATH="$CLAUDE_MD_PATH"
export ENFORCE_CLAUDE_RULES="true"
export AGENT_MUST_CHECK_RULES="true"

# Create Python rule checker if needed
RULES_CHECKER="/tmp/claude_rules_check.py"
cat > "$RULES_CHECKER" << 'EOF'
import os
import sys

def check_claude_rules():
    """Verify CLAUDE.md rules are accessible"""
    rules_path = os.environ.get('CLAUDE_RULES_PATH', '/opt/sutazaiapp/CLAUDE.md')
    if not os.path.exists(rules_path):
        print(f"ERROR: Cannot find CLAUDE.md at {rules_path}")
        sys.exit(1)
    
    with open(rules_path, 'r') as f:
        content = f.read()
        if 'Rule 1: No Fantasy Elements' in content:
            print("✓ CLAUDE.md rules loaded successfully")
            return True
    
    print("ERROR: CLAUDE.md exists but rules not found")
    return False

if __name__ == "__main__":
    check_claude_rules()
EOF

# Verify rules are accessible
echo -e "\n${YELLOW}Verifying rules accessibility...${NC}"
python3 "$RULES_CHECKER"

# Continue with normal startup
echo -e "\n${GREEN}Starting agent with rules enforcement enabled...${NC}"

# If arguments provided, use them as the command
if [ $# -gt 0 ]; then
    exec "$@"
else
    # Default to running the main agent script
    if [ -f "app.py" ]; then
        exec python3 app.py
    elif [ -f "main.py" ]; then
        exec python3 main.py
    else
        echo -e "${RED}No agent script found to run${NC}"
        exit 1
    fi
fi