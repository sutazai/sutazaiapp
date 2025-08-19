#!/bin/bash

echo "======================================"
echo "  CLAUDE ACCURACY VERIFICATION TEST  "
echo "======================================"
echo

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0

# Function to check test results
check_test() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
        ((PASS_COUNT++))
    else
        echo -e "${RED}✗${NC} $2"
        ((FAIL_COUNT++))
    fi
}

echo "1. CHECKING ANTI-HALLUCINATION HOOK"
echo "-----------------------------------"
if [ -f /root/.claude/hooks/reduce-hallucination.sh ]; then
    check_test 0 "Hook file exists"
    
    # Test if executable
    if [ -x /root/.claude/hooks/reduce-hallucination.sh ]; then
        check_test 0 "Hook is executable"
        
        # Test hook output
        OUTPUT=$(/root/.claude/hooks/reduce-hallucination.sh 2>&1)
        if [ $? -eq 0 ]; then
            check_test 0 "Hook executes without errors"
            
            # Check if output contains accuracy instructions
            if echo "$OUTPUT" | grep -q "CRITICAL INSTRUCTIONS"; then
                check_test 0 "Hook contains accuracy instructions"
            else
                check_test 1 "Hook missing accuracy instructions"
            fi
        else
            check_test 1 "Hook execution failed"
        fi
    else
        check_test 1 "Hook not executable"
    fi
else
    check_test 1 "Hook file not found"
fi

echo
echo "2. CHECKING SETTINGS CONFIGURATION"
echo "-----------------------------------"
if [ -f /root/.claude/settings.json ]; then
    check_test 0 "Settings file exists"
    
    # Check hook configuration
    if grep -q "UserPromptSubmit" /root/.claude/settings.json; then
        check_test 0 "UserPromptSubmit hook configured"
        
        # Check if our hook is referenced
        if grep -q "reduce-hallucination.sh" /root/.claude/settings.json; then
            check_test 0 "Anti-hallucination hook linked in settings"
        else
            check_test 1 "Anti-hallucination hook not linked"
        fi
    else
        check_test 1 "UserPromptSubmit hook not configured"
    fi
    
    # Check model setting
    if grep -q '"model": "opus"' /root/.claude/settings.json; then
        check_test 0 "Using Opus model (optimized for accuracy)"
    else
        check_test 1 "Not using Opus model"
    fi
else
    check_test 1 "Settings file not found"
fi

echo
echo "3. CHECKING OUTPUT STYLE"
echo "------------------------"
if [ -f /root/.claude/output-styles/accurate.md ]; then
    check_test 0 "Accurate output style exists"
    
    # Check style content
    if grep -q "name: accurate" /root/.claude/output-styles/accurate.md; then
        check_test 0 "Output style properly formatted"
    else
        check_test 1 "Output style format invalid"
    fi
    
    # Check for accuracy principles
    if grep -q "Verify Before Claiming" /root/.claude/output-styles/accurate.md; then
        check_test 0 "Style contains verification requirements"
    else
        check_test 1 "Style missing verification requirements"
    fi
else
    check_test 1 "Accurate output style not found"
fi

echo
echo "4. CHECKING CLAUDE.md UPDATES"
echo "-----------------------------"
if [ -f /opt/sutazaiapp/CLAUDE.md ]; then
    check_test 0 "CLAUDE.md exists"
    
    # Check for anti-hallucination protocol
    if grep -q "ANTI-HALLUCINATION PROTOCOL" /opt/sutazaiapp/CLAUDE.md; then
        check_test 0 "Anti-hallucination protocol added"
        
        # Check for accuracy requirements
        if grep -q "ALWAYS VERIFY" /opt/sutazaiapp/CLAUDE.md; then
            check_test 0 "Verification requirements documented"
        else
            check_test 1 "Verification requirements missing"
        fi
    else
        check_test 1 "Anti-hallucination protocol not found"
    fi
else
    check_test 1 "CLAUDE.md not found"
fi

echo
echo "5. CHECKING INTERCEPTOR TOOLS"
echo "-----------------------------"
if command -v mitmdump &> /dev/null; then
    check_test 0 "mitmproxy installed"
    
    # Check interceptor script
    if [ -f /opt/sutazaiapp/scripts/claude-temp-interceptor.py ]; then
        check_test 0 "Temperature interceptor script exists"
        
        # Check if script has correct temperature setting
        if grep -q '"temperature"] = 0.2' /opt/sutazaiapp/scripts/claude-temp-interceptor.py; then
            check_test 0 "Interceptor sets temperature to 0.2"
        else
            check_test 1 "Interceptor temperature setting incorrect"
        fi
    else
        check_test 1 "Temperature interceptor script missing"
    fi
else
    check_test 1 "mitmproxy not installed"
fi

# Check claude-code-router
if npm list -g @musistudio/claude-code-router --depth=0 2>/dev/null | grep -q "claude-code-router"; then
    check_test 0 "claude-code-router installed"
else
    check_test 1 "claude-code-router not installed"
fi

echo
echo "======================================"
echo "            TEST SUMMARY              "
echo "======================================"
echo -e "Passed: ${GREEN}$PASS_COUNT${NC}"
echo -e "Failed: ${RED}$FAIL_COUNT${NC}"
echo

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}SUCCESS!${NC} All anti-hallucination measures are properly configured."
    echo
    echo "The system is now configured to:"
    echo "• Inject accuracy instructions with every prompt"
    echo "• Use verification-focused output style"
    echo "• Document accuracy requirements in CLAUDE.md"
    echo "• Intercept API calls to set low temperature (when proxy is active)"
elif [ $PASS_COUNT -gt 15 ]; then
    echo -e "${YELLOW}MOSTLY CONFIGURED${NC} - Most anti-hallucination measures are in place."
    echo "Some components may need attention, but core functionality is working."
else
    echo -e "${RED}CONFIGURATION INCOMPLETE${NC} - Several components need to be fixed."
fi

echo
echo "To activate maximum accuracy mode:"
echo "1. Hooks are automatically active (already configured)"
echo "2. Run with proxy: bash /opt/sutazaiapp/scripts/claude-accurate-mode.sh"
echo "3. Or use: claude output-style set accurate"