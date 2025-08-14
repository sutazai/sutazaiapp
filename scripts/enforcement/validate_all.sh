#!/bin/bash
# 🔧 SUTAZAI COMPREHENSIVE RULE VALIDATION
# Complete validation and reporting system for all 20 Fundamental Rules

set -e

echo "🔧 SUTAZAI COMPREHENSIVE RULE VALIDATION"
echo "========================================================"
echo "🎯 Validating ALL 20 Fundamental Rules + Core Principles"
echo "📅 Date: $(date -u)"
echo "========================================================"

# Create reports directory
mkdir -p /opt/sutazaiapp/reports/enforcement

# Set output file
REPORT_FILE="/opt/sutazaiapp/reports/enforcement/validation_report_$(date +%Y%m%d_%H%M%S).txt"

echo "📄 Report will be saved to: $REPORT_FILE"

# Run comprehensive validation
{
    echo "🔧 SUTAZAI RULE ENFORCEMENT - COMPREHENSIVE VALIDATION"
    echo "========================================================"
    echo "Execution Date: $(date -u)"
    echo "Codebase Root: /opt/sutazaiapp"
    echo "Total Rules: 20 Fundamental Rules + 14 Core Principles"
    echo "========================================================"
    echo ""
    
    # Run the rule validator
    echo "📌 EXECUTING RULE VALIDATION..."
    echo "----------------------------------------"
    python3 /opt/sutazaiapp/scripts/enforcement/rule_validator_simple.py
    
    echo ""
    echo "========================================================"
    echo "📊 ADDITIONAL COMPLIANCE ANALYSIS"
    echo "========================================================"
    
    # File counts
    echo "📁 CODEBASE STATISTICS:"
    echo "- Python files: $(find /opt/sutazaiapp -name '*.py' | wc -l)"
    echo "- JavaScript files: $(find /opt/sutazaiapp -name '*.js' | wc -l)"
    echo "- TypeScript files: $(find /opt/sutazaiapp -name '*.ts' | wc -l)"
    echo "- Dockerfile files: $(find /opt/sutazaiapp -name 'Dockerfile*' | wc -l)"
    echo "- Test files: $(find /opt/sutazaiapp -name 'test_*.py' -o -name '*_test.py' | wc -l)"
    echo "- Documentation files: $(find /opt/sutazaiapp -name '*.md' | wc -l)"
    
    echo ""
    echo "🔍 DETAILED RULE ANALYSIS:"
    echo "----------------------------------------"
    
    # Rule 6: Documentation centralization
    echo "📋 Rule 6 - Centralized Documentation:"
    if [ -d "/opt/sutazaiapp/docs" ]; then
        echo "  ✅ /docs/ directory exists"
        echo "  📄 Documentation files in /docs/: $(find /opt/sutazaiapp/docs -name '*.md' | wc -l)"
    else
        echo "  ❌ /docs/ directory missing"
    fi
    
    # Rule 7: Script organization
    echo "📋 Rule 7 - Script Organization:"
    if [ -d "/opt/sutazaiapp/scripts" ]; then
        echo "  ✅ /scripts/ directory exists"
        echo "  📜 Script directories: $(find /opt/sutazaiapp/scripts -type d | wc -l)"
        echo "  🐍 Python scripts: $(find /opt/sutazaiapp/scripts -name '*.py' | wc -l)"
        echo "  📜 Shell scripts: $(find /opt/sutazaiapp/scripts -name '*.sh' | wc -l)"
    else
        echo "  ❌ /scripts/ directory missing"
    fi
    
    # Rule 16: Local LLM Operations
    echo "📋 Rule 16 - Local LLM Operations:"
    if [ -f "/opt/sutazaiapp/docker-compose.yml" ]; then
        if grep -q "ollama" /opt/sutazaiapp/docker-compose.yml; then
            echo "  ✅ Ollama service configured in docker-compose.yml"
        else
            echo "  ❌ Ollama service not found in docker-compose.yml"
        fi
        if grep -q "tinyllama" /opt/sutazaiapp/docker-compose.yml; then
            echo "  ✅ TinyLlama model referenced"
        else
            echo "  ⚠️  TinyLlama model not explicitly referenced"
        fi
    fi
    
    # Rule 20: MCP Server Protection
    echo "📋 Rule 20 - MCP Server Protection:"
    if [ -f "/opt/sutazaiapp/.mcp.json" ]; then
        echo "  ✅ .mcp.json exists"
        echo "  🔒 MCP servers configured: $(grep -c '"name"' /opt/sutazaiapp/.mcp.json || echo 0)"
    else
        echo "  ❌ .mcp.json missing"
    fi
    
    echo ""
    echo "========================================================"
    echo "🎯 VALIDATION COMPLETE"
    echo "========================================================"
    echo "📄 Full report saved to: $REPORT_FILE"
    echo "📅 Execution completed: $(date -u)"
    
} | tee "$REPORT_FILE"

echo ""
echo "✅ COMPREHENSIVE VALIDATION COMPLETE"
echo "📄 Report available at: $REPORT_FILE"