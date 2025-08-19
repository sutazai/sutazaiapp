#!/bin/bash
# MCP Setup Validation Script
# Validates MCP configuration and tests each server

set -e

echo "=== MCP Setup Validation ==="
echo "Time: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASS=0
FAIL=0
WARN=0

# Function to test a condition
test_condition() {
    local description="$1"
    local command="$2"
    
    echo -n "Testing: $description... "
    
    if eval "$command" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASS++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((FAIL++))
        return 1
    fi
}

# Function to check file exists
check_file() {
    local file="$1"
    local desc="$2"
    
    test_condition "$desc" "[ -f '$file' ]"
}

# Function to check directory exists
check_dir() {
    local dir="$1"
    local desc="$2"
    
    test_condition "$desc" "[ -d '$dir' ]"
}

echo "=== Configuration Files ==="
check_file "/opt/sutazaiapp/.mcp.json" ".mcp.json configuration file"
check_file "/opt/sutazaiapp/scripts/mcp/_common.sh" "MCP common functions"
check_file "/opt/sutazaiapp/scripts/mcp/init_mcp_servers.sh" "MCP initialization script"

echo ""
echo "=== MCP Directories ==="
check_dir "/opt/sutazaiapp/scripts/mcp/wrappers" "MCP wrappers directory"
check_dir "/opt/sutazaiapp/.mcp" "MCP installation directory"

echo ""
echo "=== Virtual Environments ==="
check_dir "/opt/sutazaiapp/.mcp/UltimateCoderMCP/.venv" "UltimateCoderMCP virtual environment"
test_condition "UltimateCoderMCP fastmcp module" "/opt/sutazaiapp/.mcp/UltimateCoderMCP/.venv/bin/python -c 'import fastmcp'"

echo ""
echo "=== MCP Wrapper Tests ==="

# List of working servers from our testing
WORKING_SERVERS=(
    "files"
    "github"
    "http"
    "ddg"
    "language-server"
    "mcp_ssh"
    "ultimatecoder"
    "context7"
    "compass-mcp"
    "knowledge-graph-mcp"
    "memory-bank-mcp"
    "nx-mcp"
)

for server in "${WORKING_SERVERS[@]}"; do
    wrapper="/opt/sutazaiapp/scripts/mcp/wrappers/${server}.sh"
    if [ -f "$wrapper" ]; then
        test_condition "$server wrapper selfcheck" "bash '$wrapper' --selfcheck"
    else
        echo -e "Testing: $server wrapper selfcheck... ${YELLOW}⚠ MISSING${NC}"
        ((WARN++))
    fi
done

echo ""
echo "=== Backend API Validation ==="
check_file "/opt/sutazaiapp/backend/app/api/v1/endpoints/mcp_consolidated.py" "Consolidated MCP API endpoint"
test_condition "Python syntax check for MCP API" "python3 -m py_compile /opt/sutazaiapp/backend/app/api/v1/endpoints/mcp_consolidated.py"

echo ""
echo "=== MCP Configuration Validation ==="
test_condition ".mcp.json is valid JSON" "python3 -c 'import json; json.load(open(\"/opt/sutazaiapp/.mcp.json\"))'"
test_condition ".mcp.json contains only working servers" "python3 -c '
import json
with open(\"/opt/sutazaiapp/.mcp.json\") as f:
    config = json.load(f)
    working = [\"files\", \"github\", \"http\", \"ddg\", \"language-server\", \"mcp_ssh\", 
               \"ultimatecoder\", \"context7\", \"compass-mcp\", \"knowledge-graph-mcp\", 
               \"memory-bank-mcp\", \"nx-mcp\"]
    configured = list(config[\"mcpServers\"].keys())
    for server in configured:
        assert server in working, f\"{server} is not in working servers list\"
'"

echo ""
echo "=== Summary ==="
echo -e "${GREEN}Passed:${NC} $PASS"
echo -e "${RED}Failed:${NC} $FAIL"
echo -e "${YELLOW}Warnings:${NC} $WARN"

# Generate validation report
REPORT_FILE="/opt/sutazaiapp/scripts/mcp/validation_report.json"
cat > "$REPORT_FILE" <<EOF
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "validation_results": {
        "passed": $PASS,
        "failed": $FAIL,
        "warnings": $WARN,
        "total_tests": $((PASS + FAIL + WARN))
    },
    "status": $([ $FAIL -eq 0 ] && echo '"success"' || echo '"failure"'),
    "working_servers": [
$(printf '        "%s"' "${WORKING_SERVERS[@]}" | sed 's/" "/",\n        "/g')
    ]
}
EOF

echo ""
echo "Validation report saved to: $REPORT_FILE"

# Exit with appropriate code
if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ All critical tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed. Please review and fix.${NC}"
    exit 1
fi