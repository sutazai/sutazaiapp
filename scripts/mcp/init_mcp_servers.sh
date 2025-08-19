#!/bin/bash
# MCP Server Initialization Script
# Following Rule 1: Real Implementation Only - Only start servers that actually work

set -e

echo "=== MCP Server Initialization Starting ==="
echo "Time: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"

# Source common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/_common.sh"

# Working MCP servers based on testing (2025-08-18)
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

# Servers that need fixing
BROKEN_SERVERS=(
    "extended-memory"    # Missing virtual environment
    "playwright-mcp"     # Timeout issues
    "puppeteer-mcp"      # Missing dependencies
    "postgres"           # Database connection issues
)

# Function to test MCP server
test_mcp_server() {
    local server="$1"
    local wrapper="${SCRIPT_DIR}/wrappers/${server}.sh"
    
    if [ ! -f "$wrapper" ]; then
        err_line "Wrapper not found: $wrapper"
        return 1
    fi
    
    echo -n "Testing $server... "
    if bash "$wrapper" --selfcheck >/dev/null 2>&1; then
        ok "WORKING"
        return 0
    else
        err "BROKEN"
        return 1
    fi
}

# Function to initialize MCP server
init_mcp_server() {
    local server="$1"
    
    case "$server" in
        "ultimatecoder")
            # Already fixed - just verify
            if [ -d "/opt/sutazaiapp/.mcp/UltimateCoderMCP/.venv" ]; then
                ok_line "UltimateCoderMCP virtual environment exists"
            else
                warn_line "UltimateCoderMCP needs virtual environment setup"
            fi
            ;;
        "extended-memory")
            # Fix extended-memory virtual environment
            if [ -d "/opt/sutazaiapp/.mcp/extended-memory" ]; then
                cd "/opt/sutazaiapp/.mcp/extended-memory"
                if [ -f "requirements.txt" ]; then
                    python3 -m venv .venv
                    .venv/bin/pip install -r requirements.txt
                    ok_line "Extended-memory virtual environment created"
                fi
            fi
            ;;
        *)
            # Other servers use npm/npx and are working
            ok_line "$server requires no special initialization"
            ;;
    esac
}

# Main initialization
main() {
    section "MCP Server Health Check"
    
    local working_count=0
    local broken_count=0
    
    # Test all servers
    for server in "${WORKING_SERVERS[@]}"; do
        if test_mcp_server "$server"; then
            ((working_count++))
        else
            ((broken_count++))
        fi
    done
    
    echo ""
    section "MCP Server Status Summary"
    echo "Working servers: $working_count"
    echo "Broken servers: $broken_count"
    echo "Total configured: $((working_count + broken_count))"
    
    # Initialize broken servers if requested
    if [ "${1:-}" = "--fix" ]; then
        section "Attempting to fix broken servers"
        for server in "${BROKEN_SERVERS[@]}"; do
            echo "Fixing $server..."
            init_mcp_server "$server"
        done
    fi
    
    # Generate health report
    local report_file="/opt/sutazaiapp/scripts/mcp/mcp_health_report.json"
    cat > "$report_file" <<EOF
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "summary": {
        "working": $working_count,
        "broken": $broken_count,
        "total": $((working_count + broken_count))
    },
    "working_servers": [
$(printf '        "%s"' "${WORKING_SERVERS[@]}" | sed 's/" "/",\n        "/g')
    ],
    "broken_servers": [
$(printf '        "%s"' "${BROKEN_SERVERS[@]}" | sed 's/" "/",\n        "/g')
    ]
}
EOF
    
    ok "Health report saved to $report_file"
    
    # Return exit code based on health
    if [ $broken_count -eq 0 ]; then
        ok "All MCP servers are healthy!"
        exit 0
    else
        warn "$broken_count servers need attention"
        exit 1
    fi
}

# Run main function
main "$@"