#!/bin/bash

# HTTP Fetch MCP Server Wrapper
# Uses the official @modelcontextprotocol/server-fetch

set -e

# Self-check for health monitoring
if [ "$1" = "--selfcheck" ]; then
    if command -v node >/dev/null 2>&1 && command -v npx >/dev/null 2>&1; then
        echo '{"healthy":true,"service":"http-fetch"}'
    else
        echo '{"healthy":false,"error":"node or npx not found"}'
        exit 1
    fi
    exit 0
fi

# Run the HTTP fetch MCP server
exec npx -y @modelcontextprotocol/server-fetch