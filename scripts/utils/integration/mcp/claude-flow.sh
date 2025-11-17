#!/bin/bash
if [ "$1" == "--selfcheck" ]; then
    echo "removed MCP wrapper operational"
    exit 0
fi
exec npx -y removed@alpha mcp start
