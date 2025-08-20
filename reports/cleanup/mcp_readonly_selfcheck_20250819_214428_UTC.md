# MCP Read-only Self-check
Generated: 20250819_214428_UTC

## Rule 20 Notice
This script is read-only and does not modify any MCP configuration or runtime.

## Config Presence
- .mcp.json: PRESENT
- wrappers dir: PRESENT (/opt/sutazaiapp/scripts/mcp/wrappers)

## Declared STDIO Servers in .mcp.json (grep fallback)
  10:      "type": "stdio"
  19:      "type": "stdio"
  24:      "type": "stdio"
  29:      "type": "stdio"
  34:      "type": "stdio"
  39:      "type": "stdio"
  44:      "type": "stdio"
  49:      "type": "stdio"
  54:      "type": "stdio"
  59:      "type": "stdio"
  64:      "type": "stdio"
  69:      "type": "stdio"
  74:      "type": "stdio"
  79:      "type": "stdio"
  84:      "type": "stdio"
  89:      "type": "stdio"
  94:      "type": "stdio"
  99:      "type": "stdio"
  104:      "type": "stdio"

### Commands declared
  4:      "command": "npx",
  13:      "command": "npx",
  22:      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/files.sh",
  27:      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/context7.sh",
  32:      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/http_fetch.sh",
  37:      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/ddg.sh",
  42:      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/sequentialthinking.sh",
  47:      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/nx-mcp.sh",
  52:      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/extended-memory.sh",
  57:      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/mcp_ssh.sh",
  62:      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/ultimatecoder.sh",
  67:      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/playwright-mcp.sh",
  72:      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/memory-bank-mcp.sh",
  77:      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/knowledge-graph-mcp.sh",
  82:      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/compass-mcp.sh",
  87:      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/github.sh",
  92:      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/http.sh",
  97:      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/language-server.sh",
  102:      "command": "/opt/sutazaiapp/scripts/mcp/wrappers/claude-task-runner.sh",

## Wrapper Scripts Present
  - /opt/sutazaiapp/scripts/mcp/wrappers/claude-flow.sh
  - /opt/sutazaiapp/scripts/mcp/wrappers/claude-task-runner.sh
  - /opt/sutazaiapp/scripts/mcp/wrappers/compass-mcp.sh
  - /opt/sutazaiapp/scripts/mcp/wrappers/context7.sh
  - /opt/sutazaiapp/scripts/mcp/wrappers/ddg.sh
  - /opt/sutazaiapp/scripts/mcp/wrappers/extended-memory.sh
  - /opt/sutazaiapp/scripts/mcp/wrappers/files.sh
  - /opt/sutazaiapp/scripts/mcp/wrappers/github.sh
  - /opt/sutazaiapp/scripts/mcp/wrappers/http_fetch.sh
  - /opt/sutazaiapp/scripts/mcp/wrappers/knowledge-graph-mcp.sh
  - /opt/sutazaiapp/scripts/mcp/wrappers/language-server.sh
  - /opt/sutazaiapp/scripts/mcp/wrappers/mcp_ssh.sh
  - /opt/sutazaiapp/scripts/mcp/wrappers/memory-bank-mcp.sh
  - /opt/sutazaiapp/scripts/mcp/wrappers/nx-mcp.sh
  - /opt/sutazaiapp/scripts/mcp/wrappers/playwright-mcp.sh
  - /opt/sutazaiapp/scripts/mcp/wrappers/postgres.sh
  - /opt/sutazaiapp/scripts/mcp/wrappers/puppeteer-mcp.sh
  - /opt/sutazaiapp/scripts/mcp/wrappers/ruv-swarm.sh
  - /opt/sutazaiapp/scripts/mcp/wrappers/sequentialthinking.sh
  - /opt/sutazaiapp/scripts/mcp/wrappers/ultimatecoder.sh
  - /opt/sutazaiapp/scripts/mcp/wrappers/unified-dev.sh
  - /opt/sutazaiapp/scripts/mcp/wrappers/unified-memory.sh

## Basic Dependency Checks (npx, node, python, jq)
- npx: OK (/root/.nvm/versions/node/v22.18.0/bin/npx)
- node: OK (/root/.nvm/versions/node/v22.18.0/bin/node)
- python3: OK (/usr/bin/python3)
- jq: OK (/usr/bin/jq)

## Next Steps
- Use existing scripts in scripts/mcp/* for deeper read-only validation (dry-run if supported).
- Do not start/stop any MCP servers without explicit authorization.
