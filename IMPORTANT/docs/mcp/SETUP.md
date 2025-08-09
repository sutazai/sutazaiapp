MCP Server Setup and Testing
=================================

Overview
- Primary MCP server: `mcp_server/index.js` (Node, stdio + HTTP health)
- 3-in-1 integrations covered:
  - SequentialThinking MCP container (`mcp/sequentialthinking`)
  - Upstash Context7 integration (optional)
  - SutazAI backend + Ollama

Prerequisites
- Node 18+
- Docker (for building and running SequentialThinking container)
- Docker Compose (optional but recommended)

Build the SequentialThinking Image
- Preferred (narrow context):
  docker build -t mcp/sequentialthinking \
    -f servers/src/sequentialthinking/Dockerfile \
    servers/src/sequentialthinking

- Fallback (repo root context):
  docker build -t mcp/sequentialthinking \
    -f servers/src/sequentialthinking/Dockerfile \
    servers

- Helper script:
  scripts/mcp/build_sequentialthinking.sh

Smoke Test the SequentialThinking Image
- Run:
  scripts/mcp/sequentialthinking_smoke.sh

- Direct commands equivalent:
  docker run --rm -i mcp/sequentialthinking \
    --input '{"thought":"Test","nextThoughtNeeded":false,"thoughtNumber":1,"totalThoughts":1}'

Start the MCP Server (Local Node)
- Run:
  scripts/mcp/run_server.sh

- Env vars (defaults in parentheses):
  BACKEND_API_URL (http://localhost:8000)
  OLLAMA_URL (http://localhost:11434)
  SEQUENTIAL_THINKING_IMAGE (mcp/sequentialthinking)
  MCP_HTTP_PORT (3030)

- Health endpoints:
  curl -f http://localhost:3030/health
  curl -f http://localhost:3030/info

Start via Docker Compose
- Profile: mcp
- Command:
  docker-compose --profile mcp up -d mcp-server

- Health port forwarded: 11190 -> 3030
- Check:
  curl -f http://localhost:11190/health

Claude Desktop Integration
- File: `claude.code.config.json`
- Points to:
  - sutazai-mcp-server: command `node /opt/sutazaiapp/mcp_server/index.js`
  - sequentialthinking: command `docker run --rm -i mcp/sequentialthinking`
- Ensure claude desktop picks up this config (locally or via symbolic symlink in ~/.claude or app settings).

Codex CLI Integration (stdio)
- Codex CLI can talk to MCP servers over stdio; use the same entry:
  node /opt/sutazaiapp/mcp_server/index.js

- Smoke test (headless stdio tests):
  cd mcp_server
  npm test  # requires deps installed

Testing
- Node test suite: `mcp_server/test.js` covers startup, tools, resources, and basic performance.
- Minimal flow without full stack:
  - Allows DB/Backend failures to be warned and not counted as hard failures.

Troubleshooting
- Missing Node deps:
  cd mcp_server && npm ci   # requires network

- Docker missing:
  Install Docker or skip SequentialThinking tools.

- Backend/Ollama down:
  Tools like `monitor_system` and `manage_model` will report clear errors in responses.

