#!/usr/bin/env bash
set -Eeuo pipefail

# ---------- config ----------
PROJECT_DIR="${PROJECT_DIR:-/opt/sutazaiapp}"
REPO_SLUG="${REPO_SLUG:-sutazai/sutazaiapp}"   # for optional GitHub MCP
CHROMA_URL_DEFAULT="http://sutazai-chromadb:8000"
MCP_TIMEOUT="${MCP_TIMEOUT:-20000}"
export DEBIAN_FRONTEND=noninteractive

log(){ printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }
ok(){  printf "\033[1;32m✓ %s\033[0m\n" "$*"; }
warn(){ printf "\033[1;33m! %s\033[0m\n" "$*"; }
die(){ printf "\033[1;31m✗ %s\033[0m\n" "$*"; exit 1; }

cd "$PROJECT_DIR" 2>/dev/null || die "Project dir $PROJECT_DIR not found"

command -v claude >/dev/null 2>&1 || die "Claude CLI not on PATH. Open a shell where 'claude' works."

log "Preflight"
apt-get update -y >/dev/null 2>&1 || true
apt-get install -y jq git curl ca-certificates >/dev/null 2>&1 || true

ensure_node() {
  if ! command -v node >/dev/null 2>&1 || ! command -v npx >/dev/null 2>&1; then
    export NVM_DIR="/root/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] || curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
    . "$NVM_DIR/nvm.sh"
    nvm install --lts >/dev/null
    nvm use --lts >/dev/null
  fi
}

ensure_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
    export PATH="$HOME/.local/bin:$PATH"
  fi
}

add_mcp() {
  local name="$1"; shift
  claude mcp remove "$name" -s project >/dev/null 2>&1 || true
  claude mcp add --scope project "$name" -- "$@"
}

# discover docker network + DB for Postgres MCP
DB_CONT="sutazai-postgres"
NET="$(docker network ls --format '{{.Name}}' | grep -E '^sutazai' | head -n1 || true)"
[ -n "$NET" ] || NET="bridge"

DB_URI=""
if docker inspect "$DB_CONT" >/dev/null 2>&1; then
  DB_HOST="$(docker inspect -f '{{.Name}}' "$DB_CONT" | sed 's#^/##')"
  DB_USER="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$DB_CONT" | sed -n 's/^POSTGRES_USER=//p' | tail -n1)"
  DB_PASS="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$DB_CONT" | sed -n 's/^POSTGRES_PASSWORD=//p' | tail -n1)"
  DB_NAME="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$DB_CONT" | sed -n 's/^POSTGRES_DB=//p' | tail -n1)"
  DB_URI="postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:5432/${DB_NAME}"
fi

CHROMA_URL="${CHROMA_URL:-$CHROMA_URL_DEFAULT}"

ok "Docker NET=$NET"
[ -n "$DB_URI" ] && ok "DB URI discovered" || warn "Postgres container not found—will skip 'postgres' MCP"

log "Reference MCPs (official)"
add_mcp sequentialthinking docker run --rm -i mcp/sequentialthinking
ok "sequentialthinking added"

ensure_node
add_mcp context7 npx -y @upstash/context7-mcp@latest
ok "context7 added"

# Filesystem (official)
add_mcp files npx -y @modelcontextprotocol/server-filesystem "$PROJECT_DIR"
ok "files added"

# Fetch / HTTP (official)
add_mcp http docker run --rm -i mcp/fetch
ok "http (fetch) added"

# Postgres (community docker)
if [ -n "$DB_URI" ]; then
  add_mcp postgres docker run --network "$NET" -i --rm -e DATABASE_URI="$DB_URI" crystaldba/postgres-mcp --access-mode=restricted
  ok "postgres added"
fi

log "Developer helpers"
# Extended memory (Python MCP)
ensure_uv
if [ ! -d "$PROJECT_DIR/.venvs/extended-memory" ]; then
  uv venv "$PROJECT_DIR/.venvs/extended-memory"
  "$PROJECT_DIR/.venvs/extended-memory/bin/python" -m pip install -U pip >/dev/null 2>&1 || true
  "$PROJECT_DIR/.venvs/extended-memory/bin/python" -m pip install -q extended-memory-mcp
fi
add_mcp extended-memory "$PROJECT_DIR/.venvs/extended-memory/bin/python" -m extended_memory_mcp.server
ok "extended-memory added"

# SSH (sinjab/mcp_ssh)
if [ ! -d "$PROJECT_DIR/mcp_ssh/.git" ]; then
  git clone https://github.com/sinjab/mcp_ssh "$PROJECT_DIR/mcp_ssh" >/dev/null 2>&1 || true
fi
add_mcp mcp_ssh uv --directory "$PROJECT_DIR/mcp_ssh" run mcp_ssh
ok "mcp_ssh added"

# Nx (nx-mcp)
ensure_node
add_mcp nx-mcp npx -y nx-mcp@latest
ok "nx-mcp added"

# Language server (Go + TS LSP)
ensure_node
if ! command -v mcp-language-server >/dev/null 2>&1; then
  apt-get install -y golang-go >/dev/null 2>&1 || true
  export PATH="$PATH:$(go env GOPATH 2>/dev/null)/bin"
  go install github.com/isaacphi/mcp-language-server@latest >/dev/null 2>&1 || true
fi
TS_LSP_BIN="$(command -v typescript-language-server || true)"
if [ -z "$TS_LSP_BIN" ]; then
  npm i -g typescript typescript-language-server >/dev/null 2>&1 || true
  TS_LSP_BIN="$(command -v typescript-language-server || true)"
fi
MLS_BIN="$(command -v mcp-language-server || echo /root/go/bin/mcp-language-server)"
[ -x "$MLS_BIN" ] && [ -x "$TS_LSP_BIN" ] && add_mcp language-server "$MLS_BIN" --workspace "$PROJECT_DIR" --lsp "$TS_LSP_BIN" -- --stdio && ok "language-server added" || warn "language-server skipped (missing Go LSP binaries)"

log "RAG helpers"
# Web Search MCP (mrkrsl)
ensure_node
add_mcp web-search npx -y web-search-mcp
ok "web-search added"

# Chroma MCP (privetin/chroma)
ensure_uv
mkdir -p "$PROJECT_DIR/.mcp"
if [ ! -d "$PROJECT_DIR/.mcp/chroma/.git" ]; then
  git clone https://github.com/privetin/chroma "$PROJECT_DIR/.mcp/chroma" >/dev/null 2>&1 || true
fi
add_mcp chroma uv --directory "$PROJECT_DIR/.mcp/chroma" run chroma
ok "chroma added (expects CHROMA_URL=$CHROMA_URL if the server reads it)"

log "Optional: GitHub MCP (official; needs free PAT)"
if [ -n "${GITHUB_TOKEN:-}" ]; then
  ensure_node
  add_mcp github sh -lc "GITHUB_TOKEN='${GITHUB_TOKEN}' npx -y @modelcontextprotocol/server-github --repositories '${REPO_SLUG}'"
  ok "github added for ${REPO_SLUG}"
else
  warn "Skipped github (export GITHUB_TOKEN to enable)."
fi

log "Final verification"
claude mcp reset-project-choices
MCP_TIMEOUT="$MCP_TIMEOUT" claude mcp list || true
