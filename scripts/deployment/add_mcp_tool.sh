#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <docker-image> [--no-verify]" >&2
  exit 1
fi

IMAGE="$1"
VERIFY=1
if [ "${2:-}" = "--no-verify" ]; then
  VERIFY=0
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WL_FILE="$ROOT_DIR/mcp_server/external-allowed-images.json"
WL_EXAMPLE="$ROOT_DIR/mcp_server/external-allowed-images.example.json"

if [ ! -f "$WL_FILE" ]; then
  if [ -f "$WL_EXAMPLE" ]; then
    cp "$WL_EXAMPLE" "$WL_FILE"
  else
    echo "[]" > "$WL_FILE"
  fi
fi

echo "[MCP] Adding image to whitelist: $IMAGE"
python3 - "$WL_FILE" "$IMAGE" << 'PY'
import json, sys
path, image = sys.argv[1], sys.argv[2]
try:
    with open(path, 'r') as f:
        data = json.load(f)
except Exception:
    data = []
if image not in data:
    data.append(image)
with open(path, 'w') as f:
    json.dump(data, f, indent=2)
print("OK")
PY

if [ "$VERIFY" -eq 1 ]; then
  echo "[MCP] Verifying image with --help (non-fatal if unsupported)"
  set +e
  docker run --rm "$IMAGE" --help >/dev/null 2>&1
  STATUS=$?
  set -e
  if [ $STATUS -ne 0 ]; then
    echo "[MCP] --help failed (exit $STATUS). Attempting no-arg run..."
    set +e
    docker run --rm "$IMAGE" >/dev/null 2>&1
    STATUS2=$?
    set -e
    if [ $STATUS2 -ne 0 ]; then
      echo "[MCP] Warning: basic verification failed. The image may require arguments. It is still whitelisted."
    else
      echo "[MCP] No-arg run succeeded."
    fi
  else
    echo "[MCP] Verification succeeded."
  fi
fi

echo "[MCP] Updated whitelist at $WL_FILE"
echo "[MCP] You can now call the tool via the 'docker_exec' MCP tool with:"
cat << EOF
{
  "tool": "docker_exec",
  "arguments": {
    "image": "$IMAGE",
    "args": ["--version"]
  }
}
EOF
