#!/usr/bin/env bash
set -euo pipefail

# Purpose: Run a smoke test against the built mcp/sequentialthinking image.
# Usage: scripts/mcp/sequentialthinking_smoke.sh


# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

if ! command -v docker >/dev/null 2>&1; then
  echo "[mcp] Docker not available. Install Docker or use Compose in CI."
  exit 1
fi

IMAGE=${SEQUENTIAL_THINKING_IMAGE:-mcp/sequentialthinking}

echo "[mcp] Running SequentialThinking smoke test using ${IMAGE}..."
set -x
docker run --rm -i "${IMAGE}" \
  --input '{"thought":"Test","nextThoughtNeeded":false,"thoughtNumber":1,"totalThoughts":1}'
set +x

echo "[mcp] Smoke test completed."

