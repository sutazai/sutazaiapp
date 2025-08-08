#!/usr/bin/env bash
set -euo pipefail

# Purpose: Run a smoke test against the built mcp/sequentialthinking image.
# Usage: scripts/mcp/sequentialthinking_smoke.sh

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

