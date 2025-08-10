#!/bin/bash
# Make all chaos engineering scripts executable

set -euo pipefail


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

CHAOS_DIR="/opt/sutazaiapp/chaos"

echo "Making chaos engineering scripts executable..."

# Find all shell scripts and make them executable
find "$CHAOS_DIR" -name "*.sh" -type f -exec chmod +x {} \;

# Make Python scripts executable
find "$CHAOS_DIR" -name "*.py" -type f -exec chmod +x {} \;

echo "Done. All chaos engineering scripts are now executable."

# List executable scripts
echo ""
echo "Executable scripts:"
find "$CHAOS_DIR" -name "*.sh" -o -name "*.py" | sort