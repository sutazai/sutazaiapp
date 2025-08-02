#!/bin/bash
# Make all chaos engineering scripts executable

set -euo pipefail

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