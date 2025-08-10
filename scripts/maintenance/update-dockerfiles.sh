#!/bin/bash
set -e


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

echo "🔄 Updating existing Dockerfiles to use base images..."

# Find all agent Dockerfiles and update them
find agents/ -name "Dockerfile" -type f | while read dockerfile; do
    if grep -q "FROM python:" "$dockerfile"; then
        echo "Updating $dockerfile to use python-agent-base"
        # Backup original
        cp "$dockerfile" "$dockerfile.backup"
        
        # Replace FROM line and optimize
        sed -e 's|FROM python:.*|FROM sutazai/python-agent-base:latest|' \
            -e '/RUN apt-get update/,/rm -rf \/var\/lib\/apt\/lists\*/d' \
            -e '/RUN pip install.*fastapi\|uvicorn\|pydantic/d' \
            "$dockerfile.backup" > "$dockerfile"
            
        echo "✅ Updated $dockerfile"
    fi
done

echo "🎯 Dockerfile updates complete!"
