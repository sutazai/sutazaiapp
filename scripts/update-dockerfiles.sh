#!/bin/bash
set -e

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
