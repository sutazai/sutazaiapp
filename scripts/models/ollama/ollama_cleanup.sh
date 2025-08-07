#!/bin/bash
# Purpose: Clean up unused Ollama models and cache
# Usage: ./ollama_cleanup.sh
# Requires: docker access to sutazai-ollama container

set -euo pipefail

# Remove old cache files (older than 7 days)
docker exec sutazai-ollama find /root/.ollama/cache -type f -mtime +7 -delete 2>/dev/null

# Clean temporary files
docker exec sutazai-ollama find /root/.ollama/tmp -type f -mtime +1 -delete 2>/dev/null

# Report disk usage
DISK_USAGE=$(docker exec sutazai-ollama du -sh /root/.ollama/ 2>/dev/null | cut -f1)
echo "Ollama disk usage: $DISK_USAGE"
