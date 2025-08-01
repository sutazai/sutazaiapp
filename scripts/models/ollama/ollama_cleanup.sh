#!/bin/bash
# Clean up unused models and cache

# Remove old cache files (older than 7 days)
docker exec sutazai-ollama find /root/.ollama/cache -type f -mtime +7 -delete 2>/dev/null

# Clean temporary files
docker exec sutazai-ollama find /root/.ollama/tmp -type f -mtime +1 -delete 2>/dev/null

# Report disk usage
DISK_USAGE=$(docker exec sutazai-ollama du -sh /root/.ollama/ 2>/dev/null | cut -f1)
echo "Ollama disk usage: $DISK_USAGE"
