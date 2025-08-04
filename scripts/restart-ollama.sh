#!/bin/bash
# Ollama Restart Script

echo "Restarting Ollama service..."
docker-compose -f /opt/sutazaiapp/docker-compose.yml restart ollama

echo "Waiting for service to be ready..."
sleep 10

if curl -f -s http://localhost:10104/api/tags >/dev/null; then
    echo "✅ Ollama restarted successfully"
else
    echo "❌ Ollama restart failed"
    exit 1
fi