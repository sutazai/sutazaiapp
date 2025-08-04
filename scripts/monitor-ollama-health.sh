#!/bin/bash
# Ollama Health Monitoring Script

while true; do
    echo "=== Ollama Health Check $(date) ==="
    
    # Check service status
    if curl -f -s http://localhost:10104/api/tags >/dev/null; then
        echo "✅ Service: Healthy"
    else
        echo "❌ Service: Unhealthy"
    fi
    
    # Check resource usage
    echo "📊 Resource Usage:"
    docker stats sutazai-ollama --no-stream --format "  CPU: {{.CPUPerc}}  Memory: {{.MemUsage}}"
    
    # Check loaded models
    echo "🧠 Loaded Models:"
    docker exec sutazai-ollama ollama list 2>/dev/null | tail -n +2 | while read line; do
        echo "  $line"
    done
    
    echo ""
    sleep 30
done