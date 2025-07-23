#!/bin/bash
# Ollama Health Check

check_ollama_health() {
    # Check if Ollama is responding
    if curl -f -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is healthy"
        
        # Check loaded models
        MODELS=$(curl -s http://localhost:11434/api/tags | jq -r '.models[].name' 2>/dev/null | wc -l)
        echo "Loaded models: $MODELS"
        
        # Check response time
        RESPONSE_TIME=$(curl -o /dev/null -s -w '%{time_total}' http://localhost:11434/api/tags)
        echo "API response time: ${RESPONSE_TIME}s"
        
        # Memory usage
        MEMORY=$(docker stats sutazai-ollama --no-stream --format "{{.MemUsage}}" 2>/dev/null)
        echo "Memory usage: $MEMORY"
        
        return 0
    else
        echo "Ollama is not responding"
        return 1
    fi
}

# Run health check
check_ollama_health
