#!/bin/bash
"""
Advanced OOM Prevention and Recovery System
Monitors memory usage and prevents Out of Memory conditions
"""

LOG_FILE="/opt/sutazaiapp/logs/oom-prevention.log"
MEMORY_THRESHOLD=80  # Trigger cleanup at 80% memory usage
SWAP_THRESHOLD=50    # Trigger aggressive cleanup at 50% swap usage
OLLAMA_CONTAINER="sutazai-ollama"

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [OOM-PREVENTION] $1" | tee -a "$LOG_FILE"
}

get_memory_usage() {
    free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}'
}

get_swap_usage() {
    free | grep Swap | awk '{if($2>0) printf "%.1f", $3/$2 * 100.0; else print "0.0"}'
}

get_container_memory() {
    docker stats "$OLLAMA_CONTAINER" --no-stream --format "{{.MemPerc}}" 2>/dev/null | sed 's/%//' | head -1 || echo "0"
}

unload_all_ollama_models() {
    log_message "Emergency: Unloading all Ollama models"
    
    # Get list of loaded models
    models=$(curl -s http://localhost:11434/api/ps | jq -r '.models[]?.name' 2>/dev/null || echo "")
    
    if [ -n "$models" ]; then
        echo "$models" | while read -r model; do
            if [ -n "$model" ]; then
                log_message "Unloading model: $model"
                curl -s -X DELETE "http://localhost:11434/api/delete" \
                    -H "Content-Type: application/json" \
                    -d "{\"name\": \"$model\"}" >/dev/null 2>&1
            fi
        done
    fi
    
    # Force garbage collection
    curl -s -X POST "http://localhost:11434/api/generate" \
        -H "Content-Type: application/json" \
        -d '{"model": "", "prompt": "", "stream": false}' >/dev/null 2>&1
}

restart_ollama_container() {
    log_message "Emergency: Restarting Ollama container"
    docker restart "$OLLAMA_CONTAINER" >/dev/null 2>&1
    sleep 10
    log_message "Ollama container restarted"
}

force_memory_cleanup() {
    log_message "Performing emergency memory cleanup"
    
    # Clear system caches
    sync
    echo 3 > /proc/sys/vm/drop_caches
    
    # Unload models
    unload_all_ollama_models
    
    # Clear Python caches
    find /tmp -name "*.pyc" -delete 2>/dev/null || true
    find /tmp -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Clear Docker build cache
    docker system prune -f >/dev/null 2>&1 || true
    
    log_message "Emergency cleanup completed"
}

check_oom_conditions() {
    local mem_usage
    local swap_usage
    local container_mem
    
    mem_usage=$(get_memory_usage)
    swap_usage=$(get_swap_usage)
    container_mem=$(get_container_memory)
    
    log_message "Memory: ${mem_usage}%, Swap: ${swap_usage}%, Container: ${container_mem}%"
    
    # Critical condition: High memory + high swap
    if [ $(echo "$mem_usage > 90" | bc) -eq 1 ] && [ $(echo "$swap_usage > 70" | bc) -eq 1 ]; then
        log_message "CRITICAL: Memory ${mem_usage}% + Swap ${swap_usage}% - Emergency restart"
        restart_ollama_container
        force_memory_cleanup
        return
    fi
    
    # High memory usage
    if [ $(echo "$mem_usage > $MEMORY_THRESHOLD" | bc) -eq 1 ]; then
        log_message "WARNING: High memory usage ${mem_usage}% - Unloading models"
        unload_all_ollama_models
        return
    fi
    
    # High container memory
    if [ -n "$container_mem" ] && [ "$container_mem" != "%" ] && [ $(echo "$container_mem > 75" | bc 2>/dev/null || echo 0) -eq 1 ]; then
        log_message "WARNING: High container memory ${container_mem}% - Cleanup"
        unload_all_ollama_models
        return
    fi
    
    # High swap usage
    if [ $(echo "$swap_usage > $SWAP_THRESHOLD" | bc) -eq 1 ]; then
        log_message "WARNING: High swap usage ${swap_usage}% - Memory cleanup"
        force_memory_cleanup
        return
    fi
}

# Install bc if not available
if ! command -v bc >/dev/null 2>&1; then
    apt-get update >/dev/null 2>&1 && apt-get install -y bc >/dev/null 2>&1
fi

log_message "OOM Prevention System Started"
log_message "Memory Threshold: ${MEMORY_THRESHOLD}%, Swap Threshold: ${SWAP_THRESHOLD}%"

# Main monitoring loop
while true; do
    check_oom_conditions
    sleep 30
done