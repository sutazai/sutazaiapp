#!/bin/bash
set -e

# Ultra-Memory-Optimized Ollama startup script
echo "Starting Ollama with ULTRA memory optimization..."

# Set ultra-conservative memory environment
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KEEP_ALIVE="1m"
export OLLAMA_MAX_QUEUE=1
export OLLAMA_NOHISTORY=1

# Memory thresholds (very conservative)
MEMORY_CRITICAL=80
MEMORY_WARNING=65
SWAP_WARNING=30

# Function to get memory usage percentage
get_memory_usage() {
    free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}'
}

# Function to get swap usage percentage  
get_swap_usage() {
    free | grep Swap | awk '{if($2>0) printf "%.1f", $3/$2 * 100.0; else print "0.0"}'
}

# Function to get container memory usage
get_container_memory() {
    ps -o pid,ppid,cmd,%mem --sort=-%mem | grep ollama | head -1 | awk '{print $4}' || echo "0"
}

# Enhanced memory check with multiple metrics
check_memory() {
    local mem_usage=$(get_memory_usage)
    local swap_usage=$(get_swap_usage)
    local container_mem=$(get_container_memory)
    
    echo "Memory: ${mem_usage}%, Swap: ${swap_usage}%, Process: ${container_mem}%"
    
    # Critical condition - emergency measures
    if (( $(echo "${mem_usage} > ${MEMORY_CRITICAL}" | bc -l) )) || \
       (( $(echo "${swap_usage} > ${SWAP_WARNING}" | bc -l) )) || \
       (( $(echo "${container_mem} > 15.0" | bc -l) )); then
        echo "CRITICAL: Memory emergency - aggressive cleanup!"
        emergency_cleanup
        return 1
    elif (( $(echo "${mem_usage} > ${MEMORY_WARNING}" | bc -l) )); then
        echo "WARNING: High memory usage ${mem_usage}% - preventive cleanup"
        unload_all_models
        return 0
    fi
    
    return 0
}

# Ultra-aggressive emergency cleanup
emergency_cleanup() {
    echo "EMERGENCY MEMORY CLEANUP INITIATED"
    
    # Step 1: Force unload all models immediately
    echo "Step 1: Forcing model unload..."
    unload_all_models
    
    # Step 2: Kill any hung ollama processes
    echo "Step 2: Killing hung processes..."
    pkill -f "ollama" 2>/dev/null || true
    sleep 2
    
    # Step 3: Clear system caches aggressively
    echo "Step 3: Clearing system caches..."
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    
    # Step 4: Force garbage collection via API
    echo "Step 4: Force garbage collection..."
    for i in {1..3}; do
        curl -s -X POST http://localhost:11434/api/generate \
            -H "Content-Type: application/json" \
            -d '{"model": "", "prompt": "", "stream": false, "options": {"num_keep": 0}}' >/dev/null 2>&1 || true
        sleep 1
    done
    
    # Step 5: Clear tmp files
    echo "Step 5: Clearing temporary files..."
    find /tmp -name "*ollama*" -delete 2>/dev/null || true
    find /tmp -name "*.tmp" -mmin +10 -delete 2>/dev/null || true
    
    # Step 6: Restart ollama service if needed
    echo "Step 6: Restarting Ollama if necessary..."
    if ! curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "Ollama not responding - restarting..."
        pkill -f ollama 2>/dev/null || true
        sleep 3
        ollama serve &
        OLLAMA_PID=$!
        sleep 10
    fi
    
    echo "EMERGENCY CLEANUP COMPLETED"
}

# Ultra-enhanced function to unload all models
unload_all_models() {
    echo "Unloading ALL models to free memory..."
    
    # Method 1: Use ps API to find loaded models
    local loaded_models
    loaded_models=$(curl -s http://localhost:11434/api/ps | jq -r '.models[]?.name' 2>/dev/null | head -10)
    
    if [ -n "$loaded_models" ]; then
        echo "Found loaded models via API"
        echo "$loaded_models" | while read -r model; do
            if [ -n "$model" ] && [ "$model" != "null" ]; then
                echo "API unloading: $model"
                # Use keep_alive=0 to immediately unload
                curl -s -X POST http://localhost:11434/api/generate \
                    -H "Content-Type: application/json" \
                    -d "{\"model\": \"$model\", \"prompt\": \"\", \"stream\": false, \"keep_alive\": 0}" >/dev/null 2>&1 || true
                    
                # Also try DELETE method
                curl -s -X DELETE http://localhost:11434/api/delete \
                    -H "Content-Type: application/json" \
                    -d "{\"name\": \"$model\"}" >/dev/null 2>&1 || true
            fi
        done
    fi
    
    # Method 2: Force unload common models by name
    local common_models=("llama3.2:1b" "qwen2.5:3b" "deepseek-r1:8b" "qwen2.5-coder:1.5b" "starcoder2:3b")
    for model in "${common_models[@]}"; do
        echo "Force unloading: $model"
        curl -s -X POST http://localhost:11434/api/generate \
            -H "Content-Type: application/json" \
            -d "{\"model\": \"$model\", \"prompt\": \"\", \"stream\": false, \"keep_alive\": 0}" >/dev/null 2>&1 || true
    done
    
    # Method 3: Nuclear option - clear all model cache
    echo "Clearing model cache..."
    curl -s -X POST http://localhost:11434/api/generate \
        -H "Content-Type: application/json" \
        -d '{"model": "", "prompt": "", "stream": false, "options": {"num_keep": 0, "num_ctx": 0}}' >/dev/null 2>&1 || true
    
    echo "Model unloading completed - waiting for memory to settle..."
    sleep 3
}

# Memory-conscious model loading (only if absolutely necessary)
smart_model_load() {
    local model=$1
    local min_free_gb=${2:-3}
    
    echo "Checking if safe to load model: $model"
    
    local mem_usage=$(get_memory_usage)
    local available_gb=$(free -g | grep Mem | awk '{print $7}')
    
    echo "Available memory: ${available_gb}GB, Required: ${min_free_gb}GB"
    
    if [ "$available_gb" -lt "$min_free_gb" ] || (( $(echo "${mem_usage} > 50.0" | bc -l) )); then
        echo "DENIED: Insufficient memory for $model (Usage: ${mem_usage}%, Available: ${available_gb}GB)"
        return 1
    fi
    
    echo "APPROVED: Loading $model with ${available_gb}GB available"
    return 0
}

# Cleanup function for graceful shutdown
cleanup() {
    echo "Shutting down Ollama gracefully..."
    unload_all_models
    pkill -TERM ollama 2>/dev/null || true
    wait 2>/dev/null || true
}

# Install required tools
if ! command -v bc >/dev/null 2>&1; then
    echo "Installing bc for memory calculations..."
    apt-get update >/dev/null 2>&1 && apt-get install -y bc >/dev/null 2>&1
fi

if ! command -v jq >/dev/null 2>&1; then
    echo "Installing jq for JSON parsing..."
    apt-get update >/dev/null 2>&1 && apt-get install -y jq >/dev/null 2>&1
fi

# Set up signal handlers
trap cleanup SIGTERM SIGINT EXIT

# Initial aggressive memory check
echo "=== INITIAL SYSTEM CHECK ==="
check_memory
echo "==========================="

# Start Ollama server
echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready with timeout and health checks
echo "Waiting for Ollama to be ready..."
sleep 20

# Enhanced readiness check with retries
for i in {1..90}; do
    if curl -f -m 5 http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "Ollama is ready!"
        break
    fi
    
    # Check if process died
    if ! kill -0 $OLLAMA_PID 2>/dev/null; then
        echo "ERROR: Ollama process died during startup"
        exit 1
    fi
    
    echo "Waiting for Ollama... ($i/90)"
    sleep 2
    
    # Memory check during startup
    if [ $((i % 15)) -eq 0 ]; then
        if ! check_memory; then
            echo "Memory critical during startup - aborting"
            exit 1
        fi
    fi
    
    if [ $i -eq 90 ]; then
        echo "ERROR: Ollama failed to start within 3 minutes"
        exit 1
    fi
done

# NEVER preload models - only load on demand
echo "Skipping model preload for maximum memory efficiency"

# Start ultra-aggressive memory monitoring
(
    echo "Starting memory monitoring daemon..."
    while true; do
        if ! check_memory; then
            echo "Memory check failed - emergency measures taken"
        fi
        sleep 15  # Very frequent monitoring
    done
) &

echo "=== OLLAMA STARTUP COMPLETE ==="
echo "Memory-optimized mode: ULTRA-CONSERVATIVE"
echo "Model loading: ON-DEMAND ONLY"
echo "Monitoring: AGGRESSIVE (15s intervals)"
echo "================================="

# Wait for Ollama process
wait $OLLAMA_PID