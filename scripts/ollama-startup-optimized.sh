#!/bin/bash
# Optimized Ollama startup script with aggressive memory management

set -e

echo "🚀 Starting Ollama with memory optimization..."

# Memory management functions
cleanup_memory() {
    echo "🧹 Cleaning up memory..."
    sync
    echo 1 > /proc/sys/vm/drop_caches 2>/dev/null || true
}

# Set memory limits
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE="1m"
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_QUEUE=1
export OLLAMA_FLASH_ATTENTION=false

# Start ollama server in background
echo "🔧 Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for server to be ready
echo "⏳ Waiting for Ollama server to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:11434/api/version >/dev/null 2>&1; then
        echo "✅ Ollama server is ready!"
        break
    fi
    sleep 2
done

# Function to check available memory
check_memory() {
    local available_mb=$(free -m | awk '/^Mem:/{print $7}')
    echo $available_mb
}

# Function to unload all models
unload_all_models() {
    echo "🔄 Unloading all models to free memory..."
    local models=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | grep -v "^$" || true)
    for model in $models; do
        echo "  Unloading $model..."
        curl -s -X POST http://localhost:11434/api/generate \
            -d "{\"model\":\"$model\",\"keep_alive\":0}" \
            -H "Content-Type: application/json" >/dev/null 2>&1 || true
    done
    cleanup_memory
}

# Function to load minimal model
load_minimal_model() {
    local available_mb=$(check_memory)
    echo "📊 Available memory: ${available_mb}MB"
    
    if [ $available_mb -gt 2048 ]; then
        echo "🧠 Loading llama3.2:1b (minimal model)..."
        if ! ollama pull llama3.2:1b; then
            echo "⚠️  Failed to pull llama3.2:1b, trying alternative..."
            ollama pull qwen2.5:0.5b 2>/dev/null || echo "No minimal models available"
        fi
    else
        echo "⚠️  Insufficient memory (${available_mb}MB), skipping model loading"
    fi
}

# Background monitoring
monitor_memory() {
    while true; do
        sleep 60
        local mem_usage=$(free | grep Mem | awk '{print int($3/$2 * 100.0)}')
        local available_mb=$(check_memory)
        
        echo "📊 Memory usage: ${mem_usage}%, Available: ${available_mb}MB"
        
        if [ $mem_usage -gt 85 ] || [ $available_mb -lt 512 ]; then
            echo "🚨 High memory usage detected, unloading models..."
            unload_all_models
        fi
    done
}

# Initial cleanup
cleanup_memory

# Load initial model if memory allows
load_minimal_model

# Start background monitoring
monitor_memory &
MONITOR_PID=$!

# Trap signals to clean up
trap 'echo "🛑 Shutting down..."; kill $MONITOR_PID 2>/dev/null || true; unload_all_models; kill $OLLAMA_PID 2>/dev/null || true; exit 0' SIGTERM SIGINT

# Keep the container running
echo "🎯 Ollama is running with memory optimization"
echo "📝 Logs will show memory monitoring every 60 seconds"

# Wait for ollama process
wait $OLLAMA_PID