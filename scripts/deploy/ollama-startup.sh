#!/bin/bash
# Ollama memory-efficient startup script

set -e

echo "Starting Ollama with memory optimization..."

# Start Ollama service
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready!"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

# Function to unload all models
unload_all_models() {
    echo "Unloading all models to free memory..."
    models=$(ollama list | tail -n +2 | awk '{print $1}' || true)
    for model in $models; do
        echo "Unloading model: $model"
        curl -X DELETE http://localhost:11434/api/delete -d "{\"name\": \"$model\"}" || true
    done
}

# Function to check available memory
check_memory() {
    available_mem=$(free -m | awk 'NR==2{print $7}')
    echo "Available memory: ${available_mem}MB"
    echo $available_mem
}

# Function to load model with memory check
load_model_safe() {
    local model=$1
    local min_memory_mb=${2:-2048}  # Minimum 2GB by default
    
    echo "Checking if model $model can be loaded..."
    
    # Check available memory
    available=$(check_memory)
    if [ $available -lt $min_memory_mb ]; then
        echo "Not enough memory to load $model (need ${min_memory_mb}MB, have ${available}MB)"
        unload_all_models
        available=$(check_memory)
        
        if [ $available -lt $min_memory_mb ]; then
            echo "Still not enough memory after cleanup. Skipping $model"
            return 1
        fi
    fi
    
    echo "Loading model: $model"
    if ollama pull $model; then
        echo "Model $model loaded successfully"
        return 0
    else
        echo "Failed to load model $model"
        return 1
    fi
}

# Pull essential models with memory management
echo "Loading essential models..."

# Try to load models in order of priority
# Using smaller models to save memory
if ! load_model_safe "qwen2.5-coder:1.5b" 1500; then
    echo "Warning: Could not load qwen2.5-coder:1.5b"
fi

if ! load_model_safe "qwen2.5:3b" 1200; then
    echo "Warning: Could not load qwen2.5:3b"
fi

# For code tasks, use smaller code-specific models
if ! load_model_safe "starcoder2:3b" 2500; then
    echo "Warning: Could not load starcoder2:3b"
fi

# Memory cleanup routine
cleanup_routine() {
    while true; do
        sleep 300  # Check every 5 minutes
        
        # Check memory usage
        mem_usage=$(free | grep Mem | awk '{print int($3/$2 * 100)}')
        echo "Memory usage: ${mem_usage}%"
        
        if [ $mem_usage -gt 85 ]; then
            echo "High memory usage detected. Running cleanup..."
            
            # Get list of loaded models
            loaded_models=$(curl -s http://localhost:11434/api/ps | jq -r '.models[]?.name' 2>/dev/null || echo "")
            
            # Unload least recently used models
            if [ ! -z "$loaded_models" ]; then
                for model in $loaded_models; do
                    echo "Unloading model: $model"
                    curl -X POST http://localhost:11434/api/generate \
                        -d "{\"model\": \"$model\", \"keep_alive\": 0}" || true
                done
            fi
        fi
    done
}

# Start memory cleanup routine in background
cleanup_routine &

# Keep the main Ollama process running
wait $OLLAMA_PID