#!/bin/bash
# Model Management Script for Ollama
# Pulls and manages models for the shared Ollama instance

set -e

OLLAMA_URL="${OLLAMA_URL:-http://localhost:10104}"
REQUIRED_MODELS=(
    "tinyllama"
    "tinyllama2.5-coder:7b"
    "nomic-embed-text"
    "tinyllama-coder:6.7b"
    "tinyllama:mini"
)

echo "=== Sutazai Model Management ==="
echo "Ollama URL: ${OLLAMA_URL}"
echo

# Function to check if Ollama is ready
wait_for_ollama() {
    echo "Waiting for Ollama to be ready..."
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
            echo "Ollama is ready!"
            return 0
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    echo
    echo "ERROR: Ollama not responding after ${max_attempts} attempts"
    return 1
}

# Function to pull a model
pull_model() {
    local model=$1
    echo "Pulling model: ${model}..."
    
    curl -X POST "${OLLAMA_URL}/api/pull" \
         -H "Content-Type: application/json" \
         -d "{\"name\": \"${model}\"}" \
         --no-progress-meter \
         -w "\nHTTP Status: %{http_code}\n"
}

# Function to list models
list_models() {
    echo "Currently available models:"
    curl -s "${OLLAMA_URL}/api/tags" | jq -r '.models[]?.name' 2>/dev/null || echo "No models found"
}

# Function to check model status
check_model() {
    local model=$1
    curl -s "${OLLAMA_URL}/api/tags" | jq -r '.models[]?.name' 2>/dev/null | grep -q "^${model}$"
}

# Function to remove a model
remove_model() {
    local model=$1
    echo "Removing model: ${model}..."
    
    curl -X DELETE "${OLLAMA_URL}/api/delete" \
         -H "Content-Type: application/json" \
         -d "{\"name\": \"${model}\"}" \
         --no-progress-meter \
         -w "\nHTTP Status: %{http_code}\n"
}

# Function to get model info
model_info() {
    local model=$1
    echo "Getting info for model: ${model}..."
    
    curl -s "${OLLAMA_URL}/api/show" \
         -H "Content-Type: application/json" \
         -d "{\"name\": \"${model}\"}" | jq '.' 2>/dev/null || echo "Model not found"
}

# Main execution
case "${1:-help}" in
    init)
        wait_for_ollama || exit 1
        
        echo "Initializing required models..."
        for model in "${REQUIRED_MODELS[@]}"; do
            if check_model "${model}"; then
                echo "✓ Model ${model} already exists"
            else
                echo "→ Pulling ${model}..."
                pull_model "${model}"
            fi
        done
        
        echo
        echo "Model initialization complete!"
        list_models
        ;;
        
    pull)
        if [ -z "$2" ]; then
            echo "Usage: $0 pull <model_name>"
            exit 1
        fi
        wait_for_ollama || exit 1
        pull_model "$2"
        ;;
        
    remove)
        if [ -z "$2" ]; then
            echo "Usage: $0 remove <model_name>"
            exit 1
        fi
        wait_for_ollama || exit 1
        remove_model "$2"
        ;;
        
    list)
        wait_for_ollama || exit 1
        list_models
        ;;
        
    info)
        if [ -z "$2" ]; then
            echo "Usage: $0 info <model_name>"
            exit 1
        fi
        wait_for_ollama || exit 1
        model_info "$2"
        ;;
        
    cleanup)
        wait_for_ollama || exit 1
        echo "Cleaning up unused models..."
        
        # Get all models
        all_models=$(curl -s "${OLLAMA_URL}/api/tags" | jq -r '.models[]?.name' 2>/dev/null)
        
        # Remove models not in required list
        while IFS= read -r model; do
            if [[ ! " ${REQUIRED_MODELS[@]} " =~ " ${model} " ]]; then
                echo "Removing unused model: ${model}"
                remove_model "${model}"
            fi
        done <<< "$all_models"
        
        echo "Cleanup complete!"
        list_models
        ;;
        
    benchmark)
        wait_for_ollama || exit 1
        echo "Benchmarking models..."
        
        for model in "${REQUIRED_MODELS[@]}"; do
            if check_model "${model}"; then
                echo
                echo "Testing ${model}..."
                start_time=$(date +%s.%N)
                
                response=$(curl -s -X POST "${OLLAMA_URL}/api/generate" \
                    -H "Content-Type: application/json" \
                    -d "{
                        \"model\": \"${model}\",
                        \"prompt\": \"Hello, how are you?\",
                        \"stream\": false
                    }")
                
                end_time=$(date +%s.%N)
                duration=$(echo "$end_time - $start_time" | bc)
                
                echo "Response time: ${duration} seconds"
                echo "Response: $(echo "$response" | jq -r '.response' 2>/dev/null | head -n 1)"
            else
                echo "Skipping ${model} (not installed)"
            fi
        done
        ;;
        
    *)
        echo "Usage: $0 {init|pull|remove|list|info|cleanup|benchmark} [model_name]"
        echo
        echo "Commands:"
        echo "  init       - Initialize all required models"
        echo "  pull       - Pull a specific model"
        echo "  remove     - Remove a specific model"
        echo "  list       - List all available models"
        echo "  info       - Get information about a model"
        echo "  cleanup    - Remove unused models"
        echo "  benchmark  - Test model response times"
        echo
        echo "Required models:"
        printf '%s\n' "${REQUIRED_MODELS[@]}"
        exit 1
        ;;
esac