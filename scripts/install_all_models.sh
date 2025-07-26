#!/bin/bash
# Install all required Ollama models

echo "=== Installing All Ollama Models ==="

# Function to pull model with retry
pull_model() {
    local model=$1
    local max_retries=3
    local retry=0
    
    echo "Installing $model..."
    
    while [ $retry -lt $max_retries ]; do
        if ollama pull "$model" 2>&1 | tee /tmp/ollama_pull.log; then
            echo "✓ Successfully installed $model"
            return 0
        else
            retry=$((retry + 1))
            if [ $retry -lt $max_retries ]; then
                echo "⚠ Retry $retry/$max_retries for $model..."
                sleep 5
            fi
        fi
    done
    
    echo "✗ Failed to install $model after $max_retries attempts"
    return 1
}

# Core models for the AGI system
CORE_MODELS=(
    "deepseek-r1:8b"        # Primary reasoning model
    "qwen3:8b"              # Alternative reasoning
    "llama3.2:1b"           # Fast inference model
    "llama2:7b"             # General purpose
    "codellama:7b"          # Code generation
    "mistral:7b"            # General purpose
    "phi-2"                 # Lightweight model
)

# Specialized models
SPECIALIZED_MODELS=(
    "nomic-embed-text"      # Embeddings
    "starcoder:1b"          # Code completion
    "deepseek-coder:6.7b"   # Advanced coding
)

# Optional models (install if space allows)
OPTIONAL_MODELS=(
    "mixtral:8x7b"          # Advanced but large
    "wizard-math:7b"        # Math reasoning
    "neural-chat:7b"        # Conversational
)

# Check Ollama is running
if ! ollama list >/dev/null 2>&1; then
    echo "Error: Ollama is not running. Please start Ollama first."
    exit 1
fi

# Install core models
echo ""
echo "=== Installing Core Models ==="
FAILED_MODELS=()

for model in "${CORE_MODELS[@]}"; do
    if ! pull_model "$model"; then
        FAILED_MODELS+=("$model")
    fi
done

# Install specialized models
echo ""
echo "=== Installing Specialized Models ==="

for model in "${SPECIALIZED_MODELS[@]}"; do
    if ! pull_model "$model"; then
        FAILED_MODELS+=("$model")
    fi
done

# Check available disk space
AVAILABLE_SPACE=$(df -BG /var/lib/docker | tail -1 | awk '{print $4}' | sed 's/G//')

if [ "$AVAILABLE_SPACE" -gt 50 ]; then
    echo ""
    echo "=== Installing Optional Models (${AVAILABLE_SPACE}GB available) ==="
    
    for model in "${OPTIONAL_MODELS[@]}"; do
        if ! pull_model "$model"; then
            FAILED_MODELS+=("$model")
        fi
    done
else
    echo ""
    echo "⚠ Skipping optional models due to limited disk space (${AVAILABLE_SPACE}GB available)"
fi

# List installed models
echo ""
echo "=== Installed Models ==="
ollama list

# Report failures
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo ""
    echo "⚠ Failed to install the following models:"
    for model in "${FAILED_MODELS[@]}"; do
        echo "  - $model"
    done
    echo ""
    echo "You can retry installing them manually with: ollama pull <model-name>"
fi

echo ""
echo "=== Model Installation Complete ==="