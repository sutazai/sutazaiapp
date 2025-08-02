#!/bin/bash
# Setup Ollama models with tinyllama as default

set -e

echo "==========================================
Configuring Ollama Models
=========================================="

# Wait for Ollama to be ready
echo "[$(date +%H:%M:%S)] INFO: Waiting for Ollama to be ready..."
max_attempts=30
attempt=0
while ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ $attempt -ge $max_attempts ]; then
        echo "[$(date +%H:%M:%S)] ERROR: Ollama not responding after $max_attempts attempts"
        exit 1
    fi
    echo "[$(date +%H:%M:%S)] INFO: Waiting for Ollama... (attempt $attempt/$max_attempts)"
    sleep 2
done
echo "[$(date +%H:%M:%S)] SUCCESS: Ollama is ready"

# Pull tinyllama model if not already present
echo "[$(date +%H:%M:%S)] INFO: Checking for tinyllama model..."
if ! docker exec sutazai-ollama ollama list 2>/dev/null | grep -q "tinyllama"; then
    echo "[$(date +%H:%M:%S)] INFO: Pulling tinyllama model..."
    docker exec sutazai-ollama ollama pull tinyllama:latest
    echo "[$(date +%H:%M:%S)] SUCCESS: tinyllama model pulled successfully"
else
    echo "[$(date +%H:%M:%S)] INFO: tinyllama model already exists"
fi

# Create alias for default model
echo "[$(date +%H:%M:%S)] INFO: Setting tinyllama as default model..."
docker exec sutazai-ollama ollama cp tinyllama:latest default 2>/dev/null || true

# List available models on standby
echo "[$(date +%H:%M:%S)] INFO: Available models on standby:"
docker exec sutazai-ollama ollama list 2>/dev/null | grep -v "tinyllama" || echo "No other models currently available"

# Create model configuration
cat > /tmp/ollama_modelfile << 'EOF'
FROM tinyllama:latest
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048
SYSTEM You are a helpful AI assistant running locally on the SutazAI system.
EOF

# Load the configured model
echo "[$(date +%H:%M:%S)] INFO: Loading configured tinyllama model..."
docker cp /tmp/ollama_modelfile sutazai-ollama:/tmp/modelfile
docker exec sutazai-ollama ollama create sutazai-tinyllama -f /tmp/modelfile
rm -f /tmp/ollama_modelfile

echo "[$(date +%H:%M:%S)] SUCCESS: Ollama models configured successfully"
echo "[$(date +%H:%M:%S)] INFO: Default model: tinyllama"
echo "[$(date +%H:%M:%S)] INFO: Custom model: sutazai-tinyllama (optimized for SutazAI)"