#!/bin/bash
# ULTRA PERFORMANCE FIX for Ollama

# Kill any existing Ollama processes
pkill -9 ollama || true

# Clean up temp files
rm -rf /tmp/ollama* /tmp/runner* 2>/dev/null || true

# Start Ollama with minimal settings
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_KEEP_ALIVE=30s
export OLLAMA_HOST=0.0.0.0
export OLLAMA_ORIGINS='*'
export GOMAXPROCS=2
export GOGC=25

# Start Ollama server
/bin/ollama serve &
OLLAMA_PID=$!

# Wait for server to be ready
sleep 5

# Load tinyllama with minimal config
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama",
    "prompt": "test",
    "stream": false,
    "options": {
      "num_ctx": 512,
      "num_batch": 128,
      "num_thread": 2,
      "num_gpu": 0,
      "main_gpu": 0,
      "low_vram": true,
      "f16_kv": false,
      "vocab_only": false,
      "use_mmap": false,
      "use_mlock": false
    }
  }' &

# Keep the server running
wait $OLLAMA_PID