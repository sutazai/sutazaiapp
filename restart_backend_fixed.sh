#!/bin/bash
# Restart backend with fixed Ollama support

echo "Stopping all backend processes..."
pkill -f "intelligent_backend" || true
pkill -f "simple_backend_api" || true

sleep 2

echo "Starting fixed backend with Ollama support..."
cd /opt/sutazaiapp
nohup python3 intelligent_backend_fixed_ollama.py > backend_fixed_ollama.log 2>&1 &

sleep 3

# Check if running
if ps aux | grep -v grep | grep intelligent_backend_fixed_ollama > /dev/null; then
    echo "✓ Backend started successfully"
    
    # Test health
    echo "Testing backend health..."
    curl -s http://localhost:8000/health | jq '.'
    
    echo -e "\nTesting chat endpoint..."
    curl -s -X POST http://localhost:8000/api/chat \
        -H "Content-Type: application/json" \
        -d '{"message": "Hello, testing", "model": "llama3.2:1b"}' | jq '.'
else
    echo "✗ Failed to start backend"
    tail -20 backend_fixed_ollama.log
fi