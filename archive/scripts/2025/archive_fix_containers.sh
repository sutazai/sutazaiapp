#!/bin/bash

echo "🔧 Fixing SutazAI Container Issues"
echo "================================="

# Fix Ollama health issue
echo "Installing models in Ollama..."
sudo docker exec sutazai-ollama ollama pull llama3.2:1b
sudo docker exec sutazai-ollama ollama pull tinyllama || sudo docker exec sutazai-ollama ollama pull deepseek-coder:1.3b
sudo docker exec sutazai-ollama ollama pull qwen2.5:3b || sudo docker exec sutazai-ollama ollama pull qwen:0.5b

# Fix Qdrant
echo "Checking Qdrant..."
sudo docker restart sutazai-qdrant

# Check health
echo ""
echo "Checking container health..."
sudo docker ps --format "table {{.Names}}\t{{.Status}}" | grep sutazai

echo ""
echo "✅ Container fixes applied!" 