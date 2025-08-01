#!/bin/bash
# Verify Small Model Configuration

echo "🔍 Verifying Small Model Configuration"
echo "====================================="

# Check Ollama models
echo "📦 Available Ollama models:"
curl -s http://localhost:11434/api/tags | jq -r '.models[]?.name // "No models available"' 2>/dev/null || echo "Ollama not available"

echo ""
echo "🔄 Currently loaded models:"
curl -s http://localhost:11434/api/ps | jq -r '.models[]?.name // "No models loaded"' 2>/dev/null || echo "Ollama not available"

echo ""
echo "🔧 Hardware optimizer status:"
curl -s http://localhost:8523/ollama-status | jq -r '.small_model_mode // "Unknown"' 2>/dev/null || echo "Hardware optimizer not available"

echo ""
echo "📊 Memory usage:"
free -h | grep "Mem:"

echo ""
echo "✅ Small model verification complete"
