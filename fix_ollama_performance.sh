#!/bin/bash
# ULTRA PERFORMANCE FIX SCRIPT

echo "🔧 OLLAMA ULTRA PERFORMANCE FIX"
echo "================================"

# Step 1: Stop all Ollama-related services
echo "1️⃣ Stopping services..."
docker stop sutazai-ollama sutazai-ollama-integration 2>/dev/null || true
docker rm sutazai-ollama sutazai-ollama-integration 2>/dev/null || true

# Step 2: Clean up volumes
echo "2️⃣ Cleaning up old data..."
docker volume rm sutazaiapp_ollama-data 2>/dev/null || true

# Step 3: Start fresh Ollama with minimal overhead
echo "3️⃣ Starting optimized Ollama..."
docker run -d \
  --name sutazai-ollama \
  --network sutazai-network \
  -p 10104:11434 \
  -e OLLAMA_HOST=0.0.0.0 \
  -e OLLAMA_NUM_PARALLEL=1 \
  -e OLLAMA_KEEP_ALIVE=5m \
  --cpus="4" \
  --memory="4g" \
  --restart unless-stopped \
  ollama/ollama:latest

# Step 4: Wait for Ollama to start
echo "4️⃣ Waiting for Ollama to start..."
sleep 10

# Step 5: Pull a smaller, faster model
echo "5️⃣ Pulling optimized model..."
docker exec sutazai-ollama ollama pull tinyllama:latest

# Step 6: Warm up the model
echo "6️⃣ Warming up model..."
curl -X POST http://localhost:10104/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "tinyllama", "prompt": "test", "stream": false}' \
  --max-time 60 > /dev/null 2>&1 || true

# Step 7: Test performance
echo "7️⃣ Testing performance..."
python3 /opt/sutazaiapp/test_ollama.py

echo ""
echo "✅ Fix complete! Testing final response time..."
time curl -X POST http://localhost:10104/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "tinyllama", "prompt": "Hi", "stream": false}' \
  --max-time 10 2>&1 | grep -E "response|real"

# Step 8: Restart Ollama integration service
echo ""
echo "8️⃣ Restarting integration service..."
docker compose up -d ollama-integration 2>/dev/null || true

echo ""
echo "🎉 OLLAMA PERFORMANCE FIX COMPLETE!"
echo "===================================="
echo "Expected response time: <5 seconds"
echo "If still slow, system may need more CPU resources"