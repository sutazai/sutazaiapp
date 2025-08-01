#!/bin/bash

# Deploy Essential AI Services
# ============================
# A focused deployment for core AI agent functionality

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          SUTAZAI ESSENTIAL AI DEPLOYMENT                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Function to check if service is running
check_service() {
    local name=$1
    if docker ps | grep -q "$name"; then
        echo -e "${GREEN}✓${NC} $name is running"
        return 0
    else
        echo -e "${RED}✗${NC} $name is not running"
        return 1
    fi
}

# Function to wait for service
wait_for_service() {
    local name=$1
    local port=$2
    local max_attempts=30
    local attempt=0
    
    echo "Waiting for $name to be ready..."
    while [ $attempt -lt $max_attempts ]; do
        if nc -z localhost $port 2>/dev/null; then
            echo -e "${GREEN}✓${NC} $name is ready"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    echo -e "${RED}✗${NC} $name failed to start"
    return 1
}

echo ""
echo "🔍 Checking existing services..."
echo "================================"

# Check core services
check_service "sutazai-agi-brain"
check_service "sutazai-agi-redis"
check_service "sutazai-agi-ollama"

echo ""
echo "🚀 Starting essential AI services..."
echo "===================================="

# Ensure Redis is running
if ! docker ps | grep -q "redis"; then
    echo "Starting Redis..."
    docker run -d --name sutazai-redis \
        -p 6379:6379 \
        --restart always \
        redis:alpine
    wait_for_service "Redis" 6379
fi

# Fix Ollama if unhealthy
if docker ps | grep -q "sutazai-agi-ollama.*unhealthy"; then
    echo "Fixing unhealthy Ollama service..."
    docker restart sutazai-agi-ollama
    sleep 10
fi

# Pull essential models for Ollama
echo ""
echo "📦 Ensuring essential AI models..."
echo "=================================="

ESSENTIAL_MODELS=("qwen2.5:3b" "tinyllama:latest")
for model in "${ESSENTIAL_MODELS[@]}"; do
    echo "Checking model: $model"
    if docker exec sutazai-agi-ollama ollama list | grep -q "$model"; then
        echo -e "${GREEN}✓${NC} $model already available"
    else
        echo "Pulling $model..."
        docker exec sutazai-agi-ollama ollama pull "$model" || true
    fi
done

# Create universal agent configuration
echo ""
echo "📝 Creating agent configuration..."
echo "================================="

mkdir -p "$PROJECT_ROOT/config/agents"
cat > "$PROJECT_ROOT/config/agents/essential_agents.json" <<EOF
{
  "agents": [
    {
      "id": "general-assistant",
      "name": "General AI Assistant",
      "type": "general",
      "model": "tinyllama:latest",
      "capabilities": ["conversation", "analysis", "planning"],
      "system_prompt": "You are a helpful AI assistant capable of handling general tasks."
    },
    {
      "id": "code-helper",
      "name": "Code Helper",
      "type": "code_assistant",
      "model": "qwen2.5:3b",
      "capabilities": ["code_generation", "debugging", "refactoring"],
      "system_prompt": "You are an expert programming assistant."
    },
    {
      "id": "task-planner",
      "name": "Task Planner",
      "type": "orchestrator",
      "model": "tinyllama:latest",
      "capabilities": ["task_planning", "workflow_design", "coordination"],
      "system_prompt": "You are a task planning and coordination specialist."
    }
  ]
}
EOF

echo -e "${GREEN}✓${NC} Agent configuration created"

# Create simple test script
echo ""
echo "🧪 Creating test script..."
echo "========================="

cat > "$PROJECT_ROOT/scripts/test_ai_agents.py" <<'EOF'
#!/usr/bin/env python3
"""Test essential AI agents"""

import asyncio
import aiohttp
import json
import sys

async def test_ollama():
    """Test Ollama connectivity"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:11434/api/tags') as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m['name'] for m in data.get('models', [])]
                    print(f"✅ Ollama is running with models: {models}")
                    return True
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        return False

async def test_redis():
    """Test Redis connectivity"""
    try:
        import aioredis
        redis = await aioredis.create_redis_pool('redis://localhost:6379')
        await redis.ping()
        redis.close()
        await redis.wait_closed()
        print("✅ Redis is running")
        return True
    except Exception as e:
        print(f"❌ Redis test failed: {e}")
        return False

async def test_brain():
    """Test AGI Brain"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8900/health') as resp:
                if resp.status == 200:
                    print("✅ AGI Brain is healthy")
                    return True
    except Exception as e:
        print(f"❌ AGI Brain test failed: {e}")
        return False

async def test_simple_inference():
    """Test simple AI inference"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": "tinyllama:latest",
                "prompt": "Hello, how are you?",
                "stream": False
            }
            async with session.post('http://localhost:11434/api/generate', json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    response = data.get('response', '')
                    print(f"✅ AI Response: {response[:100]}...")
                    return True
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False

async def main():
    print("\n🧪 Testing Essential AI Services")
    print("=" * 40)
    
    tests = [
        test_ollama(),
        test_redis(),
        test_brain(),
        test_simple_inference()
    ]
    
    results = await asyncio.gather(*tests, return_exceptions=True)
    
    success_count = sum(1 for r in results if r is True)
    total_count = len(tests)
    
    print("\n" + "=" * 40)
    print(f"✅ Passed: {success_count}/{total_count} tests")
    
    if success_count == total_count:
        print("\n🎉 All essential AI services are working!")
        return 0
    else:
        print("\n⚠️  Some services need attention")
        return 1

if __name__ == "__main__":
    # Try to install aioredis if not present
    try:
        import aioredis
    except ImportError:
        print("Installing aioredis...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "aioredis==1.3.1"])
    
    sys.exit(asyncio.run(main()))
EOF

chmod +x "$PROJECT_ROOT/scripts/test_ai_agents.py"
echo -e "${GREEN}✓${NC} Test script created"

# Display status
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║            ESSENTIAL AI SERVICES STATUS                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "🎯 Core Services:"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(brain|redis|ollama)" | head -5
echo ""
echo "🤖 Available Models:"
docker exec sutazai-agi-ollama ollama list 2>/dev/null || echo "Unable to list models"
echo ""
echo "📊 Next Steps:"
echo "  1. Test services: python3 scripts/test_ai_agents.py"
echo "  2. Monitor system: python3 scripts/sutazai_monitor.py"
echo "  3. View Brain UI: http://localhost:8900"
echo ""
echo "✅ Essential AI deployment complete!"