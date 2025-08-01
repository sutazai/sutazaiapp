#!/bin/bash

# Deploy Minimal Viable SutazAI System
# Optimized for low resource usage with essential agents only

set -e

echo "🚀 Deploying Minimal SutazAI System"
echo "=================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install Docker first.${NC}"
    exit 1
fi

# Stop any existing containers
echo -e "${YELLOW}🛑 Stopping existing containers...${NC}"
docker-compose -f docker-compose.yml down 2>/dev/null || true
docker-compose -f docker-compose.tinyllama.yml down 2>/dev/null || true
docker-compose -f docker-compose.minimal.yml down 2>/dev/null || true

# Clean up to save space
echo -e "${YELLOW}🧹 Cleaning up old data...${NC}"
docker system prune -f --volumes

# Create necessary directories
echo -e "${YELLOW}📁 Creating directories...${NC}"
mkdir -p data logs workspace agents/testing-qa-validator agents/senior-ai-engineer

# Copy agent files if not exists
if [ ! -f "agents/testing-qa-validator/agent.py" ]; then
    cp agents/code-generation-improver/agent.py agents/testing-qa-validator/
    cp agents/code-generation-improver/Dockerfile agents/testing-qa-validator/
fi

if [ ! -f "agents/senior-ai-engineer/agent.py" ]; then
    cp agents/code-generation-improver/agent.py agents/senior-ai-engineer/
    cp agents/code-generation-improver/Dockerfile agents/senior-ai-engineer/
fi

# Start minimal system
echo -e "${GREEN}🚀 Starting minimal services...${NC}"
docker-compose -f docker-compose.minimal.yml up -d postgres redis ollama backend

# Wait for services
echo -e "${YELLOW}⏳ Waiting for services to be ready...${NC}"
sleep 20

# Pull TinyLlama model
echo -e "${GREEN}📥 Pulling TinyLlama model...${NC}"
docker exec sutazai-ollama-minimal ollama pull tinyllama:latest || echo "Model may already exist"

# Start agents
echo -e "${GREEN}🤖 Starting essential agents...${NC}"
docker-compose -f docker-compose.minimal.yml up -d code-improver qa-validator ai-engineer

# Show status
echo -e "${GREEN}✅ Deployment complete!${NC}"
echo ""
echo "📊 System Status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "🔗 Access Points:"
echo "  - Backend API: http://localhost:8000"
echo "  - Ollama API: http://localhost:11434"
echo "  - PostgreSQL: localhost:5432"
echo "  - Redis: localhost:6379"

echo ""
echo "💾 Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

echo ""
echo "🚀 Quick Test:"
echo "  curl http://localhost:8000/health"

# Create minimal test script
cat > test_minimal.py << 'EOF'
import requests
import json

# Test backend health
try:
    response = requests.get("http://localhost:8000/health")
    print(f"✅ Backend: {response.json()}")
except:
    print("❌ Backend not responding")

# Test Ollama
try:
    response = requests.get("http://localhost:11434/api/tags")
    models = response.json().get("models", [])
    print(f"✅ Ollama: {len(models)} models loaded")
    for model in models:
        print(f"   - {model['name']}")
except:
    print("❌ Ollama not responding")

# Simple code analysis test
test_code = '''
def calculate_sum(numbers):
    total = 0
    for n in numbers:
        total = total + n
    return total
'''

print("\n🔍 Testing code analysis...")
try:
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "tinyllama",
        "prompt": f"Analyze and improve this Python code:\n{test_code}",
        "stream": False
    })
    print("✅ Analysis complete")
except Exception as e:
    print(f"❌ Analysis failed: {e}")
EOF

echo ""
echo "📝 To test the system:"
echo "  python test_minimal.py"