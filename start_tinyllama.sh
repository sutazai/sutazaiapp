#!/bin/bash

# SutazAI TinyLlama Startup Script
# Minimal resource usage with TinyLlama model

set -e

echo "🚀 Starting SutazAI with TinyLlama (Minimal Resource Mode)"
echo "=================================================="

# Check system resources
echo "📊 System Resource Check:"
free -h | grep -E "Mem:|Swap:"
echo ""

# Use TinyLlama environment
export $(cat .env.tinyllama | grep -v '^#' | xargs)

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose.tinyllama.yml down 2>/dev/null || true

# Clean up old data if requested
if [ "$1" = "--clean" ]; then
    echo "🧹 Cleaning up old data..."
    docker volume rm sutazaiapp_ollama_data sutazaiapp_postgres_data sutazaiapp_redis_data 2>/dev/null || true
fi

# Start core services
echo "🐳 Starting core services..."
docker-compose -f docker-compose.tinyllama.yml up -d ollama postgres redis

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to be ready..."
sleep 10

# Initialize TinyLlama
echo "🤖 Initializing TinyLlama model..."
docker-compose -f docker-compose.tinyllama.yml run --rm init-tinyllama

# Start LiteLLM proxy
echo "🔗 Starting LiteLLM proxy for OpenAI compatibility..."
docker-compose -f docker-compose.tinyllama.yml up -d litellm

# Start core agents
echo "🧠 Starting core agents..."
docker-compose -f docker-compose.tinyllama.yml up -d task-coordinator infra-manager ollama-specialist

# Show status
echo ""
echo "✅ SutazAI TinyLlama is starting up!"
echo ""
echo "📊 Resource Usage:"
echo "  - Model: TinyLlama (637MB)"
echo "  - RAM Usage: ~2-3GB total"
echo "  - CPU: 50% maximum"
echo ""
echo "🔗 Service URLs:"
echo "  - Ollama API: http://localhost:11434"
echo "  - LiteLLM (OpenAI compatible): http://localhost:4000"
echo "  - PostgreSQL: localhost:5432"
echo "  - Redis: localhost:6379"
echo ""
echo "📝 Quick Test Commands:"
echo "  - Test Ollama: curl http://localhost:11434/api/tags"
echo "  - Test LiteLLM: curl http://localhost:4000/health"
echo "  - Chat with TinyLlama: ollama run tinyllama:latest"
echo ""
echo "🎯 Next Steps:"
echo "  1. Wait ~30 seconds for all services to initialize"
echo "  2. Use 'docker-compose -f docker-compose.tinyllama.yml logs -f' to monitor"
echo "  3. Start using the system with minimal resources!"