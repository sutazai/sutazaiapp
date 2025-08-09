#!/bin/bash
"""
Startup script for optimized SutazAI backend with performance enhancements
"""

echo "🚀 Starting SutazAI High-Performance Backend"
echo "================================================"

# Install performance dependencies
echo "📦 Installing performance dependencies..."
pip install -q uvloop httpx asyncpg "redis[hiredis]" psutil orjson

# Set environment variables for optimization
export PYTHONUNBUFFERED=1
export UVLOOP_USE_LIBUV=1

# Start the optimized backend
echo "⚡ Starting backend with performance optimizations..."
echo "   - Connection pooling enabled"
echo "   - Redis caching active"
echo "   - Async Ollama service"
echo "   - Background task queue"
echo "   - Target: 1000+ concurrent users"
echo ""

# Run with uvicorn and multiple workers
cd /opt/sutazaiapp/backend
python -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --loop uvloop \
    --log-level info \
    --access-log