#!/bin/bash

# SutazAI Multi-Agent Orchestration Demo Launcher
# ===============================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        SutazAI Multi-Agent Orchestration Demo                ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Function to check service
check_service() {
    local name=$1
    local port=$2
    if nc -z localhost $port 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $name is running on port $port"
        return 0
    else
        echo -e "${RED}✗${NC} $name is not running on port $port"
        return 1
    fi
}

# Check prerequisites
echo ""
echo "🔍 Checking prerequisites..."
echo "=============================="

# Check Redis
if ! check_service "Redis" 6379; then
    echo "Starting Redis..."
    docker run -d --name sutazai-redis -p 6379:6379 redis:alpine 2>/dev/null || true
    sleep 2
fi

# Check Ollama
if ! check_service "Ollama" 11434; then
    echo -e "${YELLOW}⚠️${NC} Ollama not running. Please start Ollama service"
    exit 1
fi

# Check Python packages
echo ""
echo "📦 Checking Python packages..."
python3 -c "import redis" 2>/dev/null || pip install redis
python3 -c "import aiohttp" 2>/dev/null || pip install aiohttp

# Ensure demos directory
mkdir -p "$PROJECT_ROOT/demos"

# Run the demo
echo ""
echo "🚀 Launching Multi-Agent Orchestration Demo"
echo "==========================================="
echo ""
echo "Available options:"
echo "  1. Full orchestration demo (default)"
echo "  2. Real-time monitoring: $0 --monitor"
echo ""

cd "$PROJECT_ROOT"
python3 demos/sutazai_multi_agent_orchestration.py "$@"