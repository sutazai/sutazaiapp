#!/bin/bash
"""
SutazAI Agent Demo Setup Script
==============================

This script sets up the environment for running the SutazAI agent demo.
It checks dependencies, installs requirements, and provides setup guidance.

Usage:
    bash setup_agent_demo.sh [--install-deps]
"""

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🤖 SutazAI Agent Demo Setup${NC}"
echo "=================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check service status
check_service() {
    local service=$1
    local port=$2
    local name=$3
    
    if command_exists nc; then
        if nc -z localhost $port 2>/dev/null; then
            echo -e "${GREEN}✅ $name is running on port $port${NC}"
            return 0
        else
            echo -e "${YELLOW}⚠️  $name is not running on port $port${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠️  Cannot check $name status (netcat not available)${NC}"
        return 1
    fi
}

# Check Python
echo "Checking Python installation..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✅ $PYTHON_VERSION${NC}"
    PYTHON_CMD="python3"
elif command_exists python; then
    PYTHON_VERSION=$(python --version)
    echo -e "${GREEN}✅ $PYTHON_VERSION${NC}"
    PYTHON_CMD="python"
else
    echo -e "${RED}❌ Python not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Check pip
echo "Checking pip installation..."
if command_exists pip3; then
    echo -e "${GREEN}✅ pip3 available${NC}"
    PIP_CMD="pip3"
elif command_exists pip; then
    echo -e "${GREEN}✅ pip available${NC}"
    PIP_CMD="pip"
else
    echo -e "${RED}❌ pip not found. Please install pip${NC}"
    exit 1
fi

# Check Redis
echo "Checking Redis server..."
if check_service "redis" 6379 "Redis"; then
    REDIS_OK=true
else
    REDIS_OK=false
    echo -e "${YELLOW}  To start Redis: sudo systemctl start redis${NC}"
    echo -e "${YELLOW}  Or install: sudo apt-get install redis-server${NC}"
fi

# Check Ollama
echo "Checking Ollama server..."
if check_service "ollama" 11434 "Ollama"; then
    OLLAMA_OK=true
    
    # Check for required models
    echo "Checking Ollama models..."
    if command_exists curl; then
        MODELS=$(curl -s http://localhost:11434/api/tags 2>/dev/null | grep -o '"name":"[^"]*"' | cut -d'"' -f4 || echo "")
        
        if echo "$MODELS" | grep -q "codellama"; then
            echo -e "${GREEN}✅ codellama model available${NC}"
        else
            echo -e "${YELLOW}⚠️  codellama model not found${NC}"
            echo -e "${YELLOW}  Run: ollama pull codellama${NC}"
        fi
        
        if echo "$MODELS" | grep -q "llama2"; then
            echo -e "${GREEN}✅ llama2 model available${NC}"
        else
            echo -e "${YELLOW}⚠️  llama2 model not found${NC}"
            echo -e "${YELLOW}  Run: ollama pull llama2${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Cannot check models (curl not available)${NC}"
    fi
else
    OLLAMA_OK=false
    echo -e "${YELLOW}  To install Ollama: curl -fsSL https://ollama.com/install.sh | sh${NC}"
    echo -e "${YELLOW}  Then run: ollama serve${NC}"
fi

# Install Python dependencies
echo "Checking Python dependencies..."
if [[ "$1" == "--install-deps" ]] || [[ "$1" == "-i" ]]; then
    echo "Installing Python dependencies..."
    $PIP_CMD install -r agent_demo_requirements.txt
    echo -e "${GREEN}✅ Python dependencies installed${NC}"
else
    # Just check if they exist
    if $PYTHON_CMD -c "import redis, aioredis, httpx" 2>/dev/null; then
        echo -e "${GREEN}✅ Core Python dependencies available${NC}"
    else
        echo -e "${YELLOW}⚠️  Some Python dependencies missing${NC}"
        echo -e "${YELLOW}  Run: $PIP_CMD install -r agent_demo_requirements.txt${NC}"
        echo -e "${YELLOW}  Or: bash setup_agent_demo.sh --install-deps${NC}"
    fi
fi

# Summary
echo ""
echo "=================================="
echo -e "${BLUE}Setup Summary:${NC}"

if [[ "$REDIS_OK" == true ]]; then
    echo -e "${GREEN}✅ Redis server${NC}"
else
    echo -e "${RED}❌ Redis server${NC}"
fi

if [[ "$OLLAMA_OK" == true ]]; then
    echo -e "${GREEN}✅ Ollama server${NC}"
else
    echo -e "${RED}❌ Ollama server${NC}"
fi

echo -e "${GREEN}✅ Python environment${NC}"

# Check if we can run the demo
CAN_RUN=true
if [[ "$REDIS_OK" != true ]]; then
    CAN_RUN=false
fi
if [[ "$OLLAMA_OK" != true ]]; then
    CAN_RUN=false
fi

echo ""
if [[ "$CAN_RUN" == true ]]; then
    echo -e "${GREEN}🎉 Ready to run SutazAI Agent Demo!${NC}"
    echo ""
    echo "Available demo modes:"
    echo "  $PYTHON_CMD run_agent_demo.py basic      # Basic demo (quick)"
    echo "  $PYTHON_CMD run_agent_demo.py full       # Complete demo (recommended)"
    echo "  $PYTHON_CMD run_agent_demo.py custom     # Interactive demo"
    echo "  $PYTHON_CMD run_agent_demo.py benchmark  # Performance test"
    echo "  $PYTHON_CMD run_agent_demo.py monitoring # System monitoring"
    echo ""
    echo "For help:"
    echo "  $PYTHON_CMD run_agent_demo.py --help"
    echo ""
    echo "For detailed usage guide:"
    echo "  cat AGENT_DEMO_GUIDE.md"
else
    echo -e "${YELLOW}⚠️  Please fix the issues above before running the demo${NC}"
    echo ""
    echo "Quick fixes:"
    if [[ "$REDIS_OK" != true ]]; then
        echo "  # Start Redis"
        echo "  sudo systemctl start redis"
        echo "  # Or install Redis"
        echo "  sudo apt-get install redis-server"
        echo ""
    fi
    if [[ "$OLLAMA_OK" != true ]]; then
        echo "  # Install Ollama"
        echo "  curl -fsSL https://ollama.com/install.sh | sh"
        echo "  # Start Ollama"
        echo "  ollama serve &"
        echo "  # Pull required models"
        echo "  ollama pull codellama"
        echo "  ollama pull llama2"
        echo ""
    fi
fi

echo "=================================="