#!/bin/bash
# SutazAI Environment Check Script
# Validates environment is properly set up for running SutazAI

# Navigate to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo "SutazAI Environment Check"
echo "========================="
echo "Checking system requirements for SutazAI..."
echo

# Output formatting
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to display status 
display_status() {
    if [ $1 -eq 0 ]; then
        echo -e "[${GREEN}PASS${NC}] $2"
    elif [ $1 -eq 1 ]; then
        echo -e "[${YELLOW}WARN${NC}] $2"
    else
        echo -e "[${RED}FAIL${NC}] $2"
    fi
}

# System Checks
echo "System Checks:"
echo "-------------"

# Check Python version
if command_exists python; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
        display_status 0 "Python $PYTHON_VERSION"
    else
        display_status 2 "Python $PYTHON_VERSION (version 3.11.11 required)"
    fi
else
    display_status 2 "Python not found (version 3.11.11 required)"
fi

# Check pip
if command_exists pip; then
    PIP_VERSION=$(pip --version | awk '{print $2}')
    display_status 0 "pip $PIP_VERSION"
elif [ -x "venv/bin/pip" ]; then
    PIP_VERSION=$(venv/bin/pip --version | awk '{print $2}')
    display_status 0 "pip $PIP_VERSION (from virtual environment)"
else
    display_status 2 "pip not found"
fi

# Check Node.js
if command_exists node; then
    NODE_VERSION=$(node --version | cut -c 2-)
    NODE_MAJOR=$(echo $NODE_VERSION | cut -d. -f1)
    
    if [ "$NODE_MAJOR" -ge 16 ]; then
        display_status 0 "Node.js $NODE_VERSION"
    else
        display_status 2 "Node.js $NODE_VERSION (version 16+ required)"
    fi
else
    display_status 1 "Node.js not found (needed for web UI)"
fi

# Check NPM
if command_exists npm; then
    NPM_VERSION=$(npm --version)
    display_status 0 "npm $NPM_VERSION"
else
    display_status 1 "npm not found (needed for web UI)"
fi

# Check CUDA for GPU support
if command_exists nvcc; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c 2-)
    display_status 0 "CUDA $CUDA_VERSION"
else
    display_status 1 "CUDA not found (GPU acceleration won't be available)"
fi

# Check disk space
DISK_SPACE=$(df -h . | awk 'NR==2 {print $4}')
display_status 0 "Available disk space: $DISK_SPACE"

# Project Checks
echo
echo "Project Checks:"
echo "--------------"

# Check if virtual environment exists
if [ -d "venv" ]; then
    display_status 0 "Virtual environment exists"
else
    display_status 2 "Virtual environment not found"
fi

# Check if .env file exists
if [ -f ".env" ]; then
    display_status 0 ".env configuration file exists"
else
    if [ -f ".env.example" ]; then
        display_status 1 ".env file not found (example exists)"
    else
        display_status 2 ".env and .env.example files not found"
    fi
fi

# Directory structure check
echo
echo "Directory Structure:"
echo "-------------------"

REQUIRED_DIRS=("ai_agents/superagi" "backend/services" "docs" "logs" "model_management" "monitoring" "scripts" "web_ui/src/components")

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        display_status 0 "$dir"
    else
        display_status 2 "$dir (missing)"
    fi
done

# Check for model files
echo
echo "Model Checks:"
echo "------------"

if [ -d "model_management/GPT4All" ]; then
    MODEL_FILES=$(find model_management/GPT4All -name "*.bin" | wc -l)
    if [ $MODEL_FILES -gt 0 ]; then
        display_status 0 "GPT4All model files found ($MODEL_FILES file(s))"
    else
        display_status 1 "No GPT4All model files found"
    fi
else
    display_status 1 "GPT4All directory not found"
fi

if [ -d "model_management/DeepSeek-Coder-33B" ]; then
    MODEL_FILES=$(find model_management/DeepSeek-Coder-33B -name "*.bin" -o -name "*.gguf" | wc -l)
    if [ $MODEL_FILES -gt 0 ]; then
        display_status 0 "DeepSeek Coder model files found ($MODEL_FILES file(s))"
    else
        display_status 1 "No DeepSeek Coder model files found"
    fi
else
    display_status 1 "DeepSeek Coder directory not found"
fi

# Dependency check if venv exists
if [ -d "venv" ]; then
    echo
    echo "Dependencies Check:"
    echo "------------------"
    
    # Activate virtual environment and check key dependencies
    source venv/bin/activate
    
    DEPENDENCIES=("fastapi" "uvicorn" "pydantic" "numpy" "torch" "transformers" "langchain")
    
    for dep in "${DEPENDENCIES[@]}"; do
        if python -c "import $dep" 2>/dev/null; then
            DEP_VERSION=$(python -c "import $dep; print($dep.__version__)" 2>/dev/null)
            display_status 0 "$dep $DEP_VERSION"
        else
            display_status 2 "$dep (missing)"
        fi
    done
    
    # Deactivate virtual environment
    deactivate
fi

echo
echo "Environment check complete!"
echo "=========================="
