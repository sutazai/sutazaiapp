#!/bin/bash
# Script to download required AI models for SutazAI

set -e

echo "========================================="
echo "SutazAI Model Download Script"
echo "========================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Ollama API URL
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

# Required models
MODELS=(
    "qwen2.5:3b"
    "qwen2.5:3b"
    "llama3.2:3b"
    "nomic-embed-text:latest"
)

# Function to check if Ollama is running
check_ollama() {
    echo "Checking Ollama service..."
    if curl -s "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama is running${NC}"
        return 0
    else
        echo -e "${RED}✗ Ollama is not accessible at ${OLLAMA_URL}${NC}"
        return 1
    fi
}

# Function to list current models
list_models() {
    echo -e "\n${YELLOW}Current models:${NC}"
    curl -s "${OLLAMA_URL}/api/tags" | grep -o '"name":"[^"]*"' | sed 's/"name":"\([^"]*\)"/  - \1/' || echo "  No models found"
}

# Function to pull a model
pull_model() {
    local model=$1
    echo -e "\n${YELLOW}Pulling model: ${model}${NC}"
    
    # Use curl to pull the model and show progress
    curl -X POST "${OLLAMA_URL}/api/pull" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"${model}\"}" \
        --no-buffer 2>/dev/null | while IFS= read -r line; do
        if [ -n "$line" ]; then
            status=$(echo "$line" | grep -o '"status":"[^"]*"' | sed 's/"status":"\([^"]*\)"/\1/')
            if [ -n "$status" ]; then
                echo "  $status"
            fi
        fi
    done
    
    echo -e "${GREEN}✓ Model ${model} ready${NC}"
}

# Function to test a model
test_model() {
    local model=$1
    echo -e "\n${YELLOW}Testing model: ${model}${NC}"
    
    response=$(curl -s -X POST "${OLLAMA_URL}/api/generate" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${model}\",
            \"prompt\": \"Hello, please respond with 'Model working'\",
            \"stream\": false
        }" | grep -o '"response":"[^"]*"' | sed 's/"response":"\([^"]*\)"/\1/')
    
    if [ -n "$response" ]; then
        echo -e "${GREEN}✓ Model ${model} is working${NC}"
        echo "  Response: ${response:0:50}..."
    else
        echo -e "${RED}✗ Model ${model} test failed${NC}"
    fi
}

# Main execution
main() {
    # Check if Ollama is running
    if ! check_ollama; then
        echo -e "${RED}Please ensure Ollama is running and accessible${NC}"
        echo "You can start it with: docker compose up -d ollama"
        exit 1
    fi
    
    # List current models
    list_models
    
    # Pull each required model
    echo -e "\n${YELLOW}Downloading required models...${NC}"
    for model in "${MODELS[@]}"; do
        pull_model "$model"
    done
    
    # List models again
    echo -e "\n${GREEN}All models downloaded!${NC}"
    list_models
    
    # Optional: Test each model
    read -p "Do you want to test the models? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for model in "${MODELS[@]}"; do
            test_model "$model"
        done
    fi
    
    echo -e "\n${GREEN}=========================================${NC}"
    echo -e "${GREEN}Model setup complete!${NC}"
    echo -e "${GREEN}=========================================${NC}"
}

# Run main function
main