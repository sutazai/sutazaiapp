#!/bin/bash

# Setup script for GPT-OSS model in Ollama
# This script pulls and configures the GPT-OSS model for use in SutazAI

set -e

echo "========================================="
echo "    GPT-OSS Model Setup for SutazAI     "
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Ollama is running
check_ollama() {
    echo -e "${YELLOW}Checking Ollama service...${NC}"
    
    # Check if Ollama container is running
    if docker ps | grep -q "sutazai-ollama"; then
        echo -e "${GREEN}✓ Ollama container is running${NC}"
        OLLAMA_EXEC="docker exec sutazai-ollama"
        OLLAMA_URL="http://localhost:10104"
    else
        # Check if Ollama is installed locally
        if command -v ollama &> /dev/null; then
            echo -e "${GREEN}✓ Ollama is installed locally${NC}"
            OLLAMA_EXEC=""
            OLLAMA_URL="http://localhost:10104"
        else
            echo -e "${RED}✗ Ollama is not running. Please start Ollama first.${NC}"
            echo "Run: docker-compose up -d ollama"
            exit 1
        fi
    fi
    
    # Test Ollama API
    if curl -s "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama API is accessible${NC}"
    else
        echo -e "${RED}✗ Cannot connect to Ollama API at $OLLAMA_URL${NC}"
        exit 1
    fi
}

# Pull GPT-OSS model
pull_gptoss() {
    echo -e "\n${YELLOW}Pulling GPT-OSS model...${NC}"
    echo "This may take a while depending on your internet connection..."
    
    if [ -n "$OLLAMA_EXEC" ]; then
        # Running in Docker
        $OLLAMA_EXEC ollama pull tinyllama
    else
        # Running locally
        ollama pull tinyllama
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ GPT-OSS model pulled successfully${NC}"
    else
        echo -e "${RED}✗ Failed to pull GPT-OSS model${NC}"
        exit 1
    fi
}

# Remove old Mistral model (optional)
remove_tinyllama() {
    echo -e "\n${YELLOW}Checking for Mistral model...${NC}"
    
    if [ -n "$OLLAMA_EXEC" ]; then
        if $OLLAMA_EXEC ollama list | grep -q "tinyllama"; then
            echo -e "${YELLOW}Found Mistral model. Removing to save space...${NC}"
            $OLLAMA_EXEC ollama rm tinyllama
            echo -e "${GREEN}✓ Mistral model removed${NC}"
        else
            echo "No Mistral model found"
        fi
    else
        if ollama list | grep -q "tinyllama"; then
            echo -e "${YELLOW}Found Mistral model. Removing to save space...${NC}"
            ollama rm tinyllama
            echo -e "${GREEN}✓ Mistral model removed${NC}"
        else
            echo "No Mistral model found"
        fi
    fi
}

# Verify model installation
verify_installation() {
    echo -e "\n${YELLOW}Verifying GPT-OSS installation...${NC}"
    
    if [ -n "$OLLAMA_EXEC" ]; then
        MODEL_LIST=$($OLLAMA_EXEC ollama list)
    else
        MODEL_LIST=$(ollama list)
    fi
    
    if echo "$MODEL_LIST" | grep -q "tinyllama"; then
        echo -e "${GREEN}✓ GPT-OSS model is installed and ready${NC}"
        echo -e "\nInstalled models:"
        echo "$MODEL_LIST"
    else
        echo -e "${RED}✗ GPT-OSS model not found in installed models${NC}"
        exit 1
    fi
}

# Test the model
test_model() {
    echo -e "\n${YELLOW}Testing GPT-OSS model...${NC}"
    
    # Test with a simple prompt
    TEST_RESPONSE=$(curl -s -X POST "$OLLAMA_URL/api/generate" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "tinyllama",
            "prompt": "Hello, please respond with OK if you are working.",
            "stream": false,
            "options": {
                "temperature": 0.1,
                "num_predict": 10
            }
        }' | jq -r '.response' 2>/dev/null || echo "")
    
    if [ -n "$TEST_RESPONSE" ]; then
        echo -e "${GREEN}✓ GPT-OSS model is responding${NC}"
        echo "Test response: $TEST_RESPONSE"
    else
        echo -e "${RED}✗ GPT-OSS model test failed${NC}"
        echo "Please check Ollama logs for errors"
        exit 1
    fi
}

# Update environment variables
update_env() {
    echo -e "\n${YELLOW}Updating environment configuration...${NC}"
    
    ENV_FILE="/opt/sutazaiapp/.env"
    
    # Backup existing .env if it exists
    if [ -f "$ENV_FILE" ]; then
        cp "$ENV_FILE" "${ENV_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
        echo "Backed up existing .env file"
    fi
    
    # Update or add model configuration
    if grep -q "^DEFAULT_MODEL=" "$ENV_FILE" 2>/dev/null; then
        sed -i 's/^DEFAULT_MODEL=.*/DEFAULT_MODEL=tinyllama/' "$ENV_FILE"
    else
        echo "DEFAULT_MODEL=tinyllama" >> "$ENV_FILE"
    fi
    
    if grep -q "^OLLAMA_MODEL=" "$ENV_FILE" 2>/dev/null; then
        sed -i 's/^OLLAMA_MODEL=.*/OLLAMA_MODEL=tinyllama/' "$ENV_FILE"
    else
        echo "OLLAMA_MODEL=tinyllama" >> "$ENV_FILE"
    fi
    
    echo -e "${GREEN}✓ Environment configuration updated${NC}"
}

# Main execution
main() {
    echo "Starting GPT-OSS model setup..."
    echo "================================"
    
    check_ollama
    pull_gptoss
    remove_tinyllama  # Optional: remove old model to save space
    verify_installation
    test_model
    update_env
    
    echo -e "\n${GREEN}=========================================${NC}"
    echo -e "${GREEN}   GPT-OSS Setup Complete!${NC}"
    echo -e "${GREEN}=========================================${NC}"
    echo ""
    echo "GPT-OSS model is now the default model for SutazAI."
    echo ""
    echo "To use GPT-OSS directly with Ollama:"
    echo "  ollama run tinyllama"
    echo ""
    echo "To restart SutazAI with the new model:"
    echo "  docker-compose restart backend frontend"
    echo ""
}

# Run main function
main