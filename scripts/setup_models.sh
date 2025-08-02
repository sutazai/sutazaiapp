#!/bin/bash

###############################################################################
# SutazAI Model Setup Script
# Installs and configures all required AI models
###############################################################################

set -euo pipefail

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Ollama is installed
check_ollama() {
    if ! command -v ollama &> /dev/null; then
        log "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
    else
        log "Ollama is already installed"
    fi
    
    # Ensure Ollama service is running
    if ! pgrep -x "ollama" > /dev/null; then
        log "Starting Ollama service..."
        ollama serve &
        sleep 5
    fi
}

# Pull all required models
pull_models() {
    log "Pulling required AI models..."
    
    # Core models
    models=(
        # DeepSeek models for reasoning
        "qwen2.5:3b"
        
        # Qwen models for general tasks
        "qwen2.5:3b"
        
        # Code generation models
        "qwen2.5-coder:3b"
        
        # General purpose models
        "qwen2.5:3b"
        "llama3.2:3b"
        "qwen2.5:3b"
        
        # Specialized models
        "phi3:mini"
        
        # Embedding models
        "nomic-embed-text"
    )
    
    failed_models=()
    
    for model in "${models[@]}"; do
        log "Pulling model: $model"
        if ollama pull "$model"; then
            log "✅ Successfully pulled: $model"
        else
            warning "❌ Failed to pull: $model"
            failed_models+=("$model")
        fi
    done
    
    # Report results
    log "Model pulling complete!"
    log "Successfully pulled: $((${#models[@]} - ${#failed_models[@]})) models"
    
    if [ ${#failed_models[@]} -gt 0 ]; then
        warning "Failed models: ${failed_models[*]}"
        warning "You can retry pulling these models later"
    fi
}

# Configure model aliases
configure_aliases() {
    log "Configuring model aliases..."
    
    # Create model aliases for easier access
    ollama create coding-assistant -f - <<EOF
FROM qwen2.5-coder:3b
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM You are an expert coding assistant. Provide clean, efficient, and well-documented code.
EOF

    ollama create reasoning-assistant -f - <<EOF
FROM qwen2.5:3b
PARAMETER temperature 0.8
PARAMETER top_p 0.95
SYSTEM You are a reasoning assistant that thinks step by step through complex problems.
EOF

    ollama create general-assistant -f - <<EOF
FROM llama3.2:3b
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM You are a helpful AI assistant.
EOF

    log "Model aliases configured"
}

# Test models
test_models() {
    log "Testing models..."
    
    # Test a simple query
    if ollama run qwen2.5:3b "Hello, are you working?" &> /dev/null; then
        log "✅ Models are responding correctly"
    else
        error "Models are not responding"
    fi
}

# Create model management service
create_model_service() {
    log "Creating model management service..."
    
    cat > /tmp/ollama-models.service << EOF
[Unit]
Description=Ollama Model Management Service
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/ollama serve
Restart=always
RestartSec=3
Environment="OLLAMA_HOST=0.0.0.0"
Environment="OLLAMA_ORIGINS=*"
Environment="OLLAMA_MODELS=/usr/share/ollama/.ollama/models"

[Install]
WantedBy=multi-user.target
EOF

    sudo mv /tmp/ollama-models.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable ollama-models.service
    sudo systemctl start ollama-models.service
    
    log "Model service created and started"
}

# Display model information
display_info() {
    echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}🤖 SutazAI Model Setup Complete!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "\n${YELLOW}Available Models:${NC}"
    ollama list
    echo -e "\n${YELLOW}Model Aliases:${NC}"
    echo -e "  • coding-assistant - Optimized for code generation"
    echo -e "  • reasoning-assistant - Optimized for complex reasoning"
    echo -e "  • general-assistant - General purpose assistant"
    echo -e "\n${YELLOW}Usage Examples:${NC}"
    echo -e "  • ollama run qwen2.5:3b \"Explain advanced computing\""
    echo -e "  • ollama run qwen2.5-coder:3b \"Write a Python web scraper\""
    echo -e "  • ollama run coding-assistant \"Create a REST API\""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

# Main function
main() {
    log "Starting SutazAI Model Setup..."
    
    check_ollama
    pull_models
    configure_aliases
    test_models
    create_model_service
    display_info
    
    log "Model setup completed successfully!"
}

# Run main function
main "$@"