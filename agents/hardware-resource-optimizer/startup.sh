#!/bin/bash

# Agent startup script
echo "Starting hardware-resource-optimizer agent..."

# Ensure Ollama is available
if ! command -v ollama &> /dev/null; then
    echo "ERROR: Ollama not found. Please install Ollama first."
    exit 1
fi

# Start the agent
python app.py