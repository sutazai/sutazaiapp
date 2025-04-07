#!/bin/bash

# Script to ensure the Ollama server is running.
# This is a basic check; enhance as needed (e.g., systemd integration).

# Check if ollama command exists
if ! command -v ollama &> /dev/null
then
    echo "Error: ollama command not found. Please install Ollama first."
    echo "See: https://ollama.com/"
    exit 1
fi

# Check if Ollama is already running by trying to list models
# Timeout after a few seconds if it hangs
if timeout 5 ollama list &> /dev/null; then
    echo "Ollama server appears to be running."
    # Optional: Check specific required models are available
    # ollama list | grep -q "llama2" || echo "Warning: llama2 model not found."
    # ollama list | grep -q "deepseek-coder:33b" || echo "Warning: deepseek-coder:33b model not found."
    exit 0
else
    echo "Ollama server does not seem to be running. Attempting to start..."
    # Attempt to start Ollama in the background
    # This assumes 'ollama serve' runs it as a daemon or background process.
    # Adjust if 'ollama serve' runs in the foreground.
    ollama serve > /tmp/ollama_serve.log 2>&1 &

    # Wait a few seconds for the server to potentially start
    sleep 5

    # Check again
    if timeout 5 ollama list &> /dev/null; then
        echo "Ollama server started successfully."
        exit 0
    else
        echo "Error: Failed to start Ollama server. Check logs or start manually."
        echo "Log might be at /tmp/ollama_serve.log or system logs."
        exit 1
    fi
fi 