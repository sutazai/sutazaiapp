#!/bin/bash
# =============================================================================
# SutazAI System Setup Script
# =============================================================================
# This script automates the installation of Ollama and the required models.
# It is called by the ./manage.sh script.

set -e

# --- Configuration ---
# Add or remove models from this list as needed.
MODELS_TO_INSTALL=(
    "deepseek-coder:33b"
    "llama3"
    "qwen:7b"
    "codellama:7b"
)

# --- Helper Functions ---

function check_ollama() {
    if command -v ollama &> /dev/null; then
        echo "[INFO] Ollama is already installed."
        return 0
    else
        return 1
    fi
}

function install_ollama() {
    echo "[INFO] Ollama not found. Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "[SUCCESS] Ollama installed."
}

function pull_models() {
    echo "[INFO] Pulling required AI models... This may take a while."
    for model in "${MODELS_TO_INSTALL[@]}"; do
        echo "[INFO] Pulling model: $model"
        ollama pull "$model"
    done
    echo "[SUCCESS] All models have been pulled."
}

# --- Main Execution ---

if ! check_ollama; then
    install_ollama
fi

pull_models

echo "
+----------------------------------------------------+
|                                                    |
|   Setup Complete!                                  |
|                                                    |
|   You can now start the SutazAI system by running: |
|                                                    |
|   ./manage.sh start                                |
|                                                    |
+----------------------------------------------------+
"

exit 0