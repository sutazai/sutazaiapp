#!/bin/bash

# ==============================================================================
# SutazAI System Setup Script
# ==============================================================================
# This script prepares the host environment for the SutazAI system.
# It handles:
#   1. Checking for essential dependencies (curl, docker).
#   2. Installing Ollama if it's not already present.
#   3. Creating necessary directories for data and configuration.
#   4. Pulling required local LLM models via Ollama.
#
# This script should be run once before the first start of the system.
# ==============================================================================

set -e

# --- Configuration ---

# List of Ollama models to pre-pull. Add any new models here.
OLLAMA_MODELS=(
    "deepseek-coder:6.7b"
    "qwen:7b"
    "codellama:7b"
    "llama2:7b"
    "nomic-embed-text"
)

# Directories to be created
DIRECTORIES=(
    "./data/postgres"
    "./data/redis"
    "./data/chroma"
    "./data/qdrant"
    "./data/ollama"
    "./data/neo4j"
    "./data/elasticsearch"
    "./data/workspace"
    "./logs"
    "./secrets"
    "./config/nginx"
    "./config/ssl"
    "./config/vault"
    "./config/monitoring/prometheus"
    "./config/monitoring/grafana/provisioning"
)

# --- Helper Functions ---

function print_header() {
    echo "========================================"
    echo " $1"
    echo "========================================"
}

function command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# --- Main Logic ---

print_header "SutazAI System Setup Initiated"

# 1. Dependency Checks
print_header "1. Checking Dependencies"
if ! command_exists curl; then
    echo "Error: curl is not installed. Please install it and re-run." >&2
    exit 1
fi
if ! command_exists docker; then
    echo "Error: docker is not installed. Please install it and re-run." >&2
    exit 1
fi
echo "All essential dependencies are present."

# 2. Install Ollama
print_header "2. Setting up Ollama"
if ! command_exists ollama; then
    echo "Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "Ollama installed successfully."
else
    echo "Ollama is already installed."
fi

# 3. Create Directories
print_header "3. Creating System Directories"
for dir in "${DIRECTORIES[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "Created directory: $dir"
    else
        echo "Directory already exists: $dir"
    fi
done
echo "All system directories are in place."

# 4. Create Dummy Secret Files (if they don't exist)
print_header "4. Initializing Secret Files"
if [ ! -f "./secrets/postgres_password.txt" ]; then
    echo "sutazaipassword" > ./secrets/postgres_password.txt
    echo "Created dummy postgres_password.txt. CHANGE THIS IN PRODUCTION."
fi
if [ ! -f "./secrets/replication_password.txt" ]; then
    echo "sutazaireplication" > ./secrets/replication_password.txt
    echo "Created dummy replication_password.txt."
fi
if [ ! -f "./secrets/grafana_password.txt" ]; then
    echo "admin" > ./secrets/grafana_password.txt
    echo "Created dummy grafana_password.txt."
fi
if [ ! -f "./secrets/vault_token.txt" ]; then
    echo "sutazai-root-token" > ./secrets/vault_token.txt
    echo "Created dummy vault_token.txt."
fi
echo "Secret files initialized."

# 5. Pull Ollama Models
print_header "5. Pulling Local LLM Models"
echo "This may take a significant amount of time and disk space..."

# Ensure Ollama service is running
if ! pgrep -x "ollama" > /dev/null
then
    echo "Starting Ollama service in the background..."
    ollama serve > /dev/null 2>&1 &
    sleep 5 # Give it a moment to start
fi

for model in "${OLLAMA_MODELS[@]}"; do
    echo "---"
    echo "Checking for model: $model"
    if ollama list | grep -q "$model"; then
        echo "Model $model already exists. Skipping."
    else
        echo "Pulling model: $model..."
        ollama pull "$model"
        echo "Successfully pulled $model."
    fi
done
echo "All required models are present."

# Stop the background Ollama service if we started it
if [ -n "$(pgrep -f 'ollama serve')" ]; then
    echo "Stopping background Ollama service."
    pkill -f 'ollama serve'
fi


print_header "SutazAI System Setup Complete!"
echo "You can now start the system with: ./manage.sh start"
echo ""

exit 0
