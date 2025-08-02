#!/bin/bash
# SutazAI Repositories Setup Script
# This script sets up necessary repositories and downloads required model files

# Navigate to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log file
SETUP_LOG="${PROJECT_ROOT}/logs/setup.log"
mkdir -p "$(dirname "$SETUP_LOG")"

# Logging function
log() {
    local message="$1"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo -e "$message"
    echo "[$timestamp] $message" >> "$SETUP_LOG"
}

log "${BLUE}Starting SutazAI repositories setup...${NC}"
log "Setup log: $SETUP_LOG"

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Check for git
if ! command_exists git; then
    log "${RED}Error: git is not installed. Please install git and try again.${NC}"
    exit 1
fi

# Check for curl or wget
if ! command_exists curl && ! command_exists wget; then
    log "${RED}Error: Neither curl nor wget is installed. Please install one of them and try again.${NC}"
    exit 1
fi

# Function to download a file
download_file() {
    local url="$1"
    local output_file="$2"
    
    if command_exists curl; then
        curl -L -o "$output_file" "$url"
    else
        wget -O "$output_file" "$url"
    fi
    
    return $?
}

# Create model directories
mkdir -p "${PROJECT_ROOT}/model_management/GPT4All"
mkdir -p "${PROJECT_ROOT}/model_management/DeepSeek-Coder-33B"

# Set model URLs - these would need to be updated with actual download links
GPT4ALL_MODEL_URL="https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin"
GPT4ALL_MODEL_FILE="${PROJECT_ROOT}/model_management/GPT4All/gpt4all.bin"

DEEPSEEK_MODEL_URL="https://huggingface.co/TheBloke/deepseek-coder-33B-instruct-GGUF/resolve/main/deepseek-coder-33b-instruct.Q4_K_M.gguf"
DEEPSEEK_MODEL_FILE="${PROJECT_ROOT}/model_management/DeepSeek-Coder-33B/deepseek-coder-33b.gguf"

# Clone/update necessary repositories
log "${BLUE}Setting up package repositories...${NC}"

# Setup SuperAGI repositories if needed
if [ ! -d "${PROJECT_ROOT}/packages/superagi" ]; then
    log "Cloning SuperAGI repository..."
    mkdir -p "${PROJECT_ROOT}/packages"
    git clone https://github.com/TransformerOptimus/SuperAGI.git "${PROJECT_ROOT}/packages/superagi" >> "$SETUP_LOG" 2>&1
    
    if [ $? -eq 0 ]; then
        log "${GREEN}SuperAGI repository cloned successfully.${NC}"
    else
        log "${RED}Failed to clone SuperAGI repository.${NC}"
    fi
else
    log "Updating SuperAGI repository..."
    cd "${PROJECT_ROOT}/packages/superagi"
    git pull >> "$SETUP_LOG" 2>&1
    
    if [ $? -eq 0 ]; then
        log "${GREEN}SuperAGI repository updated successfully.${NC}"
    else
        log "${RED}Failed to update SuperAGI repository.${NC}"
    fi
    
    cd "$PROJECT_ROOT"
fi

# Ask if we should download models (these can be large)
log "${BLUE}Model file setup...${NC}"
log "Note: Model files can be large (several GB). Make sure you have enough disk space and bandwidth."

read -p "Download GPT4All model? (y/n): " -n 1 -r DOWNLOAD_GPT4ALL
echo
if [[ $DOWNLOAD_GPT4ALL =~ ^[Yy]$ ]]; then
    if [ -f "$GPT4ALL_MODEL_FILE" ]; then
        log "${YELLOW}GPT4All model file already exists. Overwrite? (y/n): ${NC}"
        read -n 1 -r OVERWRITE_GPT4ALL
        echo
        if [[ ! $OVERWRITE_GPT4ALL =~ ^[Yy]$ ]]; then
            log "Skipping GPT4All model download."
        else
            log "Downloading GPT4All model file (this may take a while)..."
            download_file "$GPT4ALL_MODEL_URL" "$GPT4ALL_MODEL_FILE"
            
            if [ $? -eq 0 ]; then
                log "${GREEN}GPT4All model downloaded successfully.${NC}"
            else
                log "${RED}Failed to download GPT4All model.${NC}"
            fi
        fi
    else
        log "Downloading GPT4All model file (this may take a while)..."
        download_file "$GPT4ALL_MODEL_URL" "$GPT4ALL_MODEL_FILE"
        
        if [ $? -eq 0 ]; then
            log "${GREEN}GPT4All model downloaded successfully.${NC}"
        else
            log "${RED}Failed to download GPT4All model.${NC}"
        fi
    fi
else
    log "Skipping GPT4All model download."
fi

read -p "Download DeepSeek Coder model? (y/n): " -n 1 -r DOWNLOAD_DEEPSEEK
echo
if [[ $DOWNLOAD_DEEPSEEK =~ ^[Yy]$ ]]; then
    if [ -f "$DEEPSEEK_MODEL_FILE" ]; then
        log "${YELLOW}DeepSeek Coder model file already exists. Overwrite? (y/n): ${NC}"
        read -n 1 -r OVERWRITE_DEEPSEEK
        echo
        if [[ ! $OVERWRITE_DEEPSEEK =~ ^[Yy]$ ]]; then
            log "Skipping DeepSeek Coder model download."
        else
            log "Downloading DeepSeek Coder model file (this may take a while)..."
            download_file "$DEEPSEEK_MODEL_URL" "$DEEPSEEK_MODEL_FILE"
            
            if [ $? -eq 0 ]; then
                log "${GREEN}DeepSeek Coder model downloaded successfully.${NC}"
            else
                log "${RED}Failed to download DeepSeek Coder model.${NC}"
            fi
        fi
    else
        log "Downloading DeepSeek Coder model file (this may take a while)..."
        download_file "$DEEPSEEK_MODEL_URL" "$DEEPSEEK_MODEL_FILE"
        
        if [ $? -eq 0 ]; then
            log "${GREEN}DeepSeek Coder model downloaded successfully.${NC}"
        else
            log "${RED}Failed to download DeepSeek Coder model.${NC}"
        fi
    fi
else
    log "Skipping DeepSeek Coder model download."
fi

# Create symlinks or copy necessary files from repositories
log "${BLUE}Setting up symlinks and necessary files...${NC}"

# Create other necessary directories
mkdir -p "${PROJECT_ROOT}/workspace"
mkdir -p "${PROJECT_ROOT}/outputs"
mkdir -p "${PROJECT_ROOT}/storage"

# Create version file
echo "SutazAI - Version 0.1.0" > "${PROJECT_ROOT}/version.txt"
echo "Setup Date: $(date)" >> "${PROJECT_ROOT}/version.txt"

log "${GREEN}SutazAI repositories setup completed!${NC}"
log "You may now proceed with building and running the application."

exit 0
