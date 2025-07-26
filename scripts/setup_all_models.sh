#!/bin/bash

# SutazAI Model Setup and Auto-Download Script
# Downloads and configures all required AI models for the system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
MODELS_DIR="/opt/sutazaiapp/data/models"
LOG_FILE="/opt/sutazaiapp/data/logs/model_setup.log"

# Model configurations
declare -A OLLAMA_MODELS=(
    ["llama3.2:1b"]="Fast general-purpose model (1.3GB)"
    ["deepseek-r1:8b"]="Advanced reasoning model (4.1GB)"
    ["qwen3:8b"]="Multilingual capabilities (4.5GB)"
    ["codellama:7b"]="Code generation specialist (3.8GB)"
    ["llama2:7b"]="General AI model (3.8GB)"
    ["deepseek-coder:6.7b"]="Code optimization expert (3.9GB)"
    ["mistral:7b"]="Efficient instruction following (4.1GB)"
    ["dolphin-mistral:7b"]="Uncensored conversation model (4.1GB)"
    ["neural-chat:7b"]="Chat optimized model (4.1GB)"
    ["phind-codellama:34b"]="Advanced code model (19GB)"
)

declare -A HUGGINGFACE_MODELS=(
    ["sentence-transformers/all-MiniLM-L6-v2"]="Lightweight embeddings"
    ["sentence-transformers/all-mpnet-base-v2"]="High-quality embeddings"
    ["microsoft/DialoGPT-medium"]="Conversational AI"
    ["microsoft/codebert-base"]="Code understanding"
    ["google/flan-t5-base"]="Instruction tuned model"
    ["distilbert-base-uncased"]="Efficient BERT variant"
)

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}" | tee -a "$LOG_FILE"
}

# Check system requirements
check_requirements() {
    log "🔍 Checking system requirements for model downloads..."
    
    # Check available disk space (need at least 100GB for all models)
    available_space=$(df /opt/sutazaiapp | awk 'NR==2 {print $4}')
    required_space=104857600  # 100GB in KB
    
    if [ "$available_space" -lt "$required_space" ]; then
        warn "Low disk space. At least 100GB recommended for all models"
        echo "Available: $(($available_space / 1024 / 1024))GB, Recommended: 100GB"
    else
        log "✅ Sufficient disk space available"
    fi
    
    # Check internet connectivity
    if ping -c 1 google.com >/dev/null 2>&1; then
        log "✅ Internet connectivity confirmed"
    else
        error "No internet connectivity - cannot download models"
        exit 1
    fi
    
    # Create models directory
    mkdir -p "$MODELS_DIR"
    log "✅ Models directory ready: $MODELS_DIR"
}

# Install Ollama if not present
install_ollama() {
    log "🤖 Setting up Ollama..."
    
    if ! command -v ollama >/dev/null 2>&1; then
        info "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        log "✅ Ollama installed successfully"
    else
        log "✅ Ollama already installed"
    fi
    
    # Start Ollama service
    if ! pgrep -x "ollama" > /dev/null; then
        info "Starting Ollama service..."
        ollama serve > /dev/null 2>&1 &
        sleep 10
        log "✅ Ollama service started"
    else
        log "✅ Ollama service already running"
    fi
}

# Download Ollama models
download_ollama_models() {
    log "📥 Downloading Ollama models..."
    
    total_models=${#OLLAMA_MODELS[@]}
    current=0
    
    for model in "${!OLLAMA_MODELS[@]}"; do
        current=$((current + 1))
        description="${OLLAMA_MODELS[$model]}"
        
        info "[$current/$total_models] Downloading $model - $description"
        
        if ollama list | grep -q "^$model"; then
            log "✅ Model $model already exists"
        else
            info "Pulling model $model..."
            if timeout 1800 ollama pull "$model"; then  # 30 minute timeout
                log "✅ Model $model downloaded successfully"
            else
                warn "Failed to download model: $model (continuing with others)"
            fi
        fi
    done
    
    log "✅ Ollama model downloads completed"
}

# Setup Python environment for Hugging Face models
setup_python_env() {
    log "🐍 Setting up Python environment for Hugging Face models..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "/opt/sutazaiapp/model_env" ]; then
        python3 -m venv /opt/sutazaiapp/model_env
        log "✅ Virtual environment created"
    fi
    
    # Activate environment and install dependencies
    source /opt/sutazaiapp/model_env/bin/activate
    
    pip install --upgrade pip > /dev/null 2>&1
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu > /dev/null 2>&1
    pip install transformers sentence-transformers datasets accelerate > /dev/null 2>&1
    
    log "✅ Python environment ready"
}

# Download Hugging Face models
download_huggingface_models() {
    log "🤗 Downloading Hugging Face models..."
    
    source /opt/sutazaiapp/model_env/bin/activate
    
    total_models=${#HUGGINGFACE_MODELS[@]}
    current=0
    
    for model in "${!HUGGINGFACE_MODELS[@]}"; do
        current=$((current + 1))
        description="${HUGGINGFACE_MODELS[$model]}"
        
        info "[$current/$total_models] Downloading $model - $description"
        
        # Create download script
        cat > /tmp/download_model.py << EOF
import os
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch

model_name = "$model"
cache_dir = "$MODELS_DIR/huggingface"

try:
    if "sentence-transformers" in model_name:
        model = SentenceTransformer(model_name, cache_folder=cache_dir)
        print(f"✅ Downloaded {model_name}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        print(f"✅ Downloaded {model_name}")
except Exception as e:
    print(f"❌ Failed to download {model_name}: {e}")
EOF
        
        python /tmp/download_model.py
    done
    
    rm -f /tmp/download_model.py
    log "✅ Hugging Face model downloads completed"
}

# Test model functionality
test_models() {
    log "🧪 Testing model functionality..."
    
    # Test Ollama models
    info "Testing Ollama models..."
    for model in "${!OLLAMA_MODELS[@]}"; do
        if ollama list | grep -q "^$model"; then
            info "Testing $model..."
            if echo "Hello, this is a test." | timeout 30 ollama run "$model" --format json > /dev/null 2>&1; then
                log "✅ Model $model is working"
            else
                warn "Model $model may have issues"
            fi
        fi
    done
    
    # Test Hugging Face models
    info "Testing Hugging Face models..."
    source /opt/sutazaiapp/model_env/bin/activate
    
    cat > /tmp/test_models.py << EOF
from sentence_transformers import SentenceTransformer
import os

cache_dir = "$MODELS_DIR/huggingface"

try:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder=cache_dir)
    embeddings = model.encode(["This is a test sentence."])
    print("✅ Sentence transformers working")
except Exception as e:
    print(f"❌ Sentence transformers error: {e}")
EOF
    
    python /tmp/test_models.py
    rm -f /tmp/test_models.py
    
    log "✅ Model testing completed"
}

# Create model management scripts
create_model_scripts() {
    log "📜 Creating model management scripts..."
    
    # Model list script
    cat > /opt/sutazaiapp/scripts/list_models.sh << 'EOF'
#!/bin/bash

echo "🤖 SutazAI Available Models"
echo "=========================="

echo -e "\n📦 Ollama Models:"
ollama list

echo -e "\n🤗 Hugging Face Models:"
if [ -d "/opt/sutazaiapp/data/models/huggingface" ]; then
    find /opt/sutazaiapp/data/models/huggingface -name "config.json" | \
    sed 's|/opt/sutazaiapp/data/models/huggingface/models--||' | \
    sed 's|/snapshots/.*||' | \
    sed 's|--| |g' | sort | uniq
else
    echo "No Hugging Face models found"
fi

echo -e "\n💾 Disk Usage:"
du -sh /opt/sutazaiapp/data/models/* 2>/dev/null | sort -hr
EOF
    
    # Model update script
    cat > /opt/sutazaiapp/scripts/update_models.sh << 'EOF'
#!/bin/bash

echo "🔄 Updating all models..."

# Update Ollama models
echo "Updating Ollama models..."
ollama list | grep -v "NAME" | awk '{print $1}' | while read model; do
    echo "Updating $model..."
    ollama pull "$model"
done

# Update Hugging Face models
echo "Updating Hugging Face models..."
source /opt/sutazaiapp/model_env/bin/activate
python -c "
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import os

# Update commonly used models
models = [
    'sentence-transformers/all-MiniLM-L6-v2',
    'sentence-transformers/all-mpnet-base-v2'
]

for model_name in models:
    try:
        print(f'Updating {model_name}...')
        if 'sentence-transformers' in model_name:
            SentenceTransformer(model_name, cache_folder='/opt/sutazaiapp/data/models/huggingface')
        else:
            AutoTokenizer.from_pretrained(model_name, cache_dir='/opt/sutazaiapp/data/models/huggingface')
            AutoModel.from_pretrained(model_name, cache_dir='/opt/sutazaiapp/data/models/huggingface')
        print(f'✅ Updated {model_name}')
    except Exception as e:
        print(f'❌ Failed to update {model_name}: {e}')
"

echo "✅ Model updates completed"
EOF
    
    # Model removal script
    cat > /opt/sutazaiapp/scripts/remove_model.sh << 'EOF'
#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name>"
    echo "Example: $0 llama2:7b"
    exit 1
fi

MODEL_NAME="$1"

echo "🗑️ Removing model: $MODEL_NAME"

# Check if it's an Ollama model
if ollama list | grep -q "^$MODEL_NAME"; then
    ollama rm "$MODEL_NAME"
    echo "✅ Ollama model $MODEL_NAME removed"
else
    echo "❌ Ollama model $MODEL_NAME not found"
fi

# Check if it's a Hugging Face model
HF_PATH="/opt/sutazaiapp/data/models/huggingface/models--$(echo $MODEL_NAME | sed 's|/|--|g')"
if [ -d "$HF_PATH" ]; then
    rm -rf "$HF_PATH"
    echo "✅ Hugging Face model $MODEL_NAME removed"
else
    echo "❌ Hugging Face model $MODEL_NAME not found"
fi
EOF
    
    # Make scripts executable
    chmod +x /opt/sutazaiapp/scripts/list_models.sh
    chmod +x /opt/sutazaiapp/scripts/update_models.sh
    chmod +x /opt/sutazaiapp/scripts/remove_model.sh
    
    log "✅ Model management scripts created"
}

# Create model auto-loader service
create_autoloader() {
    log "🔄 Creating model auto-loader service..."
    
    cat > /opt/sutazaiapp/scripts/model_autoloader.py << 'EOF'
#!/usr/bin/env python3
"""
SutazAI Model Auto-loader
Monitors model usage and automatically loads/unloads models based on demand
"""

import asyncio
import aiohttp
import json
import time
import psutil
import logging
from typing import Dict, List
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelAutoLoader:
    def __init__(self):
        self.loaded_models = set()
        self.model_usage = {}
        self.memory_threshold = 80  # Unload models if memory > 80%
        self.idle_timeout = 3600    # Unload models idle for 1 hour
        
    async def check_system_resources(self) -> Dict:
        """Check current system resource usage"""
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/opt/sutazaiapp')
        
        return {
            'memory_percent': memory.percent,
            'cpu_percent': cpu,
            'disk_percent': (disk.used / disk.total) * 100,
            'available_memory_gb': memory.available / (1024**3)
        }
    
    async def get_ollama_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:11434/api/tags') as response:
                    if response.status == 200:
                        data = await response.json()
                        return [model['name'] for model in data.get('models', [])]
        except Exception as e:
            logger.error(f"Failed to get Ollama models: {e}")
        return []
    
    async def load_model(self, model_name: str) -> bool:
        """Load a model into memory"""
        try:
            # Test model by sending a simple request
            async with aiohttp.ClientSession() as session:
                payload = {
                    'model': model_name,
                    'prompt': 'test',
                    'stream': False
                }
                async with session.post('http://localhost:11434/api/generate', 
                                      json=payload) as response:
                    if response.status == 200:
                        self.loaded_models.add(model_name)
                        self.model_usage[model_name] = time.time()
                        logger.info(f"✅ Model {model_name} loaded successfully")
                        return True
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
        return False
    
    async def unload_unused_models(self):
        """Unload models that haven't been used recently"""
        resources = await self.check_system_resources()
        current_time = time.time()
        
        # If memory usage is high, unload idle models
        if resources['memory_percent'] > self.memory_threshold:
            logger.info(f"High memory usage ({resources['memory_percent']:.1f}%), unloading idle models")
            
            for model_name in list(self.loaded_models):
                last_used = self.model_usage.get(model_name, 0)
                if current_time - last_used > self.idle_timeout:
                    logger.info(f"Unloading idle model: {model_name}")
                    self.loaded_models.discard(model_name)
                    del self.model_usage[model_name]
    
    async def preload_popular_models(self):
        """Preload commonly used models if resources allow"""
        resources = await self.check_system_resources()
        
        # Only preload if memory usage is low
        if resources['memory_percent'] < 60 and resources['available_memory_gb'] > 4:
            popular_models = ['llama3.2:1b', 'deepseek-coder:6.7b']
            
            for model in popular_models:
                if model not in self.loaded_models:
                    logger.info(f"Preloading popular model: {model}")
                    await self.load_model(model)
                    await asyncio.sleep(30)  # Wait between loads
    
    async def monitor_and_manage(self):
        """Main monitoring loop"""
        logger.info("🤖 Model auto-loader started")
        
        while True:
            try:
                # Check resources and manage models
                await self.unload_unused_models()
                await self.preload_popular_models()
                
                # Log current status
                resources = await self.check_system_resources()
                logger.info(f"📊 System: {resources['memory_percent']:.1f}% memory, "
                          f"{resources['cpu_percent']:.1f}% CPU, "
                          f"{len(self.loaded_models)} models loaded")
                
                # Wait before next check
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)

if __name__ == "__main__":
    autoloader = ModelAutoLoader()
    asyncio.run(autoloader.monitor_and_manage())
EOF
    
    chmod +x /opt/sutazaiapp/scripts/model_autoloader.py
    
    # Create systemd service for auto-loader
    cat > /etc/systemd/system/sutazai-model-autoloader.service << EOF
[Unit]
Description=SutazAI Model Auto-loader
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/sutazaiapp
ExecStart=/opt/sutazaiapp/model_env/bin/python /opt/sutazaiapp/scripts/model_autoloader.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    systemctl enable sutazai-model-autoloader.service
    
    log "✅ Model auto-loader service created"
}

# Generate model summary
generate_model_summary() {
    log "📋 Generating model setup summary..."
    
    cat > /opt/sutazaiapp/MODEL_SETUP_SUMMARY.md << EOF
# SutazAI Model Setup Summary

## 🤖 Ollama Models Configured

$(for model in "${!OLLAMA_MODELS[@]}"; do
    echo "- **$model**: ${OLLAMA_MODELS[$model]}"
done)

## 🤗 Hugging Face Models Configured

$(for model in "${!HUGGINGFACE_MODELS[@]}"; do
    echo "- **$model**: ${HUGGINGFACE_MODELS[$model]}"
done)

## 📁 Model Storage Locations

- **Ollama Models**: \`~/.ollama/models\`
- **Hugging Face Models**: \`/opt/sutazaiapp/data/models/huggingface\`
- **Python Environment**: \`/opt/sutazaiapp/model_env\`

## 🛠️ Management Scripts

- **List Models**: \`/opt/sutazaiapp/scripts/list_models.sh\`
- **Update Models**: \`/opt/sutazaiapp/scripts/update_models.sh\`
- **Remove Model**: \`/opt/sutazaiapp/scripts/remove_model.sh <model_name>\`
- **Auto-loader**: \`systemctl status sutazai-model-autoloader\`

## 🔧 Model Management Commands

\`\`\`bash
# List all available models
/opt/sutazaiapp/scripts/list_models.sh

# Update all models to latest versions
/opt/sutazaiapp/scripts/update_models.sh

# Remove a specific model
/opt/sutazaiapp/scripts/remove_model.sh llama2:7b

# Check model auto-loader status
systemctl status sutazai-model-autoloader

# Start/stop auto-loader
systemctl start sutazai-model-autoloader
systemctl stop sutazai-model-autoloader
\`\`\`

## 🎯 Model Recommendations

### For Code Generation:
- **deepseek-coder:6.7b** - Best for code completion and generation
- **codellama:7b** - Excellent for code understanding and debugging
- **phind-codellama:34b** - Advanced code analysis (requires more memory)

### For General Chat:
- **llama3.2:1b** - Fast responses for general questions
- **mistral:7b** - Good balance of speed and capability
- **neural-chat:7b** - Optimized for conversational AI

### For Specialized Tasks:
- **qwen3:8b** - Multilingual capabilities
- **deepseek-r1:8b** - Advanced reasoning and analysis
- **dolphin-mistral:7b** - Uncensored responses

## 🔄 Auto-Management Features

The model auto-loader service provides:

- **Automatic Model Loading**: Popular models loaded when system resources allow
- **Memory Management**: Models unloaded when memory usage is high
- **Usage Tracking**: Models unloaded after 1 hour of inactivity
- **Resource Monitoring**: Continuous system resource monitoring
- **Smart Preloading**: Preloads frequently used models during low usage

## 📊 System Requirements

| Model Size | RAM Required | Disk Space | Performance |
|------------|-------------|------------|-------------|
| 1B params | 2-4 GB | 1.3 GB | Very Fast |
| 7B params | 8-16 GB | 3.8-4.5 GB | Fast |
| 8B params | 12-20 GB | 4.1-4.5 GB | Good |
| 34B params | 32-64 GB | 19 GB | Slow |

## ✅ Setup Completed

All models have been configured and are ready for use with the SutazAI system.

---
*Generated on $(date) by SutazAI Model Setup System*
EOF
    
    log "✅ Model setup summary generated"
}

# Main function
main() {
    echo -e "${PURPLE}"
    echo "========================================================"
    echo "🤖 SutazAI Complete Model Setup & Auto-Download 🤖"
    echo "========================================================"
    echo -e "${NC}"
    
    log "🚀 Starting complete model setup process..."
    
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Run setup steps
    check_requirements
    install_ollama
    setup_python_env
    download_ollama_models
    download_huggingface_models
    test_models
    create_model_scripts
    create_autoloader
    generate_model_summary
    
    echo -e "${GREEN}"
    echo "========================================================"
    echo "🎉 Model Setup Complete! 🎉"
    echo "========================================================"
    echo -e "${NC}"
    
    echo -e "${CYAN}📋 Summary:${NC}"
    echo -e "${YELLOW}   • ${#OLLAMA_MODELS[@]} Ollama models configured${NC}"
    echo -e "${YELLOW}   • ${#HUGGINGFACE_MODELS[@]} Hugging Face models configured${NC}"
    echo -e "${YELLOW}   • Auto-loader service enabled${NC}"
    echo -e "${YELLOW}   • Management scripts created${NC}"
    echo ""
    echo -e "${CYAN}📖 View summary:${NC}"
    echo -e "${YELLOW}   cat /opt/sutazaiapp/MODEL_SETUP_SUMMARY.md${NC}"
    echo ""
    echo -e "${CYAN}🔧 Manage models:${NC}"
    echo -e "${YELLOW}   /opt/sutazaiapp/scripts/list_models.sh${NC}"
    echo ""
    
    log "🎯 Model setup completed successfully!"
}

# Run main function
main "$@"