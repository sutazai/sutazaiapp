#!/bin/bash
# 🚀 SutazAI Complete Dependencies Installation Script
# Automatically installs and configures all required dependencies

set -euo pipefail

# Configuration
PROJECT_ROOT="/opt/sutazaiapp"
LOG_FILE="logs/dependencies_install_$(date +%Y%m%d_%H%M%S).log"
OLLAMA_MODELS_DIR="./data/models"
REPOS_DIR="./repos"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
UNDERLINE='\033[4m'
NC='\033[0m'

# Logging
log_info() {
    echo -e "${BLUE}ℹ️  [$(date +'%H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}✅ [$(date +'%H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}⚠️  [$(date +'%H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}❌ [$(date +'%H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

log_header() {
    echo -e "\n${BOLD}${UNDERLINE}$1${NC}" | tee -a "$LOG_FILE"
    echo "══════════════════════════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
}

# Initialize
init_dependencies() {
    log_header "🔧 Initializing SutazAI Dependencies Installation"
    
    # Create directories
    mkdir -p logs repos data/models data/cache config/models
    mkdir -p "$OLLAMA_MODELS_DIR" "$REPOS_DIR"
    
    # Update system
    log_info "Updating system packages..."
    sudo apt-get update -y
    sudo apt-get upgrade -y
    
    # Install essential packages
    sudo apt-get install -y \
        curl \
        wget \
        git \
        python3 \
        python3-pip \
        python3-venv \
        nodejs \
        npm \
        docker.io \
        docker-compose \
        build-essential \
        cmake \
        pkg-config \
        libssl-dev \
        libffi-dev \
        libsqlite3-dev \
        redis-server \
        postgresql \
        postgresql-contrib \
        nginx \
        htop \
        tree \
        jq \
        unzip
    
    log_success "System packages updated and essential tools installed"
}

# Install Ollama and Models
install_ollama_and_models() {
    log_header "🧠 Installing Ollama and AI Models"
    
    # Install Ollama if not already installed
    if ! command -v ollama &> /dev/null; then
        log_info "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        log_success "Ollama installed successfully"
    else
        log_info "Ollama already installed"
    fi
    
    # Start Ollama service
    log_info "Starting Ollama service..."
    sudo systemctl enable ollama || true
    sudo systemctl start ollama || true
    
    # Wait for Ollama to be ready
    log_info "Waiting for Ollama to be ready..."
    max_attempts=30
    attempt=0
    while ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
        if [ $attempt -ge $max_attempts ]; then
            log_error "Ollama failed to start after ${max_attempts} attempts"
            return 1
        fi
        sleep 5
        ((attempt++))
        log_info "Waiting for Ollama... (attempt $attempt/$max_attempts)"
    done
    
    # Install models based on system capabilities
    local available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7/1024}')
    log_info "Available memory: ${available_memory}GB"
    
    # Define model sets based on memory
    if [ "$available_memory" -ge 32 ]; then
        local models=("deepseek-r1:8b" "qwen2.5:7b" "codellama:13b" "llama3.2:3b" "llama3.2:1b" "nomic-embed-text")
        log_info "High-memory system: Installing full model set"
    elif [ "$available_memory" -ge 16 ]; then
        local models=("deepseek-r1:8b" "qwen2.5:7b" "llama3.2:1b" "nomic-embed-text")
        log_info "Medium-memory system: Installing optimized model set"
    else
        local models=("llama3.2:1b" "nomic-embed-text")
        log_info "Limited-memory system: Installing minimal model set"
    fi
    
    # Download models
    for model in "${models[@]}"; do
        log_info "Downloading model: $model"
        if timeout 1800 ollama pull "$model"; then
            log_success "Model $model downloaded successfully"
        else
            log_warn "Failed to download $model (may be due to network or resources)"
        fi
    done
    
    log_success "Ollama and models installation completed"
}

# Clone and setup AI repositories
clone_ai_repositories() {
    log_header "📦 Cloning AI Repository Ecosystem"
    
    cd "$REPOS_DIR"
    
    # Model Management repositories
    log_info "Cloning Model Management repositories..."
    
    # LiteLLM
    if [ ! -d "litellm" ]; then
        git clone https://github.com/BerriAI/litellm.git
        log_success "LiteLLM repository cloned"
    fi
    
    # ChromaDB
    if [ ! -d "chromadb" ]; then
        git clone https://github.com/chroma-core/chroma.git chromadb
        log_success "ChromaDB repository cloned"
    fi
    
    # FAISS
    if [ ! -d "faiss" ]; then
        git clone https://github.com/facebookresearch/faiss.git
        log_success "FAISS repository cloned"
    fi
    
    # Context Engineering Framework
    if [ ! -d "context-engineering-framework" ]; then
        git clone https://github.com/contexte-ai/context-engineering-framework.git context-engineering-framework || \
        git clone https://github.com/openai/context-engineering.git context-engineering-framework || \
        log_warn "Context engineering framework repository not found, creating placeholder"
    fi
    
    # FSDP
    if [ ! -d "fms-fsdp" ]; then
        git clone https://github.com/foundation-model-stack/foundation-model-stack.git fms-fsdp
        log_success "FSDP repository cloned"
    fi
    
    # AI Agents repositories
    log_info "Cloning AI Agents repositories..."
    
    # AutoGPT
    if [ ! -d "AutoGPT" ]; then
        git clone https://github.com/Significant-Gravitas/AutoGPT.git
        log_success "AutoGPT repository cloned"
    fi
    
    # LocalAGI
    if [ ! -d "LocalAGI" ]; then
        git clone https://github.com/mudler/LocalAGI.git
        log_success "LocalAGI repository cloned"
    fi
    
    # TabbyML
    if [ ! -d "tabby" ]; then
        git clone https://github.com/TabbyML/tabby.git
        log_success "TabbyML repository cloned"
    fi
    
    # Semgrep
    if [ ! -d "semgrep" ]; then
        git clone https://github.com/semgrep/semgrep.git
        log_success "Semgrep repository cloned"
    fi
    
    # LangChain
    if [ ! -d "langchain" ]; then
        git clone https://github.com/langchain-ai/langchain.git
        log_success "LangChain repository cloned"
    fi
    
    # AutoGen (AG2)
    if [ ! -d "autogen" ]; then
        git clone https://github.com/ag2ai/ag2.git autogen
        log_success "AutoGen (AG2) repository cloned"
    fi
    
    # AgentZero
    if [ ! -d "agent-zero" ]; then
        git clone https://github.com/frdel/agent-zero.git
        log_success "AgentZero repository cloned"
    fi
    
    # BigAGI
    if [ ! -d "big-AGI" ]; then
        git clone https://github.com/enricoros/big-AGI.git
        log_success "BigAGI repository cloned"
    fi
    
    # Browser Use
    if [ ! -d "browser-use" ]; then
        git clone https://github.com/browser-use/browser-use.git
        log_success "Browser Use repository cloned"
    fi
    
    # Skyvern
    if [ ! -d "skyvern" ]; then
        git clone https://github.com/Skyvern-AI/skyvern.git
        log_success "Skyvern repository cloned"
    fi
    
    # PyTorch
    if [ ! -d "pytorch" ]; then
        git clone https://github.com/pytorch/pytorch.git
        log_success "PyTorch repository cloned"
    fi
    
    # TensorFlow
    if [ ! -d "tensorflow" ]; then
        git clone https://github.com/tensorflow/tensorflow.git
        log_success "TensorFlow repository cloned"
    fi
    
    # JAX
    if [ ! -d "jax" ]; then
        git clone https://github.com/jax-ml/jax.git
        log_success "JAX repository cloned"
    fi
    
    # LangFlow
    if [ ! -d "langflow" ]; then
        git clone https://github.com/langflow-ai/langflow.git
        log_success "LangFlow repository cloned"
    fi
    
    # Dify
    if [ ! -d "dify" ]; then
        git clone https://github.com/langgenius/dify.git
        log_success "Dify repository cloned"
    fi
    
    # AgentGPT
    if [ ! -d "AgentGPT" ]; then
        git clone https://github.com/reworkd/AgentGPT.git
        log_success "AgentGPT repository cloned"
    fi
    
    # CrewAI
    if [ ! -d "crewAI" ]; then
        git clone https://github.com/crewAIInc/crewAI.git
        log_success "CrewAI repository cloned"
    fi
    
    # PrivateGPT
    if [ ! -d "private-gpt" ]; then
        git clone https://github.com/zylon-ai/private-gpt.git
        log_success "PrivateGPT repository cloned"
    fi
    
    # LlamaIndex
    if [ ! -d "llama_index" ]; then
        git clone https://github.com/run-llama/llama_index.git
        log_success "LlamaIndex repository cloned"
    fi
    
    # FlowiseAI
    if [ ! -d "Flowise" ]; then
        git clone https://github.com/FlowiseAI/Flowise.git
        log_success "FlowiseAI repository cloned"
    fi
    
    # ShellGPT
    if [ ! -d "shell_gpt" ]; then
        git clone https://github.com/TheR1D/shell_gpt.git
        log_success "ShellGPT repository cloned"
    fi
    
    # PentestGPT
    if [ ! -d "PentestGPT" ]; then
        git clone https://github.com/GreyDGL/PentestGPT.git
        log_success "PentestGPT repository cloned"
    fi
    
    # JARVIS implementations
    log_info "Cloning JARVIS repositories..."
    
    # Microsoft JARVIS
    if [ ! -d "JARVIS-microsoft" ]; then
        git clone https://github.com/microsoft/JARVIS.git JARVIS-microsoft
        log_success "Microsoft JARVIS repository cloned"
    fi
    
    # Dipesh JARVIS
    if [ ! -d "Jarvis_AI" ]; then
        git clone https://github.com/Dipeshpal/Jarvis_AI.git
        log_success "Dipesh JARVIS repository cloned"
    fi
    
    # Danilo JARVIS
    if [ ! -d "jarvis-danilo" ]; then
        git clone https://github.com/danilofalcao/jarvis.git jarvis-danilo
        log_success "Danilo JARVIS repository cloned"
    fi
    
    # Additional services
    log_info "Cloning additional service repositories..."
    
    # Documind
    if [ ! -d "documind" ]; then
        git clone https://github.com/DocumindHQ/documind.git
        log_success "Documind repository cloned"
    fi
    
    # FinRobot
    if [ ! -d "FinRobot" ]; then
        git clone https://github.com/AI4Finance-Foundation/FinRobot.git
        log_success "FinRobot repository cloned"
    fi
    
    # GPT Engineer
    if [ ! -d "gpt-engineer" ]; then
        git clone https://github.com/AntonOsika/gpt-engineer.git
        log_success "GPT Engineer repository cloned"
    fi
    
    # OpenDevin
    if [ ! -d "OpenDevin" ]; then
        git clone https://github.com/OpenDevin/OpenDevin.git
        log_success "OpenDevin repository cloned"
    fi
    
    # Aider
    if [ ! -d "aider" ]; then
        git clone https://github.com/Aider-AI/aider.git
        log_success "Aider repository cloned"
    fi
    
    # Streamlit
    if [ ! -d "streamlit" ]; then
        git clone https://github.com/streamlit/streamlit.git
        log_success "Streamlit repository cloned"
    fi
    
    cd "$PROJECT_ROOT"
    log_success "All repositories cloned successfully"
}

# Install Python dependencies
install_python_dependencies() {
    log_header "🐍 Installing Python Dependencies"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv .venv
        log_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install core dependencies
    log_info "Installing core Python packages..."
    pip install --no-cache-dir \
        fastapi \
        uvicorn \
        streamlit \
        pydantic \
        sqlalchemy \
        psycopg2-binary \
        redis \
        aioredis \
        numpy \
        pandas \
        scipy \
        scikit-learn \
        matplotlib \
        seaborn \
        plotly \
        requests \
        aiohttp \
        httpx \
        websockets \
        python-multipart \
        python-dotenv \
        pyjwt \
        passlib \
        bcrypt \
        python-jose \
        transformers \
        torch \
        torchvision \
        torchaudio \
        tensorflow \
        jax \
        jaxlib \
        accelerate \
        deepspeed \
        fairscale \
        datasets \
        tokenizers \
        sentence-transformers \
        langchain \
        langchain-community \
        langchain-openai \
        langsmith \
        chromadb \
        qdrant-client \
        faiss-cpu \
        neo4j \
        openai \
        anthropic \
        cohere \
        together \
        litellm \
        ollama \
        gradio \
        chainlit \
        autogen-agentchat \
        crewai \
        asyncio \
        concurrent.futures \
        multiprocessing \
        threading \
        schedule \
        celery \
        dramatiq \
        prometheus-client \
        structlog \
        rich \
        typer \
        click \
        tqdm \
        jupyter \
        notebook \
        jupyterlab \
        ipywidgets \
        voila
    
    # Install project requirements
    if [ -f "requirements.txt" ]; then
        log_info "Installing project-specific requirements..."
        pip install -r requirements.txt
        log_success "Project requirements installed"
    fi
    
    log_success "Python dependencies installed successfully"
}

# Install Node.js dependencies
install_nodejs_dependencies() {
    log_header "📦 Installing Node.js Dependencies"
    
    # Update npm
    sudo npm install -g npm@latest
    
    # Install global packages
    log_info "Installing global Node.js packages..."
    sudo npm install -g \
        typescript \
        ts-node \
        nodemon \
        pm2 \
        nx \
        nest \
        next \
        nuxt \
        vue \
        angular \
        react \
        express \
        koa \
        fastify \
        websocket \
        socket.io \
        prisma \
        sequelize \
        typeorm \
        mongoose \
        redis \
        graphql \
        apollo \
        webpack \
        vite \
        rollup \
        babel \
        eslint \
        prettier \
        jest \
        mocha \
        cypress \
        playwright
    
    # Install project dependencies if package.json exists
    if [ -f "package.json" ]; then
        log_info "Installing project Node.js dependencies..."
        npm install
        log_success "Project Node.js dependencies installed"
    fi
    
    log_success "Node.js dependencies installed successfully"
}

# Setup database services
setup_databases() {
    log_header "🗃️ Setting Up Database Services"
    
    # PostgreSQL setup
    log_info "Configuring PostgreSQL..."
    sudo systemctl enable postgresql
    sudo systemctl start postgresql
    
    # Create database and user
    sudo -u postgres psql -c "CREATE USER sutazai WITH PASSWORD 'sutazai_password';" || true
    sudo -u postgres psql -c "CREATE DATABASE sutazai OWNER sutazai;" || true
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE sutazai TO sutazai;" || true
    
    # Redis setup
    log_info "Configuring Redis..."
    sudo systemctl enable redis-server
    sudo systemctl start redis-server
    
    # Configure Redis password
    echo "requirepass redis_password" | sudo tee -a /etc/redis/redis.conf
    sudo systemctl restart redis-server
    
    log_success "Database services configured successfully"
}

# Configure Docker environment
configure_docker() {
    log_header "🐳 Configuring Docker Environment"
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    # Enable and start Docker
    sudo systemctl enable docker
    sudo systemctl start docker
    
    # Install Docker Compose if not available
    if ! command -v docker-compose &> /dev/null; then
        log_info "Installing Docker Compose..."
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
        log_success "Docker Compose installed"
    fi
    
    # Optimize Docker for AI workloads
    log_info "Optimizing Docker configuration..."
    
    # Create Docker daemon configuration
    sudo mkdir -p /etc/docker
    sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ],
  "default-runtime": "runc",
  "runtimes": {
    "runc": {
      "path": "runc"
    }
  },
  "experimental": true,
  "features": {
    "buildkit": true
  }
}
EOF
    
    sudo systemctl restart docker
    log_success "Docker environment configured successfully"
}

# Create system optimization configurations
optimize_system() {
    log_header "⚡ Optimizing System for AI Workloads"
    
    # Increase file descriptor limits
    echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
    echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf
    
    # Optimize kernel parameters for AI workloads
    sudo tee /etc/sysctl.d/99-sutazai.conf > /dev/null <<EOF
# SutazAI System Optimizations
vm.max_map_count=262144
vm.swappiness=10
net.core.rmem_max=134217728
net.core.wmem_max=134217728
net.ipv4.tcp_rmem=4096 87380 134217728
net.ipv4.tcp_wmem=4096 65536 134217728
kernel.shmmax=68719476736
kernel.shmall=4294967296
fs.file-max=2097152
EOF
    
    sudo sysctl -p /etc/sysctl.d/99-sutazai.conf
    
    # Create swap if needed and system has enough disk space
    if [ ! -f /swapfile ] && [ "$(df / | awk 'NR==2 {print $4}')" -gt 8388608 ]; then
        log_info "Creating swap file for memory management..."
        sudo fallocate -l 8G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
        log_success "Swap file created"
    fi
    
    log_success "System optimization completed"
}

# Generate configuration files
generate_configurations() {
    log_header "⚙️ Generating Configuration Files"
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        log_info "Creating environment configuration..."
        cat > .env << EOF
# SutazAI System Configuration
SUTAZAI_ENV=production
TZ=UTC
SECRET_KEY=$(openssl rand -hex 32)

# Database Configuration
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=sutazai_password
POSTGRES_DB=sutazai
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Redis Configuration
REDIS_PASSWORD=redis_password
REDIS_HOST=localhost
REDIS_PORT=6379

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_HOST=localhost
OLLAMA_PORT=11434

# AI Models
DEFAULT_MODELS=deepseek-r1:8b,qwen2.5:7b,llama3.2:1b,nomic-embed-text
REASONING_MODEL=deepseek-r1:8b
CONVERSATION_MODEL=llama3.2:1b
EMBEDDING_MODEL=nomic-embed-text

# Feature Flags
ENABLE_GPU_SUPPORT=auto
ENABLE_MONITORING=true
ENABLE_SECURITY=true
ENABLE_AUTO_BACKUP=true
EOF
        log_success "Environment configuration created"
    fi
    
    # Create systemd service files for key components
    log_info "Creating systemd service files..."
    
    # SutazAI main service
    sudo tee /etc/systemd/system/sutazai.service > /dev/null <<EOF
[Unit]
Description=SutazAI AGI/ASI System
After=network.target docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/sutazaiapp
ExecStart=/opt/sutazaiapp/scripts/deploy_complete_system.sh start
ExecStop=/opt/sutazaiapp/scripts/deploy_complete_system.sh stop
User=root
Group=root

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    log_success "Systemd services created"
}

# Validate installation
validate_installation() {
    log_header "✅ Validating Installation"
    
    local validation_failed=0
    
    # Check Ollama
    if command -v ollama &> /dev/null && ollama list > /dev/null 2>&1; then
        log_success "Ollama: ✓ Installed and working"
    else
        log_error "Ollama: ✗ Not working properly"
        ((validation_failed++))
    fi
    
    # Check Docker
    if command -v docker &> /dev/null && docker ps > /dev/null 2>&1; then
        log_success "Docker: ✓ Installed and working"
    else
        log_error "Docker: ✗ Not working properly"
        ((validation_failed++))
    fi
    
    # Check Python environment
    if [ -d ".venv" ] && source .venv/bin/activate && python -c "import fastapi, streamlit" > /dev/null 2>&1; then
        log_success "Python Environment: ✓ Working"
    else
        log_error "Python Environment: ✗ Issues detected"
        ((validation_failed++))
    fi
    
    # Check databases
    if sudo systemctl is-active --quiet postgresql && sudo systemctl is-active --quiet redis-server; then
        log_success "Databases: ✓ Running"
    else
        log_error "Databases: ✗ Not running properly"
        ((validation_failed++))
    fi
    
    # Check repositories
    if [ -d "$REPOS_DIR" ] && [ "$(find $REPOS_DIR -maxdepth 1 -type d | wc -l)" -gt 10 ]; then
        log_success "Repositories: ✓ Cloned successfully"
    else
        log_error "Repositories: ✗ Missing or incomplete"
        ((validation_failed++))
    fi
    
    if [ $validation_failed -eq 0 ]; then
        log_success "🎉 All components validated successfully!"
        return 0
    else
        log_error "❌ $validation_failed components failed validation"
        return 1
    fi
}

# Main execution
main() {
    cd "$PROJECT_ROOT" || exit 1
    
    log_header "🚀 SutazAI Complete Dependencies Installation"
    log_info "Starting comprehensive installation process..."
    
    # Initialize
    init_dependencies
    
    # Install components
    install_ollama_and_models
    clone_ai_repositories
    install_python_dependencies
    install_nodejs_dependencies
    setup_databases
    configure_docker
    optimize_system
    generate_configurations
    
    # Validate
    if validate_installation; then
        log_success "🎉 SutazAI dependencies installation completed successfully!"
        log_info "Next steps:"
        log_info "1. Run: sudo ./scripts/deploy_complete_system.sh deploy"
        log_info "2. Access the system at: http://localhost:8501"
        log_info "3. Monitor health at: http://localhost:3000"
    else
        log_error "Installation completed with some issues. Please review the logs."
        exit 1
    fi
}

# Handle script arguments
case "${1:-install}" in
    "install"|"all")
        main
        ;;
    "ollama")
        install_ollama_and_models
        ;;
    "repos")
        clone_ai_repositories
        ;;
    "python")
        install_python_dependencies
        ;;
    "nodejs")
        install_nodejs_dependencies
        ;;
    "databases")
        setup_databases
        ;;
    "docker")
        configure_docker
        ;;
    "optimize")
        optimize_system
        ;;
    "validate")
        validate_installation
        ;;
    "help")
        echo "Usage: $0 [install|ollama|repos|python|nodejs|databases|docker|optimize|validate|help]"
        echo ""
        echo "Commands:"
        echo "  install   - Complete installation (default)"
        echo "  ollama    - Install Ollama and AI models only"
        echo "  repos     - Clone repositories only"
        echo "  python    - Install Python dependencies only"
        echo "  nodejs    - Install Node.js dependencies only"
        echo "  databases - Setup databases only"
        echo "  docker    - Configure Docker only"
        echo "  optimize  - System optimization only"
        echo "  validate  - Validate installation"
        echo "  help      - Show this help"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac 