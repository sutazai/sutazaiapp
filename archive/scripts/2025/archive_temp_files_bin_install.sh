#!/bin/bash
# SutazAI Installation Script
# This script installs the SutazAI application and its dependencies

APP_ROOT="/opt/sutazaiapp"
LOGS_DIR="$APP_ROOT/logs"
PIDS_DIR="$APP_ROOT/pids"
BACKUP_DIR="$APP_ROOT/backups"
CONFIG_DIR="$APP_ROOT/config"
INSTALL_LOG="$LOGS_DIR/installation.log"

# Function to log messages
log_message() {
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $1" >> "$INSTALL_LOG"
    echo "[$timestamp] $1"
}

# Function to colorize output
print_message() {
    local color_code="\033[0;32m"  # Green
    local reset_code="\033[0m"
    
    if [ "$2" = "error" ]; then
        color_code="\033[0;31m"  # Red
    elif [ "$2" = "warning" ]; then
        color_code="\033[0;33m"  # Yellow
    elif [ "$2" = "info" ]; then
        color_code="\033[0;34m"  # Blue
    elif [ "$2" = "title" ]; then
        color_code="\033[1;35m"  # Bold Purple
    fi
    
    echo -e "${color_code}$1${reset_code}"
    log_message "$1"
}

# Check if running as root
if [ "$(id -u)" -eq 0 ]; then
    print_message "This script should not be run as root. Exiting..." "error"
    exit 1
fi

# Ensure directories exist
mkdir -p "$LOGS_DIR"
mkdir -p "$PIDS_DIR"
mkdir -p "$BACKUP_DIR"
mkdir -p "$CONFIG_DIR"

print_message "===== SutazAI Installation Script =====" "title"
print_message "Installation log: $INSTALL_LOG" "info"

# System requirements check
print_message "\nChecking system requirements..." "info"

# Check OS
os_name=$(grep -E "^NAME=" /etc/os-release | cut -d= -f2 | tr -d '"')
os_version=$(grep -E "^VERSION_ID=" /etc/os-release | cut -d= -f2 | tr -d '"')
print_message "Operating System: $os_name $os_version" "info"

# Check CPU
cpu_cores=$(grep -c ^processor /proc/cpuinfo)
cpu_model=$(grep -m 1 "model name" /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')
print_message "CPU: $cpu_model ($cpu_cores cores)" "info"

# Check RAM
total_mem=$(free -m | awk '/Mem/{print $2}')
print_message "Total RAM: ${total_mem}MB" "info"

if [ $total_mem -lt 4000 ]; then
    print_message "WARNING: Minimum 4GB RAM recommended for SutazAI" "warning"
fi

# Check disk space
disk_free=$(df -h --output=avail $APP_ROOT | awk 'NR==2{print $1}')
print_message "Available disk space: $disk_free" "info"

# Check for required software
print_message "\nChecking for required software..." "info"

missing_software=()

# Function to check if a command exists
check_command() {
    if command -v "$1" &> /dev/null; then
        print_message "✓ $1 is installed" "info"
        return 0
    else
        print_message "✗ $1 is not installed" "warning"
        missing_software+=("$1")
        return 1
    fi
}

check_command python3
python_version=""
if [ $? -eq 0 ]; then
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    print_message "  Python version: $python_version" "info"
    
    # Check if version is at least 3.8
    py_major=$(echo "$python_version" | cut -d. -f1)
    py_minor=$(echo "$python_version" | cut -d. -f2)
    
    if [ "$py_major" -lt 3 ] || [ "$py_major" -eq 3 -a "$py_minor" -lt 8 ]; then
        print_message "  WARNING: Python 3.8 or higher is recommended" "warning"
    fi
fi

check_command pip3
check_command npm
check_command node
if [ $? -eq 0 ]; then
    node_version=$(node --version)
    print_message "  Node.js version: $node_version" "info"
fi

check_command git
check_command docker
docker_status="not running"
if [ $? -eq 0 ]; then
    if docker info &>/dev/null; then
        docker_status="running"
    fi
    print_message "  Docker status: $docker_status" "info"
fi

# Ensure models directory exists
MODELS_DIR="$APP_ROOT/models"
mkdir -p "$MODELS_DIR"
print_message "\nEnsuring models directory exists at $MODELS_DIR" "info"

# --- Start: Add Model Downloads ---

print_message "\nChecking and downloading required LLM models..." "info"

# Define model files and URLs
declare -A models
models=(
    ["llama-3-8b-instruct.gguf"]="https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/meta-llama-3-8b-instruct.Q4_K_M.gguf"
    ["deepseek-coder-33b-instruct.Q4_K_M.gguf"]="https://huggingface.co/TheBloke/deepseek-coder-33B-instruct-GGUF/resolve/main/deepseek-coder-33b-instruct.Q4_K_M.gguf"
)

# Download models if they don't exist
for filename in "${!models[@]}"; do
    target_path="$MODELS_DIR/$filename"
    url="${models[$filename]}"
    
    if [ ! -f "$target_path" ]; then
        print_message "Downloading $filename..." "info"
        # Determine download filename from URL if different
        download_filename=$(basename "$url")
        temp_target_path="$MODELS_DIR/$download_filename"

        # Use wget to download, follow redirects, output to correct path
        if wget --content-disposition -O "$temp_target_path" "$url" >> "$INSTALL_LOG" 2>&1; then
            # Rename if target filename is different from downloaded filename
            if [ "$filename" != "$download_filename" ]; then
                mv "$temp_target_path" "$target_path"
                print_message "Downloaded and renamed to $filename successfully" "info"
            else
                 print_message "Downloaded $filename successfully" "info"
            fi
        else
            print_message "Error downloading $filename from $url" "error"
            # Clean up partial download if exists
            rm -f "$temp_target_path"
        fi
    else
        print_message "$filename already exists. Skipping download." "info"
    fi
done

print_message "Model download check completed." "info"

# --- End: Add Model Downloads ---

# Install missing software
if [ ${#missing_software[@]} -gt 0 ]; then
    print_message "\nInstalling missing software..." "info"
    
    # Detect package manager
    if command -v apt-get &> /dev/null; then
        print_message "Using apt package manager" "info"
        
        print_message "Updating package lists..." "info"
        sudo apt-get update >> "$INSTALL_LOG" 2>&1
        
        for pkg in "${missing_software[@]}"; do
            case "$pkg" in
                python3)
                    print_message "Installing Python 3..." "info"
                    sudo apt-get install -y python3 python3-pip python3-venv >> "$INSTALL_LOG" 2>&1
                    ;;
                pip3)
                    print_message "Installing pip3..." "info"
                    sudo apt-get install -y python3-pip >> "$INSTALL_LOG" 2>&1
                    ;;
                npm|node)
                    print_message "Installing Node.js and npm..." "info"
                    curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash - >> "$INSTALL_LOG" 2>&1
                    sudo apt-get install -y nodejs >> "$INSTALL_LOG" 2>&1
                    ;;
                git)
                    print_message "Installing git..." "info"
                    sudo apt-get install -y git >> "$INSTALL_LOG" 2>&1
                    ;;
                docker)
                    print_message "Installing Docker..." "info"
                    curl -fsSL https://get.docker.com | sudo sh >> "$INSTALL_LOG" 2>&1
                    sudo usermod -aG docker $USER >> "$INSTALL_LOG" 2>&1
                    print_message "NOTE: Log out and back in for Docker permissions to take effect" "warning"
                    ;;
            esac
        done
    elif command -v yum &> /dev/null; then
        print_message "Using yum package manager" "info"
        
        for pkg in "${missing_software[@]}"; do
            case "$pkg" in
                python3)
                    print_message "Installing Python 3..." "info"
                    sudo yum install -y python3 python3-pip >> "$INSTALL_LOG" 2>&1
                    ;;
                pip3)
                    print_message "Installing pip3..." "info"
                    sudo yum install -y python3-pip >> "$INSTALL_LOG" 2>&1
                    ;;
                npm|node)
                    print_message "Installing Node.js and npm..." "info"
                    curl -fsSL https://rpm.nodesource.com/setup_16.x | sudo bash - >> "$INSTALL_LOG" 2>&1
                    sudo yum install -y nodejs >> "$INSTALL_LOG" 2>&1
                    ;;
                git)
                    print_message "Installing git..." "info"
                    sudo yum install -y git >> "$INSTALL_LOG" 2>&1
                    ;;
                docker)
                    print_message "Installing Docker..." "info"
                    sudo yum install -y yum-utils
                    sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo >> "$INSTALL_LOG" 2>&1
                    sudo yum install -y docker-ce docker-ce-cli containerd.io >> "$INSTALL_LOG" 2>&1
                    sudo systemctl start docker >> "$INSTALL_LOG" 2>&1
                    sudo systemctl enable docker >> "$INSTALL_LOG" 2>&1
                    sudo usermod -aG docker $USER >> "$INSTALL_LOG" 2>&1
                    print_message "NOTE: Log out and back in for Docker permissions to take effect" "warning"
                    ;;
            esac
        done
    else
        print_message "Unsupported package manager. Please install the following packages manually:" "error"
        for pkg in "${missing_software[@]}"; do
            echo " - $pkg"
        done
    fi
fi

# Create Python virtual environment
print_message "\nSetting up Python virtual environment..." "info"
if [ ! -d "$APP_ROOT/venv" ]; then
    python3 -m venv "$APP_ROOT/venv" >> "$INSTALL_LOG" 2>&1
    print_message "Virtual environment created at $APP_ROOT/venv" "info"
else
    print_message "Virtual environment already exists" "info"
fi

# Source virtual environment
source "$APP_ROOT/venv/bin/activate"

# Install Python dependencies
print_message "\nInstalling Python dependencies..." "info"
if [ -f "$APP_ROOT/requirements.txt" ]; then
    pip install -U pip
    pip install -v -r "$APP_ROOT/requirements.txt"
    print_message "Python dependencies installed from requirements.txt" "info"
else
    print_message "Installing core dependencies..." "info"
    pip install -v fastapi uvicorn pydantic sqlalchemy pydantic-settings python-dotenv pandas numpy scikit-learn qdrant-client pymongo
    
    # Create requirements.txt
    pip freeze > "$APP_ROOT/requirements.txt"
    print_message "Created requirements.txt with installed packages" "info"
fi

# Install Node.js dependencies for Web UI
if [ -d "$APP_ROOT/webui" ]; then
    print_message "\nInstalling Web UI dependencies..." "info"
    
    # Run npm install without redirection to see output/errors
    (cd "$APP_ROOT/webui" && npm install)
    if [ $? -eq 0 ]; then
        print_message "Web UI dependencies installed successfully" "info"
    else
        print_message "Error installing Web UI dependencies" "error"
    fi
else
    print_message "Web UI directory not found. Skipping npm install." "warning"
fi

# Set up Qdrant vector database
print_message "\nSetting up Vector Database (Qdrant)..." "info"
if command -v docker &> /dev/null && [ "$docker_status" = "running" ]; then
    print_message "Setting up Qdrant using Docker..." "info"
    
    # Check if Qdrant container is already running
    if ! docker ps --filter name=qdrant -q &>/dev/null; then
        # Create persistent storage directory
        mkdir -p "$APP_ROOT/vector_storage"
        
        # Pull and run Qdrant container
        docker pull qdrant/qdrant >> "$INSTALL_LOG" 2>&1
        docker run -d --name qdrant \
            -p 6333:6333 \
            -p 6334:6334 \
            -v "$APP_ROOT/vector_storage:/qdrant/storage" \
            qdrant/qdrant >> "$INSTALL_LOG" 2>&1
        
        if [ $? -eq 0 ]; then
            print_message "Qdrant container started successfully" "info"
        else
            print_message "Error starting Qdrant container" "error"
        fi
    else
        print_message "Qdrant container is already running" "info"
    fi
else
    print_message "Docker not available. Installing Qdrant via pip..." "info"
    pip install qdrant-client >> "$INSTALL_LOG" 2>&1
    
    # Create directory for Qdrant local storage
    mkdir -p "$APP_ROOT/vector_storage"
    print_message "Note: For production use, Docker-based Qdrant installation is recommended" "warning"
fi

# Set up LocalAGI
print_message "\nSetting up LocalAGI..." "info"
if [ ! -d "$APP_ROOT/ai_agents/localagi" ]; then
    mkdir -p "$APP_ROOT/ai_agents"
    
    # Configure git credentials
    git config --global user.email "sutazai01@gmail.com"
    git config --global user.name "sutazaideployment"
    git config --global credential.helper store
    
    # Try to clone LocalAGI with token directly in URL
    print_message "Cloning LocalAGI repository..." "info"
    git clone https://sutazaideployment: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@github.com/louisgv/local-agi.git "$APP_ROOT/ai_agents/localagi" >> "$INSTALL_LOG" 2>&1
    
    if [ $? -eq 0 ]; then
        cd "$APP_ROOT/ai_agents/localagi"
        
        # Install dependencies
        if [ -f "requirements.txt" ]; then
            print_message "Installing LocalAGI dependencies..." "info"
            pip install -r requirements.txt >> "$INSTALL_LOG" 2>&1
        fi
        
        print_message "LocalAGI setup completed successfully" "info"
    else
        print_message "Error cloning LocalAGI repository" "error"
        # Create minimal LocalAGI server
        mkdir -p "$APP_ROOT/ai_agents/localagi"
        cat > "$APP_ROOT/ai_agents/localagi/minimal_server.py" << 'EOF'
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

app = FastAPI(title="Minimal LocalAGI Server")

class AgentRequest(BaseModel):
    prompt: str
    system_prompt: str = "You are a helpful AI assistant."
    max_tokens: int = 1000

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "LocalAGI"}

@app.post("/v1/execute")
async def execute_agent(request: AgentRequest):
    try:
        # Simple echo response for testing
        response = f"LocalAGI response to: {request.prompt}"
        return {
            "response": response,
            "status": "success",
            "tokens": len(response.split())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("LOCALAGI_PORT", 8090))
    uvicorn.run(app, host="0.0.0.0", port=port)
EOF
        print_message "Created minimal LocalAGI server" "info"
    fi
else
    print_message "LocalAGI directory already exists" "info"
fi

# Create configuration files
print_message "\nCreating configuration files..." "info"

# Create .env file if it doesn't exist
if [ ! -f "$APP_ROOT/.env" ]; then
    cat > "$APP_ROOT/.env" << EOF
# SutazAI Environment Configuration

# Application settings
APP_ENV=development
DEBUG=true

# API settings
API_PORT=8000
API_HOST=0.0.0.0

# Web UI settings
WEBUI_PORT=3000
WEBUI_HOST=0.0.0.0

# Vector database settings
VECTOR_DB_HOST=localhost
VECTOR_DB_PORT=6333

# LocalAGI settings
LOCALAGI_HOST=localhost
LOCALAGI_PORT=8090

# Database settings
DATABASE_URL=sqlite:///./database/sutazai.db

# Logging settings
LOG_LEVEL=INFO
EOF
    print_message "Created default .env file" "info"
else
    print_message ".env file already exists" "info"
fi

# Create directory structure
print_message "\nCreating application directory structure..." "info"

directories=(
    "$APP_ROOT/backend"
    "$APP_ROOT/backend/api"
    "$APP_ROOT/backend/models"
    "$APP_ROOT/backend/services"
    "$APP_ROOT/database"
    "$APP_ROOT/logs"
    "$APP_ROOT/pids"
    "$APP_ROOT/webui"
    "$APP_ROOT/webui/public"
    "$APP_ROOT/webui/src"
    "$APP_ROOT/config"
    "$APP_ROOT/bin"
    "$APP_ROOT/data"
    "$APP_ROOT/scripts"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_message "Created directory: $dir" "info"
    fi
done

# Make executable scripts executable
print_message "\nSetting permissions for executable files..." "info"
find "$APP_ROOT/bin" -type f -name "*.sh" -exec chmod +x {} \;
find "$APP_ROOT/scripts" -type f -name "*.sh" -exec chmod +x {} \;
print_message "Made all shell scripts executable" "info"

# Verify the installation
print_message "\nVerifying installation..." "info"

# Summary of installed components
print_message "SutazAI has been installed with the following components:" "info"

components=(
    "Python Virtual Environment:$APP_ROOT/venv"
    "Backend API:$APP_ROOT/backend"
    "Web UI:$APP_ROOT/webui"
    "Vector Database:$(if docker ps --filter name=qdrant -q &>/dev/null; then echo "Docker Container"; else echo "$APP_ROOT/vector_storage"; fi)"
    "LocalAGI:$APP_ROOT/ai_agents/localagi"
    "Configuration:$APP_ROOT/.env"
    "Logs:$APP_ROOT/logs"
)

for component in "${components[@]}"; do
    name="${component%%:*}"
    path="${component#*:}"
    
    if [ -e "$path" ]; then
        print_message "✓ $name installed at $path" "info"
    else
        print_message "✗ $name not found at $path" "warning"
    fi
done

# Installation complete
print_message "\n===== SutazAI Installation Complete =====" "title"
print_message "You can now start SutazAI using the following command:" "info"
print_message "  $APP_ROOT/bin/start_all.sh" "info"
print_message "\nTo check system status:" "info"
print_message "  $APP_ROOT/bin/system_maintenance.sh" "info"

# Deactivate virtual environment
deactivate

print_message "Installation log available at: $INSTALL_LOG" "info" 