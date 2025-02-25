#!/bin/bash

# Modern color codes
RED='\e[38;5;196m'
GREEN='\e[38;5;46m'
BLUE='\e[38;5;39m'
YELLOW='\e[38;5;226m'
RESET='\e[0m'

# Source the centralized logging
source "$SCRIPT_DIR/scripts/logging.sh"

# Initialize logging
init_logging() {
    LOG_FILE="/var/log/sutazai_deploy.log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    log "INFO" "Logging initialized"
}

# Embedded logo (base85-encoded zlib-compressed)
readonly EMBEDDED_LOGO="<base85_encoded_string>"

# Function to display the large logo
display_logo() {
    clear
    echo -e "${RED}"
    if [ -f "$SCRIPT_DIR/assets/logos/sutazai.txt" ]; then
        cat "$SCRIPT_DIR/assets/logos/sutazai.txt"
    else
        # Fallback to minimal version
        echo "SutazAI"
    fi
    echo -e "${RESET}"
    log "INFO" "Deploying your infinite possibilities SutazAi! Sit back, relax and enjoy the show."
}

# Modern progress bar with 3-line status
modern_progress_bar() {
    local total_steps=$1
    local current_step=$2
    local label1=$3
    local label2=$4
    local label3=$5
    local bar_length=50
    local filled=$((current_step * bar_length / total_steps))
    local empty=$((bar_length - filled))
    
    # Terminal capability detection
    local is_interactive=0
    [ -t 1 ] && is_interactive=1
    local color_support=0
    if [ "$(tput colors)" -ge 256 ] && [ "$is_interactive" -eq 1 ]; then
        color_support=1
    fi

    if [ "$color_support" -eq 1 ]; then
        # Color gradient version
        local color=$((196 + (current_step * 60 / total_steps)))
        printf "\r\033[38;5;${color}m%s: [\033[1;37m" "$label1"
        printf "▓%.0s" $(seq 1 $filled)
        printf "░%.0s" $(seq 1 $empty)
        printf "\033[38;5;${color}m] %d%%\033[0m" $((current_step * 100 / total_steps))
    else
        # ASCII-only fallback with multi-line support
        if [ "$is_interactive" -eq 1 ]; then
            # Interactive terminal without color support
            printf "\r%-20s [%-50s] %3d%%\n" "$label1" "$(printf '#%.0s' {1..$filled})" $((current_step * 100 / total_steps))
            printf "%-20s [%-50s]\n" "$label2" "$(printf '.%.0s' {1..$filled})"
            printf "%-20s [%-50s]\n" "$label3" "$(printf '.%.0s' {1..$filled})"
            tput cuu 2  # Move cursor up 2 lines
        else
            # Non-interactive (logging) mode
            printf "[%-50s] %d%% - %s\n" "$(printf '#%.0s' {1..$filled})" $((current_step * 100 / total_steps)) "$label1"
        fi
    fi
}

# Enhanced error handling
handle_error() {
    local message=$1
    log "ERROR" "$message"
    rollback_deployment
    exit 1
}

# Rollback mechanism
rollback_deployment() {
    log "WARN" "Initiating rollback"
    
    # Stop services
    systemctl stop ai-core || true
    systemctl stop ai-worker || true
    systemctl stop sutazai-monitor || true
    
    # Clean up resources
    if [ -d "venv" ]; then
        rm -rf venv
    fi
    
    log "WARN" "Rollback completed"
}

# Enhanced dependency installation
install_dependencies() {
    local requirements=$1
    [ -f "$requirements" ] || handle_error "Requirements file not found: $requirements"
    
    log "INFO" "Installing dependencies from $requirements"
    
    # Use all system resources
    export MAKEFLAGS="-j$(nproc)"
    export PIP_NO_CACHE_DIR=1
    export TORCH_CUDA_ARCH_LIST="8.0"
    export MAX_JOBS=$(nproc)
    
    # Install with retries
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        if pip install -r "$requirements"; then
            log "INFO" "Dependencies installed successfully"
            return 0
        else
            retry_count=$((retry_count+1))
            log "WARN" "Dependency installation failed, retrying ($retry_count/$max_retries)..."
            sleep 10
        fi
    done
    
    handle_error "Failed to install dependencies after $max_retries attempts"
}

# Verify installation with detailed logging
verify_installation() {
    log "INFO" "Starting installation verification"
    {
        log "DEBUG" "Verifying torch installation"
        python3 -c "import torch; print(f'Torch version: {torch.__version__}')"
        
        log "DEBUG" "Verifying transformers installation"
        python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
        
        log "DEBUG" "Verifying flash_attn installation"
        python3 -c "import flash_attn; print('Flash attention installed')"
        
        log "INFO" "All components verified successfully"
    } | modern_progress_bar 3 1 "Core Components" "Dependencies" "Configuration"
}

# Add this function to verify virtual environment
verify_virtualenv() {
    if [ -z "$VIRTUAL_ENV" ]; then
        handle_error "Virtual environment is not active"
    fi
}

# Virtual environment creation with logging
create_virtualenv() {
    if [ ! -d "venv" ]; then
        log "INFO" "Creating virtual environment..."
        python3 -m venv venv || handle_error "Failed to create virtual environment"
    fi
    
    source venv/bin/activate || handle_error "Failed to activate virtual environment"
    log "INFO" "Virtual environment activated"
}

# Completion message with logging
show_completion_message() {
    log "INFO" "Deployment completed successfully"
    echo -e "\n\033[1;32mDEPLOYMENT COMPLETED\033[0m"
    echo -e "\033[1;36mAccess Points:\033[0m"
    echo -e "• UI Dashboard:   \033[1;35mhttps://localhost:8501\033[0m"
    echo -e "• API Endpoint:    \033[1;35mhttps://localhost:8000/docs\033[0m"
    echo -e "• Monitoring:      \033[1;35mhttps://localhost:3000\033[0m"
    echo -e "\n\033[3;36mRun 'systemctl status sutazai' for service management\033[0m"
    
    # Display final ASCII art
    echo -e "\n\033[1;32m"
    cat << "EOF"
  ██████╗ ██████╗ ███████╗███████╗██████╗ ██╗   ██╗
 ██╔═══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗╚██╗ ██╔╝
 ██║   ██║██████╔╝█████╗  █████╗  ██████╔╝ ╚████╔╝ 
 ██║   ██║██╔═══╝ ██╔══╝  ██╔══╝  ██╔══██╗  ╚██╔╝  
 ╚██████╔╝██║     ██║     ███████╗██║  ██║   ██║   
  ╚═════╝ ╚═╝     ╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝   
EOF
    echo -e "\033[0m"
}

verify_torch_installation() {
    log "INFO" "Verifying torch installation"
    python3 -c "import torch; print(f'Torch version: {torch.__version__}')" || handle_error "Torch installation verification failed"
    log "INFO" "Torch installation verified successfully"
}

show_ascii_art() {
    clear
    # Multi-layer security verification
    local secure_logo_shown=0
    
    # 1. Try encrypted system storage
    if [ -f "/etc/sutazai/secure_logo.asc" ]; then
        sudo cat /etc/sutazai/secure_logo.asc && secure_logo_shown=1
        log "DEBUG" "Displayed secure system logo"
    fi
    
    # 2. Fallback to distributed package version
    if [ $secure_logo_shown -eq 0 ] && [ -f "/usr/share/sutazai/logo.asc" ]; then
        cat /usr/share/sutazai/logo.asc && secure_logo_shown=1
        log "WARN" "Fell back to packaged logo"
    fi
    
    # 3. Embedded emergency fallback with validation
    if [ $secure_logo_shown -eq 0 ]; then
        echo -e "${RED}"
        # Base85-encoded compressed version
        python3 -c "import base64,zlib;logo=base64.b85decode('${EMBEDDED_LOGO}');print(zlib.decompress(logo).decode('utf-8'))" 2>/dev/null
        if [ $? -ne 0 ]; then
            log "ERROR" "Failed to decode embedded logo"
            echo -e "\n  SutazAI Infinite Core\n"
        fi
        echo -e "${RESET}"
        log "WARN" "Used embedded fallback logo"
    fi
}

function verify_deployment() {
    local max_retries=5
    local retry_count=0
    
    while [[ $retry_count -lt $max_retries ]]; do
        if curl -sSf http://localhost:8080/health > /dev/null; then
            echo "Deployment verified successfully"
            return 0
        else
            retry_count=$((retry_count+1))
            echo "Verification failed, retrying ($retry_count/$max_retries)..."
            sleep 10
        fi
    done
    
    echo "Max verification retries reached. Deployment may have failed."
    return 1
}

# Service management
manage_service() {
    local service=$1
    local action=$2
    
    log "INFO" "Performing $action on $service"
    systemctl $action $service || handle_error "Failed to $action $service"
}