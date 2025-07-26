#!/bin/bash
#
# SutazAI AGI/ASI Deployment Script
#
# This script automates the deployment of the full AGI/ASI system.
# It handles dependency installation, system configuration, and startup.
#
# Usage: ./deploy_agi_system.sh [--debug] [--no-gpu] [--skip-deps]
#

# Exit on any error
set -e

# Script configuration
PROJECT_ROOT=$(pwd)
LOG_DIR="$PROJECT_ROOT/logs"
CONFIG_DIR="$PROJECT_ROOT/config"
DATA_DIR="$PROJECT_ROOT/data"
VENV_DIR="$PROJECT_ROOT/venv"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$LOG_DIR/deployment_$TIMESTAMP.log"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Initialize logging
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== SutazAI AGI/ASI Deployment Started at $(date) ==="
echo "Working directory: $PROJECT_ROOT"

# Parse command line arguments
DEBUG=false
USE_GPU=true
INSTALL_DEPENDENCIES=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG=true
            shift
            ;;
        --no-gpu)
            USE_GPU=false
            shift
            ;;
        --skip-deps)
            INSTALL_DEPENDENCIES=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./deploy_agi_system.sh [--debug] [--no-gpu] [--skip-deps]"
            exit 1
            ;;
    esac
done

if [ "$DEBUG" = true ]; then
    echo "Debug mode enabled"
fi

if [ "$USE_GPU" = false ]; then
    echo "GPU support disabled"
fi

# Function to log steps
log_step() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") - $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# System check
log_step "Performing system check"

# Check if running on the correct hardware (Dell PowerEdge R720)
if command_exists dmidecode; then
    SYSTEM_MODEL=$(sudo dmidecode -s system-product-name)
    if [[ $SYSTEM_MODEL == *"PowerEdge R720"* ]]; then
        log_step "✅ Running on supported hardware: $SYSTEM_MODEL"
    else
        log_step "⚠️ Warning: Not running on Dell PowerEdge R720 (detected: $SYSTEM_MODEL)"
        log_step "Some optimizations may not be applied correctly"
    fi
else
    log_step "⚠️ Warning: Cannot determine system model (dmidecode not available)"
fi

# Check CPU cores
CPU_CORES=$(nproc)
if [ "$CPU_CORES" -ge 12 ]; then
    log_step "✅ CPU cores: $CPU_CORES (sufficient)"
else
    log_step "⚠️ Warning: Only $CPU_CORES CPU cores detected. Recommended: 12+"
fi

# Check available memory
if command_exists free; then
    TOTAL_MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_MEM_GB" -ge 100 ]; then
        log_step "✅ Memory: ${TOTAL_MEM_GB}GB (sufficient)"
    else
        log_step "⚠️ Warning: Only ${TOTAL_MEM_GB}GB memory detected. Recommended: 128GB+"
    fi
fi

# Check available disk space
DISK_SPACE_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$DISK_SPACE_GB" -ge 1000 ]; then
    log_step "✅ Disk space: ${DISK_SPACE_GB}GB (sufficient)"
else
    log_step "⚠️ Warning: Only ${DISK_SPACE_GB}GB disk space available. Recommended: 1TB+"
fi

# Check for GPU if enabled
if [ "$USE_GPU" = true ]; then
    if command_exists nvidia-smi; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1)
        log_step "✅ GPU detected: $GPU_INFO"
    else
        log_step "⚠️ Warning: NVIDIA GPU tools not found. Running without GPU acceleration."
        USE_GPU=false
    fi
fi

# Install system dependencies
if [ "$INSTALL_DEPENDENCIES" = true ]; then
    log_step "Installing system dependencies"
    
    # Check for package manager
    if command_exists apt-get; then
        log_step "Detected apt package manager"
        sudo apt-get update
        sudo apt-get install -y \
            python3.11 \
            python3.11-dev \
            python3.11-venv \
            python3-pip \
            build-essential \
            cmake \
            git \
            wget \
            curl \
            libssl-dev \
            zlib1g-dev \
            libbz2-dev \
            libreadline-dev \
            libsqlite3-dev \
            libffi-dev \
            liblzma-dev \
            tk-dev
        
        if [ "$USE_GPU" = true ]; then
            log_step "Installing NVIDIA CUDA dependencies"
            # Check if CUDA is already installed
            if ! command_exists nvcc; then
                log_step "CUDA not found. Please install CUDA manually."
                log_step "See: https://developer.nvidia.com/cuda-downloads"
            fi
        fi
        
    elif command_exists yum; then
        log_step "Detected yum package manager"
        sudo yum update -y
        sudo yum install -y \
            python3-devel \
            python3-pip \
            python3-venv \
            gcc \
            gcc-c++ \
            cmake \
            git \
            wget \
            curl \
            openssl-devel \
            bzip2-devel \
            libffi-devel \
            zlib-devel \
            sqlite-devel
    else
        log_step "⚠️ Unsupported package manager. Please install dependencies manually."
    fi
else
    log_step "Skipping dependency installation (--skip-deps flag detected)"
fi

# Create and activate Python virtual environment
log_step "Setting up Python virtual environment"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

# Install Python dependencies
log_step "Installing Python dependencies"
pip install -r requirements.txt

# Create required directories
log_step "Setting up directory structure"
mkdir -p "$CONFIG_DIR/constraints"
mkdir -p "$DATA_DIR/models"
mkdir -p "$DATA_DIR/embeddings"
mkdir -p "$DATA_DIR/vectors"

# Apply optimizations for Dell PowerEdge R720
log_step "Applying hardware-specific optimizations"

# NUMA node configuration for optimal memory access
if command_exists numactl; then
    NUMA_NODES=$(numactl --hardware | grep "available:" | awk '{print $2}')
    if [ "$NUMA_NODES" -gt 1 ]; then
        log_step "Configuring for $NUMA_NODES NUMA nodes"
        # Create NUMA configuration file
        cat > "$CONFIG_DIR/numa_config.json" << EOF
{
    "numa_nodes": $NUMA_NODES,
    "memory_strategy": "interleave",
    "cpu_binding": {
        "neuromorphic_engine": [0, 1],
        "vector_db": [2, 3],
        "agent_manager": [4, 5],
        "web_server": [6, 7]
    }
}
EOF
    fi
fi

# Create configuration files
log_step "Generating system configuration"

# Main AGI configuration
if [ ! -f "$CONFIG_DIR/agi_system.json" ]; then
    cat > "$CONFIG_DIR/agi_system.json" << EOF
{
    "system": {
        "name": "SutazAI AGI/ASI Autonomous System",
        "version": "0.1.0",
        "description": "Autonomous AGI system with self-improvement capabilities",
        "max_memory_usage_gb": $(($TOTAL_MEM_GB - 10)),
        "max_cpu_usage_percent": 90,
        "debug_mode": $([ "$DEBUG" = true ] && echo "true" || echo "false")
    },
    "neuromorphic_engine": {
        "enabled": true,
        "port": 50051,
        "energy_efficient_mode": true,
        "memory_efficient_mode": true
    },
    "self_modification": {
        "enabled": true,
        "auto_approve": false,
        "max_sandbox_time": 30,
        "max_changes_per_event": 10
    },
    "agent_manager": {
        "enabled": true,
        "max_agents": 10,
        "default_agent_timeout": 300,
        "agent_types": [
            "document_agent",
            "code_agent",
            "reasoning_agent",
            "planning_agent"
        ]
    },
    "api_server": {
        "enabled": true,
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 4,
        "ssl_enabled": false
    },
    "web_ui": {
        "enabled": true,
        "port": 8501,
        "theme": "dark"
    },
    "security": {
        "air_gap_mode": false,
        "encryption_enabled": true,
        "audit_logging": true,
        "max_token_limit": 100000
    }
}
EOF
    log_step "Created AGI system configuration"
fi

# Create starter ethical constraint
if [ ! -f "$CONFIG_DIR/constraints/safety_constraints.json" ]; then
    cat > "$CONFIG_DIR/constraints/safety_constraints.json" << EOF
{
    "id": "safety-002",
    "name": "Prevent Unauthorized Data Exfiltration",
    "description": "System must not transmit sensitive data to external systems without authorization",
    "category": "security",
    "severity": "critical",
    "scope": "system",
    "formal_specification": "∀d ∈ SensitiveData, ∀t ∈ Transmission: transmit(d, t) → authorized(t)",
    "verification_module": "ai_agents.ethical_constraints",
    "verification_function": "verify_data_exfiltration"
}
EOF
    log_step "Created starter ethical constraint configuration"
fi

# Test the installation
log_step "Testing installation"

# Generate test data
TEST_DIR="$PROJECT_ROOT/tests/test_data"
mkdir -p "$TEST_DIR"

# Create a test file if it doesn't exist
if [ ! -f "$TEST_DIR/test_document.txt" ]; then
    cat > "$TEST_DIR/test_document.txt" << EOF
This is a test document for the SutazAI AGI/ASI system.
It should be processed correctly if the system is working properly.
EOF
fi

# Run basic validation tests
python -c "
import os
import sys
import importlib.util

errors = []

# Check required components
components = [
    'neuromorphic.core',
    'ai_agents.base_agent',
    'ai_agents.protocols.self_modification',
    'ai_agents.ethical_constraints'
]

for component in components:
    try:
        module_name, class_name = component.rsplit('.', 1)
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            errors.append(f'Module {module_name} not found')
        else:
            print(f'✅ Component {component} is available')
    except ImportError as e:
        errors.append(f'Error importing {component}: {str(e)}')

# Check directories
required_dirs = [
    'logs',
    'config',
    'data',
    'neuromorphic',
    'ai_agents'
]

for dir_name in required_dirs:
    if not os.path.isdir(dir_name):
        errors.append(f'Required directory {dir_name} not found')
    else:
        print(f'✅ Directory {dir_name} exists')

if errors:
    print('❌ Validation failed with the following errors:')
    for error in errors:
        print(f'  - {error}')
    sys.exit(1)
else:
    print('✅ All validation checks passed')
"

TEST_RESULT=$?

if [ $TEST_RESULT -eq 0 ]; then
    log_step "✅ Installation tests passed"
else
    log_step "❌ Installation tests failed. Please check the logs for details."
    exit 1
fi

# Create startup script
log_step "Creating startup script"

cat > "$PROJECT_ROOT/start_agi.sh" << 'EOF'
#!/bin/bash
# AGI System Startup Script

PROJECT_ROOT=$(pwd)
VENV_DIR="$PROJECT_ROOT/venv"
LOG_DIR="$PROJECT_ROOT/logs"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$LOG_DIR/agi_system_$TIMESTAMP.log"

mkdir -p "$LOG_DIR"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Start the AGI system
echo "Starting AGI system at $(date)"
echo "Logs will be written to $LOG_FILE"

# Use nohup to keep the process running after terminal closes
nohup python scripts/initialize_agi.py "$@" > "$LOG_FILE" 2>&1 &

# Save the process ID
echo $! > "$PROJECT_ROOT/.agi.pid"
echo "AGI system started with PID: $!"
echo "To stop the system, run: ./stop_agi.sh"
EOF

# Create stop script
cat > "$PROJECT_ROOT/stop_agi.sh" << 'EOF'
#!/bin/bash
# AGI System Shutdown Script

PROJECT_ROOT=$(pwd)
PID_FILE="$PROJECT_ROOT/.agi.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "Stopping AGI system with PID: $PID"
    kill -15 $PID
    sleep 5
    if kill -0 $PID 2>/dev/null; then
        echo "AGI system didn't stop gracefully, forcing termination"
        kill -9 $PID
    fi
    rm "$PID_FILE"
    echo "AGI system stopped"
else
    echo "No PID file found. AGI system may not be running."
fi
EOF

chmod +x "$PROJECT_ROOT/start_agi.sh"
chmod +x "$PROJECT_ROOT/stop_agi.sh"

log_step "✅ Deployment completed successfully!"
log_step "To start the AGI system, run: ./start_agi.sh"
log_step "To stop the AGI system, run: ./stop_agi.sh"

# Final summary
echo ""
echo "=== DEPLOYMENT SUMMARY ==="
echo "Installation directory: $PROJECT_ROOT"
echo "Configuration: $CONFIG_DIR/agi_system.json"
echo "Logs: $LOG_DIR"
echo "Data: $DATA_DIR"
echo "Deployment log: $LOG_FILE"
echo ""
echo "System will use $([ "$USE_GPU" = true ] && echo "GPU acceleration" || echo "CPU only mode")"
echo "Debug mode: $([ "$DEBUG" = true ] && echo "Enabled" || echo "Disabled")"
echo ""
echo "=== SutazAI AGI/ASI Deployment Completed at $(date) ===" 