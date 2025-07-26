#!/bin/bash
#
# setup_transformer_environment.sh - Set up the environment for transformer optimization on Dell PowerEdge R720
#
# This script installs all required dependencies and configures the environment
# for optimal performance on E5-2640 CPUs in the Dell PowerEdge R720 server.
#

# Set up logging
LOG_DIR="/opt/sutazaiapp/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/environment_setup_$(date +%Y%m%d_%H%M%S).log"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting environment setup for transformer optimization"

# Parse command line arguments
INSTALL_DIR="/opt/sutazaiapp"
VENV_DIR="/opt/venv-sutazaiapp"
REQUIREMENTS="$INSTALL_DIR/requirements.txt"
INSTALL_DEPS=true
CREATE_VENV=true
CONFIGURE_CPU=true

# Process command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --install-dir)
            INSTALL_DIR="$2"
            shift
            shift
            ;;
        --venv-dir)
            VENV_DIR="$2"
            shift
            shift
            ;;
        --requirements)
            REQUIREMENTS="$2"
            shift
            shift
            ;;
        --skip-deps)
            INSTALL_DEPS=false
            shift
            ;;
        --skip-venv)
            CREATE_VENV=false
            shift
            ;;
        --skip-cpu-config)
            CONFIGURE_CPU=false
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "  --install-dir      Installation directory (default: /opt/sutazaiapp)"
            echo "  --venv-dir         Python virtual environment directory (default: /opt/venv-sutazaiapp)"
            echo "  --requirements     Path to requirements.txt (default: \$INSTALL_DIR/requirements.txt)"
            echo "  --skip-deps        Skip installing system dependencies"
            echo "  --skip-venv        Skip creating Python virtual environment"
            echo "  --skip-cpu-config  Skip CPU configuration"
            exit 0
            ;;
        *)
            echo "Unknown option: $key"
            echo "Use --help to see available options"
            exit 1
            ;;
    esac
done

# Check for root privileges if needed
if [ "$INSTALL_DEPS" = true ] && [ "$(id -u)" -ne 0 ]; then
    log "Warning: You are not running as root. System dependencies installation may fail."
    log "Consider running with sudo if you encounter permission issues with system dependencies."
fi

# Create directories
mkdir -p "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR/scripts"
mkdir -p "$INSTALL_DIR/models"
mkdir -p "$INSTALL_DIR/models/optimized"
mkdir -p "$INSTALL_DIR/core/neural"

log "Created directory structure in $INSTALL_DIR"

# Install system dependencies if requested
if [ "$INSTALL_DEPS" = true ]; then
    log "Installing system dependencies..."
    
    # Check what package manager to use
    if command -v apt-get &> /dev/null; then
        log "Using apt package manager"
        apt-get update
        apt-get install -y \
            python3 \
            python3-dev \
            python3-pip \
            python3-venv \
            build-essential \
            cmake \
            git \
            libblas-dev \
            liblapack-dev \
            libopenmpi-dev \
            ipmitool \
            numactl
        
    elif command -v yum &> /dev/null; then
        log "Using yum package manager"
        yum update -y
        yum install -y \
            python3 \
            python3-devel \
            python3-pip \
            gcc \
            gcc-c++ \
            make \
            cmake \
            git \
            blas-devel \
            lapack-devel \
            openmpi-devel \
            ipmitool \
            numactl
    else
        log "Warning: Unsupported package manager. Please install dependencies manually."
    fi
    
    log "System dependencies installation completed"
fi

# Create Python virtual environment if requested
if [ "$CREATE_VENV" = true ]; then
    log "Creating Python virtual environment at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    
    log "Upgrading pip in virtual environment"
    pip install --upgrade pip
    
    log "Installing wheel for better package building"
    pip install wheel
    
    # Check if requirements file exists
    if [ -f "$REQUIREMENTS" ]; then
        log "Installing Python dependencies from $REQUIREMENTS"
        pip install -r "$REQUIREMENTS"
    else
        log "Warning: Requirements file not found at $REQUIREMENTS"
        log "Installing base dependencies manually"
        
        # Install core dependencies manually
        pip install torch==1.10.0 transformers==4.25.0 accelerate==0.16.0 bitsandbytes==0.37.0
        pip install psutil numpy pandas matplotlib
        pip install intel-extension-for-pytorch mkl-service intel-openmp
    fi
    
    log "Python environment setup completed"
else
    log "Skipping Python virtual environment creation"
fi

# Configure CPU settings for E5-2640 if requested
if [ "$CONFIGURE_CPU" = true ]; then
    log "Checking CPU model for configuration..."
    CPU_MODEL=$(grep "model name" /proc/cpuinfo | head -n 1 | cut -d':' -f2 | sed 's/^[ \t]*//')
    log "Detected CPU: $CPU_MODEL"
    
    if [[ "$CPU_MODEL" == *"E5-2640"* ]]; then
        log "Confirmed E5-2640 CPU, applying optimized configuration"
        
        # Create a system profile for E5-2640
        PROFILE_DIR="/etc/profile.d"
        PROFILE_FILE="$PROFILE_DIR/intel_e5_2640_optimizations.sh"
        
        # Check if we can create the system profile
        if [ -d "$PROFILE_DIR" ] && [ "$(id -u)" -eq 0 ]; then
            log "Creating system-wide profile for E5-2640 optimizations"
            
            cat > "$PROFILE_FILE" << 'EOF'
# Intel E5-2640 CPU optimizations for Dell PowerEdge R720
# 
# This profile sets environment variables for optimal performance on E5-2640 CPUs

# Set thread count to match physical cores (E5-2640 has 6 cores per socket, 12 total)
export MKL_NUM_THREADS=12
export OMP_NUM_THREADS=12

# E5-2640 supports AVX but not AVX2
export MKL_ENABLE_INSTRUCTIONS=AVX

# Intel-specific optimizations
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0

# TensorFlow-specific optimizations
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=12

# PyTorch-specific optimizations
export TORCH_CPU_PARALLEL_BACKENDS=MKL
export TORCH_BACKENDS_CUDNN_ENABLED=false
EOF
            
            chmod +x "$PROFILE_FILE"
            log "Created system-wide profile at $PROFILE_FILE"
        else
            log "Warning: Cannot create system-wide profile. Creating local profile."
            
            # Create a local profile
            LOCAL_PROFILE="$INSTALL_DIR/scripts/activate_e5_2640_optimizations.sh"
            
            cat > "$LOCAL_PROFILE" << 'EOF'
#!/bin/bash
# Intel E5-2640 CPU optimizations for Dell PowerEdge R720
# 
# Source this file to set environment variables for optimal performance on E5-2640 CPUs

# Set thread count to match physical cores (E5-2640 has 6 cores per socket, 12 total)
export MKL_NUM_THREADS=12
export OMP_NUM_THREADS=12

# E5-2640 supports AVX but not AVX2
export MKL_ENABLE_INSTRUCTIONS=AVX

# Intel-specific optimizations
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0

# TensorFlow-specific optimizations
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=12

# PyTorch-specific optimizations
export TORCH_CPU_PARALLEL_BACKENDS=MKL
export TORCH_BACKENDS_CUDNN_ENABLED=false

echo "Intel E5-2640 CPU optimizations enabled"
EOF
            
            chmod +x "$LOCAL_PROFILE"
            log "Created local profile at $LOCAL_PROFILE"
            log "Activate with: source $LOCAL_PROFILE"
        fi
        
        # Check for NUMA configuration
        if command -v numactl &> /dev/null; then
            log "NUMA detected, recommending NUMA settings"
            log "For optimal performance, run inference with: numactl --cpunodebind=0 --membind=0 python script.py"
        fi
        
        # Recommend BIOS settings
        log "Recommended BIOS settings for Dell PowerEdge R720 with E5-2640:"
        log "1. Power Management: Set to 'System BIOS settings performance per watt (OS)'"
        log "2. CPU Power and Performance: Performance"
        log "3. Memory Frequency: Maximum Performance"
        log "4. Memory Operating Mode: Optimizer Mode"
        log "5. Node Interleaving: Disabled (for NUMA benefits)"
        
    else
        log "Warning: Not running on E5-2640 CPU, skipping CPU-specific configuration"
    fi
fi

# Create a simple test script to verify the environment
TEST_SCRIPT="$INSTALL_DIR/scripts/test_transformer_environment.py"

cat > "$TEST_SCRIPT" << 'EOF'
#!/usr/bin/env python3
# Test script to verify transformer optimization environment

import os
import sys
import platform
import torch
import numpy as np

def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 50)
    print(f" {title}")
    print("=" * 50)

def print_env_var(name):
    """Print an environment variable value."""
    value = os.environ.get(name, "Not set")
    print(f"{name}: {value}")

print_section("System Information")
print(f"Python version: {platform.python_version()}")
print(f"CPU: {platform.processor()}")
print(f"Platform: {platform.platform()}")

# Get CPU details
try:
    import psutil
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    print(f"Physical cores: {physical_cores}")
    print(f"Logical cores: {logical_cores}")
    
    memory = psutil.virtual_memory()
    print(f"Total memory: {memory.total / (1024**3):.2f} GB")
    print(f"Available memory: {memory.available / (1024**3):.2f} GB")
except ImportError:
    print("psutil not installed, skipping detailed CPU info")

print_section("PyTorch Information")
print(f"PyTorch version: {torch.__version__}")
print(f"Available backend: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
print(f"PyTorch thread count: {torch.get_num_threads()}")

# Test Intel optimizations
try:
    import intel_extension_for_pytorch as ipex
    print(f"Intel Extension for PyTorch version: {ipex.__version__}")
    print("Intel Extension for PyTorch is available")
except ImportError:
    print("Intel Extension for PyTorch not installed")

print_section("Environment Variables")
print_env_var("MKL_NUM_THREADS")
print_env_var("OMP_NUM_THREADS")
print_env_var("MKL_ENABLE_INSTRUCTIONS")
print_env_var("KMP_AFFINITY")
print_env_var("KMP_BLOCKTIME")

print_section("Tensor Performance Test")
# Simple matrix multiplication benchmark
try:
    sizes = [1000, 2000, 4000]
    results = []
    
    print("Running simple matrix multiplication benchmark...")
    for size in sizes:
        # Create random matrices
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        
        # Warm-up
        for _ in range(3):
            c = torch.matmul(a, b)
        
        # Benchmark
        import time
        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        
        results.append((size, end - start))
    
    print("\nMatrix multiplication results:")
    print("Size\tTime (s)")
    for size, duration in results:
        print(f"{size}\t{duration:.4f}")
except Exception as e:
    print(f"Error running benchmark: {e}")

print("\nEnvironment check completed.")
EOF

chmod +x "$TEST_SCRIPT"
log "Created test script at $TEST_SCRIPT"

log "Environment setup completed successfully"
log "To use the environment:"
log "1. Activate the Python virtual environment: source $VENV_DIR/bin/activate"
log "2. Run the test script: python $TEST_SCRIPT"
log "3. Optimize transformer models: $INSTALL_DIR/scripts/optimize_transformer_models.sh"

# Provide final recommendations
log "Recommended next steps:"
log "1. Review system configuration for Dell PowerEdge R720 BIOS settings"
log "2. Verify transformer optimization environment with the test script"
log "3. Download transformer models for optimization"
log "4. Run optimization with optimize_transformer_models.sh"

exit 0 