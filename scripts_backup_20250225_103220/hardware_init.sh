initialize_hardware() {
    # GPU Detection
    if lspci | grep -qi 'nvidia'; then
        echo "⚡ NVIDIA GPU Detected - Initializing CUDA"
        export CUDA_VISIBLE_DEVICES=0
        if ! nvidia-smi; then
            echo "❌ NVIDIA Drivers Missing - Installing..."
            sudo apt-get install -y nvidia-driver-535 nvidia-container-toolkit
            sudo systemctl restart docker
        fi
    else
        echo "⚠️  Using CPU Fallback Mode - Install NVIDIA GPU for Optimal Performance"
        export DISABLE_GPU=1
    fi

    # Memory Validation
    MEM_GB=$(free -g | awk '/Mem/{print $2}')
    if [ "$MEM_GB" -lt 32 ]; then
        echo "❌ Insufficient Memory: 32GB+ Required (Found ${MEM_GB}GB)" >&2
        exit 1
    fi
} 