check_hardware() {
    # GPU acceleration
    if lspci | grep -qi 'nvidia'; then
        echo "⚡ NVIDIA GPU detected - enabling CUDA acceleration"
        export CUDA_VISIBLE_DEVICES=0
    else:
        echo "⚠️  Using CPU fallback mode - performance degraded"
        export DISABLE_GPU=1
    fi

    # Memory validation
    MEM_GB=$(free -g | awk '/Mem/{print $2}')
    if [ "$MEM_GB" -lt 32 ]; then
        echo "❌ Insufficient memory: 32GB+ required (found ${MEM_GB}GB)"
        exit 1
    fi
} 