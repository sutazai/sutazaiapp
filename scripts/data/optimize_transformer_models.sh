#!/bin/bash
#
# optimize_transformer_models.sh - Script to optimize transformer models for Dell PowerEdge R720 with E5-2640 CPUs
#
# This script runs optimization and benchmarking for transformer models for optimal
# performance on the Intel E5-2640 CPUs, configuring power and performance settings
# to get the best efficiency on the Dell R720 server.
#

# Set up logging
LOG_DIR="/opt/sutazaiapp/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/transformer_optimize_$(date +%Y%m%d_%H%M%S).log"

# Log function
log() {
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $1" | tee -a "$LOG_FILE"
}

# Print header
log "===== SutazAI Transformer Optimization Tool ====="
log "Optimizing for Dell PowerEdge R720 with E5-2640 CPUs"

# Parse arguments
MODEL_PATH=""
OUTPUT_DIR="/opt/sutazaiapp/models/optimized"
MODEL_TYPE="transformer"  # Default model type (transformer or llama)
SEQ_LENGTH=512
THREADS=12
AUTO_DOWNLOAD=false
MODEL_ID=""
HF_TOKEN=""

while [[ $# -gt 0 ]]; do
    key="$1"
    
    case $key in
        --model)
            MODEL_PATH="$2"
            shift
            shift
            ;;
        --model-id)
            MODEL_ID="$2"
            AUTO_DOWNLOAD=true
            shift
            shift
            ;;
        --auto-download)
            AUTO_DOWNLOAD=true
            MODEL_ID="$2"
            shift
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift
            shift
            ;;
        --seq_length)
            SEQ_LENGTH="$2"
            shift
            shift
            ;;
        --threads)
            THREADS="$2"
            shift
            shift
            ;;
        --token)
            HF_TOKEN="$2"
            shift
            shift
            ;;
        --force)
            FORCE_DOWNLOAD=true
            shift
            ;;
        *)
            log "Unknown option: $key"
            exit 1
            ;;
    esac
done

# Check if model path or model ID is provided
if [ -z "$MODEL_PATH" ] && [ -z "$MODEL_ID" ] && [ "$AUTO_DOWNLOAD" != "true" ]; then
    log "Error: Either --model /path/to/model or --model-id model_id must be provided"
    log "Example: --model /path/to/llama3-70b.gguf or --model-id llama3-70b"
    log "You can also use --auto-download to automatically select and download an optimal model"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
log "Output directory: $OUTPUT_DIR"

# Set Intel specific environment variables
log "Setting Intel CPU optimizations..."

# Check CPU model
cpu_model=$(grep -m 1 "model name" /proc/cpuinfo | cut -d ':' -f 2 | tr -s ' ')
log "CPU Model:$cpu_model"

# Verify E5-2640 CPU
if [[ $cpu_model != *"E5-2640"* ]]; then
    log "Warning: This script is optimized for Intel E5-2640 CPUs, but detected:$cpu_model"
    log "Some optimizations may not be optimal for your CPU."
fi

# Set Intel specific environment variables
export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export MKL_ENABLE_INSTRUCTIONS=AVX  # E5-2640 supports AVX but not AVX2
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=0
log "Threads set to: $THREADS"

# Recommend BIOS settings for E5-2640
log "===== Recommended BIOS Settings for E5-2640 ====="
log "1. Power Profile: 'Performance Per Watt (DAPC)'"
log "2. CPU Power Management: 'OS Control Mode'"
log "3. CPU C-States: 'Enabled with C1E only'"
log "4. Memory Power Savings: 'Disabled'"
log "5. Fan Speed Offset: 'Low Power' (for quieter operation)"
log "NOTE: These settings can reduce power consumption by ~30% with minimal performance impact"

# If auto-download is requested or a model ID is provided, use the model_downloader
if [ "$AUTO_DOWNLOAD" = true ] || [ ! -z "$MODEL_ID" ]; then
    log "Using enterprise model auto-download system..."
    
    # Prepare the download command
    DOWNLOADER_CMD="python3 -c \"
from core.processing.model_downloader import EnterpriseModelDownloader;
downloader = EnterpriseModelDownloader();
"
    
    if [ ! -z "$HF_TOKEN" ]; then
        DOWNLOADER_CMD+="downloader.hf_token = '$HF_TOKEN';"
    fi
    
    if [ ! -z "$MODEL_ID" ]; then
        DOWNLOADER_CMD+="path = downloader.get_model('$MODEL_ID', force_download=$FORCE_DOWNLOAD);"
    else
        # If no specific model_id, use the optimal model for E5-2640
        DOWNLOADER_CMD+="
from core.processing.model_downloader import get_optimal_model_for_e5_2640;
model_id = get_optimal_model_for_e5_2640();
print(f'Auto-selecting optimal model for E5-2640: {model_id}');
path = downloader.get_model(model_id, force_download=$FORCE_DOWNLOAD);
"
    fi
    
    DOWNLOADER_CMD+="print(f'MODEL_PATH={path}');\""
    
    log "Executing model downloader..."
    eval_result=$(eval "$DOWNLOADER_CMD")
    
    # Extract the model path from the output
    MODEL_PATH=$(echo "$eval_result" | grep "MODEL_PATH=" | cut -d'=' -f2)
    
    if [ -z "$MODEL_PATH" ]; then
        log "Error: Failed to download model"
        log "Downloader output: $eval_result"
        exit 1
    fi
    
    log "Successfully downloaded model to: $MODEL_PATH"
    
    # Determine model type from filename
    if [[ "$MODEL_PATH" == *".gguf"* ]] || [[ "$MODEL_PATH" == *"llama"* ]] || [[ "$MODEL_PATH" == *"Llama"* ]]; then
        MODEL_TYPE="llama"
        log "Detected model type: llama"
    else
        MODEL_TYPE="transformer"
        log "Detected model type: transformer"
    fi
fi

# Check if using Llama3 70B model
if [[ $MODEL_TYPE == "llama" || $MODEL_PATH == *"llama"* || $MODEL_PATH == *"Llama"* || $MODEL_PATH == *".gguf"* ]]; then
    log "Detected Llama model, using llama-cpp-python optimizations"
    
    # Additional settings for Llama models
    export OMP_SCHEDULE=static
    export GGML_OPENCL_PLATFORM=0
    export GGML_OPENCL_DEVICE=0
    
    # Run the Python script with Llama-specific options
    log "Starting optimization for Llama model..."
    python3 -m core.processing.transformer_optimizer --model "$MODEL_PATH" \
        --output "$OUTPUT_DIR" \
        --model_type llama \
        --threads $THREADS \
        --log_file "$LOG_FILE" \
        --optimizations int8 context_size cpu_threads batch_size \
        --seq_length $SEQ_LENGTH
else
    # Run the Python script for standard transformer models
    log "Starting optimization for transformer model..."
    python3 -m core.processing.transformer_optimizer --model "$MODEL_PATH" \
        --output "$OUTPUT_DIR" \
        --model_type transformer \
        --threads $THREADS \
        --log_file "$LOG_FILE" \
        --optimizations int8 bettertransformer lookupffn \
        --seq_length $SEQ_LENGTH
fi

# Check if optimization was successful
if [ $? -eq 0 ]; then
    log "Optimization completed successfully."
    log "Optimized model saved to: $OUTPUT_DIR"
    log "Log file: $LOG_FILE"
    
    # Display system information for reference
    log "===== System Information ====="
    log "CPU: $(grep -m 1 'model name' /proc/cpuinfo | cut -d ':' -f 2 | tr -s ' ')"
    log "RAM: $(free -h | grep Mem | awk '{print $2}')"
    
    # Memory consumption
    mem_before=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    mem_after=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    mem_used=$(( (mem_before - mem_after) / 1024 ))
    log "Memory used during optimization: ~$mem_used MB"
    
    log "===== Optimization Summary ====="
    # If summary file exists, display it
    if [ -f "$OUTPUT_DIR/benchmark_results.json" ]; then
        log "Benchmark results available at: $OUTPUT_DIR/benchmark_results.json"
        # Extract key stats from the benchmark file
        if command -v jq &> /dev/null; then
            best_opt=$(jq -r '.recommended // "N/A"' "$OUTPUT_DIR/benchmark_results.json")
            log "Recommended optimization: $best_opt"
        else
            log "Install 'jq' to view benchmark summary details"
        fi
    fi
    
    if [ "$MODEL_TYPE" == "llama" ]; then
        log "To use the optimized Llama model:"
        log "python3 -c 'from core.processing.llama_utils import get_optimized_model; model = get_optimized_model(\"$OUTPUT_DIR/llama_optimized_config.json\")'"
    else
        log "To use the optimized transformer model, load from: $OUTPUT_DIR"
    fi
    
    exit 0
else
    log "Error: Optimization failed."
    log "Please check the log file for details: $LOG_FILE"
    exit 1
fi