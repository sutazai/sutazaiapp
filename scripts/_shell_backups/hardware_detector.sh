#!/bin/bash
set -eo pipefail

# Advanced hardware profiling
generate_hardware_profile() {
  echo "{
  \"cpu\": {
    \"cores\": $(nproc),
    \"arch\": \"$(uname -m)\",
    \"features\": \"$(grep flags /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)\"
  },
  \"memory\": {
    \"total_mb\": $(($(grep MemTotal /proc/meminfo | awk '{print $2}')/1024)),
    \"swap_mb\": $(($(grep SwapTotal /proc/meminfo | awk '{print $2}')/1024))
  },
  \"storage\": {
    \"root_gb\": $(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
  },
  \"gpu\": {
    \"count\": $(command -v nvidia-smi >/dev/null && nvidia-smi -L | wc -l || echo 0),
    \"info\": \"$(command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo 'none')\"
  }
}"
}

# Export JSON profile
export HARDWARE_PROFILE=$(generate_hardware_profile)

# Adaptive configuration
configure_system() {
  local cores=$(jq -r '.cpu.cores' <<< "$HARDWARE_PROFILE")
  
  # CPU optimization
  export OMP_NUM_THREADS=$cores
  export MKL_NUM_THREADS=$cores
  
  # GPU configuration
  if (( $(jq -r '.gpu.count' <<< "$HARDWARE_PROFILE") > 0 )); then
    export CUDA_VISIBLE_DEVICES=0
    export TF_GPU_THREAD_MODE='gpu_private'
  else
    export TF_FORCE_GPU_ALLOW_GROWTH='false'
  fi
  
  # Memory limits
  local total_mb=$(jq -r '.memory.total_mb' <<< "$HARDWARE_PROFILE")
  export JVM_HEAP_SIZE=$((total_mb * 3 / 4))m
}

# Main execution
case "$1" in
  profile)
    echo "$HARDWARE_PROFILE"
    ;;
  configure)
    configure_system
    ;;
  *)
    echo "Usage: $0 {profile|configure}"
    exit 1
esac 