#!/bin/bash
set -eo pipefail

# AI-Powered Deployment Guardian
DEPLOY_MODE="${SUTAZAI_DEV_MODE:-production}"
STRICT_MODE=true

analyze_environment() {
  local warnings=()
  local critical=()

  # Hardware analysis
  (( $(jq '.cpu.cores' <<< "$HARDWARE_PROFILE") < 2 )) && \
    critical+=("Insufficient CPU: $(jq '.cpu.cores' <<< "$HARDWARE_PROFILE") cores")
  
  (( $(jq '.memory.total_mb' <<< "$HARDWARE_PROFILE") < 4096 )) && \
    critical+=("Low memory: $(jq '.memory.total_mb' <<< "$HARDWARE_PROFILE")MB")

  # Dependency analysis
  ! command -v nvidia-smi >/dev/null && \
    warnings+=("NVIDIA tools missing - GPU features disabled")

  # Storage check
  (( $(jq '.storage.root_gb' <<< "$HARDWARE_PROFILE") < 50 )) && \
    warnings+=("Low disk space: $(jq '.storage.root_gb' <<< "$HARDWARE_PROFILE")GB")

  # Security checks
  [[ $(sysctl -n vm.swappiness) -gt 30 ]] && \
    warnings+=("High swappiness: $(sysctl -n vm.swappiness)")
  
  # Output analysis
  if [[ ${#critical[@]} -gt 0 && "$DEPLOY_MODE" == "production" ]]; then
    echo -e "${RED}CRITICAL ISSUES DETECTED:${RESET}"
    printf '  - %s\n' "${critical[@]}"
    echo -e "${YELLOW}Enable development mode: export SUTAZAI_DEV_MODE=1${RESET}"
    $STRICT_MODE && exit 1
  fi

  [[ ${#warnings[@]} -gt 0 ]] && {
    echo -e "${YELLOW}WARNINGS:${RESET}"
    printf '  - %s\n' "${warnings[@]}"
  }
}

auto_configure() {
  # Adaptive resource allocation
  local cores=$(jq '.cpu.cores' <<< "$HARDWARE_PROFILE")
  local mem=$(jq '.memory.total_mb' <<< "$HARDWARE_PROFILE")
  
  export AI_BATCH_SIZE=$(( cores * 2 ))
  export AI_MAX_MEMORY="$(( mem * 85 / 100 ))m"
  
  # GPU auto-config
  if (( $(jq '.gpu.count' <<< "$HARDWARE_PROFILE") > 0 )); then
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export TF_GPU_ALLOCATOR=cuda_malloc_async
  else
    export TF_FORCE_GPU_ALLOW_GROWTH=true
  fi
}

main() {
  analyze_environment
  auto_configure
  echo -e "${GREEN}System validated for $DEPLOY_MODE mode${RESET}"
} 