#!/bin/bash
optimize_hardware() {
  # GPU Configuration
  if lspci | grep -qi 'nvidia'; then
    nvidia-smi -pm 1
    nvidia-smi -acp 0
    nvidia-smi --auto-boost-default=0
    nvidia-smi -pl 250
  fi

  # CPU Optimization
  echo "performance" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
  sysctl -w vm.swappiness=10
  
  # Storage Optimization
  if command -v fstrim; then
    fstrim -av
    systemctl enable fstrim.timer
  fi
  
  # Network Tuning
  sysctl -w net.core.rmem_max=16777216
  sysctl -w net.core.wmem_max=16777216
} 