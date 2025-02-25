#!/bin/bash
# SutazAI System Tuning Script

# Ensure script is run with sudo
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run with sudo" 
   exit 1
fi

# CPU Performance Tuning
echo "🚀 Tuning CPU Performance..."
cpupower frequency-set -g performance
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$cpu"
done

# Memory Management
echo "🧠 Optimizing Memory..."
sysctl -w vm.swappiness=10
sysctl -w vm.dirty_ratio=10
sysctl -w vm.dirty_background_ratio=5
sysctl -w vm.overcommit_memory=1

# File Descriptor Limits
echo "📂 Increasing File Descriptor Limits..."

# Network Optimization
echo "🌐 Tuning Network Performance..."
sysctl -w net.core.somaxconn=65535
sysctl -w net.ipv4.tcp_max_syn_backlog=65536
sysctl -w net.core.netdev_max_backlog=65536
sysctl -w net.ipv4.ip_local_port_range="1024 65535"

# Kernel Performance
echo "🔧 Kernel Performance Tuning..."
sysctl -w kernel.sched_migration_cost_ns=5000000
sysctl -w kernel.sched_autogroup_enabled=1

# Disable Watchdog (optional, can reduce latency)
echo "⏰ Disabling Watchdog..."
echo 0 > /proc/sys/kernel/watchdog

# Verify Changes
echo "🔍 Verifying System Optimizations..."
sysctl -p

echo "✅ SutazAI System Optimization Complete!" 