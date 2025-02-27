#!/bin/bash

# Automated performance tuning
AUTO_TUNE() {
    echo "Starting automated performance tuning..."
    
    # CPU optimization
    tune_cpu() {
        # Set CPU governor to performance
        for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            echo performance | tee $cpu
        done
    }
    
    # Memory optimization
    tune_memory() {
        # Adjust swappiness
        sysctl vm.swappiness=10
        
        # Clear caches
        sync; echo 3 > /proc/sys/vm/drop_caches
    }
    
    # Disk optimization
    tune_disk() {
        # Enable write caching
        for disk in $(lsblk -d -o NAME | grep -v NAME); do
            hdparm -W 1 /dev/$disk
        done
        
        # Optimize I/O scheduler
        for disk in $(lsblk -d -o NAME | grep -v NAME); do
            echo deadline > /sys/block/$disk/queue/scheduler
        done
    }
    
    tune_cpu
    tune_memory
    tune_disk
    echo "Performance tuning completed at $(date)" >> /var/log/performance_tuning.log
}

echo "=== Performance Tuning Check ==="
echo "CPU Cores: $(nproc)"
echo "Load Average: $(uptime | awk -F'load average: ' '{print $2}')"