#!/bin/bash

# Automated hardware health monitoring
AUTO_HARDWARE() {
    echo "Starting automated hardware health check..."
    
    # Check CPU health
    check_cpu() {
        local temp=$(sensors | grep 'Core' | awk '{print $3}' | sed 's/+//;s/°C//')
        if (( $(echo "$temp > 80" | bc -l) )); then
            echo "CPU temperature warning: $temp°C" | mail -s "Hardware Health Alert" admin@example.com
        fi
    }
    
    # Check memory health
    check_memory() {
        if ! memtester 1G 1; then
            echo "Memory test failed" | mail -s "Hardware Health Alert" admin@example.com
        fi
    }
    
    # Check disk health
    check_disks() {
        for disk in $(lsblk -d -o NAME | grep -v NAME); do
            smartctl -H /dev/$disk | grep -q "PASSED"
            if [ $? -ne 0 ]; then
                echo "Disk health warning: /dev/$disk" | mail -s "Hardware Health Alert" admin@example.com
            fi
        done
    }
    
    check_cpu
    check_memory
    check_disks
    echo "Hardware health check completed at $(date)" >> /var/log/hardware_health.log
}

# Call the automated function
AUTO_HARDWARE 

echo "=== Hardware Health Check ==="
echo "CPU Temperature: $(sensors | grep 'Core' | awk '{print $3}')"
echo "Memory Usage: $(free -h | grep Mem | awk '{print $3 "/" $2}')" 