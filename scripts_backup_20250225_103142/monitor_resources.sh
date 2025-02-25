#!/bin/bash
# Comprehensive resource monitoring

while true; do
    # Get system metrics
    CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')
    MEM=$(free -m | awk '/Mem:/ {print $3/$2 * 100.0}')
    DISK=$(df -h / | awk '/\// {print $5}' | tr -d '%')
    NET=$(ifstat -i eth0 1 1 | awk 'NR==3 {print $1}')
    
    # Log metrics
    echo "$(date),$CPU,$MEM,$DISK,$NET" >> /var/log/resource_metrics.log
    
    # Check thresholds
    if (( $(echo "$CPU > 90" | bc -l) )); then
        ./alert.sh "High CPU: $CPU%"
    fi
    
    if (( $(echo "$MEM > 90" | bc -l) )); then
        ./alert.sh "High memory: $MEM%"
    fi
    
    if (( $DISK > 90 )); then
        ./alert.sh "High disk: $DISK%"
    fi
    
    if (( $(echo "$NET > 1000" | bc -l) )); then
        ./alert.sh "High network: $NET MB/s"
    fi
    
    sleep 5
done 