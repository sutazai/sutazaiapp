#!/bin/bash
# Send system alerts based on thresholds
CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')
if (( $(echo "$CPU > 90" | bc -l) )); then
    ./alert.sh "High CPU usage: $CPU%"
fi 