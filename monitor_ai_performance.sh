#!/bin/bash
# AI-specific performance monitoring

while true; do
    # Get AI service metrics
    AI_CPU=$(docker stats --no-stream --format "{{.CPUPerc}}" ai_service | tr -d '%')
    AI_MEM=$(docker stats --no-stream --format "{{.MemPerc}}" ai_service | tr -d '%')
    AI_LATENCY=$(curl -s -o /dev/null -w "%{time_total}" http://localhost:8000/health)
    
    # Log metrics
    echo "$(date),$AI_CPU,$AI_MEM,$AI_LATENCY" >> /var/log/ai_metrics.log
    
    # Check thresholds
    if (( $(echo "$AI_CPU > 90" | bc -l) )); then
        ./alert.sh "High AI CPU: $AI_CPU%"
    fi
    
    if (( $(echo "$AI_MEM > 90" | bc -l) )); then
        ./alert.sh "High AI memory: $AI_MEM%"
    fi
    
    if (( $(echo "$AI_LATENCY > 1" | bc -l) )); then
        ./alert.sh "High AI latency: $AI_LATENCY s"
    fi
    
    sleep 5
done 