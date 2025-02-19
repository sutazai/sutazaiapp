#!/bin/bash

# Ultra comprehensive monitoring system
services=("ai_service" "super_agent" "sutazai")
ALERT_THRESHOLDS=(
    "CPU:80"
    "MEMORY:80"
    "DISK:90"
    "NETWORK:1000"  # MB/s
)

# Automated monitoring with anomaly detection
MONITORING_INTERVAL=60
ANOMALY_THRESHOLD=3

# Add log rotation logic
LOG_DIR="/var/log/sutazai"
LOG_FILE="${LOG_DIR}/monitoring.log"

setup_log_rotation() {
    echo "🔄 Setting up log rotation..."
    cat <<EOL | sudo tee /etc/logrotate.d/sutazai
${LOG_FILE} {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
EOL
    echo "✅ Log rotation configured"
}

start_monitoring() {
    while true; do
        collect_metrics
        detect_anomalies
        if [ $ANOMALY_COUNT -gt $ANOMALY_THRESHOLD ]; then
            trigger_incident_response
        fi
        sleep $MONITORING_INTERVAL
    done
}

start_monitoring &

while true; do
    for service in "${services[@]}"; do
        # Get comprehensive metrics
        stats=$(docker stats --no-stream --format "{{.CPUPerc}},{{.MemPerc}},{{.NetIO}},{{.BlockIO}}" $service)
        IFS=',' read -r cpu mem netio blockio <<< "$stats"
        
        # Check all thresholds
        for threshold in "${ALERT_THRESHOLDS[@]}"; do
            IFS=':' read -r metric value <<< "$threshold"
            case $metric in
                CPU)
                    if (( $(echo "${cpu%\%} > $value" | bc -l) )); then
                        ./alert.sh "High CPU: $service at $cpu"
                    fi
                    ;;
                MEMORY)
                    if (( $(echo "${mem%\%} > $value" | bc -l) )); then
                        ./alert.sh "High memory: $service at $mem"
                    fi
                    ;;
                DISK)
                    disk_usage=$(df -h / | awk 'NR==2 {print $5}' | tr -d '%')
                    if (( $disk_usage > $value )); then
                        ./alert.sh "High disk usage: $disk_usage%"
                    fi
                    ;;
                NETWORK)
                    net_mb=$(echo $netio | awk '{print $1}' | sed 's/MB$//')
                    if (( $(echo "$net_mb > $value" | bc -l) )); then
                        ./alert.sh "High network usage: $net_mb MB/s"
                    fi
                    ;;
            esac
        done
        
        # Check container health
        health=$(docker inspect --format '{{.State.Health.Status}}' $service)
        if [ "$health" != "healthy" ]; then
            ./alert.sh "Unhealthy: $service is $health"
            docker-compose restart $service
        fi
    done
    
    # Check system health
    CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')
    MEM=$(free -m | awk '/Mem:/ {print $3/$2 * 100.0}')
    DISK=$(df -h / | awk '/\// {print $5}' | tr -d '%')
    
    # Send alerts if thresholds are exceeded
    if (( $(echo "$CPU > 90" | bc -l) )); then
        echo "CRITICAL: High CPU usage detected: $CPU%" | mail -s "System Alert" admin@example.com
    fi
    
    if (( $(echo "$MEM > 90" | bc -l) )); then
        echo "CRITICAL: High memory usage detected: $MEM%" | mail -s "System Alert" admin@example.com
    fi
    
    if (( $DISK > 90 )); then
        echo "CRITICAL: High disk usage detected: $DISK%" | mail -s "System Alert" admin@example.com
    fi
    
    # Log metrics
    echo "$(date),$CPU,$MEM,$DISK" >> /var/log/system_metrics.log
    
    # Optimize resources if needed
    if (( $(echo "$CPU > 80" | bc -l) )); then
        docker-compose scale super_ai=+1
    elif (( $(echo "$CPU < 20" | bc -l) )); then
        docker-compose scale super_ai=-1
    fi
    
    sleep 300
done

setup_log_rotation

# Add incident reporting
report_incident() {
    INCIDENT=$1
    curl -X POST -H "Content-Type: application/json" -d "{\"incident\": \"$INCIDENT\"}" http://monitoring-system/api/incidents
    echo "📢 Incident reported: $INCIDENT"
}

# Add incident response logic
trigger_incident_response() {
    echo "🚨 Triggering incident response..."
    report_incident "High CPU usage detected"
    echo "✅ Incident response completed"
}

# Add system health reporting
generate_health_report() {
    echo "📊 Generating system health report..."
    HEALTH_REPORT=$(docker stats --no-stream --format "{{.Name}},{{.CPUPerc}},{{.MemPerc}}")
    curl -X POST -H "Content-Type: application/json" -d "{\"report\": \"$HEALTH_REPORT\"}" http://monitoring-system/api/reports
    echo "✅ Health report generated"
}

generate_health_report

# Automated monitoring setup
echo "📊 Starting monitoring services..."

# Start Prometheus and Grafana
docker-compose -f docker-compose.yml up -d prometheus grafana

# Configure alerts
echo "Configuring alert thresholds..."
curl -X POST http://localhost:9090/-/reload

echo "Monitoring system started successfully!"

# Enhanced monitoring with auto-alerts
set -e

# Start resource monitoring
nohup ./monitor_resources.sh > /var/log/resource_monitor.log &

# Start AI performance monitoring
nohup ./monitor_ai_performance.sh > /var/log/ai_performance.log &

# Set up alerting system
./setup_alerts.sh

echo "Monitoring system fully automated and running!" 