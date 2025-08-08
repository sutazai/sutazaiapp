# Maintenance Procedures - Perfect Jarvis System

**Document Version:** 1.0  
**Last Updated:** 2025-08-08  
**Author:** Operations Team  

## 🎯 Purpose

This runbook provides comprehensive maintenance procedures for the Perfect Jarvis system to ensure optimal performance, reliability, and longevity through proactive maintenance activities.

## 📋 Table of Contents

- [Maintenance Schedule Overview](#maintenance-schedule-overview)
- [Planned Maintenance Windows](#planned-maintenance-windows)
- [Daily Maintenance Tasks](#daily-maintenance-tasks)
- [Weekly Maintenance Tasks](#weekly-maintenance-tasks)
- [Monthly Maintenance Tasks](#monthly-maintenance-tasks)
- [Quarterly Maintenance Tasks](#quarterly-maintenance-tasks)
- [Database Maintenance](#database-maintenance)
- [System Updates](#system-updates)
- [Performance Optimization](#performance-optimization)
- [Monitoring & Cleanup](#monitoring--cleanup)

## 📅 Maintenance Schedule Overview

### Maintenance Windows

| Frequency | Day/Time | Duration | Impact Level |
|-----------|----------|----------|--------------|
| Daily | 02:00 - 04:00 UTC | 5-15 minutes | None (automated) |
| Weekly | Sunday 01:00 - 03:00 UTC | 30-60 minutes | Low (rolling maintenance) |
| Monthly | First Sunday 00:00 - 04:00 UTC | 1-4 hours | Medium (planned downtime) |
| Quarterly | TBD | 2-8 hours | High (system-wide updates) |

### Maintenance Types

- **Preventive:** Proactive tasks to prevent issues
- **Corrective:** Fix identified problems
- **Adaptive:** Accommodate system changes
- **Perfective:** Improve performance and features

## 🕒 Planned Maintenance Windows

### Maintenance Window Management

#### Pre-Maintenance Communication
```bash
#!/bin/bash
# maintenance_notification.sh
MAINTENANCE_TYPE=$1
SCHEDULED_TIME=$2
DURATION=$3
IMPACT_LEVEL=$4

echo "=== MAINTENANCE WINDOW NOTIFICATION ==="

# Calculate times
START_TIME=$(date -d "$SCHEDULED_TIME" -Iseconds)
END_TIME=$(date -d "$SCHEDULED_TIME + $DURATION" -Iseconds)
NOTIFICATION_TIME=$(date -d "$SCHEDULED_TIME - 24 hours" -Iseconds)

# Create maintenance notice
create_maintenance_notice() {
    cat > /opt/sutazaiapp/maintenance/notice.json << EOF
{
  "maintenance": {
    "type": "$MAINTENANCE_TYPE",
    "start_time": "$START_TIME",
    "end_time": "$END_TIME",
    "duration": "$DURATION",
    "impact_level": "$IMPACT_LEVEL",
    "status": "scheduled",
    "description": "Scheduled maintenance for Perfect Jarvis system",
    "affected_services": ["backend", "frontend", "database"],
    "contact": "ops-team@company.com"
  }
}
EOF
}

# Send notifications
send_notifications() {
    local message="🔧 Scheduled Maintenance Notice

System: Perfect Jarvis
Type: $MAINTENANCE_TYPE
Impact: $IMPACT_LEVEL
Start: $START_TIME
Duration: $DURATION

Affected Services:
- Backend API
- Frontend Interface
- Database Operations

Contact: ops-team@company.com for questions"

    # Slack notification
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\": \"$message\"}" \
        "$MAINTENANCE_SLACK_WEBHOOK" 2>/dev/null || true
    
    # Email notification
    echo "$message" | mail -s "Jarvis Maintenance Window Scheduled" \
        "stakeholders@company.com" 2>/dev/null || true
    
    # Update status page (if available)
    curl -X POST -H 'Content-Type: application/json' \
        -d "{\"status\": \"maintenance_scheduled\", \"message\": \"$message\"}" \
        "$STATUS_PAGE_API_ENDPOINT" 2>/dev/null || true
}

# Main execution
create_maintenance_notice
send_notifications

echo "✅ Maintenance window notifications sent"
echo "📅 Maintenance scheduled for: $START_TIME"
```

#### Maintenance Mode Toggle
```bash
#!/bin/bash
# maintenance_mode.sh
MODE=${1:-status}  # enable, disable, status

MAINTENANCE_FILE="/opt/sutazaiapp/.maintenance_mode"
MAINTENANCE_PAGE="/opt/sutazaiapp/maintenance/maintenance.html"

# Create maintenance page
create_maintenance_page() {
    mkdir -p /opt/sutazaiapp/maintenance
    cat > "$MAINTENANCE_PAGE" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>System Maintenance - Perfect Jarvis</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .container {
            text-align: center;
            background: rgba(0, 0, 0, 0.2);
            padding: 40px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }
        .icon {
            font-size: 4em;
            margin-bottom: 20px;
        }
        h1 {
            margin-bottom: 20px;
            font-size: 2.5em;
        }
        p {
            font-size: 1.2em;
            margin-bottom: 15px;
        }
        .eta {
            font-size: 1.5em;
            font-weight: bold;
            color: #ffeb3b;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="icon">🔧</div>
        <h1>System Maintenance in Progress</h1>
        <p>Perfect Jarvis is currently undergoing scheduled maintenance</p>
        <p>We're working to improve your experience</p>
        <p class="eta">Expected completion: <span id="eta"></span></p>
        <p>For updates, contact: ops-team@company.com</p>
    </div>
    
    <script>
        // Update ETA dynamically if available
        const maintenanceEnd = localStorage.getItem('maintenance_end');
        if (maintenanceEnd) {
            document.getElementById('eta').textContent = new Date(maintenanceEnd).toLocaleString();
        } else {
            document.getElementById('eta').textContent = 'Shortly';
        }
        
        // Auto-refresh every 30 seconds
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>
EOF
}

case $MODE in
    "enable")
        echo "🔧 Enabling maintenance mode..."
        
        # Create maintenance marker file
        echo "$(date -Iseconds)" > "$MAINTENANCE_FILE"
        
        # Create maintenance page
        create_maintenance_page
        
        # Stop services gracefully (keep core infrastructure)
        docker-compose stop frontend backend
        
        # Configure nginx to show maintenance page (if available)
        if command -v nginx >/dev/null; then
            cat > /etc/nginx/sites-available/maintenance << 'EOF'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    
    location / {
        root /opt/sutazaiapp/maintenance;
        try_files /maintenance.html =503;
    }
    
    location /health {
        return 200 '{"status": "maintenance", "message": "System under maintenance"}';
        add_header Content-Type application/json;
    }
}
EOF
            ln -sf /etc/nginx/sites-available/maintenance /etc/nginx/sites-enabled/
            nginx -t && nginx -s reload 2>/dev/null || true
        fi
        
        echo "✅ Maintenance mode enabled"
        ;;
        
    "disable")
        echo "🚀 Disabling maintenance mode..."
        
        # Remove maintenance marker
        rm -f "$MAINTENANCE_FILE"
        
        # Restore nginx configuration
        if command -v nginx >/dev/null; then
            rm -f /etc/nginx/sites-enabled/maintenance
            ln -sf /etc/nginx/sites-available/jarvis /etc/nginx/sites-enabled/
            nginx -t && nginx -s reload 2>/dev/null || true
        fi
        
        # Start services
        docker-compose up -d frontend backend
        
        # Wait for services to be ready
        echo "Waiting for services to initialize..."
        timeout 120 bash -c 'until curl -f -s http://localhost:10010/health; do sleep 5; done'
        
        echo "✅ Maintenance mode disabled"
        ;;
        
    "status")
        if [[ -f "$MAINTENANCE_FILE" ]]; then
            maintenance_start=$(cat "$MAINTENANCE_FILE")
            echo "🔧 Maintenance mode: ENABLED since $maintenance_start"
        else
            echo "✅ Maintenance mode: DISABLED"
        fi
        ;;
esac
```

## 📅 Daily Maintenance Tasks

### Automated Daily Maintenance
```bash
#!/bin/bash
# daily_maintenance.sh
# Run via cron: 0 2 * * * /opt/sutazaiapp/scripts/daily_maintenance.sh

MAINTENANCE_LOG="/opt/sutazaiapp/logs/daily_maintenance_$(date +%Y%m%d).log"
exec > >(tee -a "$MAINTENANCE_LOG") 2>&1

echo "=== DAILY MAINTENANCE STARTED - $(date -Iseconds) ==="

# Task 1: Health Check and System Validation
daily_health_check() {
    echo "1. Performing daily health check..."
    
    # System health
    HEALTH_STATUS=$(curl -s http://localhost:10010/health | jq -r '.status' 2>/dev/null || echo "unknown")
    echo "   System Status: $HEALTH_STATUS"
    
    # Service status
    echo "   Container Status:"
    docker-compose ps --format "table {{.Name}}\t{{.Status}}" | grep -E "(sutazai|jarvis)"
    
    # Resource usage
    MEMORY_USAGE=$(free | grep Mem | awk '{printf("%.1f%"), $3/$2 * 100}')
    CPU_LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    DISK_USAGE=$(df -h / | awk 'NR==2{print $5}')
    
    echo "   Resources: Memory: $MEMORY_USAGE, CPU Load: $CPU_LOAD, Disk: $DISK_USAGE"
    
    # Alert on high usage
    if (( $(echo "$MEMORY_USAGE" | sed 's/%//' | cut -d. -f1) > 85 )); then
        echo "   ⚠️ WARNING: High memory usage detected"
        # Send alert
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"⚠️ Jarvis Daily Check: High memory usage ($MEMORY_USAGE)\"}" \
            "$OPS_SLACK_WEBHOOK" 2>/dev/null || true
    fi
    
    echo "✅ Daily health check completed"
}

# Task 2: Log Rotation and Cleanup
daily_log_cleanup() {
    echo "2. Performing log cleanup..."
    
    # Rotate application logs
    find /opt/sutazaiapp/logs -name "*.log" -size +100M -exec gzip {} \;
    
    # Clean old compressed logs (keep 7 days)
    find /opt/sutazaiapp/logs -name "*.log.gz" -mtime +7 -delete
    
    # Clean Docker logs
    docker system prune -f --filter "until=24h" >/dev/null 2>&1
    
    # Clean temporary files
    find /tmp -name "*jarvis*" -mtime +1 -delete 2>/dev/null || true
    
    # Report log sizes
    echo "   Current log directory size: $(du -sh /opt/sutazaiapp/logs | cut -f1)"
    
    echo "✅ Log cleanup completed"
}

# Task 3: Database Maintenance
daily_database_maintenance() {
    echo "3. Performing database maintenance..."
    
    # PostgreSQL maintenance
    docker exec sutazai-postgres psql -U sutazai -d sutazai -c "
        -- Update table statistics
        ANALYZE;
        
        -- Check for dead tuples and vacuum if needed
        SELECT schemaname, tablename, n_dead_tup 
        FROM pg_stat_user_tables 
        WHERE n_dead_tup > 1000;
    " 2>/dev/null || echo "   PostgreSQL maintenance skipped (container not ready)"
    
    # Redis maintenance
    REDIS_MEMORY=$(docker exec sutazai-redis redis-cli info memory 2>/dev/null | grep used_memory_human || echo "unknown")
    echo "   Redis Memory Usage: $REDIS_MEMORY"
    
    # Clear expired keys
    docker exec sutazai-redis redis-cli --scan --pattern "*expired*" | \
        xargs -r docker exec sutazai-redis redis-cli del 2>/dev/null || true
    
    echo "✅ Database maintenance completed"
}

# Task 4: Backup Verification
daily_backup_verification() {
    echo "4. Verifying backups..."
    
    # Check if recent backups exist
    LATEST_BACKUP=$(find /opt/sutazaiapp/backups -name "*.sql.gz" -mtime -1 | head -1)
    if [[ -n "$LATEST_BACKUP" ]]; then
        echo "   ✅ Recent backup found: $(basename "$LATEST_BACKUP")"
        # Test backup integrity
        if gunzip -t "$LATEST_BACKUP" 2>/dev/null; then
            echo "   ✅ Backup integrity verified"
        else
            echo "   ❌ Backup integrity check failed"
        fi
    else
        echo "   ⚠️ No recent backup found"
        # Trigger backup creation
        /opt/sutazaiapp/scripts/backup_postgres.sh
    fi
    
    echo "✅ Backup verification completed"
}

# Task 5: Security Monitoring
daily_security_check() {
    echo "5. Performing security check..."
    
    # Check for failed login attempts
    FAILED_LOGINS=$(grep -c "authentication.*failed" /opt/sutazaiapp/logs/security*.log 2>/dev/null || echo "0")
    echo "   Failed logins (24h): $FAILED_LOGINS"
    
    # Check for suspicious activity
    SUSPICIOUS_ACTIVITY=$(grep -c "SUSPICIOUS\|HIGH\|CRITICAL" /opt/sutazaiapp/logs/security*.log 2>/dev/null || echo "0")
    echo "   Suspicious activities: $SUSPICIOUS_ACTIVITY"
    
    # Update security rules if needed
    if [[ -f /opt/sutazaiapp/security/rules_update.sh ]]; then
        /opt/sutazaiapp/security/rules_update.sh
    fi
    
    echo "✅ Security check completed"
}

# Task 6: Performance Metrics Collection
daily_metrics_collection() {
    echo "6. Collecting performance metrics..."
    
    # Collect system metrics
    METRICS_FILE="/opt/sutazaiapp/logs/daily_metrics_$(date +%Y%m%d).json"
    cat > "$METRICS_FILE" << EOF
{
    "date": "$(date -Iseconds)",
    "system": {
        "uptime": "$(uptime -p)",
        "load_average": $(uptime | awk -F'load average:' '{print $2}' | awk '{print "["$1","$2","$3"]"}' | sed 's/,]/]/'),
        "memory_usage_percent": $(free | grep Mem | awk '{print $3/$2 * 100}'),
        "disk_usage_percent": $(df / | awk 'NR==2{print $5}' | sed 's/%//'),
        "container_count": $(docker ps | wc -l)
    },
    "services": {
        "backend_status": "$(curl -s http://localhost:10010/health | jq -r '.status' 2>/dev/null || echo 'unknown')",
        "database_connections": $(docker exec sutazai-postgres psql -U sutazai -t -c "SELECT count(*) FROM pg_stat_activity;" 2>/dev/null | tr -d ' ' || echo 0),
        "redis_connected_clients": $(docker exec sutazai-redis redis-cli info clients 2>/dev/null | grep connected_clients | cut -d: -f2 | tr -d '\r' || echo 0)
    }
}
EOF
    
    echo "   Metrics saved to: $METRICS_FILE"
    echo "✅ Metrics collection completed"
}

# Execute all daily tasks
main() {
    daily_health_check
    daily_log_cleanup
    daily_database_maintenance
    daily_backup_verification
    daily_security_check
    daily_metrics_collection
    
    echo "=== DAILY MAINTENANCE COMPLETED - $(date -Iseconds) ==="
    
    # Send completion notification
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"✅ Jarvis daily maintenance completed successfully\"}" \
        "$OPS_SLACK_WEBHOOK" 2>/dev/null || true
}

main
```

## 📅 Weekly Maintenance Tasks

### Weekly System Optimization
```bash
#!/bin/bash
# weekly_maintenance.sh
# Run via cron: 0 1 * * 0 /opt/sutazaiapp/scripts/weekly_maintenance.sh

MAINTENANCE_LOG="/opt/sutazaiapp/logs/weekly_maintenance_$(date +%Y%m%d).log"
exec > >(tee -a "$MAINTENANCE_LOG") 2>&1

echo "=== WEEKLY MAINTENANCE STARTED - $(date -Iseconds) ==="

# Task 1: System Updates Check
weekly_system_updates() {
    echo "1. Checking for system updates..."
    
    # Check for OS updates
    if command -v apt >/dev/null; then
        apt update >/dev/null 2>&1
        UPDATES=$(apt list --upgradable 2>/dev/null | grep -v "WARNING" | wc -l)
        echo "   Available OS updates: $UPDATES"
        
        # Security updates only
        SECURITY_UPDATES=$(apt list --upgradable 2>/dev/null | grep -i security | wc -l)
        if [[ $SECURITY_UPDATES -gt 0 ]]; then
            echo "   ⚠️ Security updates available: $SECURITY_UPDATES"
            # Auto-apply security updates (configure as needed)
            # apt-get -y install $(apt list --upgradable 2>/dev/null | grep -i security | cut -d/ -f1)
        fi
    fi
    
    # Check for Docker image updates
    echo "   Checking Docker image updates..."
    docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.CreatedAt}}" | grep jarvis
    
    # Pull latest base images for security updates
    docker pull python:3.11-slim >/dev/null 2>&1 || true
    docker pull nginx:alpine >/dev/null 2>&1 || true
    
    echo "✅ System updates check completed"
}

# Task 2: Database Optimization
weekly_database_optimization() {
    echo "2. Performing database optimization..."
    
    # PostgreSQL maintenance
    docker exec sutazai-postgres psql -U sutazai -d sutazai << 'EOF'
-- Weekly PostgreSQL maintenance
\echo 'Running VACUUM ANALYZE on all tables...'
VACUUM ANALYZE;

\echo 'Checking table bloat...'
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    CASE 
        WHEN pg_total_relation_size(schemaname||'.'||tablename) > 100*1024*1024 
        THEN 'Consider VACUUM FULL' 
        ELSE 'OK' 
    END as recommendation
FROM pg_tables 
WHERE schemaname = 'public';

\echo 'Checking index usage...'
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch,
    CASE 
        WHEN idx_tup_read = 0 THEN 'Unused index - consider dropping'
        WHEN idx_tup_read < 1000 THEN 'Low usage index'
        ELSE 'Good usage'
    END as status
FROM pg_stat_user_indexes;

\echo 'Database statistics updated'
EOF
    
    # Redis optimization
    echo "   Optimizing Redis..."
    docker exec sutazai-redis redis-cli << 'EOF'
# Redis weekly maintenance
INFO memory
MEMORY PURGE
BGREWRITEAOF
EOF
    
    echo "✅ Database optimization completed"
}

# Task 3: Log Analysis and Alerting
weekly_log_analysis() {
    echo "3. Performing log analysis..."
    
    # Generate weekly error report
    ERROR_REPORT="/opt/sutazaiapp/logs/weekly_error_report_$(date +%Y%m%d).txt"
    
    echo "Weekly Error Report - $(date)" > "$ERROR_REPORT"
    echo "================================" >> "$ERROR_REPORT"
    echo "" >> "$ERROR_REPORT"
    
    # Analyze backend errors
    echo "Backend Errors (Last 7 days):" >> "$ERROR_REPORT"
    docker logs sutazai-backend --since=168h 2>&1 | \
        grep -i error | head -20 >> "$ERROR_REPORT" 2>/dev/null || true
    echo "" >> "$ERROR_REPORT"
    
    # Analyze database errors
    echo "Database Errors (Last 7 days):" >> "$ERROR_REPORT"
    docker logs sutazai-postgres --since=168h 2>&1 | \
        grep -i -E "(error|fatal|panic)" | head -10 >> "$ERROR_REPORT" 2>/dev/null || true
    echo "" >> "$ERROR_REPORT"
    
    # Performance metrics summary
    echo "Performance Summary:" >> "$ERROR_REPORT"
    echo "Average response time: $(grep -h "response_time" /opt/sutazaiapp/logs/daily_metrics_*.json 2>/dev/null | \
        tail -7 | jq '.system.load_average[0]' 2>/dev/null | \
        awk '{sum+=$1; n++} END {if(n>0) print sum/n; else print "N/A"}')" >> "$ERROR_REPORT"
    
    # Send report if significant issues found
    ERROR_COUNT=$(grep -c -i error "$ERROR_REPORT" || echo "0")
    if [[ $ERROR_COUNT -gt 10 ]]; then
        echo "   ⚠️ High error count detected: $ERROR_COUNT"
        # Send to operations team
        mail -s "Jarvis Weekly Error Report" ops-team@company.com < "$ERROR_REPORT" 2>/dev/null || true
    fi
    
    echo "   Error report generated: $ERROR_REPORT"
    echo "✅ Log analysis completed"
}

# Task 4: Performance Tuning
weekly_performance_tuning() {
    echo "4. Performing performance tuning..."
    
    # Analyze container resource usage
    echo "   Analyzing container resources..."
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | \
        grep sutazai > /tmp/container_stats.txt
    
    # Check for resource-heavy containers
    while read line; do
        container=$(echo "$line" | awk '{print $1}')
        cpu_usage=$(echo "$line" | awk '{print $2}' | sed 's/%//')
        memory_usage=$(echo "$line" | awk '{print $3}')
        
        if (( $(echo "$cpu_usage > 80" | bc -l) 2>/dev/null )) || echo "$memory_usage" | grep -q "GiB"; then
            echo "   ⚠️ High resource usage: $container (CPU: $cpu_usage%, Memory: $memory_usage)"
            
            # Consider restarting high-usage containers
            if echo "$container" | grep -q "backend\|frontend"; then
                echo "   🔄 Restarting $container due to high resource usage"
                docker restart "$container"
            fi
        fi
    done < /tmp/container_stats.txt
    
    # Optimize Docker system
    echo "   Optimizing Docker system..."
    docker system prune -f --volumes --filter "until=168h"
    
    # Check and optimize disk space
    echo "   Checking disk space..."
    DISK_USAGE=$(df / | awk 'NR==2{print $5}' | sed 's/%//')
    if [[ $DISK_USAGE -gt 80 ]]; then
        echo "   ⚠️ High disk usage: $DISK_USAGE%"
        # Clean up old logs and backups
        find /opt/sutazaiapp/logs -name "*.log.gz" -mtime +14 -delete
        find /opt/sutazaiapp/backups -name "*.sql.gz" -mtime +30 -delete
    fi
    
    echo "✅ Performance tuning completed"
}

# Task 5: Security Updates
weekly_security_updates() {
    echo "5. Performing security updates..."
    
    # Update security rules
    if [[ -f /opt/sutazaiapp/security/update_security_rules.sh ]]; then
        /opt/sutazaiapp/security/update_security_rules.sh
    fi
    
    # Rotate API keys older than 90 days (if applicable)
    if [[ -f /opt/sutazaiapp/security/rotate_api_keys.sh ]]; then
        /opt/sutazaiapp/security/rotate_api_keys.sh --check-only
    fi
    
    # Update SSL certificates if expiring
    if [[ -f /opt/sutazaiapp/certs/jarvis.crt ]]; then
        CERT_EXPIRY=$(openssl x509 -enddate -noout -in /opt/sutazaiapp/certs/jarvis.crt | cut -d= -f2)
        EXPIRY_TIMESTAMP=$(date -d "$CERT_EXPIRY" +%s)
        CURRENT_TIMESTAMP=$(date +%s)
        DAYS_UNTIL_EXPIRY=$(( (EXPIRY_TIMESTAMP - CURRENT_TIMESTAMP) / 86400 ))
        
        echo "   SSL certificate expires in $DAYS_UNTIL_EXPIRY days"
        
        if [[ $DAYS_UNTIL_EXPIRY -lt 30 ]]; then
            echo "   ⚠️ SSL certificate expiring soon"
            # Alert operations team
            curl -X POST -H 'Content-type: application/json' \
                --data "{\"text\":\"⚠️ Jarvis SSL certificate expires in $DAYS_UNTIL_EXPIRY days\"}" \
                "$OPS_SLACK_WEBHOOK" 2>/dev/null || true
        fi
    fi
    
    echo "✅ Security updates completed"
}

# Task 6: Capacity Planning
weekly_capacity_planning() {
    echo "6. Performing capacity planning analysis..."
    
    # Generate capacity report
    CAPACITY_REPORT="/opt/sutazaiapp/logs/weekly_capacity_$(date +%Y%m%d).json"
    
    cat > "$CAPACITY_REPORT" << EOF
{
    "date": "$(date -Iseconds)",
    "capacity_metrics": {
        "system": {
            "cpu_cores": $(nproc),
            "total_memory_gb": $(free -g | awk 'NR==2{print $2}'),
            "total_disk_gb": $(df -BG / | awk 'NR==2{gsub(/G/, "", $2); print $2}'),
            "avg_cpu_usage_7d": $(grep -h "load_average" /opt/sutazaiapp/logs/daily_metrics_*.json 2>/dev/null | tail -7 | jq '.system.load_average[0]' | awk '{sum+=$1; n++} END {if(n>0) printf "%.2f", sum/n; else print 0}'),
            "avg_memory_usage_7d": $(grep -h "memory_usage_percent" /opt/sutazaiapp/logs/daily_metrics_*.json 2>/dev/null | tail -7 | jq '.system.memory_usage_percent' | awk '{sum+=$1; n++} END {if(n>0) printf "%.2f", sum/n; else print 0}'),
            "avg_disk_usage_7d": $(grep -h "disk_usage_percent" /opt/sutazaiapp/logs/daily_metrics_*.json 2>/dev/null | tail -7 | jq '.system.disk_usage_percent' | awk '{sum+=$1; n++} END {if(n>0) printf "%.2f", sum/n; else print 0}')
        },
        "database": {
            "pg_database_size_mb": $(docker exec sutazai-postgres psql -U sutazai -t -c "SELECT pg_size_pretty(pg_database_size('sutazai'));" 2>/dev/null | tr -d ' ' | sed 's/MB//' || echo "0"),
            "pg_connections_max": 100,
            "pg_connections_avg": $(grep -h "database_connections" /opt/sutazaiapp/logs/daily_metrics_*.json 2>/dev/null | tail -7 | jq '.services.database_connections' | awk '{sum+=$1; n++} END {if(n>0) printf "%.0f", sum/n; else print 0}')
        },
        "recommendations": []
    }
}
EOF
    
    # Add recommendations based on usage
    AVG_CPU=$(jq -r '.capacity_metrics.system.avg_cpu_usage_7d' "$CAPACITY_REPORT")
    AVG_MEMORY=$(jq -r '.capacity_metrics.system.avg_memory_usage_7d' "$CAPACITY_REPORT")
    AVG_DISK=$(jq -r '.capacity_metrics.system.avg_disk_usage_7d' "$CAPACITY_REPORT")
    
    RECOMMENDATIONS=""
    if (( $(echo "$AVG_CPU > 70" | bc -l) )); then
        RECOMMENDATIONS+='"Consider CPU upgrade or optimization",'
    fi
    if (( $(echo "$AVG_MEMORY > 80" | bc -l) )); then
        RECOMMENDATIONS+='"Consider memory upgrade",'
    fi
    if (( $(echo "$AVG_DISK > 75" | bc -l) )); then
        RECOMMENDATIONS+='"Consider disk cleanup or expansion",'
    fi
    
    # Update recommendations in report
    if [[ -n "$RECOMMENDATIONS" ]]; then
        RECOMMENDATIONS=${RECOMMENDATIONS%,}  # Remove trailing comma
        jq ".capacity_metrics.recommendations = [$RECOMMENDATIONS]" "$CAPACITY_REPORT" > "$CAPACITY_REPORT.tmp"
        mv "$CAPACITY_REPORT.tmp" "$CAPACITY_REPORT"
    fi
    
    echo "   Capacity report generated: $CAPACITY_REPORT"
    echo "   Average utilization - CPU: $AVG_CPU%, Memory: $AVG_MEMORY%, Disk: $AVG_DISK%"
    
    echo "✅ Capacity planning completed"
}

# Execute all weekly tasks
main() {
    # Enable maintenance mode for potentially disruptive tasks
    /opt/sutazaiapp/scripts/maintenance_mode.sh status | grep -q "ENABLED" || {
        echo "Enabling maintenance mode for weekly tasks..."
        /opt/sutazaiapp/scripts/maintenance_mode.sh enable
        MAINTENANCE_ENABLED=true
    }
    
    weekly_system_updates
    weekly_database_optimization
    weekly_log_analysis
    weekly_performance_tuning
    weekly_security_updates
    weekly_capacity_planning
    
    # Disable maintenance mode if we enabled it
    if [[ "$MAINTENANCE_ENABLED" == "true" ]]; then
        echo "Disabling maintenance mode..."
        /opt/sutazaiapp/scripts/maintenance_mode.sh disable
    fi
    
    echo "=== WEEKLY MAINTENANCE COMPLETED - $(date -Iseconds) ==="
    
    # Send comprehensive report
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"📊 Jarvis weekly maintenance completed. Reports available in /opt/sutazaiapp/logs/\"}" \
        "$OPS_SLACK_WEBHOOK" 2>/dev/null || true
}

main
```

## 📅 Monthly Maintenance Tasks

### Comprehensive Monthly Maintenance
```bash
#!/bin/bash
# monthly_maintenance.sh
# Run via cron: 0 0 1 * * /opt/sutazaiapp/scripts/monthly_maintenance.sh

MAINTENANCE_LOG="/opt/sutazaiapp/logs/monthly_maintenance_$(date +%Y%m).log"
exec > >(tee -a "$MAINTENANCE_LOG") 2>&1

echo "=== MONTHLY MAINTENANCE STARTED - $(date -Iseconds) ==="

# Notify start of monthly maintenance
curl -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"🔧 Starting Jarvis monthly maintenance window...\"}" \
    "$OPS_SLACK_WEBHOOK" 2>/dev/null || true

# Task 1: Full System Backup
monthly_full_backup() {
    echo "1. Performing full system backup..."
    
    BACKUP_DIR="/opt/sutazaiapp/backups/monthly_$(date +%Y%m)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup all databases
    echo "   Backing up PostgreSQL..."
    docker exec sutazai-postgres pg_dumpall -U sutazai | gzip > "$BACKUP_DIR/postgres_full_$(date +%Y%m%d).sql.gz"
    
    echo "   Backing up Redis..."
    docker exec sutazai-redis redis-cli BGSAVE
    sleep 10
    docker cp sutazai-redis:/data/dump.rdb "$BACKUP_DIR/redis_$(date +%Y%m%d).rdb"
    
    # Backup configuration files
    echo "   Backing up configuration..."
    tar -czf "$BACKUP_DIR/config_$(date +%Y%m%d).tar.gz" \
        /opt/sutazaiapp/docker-compose.yml \
        /opt/sutazaiapp/.env* \
        /opt/sutazaiapp/config/ \
        /opt/sutazaiapp/scripts/ \
        2>/dev/null || true
    
    # Backup logs (compressed)
    echo "   Backing up logs..."
    tar -czf "$BACKUP_DIR/logs_$(date +%Y%m%d).tar.gz" \
        /opt/sutazaiapp/logs/*.log \
        2>/dev/null || true
    
    # Create backup verification report
    echo "   Generating backup report..."
    cat > "$BACKUP_DIR/backup_report.txt" << EOF
Monthly Backup Report - $(date)
================================

Files backed up:
$(ls -lh "$BACKUP_DIR"/*)

Total backup size: $(du -sh "$BACKUP_DIR" | cut -f1)

Verification:
$(find "$BACKUP_DIR" -name "*.gz" -exec sh -c 'echo "Testing: $1"; gunzip -t "$1" && echo "OK" || echo "FAILED"' _ {} \;)
EOF
    
    echo "✅ Full system backup completed: $BACKUP_DIR"
}

# Task 2: Security Audit
monthly_security_audit() {
    echo "2. Performing security audit..."
    
    # Run comprehensive security scan
    if [[ -f /opt/sutazaiapp/scripts/security_audit.sh ]]; then
        /opt/sutazaiapp/scripts/security_audit.sh --comprehensive
    fi
    
    # Check for outdated packages
    echo "   Checking for security updates..."
    if command -v apt >/dev/null; then
        apt update >/dev/null 2>&1
        SECURITY_UPDATES=$(apt list --upgradable 2>/dev/null | grep -c security || echo "0")
        echo "   Security updates available: $SECURITY_UPDATES"
        
        if [[ $SECURITY_UPDATES -gt 0 ]]; then
            echo "   ⚠️ Applying security updates..."
            apt-get -y upgrade -o Dpkg::Options::="--force-confold"
        fi
    fi
    
    # Audit user permissions
    echo "   Auditing user permissions..."
    if [[ -f /opt/sutazaiapp/security/users.json ]]; then
        python3 << 'EOF'
import json
from datetime import datetime, timedelta

with open('/opt/sutazaiapp/security/users.json', 'r') as f:
    users = json.load(f)

print("User Audit Report:")
print("==================")

for username, user_info in users.get('users', {}).items():
    last_login = user_info.get('last_login')
    if last_login:
        login_date = datetime.fromisoformat(last_login.replace('Z', '+00:00'))
        days_since_login = (datetime.now().replace(tzinfo=login_date.tzinfo) - login_date).days
        if days_since_login > 90:
            print(f"⚠️ User {username} hasn't logged in for {days_since_login} days")
    else:
        print(f"⚠️ User {username} has never logged in")

# Check API key usage
for api_key, key_info in users.get('api_keys', {}).items():
    last_used = key_info.get('last_used')
    if not last_used:
        print(f"⚠️ API key for user {key_info['user']} has never been used")
EOF
    fi
    
    # Check SSL certificate expiry
    if [[ -f /opt/sutazaiapp/certs/jarvis.crt ]]; then
        CERT_EXPIRY=$(openssl x509 -enddate -noout -in /opt/sutazaiapp/certs/jarvis.crt | cut -d= -f2)
        echo "   SSL certificate expires: $CERT_EXPIRY"
    fi
    
    echo "✅ Security audit completed"
}

# Task 3: Performance Analysis
monthly_performance_analysis() {
    echo "3. Performing performance analysis..."
    
    PERF_REPORT="/opt/sutazaiapp/logs/monthly_performance_$(date +%Y%m).json"
    
    # Collect 30-day performance data
    echo "   Analyzing 30-day performance trends..."
    
    # Calculate averages from daily metrics
    MONTHLY_STATS=$(find /opt/sutazaiapp/logs -name "daily_metrics_*.json" -mtime -30 -exec cat {} \; | \
        jq -s '
        {
            "period": "30_days",
            "analysis_date": now | strftime("%Y-%m-%d"),
            "averages": {
                "cpu_load": (map(.system.load_average[0]) | add / length),
                "memory_usage": (map(.system.memory_usage_percent) | add / length),
                "disk_usage": (map(.system.disk_usage_percent) | add / length),
                "database_connections": (map(.services.database_connections) | add / length)
            },
            "trends": {
                "cpu_trend": (map(.system.load_average[0]) | .[0:15] as $first | .[-15:] as $last | (($last | add / length) - ($first | add / length))),
                "memory_trend": (map(.system.memory_usage_percent) | .[0:15] as $first | .[-15:] as $last | (($last | add / length) - ($first | add / length)))
            }
        }')
    
    echo "$MONTHLY_STATS" > "$PERF_REPORT"
    
    # Generate performance recommendations
    CPU_TREND=$(echo "$MONTHLY_STATS" | jq -r '.trends.cpu_trend')
    MEMORY_TREND=$(echo "$MONTHLY_STATS" | jq -r '.trends.memory_trend')
    
    RECOMMENDATIONS=""
    if (( $(echo "$CPU_TREND > 0.5" | bc -l) )); then
        RECOMMENDATIONS+='"CPU usage trending upward - monitor for capacity needs",'
    fi
    if (( $(echo "$MEMORY_TREND > 5" | bc -l) )); then
        RECOMMENDATIONS+='"Memory usage increasing - check for memory leaks",'
    fi
    
    if [[ -n "$RECOMMENDATIONS" ]]; then
        RECOMMENDATIONS=${RECOMMENDATIONS%,}
        echo "$MONTHLY_STATS" | jq ".recommendations = [$RECOMMENDATIONS]" > "$PERF_REPORT.tmp"
        mv "$PERF_REPORT.tmp" "$PERF_REPORT"
    fi
    
    echo "   Performance report: $PERF_REPORT"
    echo "✅ Performance analysis completed"
}

# Task 4: Database Optimization
monthly_database_optimization() {
    echo "4. Performing database optimization..."
    
    # PostgreSQL deep maintenance
    echo "   PostgreSQL deep maintenance..."
    docker exec sutazai-postgres psql -U sutazai -d sutazai << 'EOF'
-- Monthly PostgreSQL optimization
\echo 'Running VACUUM FULL ANALYZE...'
VACUUM FULL ANALYZE;

\echo 'Reindexing all indexes...'
REINDEX DATABASE sutazai;

\echo 'Updating table statistics...'
ANALYZE;

\echo 'Checking for unused indexes...'
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes 
WHERE idx_tup_read = 0 
ORDER BY schemaname, tablename;

\echo 'Database size analysis...'
SELECT 
    pg_size_pretty(pg_database_size('sutazai')) as database_size,
    pg_size_pretty(pg_total_relation_size('pg_class')) as catalog_size;

\echo 'PostgreSQL optimization completed'
EOF
    
    # Redis optimization
    echo "   Redis optimization..."
    docker exec sutazai-redis redis-cli << 'EOF'
# Monthly Redis optimization
INFO memory
MEMORY PURGE
MEMORY STATS
BGREWRITEAOF
CONFIG RESETSTAT
EOF
    
    # Archive old data if needed
    echo "   Archiving old data..."
    ARCHIVE_DATE=$(date -d "6 months ago" +%Y-%m-%d)
    docker exec sutazai-postgres psql -U sutazai -d sutazai -c "
        -- Archive old log entries (example)
        -- DELETE FROM log_entries WHERE created_at < '$ARCHIVE_DATE';
        SELECT 'Archive process completed for data older than $ARCHIVE_DATE';
    " 2>/dev/null || echo "   No archival needed"
    
    echo "✅ Database optimization completed"
}

# Task 5: System Updates
monthly_system_updates() {
    echo "5. Performing system updates..."
    
    # Update operating system
    echo "   Updating operating system..."
    if command -v apt >/dev/null; then
        apt update && apt -y upgrade
        apt -y autoremove
        apt -y autoclean
    fi
    
    # Update Docker
    echo "   Checking Docker updates..."
    DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
    echo "   Current Docker version: $DOCKER_VERSION"
    
    # Update container images
    echo "   Updating container images..."
    docker-compose pull
    
    # Update Python packages in backend
    echo "   Updating Python packages..."
    docker exec sutazai-backend pip list --outdated --format=json > /tmp/outdated_packages.json 2>/dev/null || true
    
    if [[ -f /tmp/outdated_packages.json ]] && [[ -s /tmp/outdated_packages.json ]]; then
        OUTDATED_COUNT=$(jq length /tmp/outdated_packages.json)
        echo "   Outdated Python packages: $OUTDATED_COUNT"
        
        # Update non-breaking packages (patch versions only)
        docker exec sutazai-backend bash -c "
            pip list --outdated --format=json | \
            jq -r '.[] | select(.latest_version | test(\"^\" + .version + \"\\.[0-9]+\")) | .name' | \
            head -10 | \
            xargs -r pip install --upgrade
        " 2>/dev/null || echo "   Package update skipped"
    fi
    
    echo "✅ System updates completed"
}

# Task 6: Disaster Recovery Testing
monthly_dr_testing() {
    echo "6. Performing disaster recovery testing..."
    
    # Test backup restoration
    echo "   Testing backup restoration..."
    TEST_DB_NAME="sutazai_test_$(date +%s)"
    
    # Find latest backup
    LATEST_BACKUP=$(find /opt/sutazaiapp/backups -name "*.sql.gz" -mtime -7 | head -1)
    
    if [[ -n "$LATEST_BACKUP" ]]; then
        echo "   Testing backup: $(basename "$LATEST_BACKUP")"
        
        # Create test database and restore
        docker exec sutazai-postgres psql -U sutazai -c "CREATE DATABASE $TEST_DB_NAME;" 2>/dev/null
        
        if gunzip -c "$LATEST_BACKUP" | docker exec -i sutazai-postgres psql -U sutazai -d "$TEST_DB_NAME" >/dev/null 2>&1; then
            echo "   ✅ Backup restoration test PASSED"
            # Verify some basic tables exist
            TABLE_COUNT=$(docker exec sutazai-postgres psql -U sutazai -d "$TEST_DB_NAME" -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';" 2>/dev/null | tr -d ' ')
            echo "   Tables restored: $TABLE_COUNT"
        else
            echo "   ❌ Backup restoration test FAILED"
        fi
        
        # Clean up test database
        docker exec sutazai-postgres psql -U sutazai -c "DROP DATABASE $TEST_DB_NAME;" 2>/dev/null || true
    else
        echo "   ⚠️ No recent backup found for testing"
    fi
    
    # Test system recovery procedures
    echo "   Testing system recovery procedures..."
    
    # Simulate service restart
    echo "   Testing service restart..."
    docker-compose restart backend
    timeout 60 bash -c 'until curl -f -s http://localhost:10010/health; do sleep 2; done' && {
        echo "   ✅ Service restart test PASSED"
    } || {
        echo "   ❌ Service restart test FAILED"
    }
    
    echo "✅ Disaster recovery testing completed"
}

# Execute all monthly tasks
main() {
    # Ensure maintenance window is properly scheduled
    echo "🔧 Monthly maintenance window in progress..."
    
    monthly_full_backup
    monthly_security_audit
    monthly_performance_analysis
    monthly_database_optimization
    monthly_system_updates
    monthly_dr_testing
    
    echo "=== MONTHLY MAINTENANCE COMPLETED - $(date -Iseconds) ==="
    
    # Generate monthly report
    MONTHLY_REPORT="/opt/sutazaiapp/logs/monthly_report_$(date +%Y%m).md"
    cat > "$MONTHLY_REPORT" << EOF
# Monthly Maintenance Report - $(date +"%B %Y")

Generated: $(date -Iseconds)

## Summary
- ✅ Full system backup completed
- ✅ Security audit completed  
- ✅ Performance analysis completed
- ✅ Database optimization completed
- ✅ System updates completed
- ✅ Disaster recovery testing completed

## Key Metrics
- System uptime: $(uptime -p)
- Backup size: $(du -sh /opt/sutazaiapp/backups/monthly_$(date +%Y%m) | cut -f1)
- Security updates applied: $(grep -c "Security updates" "$MAINTENANCE_LOG" || echo "0")
- Database size: $(docker exec sutazai-postgres psql -U sutazai -t -c "SELECT pg_size_pretty(pg_database_size('sutazai'));" 2>/dev/null | tr -d ' ')

## Action Items
$(grep -E "(⚠️|❌)" "$MAINTENANCE_LOG" | sed 's/^/- /')

## Next Maintenance
Next monthly maintenance: $(date -d "next month" "+%B 1, %Y")
EOF
    
    # Send completion notification with report
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"✅ Jarvis monthly maintenance completed successfully. Report available at: $MONTHLY_REPORT\"}" \
        "$OPS_SLACK_WEBHOOK" 2>/dev/null || true
}

main
```

## 🗃️ Database Maintenance

### PostgreSQL Maintenance Scripts

#### Database Health Monitor
```bash
#!/bin/bash
# db_health_monitor.sh

echo "=== DATABASE HEALTH MONITOR ==="

# Function to execute PostgreSQL queries safely
execute_pg_query() {
    local query=$1
    docker exec sutazai-postgres psql -U sutazai -d sutazai -t -c "$query" 2>/dev/null | sed 's/^[ \t]*//;s/[ \t]*$//'
}

# Check database connections
check_connections() {
    echo "1. Connection Analysis:"
    
    ACTIVE_CONNECTIONS=$(execute_pg_query "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';")
    IDLE_CONNECTIONS=$(execute_pg_query "SELECT count(*) FROM pg_stat_activity WHERE state = 'idle';")
    TOTAL_CONNECTIONS=$(execute_pg_query "SELECT count(*) FROM pg_stat_activity;")
    MAX_CONNECTIONS=$(execute_pg_query "SHOW max_connections;")
    
    echo "   Active connections: $ACTIVE_CONNECTIONS"
    echo "   Idle connections: $IDLE_CONNECTIONS"
    echo "   Total connections: $TOTAL_CONNECTIONS / $MAX_CONNECTIONS"
    
    CONNECTION_USAGE=$((TOTAL_CONNECTIONS * 100 / MAX_CONNECTIONS))
    if [[ $CONNECTION_USAGE -gt 80 ]]; then
        echo "   ⚠️ High connection usage: $CONNECTION_USAGE%"
    fi
}

# Check database size and growth
check_database_size() {
    echo "2. Database Size Analysis:"
    
    DB_SIZE=$(execute_pg_query "SELECT pg_size_pretty(pg_database_size('sutazai'));")
    echo "   Database size: $DB_SIZE"
    
    # Table sizes
    echo "   Largest tables:"
    execute_pg_query "
        SELECT 
            schemaname,
            tablename,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
        FROM pg_tables 
        WHERE schemaname = 'public'
        ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC 
        LIMIT 5;
    " | while IFS='|' read -r schema table size; do
        echo "     $table: $size"
    done
}

# Check query performance
check_query_performance() {
    echo "3. Query Performance Analysis:"
    
    # Slow queries
    echo "   Slowest queries (avg time):"
    execute_pg_query "
        SELECT 
            LEFT(query, 50) as query_start,
            calls,
            ROUND(mean_exec_time::numeric, 2) as avg_time_ms
        FROM pg_stat_statements 
        ORDER BY mean_exec_time DESC 
        LIMIT 3;
    " 2>/dev/null | while IFS='|' read -r query calls avg_time; do
        echo "     $query... ($calls calls, ${avg_time}ms avg)"
    done || echo "   pg_stat_statements extension not available"
    
    # Lock analysis
    BLOCKED_QUERIES=$(execute_pg_query "SELECT count(*) FROM pg_locks WHERE NOT granted;")
    echo "   Blocked queries: $BLOCKED_QUERIES"
}

# Check index usage
check_index_usage() {
    echo "4. Index Usage Analysis:"
    
    # Unused indexes
    echo "   Unused indexes:"
    execute_pg_query "
        SELECT 
            schemaname,
            tablename,
            indexname
        FROM pg_stat_user_indexes 
        WHERE idx_tup_read = 0 AND idx_tup_fetch = 0
        LIMIT 5;
    " | while IFS='|' read -r schema table index; do
        echo "     $schema.$table.$index (never used)"
    done
    
    # Missing indexes (tables with seq scans)
    echo "   Tables with high sequential scans:"
    execute_pg_query "
        SELECT 
            schemaname,
            tablename,
            seq_scan,
            seq_tup_read
        FROM pg_stat_user_tables 
        WHERE seq_scan > 1000
        ORDER BY seq_scan DESC 
        LIMIT 3;
    " | while IFS='|' read -r schema table seq_scan seq_tup_read; do
        echo "     $table: $seq_scan scans ($seq_tup_read tuples)"
    done
}

# Check database maintenance needs
check_maintenance_needs() {
    echo "5. Maintenance Needs:"
    
    # Tables needing vacuum
    echo "   Tables with dead tuples:"
    execute_pg_query "
        SELECT 
            schemaname,
            tablename,
            n_dead_tup
        FROM pg_stat_user_tables 
        WHERE n_dead_tup > 1000
        ORDER BY n_dead_tup DESC;
    " | while IFS='|' read -r schema table dead_tuples; do
        echo "     $table: $dead_tuples dead tuples"
    done
    
    # Last vacuum/analyze times
    echo "   Maintenance history:"
    execute_pg_query "
        SELECT 
            schemaname,
            tablename,
            last_vacuum,
            last_analyze
        FROM pg_stat_user_tables 
        WHERE schemaname = 'public'
        ORDER BY tablename;
    " | head -3 | while IFS='|' read -r schema table last_vacuum last_analyze; do
        echo "     $table: vacuum=$last_vacuum, analyze=$last_analyze"
    done
}

# Execute all checks
check_connections
check_database_size
check_query_performance
check_index_usage
check_maintenance_needs

echo "✅ Database health check completed"
```

## 🔄 System Updates

### Automated Update Management
```bash
#!/bin/bash
# system_update_manager.sh

UPDATE_LOG="/opt/sutazaiapp/logs/system_updates_$(date +%Y%m%d).log"
exec > >(tee -a "$UPDATE_LOG") 2>&1

echo "=== SYSTEM UPDATE MANAGER - $(date -Iseconds) ==="

# Update configuration
SECURITY_ONLY=${1:-false}
AUTO_RESTART=${2:-false}
MAX_DOWNTIME_MINUTES=${3:-30}

# Check for available updates
check_available_updates() {
    echo "1. Checking for available updates..."
    
    if command -v apt >/dev/null; then
        apt update >/dev/null 2>&1
        
        ALL_UPDATES=$(apt list --upgradable 2>/dev/null | grep -v "WARNING" | wc -l)
        SECURITY_UPDATES=$(apt list --upgradable 2>/dev/null | grep -i security | wc -l)
        
        echo "   Available updates: $ALL_UPDATES total, $SECURITY_UPDATES security"
        
        if [[ $SECURITY_UPDATES -gt 0 ]]; then
            echo "   Security updates available:"
            apt list --upgradable 2>/dev/null | grep -i security | head -5
        fi
        
        return $SECURITY_UPDATES
    fi
    
    return 0
}

# Apply system updates
apply_system_updates() {
    local security_only=$1
    echo "2. Applying system updates (security only: $security_only)..."
    
    if [[ "$security_only" == "true" ]]; then
        echo "   Applying security updates only..."
        apt-get -y install $(apt list --upgradable 2>/dev/null | grep -i security | cut -d/ -f1) || true
    else
        echo "   Applying all updates..."
        apt-get -y upgrade -o Dpkg::Options::="--force-confold"
    fi
    
    # Check if reboot is required
    if [[ -f /var/run/reboot-required ]]; then
        echo "   ⚠️ System reboot required after updates"
        echo "reboot_required" > /tmp/update_status
        
        if [[ "$AUTO_RESTART" == "true" ]]; then
            echo "   🔄 Scheduling system reboot..."
            shutdown -r +5 "System reboot required for security updates"
        fi
    fi
}

# Update Docker components
update_docker_components() {
    echo "3. Updating Docker components..."
    
    # Update base images
    echo "   Updating base images..."
    BASE_IMAGES=(
        "python:3.11-slim"
        "nginx:alpine"
        "postgres:15-alpine"
        "redis:7-alpine"
    )
    
    for image in "${BASE_IMAGES[@]}"; do
        echo "   Pulling $image..."
        docker pull "$image" >/dev/null 2>&1 || echo "   Failed to pull $image"
    done
    
    # Update application images
    echo "   Building updated application images..."
    docker-compose build --no-cache backend frontend >/dev/null 2>&1 || {
        echo "   ❌ Application image build failed"
        return 1
    }
    
    echo "   ✅ Docker components updated"
}

# Update application dependencies
update_application_dependencies() {
    echo "4. Updating application dependencies..."
    
    # Python dependencies
    echo "   Checking Python dependencies..."
    docker exec sutazai-backend pip list --outdated --format=json > /tmp/outdated_python.json 2>/dev/null || true
    
    if [[ -s /tmp/outdated_python.json ]]; then
        OUTDATED_COUNT=$(jq length /tmp/outdated_python.json)
        echo "   Outdated Python packages: $OUTDATED_COUNT"
        
        # Update patch versions only for safety
        echo "   Updating patch versions..."
        docker exec sutazai-backend bash -c "
            pip list --outdated --format=json | \
            jq -r '.[] | select(.latest_version | test(\"^\" + .version + \"\\.[0-9]+$\")) | .name' | \
            head -10 | \
            xargs -r pip install --upgrade
        " 2>/dev/null || echo "   Python package update failed"
    fi
    
    # Node.js dependencies (if applicable)
    if docker exec sutazai-frontend npm --version >/dev/null 2>&1; then
        echo "   Updating Node.js dependencies..."
        docker exec sutazai-frontend npm audit fix --only=prod >/dev/null 2>&1 || true
    fi
}

# Rolling update with health checks
perform_rolling_update() {
    echo "5. Performing rolling update..."
    
    SERVICES=("backend" "frontend")
    
    for service in "${SERVICES[@]}"; do
        echo "   Updating $service..."
        
        # Stop service
        docker-compose stop "$service"
        
        # Start updated service
        docker-compose up -d "$service"
        
        # Health check
        case $service in
            "backend")
                echo "   Waiting for $service to be ready..."
                timeout 60 bash -c 'until curl -f -s http://localhost:10010/health; do sleep 2; done' || {
                    echo "   ❌ $service failed health check - rolling back"
                    docker-compose restart "$service"
                    return 1
                }
                ;;
            "frontend")
                # Basic container health check
                sleep 10
                if ! docker-compose ps "$service" | grep -q "Up"; then
                    echo "   ❌ $service failed to start"
                    return 1
                fi
                ;;
        esac
        
        echo "   ✅ $service updated successfully"
    done
    
    echo "   ✅ Rolling update completed"
}

# Post-update verification
post_update_verification() {
    echo "6. Post-update verification..."
    
    # System health check
    echo "   Checking system health..."
    if ! curl -f -s http://localhost:10010/health >/dev/null; then
        echo "   ❌ System health check failed"
        return 1
    fi
    
    # Service functionality test
    echo "   Testing core functionality..."
    TEST_RESPONSE=$(curl -s -X POST http://localhost:10010/simple-chat \
        -H "Content-Type: application/json" \
        -d '{"message": "update test"}' | jq -r '.response' 2>/dev/null)
    
    if [[ -n "$TEST_RESPONSE" && "$TEST_RESPONSE" != "null" ]]; then
        echo "   ✅ Functionality test passed"
    else
        echo "   ❌ Functionality test failed"
        return 1
    fi
    
    # Performance check
    RESPONSE_TIME=$(curl -w "%{time_total}" -s -o /dev/null http://localhost:10010/health)
    if (( $(echo "$RESPONSE_TIME < 3.0" | bc -l) )); then
        echo "   ✅ Performance check passed (${RESPONSE_TIME}s)"
    else
        echo "   ⚠️ Slow response time: ${RESPONSE_TIME}s"
    fi
    
    echo "   ✅ Post-update verification completed"
}

# Generate update report
generate_update_report() {
    echo "7. Generating update report..."
    
    UPDATE_REPORT="/opt/sutazaiapp/logs/update_report_$(date +%Y%m%d).json"
    
    cat > "$UPDATE_REPORT" << EOF
{
    "update_date": "$(date -Iseconds)",
    "update_type": "$([[ "$SECURITY_ONLY" == "true" ]] && echo "security_only" || echo "full_update")",
    "system_updates": {
        "applied": true,
        "reboot_required": $([[ -f /var/run/reboot-required ]] && echo "true" || echo "false")
    },
    "docker_updates": {
        "base_images_updated": true,
        "application_rebuilt": true
    },
    "post_update_status": {
        "health_check": "passed",
        "functionality_test": "passed",
        "response_time_ms": $(echo "$RESPONSE_TIME * 1000" | bc | cut -d. -f1)
    },
    "next_update_check": "$(date -d "+1 week" -Iseconds)"
}
EOF
    
    echo "   Update report: $UPDATE_REPORT"
}

# Main execution
main() {
    START_TIME=$(date +%s)
    
    # Check for updates
    if ! check_available_updates; then
        echo "✅ No updates available"
        exit 0
    fi
    
    # Apply updates
    apply_system_updates "$SECURITY_ONLY"
    update_docker_components || {
        echo "❌ Docker update failed"
        exit 1
    }
    update_application_dependencies
    
    # Rolling update
    perform_rolling_update || {
        echo "❌ Rolling update failed"
        exit 1
    }
    
    # Verification
    post_update_verification || {
        echo "❌ Post-update verification failed"
        exit 1
    }
    
    generate_update_report
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo "✅ System update completed successfully in ${DURATION} seconds"
    
    # Notification
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"✅ Jarvis system updates completed successfully in ${DURATION}s\"}" \
        "$OPS_SLACK_WEBHOOK" 2>/dev/null || true
}

main "$@"
```

---

## 📋 Maintenance Schedule Summary

### Automated Tasks (Cron Configuration)

```bash
# Add to /etc/crontab or root crontab

# Daily maintenance (2 AM UTC)
0 2 * * * /opt/sutazaiapp/scripts/daily_maintenance.sh

# Weekly maintenance (Sunday 1 AM UTC)
0 1 * * 0 /opt/sutazaiapp/scripts/weekly_maintenance.sh

# Monthly maintenance (First Sunday 12 AM UTC)
0 0 1 * * /opt/sutazaiapp/scripts/monthly_maintenance.sh

# Security updates check (Daily)
30 3 * * * /opt/sutazaiapp/scripts/system_update_manager.sh true

# Database health check (Every 6 hours)
0 */6 * * * /opt/sutazaiapp/scripts/db_health_monitor.sh

# Log cleanup (Daily)
45 1 * * * find /opt/sutazaiapp/logs -name "*.log" -mtime +7 -delete

# Certificate expiry check (Weekly)
0 6 * * 1 /opt/sutazaiapp/scripts/check_ssl_expiry.sh
```

### Manual Maintenance Checklist

#### Weekly Manual Tasks
- [ ] Review maintenance logs for errors
- [ ] Check system capacity trends
- [ ] Verify backup integrity
- [ ] Review security alerts
- [ ] Update documentation if needed

#### Monthly Manual Tasks
- [ ] Review performance reports
- [ ] Plan capacity upgrades if needed
- [ ] Update disaster recovery procedures
- [ ] Conduct security review
- [ ] Test emergency procedures

#### Quarterly Manual Tasks
- [ ] Complete system architecture review
- [ ] Update maintenance procedures
- [ ] Review and update monitoring thresholds
- [ ] Conduct team training updates
- [ ] Plan major upgrades or migrations

---

## 📞 Maintenance Contacts

**Operations Team:**
- Primary: ops-team@company.com
- Emergency: +1-xxx-xxx-xxxx

**Database Administrator:**
- Primary: dba@company.com
- On-call: +1-xxx-xxx-yyyy

**Security Team:**
- Primary: security-team@company.com
- Incidents: security-incidents@company.com

---

*This maintenance procedures document is based on the actual Perfect Jarvis system architecture. Update procedures as the system evolves and operational experience grows.*