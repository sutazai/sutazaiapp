#!/bin/bash

# Automated log analysis with anomaly detection
AUTO_ANALYZE() {
    echo "Starting automated log analysis..."
    
    # Analyze system logs
    analyze_syslog() {
        # Detect critical errors
        grep -i -E '(error|critical|fail|fatal)' /var/log/syslog | sort | uniq -c | sort -nr
    }
    
    # Analyze auth logs
    analyze_authlog() {
        # Detect failed login attempts
        grep -i 'failed' /var/log/auth.log | awk '{print $1,$2,$3,$9}' | sort | uniq -c | sort -nr
    }
    
    # Generate report
    generate_report() {
        echo "Log Analysis Report - $(date)" > /var/log/log_analysis.log
        echo "System Log Analysis:" >> /var/log/log_analysis.log
        analyze_syslog >> /var/log/log_analysis.log
        echo "Auth Log Analysis:" >> /var/log/log_analysis.log
        analyze_authlog >> /var/log/log_analysis.log
        mail -s "Log Analysis Report" admin@example.com < /var/log/log_analysis.log
    }
    
    generate_report
    echo "Log analysis completed at $(date)"
}

echo "=== Log Analysis ==="
echo "Top 5 Most Frequent Log Messages:"
journalctl --no-pager | awk '{print $5}' | sort | uniq -c | sort -nr | head -5