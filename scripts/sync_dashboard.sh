#!/bin/bash

setup_sync_dashboard() {
    # Create Grafana dashboard
    local dashboard=$(cat <<EOF
{
    "dashboard": {
        "title": "Project Sync Monitoring",
        "panels": [
            {
                "type": "graph",
                "title": "Sync Status",
                "targets": [
                    {
                        "expr": "up{job=~\"code-sync|auto-detection-engine|resource-monitor\"}"
                    }
                ]
            }
        ]
    }
}
EOF
    )
    
    curl -X POST -H "Content-Type: application/json" -d "$dashboard" \
        http://admin:admin@localhost:3000/api/dashboards/db
} 