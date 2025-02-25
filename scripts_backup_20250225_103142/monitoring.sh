#!/bin/bash
setup_monitoring() {
  # Prometheus Configuration
  cat > /etc/prometheus/prometheus.yml <<EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ai_system'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['localhost:8001', 'localhost:8000']
        
  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
EOF

  # Grafana Provisioning
  docker run -d --name=grafana -p 3000:3000 grafana/grafana-enterprise
  curl -X POST -H "Content-Type: application/json" -d @monitoring/dashboards.json http://admin:admin@localhost:3000/api/dashboards/import
}

configure_document_monitoring() {
    grafana-cli --as-admin create-document-dashboard \
        --title "Document Processing Metrics" \
        --metrics latency,throughput,error_rate
}

setup_self_improvement_monitoring() {
    grafana-cli --as-admin create-dashboard \
        -n "Autonomous Improvements" \
        -m improvement_success_rate,rollback_events \
        -p /etc/grafana/provisioning/dashboards/self_improvement.json
} 