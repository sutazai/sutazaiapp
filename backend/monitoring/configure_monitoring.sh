#!/bin/bash

# SutazAI AGI Monitoring Configuration Script
# This script configures all components of the SutazAI AGI monitoring system

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
BASE_DIR="/opt/sutazaiapp"
LOG_DIR="${BASE_DIR}/logs"

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p "${LOG_DIR}"
mkdir -p "${LOG_DIR}/neural"
mkdir -p "${LOG_DIR}/ethics"
mkdir -p "${LOG_DIR}/security"
mkdir -p "${LOG_DIR}/hardware"
mkdir -p "${LOG_DIR}/self_mod"

mkdir -p "${SCRIPT_DIR}/prometheus"
mkdir -p "${SCRIPT_DIR}/grafana/provisioning/dashboards"
mkdir -p "${SCRIPT_DIR}/grafana/provisioning/datasources"
mkdir -p "${SCRIPT_DIR}/elk/logstash/config"
mkdir -p "${SCRIPT_DIR}/elk/logstash/pipeline"
mkdir -p "${SCRIPT_DIR}/elk/filebeat"

echo -e "${GREEN}✓ Directories created${NC}"

# Check Python dependencies
echo -e "${YELLOW}Checking Python dependencies...${NC}"
PIP_PACKAGES="prometheus-client psutil fastapi uvicorn numpy"

if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}Error: pip3 not found. Please install Python3 and pip3.${NC}"
    exit 1
fi

MISSING_PACKAGES=()
for package in $PIP_PACKAGES; do
    if ! pip3 list | grep -i "$package" &> /dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "${YELLOW}Installing missing Python packages: ${MISSING_PACKAGES[*]}${NC}"
    pip3 install ${MISSING_PACKAGES[*]}
fi

echo -e "${GREEN}✓ Python dependencies installed${NC}"

# Configure Prometheus
echo -e "${YELLOW}Configuring Prometheus...${NC}"
cat > "${SCRIPT_DIR}/prometheus/prometheus.yml" << EOF
global:
  scrape_interval:     15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
      
  - job_name: 'sutazai_api'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['host.docker.internal:8000', 'host.docker.internal:8100']
      
  - job_name: 'node_exporter'
    static_configs:
      - targets: ['node-exporter:9100']
EOF

# Create alert rules
cat > "${SCRIPT_DIR}/prometheus/alert_rules.yml" << EOF
groups:
- name: sutazai_alerts
  rules:
  - alert: HighCpuUsage
    expr: 100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High CPU usage detected
      description: CPU usage is above 80% for the last 5 minutes
      
  - alert: EthicalBoundaryViolation
    expr: sutazai_ethics_boundary_violations_total > 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: Ethical boundary violation detected
      description: A potential ethical boundary violation has been detected in the AGI system
      
  - alert: SecurityBreachAttempt
    expr: sutazai_security_unauthorized_access_attempts_total > 2
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: Security breach attempt detected
      description: Multiple unauthorized access attempts detected
      
  - alert: NeuralArchitectureAnomaly
    expr: sutazai_neural_plasticity_change_rate > 0.5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: Neural architecture anomaly detected
      description: Abnormal synaptic plasticity changes detected in the neural architecture
      
  - alert: SelfModificationDetected
    expr: rate(sutazai_self_mod_changes_total[1h]) > 0
    for: 0m
    labels:
      severity: warning
    annotations:
      summary: Self-modification detected
      description: The AGI system has attempted to modify its own code or parameters
EOF

echo -e "${GREEN}✓ Prometheus configured${NC}"

# Configure AlertManager
echo -e "${YELLOW}Configuring AlertManager...${NC}"
cat > "${SCRIPT_DIR}/prometheus/alertmanager.yml" << EOF
global:
  resolve_timeout: 5m
  smtp_smarthost: 'smtp.example.com:587'
  smtp_from: 'alertmanager@example.com'
  smtp_auth_username: 'alertmanager'
  smtp_auth_password: 'password'
  slack_api_url: 'https://hooks.slack.com/services/XXXX/YYYY/ZZZZ'

route:
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 12h
  receiver: 'email-team'
  routes:
  - match:
      severity: critical
    receiver: 'alert-critical'
    continue: true

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname']

receivers:
- name: 'email-team'
  email_configs:
  - to: 'team@example.com'
    send_resolved: true
  slack_configs:
  - channel: '#monitoring'
    send_resolved: true

- name: 'alert-critical'
  email_configs:
  - to: 'oncall@example.com'
    send_resolved: true
  slack_configs:
  - channel: '#alerts-critical'
    send_resolved: true
  pagerduty_configs:
  - service_key: '1234567890'
    send_resolved: true
EOF

echo -e "${GREEN}✓ AlertManager configured${NC}"

# Configure Grafana
echo -e "${YELLOW}Configuring Grafana...${NC}"
cat > "${SCRIPT_DIR}/grafana/provisioning/datasources/datasources.yaml" << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
    
  - name: Elasticsearch
    type: elasticsearch
    access: proxy
    url: http://elasticsearch:9200
    database: filebeat-*
    editable: false
    jsonData:
      timeField: "@timestamp"
      esVersion: 7.10.0
EOF

cat > "${SCRIPT_DIR}/grafana/provisioning/dashboards/dashboards.yaml" << EOF
apiVersion: 1

providers:
  - name: 'SutazAI'
    orgId: 1
    folder: 'SutazAI'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
      foldersFromFilesStructure: true
EOF

# Copy the dashboard JSON we created earlier to the provisioning directory
cp "${SCRIPT_DIR}/grafana/dashboards/agi_monitoring_dashboard.json" "${SCRIPT_DIR}/grafana/provisioning/dashboards/"

echo -e "${GREEN}✓ Grafana configured${NC}"

# Configure Logstash
echo -e "${YELLOW}Configuring Logstash...${NC}"
cat > "${SCRIPT_DIR}/elk/logstash/config/logstash.yml" << EOF
http.host: "0.0.0.0"
xpack.monitoring.elasticsearch.hosts: ["http://elasticsearch:9200"]
EOF

cat > "${SCRIPT_DIR}/elk/logstash/pipeline/sutazai.conf" << EOF
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][log_type] == "application" {
    json {
      source => "message"
      skip_on_invalid_json => true
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
      target => "@timestamp"
      remove_field => [ "timestamp" ]
      timezone => "UTC"
    }
  }
  
  if [fields][log_type] == "neural" {
    json {
      source => "message"
      skip_on_invalid_json => true
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
      target => "@timestamp"
      remove_field => [ "timestamp" ]
      timezone => "UTC"
    }
  }
  
  if [fields][log_type] == "ethics" {
    json {
      source => "message"
      skip_on_invalid_json => true
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
      target => "@timestamp"
      remove_field => [ "timestamp" ]
      timezone => "UTC"
    }
    
    if [violation_type] {
      mutate {
        add_tag => ["ethics_violation"]
      }
    }
  }
  
  if [fields][log_type] == "security" {
    json {
      source => "message"
      skip_on_invalid_json => true
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
      target => "@timestamp"
      remove_field => [ "timestamp" ]
      timezone => "UTC"
    }
    
    if [severity] == "critical" or [severity] == "high" {
      mutate {
        add_tag => ["security_alert"]
      }
    }
  }
  
  if [fields][log_type] == "self_mod" {
    json {
      source => "message"
      skip_on_invalid_json => true
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
      target => "@timestamp"
      remove_field => [ "timestamp" ]
      timezone => "UTC"
    }
    
    if [change_type] == "unauthorized" {
      mutate {
        add_tag => ["unauthorized_change"]
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "%{[fields][log_type]}-%{+YYYY.MM.dd}"
    
    # Special index for ethics violations
    if "ethics_violation" in [tags] {
      index => "ethics-violations-%{+YYYY.MM.dd}"
    }
    
    # Special index for security alerts
    if "security_alert" in [tags] {
      index => "security-alerts-%{+YYYY.MM.dd}"
    }
    
    # Special index for unauthorized changes
    if "unauthorized_change" in [tags] {
      index => "unauthorized-changes-%{+YYYY.MM.dd}"
    }
  }
}
EOF

echo -e "${GREEN}✓ Logstash configured${NC}"

# Configure Filebeat
echo -e "${YELLOW}Configuring Filebeat...${NC}"
cat > "${SCRIPT_DIR}/elk/filebeat/filebeat.yml" << EOF
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /opt/sutazaiapp/logs/*.log
  fields:
    log_type: application
  fields_under_root: false
  json.keys_under_root: true
  json.add_error_key: true

- type: log
  enabled: true
  paths:
    - /opt/sutazaiapp/logs/neural/*.log
  fields:
    log_type: neural
  fields_under_root: false
  json.keys_under_root: true
  json.add_error_key: true

- type: log
  enabled: true
  paths:
    - /opt/sutazaiapp/logs/ethics/*.log
  fields:
    log_type: ethics
  fields_under_root: false
  json.keys_under_root: true
  json.add_error_key: true

- type: log
  enabled: true
  paths:
    - /opt/sutazaiapp/logs/security/*.log
  fields:
    log_type: security
  fields_under_root: false
  json.keys_under_root: true
  json.add_error_key: true

- type: log
  enabled: true
  paths:
    - /opt/sutazaiapp/logs/self_mod/*.log
  fields:
    log_type: self_mod
  fields_under_root: false
  json.keys_under_root: true
  json.add_error_key: true

- type: log
  enabled: true
  paths:
    - /opt/sutazaiapp/logs/hardware/*.log
  fields:
    log_type: hardware
  fields_under_root: false
  json.keys_under_root: true
  json.add_error_key: true

filebeat.modules:
- module: system
  syslog:
    enabled: true
  auth:
    enabled: true

processors:
- add_host_metadata: ~
- add_cloud_metadata: ~
- add_docker_metadata: ~
- add_kubernetes_metadata: ~

output.logstash:
  hosts: ["logstash:5044"]
  ssl.enabled: false

logging.level: info
logging.to_files: true
logging.files:
  path: /var/log/filebeat
  name: filebeat
  keepfiles: 7
  permissions: 0644
EOF

echo -e "${GREEN}✓ Filebeat configured${NC}"

# Create Docker Compose file
echo -e "${YELLOW}Creating Docker Compose file...${NC}"
cat > "${SCRIPT_DIR}/docker-compose-monitoring.yml" << EOF
version: '3'

services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./prometheus/alert_rules.yml:/etc/prometheus/alert_rules.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    restart: unless-stopped

  alertmanager:
    image: prom/alertmanager:latest
    volumes:
      - ./prometheus/alertmanager.yml:/etc/alertmanager/alertmanager.yml
    ports:
      - "9093:9093"
    restart: unless-stopped
    
  grafana:
    image: grafana/grafana:latest
    volumes:
      - ./grafana/provisioning/:/etc/grafana/provisioning/
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
    ports:
      - "3000:3000"
    restart: unless-stopped
    
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.16.3
    environment:
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
      - discovery.type=single-node
      - xpack.security.enabled=false
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"
    restart: unless-stopped
    
  logstash:
    image: docker.elastic.co/logstash/logstash:7.16.3
    volumes:
      - ./elk/logstash/config/:/usr/share/logstash/config/
      - ./elk/logstash/pipeline/:/usr/share/logstash/pipeline/
    ports:
      - "5044:5044"
      - "9600:9600"
    environment:
      LS_JAVA_OPTS: "-Xmx256m -Xms256m"
    depends_on:
      - elasticsearch
    restart: unless-stopped
    
  kibana:
    image: docker.elastic.co/kibana/kibana:7.16.3
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_URL=http://elasticsearch:9200
    depends_on:
      - elasticsearch
    restart: unless-stopped
    
  filebeat:
    image: docker.elastic.co/beats/filebeat:7.16.3
    volumes:
      - ./elk/filebeat/filebeat.yml:/usr/share/filebeat/filebeat.yml:ro
      - /opt/sutazaiapp/logs/:/opt/sutazaiapp/logs/:ro
      - /var/log/:/var/log/:ro
    user: root
    depends_on:
      - elasticsearch
      - logstash
    restart: unless-stopped
    
  node-exporter:
    image: prom/node-exporter:latest
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.ignored-mount-points=^/(sys|proc|dev|host|etc)($$|/)'
    ports:
      - "9100:9100"
    restart: unless-stopped
    
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    ports:
      - "8080:8080"
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
  elasticsearch_data:
EOF

echo -e "${GREEN}✓ Docker Compose file created${NC}"

# Make scripts executable
chmod +x "${SCRIPT_DIR}/start_monitoring.py"
echo -e "${GREEN}✓ Made start_monitoring.py executable${NC}"

# Create a simple script to start everything
cat > "${SCRIPT_DIR}/run_all.sh" << EOF
#!/bin/bash

set -e

SCRIPT_DIR=\$(dirname "\$(readlink -f "\$0")")

# Start the Docker Compose stack
docker-compose -f "\${SCRIPT_DIR}/docker-compose-monitoring.yml" up -d

# Wait for services to be ready
echo "Waiting for monitoring services to start..."
sleep 10

# Start the monitoring API
python3 "\${SCRIPT_DIR}/start_monitoring.py"
EOF

chmod +x "${SCRIPT_DIR}/run_all.sh"
echo -e "${GREEN}✓ Created run_all.sh script${NC}"

echo -e "${GREEN}✓ Configuration complete!${NC}"
echo -e "${YELLOW}To start the monitoring system, run:${NC}"
echo -e "  ${GREEN}cd ${SCRIPT_DIR} && ./run_all.sh${NC}"
echo
echo -e "${YELLOW}Access points:${NC}"
echo -e "  ${GREEN}Grafana:${NC} http://localhost:3000 (admin/sutazai)"
echo -e "  ${GREEN}Prometheus:${NC} http://localhost:9090"
echo -e "  ${GREEN}AlertManager:${NC} http://localhost:9093"
echo -e "  ${GREEN}Kibana:${NC} http://localhost:5601"
echo -e "  ${GREEN}Monitoring API:${NC} http://localhost:8100" 