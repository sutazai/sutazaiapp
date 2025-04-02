#!/bin/bash
# SutazAI Monitoring System Startup Script
# This script starts the monitoring stack and configures necessary components

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==== Starting SutazAI Monitoring System ===="

# Check for required directories
mkdir -p ./prometheus/data
mkdir -p ./grafana/data
mkdir -p ./grafana/dashboards
mkdir -p ./elk/logstash/pipeline
mkdir -p ./elk/logstash/config
mkdir -p ./elk/filebeat
mkdir -p ./elk/metricbeat
mkdir -p ./sentry/data

# Make sure the alert rules file exists
if [ ! -f ./prometheus/alert_rules.yml ]; then
    echo "Alert rules file not found. Creating default alert rules..."
    mkdir -p ./prometheus
    cat > ./prometheus/alert_rules.yml << EOF
groups:
  - name: sutazai_alerts
    rules:
      # System alerts
      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High CPU usage detected
          description: CPU usage on {{ \$labels.instance }} is above 80% for the last 5 minutes
EOF
fi

# Make sure the alert manager config exists
if [ ! -f ./prometheus/alertmanager.yml ]; then
    echo "AlertManager configuration not found. Creating default configuration..."
    mkdir -p ./prometheus
    cat > ./prometheus/alertmanager.yml << EOF
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'job', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'default'

receivers:
- name: 'default'
  email_configs:
  - to: 'admin@example.com'
    send_resolved: true
EOF
fi

# Ensure logstash configuration exists
if [ ! -f ./elk/logstash/config/logstash.yml ]; then
    echo "Creating default logstash.yml"
    cat > ./elk/logstash/config/logstash.yml << EOF
http.host: "0.0.0.0"
xpack.monitoring.elasticsearch.hosts: [ "http://elasticsearch:9200" ]
EOF
fi

# Ensure Logstash pipeline is set up
if [ ! -f ./elk/logstash/pipeline/sutazai.conf ]; then
    echo "Creating Logstash pipeline configuration"
    mkdir -p ./elk/logstash/pipeline
    cat > ./elk/logstash/pipeline/sutazai.conf << EOF
input {
  # Receive logs from Filebeat
  beats {
    port => 5044
    host => "0.0.0.0"
    tags => ["filebeat"]
  }
  
  # TCP input for direct log forwarding
  tcp {
    port => 5000
    codec => json
    tags => ["direct"]
  }
}

filter {
  # Process JSON logs
  if [message] and [message] =~ /^\{.*\}$/ {
    json {
      source => "message"
      target => "log_data"
    }
  }
  
  # Add environment-specific tag
  if [environment] {
    mutate {
      add_tag => [ "%{environment}" ]
    }
  }
}

output {
  # Send everything to Elasticsearch
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "sutazai-%{+YYYY.MM.dd}"
    user => "\${ELASTIC_USERNAME:elastic}"
    password => "\${ELASTIC_PASSWORD:sutazaisecure}"
  }
}
EOF
fi

# Ensure Filebeat configuration exists
if [ ! -f ./elk/filebeat/filebeat.yml ]; then
    echo "Creating Filebeat configuration"
    mkdir -p ./elk/filebeat
    cat > ./elk/filebeat/filebeat.yml << EOF
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /opt/sutazaiapp/logs/*.log
    fields:
      log_type: application
    fields_under_root: true
    
  - type: log
    enabled: true
    paths:
      - /opt/sutazaiapp/logs/*.json.log
    fields:
      log_type: application_json
    fields_under_root: true
    json.keys_under_root: true

output.logstash:
  hosts: ["logstash:5044"]
  ssl.enabled: false
EOF
fi

# Ensure Grafana dashboards directory is properly set up
if [ ! -f ./grafana/dashboards.yaml ]; then
    echo "Creating Grafana dashboard provisioning"
    mkdir -p ./grafana/provisioning/dashboards
    cat > ./grafana/provisioning/dashboards/dashboards.yaml << EOF
apiVersion: 1

providers:
- name: 'SutazAI Dashboards'
  orgId: 1
  folder: ''
  type: file
  disableDeletion: false
  updateIntervalSeconds: 10
  allowUiUpdates: true
  options:
    path: /etc/grafana/dashboards
    foldersFromFilesStructure: true
EOF
fi

# Ensure data source provisioning is set up
mkdir -p ./grafana/provisioning/datasources
if [ ! -f ./grafana/provisioning/datasources/datasources.yaml ]; then
    echo "Creating Grafana datasource provisioning"
    cat > ./grafana/provisioning/datasources/datasources.yaml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true

  - name: Elasticsearch
    type: elasticsearch
    access: proxy
    url: http://elasticsearch:9200
    database: "sutazai-*"
    jsonData:
      timeField: "@timestamp"
      esVersion: 7.16.3
    secureJsonData:
      basicAuth: true
      basicAuthUser: elastic
      basicAuthPassword: sutazaisecure
    editable: true
EOF
fi

# Copy dashboard definitions
echo "Copying Grafana dashboards"
cp -f ./grafana/dashboards/*.json ./grafana/dashboards/ 2>/dev/null || true

# Check if we need to install dependencies
if [ ! -f ./requirements.txt ] || [ ! -d ./venv ]; then
    echo "Setting up Python environment for monitoring utilities"
    python3 -m venv ./venv
    ./venv/bin/pip install -U pip
    
    cat > ./requirements.txt << EOF
prometheus-client>=0.14.1
structlog>=22.1.0
psutil>=5.9.4
sentry-sdk>=1.14.0
elasticsearch>=7.16.3,<8.0.0
elasticsearch-dsl>=7.4.0
python-dotenv>=0.21.0
requests>=2.28.1
EOF

    ./venv/bin/pip install -r requirements.txt
fi

# Start the Docker containers
echo "Starting monitoring stack..."
docker-compose -f docker-compose.yml up -d

# Check if we need to initialize Elasticsearch templates
echo "Initializing Elasticsearch templates and ILM policies..."
sleep 10  # Wait for Elasticsearch to start
curl -X PUT "http://localhost:9200/_template/sutazai" -u elastic:sutazaisecure -H 'Content-Type: application/json' -d'
{
  "index_patterns": ["sutazai-*"],
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "index.lifecycle.name": "sutazai-logs-policy",
    "index.lifecycle.rollover_alias": "sutazai"
  },
  "mappings": {
    "properties": {
      "@timestamp": {
        "type": "date"
      },
      "log_level": {
        "type": "keyword"
      },
      "message": {
        "type": "text"
      },
      "logger_name": {
        "type": "keyword"
      },
      "environment": {
        "type": "keyword"
      },
      "type": {
        "type": "keyword"
      }
    }
  }
}' || echo "Failed to create Elasticsearch template. Elasticsearch may not be ready yet."

# Create ILM policy
curl -X PUT "http://localhost:9200/_ilm/policy/sutazai-logs-policy" -u elastic:sutazaisecure -H 'Content-Type: application/json' -d'
{
  "policy": {
    "phases": {
      "hot": {
        "min_age": "0ms",
        "actions": {
          "rollover": {
            "max_age": "7d",
            "max_size": "10gb"
          },
          "set_priority": {
            "priority": 100
          }
        }
      },
      "warm": {
        "min_age": "30d",
        "actions": {
          "shrink": {
            "number_of_shards": 1
          },
          "set_priority": {
            "priority": 50
          }
        }
      },
      "cold": {
        "min_age": "60d",
        "actions": {
          "set_priority": {
            "priority": 0
          },
          "readonly": {}
        }
      },
      "delete": {
        "min_age": "90d",
        "actions": {
          "delete": {}
        }
      }
    }
  }
}' || echo "Failed to create ILM policy. Elasticsearch may not be ready yet."

# Start Sentry if configured
if [ -f ./sentry/.env ]; then
    echo "Starting Sentry..."
    docker-compose -f sentry/docker-compose.yml up -d
else
    echo "Sentry .env file not found. Skipping Sentry startup."
    echo "To configure Sentry, create a ./sentry/.env file with the required settings."
fi

# Check status
echo "Checking monitoring systems status:"
echo "- Prometheus:"
curl -s http://localhost:9090/-/healthy || echo "  Not responding yet..."
echo "- AlertManager:"
curl -s http://localhost:9093/-/healthy || echo "  Not responding yet..."
echo "- Grafana:"
curl -s http://localhost:3000/api/health || echo "  Not responding yet..."
echo "- Elasticsearch:"
curl -s http://localhost:9200/_cluster/health || echo "  Not responding yet..."
echo "- Kibana:"
curl -s http://localhost:5601/api/status || echo "  Not responding yet..."

echo ""
echo "==== SutazAI Monitoring System Started ===="
echo "Grafana:       http://localhost:3000 (admin/sutazai)"
echo "Prometheus:    http://localhost:9090"
echo "AlertManager:  http://localhost:9093"
echo "Kibana:        http://localhost:5601 (elastic/sutazaisecure)"
echo "Sentry:        http://localhost:9000"
echo ""
echo "For detailed logs, run: docker-compose logs -f"
echo "" 