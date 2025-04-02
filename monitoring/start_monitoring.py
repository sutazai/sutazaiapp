#!/usr/bin/env python3
"""
SutazAI Monitoring System Startup Script

This script initializes and starts the comprehensive SutazAI monitoring system,
which includes neural architecture monitoring, ethical constraints verification,
self-modification monitoring, hardware optimization, and security monitoring.
"""

import os
import sys
import argparse
import logging
import json
import importlib.util
import subprocess
import shutil
from typing import Dict, Any

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(parent_dir, "logs", "monitoring.log")),
    ],
)

logger = logging.getLogger("monitoring_startup")

# Default configuration
DEFAULT_CONFIG = {
    "system_id": "sutazai_default",
    "base_dir": "/opt/sutazaiapp",
    "log_dir": "/opt/sutazaiapp/logs",
    "prometheus_port": 9090,
    "grafana_port": 3000,
    "kibana_port": 5601,
    "enable_neural_monitoring": True,
    "enable_ethics_monitoring": True,
    "enable_self_mod_monitoring": True,
    "enable_hardware_monitoring": True,
    "enable_security_monitoring": True,
    "expose_monitoring_ui": True,
    "monitoring_ui_path": "/monitoring",
    "collection_interval": 30.0,
    "security_config": {
        "allowed_interfaces": ["docker0", "veth", "lo"],
        "restricted_ports": [22, 80, 443, 8080, 9090, 3000, 5601, 9200],
    },
}


def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    required_packages = ["fastapi", "uvicorn", "prometheus_client", "psutil", "numpy"]

    missing_packages = []

    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info(
            "Install required packages with: pip install " + " ".join(missing_packages)
        )
        return False

    return True


def start_prometheus_stack() -> bool:
    """Start the Prometheus monitoring stack using Docker Compose."""
    docker_compose_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "docker-compose-monitoring.yml"
    )

    # Check if the docker-compose file exists
    if not os.path.exists(docker_compose_file):
        logger.error(f"Docker Compose file not found: {docker_compose_file}")
        logger.info("Creating default Docker Compose file...")
        create_default_docker_compose_file(docker_compose_file)

    # Find docker-compose path
    docker_compose_path = shutil.which("docker-compose")
    if not docker_compose_path:
        logger.error(
            "docker-compose command not found in PATH. Cannot start monitoring stack."
        )
        raise FileNotFoundError("docker-compose not found")

    try:
        logger.info("Starting Prometheus monitoring stack...")
        result = subprocess.run(
            [docker_compose_path, "-f", docker_compose_file, "up", "-d"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"Prometheus stack started: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Prometheus stack: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False


def create_default_docker_compose_file(file_path: str) -> None:
    """Create a default Docker Compose file for the monitoring stack."""
    docker_compose_content = """version: '3'

services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
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
      - GF_SECURITY_ADMIN_PASSWORD=sutazai
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

volumes:
  prometheus_data:
  grafana_data:
  elasticsearch_data:
"""

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Write the Docker Compose file
    with open(file_path, "w") as f:
        f.write(docker_compose_content)

    # Create required directories and configuration files
    create_required_directories()


def create_required_directories() -> None:
    """Create required directories and default configuration files."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Create Prometheus directory and configuration
    prometheus_dir = os.path.join(base_dir, "prometheus")
    os.makedirs(prometheus_dir, exist_ok=True)

    if not os.path.exists(os.path.join(prometheus_dir, "prometheus.yml")):
        prometheus_config = """global:
  scrape_interval: 15s
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
      - targets: ['host.docker.internal:8000']
"""
        with open(os.path.join(prometheus_dir, "prometheus.yml"), "w") as f:
            f.write(prometheus_config)

    if not os.path.exists(os.path.join(prometheus_dir, "alert_rules.yml")):
        alert_rules = """groups:
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
      description: Ethical boundary has been violated
"""
        with open(os.path.join(prometheus_dir, "alert_rules.yml"), "w") as f:
            f.write(alert_rules)

    if not os.path.exists(os.path.join(prometheus_dir, "alertmanager.yml")):
        alertmanager_config = """global:
  resolve_timeout: 5m
  smtp_smarthost: 'smtp.example.com:587'
  smtp_from: 'alertmanager@example.com'
  smtp_auth_username: 'alertmanager'
  smtp_auth_type: 'password'
  # TODO: Replace hardcoded password with environment variable or config value
  # smtp_auth_password: 'password' 
  smtp_auth_password: os.environ.get('SMTP_PASSWORD', 'default_placeholder_password')
  require_tls: true
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
"""
        with open(os.path.join(prometheus_dir, "alertmanager.yml"), "w") as f:
            f.write(alertmanager_config)

    # Create Grafana directories and configuration
    grafana_dir = os.path.join(base_dir, "grafana")
    grafana_prov_dir = os.path.join(grafana_dir, "provisioning")
    grafana_dash_dir = os.path.join(grafana_prov_dir, "dashboards")
    grafana_ds_dir = os.path.join(grafana_prov_dir, "datasources")

    os.makedirs(grafana_dash_dir, exist_ok=True)
    os.makedirs(grafana_ds_dir, exist_ok=True)

    if not os.path.exists(os.path.join(grafana_ds_dir, "datasources.yaml")):
        datasources_config = """apiVersion: 1

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
"""
        with open(os.path.join(grafana_ds_dir, "datasources.yaml"), "w") as f:
            f.write(datasources_config)

    if not os.path.exists(os.path.join(grafana_dash_dir, "dashboards.yaml")):
        dashboards_config = """apiVersion: 1

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
"""
        with open(os.path.join(grafana_dash_dir, "dashboards.yaml"), "w") as f:
            f.write(dashboards_config)

    # Create ELK stack directories and configuration
    elk_dir = os.path.join(base_dir, "elk")
    logstash_dir = os.path.join(elk_dir, "logstash")
    logstash_config_dir = os.path.join(logstash_dir, "config")
    logstash_pipeline_dir = os.path.join(logstash_dir, "pipeline")
    filebeat_dir = os.path.join(elk_dir, "filebeat")

    os.makedirs(logstash_config_dir, exist_ok=True)
    os.makedirs(logstash_pipeline_dir, exist_ok=True)
    os.makedirs(filebeat_dir, exist_ok=True)

    if not os.path.exists(os.path.join(logstash_config_dir, "logstash.yml")):
        logstash_config = """http.host: "0.0.0.0"
xpack.monitoring.elasticsearch.hosts: ["http://elasticsearch:9200"]
"""
        with open(os.path.join(logstash_config_dir, "logstash.yml"), "w") as f:
            f.write(logstash_config)

    if not os.path.exists(os.path.join(logstash_pipeline_dir, "sutazai.conf")):
        logstash_pipeline = """input {
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
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "%{[fields][log_type]}-%{+YYYY.MM.dd}"
    
    # Special index for security alerts
    if "security_alert" in [tags] {
      index => "security-alerts-%{+YYYY.MM.dd}"
    }
  }
}
"""
        with open(os.path.join(logstash_pipeline_dir, "sutazai.conf"), "w") as f:
            f.write(logstash_pipeline)

    if not os.path.exists(os.path.join(filebeat_dir, "filebeat.yml")):
        filebeat_config = """filebeat.inputs:
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
"""
        with open(os.path.join(filebeat_dir, "filebeat.yml"), "w") as f:
            f.write(filebeat_config)


def start_monitoring_api(config: Dict[str, Any]) -> None:
    """Start the FastAPI monitoring application."""
    try:
        from utils.monitoring_integration import create_monitoring_system
        from fastapi import FastAPI
        import uvicorn
    except ImportError:
        logger.error(
            "Failed to import required packages. Make sure they are installed."
        )
        return

    logger.info("Initializing monitoring system...")

    # Create the FastAPI app
    app = FastAPI(
        title="SutazAI Monitoring",
        description="SutazAI Monitoring API",
        version="1.0.0",
    )

    # Create the monitoring system
    monitoring_system = create_monitoring_system(
        system_id=config["system_id"],
        base_dir=config["base_dir"],
        log_dir=config["log_dir"],
        prometheus_metrics_path="/metrics",
        enable_neural_monitoring=config["enable_neural_monitoring"],
        enable_ethics_monitoring=config["enable_ethics_monitoring"],
        enable_self_mod_monitoring=config["enable_self_mod_monitoring"],
        enable_hardware_monitoring=config["enable_hardware_monitoring"],
        enable_security_monitoring=config["enable_security_monitoring"],
        expose_monitoring_ui=config["expose_monitoring_ui"],
        monitoring_ui_path=config["monitoring_ui_path"],
        security_config=config["security_config"],
        collection_interval=config["collection_interval"],
    )

    # Set up monitoring for the FastAPI app
    monitoring_system.setup_fastapi(app)

    # Define a simple root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "SutazAI Monitoring System",
            "documentation": "/docs",
            "monitoring_ui": config["monitoring_ui_path"],
            "metrics": "/metrics",
        }

    # Start the server
    logger.info("Starting monitoring API server on port 8100...")
    uvicorn.run(app, host="0.0.0.0", port=8100)  # nosec B104 - Bind to all interfaces for container/network access


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="SutazAI Monitoring System")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--skip-prometheus", action="store_true", help="Skip starting Prometheus stack"
    )
    parser.add_argument(
        "--api-only", action="store_true", help="Start only the monitoring API"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # Load configuration
    config = DEFAULT_CONFIG.copy()
    if args.config:
        if os.path.exists(args.config):
            try:
                with open(args.config, "r") as f:
                    config.update(json.load(f))
                logger.info(f"Loaded configuration from {args.config}")
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in configuration file: {args.config}")
        else:
            logger.error(f"Configuration file not found: {args.config}")

    # Create logs directory
    os.makedirs(config["log_dir"], exist_ok=True)

    # Check dependencies
    if not check_dependencies():
        logger.error(
            "Missing required dependencies. Please install them and try again."
        )
        return

    # Start Prometheus stack
    if not args.skip_prometheus and not args.api_only:
        if not start_prometheus_stack():
            logger.error("Failed to start Prometheus stack.")
            return

    # Start the monitoring API
    try:
        start_monitoring_api(config)
    except KeyboardInterrupt:
        logger.info("Monitoring system stopped by user.")
    except Exception as e:
        logger.error(f"Error starting monitoring system: {e}")


if __name__ == "__main__":
    main()
