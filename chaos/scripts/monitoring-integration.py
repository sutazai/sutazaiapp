#!/usr/bin/env python3
"""
SutazAI Chaos Engineering - Monitoring Integration
Integrates chaos engineering with Prometheus, Grafana, and existing monitoring
"""

import sys
import os
import json
import yaml
import logging
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
import time

class PrometheusIntegration:
    """Integrates chaos engineering with Prometheus"""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("chaos_prometheus")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def create_chaos_metrics_config(self):
        """Create Prometheus configuration for chaos metrics"""
        chaos_prometheus_config = """
# Chaos Engineering Metrics Configuration for Prometheus

# Add this to your main prometheus.yml scrape_configs section:

  # Chaos Engineering Metrics
  - job_name: 'chaos-experiments'
    static_configs:
      - targets: ['chaos-engine:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s
    honor_labels: true
    scrape_timeout: 10s

  # Chaos Monkey Metrics
  - job_name: 'chaos-monkey'
    static_configs:
      - targets: ['chaos-monkey:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s
    honor_labels: true

  # Container Health Metrics (enhanced for chaos)
  - job_name: 'docker-containers'
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        port: 8080
    relabel_configs:
      - source_labels: [__meta_docker_container_label_chaos_sutazai_com_target]
        action: keep
        regex: true
      - source_labels: [__meta_docker_container_name]
        target_label: container_name
      - source_labels: [__meta_docker_container_label_chaos_sutazai_com_category]
        target_label: chaos_category
      - source_labels: [__meta_docker_container_label_chaos_sutazai_com_protected]
        target_label: chaos_protected

# Recording Rules for Chaos Engineering
groups:
  - name: chaos_engineering
    interval: 30s
    rules:
      # Experiment Success Rate
      - record: chaos:experiment_success_rate
        expr: |
          rate(chaos_experiments_success_total[5m]) / 
          rate(chaos_experiments_total[5m]) * 100

      # Mean Recovery Time
      - record: chaos:mean_recovery_time_seconds
        expr: |
          avg_over_time(chaos_recovery_time_seconds[1h])

      # System Health During Chaos
      - record: chaos:system_health_score
        expr: |
          avg(up{job=~".*sutazai.*"}) * 100

      # Service Availability During Experiments
      - record: chaos:service_availability
        expr: |
          avg by (container_name) (
            up{container_name=~"sutazai-.*"}
          ) * 100

      # Chaos Impact Score
      - record: chaos:impact_score
        expr: |
          (
            100 - chaos:system_health_score
          ) * on() group_left() (
            chaos_experiments_active > 0
          )

# Alerting Rules for Chaos Engineering
groups:
  - name: chaos_alerts
    rules:
      # High Failure Rate Alert
      - alert: ChaosExperimentHighFailureRate
        expr: chaos:experiment_success_rate < 70
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High chaos experiment failure rate detected"
          description: "Chaos experiment success rate is {{ $value }}% (threshold: 70%)"

      # Long Recovery Time Alert
      - alert: ChaosLongRecoveryTime
        expr: chaos:mean_recovery_time_seconds > 300
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Long recovery times detected during chaos experiments"
          description: "Mean recovery time is {{ $value }}s (threshold: 300s)"

      # System Health Degradation Alert
      - alert: ChaosSystemHealthDegradation
        expr: chaos:system_health_score < 80
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "System health degraded during chaos experiment"
          description: "System health score is {{ $value }}% (threshold: 80%)"

      # Chaos Monkey Down Alert
      - alert: ChaosMonkeyDown
        expr: up{job="chaos-monkey"} == 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Chaos Monkey is down"
          description: "Chaos Monkey has been down for more than 5 minutes"

      # Cascade Failure Alert
      - alert: ChaosCascadeFailure
        expr: chaos_cascade_depth > 3
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Cascade failure detected during chaos experiment"
          description: "Cascade failure depth: {{ $value }} services affected"
"""
        
        config_path = "/opt/sutazaiapp/chaos/monitoring/prometheus-chaos-config.yml"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            f.write(chaos_prometheus_config)
        
        self.logger.info(f"Created Prometheus chaos configuration: {config_path}")
    
    def push_custom_metrics(self, metrics: Dict[str, float]):
        """Push custom chaos metrics to Prometheus"""
        try:
            # This would typically use Prometheus Pushgateway
            # For now, we'll create a metrics exposition endpoint
            
            metrics_content = ""
            timestamp = int(time.time() * 1000)
            
            for metric_name, value in metrics.items():
                metrics_content += f"# HELP {metric_name} Chaos engineering metric\n"
                metrics_content += f"# TYPE {metric_name} gauge\n"
                metrics_content += f"{metric_name} {value} {timestamp}\n"
            
            # Save to file for collection
            metrics_file = "/opt/sutazaiapp/chaos/monitoring/chaos_metrics.prom"
            os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
            
            with open(metrics_file, 'w') as f:
                f.write(metrics_content)
            
            self.logger.info(f"Updated chaos metrics: {len(metrics)} metrics")
            
        except Exception as e:
            self.logger.error(f"Failed to push metrics: {e}")
    
    def query_system_health(self) -> Dict[str, float]:
        """Query system health metrics from Prometheus"""
        try:
            queries = {
                'container_health': 'avg(up{job=~".*sutazai.*"})',
                'cpu_usage': 'avg(rate(container_cpu_usage_seconds_total[5m])) * 100',
                'memory_usage': 'avg(container_memory_working_set_bytes / container_spec_memory_limit_bytes) * 100',
                'error_rate': 'rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) * 100'
            }
            
            results = {}
            
            for metric_name, query in queries.items():
                try:
                    response = requests.get(
                        f"{self.prometheus_url}/api/v1/query",
                        params={'query': query},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data['status'] == 'success' and data['data']['result']:
                            value = float(data['data']['result'][0]['value'][1])
                            results[metric_name] = value
                        else:
                            results[metric_name] = 0.0
                    else:
                        results[metric_name] = 0.0
                        
                except Exception as e:
                    self.logger.warning(f"Failed to query {metric_name}: {e}")
                    results[metric_name] = 0.0
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to query system health: {e}")
            return {}

class GrafanaIntegration:
    """Integrates chaos engineering with Grafana"""
    
    def __init__(self, grafana_url: str = "http://localhost:3000", 
                 api_key: Optional[str] = None):
        self.grafana_url = grafana_url
        self.api_key = api_key
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("chaos_grafana")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def create_chaos_dashboard(self):
        """Create Grafana dashboard for chaos engineering"""
        dashboard_json = {
            "dashboard": {
                "id": None,
                "title": "SutazAI Chaos Engineering",
                "tags": ["chaos", "resilience", "sutazai"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Active Experiments",
                        "type": "stat",
                        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
                        "targets": [
                            {
                                "expr": "chaos_experiments_active",
                                "format": "time_series",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "palette-classic"},
                                "unit": "short"
                            }
                        }
                    },
                    {
                        "id": 2,
                        "title": "Experiment Success Rate",
                        "type": "stat",
                        "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0},
                        "targets": [
                            {
                                "expr": "chaos:experiment_success_rate",
                                "format": "time_series",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "thresholds"},
                                "thresholds": {
                                    "steps": [
                                        {"color": "red", "value": 0},
                                        {"color": "yellow", "value": 70},
                                        {"color": "green", "value": 90}
                                    ]
                                },
                                "unit": "percent"
                            }
                        }
                    },
                    {
                        "id": 3,
                        "title": "Mean Recovery Time",
                        "type": "stat",
                        "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0},
                        "targets": [
                            {
                                "expr": "chaos:mean_recovery_time_seconds",
                                "format": "time_series",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "thresholds"},
                                "thresholds": {
                                    "steps": [
                                        {"color": "green", "value": 0},
                                        {"color": "yellow", "value": 60},
                                        {"color": "red", "value": 300}
                                    ]
                                },
                                "unit": "s"
                            }
                        }
                    },
                    {
                        "id": 4,
                        "title": "System Health Score",
                        "type": "stat",
                        "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0},
                        "targets": [
                            {
                                "expr": "chaos:system_health_score",
                                "format": "time_series",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "thresholds"},
                                "thresholds": {
                                    "steps": [
                                        {"color": "red", "value": 0},
                                        {"color": "yellow", "value": 80},
                                        {"color": "green", "value": 95}
                                    ]
                                },
                                "unit": "percent"
                            }
                        }
                    },
                    {
                        "id": 5,
                        "title": "Service Health During Experiments",
                        "type": "graph",
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                        "targets": [
                            {
                                "expr": "chaos:service_availability",
                                "format": "time_series",
                                "legendFormat": "{{container_name}}",
                                "refId": "A"
                            }
                        ],
                        "yAxes": [
                            {"min": 0, "max": 100, "unit": "percent"}
                        ],
                        "xAxis": {"mode": "time"}
                    },
                    {
                        "id": 6,
                        "title": "Chaos Impact Timeline",
                        "type": "graph", 
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                        "targets": [
                            {
                                "expr": "chaos:impact_score",
                                "format": "time_series",
                                "legendFormat": "Impact Score",
                                "refId": "A"
                            },
                            {
                                "expr": "chaos_experiments_active",
                                "format": "time_series",
                                "legendFormat": "Active Experiments",
                                "refId": "B"
                            }
                        ],
                        "yAxes": [
                            {"min": 0, "unit": "short"}
                        ]
                    },
                    {
                        "id": 7,
                        "title": "Experiment Types",
                        "type": "piechart",
                        "gridPos": {"h": 8, "w": 8, "x": 0, "y": 16},
                        "targets": [
                            {
                                "expr": "chaos_experiments_total by (experiment_type)",
                                "format": "time_series",
                                "legendFormat": "{{experiment_type}}",
                                "refId": "A"
                            }
                        ]
                    },
                    {
                        "id": 8,
                        "title": "Recovery Time Distribution",
                        "type": "histogram",
                        "gridPos": {"h": 8, "w": 8, "x": 8, "y": 16},
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, chaos_recovery_time_seconds_bucket)",
                                "format": "time_series",
                                "legendFormat": "95th percentile",
                                "refId": "A"
                            },
                            {
                                "expr": "histogram_quantile(0.50, chaos_recovery_time_seconds_bucket)",
                                "format": "time_series",
                                "legendFormat": "50th percentile",
                                "refId": "B"
                            }
                        ]
                    },
                    {
                        "id": 9,
                        "title": "Chaos Events Log",
                        "type": "logs",
                        "gridPos": {"h": 8, "w": 8, "x": 16, "y": 16},
                        "targets": [
                            {
                                "expr": '{job="chaos-engine"} |= "experiment"',
                                "refId": "A"
                            }
                        ]
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "30s",
                "version": 1
            },
            "overwrite": True
        }
        
        # Save dashboard JSON
        dashboard_path = "/opt/sutazaiapp/chaos/monitoring/grafana-chaos-dashboard.json"
        os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)
        
        with open(dashboard_path, 'w') as f:
            json.dump(dashboard_json, f, indent=2)
        
        self.logger.info(f"Created Grafana chaos dashboard: {dashboard_path}")
        
        return dashboard_json
    
    def create_chaos_alert_rules(self):
        """Create Grafana alert rules for chaos engineering"""
        alert_rules = {
            "groups": [
                {
                    "name": "chaos_engineering_alerts",
                    "interval": "1m",
                    "rules": [
                        {
                            "uid": "chaos_high_failure_rate",
                            "title": "Chaos Experiment High Failure Rate",
                            "condition": "C",
                            "data": [
                                {
                                    "refId": "A",
                                    "queryType": "",
                                    "relativeTimeRange": {
                                        "from": 600,
                                        "to": 0
                                    },
                                    "model": {
                                        "expr": "chaos:experiment_success_rate",
                                        "intervalMs": 1000,
                                        "maxDataPoints": 43200,
                                        "refId": "A"
                                    }
                                },
                                {
                                    "refId": "C",
                                    "queryType": "",
                                    "relativeTimeRange": {
                                        "from": 0,
                                        "to": 0
                                    },
                                    "model": {
                                        "conditions": [
                                            {
                                                "evaluator": {
                                                    "params": [70],
                                                    "type": "lt"
                                                },
                                                "operator": {
                                                    "type": "and"
                                                },
                                                "query": {
                                                    "params": ["A"]
                                                },
                                                "reducer": {
                                                    "params": [],
                                                    "type": "last"
                                                },
                                                "type": "query"
                                            }
                                        ],
                                        "refId": "C"
                                    }
                                }
                            ],
                            "noDataState": "NoData",
                            "execErrState": "Alerting",
                            "for": "5m",
                            "annotations": {
                                "summary": "Chaos experiment success rate is below 70%",
                                "description": "The chaos experiment success rate has been below 70% for more than 5 minutes"
                            },
                            "labels": {
                                "severity": "warning"
                            }
                        }
                    ]
                }
            ]
        }
        
        # Save alert rules
        alerts_path = "/opt/sutazaiapp/chaos/monitoring/grafana-chaos-alerts.json"
        
        with open(alerts_path, 'w') as f:
            json.dump(alert_rules, f, indent=2)
        
        self.logger.info(f"Created Grafana chaos alert rules: {alerts_path}")

class MonitoringIntegration:
    """Main monitoring integration coordinator"""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090",
                 grafana_url: str = "http://localhost:3000"):
        self.prometheus = PrometheusIntegration(prometheus_url)
        self.grafana = GrafanaIntegration(grafana_url)
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("chaos_monitoring_integration")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def setup_complete_monitoring(self):
        """Setup complete chaos engineering monitoring"""
        try:
            self.logger.info("Setting up chaos engineering monitoring integration")
            
            # Create Prometheus configuration
            self.prometheus.create_chaos_metrics_config()
            
            # Create Grafana dashboard
            self.grafana.create_chaos_dashboard()
            
            # Create alert rules
            self.grafana.create_chaos_alert_rules()
            
            self.logger.info("Chaos engineering monitoring setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup monitoring: {e}")
            return False
    
    def validate_monitoring_integration(self) -> Dict[str, Any]:
        """Validate monitoring integration"""
        results = {
            'prometheus_config': False,
            'grafana_dashboard': False,
            'alert_rules': False,
            'prometheus_accessible': False,
            'grafana_accessible': False,
            'errors': []
        }
        
        try:
            # Check if configuration files exist
            config_files = [
                "/opt/sutazaiapp/chaos/monitoring/prometheus-chaos-config.yml",
                "/opt/sutazaiapp/chaos/monitoring/grafana-chaos-dashboard.json",
                "/opt/sutazaiapp/chaos/monitoring/grafana-chaos-alerts.json"
            ]
            
            for config_file in config_files:
                if os.path.exists(config_file):
                    if 'prometheus' in config_file:
                        results['prometheus_config'] = True
                    elif 'dashboard' in config_file:
                        results['grafana_dashboard'] = True
                    elif 'alerts' in config_file:
                        results['alert_rules'] = True
                else:
                    results['errors'].append(f"Missing config file: {config_file}")
            
            # Test Prometheus connectivity
            try:
                health = self.prometheus.query_system_health()
                if health:
                    results['prometheus_accessible'] = True
                else:
                    results['errors'].append("Prometheus not accessible or no data")
            except Exception as e:
                results['errors'].append(f"Prometheus connectivity error: {e}")
            
            # Test Grafana connectivity
            try:
                response = requests.get(f"{self.grafana.grafana_url}/api/health", timeout=5)
                if response.status_code == 200:
                    results['grafana_accessible'] = True
                else:
                    results['errors'].append("Grafana not accessible")
            except Exception as e:
                results['errors'].append(f"Grafana connectivity error: {e}")
            
            results['overall_status'] = all([
                results['prometheus_config'],
                results['grafana_dashboard'],
                results['prometheus_accessible'],
                results['grafana_accessible']
            ])
            
        except Exception as e:
            results['errors'].append(f"Validation error: {e}")
        
        return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SutazAI Chaos Engineering Monitoring Integration")
    parser.add_argument("--prometheus-url", default="http://localhost:9090",
                       help="Prometheus URL")
    parser.add_argument("--grafana-url", default="http://localhost:3000",
                       help="Grafana URL")
    parser.add_argument("--setup", action="store_true",
                       help="Setup complete monitoring integration")
    parser.add_argument("--validate", action="store_true",
                       help="Validate monitoring integration")
    parser.add_argument("--create-dashboard", action="store_true",
                       help="Create Grafana dashboard only")
    parser.add_argument("--create-prometheus-config", action="store_true",
                       help="Create Prometheus configuration only")
    
    args = parser.parse_args()
    
    integration = MonitoringIntegration(args.prometheus_url, args.grafana_url)
    
    if args.setup:
        success = integration.setup_complete_monitoring()
        print(f"Monitoring integration setup {'successful' if success else 'failed'}")
    
    if args.validate:
        results = integration.validate_monitoring_integration()
        print(json.dumps(results, indent=2))
    
    if args.create_dashboard:
        integration.grafana.create_chaos_dashboard()
        print("Grafana dashboard created")
    
    if args.create_prometheus_config:
        integration.prometheus.create_chaos_metrics_config()
        print("Prometheus configuration created")
    
    if not any([args.setup, args.validate, args.create_dashboard, args.create_prometheus_config]):
        print("Use --help for available options")