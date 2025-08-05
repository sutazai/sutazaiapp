# Comprehensive Infrastructure and DevOps Rules for 69-Agent AI System

## System Specifications Overview
- **Infrastructure**: 69 Docker containers on 12 CPU cores, 29GB RAM
- **Service Mesh**: Consul, Kong, RabbitMQ
- **Core Databases**: PostgreSQL, Redis, Neo4j
- **Monitoring Stack**: Prometheus, Grafana, Loki
- **Critical Issues**: CPU overload, restart loops, missing resource limits

---

## 1. Container Resource Management Rules

### 1.1 Mandatory Resource Limits
**RULE**: Every container MUST have explicit resource limits and reservations defined.

```yaml
# Mandatory resource constraints for all containers
resources:
  limits:
    cpu: "0.5"        # Maximum CPU cores
    memory: "1Gi"     # Maximum memory
    ephemeral-storage: "2Gi"
  requests:
    cpu: "0.1"        # Minimum guaranteed CPU
    memory: "256Mi"   # Minimum guaranteed memory
    ephemeral-storage: "512Mi"
```

### 1.2 Resource Pool Architecture
**RULE**: Implement three-tier resource allocation based on agent criticality:

#### Tier 1: Critical Core Agents (15 agents max)
- **CPU Allocation**: 2.0 cores per agent, 8.0 cores total pool limit
- **Memory Allocation**: 4Gi per agent, 16Gi total pool limit
- **Priority**: High (always scheduled first)
- **Examples**: ai-system-architect, hardware-resource-optimizer, agent-orchestrator

#### Tier 2: Performance Agents (25 agents max)
- **CPU Allocation**: 1.0 core per agent, 6.0 cores total pool limit
- **Memory Allocation**: 2Gi per agent, 12Gi total pool limit
- **Priority**: Medium (scheduled after Tier 1)
- **Examples**: ai-senior-backend-developer, cicd-pipeline-orchestrator

#### Tier 3: Specialized Agents (29 agents max)
- **CPU Allocation**: 0.5 cores per agent, 4.0 cores total pool limit
- **Memory Allocation**: 1Gi per agent, 8Gi total pool limit
- **Priority**: Low (scheduled last, can be preempted)
- **Examples**: document-knowledge-manager, garbage-collector

### 1.3 Resource Enforcement Mechanisms
**RULE**: Implement automated resource enforcement:

```bash
# Mandatory pre-deployment resource validation
validate_resources() {
    local container_name=$1
    local cpu_limit=$(docker inspect "$container_name" --format='{{.HostConfig.CpuQuota}}')
    local memory_limit=$(docker inspect "$container_name" --format='{{.HostConfig.Memory}}')
    
    if [[ "$cpu_limit" == "0" ]] || [[ "$memory_limit" == "0" ]]; then
        echo "ERROR: Container $container_name missing resource limits"
        exit 1
    fi
}
```

### 1.4 System Resource Reservation
**RULE**: Reserve system resources for stability:
- **CPU**: Reserve 2.0 cores (16.7%) for system processes
- **Memory**: Reserve 4Gi (13.8%) for system processes
- **Total Available**: 10 CPU cores, 25Gi RAM for agents

---

## 2. Deployment Procedures

### 2.1 Phased Deployment Strategy
**RULE**: Deploy agents in phases to prevent resource exhaustion:

#### Phase 1: Infrastructure Foundation
1. Deploy core databases (PostgreSQL, Redis, Neo4j)
2. Deploy service mesh (Consul, Kong)
3. Deploy monitoring stack (Prometheus, Grafana, Loki)
4. Validate infrastructure health before proceeding

#### Phase 2: Critical Agent Deployment
1. Deploy Tier 1 agents (1 at a time)
2. Wait for health check confirmation (30s timeout)
3. Validate resource consumption < 80% before next deployment
4. Maximum 5 concurrent deployments

#### Phase 3: Performance Agent Deployment
1. Deploy Tier 2 agents (2 at a time)
2. Monitor system load during deployment
3. Pause deployment if CPU > 85% or Memory > 90%

#### Phase 4: Specialized Agent Deployment
1. Deploy Tier 3 agents (3 at a time)
2. Implement hibernation for idle agents
3. Use resource-based activation

### 2.2 Deployment Validation Checklist
**RULE**: Every deployment MUST pass these checks:

```bash
#!/bin/bash
# Mandatory deployment validation

deployment_validation() {
    local service_name=$1
    
    # 1. Resource limit validation
    validate_resources "$service_name"
    
    # 2. Health check validation
    timeout 60 docker exec "$service_name" curl -f http://localhost:8080/health || exit 1
    
    # 3. System resource validation
    cpu_usage=$(docker stats --no-stream --format "table {{.CPUPerc}}" | tail -n +2 | sed 's/%//' | awk '{sum+=$1} END {print sum}')
    if (( $(echo "$cpu_usage > 85.0" | bc -l) )); then
        echo "ERROR: System CPU usage too high: $cpu_usage%"
        exit 1
    fi
    
    # 4. Network connectivity validation
    docker exec "$service_name" nslookup consul.service.consul || exit 1
    
    echo "Deployment validation passed for $service_name"
}
```

### 2.3 Rollback Procedures
**RULE**: Implement automated rollback for failed deployments:

```bash
rollback_deployment() {
    local service_name=$1
    local previous_version=$2
    
    echo "Rolling back $service_name to $previous_version"
    docker service update --image "$previous_version" "$service_name"
    
    # Wait for rollback completion
    timeout 120 docker service ps "$service_name" --filter "desired-state=running"
    
    # Validate rollback
    deployment_validation "$service_name"
}
```

---

## 3. Infrastructure as Code (IaC)

### 3.1 Terraform Resource Management
**RULE**: All infrastructure MUST be defined in Terraform with proper resource management:

```hcl
# terraform/modules/agent-container/main.tf
resource "docker_container" "agent" {
  count = var.agent_count
  
  name  = "${var.agent_name}-${count.index}"
  image = var.agent_image
  
  # Mandatory resource constraints
  cpu_set = var.cpu_set
  memory = var.memory_limit
  memory_swap = var.memory_limit * 2
  
  # Health check configuration
  healthcheck {
    test     = ["CMD", "curl", "-f", "http://localhost:8080/health"]
    interval = "30s"
    timeout  = "10s"
    retries  = 3
  }
  
  # Network configuration
  networks_advanced {
    name = var.network_name
    ipv4_address = var.static_ip
  }
  
  # Resource monitoring labels
  labels {
    label = "sutazai.tier"
    value = var.tier
  }
  labels {
    label = "sutazai.priority"
    value = var.priority
  }
}
```

### 3.2 Configuration Management
**RULE**: Use centralized configuration management:

```yaml
# config/infrastructure.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: sutazai-infrastructure-config
data:
  # Resource allocation per tier
  tier1_cpu_limit: "2.0"
  tier1_memory_limit: "4Gi"
  tier2_cpu_limit: "1.0"
  tier2_memory_limit: "2Gi"
  tier3_cpu_limit: "0.5"
  tier3_memory_limit: "1Gi"
  
  # System thresholds
  cpu_warning_threshold: "75"
  cpu_critical_threshold: "85"
  memory_warning_threshold: "80"
  memory_critical_threshold: "90"
  
  # Service discovery
  consul_endpoint: "consul.service.consul:8500"
  kong_admin_endpoint: "kong-admin.service.consul:8001"
  rabbitmq_endpoint: "rabbitmq.service.consul:5672"
```

### 3.3 GitOps Implementation
**RULE**: Implement GitOps workflow for infrastructure changes:

```bash
# .github/workflows/infrastructure.yml
name: Infrastructure Deployment

on:
  push:
    branches: [main]
    paths: ['terraform/**', 'config/**']

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - name: Terraform Validate
        run: terraform validate
      
      - name: Resource Calculation
        run: |
          total_cpu=$(grep -r "cpu.*=" terraform/ | awk '{sum += $2} END {print sum}')
          if (( $(echo "$total_cpu > 10.0" | bc -l) )); then
            echo "ERROR: Total CPU allocation exceeds available: $total_cpu"
            exit 1
          fi
  
  deploy:
    needs: validate
    runs-on: ubuntu-latest
    steps:
      - name: Terraform Apply
        run: terraform apply -auto-approve
```

---

## 4. Monitoring Setup

### 4.1 Multi-Layer Monitoring Architecture
**RULE**: Implement comprehensive monitoring at all layers:

#### System Layer Monitoring
```yaml
# prometheus/system-monitoring.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
    scrape_interval: 10s
    
  - job_name: 'docker-daemon'
    static_configs:
      - targets: ['localhost:9323']
    metrics_path: /metrics
    
  - job_name: 'consul'
    consul_sd_configs:
      - server: 'consul.service.consul:8500'
    relabel_configs:
      - source_labels: [__meta_consul_service]
        target_label: service
```

#### Agent-Level Monitoring
```yaml
# Agent health monitoring configuration
agent_monitoring:
  health_check_interval: 30s
  metrics_collection:
    - cpu_usage
    - memory_usage
    - response_time
    - error_rate
    - task_completion_rate
  
  alerting_rules:
    - name: agent_high_cpu
      condition: cpu_usage > 80
      duration: 2m
      severity: warning
      
    - name: agent_memory_leak
      condition: memory_usage_trend > 10%/hour
      duration: 5m
      severity: critical
```

### 4.2 Resource Utilization Dashboards
**RULE**: Implement real-time resource monitoring dashboards:

```json
{
  "dashboard": {
    "title": "Sutazai Resource Utilization",
    "panels": [
      {
        "title": "CPU Utilization by Tier",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(container_cpu_usage_seconds_total[5m])) by (sutazai_tier)",
            "legendFormat": "Tier {{sutazai_tier}}"
          }
        ],
        "alert": {
          "conditions": [
            {
              "query": {"queryType": "", "refId": "A"},
              "reducer": {"type": "last", "params": []},
              "evaluator": {"params": [85], "type": "gt"}
            }
          ],
          "executionErrorState": "alerting",
          "frequency": "30s"
        }
      }
    ]
  }
}
```

### 4.3 Predictive Monitoring
**RULE**: Implement predictive monitoring to prevent resource exhaustion:

```python
# monitoring/predictive_monitor.py
import numpy as np
from sklearn.linear_model import LinearRegression
import prometheus_client

class PredictiveResourceMonitor:
    def __init__(self):
        self.cpu_history = []
        self.memory_history = []
        self.prediction_window = 300  # 5 minutes
        
    def predict_resource_exhaustion(self):
        """Predict when resources will be exhausted"""
        if len(self.cpu_history) < 10:
            return None
            
        # Linear regression for trend analysis
        X = np.array(range(len(self.cpu_history))).reshape(-1, 1)
        y = np.array(self.cpu_history)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict future values
        future_points = range(len(self.cpu_history), 
                            len(self.cpu_history) + self.prediction_window)
        predictions = model.predict(np.array(future_points).reshape(-1, 1))
        
        # Find when CPU will exceed 85%
        exhaustion_point = None
        for i, pred in enumerate(predictions):
            if pred > 85.0:
                exhaustion_point = i * 15  # 15 second intervals
                break
                
        return exhaustion_point
```

---

## 5. Network Security

### 5.1 Network Segmentation
**RULE**: Implement network segmentation with security zones:

```yaml
# docker-compose.security.yml
networks:
  # DMZ - External facing services
  dmz:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.1.0/24
    driver_opts:
      com.docker.network.bridge.enable_icc: "false"
      
  # Internal - Core services
  internal:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.2.0/24
    internal: true
    
  # Data - Database layer
  data:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.3.0/24
    internal: true
```

### 5.2 Service Mesh Security
**RULE**: Implement mTLS and access control in service mesh:

```yaml
# consul-connect/intentions.hcl
Kind = "service-intentions"
Name = "agent-orchestrator"
Sources = [
  {
    Name   = "ai-system-architect"
    Action = "allow"
  },
  {
    Name   = "hardware-resource-optimizer"
    Action = "allow"
  },
  {
    Name   = "*"
    Action = "deny"
  }
]
```

### 5.3 API Gateway Security
**RULE**: Configure Kong with security plugins:

```yaml
# kong/security-config.yml
services:
  - name: agent-api
    url: http://agent-orchestrator:8080
    plugins:
      - name: rate-limiting
        config:
          minute: 100
          hour: 1000
          policy: local
          
      - name: key-auth
        config:
          key_names: ["X-API-Key"]
          
      - name: ip-restriction
        config:
          allow: ["172.20.0.0/16"]
          
      - name: request-size-limiting
        config:
          allowed_payload_size: 1024
```

### 5.4 Container Security
**RULE**: Implement container security hardening:

```dockerfile
# Dockerfile security template
FROM alpine:3.18 AS base

# Create non-root user
RUN adduser -D -s /bin/sh sutazai

# Security hardening
RUN apk add --no-cache ca-certificates \
    && apk del --purge apk-tools \
    && rm -rf /var/cache/apk/*

# Remove unnecessary setuid/setgid binaries
RUN find / -perm /6000 -type f -exec ls -ld {} \; 2>/dev/null | \
    awk '{print $9}' | xargs -I {} rm -f {}

USER sutazai
WORKDIR /app

# Security labels
LABEL security.no-new-privileges=true
LABEL security.read-only-root-filesystem=true
LABEL security.non-root-user=sutazai
```

---

## 6. Disaster Recovery

### 6.1 Backup Strategy
**RULE**: Implement automated, tested backup procedures:

```bash
#!/bin/bash
# backup/automated-backup.sh

BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/sutazai-backups/$BACKUP_TIMESTAMP"

perform_backup() {
    mkdir -p "$BACKUP_DIR"
    
    # Database backups
    docker exec postgres pg_dump -U sutazai sutazai > "$BACKUP_DIR/postgres.sql"
    docker exec redis redis-cli --rdb "$BACKUP_DIR/redis.rdb"
    docker exec neo4j cypher-shell "CALL apoc.export.cypher.all('$BACKUP_DIR/neo4j.cypher', {})"
    
    # Configuration backups
    cp -r /opt/sutazaiapp/config "$BACKUP_DIR/"
    
    # Agent state backups
    docker exec agent-orchestrator curl http://localhost:8080/api/backup > "$BACKUP_DIR/agent-state.json"
    
    # Compress backup
    tar -czf "${BACKUP_DIR}.tar.gz" -C "$BACKUP_DIR" .
    rm -rf "$BACKUP_DIR"
    
    # Validate backup
    if [[ -f "${BACKUP_DIR}.tar.gz" ]]; then
        echo "Backup completed successfully: ${BACKUP_DIR}.tar.gz"
        
        # Test backup integrity
        tar -tzf "${BACKUP_DIR}.tar.gz" > /dev/null && echo "Backup integrity verified"
    else
        echo "ERROR: Backup failed"
        exit 1
    fi
}

# Schedule: Every 6 hours
0 */6 * * * /opt/sutazaiapp/backup/automated-backup.sh
```

### 6.2 High Availability Configuration
**RULE**: Implement HA for critical services:

```yaml
# docker-compose.ha.yml
version: '3.8'
services:
  postgres-primary:
    image: postgres:15-alpine
    environment:
      POSTGRES_REPLICATION_MODE: master
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: ${REPLICATION_PASSWORD}
    volumes:
      - postgres_primary:/var/lib/postgresql/data
      
  postgres-replica:
    image: postgres:15-alpine
    environment:
      POSTGRES_REPLICATION_MODE: slave
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: ${REPLICATION_PASSWORD}
      POSTGRES_MASTER_SERVICE: postgres-primary
    volumes:
      - postgres_replica:/var/lib/postgresql/data
    depends_on:
      - postgres-primary

  redis-cluster:
    image: redis:7-alpine
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf
    deploy:
      replicas: 3
    networks:
      - redis-cluster-net
```

### 6.3 Disaster Recovery Testing
**RULE**: Automated DR testing monthly:

```python
#!/usr/bin/env python3
# dr/disaster-recovery-test.py

import subprocess
import time
import requests
import json
from datetime import datetime

class DisasterRecoveryTest:
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
        
    def test_backup_restoration(self):
        """Test complete system restoration from backup"""
        print("Starting backup restoration test...")
        
        # 1. Create test backup
        result = subprocess.run(['./backup/automated-backup.sh'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            self.test_results['backup_creation'] = 'FAILED'
            return False
            
        # 2. Simulate disaster (stop all services)
        subprocess.run(['docker-compose', 'down'], cwd='/opt/sutazaiapp')
        
        # 3. Restore from backup
        latest_backup = self.get_latest_backup()
        restore_result = self.restore_from_backup(latest_backup)
        
        # 4. Validate system health
        time.sleep(60)  # Wait for services to start
        health_check = self.validate_system_health()
        
        self.test_results['backup_restoration'] = {
            'backup_creation': 'PASSED',
            'restoration': 'PASSED' if restore_result else 'FAILED',
            'health_validation': 'PASSED' if health_check else 'FAILED',
            'total_time': (datetime.now() - self.start_time).seconds
        }
        
        return restore_result and health_check
        
    def validate_system_health(self):
        """Validate all critical services are healthy"""
        critical_services = [
            'http://localhost:5432',  # PostgreSQL
            'http://localhost:6379',  # Redis
            'http://localhost:7474',  # Neo4j
            'http://localhost:8500',  # Consul
            'http://localhost:8000',  # Kong
        ]
        
        for service in critical_services:
            try:
                response = requests.get(f"{service}/health", timeout=10)
                if response.status_code != 200:
                    return False
            except requests.RequestException:
                return False
                
        return True

# Schedule monthly DR tests
if __name__ == "__main__":
    dr_test = DisasterRecoveryTest()
    success = dr_test.test_backup_restoration()
    
    # Generate report
    with open(f'/opt/sutazaiapp/reports/dr_test_{datetime.now().strftime("%Y%m%d")}.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'results': dr_test.test_results
        }, f, indent=2)
```

### 6.4 Emergency Shutdown Procedures
**RULE**: Implement graceful emergency shutdown:

```bash
#!/bin/bash
# emergency/graceful-shutdown.sh

emergency_shutdown() {
    echo "EMERGENCY: Initiating graceful system shutdown"
    
    # 1. Stop accepting new requests
    docker exec kong kong reload --nginx-conf=/etc/kong/nginx-shutdown.conf
    
    # 2. Drain existing connections (30 second timeout)
    sleep 30
    
    # 3. Save agent states
    for agent in $(docker ps --filter "label=sutazai.tier" --format "{{.Names}}"); do
        echo "Saving state for $agent"
        docker exec "$agent" curl -X POST http://localhost:8080/api/save-state
    done
    
    # 4. Shutdown in reverse priority order
    # Tier 3 first (lowest priority)
    docker ps --filter "label=sutazai.tier=3" --format "{{.Names}}" | \
        xargs -I {} docker stop --time=30 {}
    
    # Tier 2
    docker ps --filter "label=sutazai.tier=2" --format "{{.Names}}" | \
        xargs -I {} docker stop --time=45 {}
    
    # Tier 1 last (highest priority)
    docker ps --filter "label=sutazai.tier=1" --format "{{.Names}}" | \
        xargs -I {} docker stop --time=60 {}
    
    # 5. Shutdown infrastructure
    docker stop postgres redis neo4j consul kong rabbitmq
    
    echo "Emergency shutdown completed"
}

# Trigger on critical system alerts
emergency_shutdown
```

---

## 7. Enforcement and Compliance

### 7.1 Automated Rule Enforcement
**RULE**: Implement CI/CD pipeline validation:

```yaml
# .github/workflows/infrastructure-compliance.yml
name: Infrastructure Compliance Check

on:
  pull_request:
    paths: ['docker-compose*.yml', 'terraform/**', 'config/**']

jobs:
  compliance-check:
    runs-on: ubuntu-latest
    steps:
      - name: Resource Limit Validation
        run: |
          # Check all containers have resource limits
          missing_limits=$(grep -L "cpu.*:" docker-compose*.yml | wc -l)
          if [[ $missing_limits -gt 0 ]]; then
            echo "ERROR: $missing_limits files missing CPU limits"
            exit 1
          fi
          
      - name: Security Scan
        run: |
          # Scan for security violations
          docker run --rm -v $(pwd):/scan \
            aquasec/trivy config /scan
            
      - name: Resource Calculation
        run: |
          # Verify total resources don't exceed limits
          python3 scripts/validate-resource-allocation.py
```

### 7.2 Runtime Compliance Monitoring
**RULE**: Continuous compliance monitoring:

```python
# monitoring/compliance_monitor.py
import docker
import json
import time
from datetime import datetime

class ComplianceMonitor:
    def __init__(self):
        self.client = docker.from_env()
        self.violations = []
        
    def check_resource_limits(self):
        """Verify all containers have proper resource limits"""
        containers = self.client.containers.list()
        
        for container in containers:
            if 'sutazai' in container.name:
                config = self.client.api.inspect_container(container.id)
                
                # Check CPU limits
                cpu_quota = config['HostConfig']['CpuQuota']
                if cpu_quota == 0:
                    self.violations.append({
                        'container': container.name,
                        'violation': 'missing_cpu_limit',
                        'timestamp': datetime.now().isoformat()
                    })
                
                # Check memory limits
                memory_limit = config['HostConfig']['Memory']
                if memory_limit == 0:
                    self.violations.append({
                        'container': container.name,
                        'violation': 'missing_memory_limit',
                        'timestamp': datetime.now().isoformat()
                    })
    
    def enforce_compliance(self):
        """Automatically fix compliance violations"""
        for violation in self.violations:
            if violation['violation'] == 'missing_cpu_limit':
                self.apply_default_cpu_limit(violation['container'])
            elif violation['violation'] == 'missing_memory_limit':
                self.apply_default_memory_limit(violation['container'])

# Run compliance check every 5 minutes
if __name__ == "__main__":
    monitor = ComplianceMonitor()
    while True:
        monitor.check_resource_limits()
        if monitor.violations:
            monitor.enforce_compliance()
        time.sleep(300)
```

---

## 8. Performance Optimization Rules

### 8.1 Resource Optimization
**RULE**: Implement dynamic resource allocation:

```python
# optimization/dynamic_allocator.py
class DynamicResourceAllocator:
    def __init__(self):
        self.cpu_threshold_scale_up = 0.8
        self.cpu_threshold_scale_down = 0.3
        self.memory_threshold_scale_up = 0.85
        self.memory_threshold_scale_down = 0.4
        
    def optimize_allocation(self):
        """Dynamically adjust resource allocation based on usage"""
        containers = self.get_agent_containers()
        
        for container in containers:
            metrics = self.get_container_metrics(container)
            
            # CPU optimization
            if metrics['cpu_usage'] > self.cpu_threshold_scale_up:
                new_cpu_limit = min(metrics['cpu_limit'] * 1.2, self.get_max_cpu_for_tier(container))
                self.update_cpu_limit(container, new_cpu_limit)
                
            elif metrics['cpu_usage'] < self.cpu_threshold_scale_down:
                new_cpu_limit = max(metrics['cpu_limit'] * 0.8, self.get_min_cpu_for_tier(container))
                self.update_cpu_limit(container, new_cpu_limit)
```

### 8.2 Load Balancing Rules
**RULE**: Implement intelligent load balancing:

```nginx
# nginx/load-balancing.conf
upstream agent_pool_tier1 {
    least_conn;
    server ai-system-architect:8080 weight=3 max_fails=2 fail_timeout=30s;
    server hardware-resource-optimizer:8080 weight=2 max_fails=2 fail_timeout=30s;
    server agent-orchestrator:8080 weight=5 max_fails=1 fail_timeout=15s;
}

upstream agent_pool_tier2 {
    ip_hash;
    server ai-senior-backend-developer:8080 weight=2;
    server cicd-pipeline-orchestrator:8080 weight=2;
    server ai-qa-team-lead:8080 weight=1;
}

# Health check configuration
location /health {
    access_log off;
    return 200 "healthy\n";
    add_header Content-Type text/plain;
}
```

---

## 9. Conclusion and Implementation Timeline

### Implementation Priority:
1. **Week 1**: Resource limits and monitoring setup
2. **Week 2**: Network security and service mesh
3. **Week 3**: Backup and disaster recovery
4. **Week 4**: Performance optimization and compliance

### Success Metrics:
- Zero container restarts due to resource exhaustion
- <5% system resource utilization variance
- <30 second deployment times per agent
- 99.9% service availability
- Zero security vulnerabilities in production

### Monitoring Dashboard KPIs:
- Real-time resource utilization by tier
- Agent health and performance metrics
- Network security events
- Backup success rates
- Compliance violation alerts

This comprehensive rule set addresses all critical infrastructure and DevOps concerns for the 69-agent AI system, providing concrete, actionable guidelines for stable operation within the constrained resource environment.