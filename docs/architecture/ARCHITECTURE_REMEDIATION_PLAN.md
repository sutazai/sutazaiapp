# Architecture Remediation Plan

## Immediate Actions Required for System Stability

Based on the current system state analysis, this plan addresses critical issues that must be resolved immediately.

---

## 🚨 Priority 1: Critical Issues (Fix Within 24 Hours)

### 1.1 Ollama CPU Usage (185%)

**Issue**: Ollama is consuming 185% CPU without limits, causing system instability.

**Remediation Steps**:

```yaml
# Update ollama service in docker-compose.yml
ollama:
  image: ollama/ollama:latest
  container_name: ollama
  deploy:
    resources:
      limits:
        memory: 4Gi
        cpus: '4.0'  # Limit to 4 CPU cores
      reservations:
        memory: 2Gi
        cpus: '2.0'
  environment:
    - OLLAMA_NUM_PARALLEL=2          # Limit parallel model executions
    - OLLAMA_MAX_LOADED_MODELS=1     # Only keep 1 model in memory
    - OLLAMA_CPU_LIMIT=4             # Enforce CPU limit
    - OLLAMA_KEEP_ALIVE=5m           # Unload models after 5 min idle
  volumes:
    - ollama-models:/root/.ollama
  ports:
    - "11434:11434"
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
    interval: 30s
    timeout: 10s
    retries: 3
```

**Immediate Action Script**:

```bash
#!/bin/bash
# fix_ollama_cpu.sh

echo "Fixing Ollama CPU usage..."

# Update running container resource limits
docker update --cpus="4.0" --memory="4g" ollama

# Set environment variables
docker exec ollama sh -c 'echo "export OLLAMA_NUM_PARALLEL=2" >> ~/.bashrc'

# Restart with new limits
docker-compose stop ollama
docker-compose up -d ollama

echo "Ollama CPU limits applied. Monitoring..."
docker stats --no-stream ollama
```

### 1.2 Port 8080 Conflicts

**Issue**: Multiple services attempting to bind to port 8080.

**Remediation**:

1. **Identify conflicting services**:
```bash
#!/bin/bash
# find_port_conflicts.sh

echo "Services using port 8080:"
docker ps --format "table {{.Names}}\t{{.Ports}}" | grep 8080

# Find in docker-compose files
echo -e "\nDocker-compose definitions using 8080:"
grep -r "8080:" *.yml
```

2. **Update service configurations**:
```yaml
# Service port reassignments
services:
  # Change any service using 8080 to allocated range
  frontend:
    ports:
      - "10001:8501"  # Streamlit default
      
  backend-api:
    ports:
      - "10000:8000"  # FastAPI default
      
  # Only Kong/Traefik should use 8080 internally
  kong:
    ports:
      - "80:8080"     # External 80 -> Internal 8080
      - "443:8443"    # External 443 -> Internal 8443
```

### 1.3 Memory Limits for 34 Containers

**Issue**: 34 containers running without memory limits.

**Automated Fix Script**:

```python
#!/usr/bin/env python3
# apply_memory_limits.py

import docker
import yaml

client = docker.from_env()

# Default memory limits by container type
MEMORY_LIMITS = {
    'agent': '2Gi',
    'database': '4Gi',
    'cache': '2Gi',
    'monitoring': '1Gi',
    'default': '1Gi'
}

def get_container_type(name):
    """Determine container type from name"""
    if 'postgres' in name or 'mysql' in name:
        return 'database'
    elif 'redis' in name or 'cache' in name:
        return 'cache'
    elif 'prometheus' in name or 'grafana' in name:
        return 'monitoring'
    elif 'agent' in name:
        return 'agent'
    return 'default'

def apply_limits():
    """Apply memory limits to all containers"""
    containers = client.containers.list()
    
    for container in containers:
        # Check if memory limit exists
        if not container.attrs['HostConfig'].get('Memory'):
            container_type = get_container_type(container.name)
            memory_limit = MEMORY_LIMITS[container_type]
            
            print(f"Applying {memory_limit} memory limit to {container.name}")
            
            # Convert to bytes
            memory_bytes = int(memory_limit.replace('Gi', '')) * 1024 * 1024 * 1024
            
            try:
                container.update(mem_limit=memory_bytes)
            except Exception as e:
                print(f"Failed to update {container.name}: {e}")

if __name__ == "__main__":
    apply_limits()
```

---

## ⚠️ Priority 2: High-Impact Issues (Fix Within 48 Hours)

### 2.1 Multiple Claude Instances (2.4GB RAM)

**Issue**: Multiple Claude containers consuming excessive memory.

**Solution**: Implement Claude Instance Pool

```python
# claude_pool_manager.py

import os
import threading
from queue import Queue
import docker

class ClaudePoolManager:
    """Manage a pool of Claude instances"""
    
    def __init__(self, max_instances=3, memory_per_instance="800Mi"):
        self.max_instances = max_instances
        self.memory_limit = memory_per_instance
        self.available_instances = Queue()
        self.all_instances = []
        self.docker_client = docker.from_env()
        
        # Initialize pool
        self._initialize_pool()
        
    def _initialize_pool(self):
        """Create initial Claude instances"""
        for i in range(self.max_instances):
            instance = self._create_instance(i)
            self.available_instances.put(instance)
            self.all_instances.append(instance)
            
    def _create_instance(self, index):
        """Create a single Claude instance"""
        container = self.docker_client.containers.run(
            "claude:latest",
            name=f"claude-pool-{index}",
            detach=True,
            mem_limit=self.memory_limit,
            cpu_quota=50000,  # 0.5 CPU
            environment={
                "INSTANCE_ID": str(index),
                "POOL_MODE": "true"
            },
            network="agents",
            restart_policy={"Name": "unless-stopped"}
        )
        return container
        
    def get_instance(self, timeout=30):
        """Get an available instance from the pool"""
        return self.available_instances.get(timeout=timeout)
        
    def return_instance(self, instance):
        """Return instance to the pool"""
        self.available_instances.put(instance)
        
    def cleanup(self):
        """Stop and remove all instances"""
        for instance in self.all_instances:
            instance.stop()
            instance.remove()

# Usage
pool = ClaudePoolManager(max_instances=3)
```

**Migration Script**:

```bash
#!/bin/bash
# migrate_claude_instances.sh

echo "Migrating Claude instances to pool..."

# Stop all current Claude instances
docker ps --filter "name=claude" -q | xargs -r docker stop

# Remove old instances
docker ps -a --filter "name=claude" -q | xargs -r docker rm

# Deploy new pooled configuration
docker-compose -f docker-compose.claude-pool.yml up -d

echo "Claude pool deployed with 3 instances"
```

### 2.2 Container Restart Loops

**Issue**: Containers in restart loops due to insufficient resources or configuration issues.

**Detection and Resolution Script**:

```python
#!/usr/bin/env python3
# fix_restart_loops.py

import docker
import time
from datetime import datetime, timedelta

client = docker.from_env()

def detect_restart_loops():
    """Detect containers in restart loops"""
    problem_containers = []
    
    for container in client.containers.list(all=True):
        # Get container events
        events = client.events(
            since=datetime.now() - timedelta(minutes=10),
            until=datetime.now(),
            filters={'container': container.id}
        )
        
        restart_count = 0
        for event in events:
            if event['Action'] == 'restart':
                restart_count += 1
                
        if restart_count > 3:
            problem_containers.append({
                'name': container.name,
                'restarts': restart_count,
                'status': container.status,
                'logs': container.logs(tail=50).decode('utf-8')
            })
            
    return problem_containers

def fix_container(container_info):
    """Apply fixes to problematic container"""
    name = container_info['name']
    logs = container_info['logs']
    
    # Common fixes based on log patterns
    if 'out of memory' in logs.lower():
        print(f"Increasing memory for {name}")
        # Increase memory limit
        container = client.containers.get(name)
        current_mem = container.attrs['HostConfig']['Memory']
        new_mem = int(current_mem * 1.5) if current_mem else 1073741824  # 1GB default
        container.update(mem_limit=new_mem)
        
    elif 'connection refused' in logs.lower():
        print(f"Fixing dependencies for {name}")
        # Add restart delay
        container = client.containers.get(name)
        container.update(restart_policy={
            'Name': 'on-failure',
            'MaximumRetryCount': 3
        })
        
    elif 'address already in use' in logs.lower():
        print(f"Fixing port conflict for {name}")
        # Will need manual intervention
        print(f"Manual intervention required for {name} - port conflict")

# Run fixes
problems = detect_restart_loops()
for problem in problems:
    print(f"\nFixing {problem['name']} (restarted {problem['restarts']} times)")
    fix_container(problem)
```

### 2.3 Unbalanced CPU Load

**Issue**: CPU cores not evenly utilized due to lack of affinity settings.

**CPU Affinity Configuration**:

```yaml
# cpu_affinity_config.yml

services:
  # Infrastructure services on cores 0-3
  postgres:
    cpuset: "0-1"
    
  redis:
    cpuset: "2"
    
  consul:
    cpuset: "3"
    
  # Critical agents on cores 4-7
  ai-system-architect:
    cpuset: "4-5"
    
  orchestration-agent:
    cpuset: "6-7"
    
  # Standard agents on cores 8-11
  # Use round-robin assignment
```

**Auto-assignment Script**:

```python
#!/usr/bin/env python3
# assign_cpu_affinity.py

import docker
import yaml

client = docker.from_env()

# CPU allocation strategy
CPU_ALLOCATION = {
    'infrastructure': list(range(0, 4)),    # Cores 0-3
    'critical': list(range(4, 8)),         # Cores 4-7
    'standard': list(range(8, 12)),        # Cores 8-11
}

def get_service_category(name):
    """Categorize service by name"""
    infra_keywords = ['postgres', 'redis', 'consul', 'kong', 'rabbitmq']
    critical_keywords = ['architect', 'orchestrat', 'core', 'critical']
    
    name_lower = name.lower()
    
    if any(keyword in name_lower for keyword in infra_keywords):
        return 'infrastructure'
    elif any(keyword in name_lower for keyword in critical_keywords):
        return 'critical'
    else:
        return 'standard'

def assign_cpu_affinity():
    """Assign CPU affinity to all containers"""
    containers = client.containers.list()
    
    # Track CPU usage per category
    cpu_usage = {
        'infrastructure': 0,
        'critical': 0,
        'standard': 0
    }
    
    for container in containers:
        category = get_service_category(container.name)
        cores = CPU_ALLOCATION[category]
        
        # Round-robin within category
        core_index = cpu_usage[category] % len(cores)
        assigned_core = cores[core_index]
        
        print(f"Assigning {container.name} to core {assigned_core}")
        
        try:
            # Update container with CPU affinity
            container.update(cpuset_cpus=str(assigned_core))
            cpu_usage[category] += 1
        except Exception as e:
            print(f"Failed to update {container.name}: {e}")

if __name__ == "__main__":
    assign_cpu_affinity()
```

---

## 📋 Priority 3: Preventive Measures (Implement Within 1 Week)

### 3.1 Automated Compliance Monitoring

**Deploy Compliance Dashboard**:

```yaml
# docker-compose.compliance.yml

version: '3.8'

services:
  compliance-monitor:
    image: sutazai/compliance-monitor:latest
    container_name: compliance-monitor
    ports:
      - "10499:8080"
    environment:
      - DOCKER_HOST=unix:///var/run/docker.sock
      - ALERT_WEBHOOK=http://alerts:3002/webhook
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./ARCHITECTURE_RULES.md:/app/rules.md:ro
    deploy:
      resources:
        limits:
          memory: 512Mi
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
    restart: unless-stopped
```

### 3.2 Resource Usage Forecasting

**Predictive Resource Management**:

```python
# resource_forecaster.py

import numpy as np
from sklearn.linear_model import LinearRegression
import prometheus_client
import pickle

class ResourceForecaster:
    """Predict future resource usage"""
    
    def __init__(self):
        self.models = {}
        self.history_window = 24 * 60  # 24 hours of minutes
        
    def train_model(self, service_name, metric_type):
        """Train prediction model for service"""
        # Fetch historical data from Prometheus
        data = self._fetch_metrics(service_name, metric_type)
        
        # Prepare time series data
        X = np.arange(len(data)).reshape(-1, 1)
        y = np.array(data)
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        self.models[f"{service_name}_{metric_type}"] = model
        
    def predict_usage(self, service_name, metric_type, hours_ahead=6):
        """Predict resource usage"""
        model_key = f"{service_name}_{metric_type}"
        
        if model_key not in self.models:
            return None
            
        model = self.models[model_key]
        
        # Predict future values
        future_points = hours_ahead * 60
        X_future = np.arange(self.history_window, 
                            self.history_window + future_points).reshape(-1, 1)
        
        predictions = model.predict(X_future)
        
        return {
            'max_predicted': np.max(predictions),
            'avg_predicted': np.mean(predictions),
            'trend': 'increasing' if predictions[-1] > predictions[0] else 'decreasing'
        }
```

### 3.3 Automated Resource Optimization

**Dynamic Resource Adjustment**:

```python
# auto_optimizer.py

class AutoResourceOptimizer:
    """Automatically optimize resource allocations"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.adjustment_history = {}
        
    def optimize_all_containers(self):
        """Optimize resources for all containers"""
        containers = self.docker_client.containers.list()
        
        for container in containers:
            self._optimize_container(container)
            
    def _optimize_container(self, container):
        """Optimize single container resources"""
        stats = container.stats(stream=False)
        
        # Calculate usage percentages
        memory_usage = stats['memory_stats']['usage']
        memory_limit = stats['memory_stats']['limit']
        memory_percent = (memory_usage / memory_limit) * 100
        
        # CPU calculation
        cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                   stats['precpu_stats']['cpu_usage']['total_usage']
        system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                      stats['precpu_stats']['system_cpu_usage']
        cpu_percent = (cpu_delta / system_delta) * 100
        
        # Apply optimizations
        if memory_percent < 30 and container.name in self.adjustment_history:
            # Reduce memory if consistently underutilized
            new_limit = int(memory_limit * 0.75)
            container.update(mem_limit=new_limit)
            print(f"Reduced memory for {container.name} to {new_limit}")
            
        elif memory_percent > 85:
            # Increase memory if high usage
            new_limit = int(memory_limit * 1.25)
            container.update(mem_limit=new_limit)
            print(f"Increased memory for {container.name} to {new_limit}")
            
        self.adjustment_history[container.name] = {
            'timestamp': datetime.now(),
            'memory_percent': memory_percent,
            'cpu_percent': cpu_percent
        }
```

---

## 📊 Implementation Timeline

### Day 1 (First 24 Hours)
- [ ] Apply Ollama CPU limits
- [ ] Fix port 8080 conflicts  
- [ ] Apply memory limits to all containers
- [ ] Stop container restart loops

### Day 2-3 (48-72 Hours)
- [ ] Migrate Claude instances to pool
- [ ] Implement CPU affinity settings
- [ ] Deploy compliance monitoring
- [ ] Set up health check alerts

### Week 1
- [ ] Deploy resource forecasting
- [ ] Implement auto-optimization
- [ ] Create runbooks for common issues
- [ ] Train team on new procedures

### Ongoing
- [ ] Weekly compliance reports
- [ ] Monthly resource optimization review
- [ ] Quarterly architecture assessment
- [ ] Continuous monitoring and adjustment

---

## 🔍 Validation Checklist

After implementing each fix, validate:

- [ ] No container using >80% of allocated CPU
- [ ] All containers have memory limits
- [ ] No port conflicts exist
- [ ] Health checks passing for all services
- [ ] No containers in restart loops
- [ ] CPU cores evenly utilized
- [ ] Monitoring dashboard shows green status
- [ ] Alerts configured and working

---

## 📞 Emergency Procedures

If system becomes unstable during remediation:

1. **Rollback Script**:
```bash
#!/bin/bash
# emergency_rollback.sh

# Restore previous docker-compose
git checkout HEAD~1 docker-compose*.yml

# Stop all containers
docker-compose down

# Start with previous configuration
docker-compose up -d

# Verify critical services
./scripts/verify_critical_services.sh
```

2. **Emergency Resource Release**:
```bash
#!/bin/bash
# emergency_cleanup.sh

# Stop non-critical agents
docker stop $(docker ps -q --filter "label=priority=low")

# Clear caches
docker exec redis redis-cli FLUSHALL

# Restart critical services only
docker-compose up -d postgres redis consul ollama
```

---

This remediation plan provides concrete steps to resolve all identified issues while maintaining system stability throughout the process.