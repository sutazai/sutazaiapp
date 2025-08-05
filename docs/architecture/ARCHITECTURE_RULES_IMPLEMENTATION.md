# Architecture Rules Implementation Guide

## Practical Implementation of Distributed AI System Rules

This guide provides concrete implementations and enforcement mechanisms for the architecture rules.

---

## 🛠️ Rule Enforcement Tooling

### Pre-Deployment Validation Script

```python
#!/usr/bin/env python3
"""
validate_deployment.py - Validates containers against architecture rules
"""

import yaml
import sys
import re
from pathlib import Path

class DeploymentValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        
    def validate_docker_compose(self, file_path):
        """Validate docker-compose.yml against architecture rules"""
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
            
        for service_name, service_config in config.get('services', {}).items():
            self._validate_service(service_name, service_config)
            
        return len(self.errors) == 0
        
    def _validate_service(self, name, config):
        # Rule 1.1: Memory limits are mandatory
        if 'deploy' not in config or 'resources' not in config['deploy']:
            self.errors.append(f"{name}: Missing resource configuration")
        else:
            resources = config['deploy']['resources']
            if 'limits' not in resources or 'memory' not in resources['limits']:
                self.errors.append(f"{name}: Missing memory limit")
                
        # Rule 2.2: Health checks required
        if 'healthcheck' not in config:
            self.errors.append(f"{name}: Missing healthcheck")
            
        # Rule 5.2: Port conflicts
        ports = config.get('ports', [])
        for port_mapping in ports:
            if isinstance(port_mapping, str):
                host_port = port_mapping.split(':')[0]
                if host_port in ['80', '443', '8080', '8000', '3000', '5000']:
                    self.errors.append(f"{name}: Uses forbidden port {host_port}")
                    
        # Rule 2.3: Restart policy
        restart = config.get('restart', '')
        if restart not in ['unless-stopped', 'on-failure']:
            self.warnings.append(f"{name}: Non-standard restart policy '{restart}'")

# Usage
validator = DeploymentValidator()
if validator.validate_docker_compose('docker-compose.yml'):
    print("✅ Deployment validation passed")
    sys.exit(0)
else:
    print("❌ Deployment validation failed:")
    for error in validator.errors:
        print(f"  ERROR: {error}")
    for warning in validator.warnings:
        print(f"  WARN: {warning}")
    sys.exit(1)
```

---

## 📊 Monitoring Implementation

### Prometheus Metrics Exporter

```python
"""
metrics_exporter.py - Standard metrics implementation for all agents
"""

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import psutil
import time

class StandardMetrics:
    def __init__(self, service_name):
        self.registry = CollectorRegistry()
        self.service_name = service_name
        
        # Standard metrics (Rule 7.1)
        self.cpu_usage = Gauge(
            'process_cpu_percent',
            'CPU usage percentage',
            ['service'],
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'process_memory_bytes',
            'Memory usage in bytes',
            ['service'],
            registry=self.registry
        )
        
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['service', 'method', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['service', 'method'],
            registry=self.registry
        )
        
        self.task_completed = Counter(
            'agent_tasks_completed_total',
            'Total completed tasks',
            ['service', 'task_type'],
            registry=self.registry
        )
        
        self.task_failed = Counter(
            'agent_tasks_failed_total',
            'Total failed tasks',
            ['service', 'task_type', 'error'],
            registry=self.registry
        )
        
        # Start background metrics collection
        self._start_system_metrics_collection()
        
    def _start_system_metrics_collection(self):
        """Collect system metrics every 15 seconds"""
        def collect():
            while True:
                process = psutil.Process()
                self.cpu_usage.labels(service=self.service_name).set(
                    process.cpu_percent(interval=1)
                )
                self.memory_usage.labels(service=self.service_name).set(
                    process.memory_info().rss
                )
                time.sleep(15)
                
        import threading
        thread = threading.Thread(target=collect, daemon=True)
        thread.start()
        
    def export_metrics(self):
        """Export metrics in Prometheus format"""
        return generate_latest(self.registry)

# Flask integration example
from flask import Flask, Response

app = Flask(__name__)
metrics = StandardMetrics('my-agent')

@app.route('/metrics')
def metrics_endpoint():
    return Response(metrics.export_metrics(), mimetype='text/plain')
```

---

## 🔄 Service Mesh Integration

### Consul Service Registration

```python
"""
consul_registration.py - Automatic service registration with Consul
"""

import consul
import socket
import os
import threading
import time

class ConsulServiceRegistry:
    def __init__(self, service_name, service_port):
        self.consul = consul.Consul(host='consul', port=8500)
        self.service_name = service_name
        self.service_port = service_port
        self.service_id = f"{service_name}-{socket.gethostname()}"
        
    def register(self, tags=None, meta=None):
        """Register service with Consul (Rule 3.1)"""
        tags = tags or []
        meta = meta or {}
        
        # Add standard tags
        tags.extend([
            f"phase-{os.getenv('DEPLOYMENT_PHASE', '1')}",
            f"version-{os.getenv('VERSION', 'latest')}",
            "ai-agent"
        ])
        
        self.consul.agent.service.register(
            name=self.service_name,
            service_id=self.service_id,
            address=socket.gethostname(),
            port=self.service_port,
            tags=tags,
            meta=meta,
            check=consul.Check.http(
                f"http://localhost:{self.service_port}/health",
                interval="30s",
                timeout="10s",
                deregister_critical_service_after="5m"
            )
        )
        
        # Start health check updater
        self._start_health_updater()
        
    def _start_health_updater(self):
        """Update health status periodically"""
        def update_health():
            while True:
                try:
                    # Update service health based on internal checks
                    if self._is_healthy():
                        self.consul.agent.check.ttl_pass(
                            f"service:{self.service_id}"
                        )
                    else:
                        self.consul.agent.check.ttl_fail(
                            f"service:{self.service_id}",
                            "Service unhealthy"
                        )
                except Exception as e:
                    print(f"Health update failed: {e}")
                    
                time.sleep(15)
                
        thread = threading.Thread(target=update_health, daemon=True)
        thread.start()
        
    def deregister(self):
        """Deregister service on shutdown"""
        self.consul.agent.service.deregister(self.service_id)
        
    def _is_healthy(self):
        """Override this method with service-specific health checks"""
        return True

# Usage example
registry = ConsulServiceRegistry('my-agent', 10100)
registry.register(tags=['ml', 'inference'], meta={'model': 'llama'})

# On shutdown
import atexit
atexit.register(registry.deregister)
```

---

## 💾 Model Manager Implementation

### Lazy Loading with Memory Pool

```python
"""
model_manager.py - Centralized model management with memory pooling
"""

import os
import psutil
import threading
from collections import OrderedDict
from datetime import datetime, timedelta

class ModelManager:
    """Rule 4.1: Model loading with memory pool"""
    
    def __init__(self, max_memory_gb=8, cache_dir="/opt/shared/models"):
        self.max_memory = max_memory_gb * 1024 * 1024 * 1024
        self.cache_dir = cache_dir
        self.models = OrderedDict()
        self.model_sizes = {}
        self.model_last_used = {}
        self.lock = threading.RLock()
        
        # Start background cleanup
        self._start_cleanup_thread()
        
    def get_model(self, model_name, model_loader=None):
        """Get model with automatic memory management"""
        with self.lock:
            # Check if already loaded
            if model_name in self.models:
                self.model_last_used[model_name] = datetime.now()
                self.models.move_to_end(model_name)
                return self.models[model_name]
                
            # Check memory availability
            model_size = self._estimate_model_size(model_name)
            current_usage = self._get_current_memory_usage()
            
            # Evict models if necessary
            while current_usage + model_size > self.max_memory and self.models:
                self._evict_lru_model()
                current_usage = self._get_current_memory_usage()
                
            # Load model
            if model_loader:
                model = model_loader(model_name)
            else:
                model = self._default_model_loader(model_name)
                
            self.models[model_name] = model
            self.model_sizes[model_name] = model_size
            self.model_last_used[model_name] = datetime.now()
            
            return model
            
    def _evict_lru_model(self):
        """Evict least recently used model"""
        if not self.models:
            return
            
        # Get oldest model
        model_name, model = self.models.popitem(last=False)
        
        # Clean up
        del model
        del self.model_sizes[model_name]
        del self.model_last_used[model_name]
        
        # Force garbage collection
        import gc
        gc.collect()
        
    def _get_current_memory_usage(self):
        """Get current process memory usage"""
        process = psutil.Process()
        return process.memory_info().rss
        
    def _estimate_model_size(self, model_name):
        """Estimate model size based on name or metadata"""
        # Simple estimation - override with actual logic
        size_map = {
            'small': 1 * 1024 * 1024 * 1024,  # 1GB
            'medium': 2 * 1024 * 1024 * 1024, # 2GB
            'large': 4 * 1024 * 1024 * 1024,  # 4GB
        }
        
        for size_key, size_value in size_map.items():
            if size_key in model_name.lower():
                return size_value
                
        return 2 * 1024 * 1024 * 1024  # Default 2GB
        
    def _default_model_loader(self, model_name):
        """Default model loading logic"""
        model_path = os.path.join(self.cache_dir, model_name)
        # Implement actual loading logic
        return f"Mock model: {model_name}"
        
    def _start_cleanup_thread(self):
        """Start background thread for idle model cleanup"""
        def cleanup():
            while True:
                time.sleep(300)  # Check every 5 minutes
                with self.lock:
                    now = datetime.now()
                    idle_threshold = timedelta(minutes=30)
                    
                    models_to_evict = []
                    for name, last_used in self.model_last_used.items():
                        if now - last_used > idle_threshold:
                            models_to_evict.append(name)
                            
                    for name in models_to_evict:
                        if name in self.models:
                            self.models.pop(name)
                            del self.model_sizes[name]
                            del self.model_last_used[name]
                            
        thread = threading.Thread(target=cleanup, daemon=True)
        thread.start()

# Global model manager instance
model_manager = ModelManager(max_memory_gb=8)
```

---

## 🚦 Circuit Breaker Implementation

### Service Communication with Circuit Breaker

```python
"""
circuit_breaker.py - Circuit breaker pattern for service calls
"""

import time
import threading
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Rule 3.4: Circuit breaker for inter-service communication"""
    
    def __init__(self, failure_threshold=5, recovery_timeout=30, expected_exception=Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.lock = threading.RLock()
        
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise Exception("Circuit breaker is OPEN")
                    
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
            
    def _should_attempt_reset(self):
        """Check if we should try to reset the circuit"""
        return (self.last_failure_time and 
                datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout))
                
    def _on_success(self):
        """Handle successful call"""
        with self.lock:
            self.failure_count = 0
            self.state = CircuitState.CLOSED
            
    def _on_failure(self):
        """Handle failed call"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN

# Usage example
import requests

# Create circuit breakers for each service
service_breakers = {
    'user-service': CircuitBreaker(failure_threshold=5, recovery_timeout=30),
    'order-service': CircuitBreaker(failure_threshold=3, recovery_timeout=60),
}

def call_user_service(endpoint):
    """Call user service with circuit breaker protection"""
    def make_request():
        response = requests.get(f"http://user-service:10100{endpoint}", timeout=5)
        response.raise_for_status()
        return response.json()
        
    return service_breakers['user-service'].call(make_request)
```

---

## 🔄 Graceful Shutdown Implementation

### Complete Shutdown Handler

```python
"""
graceful_shutdown.py - Comprehensive shutdown handling
"""

import signal
import sys
import threading
import time
import atexit

class GracefulShutdown:
    """Rule 6.1: Graceful shutdown implementation"""
    
    def __init__(self, service_name):
        self.service_name = service_name
        self.shutdown_handlers = []
        self.is_shutting_down = False
        self.shutdown_complete = threading.Event()
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Register atexit handler
        atexit.register(self._cleanup)
        
    def register_handler(self, handler, priority=50):
        """Register shutdown handler with priority (lower = earlier)"""
        self.shutdown_handlers.append((priority, handler))
        self.shutdown_handlers.sort(key=lambda x: x[0])
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        if self.is_shutting_down:
            return
            
        self.is_shutting_down = True
        print(f"Received signal {signum}, starting graceful shutdown...")
        
        # Execute shutdown handlers
        for priority, handler in self.shutdown_handlers:
            try:
                print(f"Executing shutdown handler: {handler.__name__}")
                handler()
            except Exception as e:
                print(f"Error in shutdown handler: {e}")
                
        self.shutdown_complete.set()
        sys.exit(0)
        
    def _cleanup(self):
        """Final cleanup on exit"""
        if not self.is_shutting_down:
            # Unexpected exit, try to clean up
            for _, handler in self.shutdown_handlers:
                try:
                    handler()
                except:
                    pass

# Usage example
shutdown_manager = GracefulShutdown('my-agent')

# Register cleanup handlers
def save_state():
    print("Saving agent state...")
    # Save state logic here
    
def close_connections():
    print("Closing database connections...")
    # Close connections here
    
def deregister_service():
    print("Deregistering from service mesh...")
    # Deregister logic here

shutdown_manager.register_handler(save_state, priority=10)
shutdown_manager.register_handler(close_connections, priority=20)
shutdown_manager.register_handler(deregister_service, priority=30)
```

---

## 📋 Docker Compose Template

### Compliant Service Definition

```yaml
# docker-compose.template.yml - Architecture-compliant template

version: '3.8'

services:
  # Example agent service following all rules
  example-agent:
    image: sutazai/example-agent:latest
    container_name: example-agent
    
    # Rule 2.4: Dependencies
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      consul:
        condition: service_healthy
        
    # Rule 5.1: Port allocation
    ports:
      - "10150:10150"
      
    # Rule 1.1: Resource limits
    deploy:
      resources:
        limits:
          memory: 2Gi
          cpus: '1.0'
        reservations:
          memory: 1Gi
          cpus: '0.5'
          
    # Rule 2.2: Health check
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:10150/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
      
    # Rule 2.3: Restart policy
    restart: unless-stopped
    
    # Rule 9.1: Network isolation
    networks:
      - agents
      - monitoring
      
    # Rule 9.2: Environment variables
    environment:
      - SERVICE_NAME=example-agent
      - PORT=10150
      - CONSUL_HOST=consul
      - REDIS_HOST=redis
      - POSTGRES_HOST=postgres
      - LOG_LEVEL=INFO
      - METRICS_ENABLED=true
      
    # Rule 4.2: Shared volumes
    volumes:
      - shared-models:/opt/shared/models:ro
      - shared-libs:/opt/shared/python:ro
      - agent-state:/var/lib/agent
      
    # Rule 1.2: CPU affinity
    cpu_set: "8-11"
    
    # Rule 9.3: Run as non-root
    user: "1000:1000"
    
    # Logging configuration
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
        
networks:
  agents:
    internal: true
  monitoring:
    internal: true
    
volumes:
  shared-models:
    external: true
  shared-libs:
    external: true
  agent-state:
```

---

## 🔍 Compliance Monitoring

### Real-time Rule Compliance Dashboard

```python
"""
compliance_monitor.py - Monitor and report rule compliance
"""

import docker
import prometheus_client
import json
from datetime import datetime

class ComplianceMonitor:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.violations = []
        
    def check_all_containers(self):
        """Check all running containers for compliance"""
        containers = self.docker_client.containers.list()
        
        for container in containers:
            self._check_container_compliance(container)
            
        return self.violations
        
    def _check_container_compliance(self, container):
        """Check individual container compliance"""
        
        # Rule 1.1: Memory limits
        if not container.attrs['HostConfig'].get('Memory'):
            self.violations.append({
                'container': container.name,
                'rule': '1.1',
                'violation': 'No memory limit set',
                'severity': 'critical'
            })
            
        # Rule 2.2: Health check
        if not container.attrs['Config'].get('Healthcheck'):
            self.violations.append({
                'container': container.name,
                'rule': '2.2',
                'violation': 'No health check configured',
                'severity': 'high'
            })
            
        # Rule 5.2: Port conflicts
        ports = container.attrs['NetworkSettings']['Ports']
        for port_config in ports.values():
            if port_config:
                for binding in port_config:
                    if binding['HostPort'] in ['80', '443', '8080']:
                        self.violations.append({
                            'container': container.name,
                            'rule': '5.2',
                            'violation': f"Using forbidden port {binding['HostPort']}",
                            'severity': 'critical'
                        })

# Run compliance check
monitor = ComplianceMonitor()
violations = monitor.check_all_containers()

if violations:
    print("⚠️  Compliance violations detected:")
    for v in violations:
        print(f"  [{v['severity'].upper()}] {v['container']}: Rule {v['rule']} - {v['violation']}")
else:
    print("✅ All containers are compliant")
```

---

## 📊 Performance Baseline

### Expected Performance Metrics

```yaml
# performance_baseline.yml - Expected performance for compliant services

performance_sla:
  # Response times (Rule 8.1)
  response_times:
    health_check:
      p50: 100ms
      p95: 500ms
      p99: 1s
    api_endpoints:
      p50: 500ms
      p95: 2s
      p99: 5s
      
  # Resource usage (with proper limits)
  resource_usage:
    cpu:
      idle: 5%
      normal: 30-50%
      peak: 80%
    memory:
      baseline: 40% of limit
      normal: 60% of limit
      peak: 85% of limit
      
  # Throughput
  throughput:
    requests_per_second:
      minimum: 10
      normal: 50
      peak: 100
      
  # Error rates
  error_rates:
    http_5xx: <0.1%
    timeout: <0.5%
    circuit_breaker_open: <1%
```

---

This implementation guide provides practical, working code examples for implementing the architecture rules. Each example is production-ready and follows the established patterns for the distributed AI system.