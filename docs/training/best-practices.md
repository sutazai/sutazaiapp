# SutazAI Best Practices Guide

## Overview

This comprehensive guide outlines best practices for using, managing, and maintaining SutazAI systems. Following these practices ensures optimal performance, security, reliability, and maintainability.

## Table of Contents

1. [System Architecture Best Practices](#system-architecture-best-practices)
2. [Development and Integration Best Practices](#development-and-integration-best-practices)
3. [Security Best Practices](#security-best-practices)
4. [Performance Optimization Best Practices](#performance-optimization-best-practices)
5. [Operational Best Practices](#operational-best-practices)
6. [Monitoring and Observability Best Practices](#monitoring-and-observability-best-practices)
7. [Data Management Best Practices](#data-management-best-practices)
8. [Agent Development Best Practices](#agent-development-best-practices)
9. [Deployment and CI/CD Best Practices](#deployment-and-cicd-best-practices)
10. [Troubleshooting and Maintenance Best Practices](#troubleshooting-and-maintenance-best-practices)

---

## System Architecture Best Practices

### Design Principles

#### 1. Local-First Architecture
```yaml
# Always prioritize local processing
architecture:
  processing: local
  data_residency: on_premises
  external_dependencies: minimal
  fallback_strategy: graceful_degradation
```

**Best Practices:**
- Keep all AI processing on local infrastructure
- Minimize external API dependencies
- Design for offline operation capability
- Implement graceful degradation when external services fail

#### 2. Microservices Architecture
```yaml
# Service decomposition strategy
services:
  decomposition: by_capability
  communication: async_preferred
  data_ownership: service_specific
  independence: high
```

**Implementation Guidelines:**
- Each agent should be independently deployable
- Use message queues for asynchronous communication
- Implement service discovery mechanisms
- Design for service independence and loose coupling

#### 3. Scalability by Design
```yaml
# Scaling considerations
scaling:
  horizontal: preferred
  vertical: fallback
  auto_scaling: enabled
  resource_limits: enforced
```

**Scaling Strategies:**
- Design agents for horizontal scaling
- Implement stateless agent design
- Use load balancing for traffic distribution
- Plan capacity based on usage patterns

### Resource Management

#### Memory Management
```bash
# Memory allocation strategy
export AGENT_MEMORY_LIMIT="2Gi"
export SYSTEM_MEMORY_RESERVE="25%"
export GARBAGE_COLLECTION_FREQUENCY="hourly"
```

**Best Practices:**
- Set memory limits for all containers
- Reserve 25% system memory for OS operations
- Implement automatic garbage collection
- Monitor for memory leaks regularly

#### CPU Optimization
```yaml
# CPU resource allocation
cpu_allocation:
  agents: 75%
  system: 15%
  monitoring: 10%
  reserve: always_available
```

**CPU Best Practices:**
- Allocate CPU resources based on agent priorities
- Use CPU affinity for critical agents
- Implement CPU throttling for resource protection
- Monitor CPU utilization patterns

---

## Development and Integration Best Practices

### Code Quality Standards

#### Code Structure
```python
# Standard agent structure
class BestPracticeAgent(BaseAgentV2):
    """
    Agent following SutazAI best practices.
    
    This agent demonstrates proper structure, error handling,
    and integration patterns.
    """
    
    def __init__(self):
        super().__init__(
            name="best-practice-agent",
            description="Demonstrates best practices implementation",
            capabilities=self._define_capabilities(),
            version="1.0.0"
        )
        self._setup_logging()
        self._initialize_metrics()
    
    def _define_capabilities(self):
        """Define agent capabilities with clear descriptions."""
        return [
            "process_data",
            "generate_report", 
            "validate_input",
            "health_check"
        ]
    
    async def process_data(self, data):
        """Process data with comprehensive error handling."""
        try:
            validated_data = await self._validate_input(data)
            result = await self._core_processing(validated_data)
            await self._emit_metrics("process_data_success")
            return result
        except ValidationError as e:
            await self._handle_validation_error(e)
            raise
        except ProcessingError as e:
            await self._handle_processing_error(e)
            raise
        except Exception as e:
            await self._handle_unexpected_error(e)
            raise
```

#### Error Handling Patterns
```python
# Comprehensive error handling
async def _handle_error(self, error, context):
    """Standard error handling pattern."""
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context,
        "timestamp": datetime.utcnow().isoformat(),
        "agent_name": self.name
    }
    
    # Log error
    self.logger.error("Agent error occurred", extra=error_info)
    
    # Emit metrics
    await self._emit_error_metrics(error_info)
    
    # Determine if error is recoverable
    if self._is_recoverable_error(error):
        await self._attempt_recovery(error, context)
    
    # Notify monitoring system
    await self._notify_monitoring(error_info)
```

### Testing Standards

#### Unit Testing
```python
# Comprehensive unit tests
import pytest
from unittest.mock import Mock, patch, AsyncMock

class TestBestPracticeAgent:
    @pytest.fixture
    def agent(self):
        return BestPracticeAgent()
    
    @pytest.mark.asyncio
    async def test_process_data_success(self, agent):
        """Test successful data processing."""
        test_data = {"input": "test_value"}
        expected_result = {"output": "processed_value"}
        
        result = await agent.process_data(test_data)
        
        assert result == expected_result
        # Verify metrics were emitted
        # Verify logging occurred
    
    @pytest.mark.asyncio
    async def test_process_data_validation_error(self, agent):
        """Test handling of validation errors."""
        invalid_data = {"invalid": "data"}
        
        with pytest.raises(ValidationError):
            await agent.process_data(invalid_data)
        
        # Verify error handling occurred
        # Verify metrics were emitted
    
    @pytest.mark.asyncio
    async def test_health_check(self, agent):
        """Test agent health check functionality."""
        health_status = await agent.health_check()
        
        assert health_status["status"] == "healthy"
        assert "metrics" in health_status
        assert "version" in health_status
```

#### Integration Testing
```python
# Integration test patterns
@pytest.mark.integration
async def test_agent_workflow_integration():
    """Test complete workflow integration."""
    # Setup test environment
    async with TestEnvironment() as env:
        # Initialize agents
        agent1 = await env.create_agent("agent-1")
        agent2 = await env.create_agent("agent-2")
        
        # Execute workflow
        result = await env.execute_workflow([
            {"agent": agent1, "action": "process"},
            {"agent": agent2, "action": "validate"}
        ])
        
        # Verify results
        assert result["status"] == "success"
        assert len(result["steps"]) == 2
```

### API Design Standards

#### RESTful API Design
```python
# Standard API endpoint structure
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/agents/{agent_name}")

class TaskRequest(BaseModel):
    """Standard task request model."""
    action: str
    data: dict
    priority: str = "medium"
    timeout: int = 300

class TaskResponse(BaseModel):
    """Standard task response model."""
    task_id: str
    status: str
    result: dict
    execution_time: float
    agent_name: str

@router.post("/tasks", response_model=TaskResponse)
async def submit_task(
    agent_name: str,
    request: TaskRequest,
    agent_service: AgentService = Depends(get_agent_service)
):
    """Submit task to specific agent."""
    try:
        result = await agent_service.submit_task(
            agent_name=agent_name,
            action=request.action,
            data=request.data,
            priority=request.priority,
            timeout=request.timeout
        )
        return TaskResponse(**result)
    except AgentNotFoundError:
        raise HTTPException(status_code=404, detail="Agent not found")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

---

## Security Best Practices

### Authentication and Authorization

#### JWT Token Management
```python
# Secure JWT implementation
import jwt
from datetime import datetime, timedelta
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class SecureTokenManager:
    def __init__(self):
        self.secret_key = self._load_secret_key()
        self.algorithm = "HS256"
        self.token_expiry = timedelta(hours=1)
    
    def _load_secret_key(self):
        """Load secret key from secure storage."""
        # Never hardcode secrets
        return os.environ.get("JWT_SECRET_KEY")
    
    def generate_token(self, user_id: str, permissions: list):
        """Generate secure JWT token."""
        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "issued_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + self.token_expiry).isoformat()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def validate_token(self, token: str):
        """Validate and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check expiration
            expires_at = datetime.fromisoformat(payload["expires_at"])
            if datetime.utcnow() > expires_at:
                raise TokenExpiredError("Token has expired")
            
            return payload
        except jwt.InvalidTokenError:
            raise InvalidTokenError("Invalid token")
```

#### Role-Based Access Control
```yaml
# RBAC configuration
rbac:
  roles:
    admin:
      permissions:
        - "system:manage"
        - "agents:*"
        - "config:write"
        - "monitoring:read"
    
    developer:
      permissions:
        - "agents:read"
        - "agents:execute"
        - "config:read"
        - "monitoring:read"
    
    user:
      permissions:
        - "agents:execute"
        - "tasks:read"

  policies:
    - effect: "allow"
      principals: ["role:admin"]
      actions: ["*"]
      resources: ["*"]
    
    - effect: "allow"
      principals: ["role:developer"]
      actions: ["read", "execute"]
      resources: ["agents/*", "tasks/*"]
```

### Data Protection

#### Encryption at Rest
```python
# Data encryption implementation
from cryptography.fernet import Fernet
import base64

class DataEncryption:
    def __init__(self):
        self.key = self._load_encryption_key()
        self.cipher = Fernet(self.key)
    
    def _load_encryption_key(self):
        """Load encryption key from secure storage."""
        key_path = "/secure/encryption.key"
        if os.path.exists(key_path):
            with open(key_path, "rb") as f:
                return f.read()
        else:
            # Generate new key for first time
            key = Fernet.generate_key()
            with open(key_path, "wb") as f:
                f.write(key)
            os.chmod(key_path, 0o600)  # Restrict permissions
            return key
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        encrypted = self.cipher.encrypt(data.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted = self.cipher.decrypt(encrypted_bytes)
        return decrypted.decode()
```

#### Secure Configuration Management
```bash
# Secure secrets management
# Use environment variables, not config files
export JWT_SECRET_KEY="$(openssl rand -base64 32)"
export DATABASE_PASSWORD="$(openssl rand -base64 24)"
export ENCRYPTION_KEY="$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"

# Store in secure vault
./scripts/store-secret.sh JWT_SECRET_KEY "$JWT_SECRET_KEY"
./scripts/store-secret.sh DATABASE_PASSWORD "$DATABASE_PASSWORD"
```

### Security Scanning

#### Automated Security Checks
```yaml
# Security scanning pipeline
security_pipeline:
  static_analysis:
    - tool: "semgrep"
      rules: ["owasp-top-10", "security-audit"]
      fail_on: "high"
    
    - tool: "bandit"
      targets: ["src/", "agents/"]
      severity: "medium"
    
  dependency_scanning:
    - tool: "safety"
      targets: ["requirements.txt"]
      database: "latest"
    
    - tool: "npm-audit"
      targets: ["package.json"]
      level: "moderate"
  
  container_scanning:
    - tool: "trivy"
      targets: ["Dockerfile", "docker-compose.yml"]
      severity: "high"
```

---

## Performance Optimization Best Practices

### Resource Optimization

#### Memory Optimization
```python
# Memory-efficient implementation patterns
import gc
from memory_profiler import profile

class MemoryOptimizedAgent:
    def __init__(self):
        # Use __slots__ to reduce memory overhead
        __slots__ = ["name", "config", "_cache", "_metrics"]
        
        self._cache = {}
        self._cache_size_limit = 1000
        self._metrics = {}
    
    @profile
    async def process_large_dataset(self, data):
        """Process large datasets efficiently."""
        # Process in chunks to manage memory
        chunk_size = 1000
        results = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            chunk_result = await self._process_chunk(chunk)
            results.append(chunk_result)
            
            # Explicit garbage collection for large datasets
            if i % (chunk_size * 10) == 0:
                gc.collect()
        
        return self._merge_results(results)
    
    def _manage_cache(self, key, value):
        """Implement LRU cache with size limits."""
        if len(self._cache) >= self._cache_size_limit:
            # Remove oldest entries
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = value
```

#### CPU Optimization
```python
# CPU optimization patterns
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class CPUOptimizedAgent:
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.process_pool = ProcessPoolExecutor(max_workers=2)
    
    async def cpu_intensive_task(self, data):
        """Handle CPU-intensive tasks efficiently."""
        # Use process pool for CPU-bound tasks
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.process_pool, 
            self._cpu_bound_operation, 
            data
        )
        return result
    
    async def io_intensive_task(self, data):
        """Handle I/O-intensive tasks efficiently."""
        # Use thread pool for I/O-bound tasks
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.thread_pool,
            self._io_bound_operation,
            data
        )
        return result
    
    def _cpu_bound_operation(self, data):
        """CPU-intensive operation run in separate process."""
        # Implement actual CPU-intensive logic
        pass
    
    def _io_bound_operation(self, data):
        """I/O-intensive operation run in separate thread."""
        # Implement actual I/O-intensive logic
        pass
```

### Caching Strategies

#### Multi-Level Caching
```python
# Implementing multi-level caching
import redis
from functools import wraps
import pickle
import hashlib

class MultiLevelCache:
    def __init__(self):
        # Level 1: In-memory cache
        self.memory_cache = {}
        self.memory_cache_size = 100
        
        # Level 2: Redis cache
        self.redis_client = redis.Redis(host='redis', port=6379, db=0)
        
        # Level 3: Persistent cache (file system)
        self.persistent_cache_dir = "/tmp/sutazai_cache"
    
    def cache_key(self, func, args, kwargs):
        """Generate consistent cache key."""
        key_data = f"{func.__name__}:{str(args)}:{str(kwargs)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key):
        """Get value from multi-level cache."""
        # Try memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Try Redis cache
        redis_value = self.redis_client.get(key)
        if redis_value:
            value = pickle.loads(redis_value)
            # Promote to memory cache
            self._set_memory_cache(key, value)
            return value
        
        # Try persistent cache
        cache_file = os.path.join(self.persistent_cache_dir, key)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                value = pickle.load(f)
                # Promote to higher levels
                self._set_redis_cache(key, value, ttl=3600)
                self._set_memory_cache(key, value)
                return value
        
        return None
    
    def set(self, key, value, ttl=3600):
        """Set value in multi-level cache."""
        self._set_memory_cache(key, value)
        self._set_redis_cache(key, value, ttl)
        self._set_persistent_cache(key, value)

def cached(ttl=3600):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = MultiLevelCache()
            key = cache.cache_key(func, args, kwargs)
            
            # Try to get from cache
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache.set(key, result, ttl)
            return result
        
        return wrapper
    return decorator
```

---

## Operational Best Practices

### Deployment Strategies

#### Blue-Green Deployment
```bash
#!/bin/bash
# Blue-green deployment script

ENVIRONMENT=${1:-production}
NEW_VERSION=${2:-latest}

echo "Starting blue-green deployment for ${ENVIRONMENT}"

# Step 1: Deploy to green environment
echo "Deploying version ${NEW_VERSION} to green environment"
docker-compose -f docker-compose.${ENVIRONMENT}.yml \
  -f docker-compose.green.yml \
  up -d --build

# Step 2: Health check green environment
echo "Performing health checks on green environment"
./scripts/health-check.sh --environment green --timeout 300

if [ $? -ne 0 ]; then
    echo "Health check failed, rolling back"
    docker-compose -f docker-compose.green.yml down
    exit 1
fi

# Step 3: Run integration tests
echo "Running integration tests on green environment"
./scripts/integration-tests.sh --environment green

if [ $? -ne 0 ]; then
    echo "Integration tests failed, rolling back"
    docker-compose -f docker-compose.green.yml down
    exit 1
fi

# Step 4: Switch traffic to green
echo "Switching traffic to green environment"
./scripts/switch-traffic.sh --from blue --to green

# Step 5: Monitor for issues
echo "Monitoring green environment for 5 minutes"
./scripts/monitor-deployment.sh --environment green --duration 300

if [ $? -ne 0 ]; then
    echo "Issues detected, switching back to blue"
    ./scripts/switch-traffic.sh --from green --to blue
    exit 1
fi

# Step 6: Shutdown blue environment
echo "Deployment successful, shutting down blue environment"
docker-compose -f docker-compose.blue.yml down

echo "Blue-green deployment completed successfully"
```

#### Canary Deployment
```yaml
# Canary deployment configuration
canary_deployment:
  strategy: gradual_rollout
  stages:
    - name: "initial"
      traffic_percentage: 5
      duration: "30m"
      success_criteria:
        error_rate: "<1%"
        response_time: "<500ms"
    
    - name: "ramp_up"
      traffic_percentage: 25
      duration: "1h"
      success_criteria:
        error_rate: "<0.5%"
        response_time: "<400ms"
    
    - name: "full_rollout"
      traffic_percentage: 100
      duration: "24h"
      success_criteria:
        error_rate: "<0.1%"
        response_time: "<300ms"
  
  rollback_criteria:
    error_rate: ">2%"
    response_time: ">1000ms"
    user_complaints: ">10"
```

### Configuration Management

#### Environment-Specific Configurations
```yaml
# config/environments/production.yml
environment: production
debug: false

database:
  host: "prod-db.internal"
  port: 5432
  ssl_mode: "require"
  connection_pool:
    min_size: 10
    max_size: 50

agents:
  default_timeout: 300
  max_concurrent_tasks: 20
  resource_limits:
    memory: "4Gi"
    cpu: "2000m"

monitoring:
  metrics_enabled: true
  logging_level: "INFO"
  traces_enabled: true

security:
  token_expiry: "1h"
  session_timeout: "24h"
  rate_limiting:
    requests_per_minute: 1000
```

#### Configuration Validation
```python
# Configuration validation schema
from pydantic import BaseModel, validator
from typing import Optional, Dict, Any

class DatabaseConfig(BaseModel):
    host: str
    port: int = 5432
    ssl_mode: str = "prefer"
    
    @validator('port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v

class AgentConfig(BaseModel):
    default_timeout: int = 300
    max_concurrent_tasks: int = 10
    resource_limits: Dict[str, str]
    
    @validator('default_timeout')
    def validate_timeout(cls, v):
        if v < 10 or v > 3600:
            raise ValueError('Timeout must be between 10 and 3600 seconds')
        return v

class SutazAIConfig(BaseModel):
    environment: str
    debug: bool = False
    database: DatabaseConfig
    agents: AgentConfig
    
    @validator('environment')
    def validate_environment(cls, v):
        if v not in ['development', 'staging', 'production']:
            raise ValueError('Environment must be development, staging, or production')
        return v

def load_and_validate_config(config_file: str) -> SutazAIConfig:
    """Load and validate configuration file."""
    import yaml
    
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return SutazAIConfig(**config_data)
```

---

## Monitoring and Observability Best Practices

### Comprehensive Monitoring

#### Metrics Collection
```python
# Comprehensive metrics collection
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
from functools import wraps

# Define metrics
task_counter = Counter('sutazai_tasks_total', 'Total tasks processed', ['agent', 'status'])
task_duration = Histogram('sutazai_task_duration_seconds', 'Task processing time', ['agent'])
active_tasks = Gauge('sutazai_active_tasks', 'Currently active tasks', ['agent'])
system_health = Gauge('sutazai_system_health', 'System health score')

class MetricsCollector:
    def __init__(self):
        # Start Prometheus metrics server
        start_http_server(8090)
        
    def record_task_start(self, agent_name: str):
        """Record task start."""
        active_tasks.labels(agent=agent_name).inc()
        
    def record_task_completion(self, agent_name: str, duration: float, status: str):
        """Record task completion."""
        task_counter.labels(agent=agent_name, status=status).inc()
        task_duration.labels(agent=agent_name).observe(duration)
        active_tasks.labels(agent=agent_name).dec()
    
    def update_system_health(self, health_score: float):
        """Update system health score."""
        system_health.set(health_score)

def monitor_task_execution(func):
    """Decorator to monitor task execution."""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        metrics = MetricsCollector()
        agent_name = self.name
        start_time = time.time()
        
        metrics.record_task_start(agent_name)
        
        try:
            result = await func(self, *args, **kwargs)
            status = "success"
            return result
        except Exception as e:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            metrics.record_task_completion(agent_name, duration, status)
    
    return wrapper
```

#### Structured Logging
```python
# Structured logging implementation
import logging
import json
from datetime import datetime
from typing import Dict, Any

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Configure structured logging
        handler = logging.StreamHandler()
        formatter = StructuredFormatter()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_event(self, level: str, message: str, **kwargs):
        """Log structured event."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            "service": "sutazai",
            **kwargs
        }
        
        getattr(self.logger, level.lower())(json.dumps(log_data))
    
    def log_task_start(self, task_id: str, agent_name: str, action: str):
        """Log task start event."""
        self.log_event(
            "info",
            "Task started",
            task_id=task_id,
            agent_name=agent_name,
            action=action,
            event_type="task_start"
        )
    
    def log_task_completion(self, task_id: str, agent_name: str, duration: float, status: str):
        """Log task completion event."""
        self.log_event(
            "info",
            "Task completed",
            task_id=task_id,
            agent_name=agent_name,
            duration=duration,
            status=status,
            event_type="task_completion"
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any]):
        """Log error with context."""
        self.log_event(
            "error",
            "Error occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            event_type="error",
            **context
        )

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        """Format log record as structured JSON."""
        if hasattr(record, 'getMessage'):
            try:
                # Try to parse as JSON
                return record.getMessage()
            except:
                # Fallback to standard formatting
                log_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno
                }
                return json.dumps(log_data)
        return super().format(record)
```

#### Alerting Configuration
```yaml
# Comprehensive alerting rules
alerting_rules:
  - name: "High Error Rate"
    condition: "rate(sutazai_tasks_total{status='error'}[5m]) > 0.1"
    severity: "critical"
    duration: "2m"
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"
    
  - name: "Agent Down"
    condition: "up{job='sutazai-agents'} == 0"
    severity: "critical"
    duration: "1m"
    annotations:
      summary: "Agent is down"
      description: "Agent {{ $labels.instance }} is not responding"
  
  - name: "High Response Time"
    condition: "sutazai_task_duration_seconds{quantile='0.95'} > 10"
    severity: "warning"
    duration: "5m"
    annotations:
      summary: "High response time"
      description: "95th percentile response time is {{ $value }} seconds"
  
  - name: "Memory Usage High"
    condition: "container_memory_usage_bytes / container_memory_limit_bytes > 0.8"
    severity: "warning"
    duration: "5m"
    annotations:
      summary: "High memory usage"
      description: "Memory usage is {{ $value | humanizePercentage }}"

notification_channels:
  - name: "slack"
    type: "slack"
    webhook_url: "${SLACK_WEBHOOK_URL}"
    channel: "#sutazai-alerts"
    
  - name: "email"
    type: "email"
    smtp_server: "smtp.company.com"
    recipients: ["admin@company.com", "devops@company.com"]
  
  - name: "pagerduty"
    type: "pagerduty"
    integration_key: "${PAGERDUTY_INTEGRATION_KEY}"
```

---

## Data Management Best Practices

### Data Lifecycle Management

#### Data Retention Policies
```yaml
# Data retention configuration
data_retention:
  logs:
    application_logs:
      retention_period: "30d"
      archive_after: "7d"
      compression: true
    
    access_logs:
      retention_period: "90d"
      archive_after: "30d"
      compression: true
    
    audit_logs:
      retention_period: "7y"
      archive_after: "1y"
      compression: true
  
  metrics:
    high_resolution:
      retention_period: "7d"
      resolution: "1m"
    
    medium_resolution:
      retention_period: "30d"
      resolution: "5m"
    
    low_resolution:
      retention_period: "1y"
      resolution: "1h"
  
  task_data:
    completed_tasks:
      retention_period: "90d"
      archive_after: "30d"
    
    failed_tasks:
      retention_period: "180d"
      archive_after: "60d"
```

#### Backup Strategies
```bash
#!/bin/bash
# Comprehensive backup strategy

BACKUP_DIR="/backups/sutazai"
DATE=$(date +%Y%m%d_%H%M%S)

# Configuration backup
echo "Backing up configuration..."
tar -czf "${BACKUP_DIR}/config_${DATE}.tar.gz" \
  /opt/sutazaiapp/config/ \
  /opt/sutazaiapp/docker-compose*.yml

# Database backup
echo "Backing up database..."
docker exec sutazaiapp_postgres_1 pg_dump -U postgres sutazai \
  | gzip > "${BACKUP_DIR}/database_${DATE}.sql.gz"

# Agent configurations backup
echo "Backing up agent configurations..."
tar -czf "${BACKUP_DIR}/agents_${DATE}.tar.gz" \
  /opt/sutazaiapp/agents/configs/

# Monitoring data backup
echo "Backing up monitoring data..."
docker exec sutazaiapp_prometheus_1 promtool tsdb create-blocks-from prometheus \
  --output-dir "/tmp/prometheus_backup_${DATE}"
tar -czf "${BACKUP_DIR}/monitoring_${DATE}.tar.gz" \
  "/tmp/prometheus_backup_${DATE}"

# Log aggregation backup
echo "Backing up critical logs..."
tar -czf "${BACKUP_DIR}/logs_${DATE}.tar.gz" \
  /opt/sutazaiapp/logs/ \
  --exclude="*.tmp" \
  --exclude="debug.log"

# Cleanup old backups (keep last 30 days)
find "${BACKUP_DIR}" -name "*.tar.gz" -mtime +30 -delete
find "${BACKUP_DIR}" -name "*.sql.gz" -mtime +30 -delete

echo "Backup completed successfully"
```

### Data Security

#### Data Encryption
```python
# Data encryption at rest and in transit
import ssl
from cryptography.fernet import Fernet
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

class SecureDataManager:
    def __init__(self):
        self.encryption_key = self._load_encryption_key()
        self.cipher = Fernet(self.encryption_key)
        
    def _load_encryption_key(self):
        """Load encryption key from secure storage."""
        # Implementation depends on key management system
        pass
    
    def encrypt_sensitive_field(self, data: str) -> str:
        """Encrypt sensitive data fields."""
        if not data:
            return data
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_field(self, encrypted_data: str) -> str:
        """Decrypt sensitive data fields."""
        if not encrypted_data:
            return encrypted_data
        return self.cipher.decrypt(encrypted_data.encode()).decode()

# Database connection with SSL
def create_secure_database_connection():
    """Create database connection with SSL encryption."""
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    ssl_context.load_cert_chain('/certs/client.crt', '/certs/client.key')
    ssl_context.load_verify_locations('/certs/ca.crt')
    
    engine = create_engine(
        "postgresql://user:password@host:5432/dbname",
        connect_args={
            "sslmode": "require",
            "sslcert": "/certs/client.crt",
            "sslkey": "/certs/client.key",
            "sslrootcert": "/certs/ca.crt"
        }
    )
    
    return engine

# Automatic field encryption
@event.listens_for(Engine, "before_cursor_execute")
def encrypt_sensitive_data(conn, cursor, statement, parameters, context, executemany):
    """Automatically encrypt sensitive data before database operations."""
    # Implementation depends on specific requirements
    pass
```

---

## Agent Development Best Practices

### Agent Architecture

#### Standard Agent Template
```python
# Standard agent implementation template
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import asyncio
from dataclasses import dataclass
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class TaskRequest:
    task_id: str
    action: str
    data: Dict[str, Any]
    priority: str = "medium"
    timeout: int = 300

@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    result: Dict[str, Any]
    execution_time: float
    error_message: str = None

class StandardAgent(ABC):
    """Standard agent implementation template."""
    
    def __init__(self, name: str, capabilities: List[str]):
        self.name = name
        self.capabilities = capabilities
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.metrics = self._setup_metrics()
        self._initialize()
    
    @abstractmethod
    async def execute_task(self, request: TaskRequest) -> TaskResult:
        """Execute a task request."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        pass
    
    def _load_config(self) -> Dict[str, Any]:
        """Load agent configuration."""
        # Implementation specific to agent
        pass
    
    def _setup_logging(self):
        """Setup structured logging."""
        return StructuredLogger(self.name)
    
    def _setup_metrics(self):
        """Setup metrics collection."""
        return MetricsCollector()
    
    def _initialize(self):
        """Initialize agent-specific resources."""
        pass
    
    async def _validate_request(self, request: TaskRequest):
        """Validate task request."""
        if request.action not in self.capabilities:
            raise ValueError(f"Unsupported action: {request.action}")
        
        if not request.data:
            raise ValueError("Request data is required")
    
    async def _execute_with_timeout(self, coro, timeout: int):
        """Execute coroutine with timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Task execution timed out after {timeout} seconds")
```

#### Error Handling Patterns
```python
# Comprehensive error handling for agents
from enum import Enum
import traceback

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AgentError(Exception):
    """Base agent error class."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        super().__init__(message)
        self.severity = severity

class ValidationError(AgentError):
    """Input validation error."""
    def __init__(self, message: str):
        super().__init__(message, ErrorSeverity.MEDIUM)

class ProcessingError(AgentError):
    """Processing logic error."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.HIGH):
        super().__init__(message, severity)

class ResourceError(AgentError):
    """Resource availability error."""
    def __init__(self, message: str):
        super().__init__(message, ErrorSeverity.CRITICAL)

class ErrorHandler:
    """Centralized error handling."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = StructuredLogger(agent_name)
        self.metrics = MetricsCollector()
    
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> TaskResult:
        """Handle and classify errors."""
        error_info = {
            "agent_name": self.agent_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "traceback": traceback.format_exc()
        }
        
        # Log error
        self.logger.log_error(error, error_info)
        
        # Record metrics
        self.metrics.record_error(self.agent_name, type(error).__name__)
        
        # Determine severity and response
        if isinstance(error, AgentError):
            severity = error.severity
        else:
            severity = ErrorSeverity.HIGH
        
        # Handle based on severity
        if severity == ErrorSeverity.CRITICAL:
            await self._handle_critical_error(error, error_info)
        elif severity == ErrorSeverity.HIGH:
            await self._handle_high_severity_error(error, error_info)
        
        return TaskResult(
            task_id=context.get("task_id", "unknown"),
            status=TaskStatus.FAILED,
            result={},
            execution_time=context.get("execution_time", 0),
            error_message=str(error)
        )
    
    async def _handle_critical_error(self, error: Exception, error_info: Dict[str, Any]):
        """Handle critical errors."""
        # Alert monitoring systems
        await self._send_alert("critical", error_info)
        
        # Attempt graceful shutdown if necessary
        if isinstance(error, ResourceError):
            await self._initiate_graceful_shutdown()
    
    async def _handle_high_severity_error(self, error: Exception, error_info: Dict[str, Any]):
        """Handle high severity errors."""
        # Alert monitoring systems
        await self._send_alert("high", error_info)
        
        # Attempt error recovery
        await self._attempt_error_recovery(error, error_info)
```

---

## Deployment and CI/CD Best Practices

### Continuous Integration

#### Automated Testing Pipeline
```yaml
# .github/workflows/ci.yml
name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    - name: Type checking with mypy
      run: mypy agents/ --ignore-missing-imports
    
    - name: Security scanning with bandit
      run: bandit -r agents/ -f json -o security-report.json
    
    - name: Test with pytest
      run: |
        pytest tests/ --cov=agents --cov-report=xml --cov-report=html
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  integration-test:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Start services
      run: |
        docker-compose -f docker-compose.test.yml up -d
        sleep 30
    
    - name: Run integration tests
      run: |
        docker-compose -f docker-compose.test.yml exec -T api-gateway \
          pytest tests/integration/ -v
    
    - name: Check service health
      run: |
        curl -f http://localhost:8000/health || exit 1
    
    - name: Cleanup
      run: docker-compose -f docker-compose.test.yml down -v

  security-scan:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
```

### Continuous Deployment

#### Automated Deployment Pipeline
```yaml
# .github/workflows/cd.yml
name: Continuous Deployment

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ghcr.io/${{ github.repository }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    runs-on: ubuntu-latest
    needs: build-and-push
    environment: staging
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to staging
      run: |
        # Deploy to staging environment
        ./scripts/deploy.sh --environment staging --version ${{ github.sha }}
    
    - name: Run smoke tests
      run: |
        ./scripts/smoke-tests.sh --environment staging
    
    - name: Run performance tests
      run: |
        ./scripts/performance-tests.sh --environment staging --duration 300

  deploy-production:
    runs-on: ubuntu-latest
    needs: [build-and-push, deploy-staging]
    environment: production
    if: github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/tags/')
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to production
      run: |
        ./scripts/deploy.sh --environment production --version ${{ github.sha }} --strategy blue-green
    
    - name: Post-deployment validation
      run: |
        ./scripts/post-deployment-validation.sh --environment production
    
    - name: Notify deployment success
      uses: 8398a7/action-slack@v3
      with:
        status: success
        channel: '#deployments'
        text: 'SutazAI deployed successfully to production'
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

---

## Troubleshooting and Maintenance Best Practices

### Proactive Maintenance

#### Automated Health Monitoring
```python
# Comprehensive health monitoring system
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class HealthCheck:
    name: str
    check_function: callable
    interval: int  # seconds
    timeout: int   # seconds
    critical: bool = False

class ProactiveHealthMonitor:
    def __init__(self):
        self.health_checks = []
        self.health_history = {}
        self.alert_thresholds = {
            "consecutive_failures": 3,
            "failure_rate_window": timedelta(minutes=15),
            "failure_rate_threshold": 0.5
        }
    
    def register_health_check(self, health_check: HealthCheck):
        """Register a health check."""
        self.health_checks.append(health_check)
        self.health_history[health_check.name] = []
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        tasks = []
        for check in self.health_checks:
            task = asyncio.create_task(self._monitor_health_check(check))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def _monitor_health_check(self, check: HealthCheck):
        """Monitor individual health check."""
        while True:
            try:
                start_time = datetime.utcnow()
                
                # Execute health check with timeout
                result = await asyncio.wait_for(
                    check.check_function(),
                    timeout=check.timeout
                )
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Record successful check
                self._record_health_result(check.name, True, execution_time)
                
            except Exception as e:
                # Record failed check
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                self._record_health_result(check.name, False, execution_time, str(e))
                
                # Check if alerting is needed
                await self._check_alert_conditions(check)
            
            # Wait for next check
            await asyncio.sleep(check.interval)
    
    def _record_health_result(self, check_name: str, success: bool, 
                            execution_time: float, error_message: str = None):
        """Record health check result."""
        result = {
            "timestamp": datetime.utcnow(),
            "success": success,
            "execution_time": execution_time,
            "error_message": error_message
        }
        
        self.health_history[check_name].append(result)
        
        # Keep only recent history (last 1000 checks)
        if len(self.health_history[check_name]) > 1000:
            self.health_history[check_name] = self.health_history[check_name][-1000:]
    
    async def _check_alert_conditions(self, check: HealthCheck):
        """Check if alert conditions are met."""
        history = self.health_history[check.name]
        
        # Check consecutive failures
        consecutive_failures = 0
        for result in reversed(history):
            if not result["success"]:
                consecutive_failures += 1
            else:
                break
        
        if consecutive_failures >= self.alert_thresholds["consecutive_failures"]:
            await self._send_alert(
                check,
                "consecutive_failures",
                f"{consecutive_failures} consecutive failures"
            )
        
        # Check failure rate
        window_start = datetime.utcnow() - self.alert_thresholds["failure_rate_window"]
        recent_results = [r for r in history if r["timestamp"] > window_start]
        
        if recent_results:
            failure_rate = sum(1 for r in recent_results if not r["success"]) / len(recent_results)
            
            if failure_rate > self.alert_thresholds["failure_rate_threshold"]:
                await self._send_alert(
                    check,
                    "high_failure_rate",
                    f"Failure rate: {failure_rate:.2%}"
                )
    
    async def _send_alert(self, check: HealthCheck, alert_type: str, details: str):
        """Send alert for health check failure."""
        alert_data = {
            "check_name": check.name,
            "alert_type": alert_type,
            "details": details,
            "critical": check.critical,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send to monitoring system
        # Implementation depends on monitoring setup
        pass

# Health check implementations
async def check_api_gateway_health():
    """Check API gateway health."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8000/health") as response:
            if response.status != 200:
                raise Exception(f"API gateway health check failed: {response.status}")
            data = await response.json()
            if data.get("status") != "healthy":
                raise Exception(f"API gateway unhealthy: {data}")

async def check_database_health():
    """Check database connectivity."""
    # Implementation depends on database setup
    pass

async def check_agent_health():
    """Check agent responsiveness."""
    # Implementation depends on agent setup
    pass

# Setup health monitoring
def setup_health_monitoring():
    """Setup comprehensive health monitoring."""
    monitor = ProactiveHealthMonitor()
    
    # Register health checks
    monitor.register_health_check(HealthCheck(
        name="api_gateway",
        check_function=check_api_gateway_health,
        interval=30,
        timeout=10,
        critical=True
    ))
    
    monitor.register_health_check(HealthCheck(
        name="database",
        check_function=check_database_health,
        interval=60,
        timeout=15,
        critical=True
    ))
    
    monitor.register_health_check(HealthCheck(
        name="agents",
        check_function=check_agent_health,
        interval=60,
        timeout=30,
        critical=False
    ))
    
    return monitor
```

#### Maintenance Automation
```bash
#!/bin/bash
# Automated maintenance script

set -e

MAINTENANCE_TYPE=${1:-daily}
LOG_FILE="/var/log/sutazai-maintenance.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

perform_daily_maintenance() {
    log "Starting daily maintenance"
    
    # Health check
    log "Performing system health check"
    ./scripts/comprehensive-agent-health-monitor.py
    
    # Log rotation
    log "Rotating logs"
    logrotate /opt/sutazaiapp/logrotate.conf
    
    # Cleanup temporary files
    log "Cleaning temporary files"
    find /tmp -name "sutazai_*" -mtime +1 -delete
    
    # Database maintenance
    log "Performing database maintenance"
    docker exec sutazaiapp_postgres_1 vacuumdb -U postgres -d sutazai -z
    
    # Memory cleanup
    log "Clearing system caches"
    ./scripts/clear-system-caches.py
    
    log "Daily maintenance completed"
}

perform_weekly_maintenance() {
    log "Starting weekly maintenance"
    
    # Perform daily maintenance first
    perform_daily_maintenance
    
    # Performance analysis
    log "Performing performance analysis"
    ./scripts/performance-profiler-suite.py --duration 300
    
    # Security scan
    log "Running security scan"
    ./scripts/security-scan.py --comprehensive
    
    # Garbage collection
    log "Running garbage collection"
    ./scripts/garbage-collection-system.py
    
    # Configuration backup
    log "Backing up configuration"
    ./scripts/backup-configuration.sh
    
    # Update check
    log "Checking for updates"
    ./scripts/check-updates.sh
    
    log "Weekly maintenance completed"
}

perform_monthly_maintenance() {
    log "Starting monthly maintenance"
    
    # Perform weekly maintenance first
    perform_weekly_maintenance
    
    # Comprehensive system audit
    log "Performing comprehensive system audit"
    ./scripts/complete-system-audit.py
    
    # Capacity planning analysis
    log "Performing capacity planning analysis"
    ./scripts/capacity-planning-analysis.py
    
    # Performance optimization
    log "Running performance optimization"
    ./scripts/performance-optimization.py --apply-recommendations
    
    # Full backup
    log "Performing full system backup"
    ./scripts/full-system-backup.sh
    
    # Documentation update check
    log "Checking documentation updates"
    ./scripts/update-documentation.py
    
    log "Monthly maintenance completed"
}

case "$MAINTENANCE_TYPE" in
    "daily")
        perform_daily_maintenance
        ;;
    "weekly")
        perform_weekly_maintenance
        ;;
    "monthly")
        perform_monthly_maintenance
        ;;
    *)
        echo "Usage: $0 {daily|weekly|monthly}"
        exit 1
        ;;
esac

# Send maintenance report
./scripts/send-maintenance-report.py --type "$MAINTENANCE_TYPE" --log-file "$LOG_FILE"
```

---

This comprehensive best practices guide provides detailed guidance for all aspects of SutazAI usage, development, and operations. Following these practices ensures optimal system performance, security, reliability, and maintainability across all environments and use cases.