# SutazAI Resource Management and Scaling Guide

## Resource Management Strategy

### 1. CPU Resource Allocation

#### Core Services CPU Allocation

```yaml
# High Priority Services (Coordinator & Core Infrastructure)
ollama:
  deploy:
    resources:
      limits:
        cpus: '6'          # Max 6 CPU cores for model inference
        memory: 8G
      reservations:
        cpus: '2'          # Always reserve 2 cores
        memory: 2G

backend:
  deploy:
    resources:
      limits:
        cpus: '2'          # API server gets 2 cores max
        memory: 4G
      reservations:
        cpus: '1'          # Always reserve 1 core
        memory: 2G

postgres:
  deploy:
    resources:
      limits:
        cpus: '2'          # Database gets 2 cores max
        memory: 4G
      reservations:
        cpus: '0.5'        # Reserve half core minimum
        memory: 1G
```

#### AI Agents CPU Allocation

```yaml
# Medium Priority Services (AI Agents)
autogpt:
  deploy:
    resources:
      limits:
        cpus: '1'          # 1 core per agent maximum
        memory: 2G
      reservations:
        cpus: '0.25'       # Quarter core minimum
        memory: 512M

crewai:
  deploy:
    resources:
      limits:
        cpus: '2'          # Multi-agent system gets more resources
        memory: 4G
      reservations:
        cpus: '0.5'
        memory: 1G

# Low Priority Services (Support & Monitoring)
redis:
  deploy:
    resources:
      limits:
        cpus: '0.5'        # Minimal CPU for cache
        memory: 1G
      reservations:
        cpus: '0.1'
        memory: 256M
```

### 2. Memory Management Strategy

#### Memory Allocation by Service Type

```yaml
# Memory allocation guidelines by service category

Database Services:
  postgres: 4GB (2GB buffer pool + 2GB working memory)
  redis: 1GB (cache with LRU eviction)
  neo4j: 2GB (1GB heap + 1GB page cache)

Vector Databases:
  chromadb: 2GB (embedding storage and processing)
  qdrant: 2GB (vector similarity computations)
  faiss: 1GB (index operations)

AI Infrastructure:
  ollama: 8GB (model loading and inference)
  litellm: 1GB (API proxy overhead)

Application Layer:
  backend: 4GB (FastAPI + AI orchestration)
  frontend: 2GB (Streamlit interface)

AI Agents (per agent):
  Simple agents: 512MB - 1GB
  Complex agents: 1GB - 2GB
  Multi-agent systems: 2GB - 4GB
```

#### Memory Optimization Configuration

```bash
# Environment variables for memory optimization
OLLAMA_MAX_LOADED_MODELS=1      # Reduce memory by limiting concurrent models
OLLAMA_KEEP_ALIVE=2m            # Unload models after 2 minutes of inactivity
POSTGRES_SHARED_BUFFERS=2GB     # PostgreSQL buffer pool
POSTGRES_WORK_MEM=256MB         # Per-query working memory
REDIS_MAXMEMORY=1gb             # Redis memory limit
REDIS_MAXMEMORY_POLICY=allkeys-lru  # LRU eviction policy
```

### 3. Storage Management

#### Disk Space Requirements

```bash
# Storage allocation by component
Component                Size Range    Recommended
Base OS + Docker        20GB          30GB
Ollama Models          30-100GB       150GB
Vector Databases       5-50GB         100GB
Application Data       2-10GB         20GB
Logs & Monitoring      5-20GB         50GB
Agent Workspaces       5-30GB         50GB
Backups               50-200GB        300GB
---------------------------------------------------
Total                 117-430GB       700GB
```

#### Storage Optimization

```yaml
# Volume management for optimal performance
volumes:
  # SSD volumes for high-performance needs
  postgres_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/sutazaiapp/data/postgres  # Mount on SSD
  
  ollama_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/sutazaiapp/data/ollama    # Mount on SSD

  # HDD volumes for bulk storage
  agent_workspaces:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/sutazaiapp/workspace      # Can be on HDD

  logs:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/sutazaiapp/logs           # Can be on HDD
```

## Scaling Strategies

### 1. Horizontal Scaling (Scale Out)

#### Load Balancer Configuration

```yaml
# docker-compose.scale.yml
version: '3.8'

services:
  # Load balancer for backend services
  nginx-lb:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/load-balancer.conf:/etc/nginx/nginx.conf
    depends_on:
      - backend
    networks:
      - sutazai-network

  # Scalable backend service
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    deploy:
      replicas: 3  # Scale to 3 instances
    environment:
      - INSTANCE_ID={{.Task.Slot}}
    depends_on:
      - postgres
      - redis
    networks:
      - sutazai-network

  # Redis cluster for high availability
  redis-cluster:
    image: redis:7-alpine
    command: redis-server --cluster-enabled yes --cluster-config-file nodes.conf
    deploy:
      replicas: 6  # 3 masters + 3 replicas
    networks:
      - sutazai-network
```

#### NGINX Load Balancer Configuration

```nginx
# nginx/load-balancer.conf
upstream backend_cluster {
    least_conn;
    server backend_1:8000 max_fails=3 fail_timeout=30s;
    server backend_2:8000 max_fails=3 fail_timeout=30s;
    server backend_3:8000 max_fails=3 fail_timeout=30s;
    
    # Health check
    keepalive 32;
}

upstream agent_cluster {
    ip_hash;  # Session affinity for stateful agents
    server autogpt_1:8080 max_fails=2 fail_timeout=30s;
    server autogpt_2:8080 max_fails=2 fail_timeout=30s;
    server crewai_1:8096 max_fails=2 fail_timeout=30s;
    server crewai_2:8096 max_fails=2 fail_timeout=30s;
}

server {
    listen 80;
    
    # Backend API load balancing
    location /api/ {
        proxy_pass http://backend_cluster;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Connection pooling
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Agent services load balancing
    location /agents/ {
        proxy_pass http://agent_cluster;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### 2. Vertical Scaling (Scale Up)

#### Dynamic Resource Adjustment

```bash
# scripts/scale-up.sh
#!/bin/bash

# Function to scale up a service
scale_up_service() {
    local service=$1
    local cpu_limit=$2
    local memory_limit=$3
    
    echo "Scaling up $service to CPU: $cpu_limit, Memory: $memory_limit"
    
    # Update docker-compose override
    cat > docker-compose.override.yml << EOF
version: '3.8'
services:
  $service:
    deploy:
      resources:
        limits:
          cpus: '$cpu_limit'
          memory: $memory_limit
        reservations:
          cpus: '$(echo "$cpu_limit * 0.5" | bc)'
          memory: $(echo "$memory_limit" | sed 's/G/*0.5G/' | bc)G
EOF
    
    # Apply changes
    docker-compose up -d --force-recreate $service
}

# Scale up based on load
CURRENT_LOAD=$(docker stats --no-stream --format "{{.CPUPerc}}" sutazai-backend | sed 's/%//')
if (( $(echo "$CURRENT_LOAD > 80" | bc -l) )); then
    scale_up_service "backend" "4" "8G"
fi

OLLAMA_MEMORY=$(docker stats --no-stream --format "{{.MemUsage}}" sutazai-ollama | cut -d'/' -f1 | sed 's/GiB//')
if (( $(echo "$OLLAMA_MEMORY > 6" | bc -l) )); then
    scale_up_service "ollama" "8" "16G"
fi
```

### 3. Auto-Scaling Configuration

#### Docker Swarm Auto-Scaling

```yaml
# docker-stack-autoscale.yml
version: '3.8'

services:
  backend:
    image: sutazai/backend:latest
    deploy:
      replicas: 2
      placement:
        constraints:
          - node.role == worker
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      update_config:
        parallelism: 1
        delay: 10s
        failure_action: rollback
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    networks:
      - sutazai-network
    
  # Auto-scaling based on CPU usage
  autoscaler:
    image: docker/swarmkit-autoscaler
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      - TARGET_SERVICE=backend
      - MIN_REPLICAS=2
      - MAX_REPLICAS=10
      - SCALE_UP_THRESHOLD=70    # CPU percentage
      - SCALE_DOWN_THRESHOLD=30  # CPU percentage
      - COOLDOWN_PERIOD=300      # 5 minutes
    deploy:
      placement:
        constraints:
          - node.role == manager
```

#### Kubernetes Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
  namespace: sutazai-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-agents-hpa
  namespace: sutazai-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: autogpt
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
```

## Performance Monitoring and Optimization

### 1. Resource Monitoring Dashboard

```python
# scripts/resource_monitor.py
import docker
import psutil
import time
import json
from datetime import datetime

class ResourceMonitor:
    def __init__(self):
        self.client = docker.from_env()
        self.metrics = []
    
    def collect_system_metrics(self):
        """Collect system-wide metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            }
        }
    
    def collect_container_metrics(self):
        """Collect per-container metrics"""
        containers = {}
        
        for container in self.client.containers.list():
            if 'sutazai' in container.name:
                stats = container.stats(stream=False)
                
                # Calculate CPU percentage
                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                           stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                              stats['precpu_stats']['system_cpu_usage']
                cpu_percent = (cpu_delta / system_delta) * 100.0
                
                # Calculate memory usage
                memory_usage = stats['memory_stats']['usage']
                memory_limit = stats['memory_stats']['limit']
                memory_percent = (memory_usage / memory_limit) * 100.0
                
                containers[container.name] = {
                    'cpu_percent': cpu_percent,
                    'memory_usage_mb': memory_usage / (1024**2),
                    'memory_percent': memory_percent,
                    'status': container.status
                }
        
        return containers
    
    def generate_scaling_recommendations(self, metrics):
        """Generate scaling recommendations based on metrics"""
        recommendations = []
        
        for container, stats in metrics['containers'].items():
            if stats['cpu_percent'] > 80:
                recommendations.append({
                    'container': container,
                    'action': 'scale_up',
                    'reason': f"High CPU usage: {stats['cpu_percent']:.1f}%",
                    'recommended_action': 'Increase CPU limit or add replicas'
                })
            
            if stats['memory_percent'] > 85:
                recommendations.append({
                    'container': container,
                    'action': 'scale_up',
                    'reason': f"High memory usage: {stats['memory_percent']:.1f}%",
                    'recommended_action': 'Increase memory limit'
                })
        
        return recommendations
    
    def run_monitoring_loop(self, interval=60):
        """Run continuous monitoring"""
        while True:
            try:
                system_metrics = self.collect_system_metrics()
                container_metrics = self.collect_container_metrics()
                
                metrics = {
                    **system_metrics,
                    'containers': container_metrics
                }
                
                recommendations = self.generate_scaling_recommendations(metrics)
                
                # Log metrics
                print(f"Timestamp: {metrics['timestamp']}")
                print(f"System CPU: {metrics['system']['cpu_percent']:.1f}%")
                print(f"System Memory: {metrics['system']['memory_percent']:.1f}%")
                
                if recommendations:
                    print("Scaling Recommendations:")
                    for rec in recommendations:
                        print(f"  - {rec['container']}: {rec['reason']} -> {rec['recommended_action']}")
                
                # Store metrics
                self.metrics.append(metrics)
                
                # Keep only last 24 hours of metrics
                if len(self.metrics) > 1440:  # 24 hours * 60 minutes
                    self.metrics = self.metrics[-1440:]
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(interval)

if __name__ == "__main__":
    monitor = ResourceMonitor()
    monitor.run_monitoring_loop()
```

### 2. Automated Scaling Script

```bash
# scripts/auto-scale.sh
#!/bin/bash
set -euo pipefail

LOG_FILE="/opt/sutazaiapp/logs/autoscale.log"
SCALE_CONFIG="/opt/sutazaiapp/config/scaling.json"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Load scaling configuration
if [[ -f "$SCALE_CONFIG" ]]; then
    CPU_SCALE_UP_THRESHOLD=$(jq -r '.cpu_scale_up_threshold' "$SCALE_CONFIG")
    CPU_SCALE_DOWN_THRESHOLD=$(jq -r '.cpu_scale_down_threshold' "$SCALE_CONFIG")
    MEMORY_SCALE_UP_THRESHOLD=$(jq -r '.memory_scale_up_threshold' "$SCALE_CONFIG")
else
    # Default thresholds
    CPU_SCALE_UP_THRESHOLD=75
    CPU_SCALE_DOWN_THRESHOLD=25
    MEMORY_SCALE_UP_THRESHOLD=85
fi

# Function to get container resource usage
get_container_stats() {
    local container_name=$1
    docker stats --no-stream --format "{{.CPUPerc}},{{.MemPerc}}" "$container_name" | tr -d '%'
}

# Function to scale service
scale_service() {
    local service=$1
    local action=$2
    local current_replicas
    local new_replicas
    
    current_replicas=$(docker service ls --filter name="$service" --format "{{.Replicas}}" | cut -d'/' -f1)
    
    case "$action" in
        "up")
            new_replicas=$((current_replicas + 1))
            log "Scaling up $service from $current_replicas to $new_replicas replicas"
            ;;
        "down")
            new_replicas=$((current_replicas - 1))
            if [[ $new_replicas -lt 1 ]]; then
                new_replicas=1
            fi
            log "Scaling down $service from $current_replicas to $new_replicas replicas"
            ;;
    esac
    
    docker service scale "$service=$new_replicas"
}

# Function to update resource limits
update_resource_limits() {
    local service=$1
    local cpu_limit=$2
    local memory_limit=$3
    
    log "Updating resource limits for $service: CPU=$cpu_limit, Memory=$memory_limit"
    
    docker service update \
        --limit-cpu "$cpu_limit" \
        --limit-memory "$memory_limit" \
        "$service"
}

# Main scaling logic
check_and_scale() {
    local service=$1
    local container_name=$2
    
    # Get current stats
    stats=$(get_container_stats "$container_name")
    cpu_usage=$(echo "$stats" | cut -d',' -f1 | cut -d'.' -f1)
    memory_usage=$(echo "$stats" | cut -d',' -f2 | cut -d'.' -f1)
    
    log "Service: $service, CPU: ${cpu_usage}%, Memory: ${memory_usage}%"
    
    # Scale up decisions
    if [[ $cpu_usage -gt $CPU_SCALE_UP_THRESHOLD ]]; then
        log "High CPU usage detected for $service"
        
        # First try to increase resource limits
        current_cpu_limit=$(docker service inspect "$service" --format '{{.Spec.TaskTemplate.Resources.Limits.NanoCPUs}}')
        if [[ $current_cpu_limit -lt 4000000000 ]]; then  # Less than 4 CPU cores
            new_cpu_limit=$((current_cpu_limit + 1000000000))  # Add 1 CPU core
            update_resource_limits "$service" "${new_cpu_limit}n" "$(docker service inspect "$service" --format '{{.Spec.TaskTemplate.Resources.Limits.MemoryBytes}}')"
        else
            # If at max CPU, scale horizontally
            scale_service "$service" "up"
        fi
    fi
    
    if [[ $memory_usage -gt $MEMORY_SCALE_UP_THRESHOLD ]]; then
        log "High memory usage detected for $service"
        
        # Increase memory limit
        current_memory_limit=$(docker service inspect "$service" --format '{{.Spec.TaskTemplate.Resources.Limits.MemoryBytes}}')
        new_memory_limit=$((current_memory_limit + 1073741824))  # Add 1GB
        update_resource_limits "$service" "$(docker service inspect "$service" --format '{{.Spec.TaskTemplate.Resources.Limits.NanoCPUs}}')" "$new_memory_limit"
    fi
    
    # Scale down decisions
    if [[ $cpu_usage -lt $CPU_SCALE_DOWN_THRESHOLD ]] && [[ $memory_usage -lt 50 ]]; then
        log "Low resource usage detected for $service"
        
        # Check if we can scale down
        current_replicas=$(docker service ls --filter name="$service" --format "{{.Replicas}}" | cut -d'/' -f1)
        if [[ $current_replicas -gt 1 ]]; then
            scale_service "$service" "down"
        fi
    fi
}

# Main execution
main() {
    log "Starting auto-scaling check"
    
    # Define services to monitor
    services=(
        "sutazai_backend:sutazai-backend"
        "sutazai_autogpt:sutazai-autogpt"
        "sutazai_crewai:sutazai-crewai"
        "sutazai_ollama:sutazai-ollama"
    )
    
    for service_pair in "${services[@]}"; do
        IFS=':' read -r service container <<< "$service_pair"
        
        # Check if container exists and is running
        if docker ps --format "{{.Names}}" | grep -q "$container"; then
            check_and_scale "$service" "$container"
        else
            log "Container $container not found or not running"
        fi
    done
    
    log "Auto-scaling check completed"
}

# Run the scaling check
main "$@"
```

### 3. Performance Optimization Recommendations

#### Model Optimization

```bash
# scripts/optimize-models.sh
#!/bin/bash

# Function to optimize Ollama models for production
optimize_ollama_models() {
    echo "Optimizing Ollama models for production..."
    
    # Create optimized model configurations
    docker exec sutazai-ollama bash -c '
        # Create production-optimized model variants
        echo "FROM llama3.2:3b
PARAMETER num_thread 4
PARAMETER num_ctx 2048
PARAMETER temperature 0.7
PARAMETER top_p 0.9" | ollama create llama3.2-prod
        
        echo "FROM qwen2.5:3b
PARAMETER num_thread 4
PARAMETER num_ctx 2048
PARAMETER temperature 0.7" | ollama create qwen2.5-prod
        
        echo "FROM codellama:7b
PARAMETER num_thread 6
PARAMETER num_ctx 4096
PARAMETER temperature 0.1" | ollama create codellama-prod
    '
    
    # Update environment to use optimized models
    cat >> .env << EOF
# Optimized model configuration
DEFAULT_MODEL=llama3.2-prod
CODE_MODEL=codellama-prod
REASONING_MODEL=qwen2.5-prod
EOF
}

# Function to optimize database performance
optimize_databases() {
    echo "Optimizing database performance..."
    
    # PostgreSQL optimization
    docker exec sutazai-postgres psql -U sutazai -d sutazai -c "
        -- Analyze all tables for better query planning
        ANALYZE;
        
        -- Update statistics
        SELECT schemaname,tablename,n_tup_ins,n_tup_upd,n_tup_del,last_analyze,last_autoanalyze 
        FROM pg_stat_user_tables;
        
        -- Optimize frequently used queries
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_status ON agents(status);
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
    "
    
    # Redis optimization
    docker exec sutazai-redis redis-cli CONFIG SET save "900 1 300 10 60 10000"
    docker exec sutazai-redis redis-cli BGREWRITEAOF
}

# Function to optimize container resources
optimize_containers() {
    echo "Optimizing container resources..."
    
    # Create optimized docker-compose override
    cat > docker-compose.performance.yml << EOF
version: '3.8'

services:
  ollama:
    environment:
      OLLAMA_NUM_PARALLEL: 2
      OLLAMA_NUM_THREADS: 6
      OLLAMA_MAX_LOADED_MODELS: 2
      OLLAMA_KEEP_ALIVE: 5m
    deploy:
      resources:
        limits:
          cpus: '6'
          memory: 12G
        reservations:
          cpus: '3'
          memory: 6G

  backend:
    environment:
      WORKERS: 4
      MAX_REQUESTS: 1000
      TIMEOUT: 60
    deploy:
      resources:
        limits:
          cpus: '3'
          memory: 6G
        reservations:
          cpus: '1.5'
          memory: 3G

  postgres:
    environment:
      POSTGRES_SHARED_BUFFERS: 1GB
      POSTGRES_EFFECTIVE_CACHE_SIZE: 3GB
      POSTGRES_WORK_MEM: 64MB
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
EOF
}

# Main optimization execution
main() {
    echo "Starting performance optimization..."
    
    optimize_ollama_models
    optimize_databases
    optimize_containers
    
    echo "Performance optimization completed!"
    echo "Restart services to apply changes:"
    echo "docker-compose -f docker-compose.yml -f docker-compose.performance.yml up -d"
}

main "$@"
```

This comprehensive resource management and scaling guide provides detailed strategies for optimizing the SutazAI system performance and implementing both horizontal and vertical scaling solutions.