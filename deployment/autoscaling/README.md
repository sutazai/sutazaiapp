# SutazAI Auto-scaling Infrastructure

This directory contains comprehensive auto-scaling infrastructure for the SutazAI platform, enabling automatic scaling based on CPU, memory, and custom AI workload metrics.

## 🚀 Quick Start

### Deploy Auto-scaling Infrastructure

```bash
# Kubernetes deployment (recommended for production)
./deployment/scripts/deploy-autoscaling.sh kubernetes production

# Docker Swarm deployment (good for smaller deployments)
./deployment/scripts/deploy-autoscaling.sh swarm staging

# Docker Compose deployment (local development)
./deployment/scripts/deploy-autoscaling.sh compose local
```

## 📁 Directory Structure

```
deployment/autoscaling/
├── README.md                    # This file
├── hpa-enhanced.yaml           # Kubernetes HPA configurations
├── vpa-config.yaml             # Kubernetes VPA configurations
├── load-balancing/
│   ├── nginx-ingress.yaml      # Nginx ingress controller
│   ├── traefik-config.yaml     # Traefik ingress controller
│   └── ingress-rules.yaml      # Ingress routing rules
├── monitoring/
│   └── ai-metrics-exporter.yaml # Custom AI metrics exporter
├── swarm/
│   ├── docker-compose.swarm.yml # Docker Swarm configuration
│   ├── nginx.conf              # Nginx load balancer config
│   ├── upstream.conf           # Upstream server definitions
│   └── swarm-autoscaler.py     # Custom Swarm autoscaler
├── kubernetes/
│   └── core-services.yaml      # Core K8s service definitions
└── scripts/
    └── deploy-autoscaling.sh    # Main deployment script
```

## 🎯 Features

### 1. Container Auto-scaling

#### Kubernetes (HPA/VPA)
- **Horizontal Pod Autoscaler (HPA)**: Scales replicas based on CPU, memory, and custom metrics
- **Vertical Pod Autoscaler (VPA)**: Adjusts CPU and memory requests/limits
- **Custom Metrics**: AI inference queue depth, response time, task completion rate
- **Predictive Scaling**: Machine learning-based scaling decisions

#### Docker Swarm
- **Service Auto-scaling**: Automatic service replica adjustment
- **Load-aware Scaling**: Scales based on actual service load
- **Health-based Decisions**: Considers service health in scaling decisions
- **Custom Autoscaler**: Python-based autoscaler with Prometheus integration

### 2. Load Balancing

#### Nginx Ingress Controller
- **Layer 7 Load Balancing**: HTTP/HTTPS traffic distribution
- **Health Checks**: Automatic unhealthy backend removal
- **Rate Limiting**: Protection against traffic spikes
- **SSL Termination**: Automatic HTTPS handling

#### Traefik (Alternative)
- **Automatic Service Discovery**: Zero-config service exposure
- **Circuit Breakers**: Automatic failure handling
- **Metrics Collection**: Built-in Prometheus metrics
- **Dynamic Configuration**: Runtime configuration updates

### 3. Resource Monitoring

#### Built-in Metrics
- **System Metrics**: CPU, memory, disk, network utilization
- **Container Metrics**: Per-container resource usage
- **Application Metrics**: Request rates, response times, error rates

#### Custom AI Metrics
- **Inference Metrics**: Model loading time, inference latency, queue depth
- **Agent Metrics**: Task completion rate, active tasks, queue size
- **Vector Database Metrics**: Search latency, collection size, index memory

### 4. Intelligent Scaling

#### Scaling Triggers
```yaml
# CPU-based scaling
cpu_threshold_up: 70%      # Scale up when CPU > 70%
cpu_threshold_down: 30%    # Scale down when CPU < 30%

# Memory-based scaling
memory_threshold_up: 80%   # Scale up when memory > 80%
memory_threshold_down: 50% # Scale down when memory < 50%

# Custom metrics scaling
inference_queue_depth: 10  # Scale up when queue > 10 requests
response_time_p95: 5s      # Scale up when 95th percentile > 5s
```

#### Scaling Policies
- **Cooldown Periods**: Prevents oscillation with configurable delays
- **Rate Limiting**: Controlled scaling to prevent resource exhaustion
- **Minimum/Maximum Replicas**: Ensures service availability and cost control
- **Health Checks**: Only scales healthy services

## 🛠 Configuration

### Kubernetes Configuration

#### HPA Configuration
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sutazai-backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sutazai-backend
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Object
    object:
      metric:
        name: ai_agent_queue_depth
      target:
        type: AverageValue
        averageValue: "20"
```

#### VPA Configuration
```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: sutazai-backend-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sutazai-backend
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: sutazai-backend
      minAllowed:
        cpu: "1"
        memory: "2Gi"
      maxAllowed:
        cpu: "8"
        memory: "16Gi"
```

### Docker Swarm Configuration

```yaml
services:
  backend:
    image: sutazai/backend:latest
    deploy:
      replicas: 3
      placement:
        max_replicas_per_node: 2
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '1'
          memory: 2G
      restart_policy:
        condition: on-failure
        delay: 10s
        max_attempts: 3
      update_config:
        parallelism: 1
        delay: 30s
        failure_action: rollback
```

### Load Balancer Configuration

#### Nginx Configuration
```nginx
upstream backend_backend {
    least_conn;
    server sutazai-backend:8000 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

server {
    listen 80;
    location /api/ {
        proxy_pass http://backend_backend/;
        proxy_set_header Host $host;
        # Health checks and timeouts
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503;
        proxy_next_upstream_timeout 5s;
        proxy_next_upstream_tries 3;
    }
}
```

## 📊 Monitoring and Observability

### Metrics Collection

#### Prometheus Configuration
```yaml
scrape_configs:
  - job_name: 'sutazai-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'ai-metrics-exporter'
    static_configs:
      - targets: ['ai-metrics-exporter:9200']
    scrape_interval: 30s
```

#### Custom Metrics
```python
# Example custom metrics
inference_requests_total = Counter(
    'sutazai_inference_requests_total',
    'Total AI inference requests',
    ['model', 'status', 'service']
)

inference_duration_seconds = Histogram(
    'sutazai_inference_duration_seconds',
    'AI inference duration in seconds',
    ['model', 'service'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)
```

### Grafana Dashboards

Key dashboards for monitoring auto-scaling:

1. **Service Overview**: Service health, replica counts, resource usage
2. **Auto-scaling Events**: Scaling actions, thresholds, cooldown periods
3. **Resource Utilization**: CPU, memory, network, disk usage trends
4. **AI Workload Metrics**: Inference latency, queue depth, model performance
5. **Load Balancer Metrics**: Request rates, response times, error rates

### Alerting Rules

```yaml
groups:
- name: autoscaling
  rules:
  - alert: HighCPUUsage
    expr: avg(rate(container_cpu_usage_seconds_total[5m])) > 0.8
    for: 5m
    annotations:
      summary: High CPU usage detected
      
  - alert: ScalingEventFailure
    expr: increase(hpa_scaling_events_total{result="failed"}[5m]) > 0
    annotations:
      summary: Auto-scaling event failed
```

## 🚦 Testing Auto-scaling

### Load Testing

#### 1. Generate HTTP Load
```bash
# Install hey (HTTP load testing tool)
go install github.com/rakyll/hey@latest

# Generate load on backend API
hey -z 5m -c 10 http://localhost:8000/api/health

# Generate load on specific endpoints
hey -z 5m -c 20 -m POST -H "Content-Type: application/json" \
    -d '{"prompt": "test"}' http://localhost:8000/api/inference
```

#### 2. Monitor Scaling Events
```bash
# Kubernetes
kubectl get hpa -w
kubectl describe hpa sutazai-backend-hpa

# Docker Swarm
docker service ls
docker service logs sutazai_swarm-autoscaler -f

# Watch resource usage
watch 'docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"'
```

#### 3. Verify Scaling Behavior
```bash
# Check replica count changes
kubectl get pods -l app=sutazai-backend -w

# Monitor resource allocations
kubectl top pods
kubectl top nodes
```

### AI Workload Testing

#### 1. Inference Load Testing
```python
import asyncio
import aiohttp

async def generate_inference_load():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(100):
            task = session.post(
                'http://localhost:10104/api/generate',
                json={'model': 'tinyllama', 'prompt': f'Test prompt {i}'}
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)

asyncio.run(generate_inference_load())
```

#### 2. Agent Task Load Testing
```bash
# Create multiple agent tasks
for i in {1..50}; do
    curl -X POST http://localhost:8000/api/agents/autogpt/task \
         -H "Content-Type: application/json" \
         -d "{\"task\": \"Process item $i\"}"
done
```

## 🔧 Troubleshooting

### Common Issues

#### 1. HPA Not Scaling
```bash
# Check HPA status
kubectl describe hpa sutazai-backend-hpa

# Common issues:
# - Metrics server not installed
# - Resource requests not set
# - Insufficient load to trigger scaling

# Solutions:
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

#### 2. VPA Not Working
```bash
# Check VPA installation
kubectl get crd verticalpodautoscalers.autoscaling.k8s.io

# Install VPA if missing
git clone https://github.com/kubernetes/autoscaler.git
cd autoscaler/vertical-pod-autoscaler
./hack/vpa-up.sh
```

#### 3. Custom Metrics Not Available
```bash
# Check metrics exporter
kubectl logs deployment/ai-metrics-exporter -n sutazai-monitoring

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets
```

#### 4. Load Balancer Issues
```bash
# Check ingress controller
kubectl get pods -n sutazai-infrastructure

# Check ingress rules
kubectl describe ingress sutazai-core-ingress

# Test load balancer health
curl -I http://localhost/health
```

### Performance Tuning

#### 1. Scaling Thresholds
```yaml
# Aggressive scaling (fast response)
cpu_threshold_up: 60%
cpu_threshold_down: 20%
scale_up_cooldown: 30s
scale_down_cooldown: 120s

# Conservative scaling (stable)
cpu_threshold_up: 80%
cpu_threshold_down: 40%
scale_up_cooldown: 60s
scale_down_cooldown: 300s
```

#### 2. Resource Requests/Limits
```yaml
# Right-sized for auto-scaling
resources:
  requests:
    cpu: "1"
    memory: "2Gi"
  limits:
    cpu: "4"
    memory: "8Gi"
```

## 📚 Additional Resources

### Documentation
- [Kubernetes HPA Documentation](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Kubernetes VPA Documentation](https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler)
- [Docker Swarm Documentation](https://docs.docker.com/engine/swarm/)
- [Prometheus Metrics](https://prometheus.io/docs/concepts/metric_types/)

### Best Practices
- Start with conservative scaling thresholds and adjust based on observed behavior
- Always set resource requests for accurate HPA scaling
- Use readiness and liveness probes for healthy scaling
- Monitor scaling events and adjust policies based on patterns
- Test scaling behavior under realistic load conditions

### Security Considerations
- Use RBAC to limit access to scaling configurations
- Secure metrics endpoints and dashboards
- Monitor for unusual scaling patterns that might indicate attacks
- Set maximum replica limits to prevent resource exhaustion

## 🤝 Contributing

When contributing to the auto-scaling infrastructure:

1. Test changes in a development environment first
2. Update metrics and monitoring for new services
3. Document scaling behavior and thresholds
4. Add integration tests for scaling scenarios
5. Follow the existing configuration patterns

For more information, see the main [SutazAI documentation](../../README.md).