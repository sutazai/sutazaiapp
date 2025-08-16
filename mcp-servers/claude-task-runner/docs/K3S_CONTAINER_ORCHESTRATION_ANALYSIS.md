# K3s Container Orchestration Analysis & Implementation Strategy
## SutazAI Service Mesh Expansion: 25 → 27 Containers

**Date**: 2025-08-16  
**Author**: K3s Container Orchestration Specialist  
**Version**: 1.0.0  
**Status**: COMPREHENSIVE ANALYSIS

---

## Executive Summary

This document provides a comprehensive analysis of container orchestration strategies for expanding the SutazAI service mesh from 25 to 27 containers by adding task-decomposition-service and workspace-isolation-service. The analysis focuses on K3s lightweight Kubernetes deployment for edge computing efficiency, resource optimization, and seamless mesh integration.

---

## 1. Current Infrastructure Analysis

### 1.1 Existing Container Architecture (25 Services)

#### Core Infrastructure Tier (Ports 10000-10099)
- **PostgreSQL** (10000): Primary database with 2GB memory limit
- **Redis** (10001): Cache layer with 1GB memory limit  
- **Neo4j** (10002-10003): Graph database with 1GB memory limit
- **Kong API Gateway** (10005, 10015): API management layer
- **Consul** (10006): Service discovery
- **RabbitMQ** (10007-10008): Message queue system
- **FastAPI Backend** (10010): Core application backend
- **Streamlit Frontend** (10011): User interface

#### AI & Vector Services Tier (Ports 10100-10199)
- **ChromaDB** (10100): Vector database with 1GB memory limit
- **Qdrant** (10101-10102): Alternative vector store
- **Ollama** (10104): LLM server with 4GB memory limit - CRITICAL SERVICE
- **FAISS** (10103): Vector similarity search [DEFINED BUT NOT RUNNING]

#### Monitoring Stack (Ports 10200-10299)
- **Prometheus** (10200): Metrics collection
- **Grafana** (10201): Visualization dashboards
- **Loki** (10202): Log aggregation
- **AlertManager** (10203): Alert routing
- **Jaeger** (10210-10215): Distributed tracing
- Multiple exporters for comprehensive observability

#### Agent Services (Ports 11000+)
- Limited deployment of specialized agents
- Ultra System Architect (11200) currently active

### 1.2 Resource Analysis

**Current Total Resource Allocation:**
- **CPU**: ~15 cores reserved, ~30 cores limit
- **Memory**: ~12GB reserved, ~25GB limit
- **Storage**: Multiple persistent volumes for stateful services
- **Network**: Single overlay network (sutazai-network)

---

## 2. Proposed Container Expansion

### 2.1 New Services Architecture

#### task-decomposition-service (Port 10030)
```yaml
Purpose: Intelligent task breakdown and delegation
Technology: Python-based microservice
Dependencies:
  - MCP infrastructure integration
  - Claude Flow coordination APIs
  - Redis for task queue management
  - PostgreSQL for task persistence
Resource Requirements:
  - CPU: 0.5 cores reserved, 2.0 cores limit
  - Memory: 512MB reserved, 2GB limit
  - Persistent volume for task artifacts
```

#### workspace-isolation-service (Port 10031)
```yaml
Purpose: Git worktree and environment isolation
Technology: Container orchestration service
Dependencies:
  - Docker socket access (privileged mode)
  - Git repository volumes
  - Shared workspace volumes
Resource Requirements:
  - CPU: 0.25 cores reserved, 1.0 core limit
  - Memory: 256MB reserved, 1GB limit
  - Multiple bind mounts for workspace access
```

---

## 3. K3s Orchestration Strategy

### 3.1 Why K3s for This Architecture

**Advantages:**
1. **Lightweight footprint**: 40MB binary, 512MB memory minimum
2. **Built-in components**: Traefik, CoreDNS, ServiceLB included
3. **Edge optimization**: Perfect for distributed AI workloads
4. **Single-node capable**: Can run entire stack on one machine
5. **Multi-cluster ready**: Easy federation for scaling

### 3.2 K3s Deployment Architecture

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: sutazai-mesh
---
# Core Tier Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: task-decomposition
  namespace: sutazai-mesh
spec:
  replicas: 2  # HA configuration
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: task-decomposition
  template:
    metadata:
      labels:
        app: task-decomposition
        tier: core-services
        mesh: sutazai
    spec:
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - task-decomposition
              topologyKey: kubernetes.io/hostname
      containers:
      - name: task-decomposition
        image: sutazai/task-decomposition:v1.0.0
        ports:
        - containerPort: 8080
          name: http
          protocol: TCP
        - containerPort: 9090
          name: metrics
          protocol: TCP
        env:
        - name: REDIS_URL
          value: "redis://sutazai-redis:6379"
        - name: POSTGRES_URL
          valueFrom:
            secretKeyRef:
              name: postgres-credentials
              key: connection-string
        - name: MCP_ENDPOINT
          value: "http://mcp-coordinator:8000"
        - name: CLAUDE_FLOW_API
          value: "http://claude-flow:3000"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: task-artifacts
          mountPath: /data/tasks
        - name: config
          mountPath: /app/config
          readOnly: true
      volumes:
      - name: task-artifacts
        persistentVolumeClaim:
          claimName: task-artifacts-pvc
      - name: config
        configMap:
          name: task-decomposition-config
---
# Workspace Isolation Deployment
apiVersion: apps/v1
kind: DaemonSet  # One per node for local workspace access
metadata:
  name: workspace-isolation
  namespace: sutazai-mesh
spec:
  selector:
    matchLabels:
      app: workspace-isolation
  template:
    metadata:
      labels:
        app: workspace-isolation
        tier: infrastructure
        mesh: sutazai
    spec:
      hostNetwork: true  # Direct host network access
      hostPID: true      # Process namespace sharing
      containers:
      - name: workspace-isolation
        image: sutazai/workspace-isolation:v1.0.0
        securityContext:
          privileged: true  # Required for Docker-in-Docker
          capabilities:
            add:
            - SYS_ADMIN
            - NET_ADMIN
        ports:
        - containerPort: 8081
          hostPort: 10031
          name: http
        env:
        - name: DOCKER_HOST
          value: "unix:///var/run/docker.sock"
        - name: GIT_WORKSPACE_ROOT
          value: "/workspace"
        - name: ISOLATION_MODE
          value: "strict"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        volumeMounts:
        - name: docker-socket
          mountPath: /var/run/docker.sock
        - name: workspace-root
          mountPath: /workspace
        - name: git-repos
          mountPath: /repos
      volumes:
      - name: docker-socket
        hostPath:
          path: /var/run/docker.sock
          type: Socket
      - name: workspace-root
        hostPath:
          path: /opt/sutazai/workspaces
          type: DirectoryOrCreate
      - name: git-repos
        hostPath:
          path: /opt/sutazai/repos
          type: DirectoryOrCreate
```

### 3.3 Service Mesh Integration

```yaml
# Service definitions for mesh participation
apiVersion: v1
kind: Service
metadata:
  name: task-decomposition-service
  namespace: sutazai-mesh
  labels:
    app: task-decomposition
    monitoring: prometheus
spec:
  type: LoadBalancer
  ports:
  - port: 10030
    targetPort: 8080
    protocol: TCP
    name: http
  - port: 19030
    targetPort: 9090
    protocol: TCP
    name: metrics
  selector:
    app: task-decomposition
---
apiVersion: v1
kind: Service
metadata:
  name: workspace-isolation-service
  namespace: sutazai-mesh
  labels:
    app: workspace-isolation
    monitoring: prometheus
spec:
  type: NodePort
  ports:
  - port: 10031
    targetPort: 8081
    nodePort: 30031
    protocol: TCP
    name: http
  selector:
    app: workspace-isolation
```

---

## 4. Resource Management Strategy

### 4.1 Resource Allocation Model

**Tier-Based Resource Prioritization:**

```yaml
Priority Classes:
  1. Critical (1000): Database, Cache, Message Queue
  2. High (800): Core Services, API Gateway
  3. Medium (600): AI/ML Services, New Services
  4. Low (400): Monitoring, Logging
  5. BestEffort (200): Development, Testing

Resource Quotas per Namespace:
  sutazai-core:
    CPU: 10 cores
    Memory: 16GB
    Storage: 100GB
  sutazai-ai:
    CPU: 8 cores
    Memory: 12GB
    Storage: 50GB
  sutazai-monitoring:
    CPU: 4 cores
    Memory: 6GB
    Storage: 20GB
```

### 4.2 Horizontal Pod Autoscaling (HPA)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: task-decomposition-hpa
  namespace: sutazai-mesh
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: task-decomposition
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
  - type: Pods
    pods:
      metric:
        name: task_queue_depth
      target:
        type: AverageValue
        averageValue: "30"
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
      - type: Pods
        value: 2
        periodSeconds: 60
```

### 4.3 Vertical Pod Autoscaling (VPA)

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: workspace-isolation-vpa
  namespace: sutazai-mesh
spec:
  targetRef:
    apiVersion: apps/v1
    kind: DaemonSet
    name: workspace-isolation
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: workspace-isolation
      minAllowed:
        cpu: 100m
        memory: 128Mi
      maxAllowed:
        cpu: 2000m
        memory: 2Gi
      controlledResources: ["cpu", "memory"]
```

---

## 5. Networking Patterns

### 5.1 Service Mesh Architecture

```yaml
# NetworkPolicy for service isolation
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: task-decomposition-netpol
  namespace: sutazai-mesh
spec:
  podSelector:
    matchLabels:
      app: task-decomposition
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: sutazai-mesh
    - podSelector:
        matchLabels:
          tier: gateway
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: sutazai-mesh
    ports:
    - protocol: TCP
      port: 6379  # Redis
    - protocol: TCP
      port: 5432  # PostgreSQL
  - to:
    - podSelector:
        matchLabels:
          app: mcp-coordinator
    ports:
    - protocol: TCP
      port: 8000
```

### 5.2 Ingress Configuration

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sutazai-services-ingress
  namespace: sutazai-mesh
  annotations:
    kubernetes.io/ingress.class: traefik
    traefik.ingress.kubernetes.io/router.entrypoints: web,websecure
    traefik.ingress.kubernetes.io/router.middlewares: default-ratelimit@kubernetescrd
spec:
  rules:
  - host: tasks.sutazai.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: task-decomposition-service
            port:
              number: 10030
  - host: workspace.sutazai.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: workspace-isolation-service
            port:
              number: 10031
  tls:
  - hosts:
    - tasks.sutazai.local
    - workspace.sutazai.local
    secretName: sutazai-tls-cert
```

---

## 6. Health Checks & Monitoring

### 6.1 Comprehensive Health Check Strategy

```yaml
# Liveness, Readiness, and Startup Probes
livenessProbe:
  httpGet:
    path: /health/live
    port: 8080
    httpHeaders:
    - name: X-Health-Check
      value: liveness
  initialDelaySeconds: 60
  periodSeconds: 10
  timeoutSeconds: 5
  successThreshold: 1
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8080
    httpHeaders:
    - name: X-Health-Check
      value: readiness
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  successThreshold: 1
  failureThreshold: 3

startupProbe:
  httpGet:
    path: /health/startup
    port: 8080
  initialDelaySeconds: 0
  periodSeconds: 10
  timeoutSeconds: 10
  successThreshold: 1
  failureThreshold: 30
```

### 6.2 Prometheus Monitoring Integration

```yaml
apiVersion: v1
kind: ServiceMonitor
metadata:
  name: task-decomposition-monitor
  namespace: sutazai-mesh
spec:
  selector:
    matchLabels:
      app: task-decomposition
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
    scrapeTimeout: 10s
    relabelings:
    - sourceLabels: [__meta_kubernetes_pod_node_name]
      targetLabel: node
    - sourceLabels: [__meta_kubernetes_namespace]
      targetLabel: namespace
    - sourceLabels: [__meta_kubernetes_pod_name]
      targetLabel: pod
    - sourceLabels: [__meta_kubernetes_pod_container_name]
      targetLabel: container
```

### 6.3 Custom Metrics for Autoscaling

```yaml
# Custom metrics for HPA
apiVersion: v1
kind: ConfigMap
metadata:
  name: adapter-config
  namespace: monitoring
data:
  config.yaml: |
    rules:
    - seriesQuery: 'task_queue_depth{namespace="sutazai-mesh"}'
      resources:
        overrides:
          namespace: {resource: "namespace"}
          pod: {resource: "pod"}
      name:
        matches: "^task_queue_depth"
        as: "task_queue_depth"
      metricsQuery: 'avg_over_time(task_queue_depth{<<.LabelMatchers>>}[2m])'
```

---

## 7. Security Implications

### 7.1 Container Security Policies

```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: sutazai-restricted
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
  - ALL
  volumes:
  - 'configMap'
  - 'emptyDir'
  - 'projected'
  - 'secret'
  - 'downwardAPI'
  - 'persistentVolumeClaim'
  hostNetwork: false
  hostIPC: false
  hostPID: false
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  supplementalGroups:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
  readOnlyRootFilesystem: true
---
# Exception for workspace-isolation (needs privileged)
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: workspace-isolation-psp
spec:
  privileged: true
  allowPrivilegeEscalation: true
  allowedCapabilities:
  - SYS_ADMIN
  - NET_ADMIN
  volumes:
  - '*'
  hostNetwork: true
  hostPID: true
  hostIPC: false
  runAsUser:
    rule: 'RunAsAny'
  seLinux:
    rule: 'RunAsAny'
  supplementalGroups:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

### 7.2 RBAC Configuration

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: task-decomposition-role
  namespace: sutazai-mesh
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list"]
- apiGroups: ["batch"]
  resources: ["jobs"]
  verbs: ["create", "get", "list", "watch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: workspace-isolation-role
rules:
- apiGroups: [""]
  resources: ["nodes", "pods", "namespaces"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods/exec"]
  verbs: ["create"]
- apiGroups: ["apps"]
  resources: ["deployments", "daemonsets"]
  verbs: ["get", "list"]
```

### 7.3 Network Security

```yaml
# Calico NetworkPolicy for fine-grained control
apiVersion: projectcalico.org/v3
kind: NetworkPolicy
metadata:
  name: task-decomposition-calico
  namespace: sutazai-mesh
spec:
  selector: app == 'task-decomposition'
  types:
  - Ingress
  - Egress
  ingress:
  - action: Allow
    protocol: TCP
    source:
      selector: tier == 'gateway'
    destination:
      ports:
      - 8080
  egress:
  - action: Allow
    protocol: TCP
    destination:
      selector: app == 'redis'
      ports:
      - 6379
  - action: Allow
    protocol: TCP
    destination:
      selector: app == 'postgres'
      ports:
      - 5432
  - action: Log
    protocol: TCP
    destination:
      nets:
      - 0.0.0.0/0
```

---

## 8. Bottleneck Analysis

### 8.1 Identified Bottlenecks

1. **Database Connections**
   - Current: 100 max connections
   - With 27 services: Potential exhaustion
   - Solution: Connection pooling, PgBouncer

2. **Message Queue Throughput**
   - Current: Single RabbitMQ instance
   - Risk: Message backlog with task decomposition
   - Solution: RabbitMQ cluster, partition tolerance

3. **Network Bandwidth**
   - Inter-container communication overhead
   - Solution: Container affinity, local communication

4. **Storage I/O**
   - Shared volume contention
   - Solution: Dedicated volumes, SSD storage class

### 8.2 Mitigation Strategies

```yaml
# PgBouncer for connection pooling
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pgbouncer
  namespace: sutazai-mesh
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pgbouncer
  template:
    metadata:
      labels:
        app: pgbouncer
    spec:
      containers:
      - name: pgbouncer
        image: pgbouncer/pgbouncer:1.22.0
        ports:
        - containerPort: 5432
        env:
        - name: DATABASES_HOST
          value: "sutazai-postgres"
        - name: DATABASES_PORT
          value: "5432"
        - name: DATABASES_DATABASE
          value: "sutazai"
        - name: POOL_MODE
          value: "transaction"
        - name: MAX_CLIENT_CONN
          value: "1000"
        - name: DEFAULT_POOL_SIZE
          value: "25"
        resources:
          requests:
            memory: "64Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "500m"
```

---

## 9. Implementation Roadmap

### Phase 1: Infrastructure Preparation (Week 1)
- [ ] K3s cluster deployment
- [ ] Namespace creation and RBAC setup
- [ ] Storage class configuration
- [ ] Network policy implementation

### Phase 2: Service Migration (Week 2)
- [ ] Convert Docker Compose to K8s manifests
- [ ] Deploy existing 25 services to K3s
- [ ] Validate service mesh connectivity
- [ ] Performance baseline establishment

### Phase 3: New Service Integration (Week 3)
- [ ] Deploy task-decomposition-service
- [ ] Deploy workspace-isolation-service
- [ ] Configure service mesh participation
- [ ] Implement health checks and monitoring

### Phase 4: Optimization (Week 4)
- [ ] Enable autoscaling policies
- [ ] Tune resource limits
- [ ] Implement caching strategies
- [ ] Performance testing and validation

### Phase 5: Production Readiness (Week 5)
- [ ] Disaster recovery testing
- [ ] Security audit
- [ ] Documentation completion
- [ ] Team training

---

## 10. Validation Checklist

### Pre-Deployment
- [ ] Resource capacity analysis completed
- [ ] Network policies defined
- [ ] Security policies implemented
- [ ] Monitoring dashboards created
- [ ] Backup strategies defined

### During Deployment
- [ ] Services starting successfully
- [ ] Health checks passing
- [ ] Metrics being collected
- [ ] Logs aggregating properly
- [ ] Network connectivity verified

### Post-Deployment
- [ ] Performance metrics meeting SLAs
- [ ] Autoscaling functioning correctly
- [ ] No resource contention detected
- [ ] Security scans passing
- [ ] Documentation updated

---

## 11. Conclusion

The expansion from 25 to 27 containers using K3s orchestration provides:

1. **Scalability**: Horizontal and vertical autoscaling
2. **Reliability**: Health checks, self-healing, rolling updates
3. **Security**: RBAC, network policies, pod security
4. **Observability**: Comprehensive monitoring and tracing
5. **Efficiency**: Resource optimization, connection pooling

The K3s approach offers lightweight Kubernetes orchestration perfect for edge computing scenarios while maintaining enterprise-grade features for production workloads.

---

## Appendix A: K3s Installation Script

```bash
#!/bin/bash
# K3s installation for SutazAI mesh

# Install K3s server
curl -sfL https://get.k3s.io | sh -s - \
  --write-kubeconfig-mode 644 \
  --disable traefik \
  --disable servicelb \
  --disable-network-policy \
  --flannel-backend=vxlan \
  --node-name="sutazai-master" \
  --cluster-cidr="10.42.0.0/16" \
  --service-cidr="10.43.0.0/16" \
  --cluster-dns="10.43.0.10"

# Wait for K3s to be ready
kubectl wait --for=condition=Ready nodes --all --timeout=300s

# Install Calico for network policies
kubectl apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.26.0/manifests/tigera-operator.yaml
kubectl apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.26.0/manifests/custom-resources.yaml

# Install MetalLB for LoadBalancer services
kubectl apply -f https://raw.githubusercontent.com/metallb/metallb/v0.13.12/config/manifests/metallb-native.yaml

# Configure MetalLB IP pool
cat <<EOF | kubectl apply -f -
apiVersion: metallb.io/v1beta1
kind: IPAddressPool
metadata:
  name: sutazai-pool
  namespace: metallb-system
spec:
  addresses:
  - 192.168.1.240-192.168.1.250
---
apiVersion: metallb.io/v1beta1
kind: L2Advertisement
metadata:
  name: sutazai-l2
  namespace: metallb-system
EOF

# Install metrics-server for HPA
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Create namespaces
kubectl create namespace sutazai-mesh
kubectl create namespace sutazai-monitoring
kubectl label namespace sutazai-mesh name=sutazai-mesh

echo "K3s cluster ready for SutazAI deployment"
```

---

## Appendix B: Helm Chart Structure

```yaml
# Chart.yaml
apiVersion: v2
name: sutazai-mesh
description: SutazAI Service Mesh Helm Chart
type: application
version: 1.0.0
appVersion: "2.0.0"

# values.yaml
global:
  imageRegistry: "docker.io/sutazai"
  imagePullSecrets: []
  storageClass: "local-path"
  
taskDecomposition:
  enabled: true
  replicaCount: 2
  image:
    repository: task-decomposition
    tag: "v1.0.0"
    pullPolicy: IfNotPresent
  service:
    type: LoadBalancer
    port: 10030
  resources:
    requests:
      memory: "512Mi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "2000m"
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
    
workspaceIsolation:
  enabled: true
  image:
    repository: workspace-isolation
    tag: "v1.0.0"
    pullPolicy: IfNotPresent
  service:
    type: NodePort
    port: 10031
    nodePort: 30031
  resources:
    requests:
      memory: "256Mi"
      cpu: "250m"
    limits:
      memory: "1Gi"
      cpu: "1000m"
  securityContext:
    privileged: true
    capabilities:
      add:
      - SYS_ADMIN
      - NET_ADMIN
```

---

**Document Version**: 1.0.0  
**Last Updated**: 2025-08-16  
**Next Review**: 2025-08-23  
**Status**: READY FOR IMPLEMENTATION