# SutazAI Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying SutazAI to production using Kubernetes, Docker, and Infrastructure as Code.

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   CloudFront    │────▶│ Load Balancer│────▶│   Kubernetes    │
│      (CDN)      │     │    (ALB)     │     │    Cluster      │
└─────────────────┘     └──────────────┘     └─────────────────┘
                                                      │
                              ┌───────────────────────┼───────────────────────┐
                              │                       │                       │
                        ┌─────▼─────┐          ┌─────▼─────┐          ┌─────▼─────┐
                        │  Backend  │          │ Frontend  │          │  Ollama   │
                        │   Pods    │          │   Pods    │          │   Pods    │
                        └─────┬─────┘          └───────────┘          └───────────┘
                              │
                    ┌─────────┼─────────┬─────────────┬────────────┐
                    │         │         │             │            │
              ┌─────▼───┐ ┌──▼───┐ ┌──▼────┐ ┌─────▼────┐ ┌─────▼────┐
              │Postgres │ │Redis │ │ChromaDB│ │  Qdrant  │ │  Neo4j   │
              │  (RDS)  │ │(Cache)│ │(Vector)│ │ (Vector) │ │ (Graph)  │
              └─────────┘ └──────┘ └────────┘ └──────────┘ └──────────┘
```

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **Domain Name** (sutazai.ai) configured in Route53
3. **SSL Certificates** via AWS Certificate Manager
4. **Docker Hub Account** for image registry
5. **Tools Installed**:
   - Docker (>= 20.10)
   - kubectl (>= 1.28)
   - helm (>= 3.12)
   - terraform (>= 1.5)
   - aws-cli (>= 2.0)

## Deployment Steps

### 1. Infrastructure Provisioning

```bash
# Clone the repository
git clone https://github.com/sutazai/sutazai-app.git
cd sutazai-app

# Configure AWS credentials
aws configure

# Deploy infrastructure with Terraform
cd terraform/environments/prod
terraform init
terraform plan -out=tfplan
terraform apply tfplan
```

### 2. Build and Push Docker Images

```bash
# Set version
export VERSION=v1.0.0

# Build images
docker build -f docker/production/backend.Dockerfile -t sutazai/backend:$VERSION .
docker build -f docker/production/frontend.Dockerfile -t sutazai/frontend:$VERSION .

# Login to registry
docker login

# Push images
docker push sutazai/backend:$VERSION
docker push sutazai/frontend:$VERSION
```

### 3. Deploy to Kubernetes

```bash
# Update kubeconfig
aws eks update-kubeconfig --name sutazai-prod --region us-west-2

# Create namespace
kubectl create namespace sutazai

# Create secrets
kubectl create secret generic backend-secret \
  --from-literal=SECRET_KEY='your-secret-key' \
  --from-literal=DATABASE_URL='postgresql://user:pass@host/db' \
  -n sutazai

# Deploy with Kustomize
kubectl apply -k k8s/overlays/prod

# Verify deployment
kubectl get pods -n sutazai
kubectl get services -n sutazai
```

### 4. Configure DNS

```bash
# Get Load Balancer URL
LB_URL=$(kubectl get service -n sutazai sutazai-ingress-nginx -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

# Update Route53 records
# Point sutazai.ai and *.sutazai.ai to the Load Balancer
```

### 5. Deploy Monitoring

```bash
# Create monitoring namespace
kubectl create namespace monitoring

# Install Prometheus
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --values monitoring/helm/prometheus-values.yaml

# Install Loki
helm repo add grafana https://grafana.github.io/helm-charts
helm install loki grafana/loki-stack \
  --namespace monitoring \
  --values monitoring/helm/loki-values.yaml

# Access Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
# Default credentials: admin/prom-operator
```

### 6. Run Database Migrations

```bash
# Create migration job
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migration
  namespace: sutazai
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: migrate
        image: sutazai/backend:$VERSION
        command: ["alembic", "upgrade", "head"]
        envFrom:
        - secretRef:
            name: backend-secret
EOF

# Check migration status
kubectl logs -n sutazai job/db-migration
```

### 7. Load AI Models

```bash
# Exec into Ollama pod
kubectl exec -it -n sutazai deployment/ollama -- bash

# Pull required models
ollama pull tinyllama
ollama pull deepseek-r1:8b
ollama pull qwen2.5:8b
ollama pull codellama:7b
```

## Post-Deployment

### Health Checks

```bash
# API Health
curl https://api.sutazai.ai/health

# Frontend Health
curl https://sutazai.ai/healthz

# Metrics
curl https://prometheus.sutazai.ai/api/v1/query?query=up
```

### Scaling

```bash
# Scale backend
kubectl scale deployment backend -n sutazai --replicas=10

# Enable HPA
kubectl autoscale deployment backend -n sutazai \
  --min=3 --max=20 --cpu-percent=70
```

### Backup Strategy

1. **Database**: Automated RDS snapshots daily
2. **Persistent Volumes**: EBS snapshots via AWS Backup
3. **Configuration**: GitOps with version control
4. **Secrets**: AWS Secrets Manager with rotation

## Monitoring and Alerts

### Key Metrics to Monitor

1. **Application Metrics**
   - Request rate and latency
   - Error rate
   - Active connections
   - Queue depth

2. **Infrastructure Metrics**
   - CPU and Memory usage
   - Disk I/O
   - Network throughput
   - Pod restarts

3. **Business Metrics**
   - User registrations
   - API usage
   - Model inference time
   - Token consumption

### Alert Configuration

Alerts are configured in `monitoring/alerts/alerts.yml` for:
- Service downtime
- High error rates
- Resource exhaustion
- Security incidents

## Security Considerations

1. **Network Security**
   - Private subnets for compute resources
   - Security groups with least privilege
   - Network policies in Kubernetes

2. **Data Security**
   - Encryption at rest (KMS)
   - Encryption in transit (TLS)
   - Secrets management

3. **Access Control**
   - RBAC for Kubernetes
   - IAM roles for AWS resources
   - API authentication/authorization

## Disaster Recovery

### Backup Procedures

```bash
# Manual database backup
kubectl exec -n sutazai postgres-0 -- pg_dump -U sutazai sutazai > backup.sql

# Restore from backup
kubectl exec -i -n sutazai postgres-0 -- psql -U sutazai sutazai < backup.sql
```

### Rollback Procedures

```bash
# Rollback deployment
kubectl rollout undo deployment/backend -n sutazai

# Rollback to specific revision
kubectl rollout undo deployment/backend -n sutazai --to-revision=2
```

## Troubleshooting

### Common Issues

1. **Pods not starting**
   ```bash
   kubectl describe pod <pod-name> -n sutazai
   kubectl logs <pod-name> -n sutazai
   ```

2. **Database connection issues**
   ```bash
   kubectl exec -it deployment/backend -n sutazai -- /bin/bash
   nc -zv postgres 5432
   ```

3. **High memory usage**
   ```bash
   kubectl top pods -n sutazai
   kubectl top nodes
   ```

## Cost Optimization

1. **Use Spot Instances** for non-critical workloads
2. **Enable autoscaling** with appropriate thresholds
3. **Use Reserved Instances** for baseline capacity
4. **Implement request/limit ratios** to avoid overprovisioning
5. **Use S3 lifecycle policies** for log retention

## Maintenance

### Regular Tasks

- **Weekly**: Review metrics and logs
- **Monthly**: Security patches and updates
- **Quarterly**: Disaster recovery drills
- **Annually**: Architecture review and optimization

### Update Procedures

```bash
# Update application
./scripts/deploy-production.sh

# Update Kubernetes
eksctl upgrade cluster --name sutazai-prod

# Update infrastructure
cd terraform/environments/prod
terraform plan
terraform apply
```

## Support

For issues or questions:
- **Documentation**: /docs
- **Logs**: Grafana/Loki dashboard
- **Metrics**: Prometheus/Grafana
- **Alerts**: Configure PagerDuty/Slack integration