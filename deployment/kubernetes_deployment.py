"""
SutazAI Kubernetes Deployment System
Enterprise-scale orchestration for the AGI/ASI system

This module provides comprehensive Kubernetes deployment capabilities
including auto-scaling, service mesh, monitoring, and high availability.
"""

import os
import json
import yaml
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KubernetesEnvironment(Enum):
    """Kubernetes environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class KubernetesConfig:
    """Kubernetes configuration settings"""
    namespace: str = "sutazai"
    image_name: str = "sutazai/agi-system"
    image_tag: str = "latest"
    replicas: int = 3
    max_replicas: int = 10
    min_replicas: int = 2
    cpu_request: str = "100m"
    cpu_limit: str = "1000m"
    memory_request: str = "512Mi"
    memory_limit: str = "2Gi"
    service_port: int = 8000
    ingress_host: str = "sutazai.example.com"
    storage_size: str = "10Gi"
    storage_class: str = "fast-ssd"

class KubernetesDeploymentManager:
    """Manages Kubernetes-based deployment of the SutazAI system"""
    
    def __init__(self, config: KubernetesConfig = None):
        self.config = config or KubernetesConfig()
        self.base_dir = Path("/opt/sutazaiapp")
        self.deployment_dir = self.base_dir / "deployment" / "kubernetes"
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Kubernetes client
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        
        self.k8s_client = client.ApiClient()
        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
        self.networking_v1 = client.NetworkingV1Api()
        self.autoscaling_v1 = client.AutoscalingV1Api()
        
        logger.info("Kubernetes Deployment Manager initialized")
    
    def create_namespace_manifest(self) -> str:
        """Create namespace manifest"""
        namespace_manifest = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": self.config.namespace,
                "labels": {
                    "name": self.config.namespace,
                    "app": "sutazai-agi"
                }
            }
        }
        
        manifest_path = self.deployment_dir / "namespace.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(namespace_manifest, f, default_flow_style=False)
        
        logger.info(f"Created namespace manifest")
        return str(manifest_path)
    
    def create_configmap_manifest(self, environment: KubernetesEnvironment) -> str:
        """Create ConfigMap manifest for application configuration"""
        configmap_data = {
            "settings.json": json.dumps({
                "database": {
                    "host": "postgres-service",
                    "port": 5432,
                    "name": "sutazai",
                    "user": "sutazai"
                },
                "redis": {
                    "host": "redis-service",
                    "port": 6379
                },
                "neural_network": {
                    "default_nodes": 100,
                    "learning_rate": 0.01,
                    "activation_threshold": 0.5
                },
                "security": {
                    "jwt_expiration": 3600,
                    "authorized_user": "chrissuta01@gmail.com"
                },
                "performance": {
                    "max_workers": 10,
                    "task_queue_size": 1000,
                    "memory_limit": "2G"
                }
            }, indent=2),
            "nginx.conf": self._create_nginx_config_content()
        }
        
        configmap_manifest = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"sutazai-config-{environment.value}",
                "namespace": self.config.namespace
            },
            "data": configmap_data
        }
        
        manifest_path = self.deployment_dir / f"configmap-{environment.value}.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(configmap_manifest, f, default_flow_style=False)
        
        logger.info(f"Created ConfigMap manifest for {environment.value}")
        return str(manifest_path)
    
    def create_secret_manifest(self, environment: KubernetesEnvironment) -> str:
        """Create Secret manifest for sensitive data"""
        import base64
        
        secret_data = {
            "database-password": base64.b64encode(b"secure_password_123").decode('utf-8'),
            "jwt-secret": base64.b64encode(b"jwt_secret_key_123").decode('utf-8'),
            "redis-password": base64.b64encode(b"redis_password_123").decode('utf-8')
        }
        
        secret_manifest = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": f"sutazai-secrets-{environment.value}",
                "namespace": self.config.namespace
            },
            "type": "Opaque",
            "data": secret_data
        }
        
        manifest_path = self.deployment_dir / f"secret-{environment.value}.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(secret_manifest, f, default_flow_style=False)
        
        logger.info(f"Created Secret manifest for {environment.value}")
        return str(manifest_path)
    
    def create_persistent_volume_claim(self, environment: KubernetesEnvironment) -> str:
        """Create PersistentVolumeClaim manifest"""
        pvc_manifest = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": f"sutazai-data-{environment.value}",
                "namespace": self.config.namespace
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "storageClassName": self.config.storage_class,
                "resources": {
                    "requests": {
                        "storage": self.config.storage_size
                    }
                }
            }
        }
        
        manifest_path = self.deployment_dir / f"pvc-{environment.value}.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(pvc_manifest, f, default_flow_style=False)
        
        logger.info(f"Created PVC manifest for {environment.value}")
        return str(manifest_path)
    
    def create_deployment_manifest(self, environment: KubernetesEnvironment) -> str:
        """Create Deployment manifest for the main application"""
        deployment_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"sutazai-agi-{environment.value}",
                "namespace": self.config.namespace,
                "labels": {
                    "app": "sutazai-agi",
                    "environment": environment.value
                }
            },
            "spec": {
                "replicas": self.config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": "sutazai-agi",
                        "environment": environment.value
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "sutazai-agi",
                            "environment": environment.value
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "sutazai-agi",
                                "image": f"{self.config.image_name}:{environment.value}",
                                "ports": [
                                    {
                                        "containerPort": self.config.service_port,
                                        "protocol": "TCP"
                                    }
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": self.config.cpu_request,
                                        "memory": self.config.memory_request
                                    },
                                    "limits": {
                                        "cpu": self.config.cpu_limit,
                                        "memory": self.config.memory_limit
                                    }
                                },
                                "env": [
                                    {
                                        "name": "SUTAZAI_ENV",
                                        "value": environment.value
                                    },
                                    {
                                        "name": "SUTAZAI_CONFIG_PATH",
                                        "value": "/app/config/settings.json"
                                    },
                                    {
                                        "name": "DATABASE_PASSWORD",
                                        "valueFrom": {
                                            "secretKeyRef": {
                                                "name": f"sutazai-secrets-{environment.value}",
                                                "key": "database-password"
                                            }
                                        }
                                    },
                                    {
                                        "name": "JWT_SECRET",
                                        "valueFrom": {
                                            "secretKeyRef": {
                                                "name": f"sutazai-secrets-{environment.value}",
                                                "key": "jwt-secret"
                                            }
                                        }
                                    }
                                ],
                                "volumeMounts": [
                                    {
                                        "name": "config-volume",
                                        "mountPath": "/app/config"
                                    },
                                    {
                                        "name": "data-volume",
                                        "mountPath": "/app/data"
                                    }
                                ],
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": "/health",
                                        "port": self.config.service_port
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": "/health",
                                        "port": self.config.service_port
                                    },
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5
                                }
                            }
                        ],
                        "volumes": [
                            {
                                "name": "config-volume",
                                "configMap": {
                                    "name": f"sutazai-config-{environment.value}"
                                }
                            },
                            {
                                "name": "data-volume",
                                "persistentVolumeClaim": {
                                    "claimName": f"sutazai-data-{environment.value}"
                                }
                            }
                        ]
                    }
                }
            }
        }
        
        manifest_path = self.deployment_dir / f"deployment-{environment.value}.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(deployment_manifest, f, default_flow_style=False)
        
        logger.info(f"Created Deployment manifest for {environment.value}")
        return str(manifest_path)
    
    def create_service_manifest(self, environment: KubernetesEnvironment) -> str:
        """Create Service manifest"""
        service_manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"sutazai-agi-service-{environment.value}",
                "namespace": self.config.namespace,
                "labels": {
                    "app": "sutazai-agi",
                    "environment": environment.value
                }
            },
            "spec": {
                "selector": {
                    "app": "sutazai-agi",
                    "environment": environment.value
                },
                "ports": [
                    {
                        "port": 80,
                        "targetPort": self.config.service_port,
                        "protocol": "TCP"
                    }
                ],
                "type": "ClusterIP"
            }
        }
        
        manifest_path = self.deployment_dir / f"service-{environment.value}.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(service_manifest, f, default_flow_style=False)
        
        logger.info(f"Created Service manifest for {environment.value}")
        return str(manifest_path)
    
    def create_ingress_manifest(self, environment: KubernetesEnvironment) -> str:
        """Create Ingress manifest for external access"""
        ingress_manifest = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": f"sutazai-agi-ingress-{environment.value}",
                "namespace": self.config.namespace,
                "annotations": {
                    "kubernetes.io/ingress.class": "nginx",
                    "nginx.ingress.kubernetes.io/rewrite-target": "/",
                    "nginx.ingress.kubernetes.io/ssl-redirect": "true",
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod"
                }
            },
            "spec": {
                "tls": [
                    {
                        "hosts": [self.config.ingress_host],
                        "secretName": f"sutazai-tls-{environment.value}"
                    }
                ],
                "rules": [
                    {
                        "host": self.config.ingress_host,
                        "http": {
                            "paths": [
                                {
                                    "path": "/",
                                    "pathType": "Prefix",
                                    "backend": {
                                        "service": {
                                            "name": f"sutazai-agi-service-{environment.value}",
                                            "port": {
                                                "number": 80
                                            }
                                        }
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }
        
        manifest_path = self.deployment_dir / f"ingress-{environment.value}.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(ingress_manifest, f, default_flow_style=False)
        
        logger.info(f"Created Ingress manifest for {environment.value}")
        return str(manifest_path)
    
    def create_hpa_manifest(self, environment: KubernetesEnvironment) -> str:
        """Create HorizontalPodAutoscaler manifest"""
        hpa_manifest = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"sutazai-agi-hpa-{environment.value}",
                "namespace": self.config.namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"sutazai-agi-{environment.value}"
                },
                "minReplicas": self.config.min_replicas,
                "maxReplicas": self.config.max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 70
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80
                            }
                        }
                    }
                ]
            }
        }
        
        manifest_path = self.deployment_dir / f"hpa-{environment.value}.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(hpa_manifest, f, default_flow_style=False)
        
        logger.info(f"Created HPA manifest for {environment.value}")
        return str(manifest_path)
    
    def create_database_deployment(self, environment: KubernetesEnvironment) -> str:
        """Create PostgreSQL database deployment"""
        db_deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"postgres-{environment.value}",
                "namespace": self.config.namespace
            },
            "spec": {
                "replicas": 1,
                "selector": {
                    "matchLabels": {
                        "app": "postgres",
                        "environment": environment.value
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "postgres",
                            "environment": environment.value
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "postgres",
                                "image": "postgres:15-alpine",
                                "env": [
                                    {
                                        "name": "POSTGRES_DB",
                                        "value": "sutazai"
                                    },
                                    {
                                        "name": "POSTGRES_USER",
                                        "value": "sutazai"
                                    },
                                    {
                                        "name": "POSTGRES_PASSWORD",
                                        "valueFrom": {
                                            "secretKeyRef": {
                                                "name": f"sutazai-secrets-{environment.value}",
                                                "key": "database-password"
                                            }
                                        }
                                    }
                                ],
                                "ports": [
                                    {
                                        "containerPort": 5432
                                    }
                                ],
                                "volumeMounts": [
                                    {
                                        "name": "postgres-data",
                                        "mountPath": "/var/lib/postgresql/data"
                                    }
                                ]
                            }
                        ],
                        "volumes": [
                            {
                                "name": "postgres-data",
                                "persistentVolumeClaim": {
                                    "claimName": f"postgres-data-{environment.value}"
                                }
                            }
                        ]
                    }
                }
            }
        }
        
        # Create corresponding service
        db_service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "postgres-service",
                "namespace": self.config.namespace
            },
            "spec": {
                "selector": {
                    "app": "postgres",
                    "environment": environment.value
                },
                "ports": [
                    {
                        "port": 5432,
                        "targetPort": 5432
                    }
                ]
            }
        }
        
        # Create PVC for database
        db_pvc = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": f"postgres-data-{environment.value}",
                "namespace": self.config.namespace
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "resources": {
                    "requests": {
                        "storage": "20Gi"
                    }
                }
            }
        }
        
        # Combine all database resources
        db_resources = {
            "apiVersion": "v1",
            "kind": "List",
            "items": [db_deployment, db_service, db_pvc]
        }
        
        manifest_path = self.deployment_dir / f"database-{environment.value}.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(db_resources, f, default_flow_style=False)
        
        logger.info(f"Created database deployment for {environment.value}")
        return str(manifest_path)
    
    def create_monitoring_deployment(self, environment: KubernetesEnvironment) -> str:
        """Create monitoring stack deployment (Prometheus + Grafana)"""
        monitoring_resources = {
            "apiVersion": "v1",
            "kind": "List",
            "items": [
                # Prometheus deployment
                {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "metadata": {
                        "name": f"prometheus-{environment.value}",
                        "namespace": self.config.namespace
                    },
                    "spec": {
                        "replicas": 1,
                        "selector": {
                            "matchLabels": {
                                "app": "prometheus",
                                "environment": environment.value
                            }
                        },
                        "template": {
                            "metadata": {
                                "labels": {
                                    "app": "prometheus",
                                    "environment": environment.value
                                }
                            },
                            "spec": {
                                "containers": [
                                    {
                                        "name": "prometheus",
                                        "image": "prom/prometheus:latest",
                                        "ports": [
                                            {
                                                "containerPort": 9090
                                            }
                                        ],
                                        "args": [
                                            "--config.file=/etc/prometheus/prometheus.yml",
                                            "--storage.tsdb.path=/prometheus/",
                                            "--web.console.libraries=/etc/prometheus/console_libraries",
                                            "--web.console.templates=/etc/prometheus/consoles",
                                            "--web.enable-lifecycle"
                                        ],
                                        "volumeMounts": [
                                            {
                                                "name": "prometheus-config",
                                                "mountPath": "/etc/prometheus"
                                            }
                                        ]
                                    }
                                ],
                                "volumes": [
                                    {
                                        "name": "prometheus-config",
                                        "configMap": {
                                            "name": f"prometheus-config-{environment.value}"
                                        }
                                    }
                                ]
                            }
                        }
                    }
                },
                # Prometheus service
                {
                    "apiVersion": "v1",
                    "kind": "Service",
                    "metadata": {
                        "name": "prometheus-service",
                        "namespace": self.config.namespace
                    },
                    "spec": {
                        "selector": {
                            "app": "prometheus",
                            "environment": environment.value
                        },
                        "ports": [
                            {
                                "port": 9090,
                                "targetPort": 9090
                            }
                        ]
                    }
                },
                # Grafana deployment
                {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "metadata": {
                        "name": f"grafana-{environment.value}",
                        "namespace": self.config.namespace
                    },
                    "spec": {
                        "replicas": 1,
                        "selector": {
                            "matchLabels": {
                                "app": "grafana",
                                "environment": environment.value
                            }
                        },
                        "template": {
                            "metadata": {
                                "labels": {
                                    "app": "grafana",
                                    "environment": environment.value
                                }
                            },
                            "spec": {
                                "containers": [
                                    {
                                        "name": "grafana",
                                        "image": "grafana/grafana:latest",
                                        "ports": [
                                            {
                                                "containerPort": 3000
                                            }
                                        ],
                                        "env": [
                                            {
                                                "name": "GF_SECURITY_ADMIN_PASSWORD",
                                                "value": "admin"
                                            }
                                        ]
                                    }
                                ]
                            }
                        }
                    }
                },
                # Grafana service
                {
                    "apiVersion": "v1",
                    "kind": "Service",
                    "metadata": {
                        "name": "grafana-service",
                        "namespace": self.config.namespace
                    },
                    "spec": {
                        "selector": {
                            "app": "grafana",
                            "environment": environment.value
                        },
                        "ports": [
                            {
                                "port": 3000,
                                "targetPort": 3000
                            }
                        ]
                    }
                }
            ]
        }
        
        manifest_path = self.deployment_dir / f"monitoring-{environment.value}.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(monitoring_resources, f, default_flow_style=False)
        
        logger.info(f"Created monitoring deployment for {environment.value}")
        return str(manifest_path)
    
    def _create_nginx_config_content(self) -> str:
        """Create Nginx configuration content"""
        return """
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server 127.0.0.1:8000;
    }
    
    server {
        listen 80;
        location / {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
    }
}
"""
    
    def deploy(self, environment: KubernetesEnvironment) -> Dict[str, Any]:
        """Deploy the complete SutazAI system to Kubernetes"""
        try:
            logger.info(f"Starting Kubernetes deployment for {environment.value}")
            
            # Create all manifest files
            manifests = {}
            manifests["namespace"] = self.create_namespace_manifest()
            manifests["configmap"] = self.create_configmap_manifest(environment)
            manifests["secret"] = self.create_secret_manifest(environment)
            manifests["pvc"] = self.create_persistent_volume_claim(environment)
            manifests["deployment"] = self.create_deployment_manifest(environment)
            manifests["service"] = self.create_service_manifest(environment)
            manifests["ingress"] = self.create_ingress_manifest(environment)
            manifests["hpa"] = self.create_hpa_manifest(environment)
            manifests["database"] = self.create_database_deployment(environment)
            
            if environment == KubernetesEnvironment.PRODUCTION:
                manifests["monitoring"] = self.create_monitoring_deployment(environment)
            
            # Apply manifests in order
            deployment_results = {}
            
            # Apply namespace first
            deployment_results["namespace"] = self._apply_manifest(manifests["namespace"])
            
            # Apply secrets and configmaps
            deployment_results["secret"] = self._apply_manifest(manifests["secret"])
            deployment_results["configmap"] = self._apply_manifest(manifests["configmap"])
            
            # Apply storage
            deployment_results["pvc"] = self._apply_manifest(manifests["pvc"])
            
            # Apply database
            deployment_results["database"] = self._apply_manifest(manifests["database"])
            
            # Wait for database to be ready
            self._wait_for_deployment(f"postgres-{environment.value}", timeout=300)
            
            # Apply main application
            deployment_results["deployment"] = self._apply_manifest(manifests["deployment"])
            deployment_results["service"] = self._apply_manifest(manifests["service"])
            
            # Wait for deployment to be ready
            self._wait_for_deployment(f"sutazai-agi-{environment.value}", timeout=600)
            
            # Apply ingress and autoscaling
            deployment_results["ingress"] = self._apply_manifest(manifests["ingress"])
            deployment_results["hpa"] = self._apply_manifest(manifests["hpa"])
            
            # Apply monitoring if production
            if environment == KubernetesEnvironment.PRODUCTION:
                deployment_results["monitoring"] = self._apply_manifest(manifests["monitoring"])
            
            logger.info(f"Successfully deployed {environment.value} environment")
            
            return {
                "status": "success",
                "environment": environment.value,
                "namespace": self.config.namespace,
                "manifests": manifests,
                "deployment_results": deployment_results,
                "services": self._get_service_urls(environment)
            }
            
        except Exception as e:
            logger.error(f"Kubernetes deployment failed: {e}")
            raise
    
    def _apply_manifest(self, manifest_path: str) -> Dict[str, Any]:
        """Apply a Kubernetes manifest file"""
        try:
            result = subprocess.run(
                ["kubectl", "apply", "-f", manifest_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise Exception(f"Failed to apply manifest {manifest_path}: {result.stderr}")
            
            return {
                "status": "success",
                "manifest": manifest_path,
                "output": result.stdout
            }
            
        except Exception as e:
            logger.error(f"Failed to apply manifest {manifest_path}: {e}")
            raise
    
    def _wait_for_deployment(self, deployment_name: str, timeout: int = 300):
        """Wait for a deployment to be ready"""
        logger.info(f"Waiting for deployment {deployment_name} to be ready...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                deployment = self.apps_v1.read_namespaced_deployment(
                    name=deployment_name,
                    namespace=self.config.namespace
                )
                
                if (deployment.status.ready_replicas and 
                    deployment.status.ready_replicas == deployment.spec.replicas):
                    logger.info(f"Deployment {deployment_name} is ready")
                    return
                
            except ApiException as e:
                if e.status != 404:
                    logger.error(f"Error checking deployment status: {e}")
            
            time.sleep(10)
        
        raise Exception(f"Deployment {deployment_name} not ready after {timeout} seconds")
    
    def _get_service_urls(self, environment: KubernetesEnvironment) -> Dict[str, str]:
        """Get service URLs for the deployed application"""
        try:
            urls = {}
            
            # Main application URL
            urls["main_app"] = f"http://{self.config.ingress_host}"
            
            # API documentation
            urls["api_docs"] = f"http://{self.config.ingress_host}/api/docs"
            
            # Health check
            urls["health"] = f"http://{self.config.ingress_host}/health"
            
            # Monitoring URLs (if production)
            if environment == KubernetesEnvironment.PRODUCTION:
                urls["prometheus"] = f"http://{self.config.ingress_host}:9090"
                urls["grafana"] = f"http://{self.config.ingress_host}:3000"
            
            return urls
            
        except Exception as e:
            logger.error(f"Failed to get service URLs: {e}")
            return {}
    
    def get_status(self, environment: KubernetesEnvironment) -> Dict[str, Any]:
        """Get deployment status"""
        try:
            # Get deployment status
            deployment = self.apps_v1.read_namespaced_deployment(
                name=f"sutazai-agi-{environment.value}",
                namespace=self.config.namespace
            )
            
            # Get service status
            service = self.core_v1.read_namespaced_service(
                name=f"sutazai-agi-service-{environment.value}",
                namespace=self.config.namespace
            )
            
            # Get pod status
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.config.namespace,
                label_selector=f"app=sutazai-agi,environment={environment.value}"
            )
            
            return {
                "deployment": {
                    "name": deployment.metadata.name,
                    "replicas": deployment.spec.replicas,
                    "ready_replicas": deployment.status.ready_replicas or 0,
                    "available_replicas": deployment.status.available_replicas or 0
                },
                "service": {
                    "name": service.metadata.name,
                    "cluster_ip": service.spec.cluster_ip,
                    "ports": [{"port": p.port, "target_port": p.target_port} for p in service.spec.ports]
                },
                "pods": [
                    {
                        "name": pod.metadata.name,
                        "status": pod.status.phase,
                        "ready": all(c.status for c in pod.status.container_statuses or [])
                    }
                    for pod in pods.items
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            raise
    
    def scale(self, environment: KubernetesEnvironment, replicas: int) -> Dict[str, Any]:
        """Scale the deployment"""
        try:
            # Update deployment replicas
            deployment = self.apps_v1.read_namespaced_deployment(
                name=f"sutazai-agi-{environment.value}",
                namespace=self.config.namespace
            )
            
            deployment.spec.replicas = replicas
            
            self.apps_v1.patch_namespaced_deployment(
                name=f"sutazai-agi-{environment.value}",
                namespace=self.config.namespace,
                body=deployment
            )
            
            logger.info(f"Scaled deployment to {replicas} replicas")
            
            return {
                "status": "success",
                "deployment": f"sutazai-agi-{environment.value}",
                "replicas": replicas
            }
            
        except Exception as e:
            logger.error(f"Failed to scale deployment: {e}")
            raise
    
    def cleanup(self, environment: KubernetesEnvironment) -> Dict[str, Any]:
        """Clean up deployment resources"""
        try:
            logger.info(f"Cleaning up {environment.value} deployment")
            
            # Delete all resources in the namespace
            result = subprocess.run(
                ["kubectl", "delete", "all", "--all", "-n", self.config.namespace],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.warning(f"Some resources may not have been deleted: {result.stderr}")
            
            return {
                "status": "success",
                "environment": environment.value,
                "message": "Deployment cleaned up successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to cleanup deployment: {e}")
            raise

def create_kubernetes_manager(config: KubernetesConfig = None) -> KubernetesDeploymentManager:
    """Create a new Kubernetes deployment manager"""
    return KubernetesDeploymentManager(config)

if __name__ == "__main__":
    # Example usage
    k8s_manager = create_kubernetes_manager()
    
    # Deploy production environment
    result = k8s_manager.deploy(KubernetesEnvironment.PRODUCTION)
    print(f"Deployment result: {json.dumps(result, indent=2)}")
    
    # Get deployment status
    status = k8s_manager.get_status(KubernetesEnvironment.PRODUCTION)
    print(f"Deployment status: {json.dumps(status, indent=2)}")