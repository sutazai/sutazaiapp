"""
Edge Deployment Tools - Automated deployment and management for edge inference systems
"""

import asyncio
import os
import subprocess
import logging
import json
import yaml
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import aiofiles
import aiohttp

logger = logging.getLogger(__name__)

class DeploymentStatus(Enum):
    """Deployment status"""
    PENDING = "pending"
    PREPARING = "preparing"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"

class EdgePlatform(Enum):
    """Edge platforms"""
    KUBERNETES = "kubernetes"
    DOCKER_SWARM = "docker_swarm"
    NOMAD = "nomad"
    BARE_METAL = "bare_metal"
    EDGE_COMPUTING = "edge_computing"

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    name: str
    platform: EdgePlatform
    resources: Dict[str, Any]
    environment: Dict[str, str] = field(default_factory=dict)
    replicas: int = 1
    health_check: Dict[str, Any] = field(default_factory=dict)
    networking: Dict[str, Any] = field(default_factory=dict)
    storage: Dict[str, Any] = field(default_factory=dict)
    security: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeploymentJob:
    """Deployment job tracking"""
    job_id: str
    config: DeploymentConfig
    status: DeploymentStatus
    created_at: datetime
    updated_at: datetime
    logs: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    deployed_endpoints: List[str] = field(default_factory=list)

class KubernetesDeployer:
    """Kubernetes deployment manager"""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        self.kubeconfig_path = kubeconfig_path or os.path.expanduser("~/.kube/config")
    
    async def deploy(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy to Kubernetes"""
        logger.info(f"Deploying {config.name} to Kubernetes")
        
        # Generate Kubernetes manifests
        manifests = self._generate_k8s_manifests(config)
        
        # Create temporary directory for manifests
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "manifests.yaml"
            
            async with aiofiles.open(manifest_path, 'w') as f:
                await f.write(yaml.dump_all(manifests))
            
            # Apply manifests
            result = await self._run_kubectl(["apply", "-f", str(manifest_path)])
            
            if result["success"]:
                # Get service endpoints
                endpoints = await self._get_service_endpoints(config.name)
                return {
                    "success": True,
                    "endpoints": endpoints,
                    "message": "Deployment successful"
                }
            else:
                return {
                    "success": False,
                    "error": result["error"],
                    "message": "Deployment failed"
                }
    
    def _generate_k8s_manifests(self, config: DeploymentConfig) -> List[Dict[str, Any]]:
        """Generate Kubernetes manifests"""
        manifests = []
        
        # Deployment manifest
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": config.name,
                "labels": {"app": config.name}
            },
            "spec": {
                "replicas": config.replicas,
                "selector": {"matchLabels": {"app": config.name}},
                "template": {
                    "metadata": {"labels": {"app": config.name}},
                    "spec": {
                        "containers": [{
                            "name": config.name,
                            "image": config.resources.get("image", "edge-inference:latest"),
                            "ports": [{"containerPort": 8000}],
                            "env": [{"name": k, "value": str(v)} for k, v in config.environment.items()],
                            "resources": {
                                "requests": {
                                    "cpu": config.resources.get("cpu_request", "100m"),
                                    "memory": config.resources.get("memory_request", "128Mi")
                                },
                                "limits": {
                                    "cpu": config.resources.get("cpu_limit", "500m"),
                                    "memory": config.resources.get("memory_limit", "512Mi")
                                }
                            }
                        }]
                    }
                }
            }
        }
        
        # Add health checks if configured
        if config.health_check:
            health_config = config.health_check
            container = deployment["spec"]["template"]["spec"]["containers"][0]
            if health_config.get("http_path"):
                container["livenessProbe"] = {
                    "httpGet": {
                        "path": health_config["http_path"],
                        "port": health_config.get("port", 8000)
                    },
                    "initialDelaySeconds": health_config.get("initial_delay", 30),
                    "periodSeconds": health_config.get("period", 10)
                }
                container["readinessProbe"] = container["livenessProbe"].copy()
        
        manifests.append(deployment)
        
        # Service manifest
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{config.name}-service",
                "labels": {"app": config.name}
            },
            "spec": {
                "selector": {"app": config.name},
                "ports": [{
                    "port": 80,
                    "targetPort": 8000,
                    "protocol": "TCP"
                }],
                "type": config.networking.get("service_type", "ClusterIP")
            }
        }
        manifests.append(service)
        
        # Ingress if external access needed
        if config.networking.get("external_access"):
            ingress = {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "Ingress",
                "metadata": {
                    "name": f"{config.name}-ingress",
                    "annotations": {
                        "nginx.ingress.kubernetes.io/rewrite-target": "/"
                    }
                },
                "spec": {
                    "rules": [{
                        "host": config.networking.get("hostname", f"{config.name}.local"),
                        "http": {
                            "paths": [{
                                "path": "/",
                                "pathType": "Prefix",
                                "backend": {
                                    "service": {
                                        "name": f"{config.name}-service",
                                        "port": {"number": 80}
                                    }
                                }
                            }]
                        }
                    }]
                }
            }
            manifests.append(ingress)
        
        return manifests
    
    async def _run_kubectl(self, args: List[str]) -> Dict[str, Any]:
        """Run kubectl command"""
        cmd = ["kubectl"] + args
        if self.kubeconfig_path:
            cmd.extend(["--kubeconfig", self.kubeconfig_path])
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "error": stderr.decode() if process.returncode != 0 else None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_service_endpoints(self, service_name: str) -> List[str]:
        """Get service endpoints"""
        result = await self._run_kubectl([
            "get", "service", f"{service_name}-service", 
            "-o", "jsonpath='{.status.loadBalancer.ingress[*].ip}'"
        ])
        
        if result["success"] and result["stdout"]:
            ips = result["stdout"].strip("'").split()
            return [f"http://{ip}" for ip in ips if ip]
        
        return []
    
    async def undeploy(self, deployment_name: str) -> Dict[str, Any]:
        """Remove deployment"""
        result = await self._run_kubectl([
            "delete", "deployment,service,ingress", 
            "-l", f"app={deployment_name}"
        ])
        
        return {
            "success": result["success"],
            "message": "Undeployment successful" if result["success"] else "Undeployment failed",
            "error": result.get("error")
        }

class DockerDeployer:
    """Docker deployment manager"""
    
    def __init__(self):
        pass
    
    async def deploy(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy using Docker"""
        logger.info(f"Deploying {config.name} with Docker")
        
        try:
            # Generate docker-compose file
            compose_config = self._generate_docker_compose(config)
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                compose_path = Path(temp_dir) / "docker-compose.yml"
                
                async with aiofiles.open(compose_path, 'w') as f:
                    await f.write(yaml.dump(compose_config))
                
                # Deploy with docker-compose
                result = await self._run_docker_compose(str(compose_path), ["up", "-d"])
                
                if result["success"]:
                    endpoints = await self._get_container_endpoints(config.name)
                    return {
                        "success": True,
                        "endpoints": endpoints,
                        "message": "Docker deployment successful"
                    }
                else:
                    return {
                        "success": False,
                        "error": result["error"],
                        "message": "Docker deployment failed"
                    }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Docker deployment failed"
            }
    
    def _generate_docker_compose(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Generate docker-compose configuration"""
        service_config = {
            "image": config.resources.get("image", "edge-inference:latest"),
            "ports": [f"{config.networking.get('host_port', 8000)}:8000"],
            "environment": config.environment,
            "restart": "unless-stopped"
        }
        
        # Add resource limits if specified
        if config.resources.get("memory_limit"):
            service_config["mem_limit"] = config.resources["memory_limit"]
        
        if config.resources.get("cpu_limit"):
            service_config["cpus"] = config.resources["cpu_limit"]
        
        # Add health check
        if config.health_check.get("http_path"):
            service_config["healthcheck"] = {
                "test": f"curl -f http://localhost:8000{config.health_check['http_path']} || exit 1",
                "interval": f"{config.health_check.get('period', 30)}s",
                "timeout": "10s",
                "retries": 3
            }
        
        # Add volumes if specified
        if config.storage.get("volumes"):
            service_config["volumes"] = config.storage["volumes"]
        
        return {
            "version": "3.8",
            "services": {
                config.name: service_config
            }
        }
    
    async def _run_docker_compose(self, compose_file: str, args: List[str]) -> Dict[str, Any]:
        """Run docker-compose command"""
        cmd = ["docker-compose", "-f", compose_file] + args
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "error": stderr.decode() if process.returncode != 0 else None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_container_endpoints(self, container_name: str) -> List[str]:
        """Get container endpoints"""
        try:
            result = await self._run_docker_command([
                "inspect", container_name,
                "--format", "{{range .NetworkSettings.Ports}}{{.}}{{end}}"
            ])
            
            if result["success"]:
                # Parse port mappings and construct endpoints
                return [f"http://localhost:8000"]  # Simplified
            
            return []
        except Exception as e:
            logger.warning(f"Exception caught, returning: {e}")
            return []
    
    async def _run_docker_command(self, args: List[str]) -> Dict[str, Any]:
        """Run docker command"""
        cmd = ["docker"] + args
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode(),
                "stderr": stderr.decode()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

class BareMetalDeployer:
    """Bare metal deployment manager"""
    
    def __init__(self):
        pass
    
    async def deploy(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy on bare metal"""
        logger.info(f"Deploying {config.name} on bare metal")
        
        try:
            # Create deployment directory
            deploy_dir = Path(f"/opt/edge_inference/{config.name}")
            deploy_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate systemd service file
            service_config = self._generate_systemd_service(config, deploy_dir)
            service_file = f"/etc/systemd/system/{config.name}.service"
            
            async with aiofiles.open(service_file, 'w') as f:
                await f.write(service_config)
            
            # Generate startup script
            startup_script = self._generate_startup_script(config)
            script_path = deploy_dir / "start.sh"
            
            async with aiofiles.open(script_path, 'w') as f:
                await f.write(startup_script)
            
            # Make script executable
            os.chmod(script_path, 0o755)
            
            # Enable and start service
            await self._run_systemctl(["daemon-reload"])
            await self._run_systemctl(["enable", config.name])
            result = await self._run_systemctl(["start", config.name])
            
            if result["success"]:
                return {
                    "success": True,
                    "endpoints": [f"http://localhost:{config.networking.get('port', 8000)}"],
                    "message": "Bare metal deployment successful"
                }
            else:
                return {
                    "success": False,
                    "error": result["error"],
                    "message": "Bare metal deployment failed"
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Bare metal deployment failed"
            }
    
    def _generate_systemd_service(self, config: DeploymentConfig, deploy_dir: Path) -> str:
        """Generate systemd service file"""
        return f"""[Unit]
Description=Edge Inference Service - {config.name}
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory={deploy_dir}
ExecStart={deploy_dir}/start.sh
Restart=always
RestartSec=10
Environment=PYTHONPATH={deploy_dir}

[Install]
WantedBy=multi-user.target
"""
    
    def _generate_startup_script(self, config: DeploymentConfig) -> str:
        """Generate startup script"""
        env_vars = "\n".join([f"export {k}={v}" for k, v in config.environment.items()])
        
        return f"""#!/bin/bash
set -e

# Set environment variables
{env_vars}

# Start the edge inference service
cd /opt/sutazaiapp/backend
python -m edge_inference.proxy --port {config.networking.get('port', 8000)}
"""
    
    async def _run_systemctl(self, args: List[str]) -> Dict[str, Any]:
        """Run systemctl command"""
        cmd = ["systemctl"] + args
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "error": stderr.decode() if process.returncode != 0 else None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

class EdgeDeploymentManager:
    """Main deployment manager for edge inference systems"""
    
    def __init__(self):
        self.deployers = {
            EdgePlatform.KUBERNETES: KubernetesDeployer(),
            EdgePlatform.DOCKER_SWARM: DockerDeployer(),
            EdgePlatform.BARE_METAL: BareMetalDeployer()
        }
        
        self.jobs: Dict[str, DeploymentJob] = {}
        self._job_counter = 0
        self._lock = asyncio.Lock()
    
    async def deploy(self, config: DeploymentConfig) -> str:
        """Deploy edge inference system"""
        async with self._lock:
            # Create deployment job
            job_id = f"deploy_{self._job_counter}_{int(asyncio.get_event_loop().time())}"
            self._job_counter += 1
            
            job = DeploymentJob(
                job_id=job_id,
                config=config,
                status=DeploymentStatus.PENDING,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.jobs[job_id] = job
        
        # Start deployment task
        asyncio.create_task(self._execute_deployment(job_id))
        
        return job_id
    
    async def _execute_deployment(self, job_id: str) -> None:
        """Execute deployment job"""
        job = self.jobs[job_id]
        
        try:
            # Update status
            job.status = DeploymentStatus.PREPARING
            job.updated_at = datetime.now()
            job.logs.append(f"Starting deployment for {job.config.name}")
            
            # Get appropriate deployer
            deployer = self.deployers.get(job.config.platform)
            if not deployer:
                raise ValueError(f"Unsupported platform: {job.config.platform}")
            
            # Execute deployment
            job.status = DeploymentStatus.DEPLOYING
            job.updated_at = datetime.now()
            job.logs.append(f"Deploying to {job.config.platform.value}")
            
            result = await deployer.deploy(job.config)
            
            if result["success"]:
                job.status = DeploymentStatus.DEPLOYED
                job.deployed_endpoints = result.get("endpoints", [])
                job.logs.append("Deployment successful")
            else:
                job.status = DeploymentStatus.FAILED
                job.error_message = result.get("error", "Unknown error")
                job.logs.append(f"Deployment failed: {job.error_message}")
            
        except Exception as e:
            job.status = DeploymentStatus.FAILED
            job.error_message = str(e)
            job.logs.append(f"Deployment failed with exception: {e}")
        
        finally:
            job.updated_at = datetime.now()
    
    async def get_deployment_status(self, job_id: str) -> Optional[DeploymentJob]:
        """Get deployment job status"""
        return self.jobs.get(job_id)
    
    async def list_deployments(self) -> List[DeploymentJob]:
        """List all deployment jobs"""
        return list(self.jobs.values())
    
    async def create_edge_cluster_config(self,
                                       cluster_name: str,
                                       nodes: List[Dict[str, Any]],
                                       models: List[str]) -> DeploymentConfig:
        """Create deployment config for edge cluster"""
        
        # Calculate resource requirements based on models and nodes
        total_cpu = sum(node.get("cpu_cores", 2) for node in nodes)
        total_memory = sum(node.get("memory_gb", 4) for node in nodes)
        
        # Estimate resource needs per model
        cpu_per_model = max(0.5, total_cpu / len(models)) if models else 1.0
        memory_per_model = max(1.0, total_memory / len(models)) if models else 2.0
        
        return DeploymentConfig(
            name=cluster_name,
            platform=EdgePlatform.KUBERNETES,
            replicas=len(nodes),
            resources={
                "image": "edge-inference:latest",
                "cpu_request": f"{cpu_per_model * 0.5}",
                "cpu_limit": f"{cpu_per_model}",
                "memory_request": f"{int(memory_per_model * 0.5 * 1024)}Mi",
                "memory_limit": f"{int(memory_per_model * 1024)}Mi"
            },
            environment={
                "MODELS": ",".join(models),
                "NODE_COUNT": str(len(nodes)),
                "CLUSTER_NAME": cluster_name
            },
            health_check={
                "http_path": "/health",
                "port": 8000,
                "initial_delay": 30,
                "period": 10
            },
            networking={
                "external_access": True,
                "hostname": f"{cluster_name}.edge.local",
                "service_type": "LoadBalancer"
            }
        )
    
    async def generate_deployment_templates(self) -> Dict[str, str]:
        """Generate deployment templates for different scenarios"""
        templates = {}
        
        # Single node template
        single_node_config = DeploymentConfig(
            name="edge-inference-single",
            platform=EdgePlatform.DOCKER_SWARM,
            resources={
                "image": "edge-inference:latest",
                "cpu_limit": "2",
                "memory_limit": "4Gi"
            },
            environment={
                "MODEL_CACHE_SIZE": "2GB",
                "MAX_CONCURRENT_REQUESTS": "10"
            },
            health_check={
                "http_path": "/health",
                "period": 30
            },
            networking={
                "host_port": 8000
            }
        )
        templates["single_node"] = yaml.dump(single_node_config.__dict__)
        
        # High availability template
        ha_config = DeploymentConfig(
            name="edge-inference-ha",
            platform=EdgePlatform.KUBERNETES,
            replicas=3,
            resources={
                "image": "edge-inference:latest",
                "cpu_request": "500m",
                "cpu_limit": "1",
                "memory_request": "1Gi",
                "memory_limit": "2Gi"
            },
            environment={
                "ENABLE_CLUSTERING": "true",
                "ENABLE_FAILOVER": "true"
            },
            health_check={
                "http_path": "/health",
                "initial_delay": 30,
                "period": 10
            },
            networking={
                "external_access": True,
                "service_type": "LoadBalancer"
            }
        )
        templates["high_availability"] = yaml.dump(ha_config.__dict__)
        
        # Edge IoT template
        iot_config = DeploymentConfig(
            name="edge-inference-iot",
            platform=EdgePlatform.BARE_METAL,
            resources={
                "cpu_limit": "1",
                "memory_limit": "1Gi"
            },
            environment={
                "LIGHTWEIGHT_MODE": "true",
                "MODEL_QUANTIZATION": "int8"
            },
            networking={
                "port": 8000
            }
        )
        templates["iot_edge"] = yaml.dump(iot_config.__dict__)
        
        return templates
    
    async def validate_deployment_requirements(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Validate deployment requirements"""
        issues = []
        warnings = []
        
        # Check platform availability
        if config.platform not in self.deployers:
            issues.append(f"Platform {config.platform.value} is not supported")
        
        # Check resource requirements
        if config.resources.get("memory_limit"):
            memory_str = config.resources["memory_limit"]
            if "Gi" in memory_str:
                memory_gb = float(memory_str.replace("Gi", ""))
                if memory_gb > 8:
                    warnings.append("High memory requirement may not be suitable for edge devices")
        
        # Check networking configuration
        if config.networking.get("external_access") and not config.networking.get("hostname"):
            warnings.append("External access enabled but no hostname specified")
        
        # Check health check configuration
        if not config.health_check:
            warnings.append("No health check configured - recommended for production")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }

# Global deployment manager instance
_global_deployment_manager: Optional[EdgeDeploymentManager] = None

def get_global_deployment_manager() -> EdgeDeploymentManager:
    """Get or create global deployment manager instance"""
    global _global_deployment_manager
    if _global_deployment_manager is None:
        _global_deployment_manager = EdgeDeploymentManager()
    return _global_deployment_manager