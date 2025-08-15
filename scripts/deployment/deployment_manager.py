#!/usr/bin/env python3
"""

logger = logging.getLogger(__name__)
Deployment Manager for SutazAI
==============================

Consolidated deployment and orchestration module that replaces 101+ individual deployment scripts.
Provides comprehensive deployment automation, service management, and orchestration capabilities.
"""

import os
import sys
import time
import json
import yaml
import subprocess
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from enum import Enum
import threading
import tempfile
import shutil

from ..utils.common_utils import (
    setup_logging, run_command, health_check_url, retry_operation,
    ensure_directory, load_config
)
from ..utils.docker_utils import DockerManager
from ..utils.network_utils import validate_sutazai_services, check_port_availability

logger = setup_logging('deployment_manager')

class DeploymentStatus(Enum):
    """Deployment status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    SUCCESS = "success"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"

@dataclass
class ServiceConfig:
    """Configuration for a single service"""
    name: str
    image: str
    ports: Dict[str, int]
    environment: Dict[str, str]
    volumes: List[str]
    dependencies: List[str]
    health_check: Optional[str] = None
    restart_policy: str = "unless-stopped"
    memory_limit: Optional[str] = None
    cpu_limit: Optional[float] = None
    user: Optional[str] = None
    networks: List[str] = None
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.networks is None:
            self.networks = ["sutazai-network"]
        if self.labels is None:
            self.labels = {}

@dataclass
class DeploymentConfig:
    """Configuration for deployment"""
    project_name: str = "sutazai"
    environment: str = "development"
    services: Dict[str, ServiceConfig] = None
    networks: Dict[str, Dict[str, Any]] = None
    volumes: Dict[str, Dict[str, Any]] = None
    deployment_order: List[List[str]] = None  # Tiered deployment
    health_check_timeout: int = 120
    health_check_interval: int = 5
    rollback_on_failure: bool = True
    parallel_deployment: bool = True
    max_parallel_services: int = 5
    
    def __post_init__(self):
        if self.services is None:
            self.services = {}
        if self.networks is None:
            self.networks = {
                "sutazai-network": {
                    "driver": "bridge",
                    "ipam": {
                        "config": [{"subnet": "172.20.0.0/16"}]
                    }
                }
            }
        if self.volumes is None:
            self.volumes = {}
        if self.deployment_order is None:
            self.deployment_order = [
                ["postgres", "redis", "neo4j"],  # Tier 1: Databases
                ["rabbitmq", "ollama"],          # Tier 2: Message queue & AI
                ["backend", "frontend"],         # Tier 3: Core services
                ["hardware-resource-optimizer", "ai-agent-orchestrator"]  # Tier 4: Agents
            ]

@dataclass
class DeploymentResult:
    """Result of a deployment operation"""
    status: DeploymentStatus
    services_deployed: List[str]
    services_failed: List[str]
    deployment_time: float
    error_messages: List[str]
    rollback_performed: bool = False
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class ServiceManager:
    """Manages individual service deployment and lifecycle"""
    
    def __init__(self, docker_manager: DockerManager):
        self.docker = docker_manager
        self.deployment_history: Dict[str, List[DeploymentResult]] = {}
    
    def deploy_service(
        self,
        service_name: str,
        service_config: ServiceConfig,
        compose_file: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Deploy a single service"""
        logger.info(f"Deploying service: {service_name}")
        
        try:
            if compose_file:
                # Use docker-compose for deployment
                return self._deploy_with_compose(service_name, compose_file)
            else:
                # Direct Docker deployment
                return self._deploy_with_docker(service_name, service_config)
                
        except Exception as e:
            error_msg = f"Failed to deploy {service_name}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def _deploy_with_compose(self, service_name: str, compose_file: str) -> Tuple[bool, str]:
        """Deploy service using docker-compose"""
        cmd = [
            "docker-compose", "-f", compose_file,
            "up", "-d", service_name
        ]
        
        returncode, stdout, stderr = run_command(cmd, timeout=120)
        
        if returncode == 0:
            logger.info(f"Successfully deployed {service_name} with compose")
            return True, "Deployment successful"
        else:
            error_msg = f"Compose deployment failed: {stderr}"
            logger.error(error_msg)
            return False, error_msg
    
    def _deploy_with_docker(self, service_name: str, config: ServiceConfig) -> Tuple[bool, str]:
        """Deploy service using direct Docker commands"""
        # Stop and remove existing container
        self.stop_service(service_name)
        self.remove_service(service_name)
        
        # Build Docker run command
        cmd = ["docker", "run", "-d", "--name", service_name]
        
        # Add restart policy
        cmd.extend(["--restart", config.restart_policy])
        
        # Add port mappings
        for internal_port, external_port in config.ports.items():
            cmd.extend(["-p", f"{external_port}:{internal_port}"])
        
        # Add environment variables
        for key, value in config.environment.items():
            cmd.extend(["-e", f"{key}={value}"])
        
        # Add volume mounts
        for volume in config.volumes:
            cmd.extend(["-v", volume])
        
        # Add networks
        for network in config.networks:
            cmd.extend(["--network", network])
        
        # Add resource limits
        if config.memory_limit:
            cmd.extend(["-m", config.memory_limit])
        if config.cpu_limit:
            cmd.extend(["--cpus", str(config.cpu_limit)])
        
        # Add user
        if config.user:
            cmd.extend(["--user", config.user])
        
        # Add labels
        for key, value in config.labels.items():
            cmd.extend(["--label", f"{key}={value}"])
        
        # Add image
        cmd.append(config.image)
        
        # Execute deployment
        returncode, stdout, stderr = run_command(cmd, timeout=60)
        
        if returncode == 0:
            container_id = stdout.strip()
            logger.info(f"Successfully deployed {service_name} (ID: {container_id[:12]})")
            return True, f"Container started: {container_id[:12]}"
        else:
            error_msg = f"Docker deployment failed: {stderr}"
            logger.error(error_msg)
            return False, error_msg
    
    def stop_service(self, service_name: str) -> bool:
        """Stop a service"""
        try:
            if self.docker.client:
                container = self.docker.client.containers.get(service_name)
                container.stop(timeout=10)
                logger.info(f"Stopped service: {service_name}")
                return True
        except Exception as e:
            logger.debug(f"Service {service_name} not running or doesn't exist: {e}")
        
        return False
    
    def remove_service(self, service_name: str) -> bool:
        """Remove a service container"""
        try:
            if self.docker.client:
                container = self.docker.client.containers.get(service_name)
                container.remove(force=True)
                logger.info(f"Removed service: {service_name}")
                return True
        except Exception as e:
            logger.debug(f"Service {service_name} doesn't exist: {e}")
        
        return False
    
    def health_check_service(
        self,
        service_name: str,
        health_check_url: Optional[str] = None,
        timeout: int = 120
    ) -> Tuple[bool, str]:
        """Perform health check on deployed service"""
        logger.info(f"Health checking service: {service_name}")
        
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            try:
                # Check if container is running
                if self.docker.client:
                    container = self.docker.client.containers.get(service_name)
                    if container.status != 'running':
                        logger.debug(f"Container {service_name} status: {container.status}")
                        time.sleep(2)
                        continue
                
                # Check health endpoint if provided
                if health_check_url:
                    from ..utils.common_utils import health_check_url as check_url
                    healthy, message = check_url(health_check_url, timeout=5)
                    if healthy:
                        logger.info(f"Service {service_name} is healthy")
                        return True, "Health check passed"
                    else:
                        logger.debug(f"Health check failed: {message}")
                else:
                    # Just check if container is running
                    logger.info(f"Service {service_name} is running")
                    return True, "Container is running"
                
            except Exception as e:
                logger.debug(f"Health check error for {service_name}: {e}")
            
            time.sleep(5)
        
        error_msg = f"Health check timeout for {service_name}"
        logger.error(error_msg)
        return False, error_msg
    
    def get_service_logs(self, service_name: str, lines: int = 50) -> str:
        """Get service logs"""
        try:
            if self.docker.client:
                container = self.docker.client.containers.get(service_name)
                logs = container.logs(tail=lines, timestamps=True).decode('utf-8')
                return logs
        except Exception as e:
            logger.error(f"Error getting logs for {service_name}: {e}")
        
        return ""

class OrchestrationEngine:
    """Orchestrates multi-service deployments with dependency management"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.docker = DockerManager()
        self.service_manager = ServiceManager(self.docker)
        self.deployment_state: Dict[str, Any] = {
            'current_deployment': None,
            'deployment_history': [],
            'failed_services': [],
            'successful_services': []
        }
    
    def deploy_all_services(
        self,
        compose_file: str = "docker-compose.yml"
    ) -> DeploymentResult:
        """Deploy all services according to configuration"""
        logger.info("Starting full system deployment")
        
        start_time = time.time()
        result = DeploymentResult(
            status=DeploymentStatus.IN_PROGRESS,
            services_deployed=[],
            services_failed=[],
            deployment_time=0,
            error_messages=[]
        )
        
        try:
            # Create networks first
            self._create_networks()
            
            # Deploy in tiers
            for tier_index, tier_services in enumerate(self.config.deployment_order):
                logger.info(f"Deploying tier {tier_index + 1}: {tier_services}")
                
                if self.config.parallel_deployment:
                    tier_result = self._deploy_tier_parallel(tier_services, compose_file)
                else:
                    tier_result = self._deploy_tier_sequential(tier_services, compose_file)
                
                result.services_deployed.extend(tier_result.services_deployed)
                result.services_failed.extend(tier_result.services_failed)
                result.error_messages.extend(tier_result.error_messages)
                
                # Stop if tier deployment failed and rollback is enabled
                if tier_result.services_failed and self.config.rollback_on_failure:
                    logger.warning(f"Tier {tier_index + 1} deployment failed, initiating rollback")
                    self._rollback_deployment(result.services_deployed)
                    result.status = DeploymentStatus.ROLLED_BACK
                    result.rollback_performed = True
                    break
            
            # Set final status
            if result.status == DeploymentStatus.IN_PROGRESS:
                if result.services_failed:
                    result.status = DeploymentStatus.FAILED
                else:
                    result.status = DeploymentStatus.SUCCESS
            
            result.deployment_time = time.time() - start_time
            
            # Wait for services to be healthy
            if result.status == DeploymentStatus.SUCCESS:
                logger.info("Waiting for services to become healthy...")
                self._wait_for_services_health(result.services_deployed)
            
        except Exception as e:
            error_msg = f"Deployment orchestration failed: {str(e)}"
            logger.error(error_msg)
            result.status = DeploymentStatus.FAILED
            result.error_messages.append(error_msg)
            result.deployment_time = time.time() - start_time
        
        # Record deployment
        self.deployment_state['deployment_history'].append(result)
        self.deployment_state['current_deployment'] = result
        
        logger.info(f"Deployment completed: {result.status.value} "
                   f"({len(result.services_deployed)} successful, "
                   f"{len(result.services_failed)} failed)")
        
        return result
    
    def _create_networks(self) -> None:
        """Create required networks"""
        for network_name, network_config in self.config.networks.items():
            try:
                if self.docker.client:
                    # Check if network exists
                    try:
                        self.docker.client.networks.get(network_name)
                        logger.debug(f"Network {network_name} already exists")
                        continue
                    except Exception as e:
                        # Suppressed exception (was bare except)
                        logger.debug(f"Suppressed exception: {e}")
                        pass
                    
                    # Create network
                    self.docker.client.networks.create(
                        network_name,
                        driver=network_config.get('driver', 'bridge'),
                        ipam=network_config.get('ipam')
                    )
                    logger.info(f"Created network: {network_name}")
                    
            except Exception as e:
                logger.warning(f"Error creating network {network_name}: {e}")
    
    def _deploy_tier_parallel(
        self,
        services: List[str],
        compose_file: str
    ) -> DeploymentResult:
        """Deploy services in a tier in parallel"""
        result = DeploymentResult(
            status=DeploymentStatus.IN_PROGRESS,
            services_deployed=[],
            services_failed=[],
            deployment_time=0,
            error_messages=[]
        )
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_services) as executor:
            future_to_service = {
                executor.submit(self._deploy_single_service, service, compose_file): service
                for service in services
            }
            
            for future in as_completed(future_to_service):
                service_name = future_to_service[future]
                try:
                    success, message = future.result()
                    if success:
                        result.services_deployed.append(service_name)
                        logger.info(f"✓ {service_name}: {message}")
                    else:
                        result.services_failed.append(service_name)
                        result.error_messages.append(f"{service_name}: {message}")
                        logger.error(f"✗ {service_name}: {message}")
                        
                except Exception as e:
                    result.services_failed.append(service_name)
                    error_msg = f"{service_name}: {str(e)}"
                    result.error_messages.append(error_msg)
                    logger.error(f"✗ {error_msg}")
        
        result.deployment_time = time.time() - start_time
        return result
    
    def _deploy_tier_sequential(
        self,
        services: List[str],
        compose_file: str
    ) -> DeploymentResult:
        """Deploy services in a tier sequentially"""
        result = DeploymentResult(
            status=DeploymentStatus.IN_PROGRESS,
            services_deployed=[],
            services_failed=[],
            deployment_time=0,
            error_messages=[]
        )
        
        start_time = time.time()
        
        for service_name in services:
            try:
                success, message = self._deploy_single_service(service_name, compose_file)
                if success:
                    result.services_deployed.append(service_name)
                    logger.info(f"✓ {service_name}: {message}")
                else:
                    result.services_failed.append(service_name)
                    result.error_messages.append(f"{service_name}: {message}")
                    logger.error(f"✗ {service_name}: {message}")
                    
                    # Stop deployment on failure if not parallel
                    break
                    
            except Exception as e:
                result.services_failed.append(service_name)
                error_msg = f"{service_name}: {str(e)}"
                result.error_messages.append(error_msg)
                logger.error(f"✗ {error_msg}")
                break
        
        result.deployment_time = time.time() - start_time
        return result
    
    def _deploy_single_service(self, service_name: str, compose_file: str) -> Tuple[bool, str]:
        """Deploy a single service with retry logic"""
        def deploy_operation():
            return self.service_manager.deploy_service(
                service_name,
                self.config.services.get(service_name),
                compose_file
            )
        
        try:
            return retry_operation(
                deploy_operation,
                max_attempts=2,
                delay=5.0,
                exceptions=(Exception,)
            )
        except Exception as e:
            return False, str(e)
    
    def _wait_for_services_health(self, services: List[str]) -> None:
        """Wait for all services to become healthy"""
        logger.info("Performing health checks on deployed services...")
        
        # Health check URLs for services that support HTTP health checks
        health_endpoints = {
            'backend': 'http://localhost:10010/health',
            'frontend': 'http://localhost:10011',
            'ollama': 'http://localhost:10104/api/tags',
            'grafana': 'http://localhost:10201/api/health',
            'prometheus': 'http://localhost:10200/-/healthy',
            'hardware-resource-optimizer': 'http://localhost:11110/health',
            'ai-agent-orchestrator': 'http://localhost:8589/health'
        }
        
        for service in services:
            health_url = health_endpoints.get(service)
            success, message = self.service_manager.health_check_service(
                service,
                health_url,
                timeout=self.config.health_check_timeout
            )
            
            if success:
                logger.info(f"✓ {service} health check passed")
            else:
                logger.warning(f"⚠ {service} health check failed: {message}")
    
    def _rollback_deployment(self, services: List[str]) -> None:
        """Rollback deployment by stopping services"""
        logger.info("Rolling back deployment...")
        
        for service in reversed(services):  # Stop in reverse order
            try:
                self.service_manager.stop_service(service)
                logger.info(f"Rolled back service: {service}")
            except Exception as e:
                logger.error(f"Error rolling back {service}: {e}")
    
    def stop_all_services(self) -> DeploymentResult:
        """Stop all running services"""
        logger.info("Stopping all services")
        
        start_time = time.time()
        result = DeploymentResult(
            status=DeploymentStatus.IN_PROGRESS,
            services_deployed=[],  # Will contain stopped services
            services_failed=[],
            deployment_time=0,
            error_messages=[]
        )
        
        # Get all running services
        running_services = []
        try:
            containers = self.docker.list_containers()
            running_services = [c.name for c in containers if c.status == 'running']
        except Exception as e:
            logger.error(f"Error listing containers: {e}")
        
        # Stop services in reverse deployment order
        all_services = [s for tier in self.config.deployment_order for s in tier]
        services_to_stop = [s for s in reversed(all_services) if s in running_services]
        
        for service in services_to_stop:
            try:
                success = self.service_manager.stop_service(service)
                if success:
                    result.services_deployed.append(service)
                else:
                    result.services_failed.append(service)
            except Exception as e:
                result.services_failed.append(service)
                result.error_messages.append(f"{service}: {str(e)}")
        
        result.deployment_time = time.time() - start_time
        result.status = DeploymentStatus.SUCCESS if not result.services_failed else DeploymentStatus.FAILED
        
        logger.info(f"Stopped {len(result.services_deployed)} services")
        return result
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        return {
            'current_deployment': asdict(self.deployment_state['current_deployment']) if self.deployment_state['current_deployment'] else None,
            'deployment_history': [asdict(d) for d in self.deployment_state['deployment_history'][-10:]],  # Last 10
            'total_deployments': len(self.deployment_state['deployment_history']),
            'timestamp': datetime.now().isoformat()
        }

class DeploymentManager:
    """Main deployment manager class"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_config()
        self.orchestrator = OrchestrationEngine(self.config)
        self.project_root = Path(__file__).parent.parent.parent
        
    def _load_config(self) -> DeploymentConfig:
        """Load deployment configuration"""
        if self.config_path and Path(self.config_path).exists():
            try:
                config_data = load_config(self.config_path)
                # Convert to DeploymentConfig object
                # This is a simplified version - in practice you'd want proper deserialization
                return DeploymentConfig(**config_data)
            except Exception as e:
                logger.warning(f"Error loading config {self.config_path}: {e}")
        
        # Return default configuration
        return DeploymentConfig()
    
    def deploy(
        self,
        services: Optional[List[str]] = None,
        compose_file: str = "docker-compose.yml"
    ) -> DeploymentResult:
        """Deploy services"""
        if services:
            logger.info(f"Deploying specific services: {services}")
            # Filter config to only include specified services
            filtered_config = DeploymentConfig()
            filtered_config.deployment_order = [
                [s for s in tier if s in services]
                for tier in self.config.deployment_order
            ]
            # Remove empty tiers
            filtered_config.deployment_order = [
                tier for tier in filtered_config.deployment_order if tier
            ]
            
            orchestrator = OrchestrationEngine(filtered_config)
            return orchestrator.deploy_all_services(compose_file)
        else:
            return self.orchestrator.deploy_all_services(compose_file)
    
    def stop(self, services: Optional[List[str]] = None) -> DeploymentResult:
        """Stop services"""
        if services:
            # Stop specific services
            result = DeploymentResult(
                status=DeploymentStatus.SUCCESS,
                services_deployed=[],
                services_failed=[],
                deployment_time=0,
                error_messages=[]
            )
            
            for service in services:
                success = self.orchestrator.service_manager.stop_service(service)
                if success:
                    result.services_deployed.append(service)
                else:
                    result.services_failed.append(service)
            
            return result
        else:
            return self.orchestrator.stop_all_services()
    
    def restart(self, services: Optional[List[str]] = None) -> DeploymentResult:
        """Restart services"""
        # Stop then deploy
        stop_result = self.stop(services)
        
        if stop_result.status == DeploymentStatus.SUCCESS:
            time.sleep(2)  # Brief pause
            return self.deploy(services)
        else:
            return stop_result
    
    def status(self) -> Dict[str, Any]:
        """Get deployment status"""
        return self.orchestrator.get_deployment_status()
    
    def logs(self, service: str, lines: int = 50) -> str:
        """Get service logs"""
        return self.orchestrator.service_manager.get_service_logs(service, lines)
    
    def health_check(self, services: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform health check on services"""
        if services is None:
            services = [s for tier in self.config.deployment_order for s in tier]
        
        results = {}
        
        for service in services:
            # Determine health check URL
            health_endpoints = {
                'backend': 'http://localhost:10010/health',
                'frontend': 'http://localhost:10011',
                'ollama': 'http://localhost:10104/api/tags',
                'grafana': 'http://localhost:10201/api/health',
                'prometheus': 'http://localhost:10200/-/healthy',
                'hardware-resource-optimizer': 'http://localhost:11110/health',
                'ai-agent-orchestrator': 'http://localhost:8589/health'
            }
            
            health_url = health_endpoints.get(service)
            success, message = self.orchestrator.service_manager.health_check_service(
                service, health_url, timeout=30
            )
            
            results[service] = {
                'healthy': success,
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
        
        return results

# CLI interface functions
def create_deployment_manager(config_path: Optional[str] = None) -> DeploymentManager:
    """Create deployment manager with configuration"""
    return DeploymentManager(config_path)

def quick_deploy(services: List[str], compose_file: str = "docker-compose.yml") -> bool:
    """Quick deployment of specific services"""
    manager = DeploymentManager()
    result = manager.deploy(services, compose_file)
    return result.status == DeploymentStatus.SUCCESS

def deploy_ _stack() -> bool:
    """Deploy   SutazAI stack"""
    services = ["postgres", "redis", "ollama", "backend", "frontend"]
    return quick_deploy(services)

def deploy_full_stack() -> bool:
    """Deploy full SutazAI stack"""
    manager = DeploymentManager()
    result = manager.deploy()
    return result.status == DeploymentStatus.SUCCESS

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SutazAI Deployment Manager')
    parser.add_argument('action', choices=['deploy', 'stop', 'restart', 'status', 'health', 'logs'])
    parser.add_argument('--services', nargs='+', help='Specific services to target')
    parser.add_argument('--compose-file', default='docker-compose.yml', help='Docker compose file')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--lines', type=int, default=50, help='Number of log lines (for logs action)')
    
    args = parser.parse_args()
    
    # Create deployment manager
    manager = DeploymentManager(args.config)
    
    if args.action == 'deploy':
        result = manager.deploy(args.services, args.compose_file)
        logger.info(f"Deployment result: {result.status.value}")
        if result.error_messages:
            logger.error("Errors:")
            for error in result.error_messages:
                logger.error(f"  - {error}")
    
    elif args.action == 'stop':
        result = manager.stop(args.services)
        logger.info(f"Stop result: {result.status.value}")
    
    elif args.action == 'restart':
        result = manager.restart(args.services)
        logger.info(f"Restart result: {result.status.value}")
    
    elif args.action == 'status':
        status = manager.status()
        logger.info(json.dumps(status, indent=2))
    
    elif args.action == 'health':
        health = manager.health_check(args.services)
        logger.info(json.dumps(health, indent=2))
    
    elif args.action == 'logs':
        if not args.services or len(args.services) != 1:
            logger.error("Error: logs action requires exactly one service")
            sys.exit(1)
        
        logs = manager.logs(args.services[0], args.lines)
        logger.info(logs)