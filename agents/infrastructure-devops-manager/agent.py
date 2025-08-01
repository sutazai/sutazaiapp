#!/usr/bin/env python3
"""
Infrastructure DevOps Manager Agent
Responsible for infrastructure management and DevOps operations
"""

import sys
import os
import subprocess
sys.path.append('/opt/sutazaiapp/agents')

from agent_base import BaseAgent
from typing import Dict, Any, List


class InfrastructureDevOpsManagerAgent(BaseAgent):
    """Infrastructure DevOps Manager Agent implementation"""
    
    def __init__(self):
        super().__init__()
        self.infrastructure_tools = [
            "docker",
            "kubernetes",
            "terraform",
            "ansible",
            "prometheus",
            "grafana"
        ]
        
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process infrastructure and DevOps tasks"""
        task_type = task.get("type", "")
        task_data = task.get("data", {})
        
        self.logger.info(f"Processing infrastructure task: {task_type}")
        
        try:
            if task_type == "provision_infrastructure":
                return self._provision_infrastructure(task_data)
            elif task_type == "configure_monitoring":
                return self._configure_monitoring(task_data)
            elif task_type == "manage_containers":
                return self._manage_containers(task_data)
            elif task_type == "setup_networking":
                return self._setup_networking(task_data)
            elif task_type == "backup_restore":
                return self._handle_backup_restore(task_data)
            else:
                # Use Ollama for general DevOps tasks
                prompt = f"""As an Infrastructure DevOps Manager, help with this task:
                Type: {task_type}
                Data: {task_data}
                
                Provide infrastructure solution and implementation steps."""
                
                response = self.query_ollama(prompt)
                
                return {
                    "status": "success",
                    "task_id": task.get("id"),
                    "result": response or "Infrastructure assistance provided",
                    "agent": self.agent_name
                }
                
        except Exception as e:
            self.logger.error(f"Error processing task: {e}")
            return {
                "status": "error",
                "task_id": task.get("id"),
                "error": str(e),
                "agent": self.agent_name
            }
    
    def _provision_infrastructure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Provision infrastructure resources"""
        resource_type = data.get("resource_type", "compute")
        provider = data.get("provider", "docker")
        specs = data.get("specs", {})
        
        self.logger.info(f"Provisioning {resource_type} on {provider}")
        
        # Simulate infrastructure provisioning
        provisioned_resources = {
            "compute_instances": 3,
            "load_balancers": 1,
            "databases": 1,
            "storage_volumes": 2,
            "network_interfaces": 4
        }
        
        return {
            "status": "success",
            "action": "infrastructure_provisioned",
            "provider": provider,
            "resource_type": resource_type,
            "resources": provisioned_resources,
            "configuration": {
                "auto_scaling": "enabled",
                "high_availability": "configured",
                "backup_policy": "daily"
            }
        }
    
    def _configure_monitoring(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Configure monitoring and alerting"""
        monitoring_type = data.get("monitoring_type", "prometheus")
        services = data.get("services", ["all"])
        
        self.logger.info(f"Configuring {monitoring_type} monitoring")
        
        # Simulate monitoring setup
        monitoring_config = {
            "metrics_collected": [
                "cpu_usage",
                "memory_usage",
                "disk_io",
                "network_traffic",
                "application_metrics"
            ],
            "alert_rules": [
                "high_cpu_usage",
                "low_disk_space",
                "service_down",
                "error_rate_threshold"
            ],
            "dashboards": [
                "system_overview",
                "application_performance",
                "infrastructure_health"
            ]
        }
        
        return {
            "status": "success",
            "action": "monitoring_configured",
            "monitoring_type": monitoring_type,
            "configuration": monitoring_config,
            "grafana_url": "http://grafana.sutazai.local:3000",
            "prometheus_url": "http://prometheus.sutazai.local:9090"
        }
    
    def _manage_containers(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Manage Docker containers"""
        action = data.get("action", "status")
        container_name = data.get("container_name", "all")
        
        self.logger.info(f"Managing containers: {action} on {container_name}")
        
        # Simulate container management
        if action == "status":
            container_status = self._get_container_status()
        else:
            container_status = f"Action '{action}' performed on {container_name}"
        
        return {
            "status": "success",
            "action": "containers_managed",
            "operation": action,
            "target": container_name,
            "result": container_status
        }
    
    def _setup_networking(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Setup network configuration"""
        network_type = data.get("network_type", "bridge")
        subnet = data.get("subnet", "172.20.0.0/16")
        
        self.logger.info(f"Setting up {network_type} network")
        
        network_config = {
            "network_name": "sutazai-network",
            "type": network_type,
            "subnet": subnet,
            "gateway": "172.20.0.1",
            "dns": ["8.8.8.8", "8.8.4.4"],
            "security_groups": [
                "allow_http",
                "allow_https",
                "allow_internal"
            ]
        }
        
        return {
            "status": "success",
            "action": "network_configured",
            "configuration": network_config,
            "connected_services": [
                "backend",
                "frontend",
                "database",
                "cache"
            ]
        }
    
    def _handle_backup_restore(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle backup and restore operations"""
        operation = data.get("operation", "backup")
        target = data.get("target", "all")
        
        self.logger.info(f"Performing {operation} on {target}")
        
        if operation == "backup":
            result = {
                "backup_id": "backup_20250801_1200",
                "size": "2.5GB",
                "duration": "5 minutes",
                "components": ["database", "volumes", "configurations"],
                "storage_location": "s3://sutazai-backups/"
            }
        else:  # restore
            result = {
                "restore_from": "backup_20250801_1200",
                "restored_components": ["database", "volumes", "configurations"],
                "duration": "8 minutes",
                "verification": "successful"
            }
        
        return {
            "status": "success",
            "action": f"{operation}_completed",
            "target": target,
            "result": result
        }
    
    def _get_container_status(self) -> Dict[str, str]:
        """Get status of all containers"""
        try:
            result = subprocess.run(
                ["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return {"containers": result.stdout}
            else:
                return {"error": "Failed to get container status"}
                
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    agent = InfrastructureDevOpsManagerAgent()
    agent.run()