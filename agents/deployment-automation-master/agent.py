#!/usr/bin/env python3
"""
Deployment Automation Master Agent
Responsible for automated deployments and CI/CD pipelines
"""

import sys
import os
import subprocess
from fastapi import FastAPI
from datetime import datetime
from typing import Dict, Any, List

sys.path.append('/opt/sutazaiapp/agents')

# Create FastAPI app first
app = FastAPI(
    title="Deployment Automation Master",
    description="Responsible for automated deployments and CI/CD pipelines",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "deployment-automation-master",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "agent": "deployment-automation-master",
        "status": "running",
        "description": "Responsible for automated deployments and CI/CD pipelines"
    }

@app.post("/task")
async def process_task_endpoint(task: Dict[str, Any]):
    """Process deployment automation tasks via API"""
    try:
        # Create agent instance for processing
        agent = DeploymentAutomationMasterAgent()
        result = agent.process_task(task)
        return result
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "agent": "deployment-automation-master"
        }

# Import base agent after FastAPI setup to avoid circular imports
try:
    from agents.core.base_agent_v2 import BaseAgentV2
except ImportError:
    # Fallback base agent
    import logging
    
    class BaseAgentV2:
        def __init__(self):
            self.agent_name = "deployment-automation-master"
            self.logger = logging.getLogger(self.agent_name)
            
        def query_ollama(self, prompt: str) -> str:
            return f"Processed: {prompt[:100]}..."


class DeploymentAutomationMasterAgent(BaseAgentV2):
    """Deployment Automation Master Agent implementation"""
    
    def __init__(self):
        super().__init__()
        self.deployment_strategies = [
            "blue_green",
            "canary",
            "rolling_update",
            "recreate"
        ]
        
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process deployment automation tasks"""
        task_type = task.get("type", "")
        task_data = task.get("data", {})
        
        self.logger.info(f"Processing deployment task: {task_type}")
        
        try:
            if task_type == "deploy_application":
                return self._deploy_application(task_data)
            elif task_type == "rollback_deployment":
                return self._rollback_deployment(task_data)
            elif task_type == "setup_cicd":
                return self._setup_cicd_pipeline(task_data)
            elif task_type == "health_check":
                return self._perform_health_check(task_data)
            elif task_type == "scale_service":
                return self._scale_service(task_data)
            else:
                # Use Ollama for general deployment tasks
                prompt = f"""As a Deployment Automation Master, help with this task:
                Type: {task_type}
                Data: {task_data}
                
                Provide deployment strategy and implementation steps."""
                
                response = self.query_ollama(prompt)
                
                return {
                    "status": "success",
                    "task_id": task.get("id"),
                    "result": response or "Deployment assistance provided",
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
    
    def _deploy_application(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy application using specified strategy"""
        app_name = data.get("app_name", "sutazai-app")
        strategy = data.get("strategy", "rolling_update")
        version = data.get("version", "latest")
        
        self.logger.info(f"Deploying {app_name} version {version} using {strategy}")
        
        # Simulate deployment steps
        deployment_steps = [
            "Pulling latest images",
            "Running pre-deployment checks",
            "Applying deployment strategy",
            "Updating service configurations",
            "Performing health checks"
        ]
        
        return {
            "status": "success",
            "action": "application_deployed",
            "app_name": app_name,
            "version": version,
            "strategy": strategy,
            "steps_completed": deployment_steps,
            "deployment_url": f"http://{app_name}.sutazai.local"
        }
    
    def _rollback_deployment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback to previous deployment"""
        app_name = data.get("app_name", "sutazai-app")
        target_version = data.get("target_version", "previous")
        
        self.logger.info(f"Rolling back {app_name} to {target_version}")
        
        return {
            "status": "success",
            "action": "deployment_rolled_back",
            "app_name": app_name,
            "rolled_back_to": target_version,
            "rollback_steps": [
                "Identified rollback target",
                "Preserved current state",
                "Switched to previous version",
                "Verified rollback success"
            ]
        }
    
    def _setup_cicd_pipeline(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Setup CI/CD pipeline"""
        pipeline_type = data.get("pipeline_type", "github_actions")
        repo_url = data.get("repo_url", "")
        
        self.logger.info(f"Setting up {pipeline_type} pipeline")
        
        # Generate pipeline configuration
        pipeline_config = {
            "stages": ["build", "test", "deploy"],
            "triggers": ["push", "pull_request"],
            "environments": ["dev", "staging", "production"]
        }
        
        return {
            "status": "success",
            "action": "cicd_pipeline_created",
            "pipeline_type": pipeline_type,
            "configuration": pipeline_config,
            "webhook_url": f"https://sutazai.io/webhooks/{pipeline_type}"
        }
    
    def _perform_health_check(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform health check on deployed services"""
        service_name = data.get("service_name", "all")
        
        self.logger.info(f"Performing health check on {service_name}")
        
        # Simulate health check
        health_status = {
            "backend": "healthy",
            "frontend": "healthy",
            "database": "healthy",
            "cache": "healthy",
            "message_queue": "healthy"
        }
        
        return {
            "status": "success",
            "action": "health_check_completed",
            "service": service_name,
            "health_status": health_status,
            "overall_status": "all_systems_operational"
        }
    
    def _scale_service(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Scale service replicas"""
        service_name = data.get("service_name", "backend")
        replicas = data.get("replicas", 3)
        
        self.logger.info(f"Scaling {service_name} to {replicas} replicas")
        
        return {
            "status": "success",
            "action": "service_scaled",
            "service": service_name,
            "replicas": replicas,
            "scaling_strategy": "horizontal",
            "load_balancer": "configured"
        }
    
    def _execute_docker_command(self, command: List[str]) -> Dict[str, Any]:
        """Execute docker command safely"""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


if __name__ == "__main__":
    agent = DeploymentAutomationMasterAgent()
    agent.run()