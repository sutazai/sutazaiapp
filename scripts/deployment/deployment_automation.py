#!/usr/bin/env python3
"""
Deployment Automation Workflow
Practical example of automating deployments with SutazAI
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import asyncio
import httpx
import json
import os
import subprocess
from datetime import datetime

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class DeploymentWorkflow:
    """Automated deployment workflow"""
    
    def __init__(self):
        self.api_url = f"{API_BASE_URL}/api/v1"
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def pre_deployment_checks(self) -> Dict[str, Any]:
        """Run pre-deployment validation checks"""
        logger.info("🔍 Running pre-deployment checks...")
        
        checks = {
            "timestamp": datetime.now().isoformat(),
            "checks": []
        }
        
        # Check 1: Verify Docker is running
        docker_check = self._check_docker()
        checks["checks"].append(docker_check)
        
        # Check 2: Verify required services
        services_check = await self._check_services()
        checks["checks"].append(services_check)
        
        # Check 3: Run tests
        test_check = await self._run_tests()
        checks["checks"].append(test_check)
        
        # Check 4: Security scan
        security_check = await self._security_scan()
        checks["checks"].append(security_check)
        
        # Determine if deployment should proceed
        all_passed = all(check["passed"] for check in checks["checks"])
        checks["ready_to_deploy"] = all_passed
        
        return checks
    
    def _check_docker(self) -> Dict[str, Any]:
        """Check if Docker is running"""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=5
            )
            passed = result.returncode == 0
            return {
                "name": "Docker Status",
                "passed": passed,
                "message": "Docker is running" if passed else "Docker is not running"
            }
        except Exception as e:
            return {
                "name": "Docker Status",
                "passed": False,
                "message": f"Failed to check Docker: {e}"
            }
    
    async def _check_services(self) -> Dict[str, Any]:
        """Check if required services are healthy"""
        services = ["postgres", "redis", "ollama"]
        healthy = []
        unhealthy = []
        
        for service in services:
            try:
                result = subprocess.run(
                    ["docker", "ps", "--filter", f"name=sutazai-{service}", "--format", "{{.Status}}"],
                    capture_output=True,
                    text=True
                )
                if "healthy" in result.stdout or "Up" in result.stdout:
                    healthy.append(service)
                else:
                    unhealthy.append(service)
            except (IOError, OSError, FileNotFoundError) as e:
                logger.error(f"Unexpected exception: {e}", exc_info=True)
                unhealthy.append(service)
        
        passed = len(unhealthy) == 0
        return {
            "name": "Service Health",
            "passed": passed,
            "message": f"Healthy: {healthy}, Unhealthy: {unhealthy}" if unhealthy else "All services healthy",
            "details": {
                "healthy": healthy,
                "unhealthy": unhealthy
            }
        }
    
    async def _run_tests(self) -> Dict[str, Any]:
        """Run automated tests"""
        # In a real implementation, this would run actual tests
        # For now, we'll simulate using the testing agent
        
        try:
            response = await self.client.post(
                f"{self.api_url}/agents/execute",
                json={
                    "agent": "testing-qa-validator",
                    "task": "run_tests",
                    "data": {
                        "test_suite": "deployment",
                        "directories": ["./backend", "./workflows"]
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                passed = result.get("all_tests_passed", False)
                return {
                    "name": "Automated Tests",
                    "passed": passed,
                    "message": f"{result.get('passed', 0)} tests passed, {result.get('failed', 0)} failed"
                }
            else:
                return {
                    "name": "Automated Tests",
                    "passed": False,
                    "message": "Failed to run tests"
                }
        except Exception as e:
            return {
                "name": "Automated Tests",
                "passed": False,
                "message": f"Test execution failed: {e}"
            }
    
    async def _security_scan(self) -> Dict[str, Any]:
        """Run security scan"""
        try:
            response = await self.client.post(
                f"{self.api_url}/agents/execute",
                json={
                    "agent": "security-pentesting-specialist",
                    "task": "quick_scan",
                    "data": {
                        "target": "./",
                        "scan_type": "deployment"
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                vulnerabilities = result.get("vulnerabilities", [])
                critical = [v for v in vulnerabilities if v.get("severity") == "CRITICAL"]
                
                return {
                    "name": "Security Scan",
                    "passed": len(critical) == 0,
                    "message": f"Found {len(critical)} critical vulnerabilities" if critical else "No critical vulnerabilities found"
                }
            else:
                return {
                    "name": "Security Scan",
                    "passed": False,
                    "message": "Security scan failed"
                }
        except Exception as e:
            return {
                "name": "Security Scan",
                "passed": False,
                "message": f"Security scan error: {e}"
            }
    
    async def create_deployment_plan(self, environment: str = "production") -> Dict[str, Any]:
        """Create a deployment plan"""
        logger.info(f"📋 Creating deployment plan for {environment}...")
        
        try:
            response = await self.client.post(
                f"{self.api_url}/agents/execute",
                json={
                    "agent": "deployment-automation-master",
                    "task": "create_plan",
                    "data": {
                        "environment": environment,
                        "strategy": "rolling_update",
                        "services": ["backend", "frontend", "workers"]
                    }
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": "Failed to create deployment plan"}
        except Exception as e:
            return {"error": f"Plan creation failed: {e}"}
    
    async def execute_deployment(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the deployment plan"""
        logger.info("🚀 Executing deployment...")
        
        results = {
            "started_at": datetime.now().isoformat(),
            "steps": []
        }
        
        # Simulate deployment steps
        steps = [
            {"name": "Build Docker images", "command": "docker-compose build"},
            {"name": "Stop current services", "command": "docker-compose down"},
            {"name": "Start new services", "command": "docker-compose up -d"},
            {"name": "Run migrations", "command": "docker-compose exec backend python -m alembic upgrade head"},
            {"name": "Health check", "command": "curl http://localhost:8000/health"}
        ]
        
        for step in steps:
            logger.info(f"  ▶️  {step['name']}...")
            
            try:
                # In production, you'd execute actual commands
                # For demo, we'll simulate
                await asyncio.sleep(1)  # Simulate work
                
                results["steps"].append({
                    "name": step["name"],
                    "status": "success",
                    "message": f"Completed: {step['command']}"
                })
            except Exception as e:
                results["steps"].append({
                    "name": step["name"],
                    "status": "failed",
                    "message": str(e)
                })
                break
        
        results["completed_at"] = datetime.now().isoformat()
        results["success"] = all(step["status"] == "success" for step in results["steps"])
        
        return results
    
    async def post_deployment_validation(self) -> Dict[str, Any]:
        """Validate deployment success"""
        logger.info("✅ Running post-deployment validation...")
        
        validations = []
        
        # Check API health
        try:
            response = await self.client.get(f"{API_BASE_URL}/health")
            validations.append({
                "check": "API Health",
                "passed": response.status_code == 200,
                "details": response.json() if response.status_code == 200 else "API not responding"
            })
        except Exception as e:
            validations.append({
                "check": "API Health",
                "passed": False,
                "details": str(e)
            })
        
        # Check database connectivity
        try:
            response = await self.client.get(f"{self.api_url}/system/database/status")
            validations.append({
                "check": "Database Connection",
                "passed": response.status_code == 200,
                "details": "Connected" if response.status_code == 200 else "Not connected"
            })
        except (IOError, OSError, FileNotFoundError) as e:
            logger.error(f"Unexpected exception: {e}", exc_info=True)
            validations.append({
                "check": "Database Connection",
                "passed": False,
                "details": "Failed to check"
            })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "validations": validations,
            "deployment_successful": all(v["passed"] for v in validations)
        }
    
    async def close(self):
        """Clean up resources"""
        await self.client.aclose()


async def main():
    """Example deployment workflow"""
    workflow = DeploymentWorkflow()
    
    try:
        # Step 1: Pre-deployment checks
        logger.info("🚀 Starting deployment workflow...")
        checks = await workflow.pre_deployment_checks()
        
        logger.info("\n📊 Pre-deployment check results:")
        for check in checks["checks"]:
            status = "✅" if check["passed"] else "❌"
            logger.info(f"  {status} {check['name']}: {check['message']}")
        
        if not checks["ready_to_deploy"]:
            logger.error("\n❌ Pre-deployment checks failed. Aborting deployment.")
            return
        
        # Step 2: Create deployment plan
        plan = await workflow.create_deployment_plan("production")
        
        if "error" in plan:
            logger.error(f"\n❌ Failed to create deployment plan: {plan['error']}")
            return
        
        logger.info("\n📋 Deployment plan created successfully")
        
        # Step 3: Execute deployment
        deployment_result = await workflow.execute_deployment(plan)
        
        logger.info("\n📊 Deployment results:")
        for step in deployment_result["steps"]:
            status = "✅" if step["status"] == "success" else "❌"
            logger.info(f"  {status} {step['name']}: {step['message']}")
        
        if not deployment_result["success"]:
            logger.error("\n❌ Deployment failed!")
            return
        
        # Step 4: Post-deployment validation
        validation = await workflow.post_deployment_validation()
        
        logger.info("\n📊 Post-deployment validation:")
        for val in validation["validations"]:
            status = "✅" if val["passed"] else "❌"
            logger.info(f"  {status} {val['check']}: {val['details']}")
        
        if validation["deployment_successful"]:
            logger.info("\n✅ Deployment completed successfully!")
        else:
            logger.warning("\n⚠️  Deployment completed with warnings. Please check the validation results.")
        
        # Save deployment report
        report = {
            "deployment_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "pre_checks": checks,
            "deployment_plan": plan,
            "execution_results": deployment_result,
            "post_validation": validation
        }
        
        report_path = f"deployment_report_{report['deployment_id']}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n📄 Deployment report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"\n❌ Deployment workflow error: {e}")
    
    finally:
        await workflow.close()


if __name__ == "__main__":
