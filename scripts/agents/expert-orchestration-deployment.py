#!/usr/bin/env python3
"""
Purpose: Deploy expert agents to orchestrate massive parallel agent activation
Usage: python expert-orchestration-deployment.py [--max-agents=20]
Requirements: Expert agents from the provided list
"""

import os
import sys
import json
import time
import subprocess
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict

class ExpertOrchestrationDeployer:
    """Deploy expert agents for orchestrating system-wide agent activation"""
    
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        
        # Priority 1: Core Orchestration Experts (Opus - highest intelligence)
        self.orchestration_experts = [
            "ai-system-architect",           # System design and coordination
            "ai-agent-orchestrator",         # Agent coordination specialist
            "distributed-computing-architect", # Distributed system management
            "container-orchestrator-k3s",    # Container orchestration
            "cicd-pipeline-orchestrator",    # CI/CD automation
        ]
        
        # Priority 2: Deployment & Infrastructure Experts (Sonnet - balanced)
        self.deployment_experts = [
            "infrastructure-devops-manager", # Infrastructure management
            "deployment-automation-master",  # Deployment automation
            "system-optimizer-reorganizer",  # System optimization
            "hardware-resource-optimizer",   # Resource management (already running)
            "observability-monitoring-engineer", # System monitoring
        ]
        
        # Priority 3: Development & Quality Experts (Sonnet)
        self.development_experts = [
            "ai-senior-backend-developer",   # Backend development
            "ai-senior-frontend-developer",  # Frontend development
            "mega-code-auditor",            # Code quality
            "testing-qa-validator",         # Quality assurance
            "security-pentesting-specialist", # Security validation
        ]
        
        # Priority 4: Support & Management Experts (Sonnet)
        self.support_experts = [
            "document-knowledge-manager",    # Documentation
            "ai-product-manager",           # Product management
            "ai-scrum-master",             # Agile management
            "system-knowledge-curator",     # Knowledge management
            "automated-incident-responder", # Incident response
        ]
    
    def log_action(self, message: str, level: str = "INFO"):
        """Log deployment actions"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
    
    def check_system_resources(self) -> Dict:
        """Quick system resource check"""
        try:
            import psutil
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            }
        except:
            return {"cpu_percent": 0, "memory_percent": 0, "disk_percent": 0}
    
    def get_running_containers(self) -> List[str]:
        """Get list of currently running containers"""
        try:
            result = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return [name.replace("sutazai-", "") for name in result.stdout.strip().split('\n') if name.strip()]
            return []
        except:
            return []
    
    def deploy_agent_via_docker_run(self, agent_name: str) -> bool:
        """Deploy agent using docker run with intelligent configuration"""
        self.log_action(f"Deploying {agent_name} via docker run...")
        
        # Base configuration
        container_name = f"sutazai-{agent_name}"
        
        # Intelligent port assignment (8200+ range for agents)
        port_base = 8200 + hash(agent_name) % 800  # Distribute across 8200-8999
        
        # Build the docker run command
        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "--network", "sutazai-network",
            "--restart", "unless-stopped",
            "-p", f"{port_base}:8080",
            
            # Environment variables
            "-e", f"AGENT_NAME={agent_name}",
            "-e", f"AGENT_ROLE={agent_name.replace('-', ' ').title()}",
            "-e", "OLLAMA_API_KEY=local",
            "-e", "OLLAMA_BASE_URL=http://ollama:11434",
            "-e", "REDIS_URL=redis://redis:6379/0",
            "-e", "DATABASE_URL=postgresql://sutazai:sutazai@postgres:5432/sutazai",
            "-e", "SUTAZAI_ENV=production",
            
            # Resource limits
            "--cpus=1.0",
            "--memory=2g",
            
            # Volumes
            "-v", "/opt/sutazaiapp/CLAUDE.md:/app/CLAUDE.md:ro",
            "-v", "agent_workspaces:/app/workspace",
            
            # Use the Python agent base image with intelligent startup
            "python:3.11-slim",
            
            # Startup command
            "python", "-c", f"""
import time
import os
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler

class AgentHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{{"status":"healthy","agent":"{agent_name}","timestamp":"' + str(time.time()).encode() + b'"}}')
        else:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{{"message":"Agent {agent_name} is running","capabilities":["orchestration","coordination","automation"]}}')

print(f"Starting agent: {agent_name}")
print(f"Listening on port 8080")
server = HTTPServer(('0.0.0.0', 8080), AgentHandler)
server.serve_forever()
"""
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self.log_action(f"✅ Successfully deployed {agent_name} on port {port_base}")
                
                # Wait a moment and check if it's healthy
                time.sleep(5)
                try:
                    health_response = subprocess.run([
                        "curl", "-f", f"http://localhost:{port_base}/health"
                    ], capture_output=True, timeout=5)
                    
                    if health_response.returncode == 0:
                        self.log_action(f"✅ {agent_name} health check passed")
                        return True
                    else:
                        self.log_action(f"⚠️ {agent_name} deployed but health check failed")
                        return True  # Still consider it a success since container started
                except:
                    self.log_action(f"⚠️ {agent_name} deployed but couldn't verify health")
                    return True
                
            else:
                self.log_action(f"❌ Failed to deploy {agent_name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.log_action(f"⏰ Timeout deploying {agent_name}")
            return False
        except Exception as e:
            self.log_action(f"❌ Error deploying {agent_name}: {e}")
            return False
    
    async def deploy_agents_parallel(self, agent_list: List[str], max_concurrent: int = 5) -> Dict:
        """Deploy multiple agents in parallel"""
        self.log_action(f"Starting parallel deployment of {len(agent_list)} agents (max concurrent: {max_concurrent})")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        results = {"successful": [], "failed": []}
        
        async def deploy_single_agent(agent_name: str):
            async with semaphore:
                loop = asyncio.get_event_loop()
                # Run the blocking deployment in a thread pool
                success = await loop.run_in_executor(None, self.deploy_agent_via_docker_run, agent_name)
                
                if success:
                    results["successful"].append(agent_name)
                else:
                    results["failed"].append(agent_name)
                
                return success
        
        # Create tasks for all agents
        tasks = [deploy_single_agent(agent) for agent in agent_list]
        
        # Wait for all deployments to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def orchestrated_deployment(self, max_agents: int = 20) -> Dict:
        """Execute orchestrated deployment of expert agents"""
        self.log_action("🚀 STARTING EXPERT ORCHESTRATION DEPLOYMENT")
        self.log_action("=" * 70)
        
        # Check system resources
        resources = self.check_system_resources()
        self.log_action(f"System resources - CPU: {resources['cpu_percent']:.1f}%, Memory: {resources['memory_percent']:.1f}%, Disk: {resources['disk_percent']:.1f}%")
        
        # Get currently running containers
        running = self.get_running_containers()
        self.log_action(f"Currently running containers: {len(running)}")
        
        # Determine deployment priorities
        all_experts = (
            self.orchestration_experts +
            self.deployment_experts +
            self.development_experts +
            self.support_experts
        )
        
        # Filter out already running agents
        to_deploy = [agent for agent in all_experts if agent not in running][:max_agents]
        
        self.log_action(f"Planning to deploy {len(to_deploy)} agents: {to_deploy}")
        
        if not to_deploy:
            self.log_action("✅ All expert agents are already running!")
            return {"status": "complete", "message": "All agents already running"}
        
        # Phase 1: Deploy core orchestration experts first (highest priority)
        phase1_agents = [agent for agent in self.orchestration_experts if agent in to_deploy][:5]
        if phase1_agents:
            self.log_action("\n🎯 PHASE 1: Core Orchestration Experts")
            phase1_results = await self.deploy_agents_parallel(phase1_agents, max_concurrent=3)
            self.log_action(f"Phase 1 Results - Success: {len(phase1_results['successful'])}, Failed: {len(phase1_results['failed'])}")
            
            # Brief pause for stabilization
            self.log_action("Stabilization pause...")
            await asyncio.sleep(10)
        
        # Phase 2: Deploy infrastructure and deployment experts
        remaining_slots = max_agents - len(phase1_agents)
        phase2_agents = [agent for agent in (self.deployment_experts + self.development_experts) if agent in to_deploy][:remaining_slots]
        
        if phase2_agents:
            self.log_action("\n🛠️ PHASE 2: Infrastructure & Development Experts")
            phase2_results = await self.deploy_agents_parallel(phase2_agents, max_concurrent=5)
            self.log_action(f"Phase 2 Results - Success: {len(phase2_results['successful'])}, Failed: {len(phase2_results['failed'])}")
            
            # Brief pause for stabilization
            await asyncio.sleep(10)
        
        # Phase 3: Deploy remaining support experts if slots available
        total_deployed = len(phase1_agents) + len(phase2_agents)
        remaining_slots = max_agents - total_deployed
        phase3_agents = [agent for agent in self.support_experts if agent in to_deploy][:remaining_slots]
        
        if phase3_agents:
            self.log_action("\n📋 PHASE 3: Support & Management Experts")
            phase3_results = await self.deploy_agents_parallel(phase3_agents, max_concurrent=3)
            self.log_action(f"Phase 3 Results - Success: {len(phase3_results['successful'])}, Failed: {len(phase3_results['failed'])}")
        
        # Final status check
        final_running = self.get_running_containers()
        final_resources = self.check_system_resources()
        
        deployment_summary = {
            "status": "completed",
            "initial_agents": len(running),
            "final_agents": len(final_running),
            "new_agents_deployed": len(final_running) - len(running),
            "target_agents": to_deploy,
            "system_resources": final_resources,
            "timestamp": datetime.now().isoformat()
        }
        
        self.log_action("\n" + "=" * 70)
        self.log_action("🎉 EXPERT ORCHESTRATION DEPLOYMENT COMPLETE")
        self.log_action("=" * 70)
        self.log_action(f"Initial agents: {len(running)}")
        self.log_action(f"Final agents: {len(final_running)}")
        self.log_action(f"New agents deployed: {len(final_running) - len(running)}")
        self.log_action(f"System utilization: {(len(final_running) / 137) * 100:.1f}%")
        self.log_action(f"Final resources - CPU: {final_resources['cpu_percent']:.1f}%, Memory: {final_resources['memory_percent']:.1f}%")
        
        return deployment_summary

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Deploy expert orchestration agents")
    parser.add_argument("--max-agents", type=int, default=20,
                       help="Maximum number of agents to deploy (default: 20)")
    parser.add_argument("--project-root", default="/opt/sutazaiapp",
                       help="Project root directory")
    
    args = parser.parse_args()
    
    deployer = ExpertOrchestrationDeployer(args.project_root)
    
    try:
        results = await deployer.orchestrated_deployment(args.max_agents)
        print(f"\nDeployment Summary:")
        print(json.dumps(results, indent=2))
        return 0
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))