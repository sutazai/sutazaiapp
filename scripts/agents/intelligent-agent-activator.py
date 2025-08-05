#!/usr/bin/env python3
"""
Intelligent Agent Activator - Smart coordination for SutazAI agent system
Manages systematic startup of agents based on resource allocation and priorities
"""

import asyncio
import docker
import json
import psutil
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntelligentAgentActivator:
    """Smart agent activation with resource awareness and health monitoring"""
    
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.docker_client = docker.from_env()
        
        # Load configurations
        self.load_configurations()
        
        # State tracking
        self.active_agents: Set[str] = set()
        self.failed_agents: Set[str] = set()
        self.agent_health: Dict[str, Dict] = {}
        self.resource_usage = {"cpu": 0, "memory": 0}
        
        # Core services that must be healthy before agent activation
        self.core_services = ["postgres", "redis", "ollama", "neo4j", "qdrant"]
        
    def load_configurations(self):
        """Load agent registry and resource allocation configs"""
        try:
            # Load agent registry
            registry_path = self.project_root / "agents" / "agent_registry.json"
            with open(registry_path, 'r') as f:
                self.agent_registry = json.load(f).get("agents", {})
            
            # Load resource allocation
            resource_path = self.project_root / "config" / "agent-resource-allocation.yml"
            with open(resource_path, 'r') as f:
                self.resource_config = yaml.safe_load(f)
                
            # Load communication config
            comm_path = self.project_root / "agents" / "communication_config.json"
            with open(comm_path, 'r') as f:
                self.comm_config = json.load(f)
                
            logger.info("✅ Configurations loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load configurations: {e}")
            raise
    
    def get_system_resources(self) -> Dict[str, float]:
        """Get current system resource utilization"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3),
            "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
        }
    
    def check_resource_constraints(self) -> bool:
        """Check if system can handle more agents"""
        resources = self.get_system_resources()
        thresholds = self.resource_config["system_constraints"]["activation_thresholds"]
        
        if resources["cpu_percent"] > thresholds["cpu_critical"]:
            logger.warning(f"⚠️ CPU usage critical: {resources['cpu_percent']:.1f}%")
            return False
            
        if resources["memory_percent"] > thresholds["memory_critical"]:
            logger.warning(f"⚠️ Memory usage critical: {resources['memory_percent']:.1f}%")
            return False
            
        if resources["disk_percent"] > thresholds["disk_critical"]:
            logger.warning(f"⚠️ Disk usage critical: {resources['disk_percent']:.1f}%")
            return False
            
        return True
    
    def get_container_status(self, container_name: str) -> Dict[str, str]:
        """Get detailed container status"""
        try:
            container = self.docker_client.containers.get(container_name)
            health_status = "unknown"
            
            # Check health if available
            health = container.attrs.get("State", {}).get("Health", {})
            if health:
                health_status = health.get("Status", "unknown")
            
            return {
                "status": container.status,
                "health": health_status,
                "running": container.status == "running",
                "healthy": health_status in ["healthy", "unknown"] and container.status == "running"
            }
        except docker.errors.NotFound:
            return {"status": "not_found", "health": "not_found", "running": False, "healthy": False}
        except Exception as e:
            logger.warning(f"Error checking container {container_name}: {e}")
            return {"status": "error", "health": "error", "running": False, "healthy": False}
    
    async def ensure_core_services(self) -> bool:
        """Ensure core services are healthy before agent activation"""
        logger.info("🔧 Checking core services health...")
        
        unhealthy_services = []
        
        for service in self.core_services:
            container_name = f"sutazai-{service}"
            status = self.get_container_status(container_name)
            
            if not status["healthy"]:
                unhealthy_services.append(service)
                logger.warning(f"⚠️ {service} is not healthy: {status}")
                
                # Try to start/restart the service
                try:
                    container = self.docker_client.containers.get(container_name)
                    if container.status != "running":
                        logger.info(f"🔄 Restarting {service}...")
                        container.restart()
                        await asyncio.sleep(10)  # Wait for startup
                except Exception as e:
                    logger.error(f"❌ Failed to restart {service}: {e}")
        
        # Re-check after restart attempts
        if unhealthy_services:
            await asyncio.sleep(15)  # Additional wait time
            remaining_unhealthy = []
            
            for service in unhealthy_services:
                container_name = f"sutazai-{service}"
                status = self.get_container_status(container_name)
                if not status["healthy"]:
                    remaining_unhealthy.append(service)
            
            if remaining_unhealthy:
                logger.error(f"❌ Core services still unhealthy: {remaining_unhealthy}")
                return False
        
        logger.info("✅ All core services are healthy")
        return True
    
    def prioritize_agents(self) -> List[Tuple[str, int, str]]:
        """Prioritize agents for activation based on phases and dependencies"""
        prioritized = []
        
        # Phase 1: Critical agents (highest priority)
        phase1_agents = [
            "hardware-resource-optimizer",
            "ai-system-architect", 
            "infrastructure-devops-manager",
            "ai-agent-orchestrator",
            "deployment-automation-master"
        ]
        
        for agent in phase1_agents:
            if agent in self.agent_registry:
                prioritized.append((agent, 1, "critical"))
        
        # Phase 2: Performance agents
        phase2_agents = [
            "observability-monitoring-engineer",
            "system-optimizer-reorganizer",
            "garbage-collector-coordinator",
            "ai-senior-backend-developer",
            "ai-senior-frontend-developer"
        ]
        
        for agent in phase2_agents:
            if agent in self.agent_registry:
                prioritized.append((agent, 2, "performance"))
        
        # Phase 3: Specialized agents (limited activation)
        phase3_agents = [
            "document-knowledge-manager",
            "testing-qa-validator",
            "security-pentesting-specialist",
            "ai-product-manager",
            "ai-scrum-master"
        ]
        
        for agent in phase3_agents:
            if agent in self.agent_registry:
                prioritized.append((agent, 3, "specialized"))
        
        return prioritized
    
    async def start_agent_container(self, agent_name: str, phase: int) -> bool:
        """Start a single agent container with proper resource limits"""
        try:
            container_name = f"sutazai-{agent_name}"
            
            # Check if already running
            status = self.get_container_status(container_name)
            if status["healthy"]:
                logger.info(f"✅ {agent_name} already healthy")
                self.active_agents.add(agent_name)
                return True
            
            # Get resource limits for phase
            pool_config = self.resource_config["resource_pools"]
            if phase == 1:
                limits = pool_config["critical_pool"]["agent_limits"]
            elif phase == 2:
                limits = pool_config["performance_pool"]["agent_limits"]
            else:
                limits = pool_config["specialized_pool"]["agent_limits"]
            
            # Check for agent-specific overrides
            overrides = self.resource_config.get("agent_overrides", {})
            if agent_name in overrides:
                limits.update(overrides[agent_name])
            
            logger.info(f"🚀 Starting {agent_name} (Phase {phase}) with limits: {limits}")
            
            # Try to get existing container
            try:
                container = self.docker_client.containers.get(container_name)
                
                # Remove if not running properly
                if container.status in ["exited", "created", "dead"]:
                    logger.info(f"🗑️ Removing old container for {agent_name}")
                    container.remove(force=True)
                    await asyncio.sleep(2)
                elif container.status == "restarting":
                    logger.info(f"🔄 Stopping restarting container for {agent_name}")
                    container.stop(timeout=10)
                    container.remove(force=True)
                    await asyncio.sleep(2)
                else:
                    # Try to start if stopped
                    container.start()
                    await asyncio.sleep(5)
                    
                    # Check if healthy now
                    status = self.get_container_status(container_name)
                    if status["healthy"]:
                        self.active_agents.add(agent_name)
                        return True
            
            except docker.errors.NotFound:
                logger.warning(f"⚠️ Container {container_name} not found - may need to be built first")
                return False
            
            # Wait for container to start
            await asyncio.sleep(10)
            
            # Verify startup
            status = self.get_container_status(container_name)
            if status["healthy"]:
                logger.info(f"✅ {agent_name} started successfully")
                self.active_agents.add(agent_name)
                return True
            else:
                logger.error(f"❌ {agent_name} failed to start properly: {status}")
                self.failed_agents.add(agent_name)
                return False
                
        except Exception as e:
            logger.error(f"❌ Error starting {agent_name}: {e}")
            self.failed_agents.add(agent_name)
            return False
    
    async def activate_agent_batch(self, batch: List[Tuple[str, int, str]], max_concurrent: int = 3) -> Dict:
        """Activate a batch of agents with concurrency control"""
        logger.info(f"📦 Activating batch of {len(batch)} agents (max concurrent: {max_concurrent})")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def activate_with_semaphore(agent_info):
            async with semaphore:
                agent_name, phase, category = agent_info
                return await self.start_agent_container(agent_name, phase)
        
        # Execute with concurrency control
        results = await asyncio.gather(
            *[activate_with_semaphore(agent_info) for agent_info in batch],
            return_exceptions=True
        )
        
        # Count results
        successful = sum(1 for r in results if r is True)
        failed = len(results) - successful
        
        return {"successful": successful, "failed": failed, "total": len(batch)}
    
    async def intelligent_activation(self, max_agents: int = 15) -> Dict:
        """Intelligently activate agents based on system capacity and priorities"""
        logger.info("🧠 Starting intelligent agent activation...")
        
        # Step 1: Ensure core services
        if not await self.ensure_core_services():
            return {"status": "failed", "error": "Core services not healthy"}
        
        # Step 2: Check initial system resources
        if not self.check_resource_constraints():
            return {"status": "failed", "error": "System resources exceeded limits"}
        
        # Step 3: Get prioritized agent list
        prioritized_agents = self.prioritize_agents()
        logger.info(f"📋 Found {len(prioritized_agents)} agents to potentially activate")
        
        # Step 4: Activate in batches with monitoring
        activation_results = {"phases": {}, "total_activated": 0, "total_failed": 0, "agents": {}}
        
        current_batch = []
        current_phase = None
        agents_activated = 0
        
        for agent_name, phase, category in prioritized_agents:
            # Check if we've hit our limit
            if agents_activated >= max_agents:
                logger.info(f"🛑 Reached maximum agent limit ({max_agents})")
                break
            
            # Process batch when phase changes
            if current_phase is not None and phase != current_phase and current_batch:
                logger.info(f"🔄 Processing Phase {current_phase} batch ({len(current_batch)} agents)")
                
                # Check resources before processing
                if not self.check_resource_constraints():
                    logger.warning("⚠️ Resource constraints exceeded, stopping activation")
                    break
                
                # Activate batch
                batch_result = await self.activate_agent_batch(current_batch, max_concurrent=2)
                activation_results["phases"][current_phase] = batch_result
                agents_activated += batch_result["successful"]
                
                # Clear batch
                current_batch = []
                
                # Inter-phase delay
                await asyncio.sleep(10)
            
            current_batch.append((agent_name, phase, category))
            current_phase = phase
        
        # Process final batch
        if current_batch and agents_activated < max_agents:
            logger.info(f"🔄 Processing final Phase {current_phase} batch ({len(current_batch)} agents)")
            
            if self.check_resource_constraints():
                batch_result = await self.activate_agent_batch(current_batch, max_concurrent=2)
                activation_results["phases"][current_phase] = batch_result
                agents_activated += batch_result["successful"]
        
        # Final summary
        activation_results.update({
            "status": "completed",
            "total_activated": len(self.active_agents),
            "total_failed": len(self.failed_agents),
            "active_agents": list(self.active_agents),
            "failed_agents": list(self.failed_agents),
            "system_resources": self.get_system_resources(),
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"🎉 Activation complete: {len(self.active_agents)} agents active, {len(self.failed_agents)} failed")
        
        return activation_results
    
    async def health_monitor_loop(self, duration: int = 300):
        """Monitor agent health for specified duration"""
        logger.info(f"🔍 Starting health monitoring for {duration} seconds...")
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            unhealthy_count = 0
            
            for agent_name in list(self.active_agents):
                container_name = f"sutazai-{agent_name}"
                status = self.get_container_status(container_name)
                
                if not status["healthy"]:
                    unhealthy_count += 1
                    logger.warning(f"⚠️ {agent_name} is unhealthy: {status}")
                    
                    # Try to restart unhealthy agents
                    try:
                        container = self.docker_client.containers.get(container_name)
                        if container.status != "running":
                            logger.info(f"🔄 Restarting {agent_name}")
                            container.restart()
                    except Exception as e:
                        logger.error(f"❌ Failed to restart {agent_name}: {e}")
                        self.active_agents.discard(agent_name)
                        self.failed_agents.add(agent_name)
            
            if unhealthy_count > 0:
                logger.warning(f"⚠️ {unhealthy_count} agents are unhealthy")
            
            await asyncio.sleep(30)  # Check every 30 seconds
        
        logger.info("✅ Health monitoring completed")

async def main():
    """Main execution function"""
    activator = IntelligentAgentActivator()
    
    try:
        logger.info("🤖 SutazAI Intelligent Agent Activator Starting...")
        
        # Activate agents intelligently
        result = await activator.intelligent_activation(max_agents=15)
        
        if result["total_activated"] > 0:
            logger.info(f"✅ Successfully activated {result['total_activated']} agents")
            
            # Run health monitoring
            await activator.health_monitor_loop(duration=300)  # 5 minutes
        else:
            logger.error("❌ No agents were successfully activated")
        
        # Save results
        results_path = Path("/opt/sutazaiapp/logs/activation-results.json")
        results_path.parent.mkdir(exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"📄 Results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"❌ Activation failed: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))