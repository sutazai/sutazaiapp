#!/usr/bin/env python3
"""
Core Agent Startup - Start essential agents in controlled manner
"""

import asyncio
import docker
import logging
import time
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoreAgentStarter:
    """Start core agents with proper sequencing and health checks"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        
        # Define core agents in startup order (most critical first)
        self.core_agents = [
            "hardware-resource-optimizer",     # Already running - resource monitoring
            "garbage-collector-coordinator",   # System cleanup
            "ai-metrics-exporter",            # Metrics collection 
            "edge-inference-proxy",           # Local AI processing
            "data-drift-detector",            # Data monitoring
            "experiment-tracker"              # ML experiment tracking
        ]
        
        # Additional important agents to start if available
        self.important_agents = [
            "prompt-injection-guard",
            "resource-visualiser", 
            "metrics-collector-prometheus",
            "ram-hardware-optimizer",
            "cpu-only-hardware-optimizer"
        ]
    
    def get_container_status(self, agent_name: str) -> Dict:
        """Get container status"""
        try:
            container_name = f"sutazai-{agent_name}"
            container = self.docker_client.containers.get(container_name)
            
            return {
                "exists": True,
                "status": container.status,
                "running": container.status == "running",
                "can_start": container.status in ["created", "exited"],
                "container": container
            }
        except docker.errors.NotFound:
            return {"exists": False, "status": "not_found", "running": False, "can_start": False}
        except Exception as e:
            logger.error(f"Error checking {agent_name}: {e}")
            return {"exists": False, "status": "error", "running": False, "can_start": False}
    
    async def start_agent(self, agent_name: str) -> bool:
        """Start a single agent and verify it's healthy"""
        logger.info(f"🚀 Starting {agent_name}...")
        
        status = self.get_container_status(agent_name)
        
        if not status["exists"]:
            logger.warning(f"⚠️ {agent_name} container not found")
            return False
        
        if status["running"]:
            logger.info(f"✅ {agent_name} already running")
            return True
        
        if not status["can_start"]:
            logger.warning(f"⚠️ {agent_name} cannot be started (status: {status['status']})")
            return False
        
        try:
            # Start the container
            container = status["container"]
            container.start()
            
            # Wait for startup
            logger.info(f"⏳ Waiting for {agent_name} to start...")
            await asyncio.sleep(8)
            
            # Check if it's running
            container.reload()
            if container.status == "running":
                logger.info(f"✅ {agent_name} started successfully")
                return True
            else:
                logger.error(f"❌ {agent_name} failed to start (status: {container.status})")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error starting {agent_name}: {e}")
            return False
    
    async def start_agent_group(self, agent_list: List[str], group_name: str) -> Dict:
        """Start a group of agents"""
        logger.info(f"📦 Starting {group_name} agents...")
        
        results = {"successful": [], "failed": [], "already_running": []}
        
        for agent_name in agent_list:
            status = self.get_container_status(agent_name)
            
            if status["running"]:
                results["already_running"].append(agent_name)
                continue
            
            success = await self.start_agent(agent_name)
            if success:
                results["successful"].append(agent_name)
            else:
                results["failed"].append(agent_name)
            
            # Brief pause between starts
            await asyncio.sleep(3)
        
        total_healthy = len(results["successful"]) + len(results["already_running"])
        logger.info(f"📊 {group_name} results: {total_healthy} healthy, {len(results['failed'])} failed")
        
        return results
    
    async def health_check_all(self) -> Dict:
        """Check health of all agents"""
        logger.info("🔍 Performing health check...")
        
        health_status = {"healthy": [], "unhealthy": [], "not_found": []}
        
        all_agents = self.core_agents + self.important_agents
        
        for agent_name in all_agents:
            status = self.get_container_status(agent_name)
            
            if not status["exists"]:
                health_status["not_found"].append(agent_name)
            elif status["running"]:
                health_status["healthy"].append(agent_name)
            else:
                health_status["unhealthy"].append(agent_name)
        
        logger.info(f"💚 Healthy: {len(health_status['healthy'])}")
        logger.info(f"🔴 Unhealthy: {len(health_status['unhealthy'])}")
        logger.info(f"❓ Not found: {len(health_status['not_found'])}")
        
        return health_status
    
    async def startup_sequence(self) -> Dict:
        """Execute complete startup sequence"""
        logger.info("🤖 Starting SutazAI Core Agent Startup Sequence...")
        
        # Initial health check
        initial_health = await self.health_check_all()
        logger.info(f"Initial state: {len(initial_health['healthy'])} agents already healthy")
        
        # Start core agents first
        core_results = await self.start_agent_group(self.core_agents, "Core")
        
        # Brief delay before starting important agents
        await asyncio.sleep(10)
        
        # Start important agents
        important_results = await self.start_agent_group(self.important_agents, "Important")
        
        # Final health check
        final_health = await self.health_check_all()
        
        # Summary
        total_started = len(core_results["successful"]) + len(important_results["successful"])
        total_healthy = len(final_health["healthy"])
        
        summary = {
            "total_agents_started": total_started,
            "total_healthy_agents": total_healthy,
            "core_results": core_results,
            "important_results": important_results,
            "final_health": final_health,
            "success_rate": (total_healthy / len(self.core_agents + self.important_agents)) * 100
        }
        
        logger.info("🎉 Startup sequence completed!")
        logger.info(f"📊 Summary: {total_healthy} healthy agents ({summary['success_rate']:.1f}% success rate)")
        
        return summary

async def main():
    """Main execution"""
    starter = CoreAgentStarter()
    
    try:
        result = await starter.startup_sequence()
        
        # Print summary
        print("=" * 50)
        print("CORE AGENT STARTUP SUMMARY")
        print("=" * 50)
        print(f"Total Healthy Agents: {result['total_healthy_agents']}")
        print(f"Success Rate: {result['success_rate']:.1f}%")
        print(f"Agents Started: {result['total_agents_started']}")
        
        if result['final_health']['healthy']:
            print("\n✅ Healthy Agents:")
            for agent in result['final_health']['healthy']:
                print(f"  - {agent}")
        
        if result['final_health']['unhealthy']:
            print("\n❌ Unhealthy Agents:")
            for agent in result['final_health']['unhealthy']:
                print(f"  - {agent}")
        
        return 0 if result['total_healthy_agents'] > 0 else 1
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))