#!/usr/bin/env python3
"""
Agent Stability Manager - Monitor and maintain agent health
"""

import asyncio
import docker
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentStabilityManager:
    """Continuously monitor and stabilize agent health"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.problematic_agents: Set[str] = set()
        self.restart_counts: Dict[str, int] = {}
        self.stability_report = {
            "start_time": datetime.now().isoformat(),
            "actions_taken": [],
            "agent_status": {}
        }
    
    def log_action(self, action: str, agent: str = None):
        """Log stability actions"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "agent": agent
        }
        self.stability_report["actions_taken"].append(entry)
        logger.info(f"[ACTION] {action}" + (f" ({agent})" if agent else ""))
    
    def get_agent_containers(self) -> List[str]:
        """Get list of all SutazAI agent containers"""
        agents = []
        try:
            containers = self.docker_client.containers.list(all=True, filters={"name": "sutazai-"})
            for container in containers:
                name = container.name.replace("sutazai-", "")
                # Skip core services
                if name not in ["postgres", "redis", "ollama", "neo4j", "qdrant", "chromadb", "backend"]:
                    agents.append(name)
        except Exception as e:
            logger.error(f"Error discovering agents: {e}")
        
        return sorted(agents)
    
    def analyze_container_health(self, agent_name: str) -> Dict:
        """Analyze detailed container health"""
        try:
            container_name = f"sutazai-{agent_name}"
            container = self.docker_client.containers.get(container_name)
            
            # Get container state
            state = container.attrs.get("State", {})
            
            health_info = {
                "name": agent_name,
                "status": container.status,
                "running": container.status == "running",
                "restart_count": state.get("RestartCount", 0),
                "exit_code": state.get("ExitCode", 0),
                "started_at": state.get("StartedAt", ""),
                "finished_at": state.get("FinishedAt", ""),
                "error": state.get("Error", ""),
                "healthy": False
            }
            
            # Assess health
            if container.status == "running":
                # Check if recently restarted
                if health_info["restart_count"] < 3:
                    health_info["healthy"] = True
                else:
                    health_info["health_issue"] = "excessive_restarts"
            elif container.status == "restarting":
                health_info["health_issue"] = "restart_loop"
            elif container.status == "exited":
                if health_info["exit_code"] != 0:
                    health_info["health_issue"] = f"exit_code_{health_info['exit_code']}"
                else:
                    health_info["health_issue"] = "normal_exit"
            else:
                health_info["health_issue"] = f"status_{container.status}"
            
            return health_info
            
        except docker.errors.NotFound:
            return {"name": agent_name, "status": "not_found", "healthy": False, "health_issue": "container_missing"}
        except Exception as e:
            return {"name": agent_name, "status": "error", "healthy": False, "health_issue": f"analysis_error: {e}"}
    
    async def stabilize_agent(self, agent_name: str, health_info: Dict) -> bool:
        """Take corrective action for unhealthy agent"""
        try:
            container_name = f"sutazai-{agent_name}"
            
            if health_info["status"] == "not_found":
                self.log_action(f"Container missing - cannot stabilize", agent_name)
                return False
            
            # Handle restart loops
            if health_info.get("health_issue") == "restart_loop":
                self.log_action(f"Stopping restart loop", agent_name)
                container = self.docker_client.containers.get(container_name)
                container.stop(timeout=10)
                await asyncio.sleep(5)
                
                # Try to start with fresh state
                container.start()
                await asyncio.sleep(10)
                return True
            
            # Handle excessive restarts
            elif health_info.get("health_issue") == "excessive_restarts":
                restart_count = self.restart_counts.get(agent_name, 0)
                if restart_count < 2:  # Allow 2 restart attempts
                    self.log_action(f"Restarting due to excessive restarts (attempt {restart_count + 1})", agent_name)
                    container = self.docker_client.containers.get(container_name)
                    container.restart()
                    self.restart_counts[agent_name] = restart_count + 1
                    await asyncio.sleep(10)
                    return True
                else:
                    self.log_action(f"Marking as problematic - too many restart attempts", agent_name)
                    self.problematic_agents.add(agent_name)
                    return False
            
            # Handle exited containers
            elif health_info["status"] == "exited":
                self.log_action(f"Restarting exited container", agent_name)
                container = self.docker_client.containers.get(container_name)
                container.start()
                await asyncio.sleep(8)
                return True
            
            # Handle created but not started
            elif health_info["status"] == "created":
                self.log_action(f"Starting created container", agent_name)
                container = self.docker_client.containers.get(container_name)
                container.start()
                await asyncio.sleep(8)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error stabilizing {agent_name}: {e}")
            return False
    
    async def stability_sweep(self) -> Dict:
        """Perform one complete stability sweep"""
        logger.info("🔍 Performing stability sweep...")
        
        agents = self.get_agent_containers()
        sweep_results = {
            "total_agents": len(agents),
            "healthy": 0,
            "stabilized": 0,
            "problematic": 0,
            "actions": []
        }
        
        for agent_name in agents:
            if agent_name in self.problematic_agents:
                continue  # Skip known problematic agents
            
            health_info = self.analyze_container_health(agent_name)
            self.stability_report["agent_status"][agent_name] = health_info
            
            if health_info["healthy"]:
                sweep_results["healthy"] += 1
            else:
                # Try to stabilize
                self.log_action(f"Attempting to stabilize unhealthy agent", agent_name)
                success = await self.stabilize_agent(agent_name, health_info)
                
                if success:
                    sweep_results["stabilized"] += 1
                    sweep_results["actions"].append(f"Stabilized {agent_name}")
                else:
                    sweep_results["problematic"] += 1
                    sweep_results["actions"].append(f"Failed to stabilize {agent_name}")
        
        logger.info(f"📊 Sweep complete: {sweep_results['healthy']} healthy, "
                   f"{sweep_results['stabilized']} stabilized, {sweep_results['problematic']} problematic")
        
        return sweep_results
    
    async def continuous_monitoring(self, duration: int = 900, sweep_interval: int = 60):
        """Run continuous monitoring for specified duration"""
        logger.info(f"🔄 Starting continuous monitoring for {duration} seconds...")
        
        start_time = asyncio.get_event_loop().time()
        sweep_count = 0
        
        while (asyncio.get_event_loop().time() - start_time) < duration:
            sweep_count += 1
            logger.info(f"🔍 Starting stability sweep #{sweep_count}")
            
            sweep_results = await self.stability_sweep()
            
            # Log summary
            total_healthy = sweep_results["healthy"] + sweep_results["stabilized"]
            logger.info(f"✅ Sweep #{sweep_count}: {total_healthy}/{sweep_results['total_agents']} agents healthy")
            
            # Wait for next sweep
            await asyncio.sleep(sweep_interval)
        
        logger.info("✅ Continuous monitoring completed")
        return self.stability_report
    
    def generate_stability_report(self) -> Dict:
        """Generate final stability report"""
        agents = self.get_agent_containers()
        
        # Get final status of all agents
        final_status = {"healthy": [], "unhealthy": [], "problematic": []}
        
        for agent_name in agents:
            if agent_name in self.problematic_agents:
                final_status["problematic"].append(agent_name)
            else:
                health_info = self.analyze_container_health(agent_name)
                if health_info["healthy"]:
                    final_status["healthy"].append(agent_name)
                else:
                    final_status["unhealthy"].append(agent_name)
        
        self.stability_report.update({
            "end_time": datetime.now().isoformat(),
            "final_status": final_status,
            "total_agents": len(agents),
            "healthy_count": len(final_status["healthy"]),
            "unhealthy_count": len(final_status["unhealthy"]),
            "problematic_count": len(final_status["problematic"]),
            "stability_rate": (len(final_status["healthy"]) / len(agents)) * 100 if agents else 0
        })
        
        return self.stability_report

async def main():
    """Main stability management"""
    manager = AgentStabilityManager()
    
    try:
        logger.info("🤖 SutazAI Agent Stability Manager Starting...")
        
        # Run continuous monitoring for 15 minutes
        await manager.continuous_monitoring(duration=900, sweep_interval=60)
        
        # Generate final report
        report = manager.generate_stability_report()
        
        # Save report
        report_path = Path("/opt/sutazaiapp/logs/agent-stability-report.json")
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("=" * 60)
        print("AGENT STABILITY MANAGEMENT SUMMARY")
        print("=" * 60)
        print(f"Total Agents: {report['total_agents']}")
        print(f"Healthy Agents: {report['healthy_count']}")
        print(f"Unhealthy Agents: {report['unhealthy_count']}")
        print(f"Problematic Agents: {report['problematic_count']}")
        print(f"Stability Rate: {report['stability_rate']:.1f}%")
        print(f"Actions Taken: {len(report['actions_taken'])}")
        print(f"Report saved: {report_path}")
        
        if report['final_status']['healthy']:
            print("\n✅ Healthy Agents:")
            for agent in sorted(report['final_status']['healthy']):
                print(f"  - {agent}")
        
        if report['final_status']['problematic']:
            print("\n❌ Problematic Agents:")
            for agent in sorted(report['final_status']['problematic']):
                print(f"  - {agent}")
        
        return 0 if report['healthy_count'] > 0 else 1
        
    except Exception as e:
        logger.error(f"❌ Stability management failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))