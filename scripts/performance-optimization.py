#!/usr/bin/env python3
"""
SutazAI System Performance Optimization Script
Implements dynamic resource management and container health fixes
"""

import docker
import psutil
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set
from dataclasses import dataclass
from collections import deque
import asyncio
import aiohttp
import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sutazai-optimizer')

@dataclass
class AgentStatus:
    name: str
    container_id: str
    status: str
    cpu_percent: float
    memory_usage: int
    memory_limit: int
    last_activity: datetime
    health_status: str
    port: int

@dataclass
class SystemMetrics:
    cpu_percent: float
    memory_percent: float
    memory_available: int
    load_average: float
    container_count: int
    healthy_containers: int
    active_agents: int

class AgentResourceManager:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.redis_client = redis.Redis(host='localhost', port=10001, db=0)
        self.active_agents: Set[str] = set()
        self.standby_agents: Set[str] = set()
        self.agent_queue: deque = deque()
        self.max_active_agents = 20  # Based on resource analysis
        self.max_memory_percent = 80
        self.max_cpu_load = 8.0
        
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system resource metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        load_avg = psutil.getloadavg()[0]
        
        containers = self.docker_client.containers.list()
        healthy_count = sum(1 for c in containers if c.attrs.get('State', {}).get('Health', {}).get('Status') == 'healthy')
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available=memory.available,
            load_average=load_avg,
            container_count=len(containers),
            healthy_containers=healthy_count,
            active_agents=len(self.active_agents)
        )
    
    def get_agent_status(self) -> List[AgentStatus]:
        """Get status of all AI agent containers"""
        agents = []
        containers = self.docker_client.containers.list(all=True)
        
        for container in containers:
            if 'sutazai-' in container.name and 'phase3' in container.name:
                try:
                    stats = container.stats(stream=False)
                    
                    # Calculate CPU percentage
                    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                               stats['precpu_stats']['cpu_usage']['total_usage']
                    system_cpu_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                      stats['precpu_stats']['system_cpu_usage']
                    cpu_percent = (cpu_delta / system_cpu_delta) * 100.0 if system_cpu_delta > 0 else 0
                    
                    # Get memory usage
                    memory_usage = stats['memory_stats'].get('usage', 0)
                    memory_limit = stats['memory_stats'].get('limit', 0)
                    
                    # Get port mapping
                    port_bindings = container.attrs.get('NetworkSettings', {}).get('Ports', {})
                    port = None
                    for port_key in port_bindings:
                        if port_bindings[port_key]:
                            port = int(port_bindings[port_key][0]['HostPort'])
                            break
                    
                    agent = AgentStatus(
                        name=container.name,
                        container_id=container.id,
                        status=container.status,
                        cpu_percent=cpu_percent,
                        memory_usage=memory_usage,
                        memory_limit=memory_limit,
                        last_activity=datetime.now(),
                        health_status=container.attrs.get('State', {}).get('Health', {}).get('Status', 'unknown'),
                        port=port
                    )
                    agents.append(agent)
                    
                except Exception as e:
                    logger.warning(f"Error getting stats for {container.name}: {e}")
                    
        return agents
    
    def optimize_container_resources(self):
        """Optimize Docker container resource allocation"""
        logger.info("Starting container resource optimization...")
        
        try:
            # Clean up unused Docker resources
            logger.info("Cleaning up unused Docker resources...")
            pruned = self.docker_client.images.prune(filters={'dangling': True})
            logger.info(f"Removed {len(pruned.get('ImagesDeleted', []))} unused images")
            
            pruned_containers = self.docker_client.containers.prune()
            logger.info(f"Removed {pruned_containers['ContainersDeleted']} stopped containers")
            
            pruned_volumes = self.docker_client.volumes.prune()
            logger.info(f"Removed {len(pruned_volumes.get('VolumesDeleted', []))} unused volumes")
            
        except Exception as e:
            logger.error(f"Error during Docker cleanup: {e}")
    
    def fix_unhealthy_containers(self):
        """Attempt to fix unhealthy containers"""
        logger.info("Checking and fixing unhealthy containers...")
        
        containers = self.docker_client.containers.list(all=True)
        unhealthy_containers = []
        
        for container in containers:
            health_status = container.attrs.get('State', {}).get('Health', {}).get('Status')
            if health_status == 'unhealthy' or container.status == 'restarting':
                unhealthy_containers.append(container)
        
        logger.info(f"Found {len(unhealthy_containers)} unhealthy containers")
        
        for container in unhealthy_containers:
            try:
                logger.info(f"Attempting to fix {container.name}")
                
                # Try restarting the container
                container.restart(timeout=30)
                time.sleep(10)  # Wait for restart
                
                # Check if it's healthy now
                container.reload()
                new_status = container.attrs.get('State', {}).get('Health', {}).get('Status')
                logger.info(f"{container.name} new status: {new_status}")
                
            except Exception as e:
                logger.error(f"Error fixing {container.name}: {e}")
    
    def implement_agent_queue_system(self):
        """Implement queue-based agent activation system"""
        agents = self.get_agent_status()
        system_metrics = self.get_system_metrics()
        
        # Determine if we need to scale up or down
        if (system_metrics.memory_percent > self.max_memory_percent or 
            system_metrics.load_average > self.max_cpu_load):
            # Scale down - move agents to standby
            self.scale_down_agents(agents)
        elif (system_metrics.memory_percent < 60 and 
              system_metrics.load_average < 4.0 and 
              len(self.active_agents) < self.max_active_agents):
            # Scale up - activate queued agents
            self.scale_up_agents(agents)
    
    def scale_down_agents(self, agents: List[AgentStatus]):
        """Scale down least active agents"""
        logger.info("System under pressure - scaling down agents")
        
        # Sort agents by activity (CPU usage as proxy)
        active_agents = [a for a in agents if a.name in self.active_agents]
        active_agents.sort(key=lambda x: x.cpu_percent)
        
        # Move least active agents to standby
        agents_to_standby = active_agents[:len(active_agents)//4]  # 25% to standby
        
        for agent in agents_to_standby:
            self.move_agent_to_standby(agent)
    
    def scale_up_agents(self, agents: List[AgentStatus]):
        """Scale up agents from standby queue"""
        logger.info("System has capacity - scaling up agents")
        
        # Get standby agents
        standby_agents = [a for a in agents if a.name in self.standby_agents]
        
        # Activate agents up to max limit
        available_slots = self.max_active_agents - len(self.active_agents)
        agents_to_activate = standby_agents[:available_slots]
        
        for agent in agents_to_activate:
            self.activate_agent(agent)
    
    def move_agent_to_standby(self, agent: AgentStatus):
        """Move agent to standby mode (reduce resources)"""
        try:
            container = self.docker_client.containers.get(agent.container_id)
            
            # Update container resource limits (would require container recreation)
            logger.info(f"Moving {agent.name} to standby mode")
            
            self.active_agents.discard(agent.name)
            self.standby_agents.add(agent.name)
            
            # Store in Redis for persistence
            self.redis_client.sadd('standby_agents', agent.name)
            self.redis_client.srem('active_agents', agent.name)
            
        except Exception as e:
            logger.error(f"Error moving {agent.name} to standby: {e}")
    
    def activate_agent(self, agent: AgentStatus):
        """Activate agent from standby mode"""
        try:
            logger.info(f"Activating {agent.name}")
            
            self.standby_agents.discard(agent.name)
            self.active_agents.add(agent.name)
            
            # Store in Redis for persistence
            self.redis_client.sadd('active_agents', agent.name)
            self.redis_client.srem('standby_agents', agent.name)
            
        except Exception as e:
            logger.error(f"Error activating {agent.name}: {e}")
    
    def monitor_and_report(self):
        """Generate monitoring report"""
        system_metrics = self.get_system_metrics()
        agents = self.get_agent_status()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': {
                'cpu_percent': system_metrics.cpu_percent,
                'memory_percent': system_metrics.memory_percent,
                'memory_available_gb': system_metrics.memory_available / (1024**3),
                'load_average': system_metrics.load_average,
                'container_count': system_metrics.container_count,
                'healthy_containers': system_metrics.healthy_containers,
                'container_health_rate': (system_metrics.healthy_containers / system_metrics.container_count * 100) if system_metrics.container_count else 0
            },
            'agent_metrics': {
                'total_agents': len(agents),
                'active_agents': len(self.active_agents),
                'standby_agents': len(self.standby_agents),
                'healthy_agents': sum(1 for a in agents if a.health_status == 'healthy'),
                'unhealthy_agents': sum(1 for a in agents if a.health_status == 'unhealthy'),
                'restarting_agents': sum(1 for a in agents if a.status == 'restarting')
            },
            'resource_utilization': {
                'total_agent_memory_mb': sum(a.memory_usage for a in agents) / (1024*1024),
                'average_agent_cpu': sum(a.cpu_percent for a in agents) / len(agents) if agents else 0
            }
        }
        
        # Save report
        with open(f'/opt/sutazaiapp/logs/performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"System Health: {report['system_metrics']['container_health_rate']:.1f}% containers healthy")
        logger.info(f"Memory: {report['system_metrics']['memory_percent']:.1f}% used")
        logger.info(f"CPU Load: {report['system_metrics']['load_average']:.2f}")
        logger.info(f"Active Agents: {report['agent_metrics']['active_agents']}/{report['agent_metrics']['total_agents']}")
        
        return report

class PerformanceOptimizer:
    def __init__(self):
        self.resource_manager = AgentResourceManager()
        self.optimization_interval = 60  # seconds
        
    async def run_optimization_cycle(self):
        """Run one optimization cycle"""
        logger.info("Starting optimization cycle...")
        
        # 1. Fix unhealthy containers
        self.resource_manager.fix_unhealthy_containers()
        
        # 2. Optimize resources
        self.resource_manager.optimize_container_resources()
        
        # 3. Implement queue system
        self.resource_manager.implement_agent_queue_system()
        
        # 4. Generate report
        report = self.resource_manager.monitor_and_report()
        
        logger.info("Optimization cycle completed")
        return report
    
    async def run_continuous_optimization(self):
        """Run continuous optimization loop"""
        logger.info("Starting continuous performance optimization...")
        
        while True:
            try:
                await self.run_optimization_cycle()
                await asyncio.sleep(self.optimization_interval)
                
            except KeyboardInterrupt:
                logger.info("Optimization stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in optimization cycle: {e}")
                await asyncio.sleep(30)  # Wait before retrying

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SutazAI Performance Optimizer')
    parser.add_argument('--mode', choices=['once', 'continuous'], default='once',
                       help='Run optimization once or continuously')
    parser.add_argument('--fix-containers', action='store_true',
                       help='Fix unhealthy containers')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up Docker resources')
    parser.add_argument('--report', action='store_true',
                       help='Generate performance report')
    
    args = parser.parse_args()
    
    optimizer = PerformanceOptimizer()
    
    if args.mode == 'once':
        asyncio.run(optimizer.run_optimization_cycle())
    else:
        asyncio.run(optimizer.run_continuous_optimization())

if __name__ == '__main__':
    main()