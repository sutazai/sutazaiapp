#!/usr/bin/env python3
"""
SutazAI Startup Sequence Optimizer
Optimizes the startup sequence for 69 agents to reduce startup time by 50%
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import subprocess
import sys
import time
import yaml
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import docker
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/startup_optimizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ServiceConfig:
    """Configuration for a service/agent"""
    name: str
    dependencies: List[str]
    startup_time: float
    memory_required: int
    cpu_required: float
    priority: int  # 1=critical, 2=important, 3=optional
    parallel_group: Optional[str] = None
    health_check_timeout: int = 30
    can_delay_start: bool = False
    
@dataclass
class StartupGroup:
    """Group of services that can start in parallel"""
    name: str
    services: List[str]
    max_parallelism: int
    total_memory: int
    total_cpu: float
    dependencies: Set[str]

class SystemResourceMonitor:
    """Monitor system resources during startup"""
    
    def __init__(self):
        self.cpu_threshold = 80.0  # Max CPU utilization %
        self.memory_threshold = 85.0  # Max memory utilization %
        self.monitoring = False
        
    async def monitor_resources(self):
        """Monitor system resources"""
        while self.monitoring:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            logger.debug(f"CPU: {cpu_percent}%, Memory: {memory_percent}%")
            
            if cpu_percent > self.cpu_threshold or memory_percent > self.memory_threshold:
                logger.warning(f"High resource usage - CPU: {cpu_percent}%, Memory: {memory_percent}%")
                await asyncio.sleep(2)  # Throttle when resources are high
            else:
                await asyncio.sleep(0.5)
    
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False

class StartupOptimizer:
    """Optimizes the startup sequence for SutazAI agents"""
    
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.docker_client = docker.from_env()
        self.resource_monitor = SystemResourceMonitor()
        self.service_configs = {}
        self.startup_groups = []
        self.startup_times = {}
        
        # System capabilities
        self.max_cpu_cores = psutil.cpu_count()
        self.max_memory_gb = psutil.virtual_memory().total // (1024**3)
        self.max_parallel_services = min(self.max_cpu_cores * 2, 20)
        
        logger.info(f"System: {self.max_cpu_cores} cores, {self.max_memory_gb}GB RAM")
        
    def load_service_configurations(self):
        """Load service configurations from docker-compose.yml"""
        compose_file = self.project_root / "docker-compose.yml"
        
        if not compose_file.exists():
            raise FileNotFoundError(f"Docker compose file not found: {compose_file}")
        
        with open(compose_file, 'r') as f:
            compose_data = yaml.safe_load(f)
        
        services = compose_data.get('services', {})
        
        # Define service categories and priorities
        critical_services = [
            'postgres', 'redis', 'neo4j'  # Core databases
        ]
        
        important_services = [
            'ollama', 'chromadb', 'qdrant', 'faiss',  # AI infrastructure
            'backend', 'frontend'  # Core application
        ]
        
        # AI agents - can be started in parallel groups
        ai_agents = [
            name for name in services.keys() 
            if name not in critical_services + important_services
        ]
        
        for service_name, service_config in services.items():
            # Determine priority
            if service_name in critical_services:
                priority = 1
            elif service_name in important_services:
                priority = 2
            else:
                priority = 3
            
            # Extract dependencies
            depends_on = service_config.get('depends_on', [])
            if isinstance(depends_on, dict):
                dependencies = list(depends_on.keys())
            else:
                dependencies = depends_on if isinstance(depends_on, list) else []
            
            # Estimate resource requirements
            deploy_config = service_config.get('deploy', {})
            resources = deploy_config.get('resources', {})
            limits = resources.get('limits', {})
            
            memory_required = self._parse_memory(limits.get('memory', '512M'))
            cpu_required = float(limits.get('cpus', '0.5'))
            
            # Estimate startup time based on service type
            startup_time = self._estimate_startup_time(service_name, priority)
            
            # Determine if service can delay start
            can_delay_start = priority == 3  # AI agents can delay start
            
            self.service_configs[service_name] = ServiceConfig(
                name=service_name,
                dependencies=dependencies,
                startup_time=startup_time,
                memory_required=memory_required,
                cpu_required=cpu_required,
                priority=priority,
                can_delay_start=can_delay_start,
                health_check_timeout=60 if priority == 1 else 30
            )
        
        logger.info(f"Loaded {len(self.service_configs)} service configurations")
    
    def _parse_memory(self, memory_str: str) -> int:
        """Parse memory string to MB"""
        if isinstance(memory_str, (int, float)):
            return int(memory_str)
        
        memory_str = str(memory_str).upper()
        if memory_str.endswith('G'):
            return int(float(memory_str[:-1]) * 1024)
        elif memory_str.endswith('M'):
            return int(float(memory_str[:-1]))
        elif memory_str.endswith('K'):
            return int(float(memory_str[:-1]) / 1024)
        else:
            return 512  # Default 512MB
    
    def _estimate_startup_time(self, service_name: str, priority: int) -> float:
        """Estimate startup time for a service"""
        base_times = {
            1: 15.0,  # Critical services - databases
            2: 10.0,  # Important services - AI infrastructure
            3: 5.0    # AI agents
        }
        
        # Special cases
        if service_name == 'ollama':
            return 30.0  # Ollama takes longer to start
        elif service_name == 'neo4j':
            return 25.0  # Neo4j is slow to start
        elif service_name in ['postgres', 'redis']:
            return 10.0  # Databases are reasonably fast
        elif 'ai-' in service_name or 'agent' in service_name:
            return 8.0   # AI agents are typically fast
        
        return base_times.get(priority, 10.0)
    
    def create_dependency_graph(self) -> Dict[str, Set[str]]:
        """Create dependency graph for topological sorting"""
        graph = defaultdict(set)
        
        for service_name, config in self.service_configs.items():
            for dependency in config.dependencies:
                if dependency in self.service_configs:
                    graph[dependency].add(service_name)
        
        return dict(graph)
    
    def topological_sort(self) -> List[List[str]]:
        """Perform topological sort to determine startup order"""
        # Calculate in-degrees
        in_degree = defaultdict(int)
        graph = self.create_dependency_graph()
        
        for service in self.service_configs:
            in_degree[service] = len(self.service_configs[service].dependencies)
        
        # Initialize queue with services that have no dependencies
        queue = deque([service for service, degree in in_degree.items() if degree == 0])
        levels = []
        
        while queue:
            current_level = []
            next_queue = deque()
            
            # Process all services at current level
            while queue:
                service = queue.popleft()
                current_level.append(service)
                
                # Update in-degrees of dependent services
                for dependent in graph.get(service, []):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_queue.append(dependent)
            
            if current_level:
                levels.append(current_level)
                queue = next_queue
        
        logger.info(f"Topological sort created {len(levels)} startup levels")
        return levels
    
    def optimize_startup_groups(self, levels: List[List[str]]) -> List[StartupGroup]:
        """Optimize services into parallel startup groups"""
        optimized_groups = []
        
        for level_idx, level_services in enumerate(levels):
            # Sort services by priority and resource requirements
            level_services.sort(key=lambda s: (
                self.service_configs[s].priority,
                -self.service_configs[s].memory_required
            ))
            
            # Group services by priority and resource constraints
            priority_groups = defaultdict(list)
            for service in level_services:
                priority = self.service_configs[service].priority
                priority_groups[priority].append(service)
            
            # Create startup groups for each priority
            for priority, services in priority_groups.items():
                if not services:
                    continue
                
                # Calculate optimal parallelism based on system resources
                total_memory = sum(self.service_configs[s].memory_required for s in services)
                total_cpu = sum(self.service_configs[s].cpu_required for s in services)
                
                # Determine max parallelism based on resources and priority
                if priority == 1:  # Critical services
                    max_parallel = min(len(services), 3)  # Conservative for critical
                elif priority == 2:  # Important services
                    max_parallel = min(len(services), 5)
                else:  # AI agents
                    max_parallel = min(len(services), self.max_parallel_services)
                
                # Adjust for resource constraints
                memory_limit = self.max_memory_gb * 1024 * 0.8  # Use 80% of available memory
                if total_memory > memory_limit:
                    memory_ratio = memory_limit / total_memory
                    max_parallel = max(1, int(max_parallel * memory_ratio))
                
                group_name = f"level_{level_idx}_priority_{priority}"
                
                # Split large groups into smaller chunks
                if len(services) > max_parallel:
                    chunks = [services[i:i + max_parallel] for i in range(0, len(services), max_parallel)]
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_name = f"{group_name}_chunk_{chunk_idx}"
                        optimized_groups.append(StartupGroup(
                            name=chunk_name,
                            services=chunk,
                            max_parallelism=len(chunk),
                            total_memory=sum(self.service_configs[s].memory_required for s in chunk),
                            total_cpu=sum(self.service_configs[s].cpu_required for s in chunk),
                            dependencies=set()
                        ))
                else:
                    optimized_groups.append(StartupGroup(
                        name=group_name,
                        services=services,
                        max_parallelism=max_parallel,
                        total_memory=total_memory,
                        total_cpu=total_cpu,
                        dependencies=set()
                    ))
        
        self.startup_groups = optimized_groups
        logger.info(f"Created {len(optimized_groups)} optimized startup groups")
        return optimized_groups
    
    async def start_service(self, service_name: str, semaphore: asyncio.Semaphore) -> bool:
        """Start a single service"""
        async with semaphore:
            start_time = time.time()
            config = self.service_configs[service_name]
            
            try:
                logger.info(f"Starting service: {service_name}")
                
                # Start the service using docker-compose
                cmd = [
                    'docker', 'compose', 
                    '-f', str(self.project_root / 'docker-compose.yml'),
                    'up', '-d', service_name
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.project_root
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    logger.error(f"Failed to start {service_name}: {stderr.decode()}")
                    return False
                
                # Wait for health check
                if await self.wait_for_service_health(service_name, config.health_check_timeout):
                    elapsed = time.time() - start_time
                    self.startup_times[service_name] = elapsed
                    logger.info(f"Service {service_name} started in {elapsed:.2f}s")
                    return True
                else:
                    logger.warning(f"Service {service_name} started but failed health check")
                    return False
                    
            except Exception as e:
                logger.error(f"Error starting service {service_name}: {e}")
                return False
    
    async def wait_for_service_health(self, service_name: str, timeout: int) -> bool:
        """Wait for service to become healthy"""
        start_time = time.time()
        container_name = f"sutazai-{service_name}"
        
        while time.time() - start_time < timeout:
            try:
                container = self.docker_client.containers.get(container_name)
                
                # Check if container is running
                if container.status != 'running':
                    await asyncio.sleep(1)
                    continue
                
                # Check health status if available
                health = container.attrs.get('State', {}).get('Health', {})
                if health:
                    health_status = health.get('Status', 'none')
                    if health_status == 'healthy':
                        return True
                    elif health_status == 'unhealthy':
                        logger.warning(f"Service {service_name} is unhealthy")
                        return False
                else:
                    # No health check defined, assume healthy if running
                    return True
                
                await asyncio.sleep(2)
                
            except docker.errors.NotFound:
                logger.debug(f"Container {container_name} not found yet")
                await asyncio.sleep(1)
            except Exception as e:
                logger.debug(f"Health check error for {service_name}: {e}")
                await asyncio.sleep(1)
        
        logger.warning(f"Health check timeout for service {service_name}")
        return False
    
    async def start_group(self, group: StartupGroup) -> Dict[str, bool]:
        """Start a group of services in parallel"""
        logger.info(f"Starting group {group.name} with {len(group.services)} services")
        
        # Create semaphore to limit parallelism
        semaphore = asyncio.Semaphore(min(group.max_parallelism, self.max_parallel_services))
        
        # Start all services in the group
        tasks = []
        for service_name in group.services:
            task = asyncio.create_task(
                self.start_service(service_name, semaphore),
                name=f"start_{service_name}"
            )
            tasks.append((service_name, task))
        
        # Wait for all services to complete
        results = {}
        for service_name, task in tasks:
            try:
                success = await task
                results[service_name] = success
            except Exception as e:
                logger.error(f"Failed to start {service_name}: {e}")
                results[service_name] = False
        
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Group {group.name} completed: {successful}/{len(group.services)} successful")
        
        return results
    
    async def start_all_services(self, enable_fast_mode: bool = True) -> Dict[str, bool]:
        """Start all services with optimized sequence"""
        logger.info("Starting optimized service startup sequence")
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        monitor_task = asyncio.create_task(self.resource_monitor.monitor_resources())
        
        total_start_time = time.time()
        all_results = {}
        
        try:
            # Fast mode: Start critical services, then start others in background
            if enable_fast_mode:
                await self.start_fast_mode()
            else:
                # Normal mode: Start all groups sequentially
                for group in self.startup_groups:
                    group_results = await self.start_group(group)
                    all_results.update(group_results)
                    
                    # Brief pause between groups for resource management
                    await asyncio.sleep(2)
        
        finally:
            # Stop resource monitoring
            self.resource_monitor.stop_monitoring()
            monitor_task.cancel()
            
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        
        total_time = time.time() - total_start_time
        successful_services = sum(1 for success in all_results.values() if success)
        
        logger.info(f"Startup completed in {total_time:.2f}s")
        logger.info(f"Successfully started {successful_services}/{len(all_results)} services")
        
        return all_results
    
    async def start_fast_mode(self):
        """Fast startup mode - prioritize critical services, delay optional ones"""
        logger.info("Starting FAST MODE - Critical services first, others in background")
        
        # Group 1: Critical databases (must start first)
        critical_group = StartupGroup(
            name="critical_infrastructure",
            services=[s for s in self.service_configs.keys() 
                     if self.service_configs[s].priority == 1],
            max_parallelism=3,
            total_memory=0,
            total_cpu=0,
            dependencies=set()
        )
        
        if critical_group.services:
            await self.start_group(critical_group)
            await asyncio.sleep(5)  # Brief pause for stability
        
        # Group 2: Important services (AI infrastructure + core app)
        important_group = StartupGroup(
            name="important_services",
            services=[s for s in self.service_configs.keys() 
                     if self.service_configs[s].priority == 2],
            max_parallelism=5,
            total_memory=0,
            total_cpu=0,
            dependencies=set()
        )
        
        if important_group.services:
            await self.start_group(important_group)
            await asyncio.sleep(3)
        
        # Group 3: AI Agents (start in background batches)
        ai_agents = [s for s in self.service_configs.keys() 
                    if self.service_configs[s].priority == 3]
        
        if ai_agents:
            # Split AI agents into smaller batches for better resource management
            batch_size = min(10, self.max_parallel_services)
            batches = [ai_agents[i:i + batch_size] for i in range(0, len(ai_agents), batch_size)]
            
            # Start batches with overlap for faster overall startup
            batch_tasks = []
            for batch_idx, batch in enumerate(batches):
                task = asyncio.create_task(
                    self.start_group(StartupGroup(
                        name=f"ai_agents_batch_{batch_idx}",
                        services=batch,
                        max_parallelism=len(batch),
                        total_memory=0,
                        total_cpu=0,
                        dependencies=set()
                    )),
                    name=f"batch_{batch_idx}"
                )
                batch_tasks.append(task)
                
                # Stagger batch starts
                await asyncio.sleep(2)
            
            # Wait for all batches to complete
            await asyncio.gather(*batch_tasks, return_exceptions=True)
    
    def generate_startup_report(self, results: Dict[str, bool]):
        """Generate startup optimization report"""
        report_path = self.project_root / "logs" / f"startup_report_{int(time.time())}.json"
        
        successful_services = [s for s, success in results.items() if success]
        failed_services = [s for s, success in results.items() if not success]
        
        total_estimated_time = sum(config.startup_time for config in self.service_configs.values())
        actual_total_time = sum(self.startup_times.values())
        
        optimization_ratio = (total_estimated_time - actual_total_time) / total_estimated_time * 100
        
        report = {
            "timestamp": time.time(),
            "optimization_summary": {
                "total_services": len(self.service_configs),
                "successful_starts": len(successful_services),
                "failed_starts": len(failed_services),
                "estimated_sequential_time_s": total_estimated_time,
                "actual_parallel_time_s": actual_total_time,
                "optimization_percentage": optimization_ratio,
                "target_achieved": optimization_ratio >= 50.0
            },
            "startup_groups": [
                {
                    "name": group.name,
                    "services": group.services,
                    "max_parallelism": group.max_parallelism,
                    "total_memory_mb": group.total_memory,
                    "total_cpu": group.total_cpu
                }
                for group in self.startup_groups
            ],
            "service_timings": self.startup_times,
            "successful_services": successful_services,
            "failed_services": failed_services,
            "system_info": {
                "cpu_cores": self.max_cpu_cores,
                "memory_gb": self.max_memory_gb,
                "max_parallel_services": self.max_parallel_services
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Startup report generated: {report_path}")
        logger.info(f"Optimization achieved: {optimization_ratio:.1f}% time reduction")
        
        return report

async def main():
    """Main function to run startup optimization"""
    optimizer = StartupOptimizer()
    
    try:
        # Load service configurations
        optimizer.load_service_configurations()
        
        # Create dependency-aware startup order
        levels = optimizer.topological_sort()
        
        # Optimize into parallel groups
        optimizer.optimize_startup_groups(levels)
        
        # Start all services with optimization
        results = await optimizer.start_all_services(enable_fast_mode=True)
        
        # Generate report
        report = optimizer.generate_startup_report(results)
        
        # Check if we achieved the 50% reduction target
        if report["optimization_summary"]["target_achieved"]:
            logger.info("✅ TARGET ACHIEVED: 50%+ startup time reduction!")
        else:
            logger.warning(f"⚠️ Target not achieved: {report['optimization_summary']['optimization_percentage']:.1f}% reduction")
        
        return report
        
    except Exception as e:
        logger.error(f"Startup optimization failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())