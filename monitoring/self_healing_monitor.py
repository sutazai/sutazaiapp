#!/usr/bin/env python3
"""
Self-Healing Monitor for SutazAI System
Monitors critical services and automatically restarts failed containers
"""

import os
import json
import logging
import asyncio
import aiohttp
import docker
import psutil
from dataclasses import dataclass, asdict
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/self_healing_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ServiceHealth:
    name: str
    status: str
    healthy: bool
    last_check: datetime
    failure_count: int
    restart_count: int
    uptime: Optional[float] = None
    cpu_usage: Optional[float] = None
    memory_usage: Optional[int] = None

class SelfHealingMonitor:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.monitor_interval = int(os.getenv('MONITOR_INTERVAL', '30'))
        self.restart_threshold = int(os.getenv('RESTART_THRESHOLD', '3'))
        self.health_check_timeout = int(os.getenv('HEALTH_CHECK_TIMEOUT', '10'))
        
        # Critical services to monitor
        self.critical_services = {
            'sutazai-postgres': {
                'health_url': None,
                'health_command': 'pg_isready -U sutazai -d sutazai',
                'restart_delay': 10,
                'max_restarts': 5
            },
            'sutazai-redis': {
                'health_url': None,
                'health_command': 'redis-cli ping',
                'restart_delay': 5,
                'max_restarts': 10
            },
            'sutazai-neo4j': {
                'health_url': 'http://localhost:7474/db/data/',
                'health_command': None,
                'restart_delay': 15,
                'max_restarts': 3
            },
            'sutazai-ollama': {
                'health_url': 'http://localhost:11434/api/version',
                'health_command': 'ollama list',
                'restart_delay': 30,
                'max_restarts': 3
            }
        }
        
        # Service health tracking
        self.service_health: Dict[str, ServiceHealth] = {}
        self.circuit_breaker_state = {}
        self.last_restart_times = {}
        
        # Initialize service health tracking
        for service_name in self.critical_services.keys():
            self.service_health[service_name] = ServiceHealth(
                name=service_name,
                status='unknown',
                healthy=False,
                last_check=datetime.now(),
                failure_count=0,
                restart_count=0
            )
            self.circuit_breaker_state[service_name] = 'closed'  # closed, open, half-open

    async def check_service_health(self, service_name: str, config: Dict) -> bool:
        """Check health of a specific service"""
        try:
            container = self.docker_client.containers.get(service_name)
            
            # Check if container is running
            if container.status != 'running':
                logger.warning(f"Container {service_name} is not running: {container.status}")
                return False
            
            # Get container stats
            stats = container.stats(stream=False)
            cpu_usage = self._calculate_cpu_usage(stats)
            memory_usage = stats['memory_stats'].get('usage', 0)
            
            # Update service health with stats
            self.service_health[service_name].cpu_usage = cpu_usage
            self.service_health[service_name].memory_usage = memory_usage
            
            # Check health URL if available
            if config.get('health_url'):
                try:
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.health_check_timeout)) as session:
                        async with session.get(config['health_url']) as response:
                            if response.status != 200:
                                logger.warning(f"Health check failed for {service_name}: HTTP {response.status}")
                                return False
                except Exception as e:
                    logger.warning(f"Health check connection failed for {service_name}: {e}")
                    return False
            
            # Execute health command if available
            if config.get('health_command'):
                try:
                    result = container.exec_run(config['health_command'], timeout=self.health_check_timeout)
                    if result.exit_code != 0:
                        logger.warning(f"Health command failed for {service_name}: {result.output.decode()}")
                        return False
                except Exception as e:
                    logger.warning(f"Health command execution failed for {service_name}: {e}")
                    return False
            
            return True
            
        except docker.errors.NotFound:
            logger.error(f"Container {service_name} not found")
            return False
        except Exception as e:
            logger.error(f"Error checking health for {service_name}: {e}")
            return False

    def _calculate_cpu_usage(self, stats: Dict) -> float:
        """Calculate CPU usage percentage from container stats"""
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            if system_delta > 0 and cpu_delta > 0:
                cpu_usage = (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100.0
                return round(cpu_usage, 2)
        except (KeyError, ZeroDivisionError):
            pass
        return 0.0

    async def restart_service(self, service_name: str) -> bool:
        """Restart a failed service with exponential backoff"""
        try:
            config = self.critical_services[service_name]
            current_time = datetime.now()
            
            # Check if we're within restart limits
            if self.service_health[service_name].restart_count >= config['max_restarts']:
                logger.error(f"Max restart attempts reached for {service_name}")
                self.circuit_breaker_state[service_name] = 'open'
                return False
            
            # Implement restart delay
            last_restart = self.last_restart_times.get(service_name)
            if last_restart:
                time_since_restart = (current_time - last_restart).total_seconds()
                if time_since_restart < config['restart_delay']:
                    logger.info(f"Waiting for restart delay for {service_name}")
                    return False
            
            logger.info(f"Attempting to restart {service_name}")
            
            container = self.docker_client.containers.get(service_name)
            container.restart(timeout=30)
            
            # Update tracking
            self.service_health[service_name].restart_count += 1
            self.last_restart_times[service_name] = current_time
            
            # Wait for container to start
            await asyncio.sleep(config['restart_delay'])
            
            # Verify restart was successful
            container.reload()
            if container.status == 'running':
                logger.info(f"Successfully restarted {service_name}")
                self.service_health[service_name].failure_count = 0
                return True
            else:
                logger.error(f"Failed to restart {service_name}: {container.status}")
                return False
                
        except Exception as e:
            logger.error(f"Error restarting {service_name}: {e}")
            return False

    async def manage_circuit_breaker(self, service_name: str, healthy: bool):
        """Manage circuit breaker state for service"""
        current_state = self.circuit_breaker_state[service_name]
        
        if healthy:
            if current_state == 'half-open':
                logger.info(f"Service {service_name} recovered, closing circuit breaker")
                self.circuit_breaker_state[service_name] = 'closed'
                self.service_health[service_name].failure_count = 0
        else:
            failure_count = self.service_health[service_name].failure_count
            
            if current_state == 'closed' and failure_count >= self.restart_threshold:
                logger.warning(f"Opening circuit breaker for {service_name}")
                self.circuit_breaker_state[service_name] = 'open'
            elif current_state == 'open':
                # Check if we should try half-open
                last_check = self.service_health[service_name].last_check
                if (datetime.now() - last_check).total_seconds() > 300:  # 5 minutes
                    logger.info(f"Trying half-open state for {service_name}")
                    self.circuit_breaker_state[service_name] = 'half-open'

    async def monitor_services(self):
        """Main monitoring loop"""
        logger.info("Starting self-healing monitor")
        
        while True:
            try:
                for service_name, config in self.critical_services.items():
                    current_time = datetime.now()
                    
                    # Skip if circuit breaker is open
                    if self.circuit_breaker_state[service_name] == 'open':
                        # Check if we should transition to half-open
                        await self.manage_circuit_breaker(service_name, False)
                        continue
                    
                    # Check service health
                    healthy = await self.check_service_health(service_name, config)
                    
                    # Update service health record
                    health_record = self.service_health[service_name]
                    health_record.healthy = healthy
                    health_record.last_check = current_time
                    health_record.status = 'healthy' if healthy else 'unhealthy'
                    
                    if not healthy:
                        health_record.failure_count += 1
                        logger.warning(f"Health check failed for {service_name} (failures: {health_record.failure_count})")
                        
                        # Attempt restart if threshold is reached
                        if (health_record.failure_count >= self.restart_threshold and 
                            self.circuit_breaker_state[service_name] != 'open'):
                            
                            restart_success = await self.restart_service(service_name)
                            if not restart_success:
                                await self.manage_circuit_breaker(service_name, False)
                    else:
                        if health_record.failure_count > 0:
                            logger.info(f"Service {service_name} recovered")
                        health_record.failure_count = 0
                        await self.manage_circuit_breaker(service_name, True)
                
                # Save health report
                await self.save_health_report()
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)

    async def save_health_report(self):
        """Save current health status to file"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'services': {name: asdict(health) for name, health in self.service_health.items()},
                'circuit_breakers': self.circuit_breaker_state,
                'system_stats': {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_percent': psutil.disk_usage('/').percent
                }
            }
            
            # Convert datetime objects to strings
            for service_data in report['services'].values():
                if 'last_check' in service_data:
                    service_data['last_check'] = service_data['last_check'].isoformat()
            
            with open('/app/logs/self_healing_report.json', 'w') as f:
                json.dump(report, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving health report: {e}")

    async def health_endpoint(self, request):
        """Health endpoint for the monitor itself"""
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services_monitored': len(self.critical_services),
            'circuit_breakers': self.circuit_breaker_state
        })

def signal_handler(signum, frame):
    logger.info("Received shutdown signal, stopping monitor")
    sys.exit(0)

async def main():
    monitor = SelfHealingMonitor()
    
    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start monitoring
    await monitor.monitor_services()

if __name__ == '__main__':
    asyncio.run(main())