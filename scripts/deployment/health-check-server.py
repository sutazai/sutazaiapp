#!/usr/bin/env python3
"""
Health Check Server for SutazAI Services
Provides health check endpoints for all services in the mesh
"""

import asyncio
import aiohttp
import json
import logging
import os
import sys
import time
from aiohttp import web, ClientSession, ClientTimeout
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import redis
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class HealthStatus:
    """Health status information"""
    service: str
    status: str  # healthy, unhealthy, warning
    timestamp: datetime
    response_time: float
    details: Dict[str, Any]
    dependencies: List[str]

class HealthCheckServer:
    """Health check server for service mesh monitoring"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.app = web.Application()
        self.services = {}
        self.health_cache = {}
        self.redis_client = None
        
        # Setup routes
        self.setup_routes()
        
    def setup_routes(self):
        """Setup HTTP routes"""
        self.app.router.add_get('/health', self.health_check_handler)
        self.app.router.add_get('/health/{service}', self.service_health_handler)
        self.app.router.add_get('/health/deep/{service}', self.deep_health_check_handler)
        self.app.router.add_get('/services', self.list_services_handler)
        self.app.router.add_get('/metrics', self.metrics_handler)
        self.app.router.add_post('/register', self.register_service_handler)
        self.app.router.add_delete('/register/{service}', self.deregister_service_handler)
    
    async def initialize(self):
        """Initialize health check server"""
        try:
            # Initialize Redis connection for caching
            redis_url = os.getenv('REDIS_URL', 'redis://redis:6379/0')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Load service configurations
            await self.load_service_configs()
            
            logger.info("Health check server initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize health check server: {e}")
            return False
    
    async def load_service_configs(self):
        """Load service configurations from Consul services config"""
        try:
            config_path = '/opt/sutazaiapp/config/consul/services.json'
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                for service in config.get('services', []):
                    self.services[service['name']] = {
                        'id': service['id'],
                        'address': service['address'],
                        'port': service['port'],
                        'tags': service.get('tags', []),
                        'health_check': service.get('check', {}),
                        'dependencies': self.get_service_dependencies(service['name'])
                    }
                
                logger.info(f"Loaded {len(self.services)} service configurations")
                
        except Exception as e:
            logger.error(f"Error loading service configs: {e}")
    
    def get_service_dependencies(self, service_name: str) -> List[str]:
        """Get service dependencies based on service type"""
        dependencies = {
            'backend': ['postgres', 'redis', 'ollama'],
            'frontend': ['backend'],
            'autogpt': ['ollama', 'backend'],
            'crewai': ['ollama', 'backend'],
            'letta': ['postgres', 'ollama', 'backend'],
            'aider': ['ollama'],
            'langflow': ['postgres', 'redis'],
            'flowise': [],
            'dify': ['postgres', 'redis', 'ollama'],
            'n8n': [],
            'mcp-server': ['postgres', 'redis', 'backend', 'ollama']
        }
        return dependencies.get(service_name, [])
    
    async def check_service_health(self, service_name: str) -> HealthStatus:
        """Check health of a specific service"""
        if service_name not in self.services:
            return HealthStatus(
                service=service_name,
                status='unknown',
                timestamp=datetime.now(),
                response_time=0.0,
                details={'error': 'Service not registered'},
                dependencies=[]
            )
        
        service_config = self.services[service_name]
        start_time = time.time()
        
        try:
            # Check if service responds to health check
            health_check = service_config.get('health_check', {})
            
            if 'http' in health_check:
                async with ClientSession(timeout=ClientTimeout(total=10)) as session:
                    async with session.get(health_check['http']) as response:
                        response_time = time.time() - start_time
                        
                        if response.status == 200:
                            # Check dependencies
                            dep_status = await self.check_dependencies(service_config['dependencies'])
                            
                            status = 'healthy' if dep_status['all_healthy'] else 'warning'
                            
                            return HealthStatus(
                                service=service_name,
                                status=status,
                                timestamp=datetime.now(),
                                response_time=response_time,
                                details={
                                    'http_status': response.status,
                                    'dependencies': dep_status
                                },
                                dependencies=service_config['dependencies']
                            )
                        else:
                            return HealthStatus(
                                service=service_name,
                                status='unhealthy',
                                timestamp=datetime.now(),
                                response_time=response_time,
                                details={'http_status': response.status},
                                dependencies=service_config['dependencies']
                            )
            
            elif 'tcp' in health_check:
                # TCP health check
                address = service_config['address']
                port = service_config['port']
                
                try:
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_connection(address, port),
                        timeout=5.0
                    )
                    writer.close()
                    await writer.wait_closed()
                    
                    response_time = time.time() - start_time
                    
                    return HealthStatus(
                        service=service_name,
                        status='healthy',
                        timestamp=datetime.now(),
                        response_time=response_time,
                        details={'tcp_check': 'passed'},
                        dependencies=service_config['dependencies']
                    )
                    
                except Exception as tcp_error:
                    return HealthStatus(
                        service=service_name,
                        status='unhealthy',
                        timestamp=datetime.now(),
                        response_time=time.time() - start_time,
                        details={'tcp_error': str(tcp_error)},
                        dependencies=service_config['dependencies']
                    )
            
            else:
                # No health check defined - assume healthy if registered
                return HealthStatus(
                    service=service_name,
                    status='healthy',
                    timestamp=datetime.now(),
                    response_time=0.0,
                    details={'check_type': 'registration_only'},
                    dependencies=service_config['dependencies']
                )
                
        except Exception as e:
            return HealthStatus(
                service=service_name,
                status='unhealthy',
                timestamp=datetime.now(),
                response_time=time.time() - start_time,
                details={'error': str(e)},
                dependencies=service_config['dependencies']
            )
    
    async def check_dependencies(self, dependencies: List[str]) -> Dict[str, Any]:
        """Check health of service dependencies"""
        dep_results = {}
        all_healthy = True
        
        for dep in dependencies:
            if dep in self.health_cache:
                # Use cached result if recent
                cached = self.health_cache[dep]
                if datetime.now() - cached.timestamp < timedelta(seconds=30):
                    dep_results[dep] = cached.status
                    if cached.status != 'healthy':
                        all_healthy = False
                    continue
            
            # Check dependency health
            dep_health = await self.check_service_health(dep)
            dep_results[dep] = dep_health.status
            self.health_cache[dep] = dep_health
            
            if dep_health.status != 'healthy':
                all_healthy = False
        
        return {
            'all_healthy': all_healthy,
            'details': dep_results
        }
    
    async def health_check_handler(self, request):
        """Main health check endpoint"""
        try:
            # Check system health
            system_health = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'timestamp': datetime.now().isoformat()
            }
            
            # Check all services
            service_statuses = {}
            overall_healthy = True
            
            for service_name in self.services.keys():
                health_status = await self.check_service_health(service_name)
                service_statuses[service_name] = asdict(health_status)
                service_statuses[service_name]['timestamp'] = health_status.timestamp.isoformat()
                
                if health_status.status != 'healthy':
                    overall_healthy = False
                
                # Cache the result
                self.health_cache[service_name] = health_status
            
            response_data = {
                'status': 'healthy' if overall_healthy else 'degraded',
                'timestamp': datetime.now().isoformat(),
                'system': system_health,
                'services': service_statuses,
                'summary': {
                    'total_services': len(self.services),
                    'healthy_services': sum(1 for s in service_statuses.values() if s['status'] == 'healthy'),
                    'unhealthy_services': sum(1 for s in service_statuses.values() if s['status'] == 'unhealthy'),
                    'warning_services': sum(1 for s in service_statuses.values() if s['status'] == 'warning')
                }
            }
            
            status_code = 200 if overall_healthy else 503
            return web.json_response(response_data, status=status_code)
            
        except Exception as e:
            logger.error(f"Error in health check handler: {e}")
            return web.json_response(
                {'status': 'error', 'message': str(e)}, 
                status=500
            )
    
    async def service_health_handler(self, request):
        """Individual service health check"""
        service_name = request.match_info['service']
        
        try:
            health_status = await self.check_service_health(service_name)
            response_data = asdict(health_status)
            response_data['timestamp'] = health_status.timestamp.isoformat()
            
            status_code = 200 if health_status.status == 'healthy' else 503
            return web.json_response(response_data, status=status_code)
            
        except Exception as e:
            logger.error(f"Error checking service {service_name}: {e}")
            return web.json_response(
                {'service': service_name, 'status': 'error', 'message': str(e)}, 
                status=500
            )
    
    async def deep_health_check_handler(self, request):
        """Deep health check including dependencies"""
        service_name = request.match_info['service']
        
        try:
            # Get service health
            health_status = await self.check_service_health(service_name)
            
            # Get dependency health details
            if health_status.dependencies:
                dep_details = {}
                for dep in health_status.dependencies:
                    dep_health = await self.check_service_health(dep)
                    dep_details[dep] = asdict(dep_health)
                    dep_details[dep]['timestamp'] = dep_health.timestamp.isoformat()
                
                response_data = asdict(health_status)
                response_data['timestamp'] = health_status.timestamp.isoformat()
                response_data['dependency_details'] = dep_details
            else:
                response_data = asdict(health_status)
                response_data['timestamp'] = health_status.timestamp.isoformat()
            
            status_code = 200 if health_status.status == 'healthy' else 503
            return web.json_response(response_data, status=status_code)
            
        except Exception as e:
            logger.error(f"Error in deep health check for {service_name}: {e}")
            return web.json_response(
                {'service': service_name, 'status': 'error', 'message': str(e)}, 
                status=500
            )
    
    async def list_services_handler(self, request):
        """List all registered services"""
        return web.json_response({
            'services': list(self.services.keys()),
            'count': len(self.services)
        })
    
    async def metrics_handler(self, request):
        """Prometheus-style metrics endpoint"""
        try:
            metrics = []
            
            # System metrics
            metrics.append(f'system_cpu_percent {psutil.cpu_percent()}')
            metrics.append(f'system_memory_percent {psutil.virtual_memory().percent}')
            metrics.append(f'system_disk_percent {psutil.disk_usage("/").percent}')
            
            # Service metrics
            for service_name in self.services.keys():
                health_status = await self.check_service_health(service_name)
                
                status_value = 1 if health_status.status == 'healthy' else 0
                metrics.append(f'service_health{{service="{service_name}"}} {status_value}')
                metrics.append(f'service_response_time{{service="{service_name}"}} {health_status.response_time}')
            
            return web.Response(text='\n'.join(metrics), content_type='text/plain')
            
        except Exception as e:
            logger.error(f"Error generating metrics: {e}")
            return web.Response(text='', status=500)
    
    async def register_service_handler(self, request):
        """Register a new service for health checking"""
        try:
            data = await request.json()
            service_name = data['name']
            
            self.services[service_name] = {
                'id': data.get('id', service_name),
                'address': data['address'],
                'port': data['port'],
                'tags': data.get('tags', []),
                'health_check': data.get('health_check', {}),
                'dependencies': data.get('dependencies', [])
            }
            
            logger.info(f"Registered service for health checking: {service_name}")
            return web.json_response({'status': 'registered', 'service': service_name})
            
        except Exception as e:
            logger.error(f"Error registering service: {e}")
            return web.json_response({'error': str(e)}, status=400)
    
    async def deregister_service_handler(self, request):
        """Deregister a service"""
        service_name = request.match_info['service']
        
        if service_name in self.services:
            del self.services[service_name]
            if service_name in self.health_cache:
                del self.health_cache[service_name]
            
            logger.info(f"Deregistered service: {service_name}")
            return web.json_response({'status': 'deregistered', 'service': service_name})
        else:
            return web.json_response({'error': 'Service not found'}, status=404)

async def main():
    """Main health check server"""
    port = int(os.getenv('HEALTH_CHECK_PORT', '8080'))
    server = HealthCheckServer(port=port)
    
    try:
        if not await server.initialize():
            logger.error("Failed to initialize health check server")
            sys.exit(1)
        
        # Start the web server
        logger.info(f"Starting health check server on port {port}")
        runner = web.AppRunner(server.app)
        await runner.setup()
        
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        
        logger.info(f"Health check server running on http://0.0.0.0:{port}")
        
        # Keep the server running
        while True:
            await asyncio.sleep(3600)  # Sleep for 1 hour
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())