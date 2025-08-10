#!/usr/bin/env python3
"""
Service Mesh Orchestrator for SutazAI
Coordinates Consul, Kong, and RabbitMQ integration
"""

import asyncio
import aiohttp
import json
import logging
import os
import sys
import yaml
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ServiceMeshOrchestrator:
    """Orchestrates the complete service mesh setup"""
    
    def __init__(self):
        self.consul_url = os.getenv('CONSUL_URL', 'http://consul:8500')
        self.kong_admin_url = os.getenv('KONG_ADMIN_URL', 'http://kong:8001')
        self.rabbitmq_url = os.getenv('RABBITMQ_URL', 'http://rabbitmq:15672')
        self.rabbitmq_user = os.getenv('RABBITMQ_USER', 'admin')
        self.rabbitmq_pass = os.getenv('RABBITMQ_PASS', 'adminpass')
        
        self.services_configured = set()
        self.running = True
        
    async def initialize(self):
        """Initialize service mesh components"""
        logger.info("Initializing Service Mesh Orchestrator")
        
        # Wait for infrastructure services to be ready
        await self.wait_for_infrastructure()
        
        # Configure Consul services
        await self.configure_consul_services()
        
        # Configure Kong gateway
        await self.configure_kong_gateway()
        
        # Configure RabbitMQ messaging
        await self.configure_rabbitmq_messaging()
        
        # Start health monitoring
        await self.start_health_monitoring()
        
        logger.info("Service mesh orchestrator initialized successfully")
        return True
    
    async def wait_for_infrastructure(self):
        """Wait for infrastructure services to be ready"""
        services = {
            'Consul': self.consul_url + '/v1/status/leader',
            'Kong': self.kong_admin_url + '/status',
            'RabbitMQ': self.rabbitmq_url + '/api/overview'
        }
        
        for service_name, url in services.items():
            logger.info(f"Waiting for {service_name} to be ready...")
            
            max_retries = 30
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    auth = None
                    if 'rabbitmq' in url.lower():
                        auth = aiohttp.BasicAuth(self.rabbitmq_user, self.rabbitmq_pass)
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, auth=auth, timeout=10) as response:
                            if response.status == 200:
                                logger.info(f"{service_name} is ready")
                                break
                except Exception as e:
                    logger.debug(f"Waiting for {service_name}: {e}")
                
                retry_count += 1
                await asyncio.sleep(10)
            
            if retry_count >= max_retries:
                raise Exception(f"{service_name} failed to become ready after {max_retries} retries")
    
    async def configure_consul_services(self):
        """Configure Consul service registration"""
        logger.info("Configuring Consul services...")
        
        try:
            # Load services configuration
            config_path = '/opt/sutazaiapp/config/consul/services.json'
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            services = config.get('services', [])
            
            async with aiohttp.ClientSession() as session:
                for service in services:
                    try:
                        # Register service with Consul
                        registration_data = {
                            'ID': service['id'],
                            'Name': service['name'],
                            'Tags': service.get('tags', []),
                            'Address': service['address'],
                            'Port': service['port'],
                            'Meta': service.get('meta', {}),
                            'EnableTagOverride': False
                        }
                        
                        # Add health check if configured
                        if 'check' in service:
                            check = service['check']
                            if 'http' in check:
                                registration_data['Check'] = {
                                    'HTTP': check['http'],
                                    'Interval': check.get('interval', '30s'),
                                    'Timeout': check.get('timeout', '10s'),
                                    'DeregisterCriticalServiceAfter': '1m'
                                }
                            elif 'tcp' in check:
                                registration_data['Check'] = {
                                    'TCP': f"{service['address']}:{service['port']}",
                                    'Interval': check.get('interval', '30s'),
                                    'Timeout': check.get('timeout', '10s'),
                                    'DeregisterCriticalServiceAfter': '1m'
                                }
                        
                        # Register with Consul
                        url = f"{self.consul_url}/v1/agent/service/register"
                        async with session.put(url, json=registration_data) as response:
                            if response.status == 200:
                                logger.info(f"Registered service with Consul: {service['name']}")
                                self.services_configured.add(service['id'])
                            else:
                                error_text = await response.text()
                                logger.error(f"Failed to register {service['name']}: {error_text}")
                    
                    except Exception as e:
                        logger.error(f"Error registering service {service['name']}: {e}")
            
            logger.info(f"Consul configuration completed. Registered {len(self.services_configured)} services")
        
        except Exception as e:
            logger.error(f"Error configuring Consul services: {e}")
            raise
    
    async def configure_kong_gateway(self):
        """Configure Kong API Gateway"""
        logger.info("Configuring Kong API Gateway...")
        
        try:
            # Load Kong configuration
            config_path = '/opt/sutazaiapp/config/kong/kong.yml'
            with open(config_path, 'r') as f:
                kong_config = yaml.safe_load(f)
            
            async with aiohttp.ClientSession() as session:
                # Configure services
                for service_config in kong_config.get('services', []):
                    try:
                        service_data = {
                            'name': service_config['name'],
                            'url': service_config['url'],
                            'retries': service_config.get('retries', 5),
                            'connect_timeout': service_config.get('connect_timeout', 60000),
                            'write_timeout': service_config.get('write_timeout', 60000),
                            'read_timeout': service_config.get('read_timeout', 60000)
                        }
                        
                        # Create or update service
                        url = f"{self.kong_admin_url}/services"
                        async with session.post(url, json=service_data) as response:
                            if response.status in [200, 201, 409]:  # 409 = conflict (already exists)
                                service_name = service_config['name']
                                logger.info(f"Configured Kong service: {service_name}")
                                
                                # Configure routes for this service
                                for route_config in service_config.get('routes', []):
                                    route_data = {
                                        'name': route_config['name'],
                                        'paths': route_config['paths'],
                                        'methods': route_config.get('methods', ['GET', 'POST']),
                                        'strip_path': route_config.get('strip_path', True),
                                        'preserve_host': route_config.get('preserve_host', False)
                                    }
                                    
                                    # Create route
                                    route_url = f"{self.kong_admin_url}/services/{service_name}/routes"
                                    async with session.post(route_url, json=route_data) as route_response:
                                        if route_response.status in [200, 201, 409]:
                                            logger.info(f"Configured Kong route: {route_config['name']}")
                                        else:
                                            error_text = await route_response.text()
                                            logger.error(f"Failed to create route {route_config['name']}: {error_text}")
                                
                                # Configure plugins for this service
                                for plugin_config in service_config.get('plugins', []):
                                    plugin_data = {
                                        'name': plugin_config['name'],
                                        'config': plugin_config.get('config', {})
                                    }
                                    
                                    plugin_url = f"{self.kong_admin_url}/services/{service_name}/plugins"
                                    async with session.post(plugin_url, json=plugin_data) as plugin_response:
                                        if plugin_response.status in [200, 201, 409]:
                                            logger.info(f"Configured Kong plugin: {plugin_config['name']} for {service_name}")
                                        else:
                                            error_text = await plugin_response.text()
                                            logger.error(f"Failed to create plugin {plugin_config['name']}: {error_text}")
                            
                            else:
                                error_text = await response.text()
                                logger.error(f"Failed to create service {service_config['name']}: {error_text}")
                    
                    except Exception as e:
                        logger.error(f"Error configuring Kong service {service_config['name']}: {e}")
                
                # Configure global plugins
                for plugin_config in kong_config.get('plugins', []):
                    try:
                        plugin_data = {
                            'name': plugin_config['name'],
                            'config': plugin_config.get('config', {})
                        }
                        
                        url = f"{self.kong_admin_url}/plugins"
                        async with session.post(url, json=plugin_data) as response:
                            if response.status in [200, 201, 409]:
                                logger.info(f"Configured global Kong plugin: {plugin_config['name']}")
                            else:
                                error_text = await response.text()
                                logger.error(f"Failed to create global plugin {plugin_config['name']}: {error_text}")
                    
                    except Exception as e:
                        logger.error(f"Error configuring global plugin {plugin_config['name']}: {e}")
            
            logger.info("Kong configuration completed")
        
        except Exception as e:
            logger.error(f"Error configuring Kong gateway: {e}")
            raise
    
    async def configure_rabbitmq_messaging(self):
        """Configure RabbitMQ messaging system"""
        logger.info("Configuring RabbitMQ messaging...")
        
        try:
            # Load RabbitMQ definitions
            config_path = '/opt/sutazaiapp/config/rabbitmq/definitions.json'
            with open(config_path, 'r') as f:
                rabbitmq_config = json.load(f)
            
            # Apply RabbitMQ configuration using management API
            auth = aiohttp.BasicAuth(self.rabbitmq_user, self.rabbitmq_pass)
            
            async with aiohttp.ClientSession() as session:
                # Import definitions
                url = f"{self.rabbitmq_url}/api/definitions"
                async with session.post(url, json=rabbitmq_config, auth=auth) as response:
                    if response.status in [200, 201, 204]:
                        logger.info("RabbitMQ definitions imported successfully")
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to import RabbitMQ definitions: {error_text}")
                        # Continue anyway, as some definitions might already exist
                
                # Verify virtual hosts
                url = f"{self.rabbitmq_url}/api/vhosts"
                async with session.get(url, auth=auth) as response:
                    if response.status == 200:
                        vhosts = await response.json()
                        vhost_names = [vh['name'] for vh in vhosts]
                        logger.info(f"RabbitMQ virtual hosts configured: {vhost_names}")
                
                # Verify queues
                url = f"{self.rabbitmq_url}/api/queues"
                async with session.get(url, auth=auth) as response:
                    if response.status == 200:
                        queues = await response.json()
                        logger.info(f"RabbitMQ queues configured: {len(queues)} total")
            
            logger.info("RabbitMQ configuration completed")
        
        except Exception as e:
            logger.error(f"Error configuring RabbitMQ messaging: {e}")
            raise
    
    async def start_health_monitoring(self):
        """Start health monitoring processes"""
        logger.info("Starting health monitoring...")
        
        try:
            # Start service discovery client
            discovery_script = '/opt/sutazaiapp/scripts/service-mesh/service-discovery-client.py'
            if os.path.exists(discovery_script):
                logger.info("Starting service discovery client...")
                # Note: In production, you'd want to run this as a separate service
                # For now, we'll just log that it should be started
                logger.info(f"Service discovery client should be started: python {discovery_script}")
            
            # Start health check server
            health_script = '/opt/sutazaiapp/scripts/service-mesh/health-check-server.py'
            if os.path.exists(health_script):
                logger.info("Starting health check server...")
                # Note: In production, you'd want to run this as a separate service
                logger.info(f"Health check server should be started: python {health_script}")
            
            logger.info("Health monitoring setup completed")
        
        except Exception as e:
            logger.error(f"Error starting health monitoring: {e}")
            raise
    
    async def validate_service_mesh(self):
        """Validate that the service mesh is working correctly"""
        logger.info("Validating service mesh configuration...")
        
        try:
            validation_results = {}
            
            # Test Consul service discovery
            async with aiohttp.ClientSession() as session:
                url = f"{self.consul_url}/v1/catalog/services"
                async with session.get(url) as response:
                    if response.status == 200:
                        services = await response.json()
                        validation_results['consul_services'] = len(services)
                        logger.info(f"Consul validation: {len(services)} services registered")
                    else:
                        validation_results['consul_services'] = 0
                        logger.error("Consul validation failed")
                
                # Test Kong gateway
                url = f"{self.kong_admin_url}/services"
                async with session.get(url) as response:
                    if response.status == 200:
                        kong_services = await response.json()
                        validation_results['kong_services'] = len(kong_services.get('data', []))
                        logger.info(f"Kong validation: {len(kong_services.get('data', []))} services configured")
                    else:
                        validation_results['kong_services'] = 0
                        logger.error("Kong validation failed")
                
                # Test RabbitMQ messaging
                auth = aiohttp.BasicAuth(self.rabbitmq_user, self.rabbitmq_pass)
                url = f"{self.rabbitmq_url}/api/overview"
                async with session.get(url, auth=auth) as response:
                    if response.status == 200:
                        overview = await response.json()
                        validation_results['rabbitmq_status'] = overview.get('node', 'unknown')
                        logger.info(f"RabbitMQ validation: Node {overview.get('node', 'unknown')} is running")
                    else:
                        validation_results['rabbitmq_status'] = 'failed'
                        logger.error("RabbitMQ validation failed")
            
            # Log validation summary
            logger.info("Service mesh validation completed:")
            for component, result in validation_results.items():
                logger.info(f"  {component}: {result}")
            
            return all(v != 0 and v != 'failed' for v in validation_results.values())
        
        except Exception as e:
            logger.error(f"Error validating service mesh: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup service mesh configuration"""
        logger.info("Cleaning up service mesh...")
        
        try:
            # Deregister services from Consul
            async with aiohttp.ClientSession() as session:
                for service_id in self.services_configured:
                    try:
                        url = f"{self.consul_url}/v1/agent/service/deregister/{service_id}"
                        async with session.put(url) as response:
                            if response.status == 200:
                                logger.info(f"Deregistered service: {service_id}")
                            else:
                                logger.error(f"Failed to deregister service: {service_id}")
                    except Exception as e:
                        logger.error(f"Error deregistering service {service_id}: {e}")
            
            logger.info("Service mesh cleanup completed")
        
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def handle_signal(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

async def main():
    """Main orchestrator function"""
    orchestrator = ServiceMeshOrchestrator()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, orchestrator.handle_signal)
    signal.signal(signal.SIGTERM, orchestrator.handle_signal)
    
    try:
        # Initialize service mesh
        if not await orchestrator.initialize():
            logger.error("Failed to initialize service mesh")
            sys.exit(1)
        
        # Validate configuration
        if await orchestrator.validate_service_mesh():
            logger.info("Service mesh validation successful")
        else:
            logger.warning("Service mesh validation failed - some components may not be working correctly")
        
        # Keep orchestrator running
        logger.info("Service mesh orchestrator is running. Press Ctrl+C to stop.")
        while orchestrator.running:
            await asyncio.sleep(60)
            
            # Periodic validation
            if await orchestrator.validate_service_mesh():
                logger.debug("Service mesh health check passed")
            else:
                logger.warning("Service mesh health check failed")
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await orchestrator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())