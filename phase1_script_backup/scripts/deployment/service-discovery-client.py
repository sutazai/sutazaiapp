#!/usr/bin/env python3
"""
Service Discovery Client for SutazAI Service Mesh
Integrates with Consul for service registration and discovery
"""

import asyncio
import consul
import logging
import json
import os
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ServiceConfig:
    """Service configuration for registration"""
    name: str
    id: str
    address: str
    port: int
    tags: List[str]
    meta: Dict[str, str]
    health_check: Dict[str, Any]
    weights: Optional[Dict[str, int]] = None

class ServiceDiscoveryClient:
    """Client for service discovery and registration with Consul"""
    
    def __init__(self, consul_host: str = "consul", consul_port: int = 8500):
        self.consul_host = consul_host
        self.consul_port = consul_port
        self.consul_client = None
        self.registered_services = set()
        
    async def initialize(self):
        """Initialize Consul client connection"""
        try:
            self.consul_client = consul.Consul(
                host=self.consul_host,
                port=self.consul_port,
                timeout=10
            )
            
            # Test connection
            self.consul_client.agent.self()
            logger.info(f"Connected to Consul at {self.consul_host}:{self.consul_port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Consul: {e}")
            return False
    
    async def register_service(self, service_config: ServiceConfig) -> bool:
        """Register a service with Consul"""
        try:
            check_config = None
            if service_config.health_check:
                check_config = consul.Check.http(
                    url=service_config.health_check.get('http'),
                    interval=service_config.health_check.get('interval', '30s'),
                    timeout=service_config.health_check.get('timeout', '10s'),
                    deregister=service_config.health_check.get('deregister', '1m')
                ) if 'http' in service_config.health_check else consul.Check.tcp(
                    host=service_config.address,
                    port=service_config.port,
                    interval=service_config.health_check.get('interval', '30s'),
                    timeout=service_config.health_check.get('timeout', '10s'),
                    deregister=service_config.health_check.get('deregister', '1m')
                )
            
            success = self.consul_client.agent.service.register(
                name=service_config.name,
                service_id=service_config.id,
                address=service_config.address,
                port=service_config.port,
                tags=service_config.tags,
                meta=service_config.meta,
                check=check_config,
                weights=service_config.weights
            )
            
            if success:
                self.registered_services.add(service_config.id)
                logger.info(f"Registered service: {service_config.name} ({service_config.id})")
                return True
            else:
                logger.error(f"Failed to register service: {service_config.name}")
                return False
                
        except Exception as e:
            logger.error(f"Error registering service {service_config.name}: {e}")
            return False
    
    async def deregister_service(self, service_id: str) -> bool:
        """Deregister a service from Consul"""
        try:
            success = self.consul_client.agent.service.deregister(service_id)
            if success:
                self.registered_services.discard(service_id)
                logger.info(f"Deregistered service: {service_id}")
                return True
            else:
                logger.error(f"Failed to deregister service: {service_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deregistering service {service_id}: {e}")
            return False
    
    async def discover_services(self, service_name: str = None, tags: List[str] = None) -> List[Dict]:
        """Discover services from Consul"""
        try:
            if service_name:
                # Get specific service
                _, services = self.consul_client.health.service(
                    service_name, 
                    passing=True, 
                    tag=tags[0] if tags else None
                )
            else:
                # Get all services
                _, services = self.consul_client.health.state('passing')
                
            discovered = []
            for service in services:
                service_info = {
                    'name': service.get('ServiceName'),
                    'id': service.get('ServiceID'),
                    'address': service.get('ServiceAddress'),
                    'port': service.get('ServicePort'),
                    'tags': service.get('ServiceTags', []),
                    'meta': service.get('ServiceMeta', {}),
                    'status': 'healthy' if service.get('Status') == 'passing' else 'unhealthy'
                }
                discovered.append(service_info)
                
            logger.info(f"Discovered {len(discovered)} services")
            return discovered
            
        except Exception as e:
            logger.error(f"Error discovering services: {e}")
            return []
    
    async def get_service_endpoints(self, service_name: str) -> List[str]:
        """Get healthy service endpoints for load balancing"""
        try:
            _, services = self.consul_client.health.service(service_name, passing=True)
            endpoints = []
            
            for service in services:
                address = service['Service']['Address']
                port = service['Service']['Port']
                endpoints.append(f"{address}:{port}")
                
            logger.info(f"Found {len(endpoints)} healthy endpoints for {service_name}")
            return endpoints
            
        except Exception as e:
            logger.error(f"Error getting service endpoints for {service_name}: {e}")
            return []
    
    async def watch_service_changes(self, callback, service_name: str = None):
        """Watch for service changes and trigger callback"""
        try:
            index = None
            while True:
                if service_name:
                    index, services = self.consul_client.health.service(
                        service_name, 
                        index=index, 
                        wait='30s'
                    )
                else:
                    index, services = self.consul_client.catalog.services(
                        index=index, 
                        wait='30s'
                    )
                
                await callback(services)
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error watching service changes: {e}")
    
    async def load_services_from_config(self, config_path: str) -> bool:
        """Load and register services from configuration file"""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.error(f"Configuration file not found: {config_path}")
                return False
                
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            services = config.get('services', [])
            registered_count = 0
            
            for service_def in services:
                service_config = ServiceConfig(
                    name=service_def['name'],
                    id=service_def['id'],
                    address=service_def['address'],
                    port=service_def['port'],
                    tags=service_def.get('tags', []),
                    meta=service_def.get('meta', {}),
                    health_check=service_def.get('check', {}),
                    weights=service_def.get('weights')
                )
                
                if await self.register_service(service_config):
                    registered_count += 1
            
            logger.info(f"Registered {registered_count}/{len(services)} services from config")
            return registered_count == len(services)
            
        except Exception as e:
            logger.error(f"Error loading services from config: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup - deregister all registered services"""
        for service_id in list(self.registered_services):
            await self.deregister_service(service_id)
        logger.info("Service discovery client cleanup completed")

async def main():
    """Main service discovery client"""
    client = ServiceDiscoveryClient(
        consul_host=os.getenv('CONSUL_HOST', 'consul'),
        consul_port=int(os.getenv('CONSUL_PORT', '8500'))
    )
    
    try:
        if not await client.initialize():
            logger.error("Failed to initialize service discovery client")
            sys.exit(1)
        
        # Load services from configuration
        config_path = os.getenv('SERVICES_CONFIG_PATH', '/opt/sutazaiapp/config/consul/services.json')
        if await client.load_services_from_config(config_path):
            logger.info("All services registered successfully")
        else:
            logger.warning("Some services failed to register")
        
        # Keep the client running
        logger.info("Service discovery client running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(60)
            # Periodic health check of registered services
            services = await client.discover_services()
            healthy_services = [s for s in services if s['status'] == 'healthy']
            logger.info(f"Health check: {len(healthy_services)}/{len(services)} services healthy")
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())