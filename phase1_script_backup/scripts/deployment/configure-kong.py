#!/usr/bin/env python3
"""
Purpose: Configure Kong API Gateway with routes for all SutazAI services.
Usage: python configure-kong.py [--kong-admin-url=http://localhost:10007]
Requirements: requests, PyYAML libraries, running Kong instance
"""

import yaml
import requests
import argparse
import time
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KongConfigurator:
    """Handle Kong API Gateway configuration."""
    
    def __init__(self, admin_url: str = "http://localhost:10007"):
        self.admin_url = admin_url
        self.session = requests.Session()
        
    def check_kong_health(self) -> bool:
        """Check if Kong is healthy and accepting connections."""
        try:
            response = self.session.get(f"{self.admin_url}/status")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to connect to Kong: {e}")
            return False
    
    def get_existing_services(self) -> List[Dict[str, Any]]:
        """Get all existing services from Kong."""
        try:
            response = self.session.get(f"{self.admin_url}/services")
            if response.status_code == 200:
                return response.json().get('data', [])
            return []
        except Exception as e:
            logger.error(f"Error fetching services: {e}")
            return []
    
    def create_or_update_service(self, service_config: Dict[str, Any]) -> Optional[str]:
        """Create or update a service in Kong."""
        try:
            service_name = service_config['name']
            service_url = service_config['url']
            
            # Check if service exists
            response = self.session.get(f"{self.admin_url}/services/{service_name}")
            
            if response.status_code == 200:
                # Update existing service
                logger.info(f"Updating existing service: {service_name}")
                response = self.session.patch(
                    f"{self.admin_url}/services/{service_name}",
                    json={"url": service_url}
                )
            else:
                # Create new service
                logger.info(f"Creating new service: {service_name}")
                response = self.session.post(
                    f"{self.admin_url}/services",
                    json={
                        "name": service_name,
                        "url": service_url,
                        "connect_timeout": 60000,
                        "write_timeout": 60000,
                        "read_timeout": 60000
                    }
                )
            
            if response.status_code in [200, 201]:
                logger.info(f"Successfully configured service: {service_name}")
                return service_name
            else:
                logger.error(f"Failed to configure service {service_name}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error configuring service {service_config.get('name')}: {e}")
            return None
    
    def create_or_update_route(self, service_name: str, route_config: Dict[str, Any]) -> bool:
        """Create or update a route for a service."""
        try:
            route_name = route_config['name']
            
            # Check if route exists
            response = self.session.get(f"{self.admin_url}/routes/{route_name}")
            
            route_data = {
                "name": route_name,
                "paths": route_config.get('paths', []),
                "methods": route_config.get('methods', ["GET", "POST", "PUT", "DELETE", "OPTIONS"]),
                "strip_path": route_config.get('strip_path', False),
                "preserve_host": route_config.get('preserve_host', False),
                "service": {"name": service_name}
            }
            
            if response.status_code == 200:
                # Update existing route
                logger.info(f"Updating existing route: {route_name}")
                response = self.session.patch(
                    f"{self.admin_url}/routes/{route_name}",
                    json=route_data
                )
            else:
                # Create new route
                logger.info(f"Creating new route: {route_name}")
                response = self.session.post(
                    f"{self.admin_url}/routes",
                    json=route_data
                )
            
            if response.status_code in [200, 201]:
                logger.info(f"Successfully configured route: {route_name}")
                return True
            else:
                logger.error(f"Failed to configure route {route_name}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error configuring route {route_config.get('name')}: {e}")
            return False
    
    def add_plugin(self, service_name: str, plugin_config: Dict[str, Any]) -> bool:
        """Add a plugin to a service."""
        try:
            plugin_name = plugin_config['name']
            config = plugin_config.get('config', {})
            
            # Check if plugin already exists for this service
            response = self.session.get(
                f"{self.admin_url}/services/{service_name}/plugins",
                params={"name": plugin_name}
            )
            
            existing_plugins = response.json().get('data', []) if response.status_code == 200 else []
            
            if existing_plugins:
                # Update existing plugin
                plugin_id = existing_plugins[0]['id']
                logger.info(f"Updating plugin {plugin_name} for service {service_name}")
                response = self.session.patch(
                    f"{self.admin_url}/plugins/{plugin_id}",
                    json={"config": config}
                )
            else:
                # Create new plugin
                logger.info(f"Adding plugin {plugin_name} to service {service_name}")
                response = self.session.post(
                    f"{self.admin_url}/services/{service_name}/plugins",
                    json={
                        "name": plugin_name,
                        "config": config
                    }
                )
            
            if response.status_code in [200, 201]:
                logger.info(f"Successfully configured plugin {plugin_name} for {service_name}")
                return True
            else:
                logger.error(f"Failed to configure plugin {plugin_name}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error configuring plugin {plugin_config.get('name')}: {e}")
            return False
    
    def configure_global_plugins(self, plugins: List[Dict[str, Any]]) -> int:
        """Configure global plugins."""
        success_count = 0
        
        for plugin in plugins:
            try:
                plugin_name = plugin['name']
                config = plugin.get('config', {})
                
                # Check if global plugin exists
                response = self.session.get(
                    f"{self.admin_url}/plugins",
                    params={"name": plugin_name}
                )
                
                existing_plugins = response.json().get('data', []) if response.status_code == 200 else []
                global_plugins = [p for p in existing_plugins if p.get('service') is None]
                
                if global_plugins:
                    # Update existing global plugin
                    plugin_id = global_plugins[0]['id']
                    logger.info(f"Updating global plugin: {plugin_name}")
                    response = self.session.patch(
                        f"{self.admin_url}/plugins/{plugin_id}",
                        json={"config": config}
                    )
                else:
                    # Create new global plugin
                    logger.info(f"Creating global plugin: {plugin_name}")
                    response = self.session.post(
                        f"{self.admin_url}/plugins",
                        json={
                            "name": plugin_name,
                            "config": config
                        }
                    )
                
                if response.status_code in [200, 201]:
                    logger.info(f"Successfully configured global plugin: {plugin_name}")
                    success_count += 1
                else:
                    logger.error(f"Failed to configure global plugin {plugin_name}: {response.text}")
                    
            except Exception as e:
                logger.error(f"Error configuring global plugin: {e}")
        
        return success_count
    
    def create_consumers(self, consumers: List[Dict[str, Any]]) -> int:
        """Create consumers in Kong."""
        success_count = 0
        
        for consumer in consumers:
            try:
                username = consumer['username']
                custom_id = consumer.get('custom_id')
                
                # Check if consumer exists
                response = self.session.get(f"{self.admin_url}/consumers/{username}")
                
                consumer_data = {"username": username}
                if custom_id:
                    consumer_data["custom_id"] = custom_id
                
                if response.status_code == 200:
                    logger.info(f"Consumer {username} already exists")
                    success_count += 1
                else:
                    # Create new consumer
                    logger.info(f"Creating consumer: {username}")
                    response = self.session.post(
                        f"{self.admin_url}/consumers",
                        json=consumer_data
                    )
                    
                    if response.status_code in [200, 201]:
                        logger.info(f"Successfully created consumer: {username}")
                        success_count += 1
                    else:
                        logger.error(f"Failed to create consumer {username}: {response.text}")
                        
            except Exception as e:
                logger.error(f"Error creating consumer: {e}")
        
        return success_count
    
    def configure_kong_from_file(self, config_file: str) -> Dict[str, Any]:
        """Configure Kong from a YAML configuration file."""
        try:
            # Load configuration
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            results = {
                "services": {"success": 0, "failed": 0},
                "routes": {"success": 0, "failed": 0},
                "plugins": {"success": 0, "failed": 0},
                "global_plugins": {"success": 0, "failed": 0},
                "consumers": {"success": 0, "failed": 0}
            }
            
            # Configure services
            services = config.get('services', [])
            for service_config in services:
                service_name = self.create_or_update_service(service_config)
                
                if service_name:
                    results["services"]["success"] += 1
                    
                    # Configure routes for this service
                    routes = service_config.get('routes', [])
                    for route in routes:
                        if self.create_or_update_route(service_name, route):
                            results["routes"]["success"] += 1
                        else:
                            results["routes"]["failed"] += 1
                    
                    # Configure plugins for this service
                    plugins = service_config.get('plugins', [])
                    for plugin in plugins:
                        if self.add_plugin(service_name, plugin):
                            results["plugins"]["success"] += 1
                        else:
                            results["plugins"]["failed"] += 1
                else:
                    results["services"]["failed"] += 1
                
                # Small delay to avoid overwhelming Kong
                time.sleep(0.1)
            
            # Configure global plugins
            global_plugins = config.get('plugins', [])
            results["global_plugins"]["success"] = self.configure_global_plugins(global_plugins)
            results["global_plugins"]["failed"] = len(global_plugins) - results["global_plugins"]["success"]
            
            # Create consumers
            consumers = config.get('consumers', [])
            results["consumers"]["success"] = self.create_consumers(consumers)
            results["consumers"]["failed"] = len(consumers) - results["consumers"]["success"]
            
            return results
            
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
            return {}


def main():
    """Main function to configure Kong."""
    parser = argparse.ArgumentParser(description="Configure Kong API Gateway for SutazAI")
    parser.add_argument("--kong-admin-url", default="http://localhost:10007", 
                        help="Kong Admin API URL")
    parser.add_argument("--config", default="/opt/sutazaiapp/config/kong/kong.yml",
                        help="Path to Kong configuration file")
    
    args = parser.parse_args()
    
    # Create configurator instance
    configurator = KongConfigurator(args.kong_admin_url)
    
    # Check Kong connectivity
    logger.info(f"Connecting to Kong Admin API at {args.kong_admin_url}")
    if not configurator.check_kong_health():
        logger.error("Kong is not healthy or not reachable")
        return 1
    
    logger.info("Kong is healthy and accepting connections")
    
    # Get existing services
    existing_services = configurator.get_existing_services()
    logger.info(f"Found {len(existing_services)} existing services in Kong")
    
    # Configure Kong from file
    logger.info(f"Loading configuration from {args.config}")
    results = configurator.configure_kong_from_file(args.config)
    
    # Report results
    if results:
        logger.info("\nConfiguration Summary:")
        logger.info(f"Services: {results['services']['success']} success, {results['services']['failed']} failed")
        logger.info(f"Routes: {results['routes']['success']} success, {results['routes']['failed']} failed")
        logger.info(f"Plugins: {results['plugins']['success']} success, {results['plugins']['failed']} failed")
        logger.info(f"Global Plugins: {results['global_plugins']['success']} success, {results['global_plugins']['failed']} failed")
        logger.info(f"Consumers: {results['consumers']['success']} success, {results['consumers']['failed']} failed")
        
        # Check final configuration
        final_services = configurator.get_existing_services()
        logger.info(f"\nFinal service count in Kong: {len(final_services)}")
        
        total_failed = sum(r['failed'] for r in results.values())
        return 0 if total_failed == 0 else 1
    else:
        logger.error("Configuration failed")
        return 1


if __name__ == "__main__":
    exit(main())