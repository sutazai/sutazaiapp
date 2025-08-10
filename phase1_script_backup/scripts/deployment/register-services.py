#!/usr/bin/env python3
"""
Purpose: Register all SutazAI services with Consul for service discovery.
Usage: python register-services.py [--consul-host=localhost] [--consul-port=10006]
Requirements: requests library, running Consul instance
"""

import json
import requests
import argparse
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConsulServiceRegistrar:
    """Handle service registration with Consul."""
    
    def __init__(self, consul_host: str = "localhost", consul_port: int = 10006):
        self.consul_url = f"http://{consul_host}:{consul_port}"
        self.session = requests.Session()
        
    def check_consul_health(self) -> bool:
        """Check if Consul is healthy and accepting connections."""
        try:
            response = self.session.get(f"{self.consul_url}/v1/status/leader")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to connect to Consul: {e}")
            return False
    
    def register_service(self, service: Dict[str, Any]) -> bool:
        """Register a single service with Consul."""
        try:
            # Prepare the service registration payload
            payload = {
                "ID": service["id"],
                "Name": service["name"],
                "Tags": service.get("tags", []),
                "Address": service["address"],
                "Port": service["port"],
                "Meta": service.get("meta", {}),
                "Check": service.get("check", {})
            }
            
            # Add weights if present
            if "weights" in service:
                payload["Weights"] = service["weights"]
            
            # Register the service
            response = self.session.put(
                f"{self.consul_url}/v1/agent/service/register",
                json=payload
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully registered service: {service['name']} ({service['id']})")
                return True
            else:
                logger.error(f"Failed to register {service['name']}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error registering {service['name']}: {e}")
            return False
    
    def deregister_service(self, service_id: str) -> bool:
        """Deregister a service from Consul."""
        try:
            response = self.session.put(
                f"{self.consul_url}/v1/agent/service/deregister/{service_id}"
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error deregistering {service_id}: {e}")
            return False
    
    def get_registered_services(self) -> Dict[str, Any]:
        """Get all currently registered services."""
        try:
            response = self.session.get(f"{self.consul_url}/v1/agent/services")
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            logger.error(f"Error fetching registered services: {e}")
            return {}
    
    def register_all_services(self, services_config_path: str) -> Dict[str, bool]:
        """Register all services from configuration file."""
        try:
            # Load services configuration
            with open(services_config_path, 'r') as f:
                config = json.load(f)
            
            services = config.get("services", [])
            results = {}
            
            # Get currently registered services
            current_services = self.get_registered_services()
            logger.info(f"Currently registered services: {list(current_services.keys())}")
            
            # Register each service
            for service in services:
                service_id = service["id"]
                
                # Check if service needs to be re-registered
                if service_id in current_services:
                    logger.info(f"Service {service_id} already registered, updating...")
                    self.deregister_service(service_id)
                
                # Register the service
                success = self.register_service(service)
                results[service_id] = success
                
                # Small delay to avoid overwhelming Consul
                time.sleep(0.1)
            
            return results
            
        except Exception as e:
            logger.error(f"Error loading services configuration: {e}")
            return {}
    
    def enable_service_health_checks(self) -> None:
        """Enable maintenance mode for unhealthy services."""
        try:
            # Get all service health statuses
            response = self.session.get(f"{self.consul_url}/v1/health/state/any")
            if response.status_code == 200:
                health_checks = response.json()
                
                for check in health_checks:
                    if check["Status"] == "critical":
                        service_id = check.get("ServiceID")
                        if service_id:
                            logger.warning(f"Service {service_id} is unhealthy")
                            
        except Exception as e:
            logger.error(f"Error checking service health: {e}")


def main():
    """Main function to register services with Consul."""
    parser = argparse.ArgumentParser(description="Register SutazAI services with Consul")
    parser.add_argument("--consul-host", default="localhost", help="Consul host address")
    parser.add_argument("--consul-port", type=int, default=10006, help="Consul port")
    parser.add_argument("--config", default="/opt/sutazaiapp/config/consul/services.json",
                        help="Path to services configuration file")
    parser.add_argument("--check-health", action="store_true", help="Check service health after registration")
    
    args = parser.parse_args()
    
    # Create registrar instance
    registrar = ConsulServiceRegistrar(args.consul_host, args.consul_port)
    
    # Check Consul connectivity
    logger.info(f"Connecting to Consul at {args.consul_host}:{args.consul_port}")
    if not registrar.check_consul_health():
        logger.error("Consul is not healthy or not reachable")
        return 1
    
    logger.info("Consul is healthy and accepting connections")
    
    # Register all services
    logger.info(f"Loading services from {args.config}")
    results = registrar.register_all_services(args.config)
    
    # Report results
    successful = sum(1 for success in results.values() if success)
    failed = len(results) - successful
    
    logger.info(f"\nRegistration Summary:")
    logger.info(f"Total services: {len(results)}")
    logger.info(f"Successfully registered: {successful}")
    logger.info(f"Failed registrations: {failed}")
    
    if failed > 0:
        logger.error("\nFailed services:")
        for service_id, success in results.items():
            if not success:
                logger.error(f"  - {service_id}")
    
    # Check health if requested
    if args.check_health:
        logger.info("\nChecking service health...")
        registrar.enable_service_health_checks()
    
    # Show final registered services
    final_services = registrar.get_registered_services()
    logger.info(f"\nFinal registered services: {len(final_services)}")
    for service_id, service_info in final_services.items():
        logger.info(f"  - {service_id}: {service_info.get('Service', 'unknown')}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())