#!/usr/bin/env python3
"""
Purpose: Configure RabbitMQ with exchanges, queues, and bindings for SutazAI service mesh.
Usage: python configure-rabbitmq.py [--rabbitmq-host=localhost] [--port=10042]
Requirements: requests library, running RabbitMQ instance
"""

import json
import requests
import argparse
import logging
from typing import Dict, List, Any
from requests.auth import HTTPBasicAuth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RabbitMQConfigurator:
    """Handle RabbitMQ configuration for service mesh."""
    
    def __init__(self, host: str = "localhost", port: int = 10042, 
                 username: str = "admin", password: str = "adminpass"):
        self.base_url = f"http://{host}:{port}/api"
        self.auth = HTTPBasicAuth(username, password)
        self.session = requests.Session()
        self.session.auth = self.auth
        
    def check_rabbitmq_health(self) -> bool:
        """Check if RabbitMQ is healthy and accepting connections."""
        try:
            response = self.session.get(f"{self.base_url}/overview")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            return False
    
    def create_vhost(self, vhost_config: Dict[str, Any]) -> bool:
        """Create a virtual host in RabbitMQ."""
        try:
            vhost_name = vhost_config['name']
            description = vhost_config.get('description', '')
            
            # Check if vhost exists
            response = self.session.get(f"{self.base_url}/vhosts/{vhost_name}")
            
            if response.status_code == 200:
                logger.info(f"Virtual host {vhost_name} already exists")
                return True
            
            # Create vhost
            response = self.session.put(
                f"{self.base_url}/vhosts/{vhost_name}",
                json={"description": description, "tags": ""}
            )
            
            if response.status_code in [201, 204]:
                logger.info(f"Successfully created virtual host: {vhost_name}")
                return True
            else:
                logger.error(f"Failed to create vhost {vhost_name}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating vhost: {e}")
            return False
    
    def set_permissions(self, permission: Dict[str, Any]) -> bool:
        """Set permissions for a user on a vhost."""
        try:
            user = permission['user']
            vhost = permission['vhost']
            
            response = self.session.put(
                f"{self.base_url}/permissions/{vhost}/{user}",
                json={
                    "configure": permission.get('configure', '.*'),
                    "write": permission.get('write', '.*'),
                    "read": permission.get('read', '.*')
                }
            )
            
            if response.status_code in [201, 204]:
                logger.info(f"Successfully set permissions for {user} on {vhost}")
                return True
            else:
                logger.error(f"Failed to set permissions: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting permissions: {e}")
            return False
    
    def create_exchange(self, exchange: Dict[str, Any]) -> bool:
        """Create an exchange in RabbitMQ."""
        try:
            vhost = exchange['vhost']
            name = exchange['name']
            
            response = self.session.put(
                f"{self.base_url}/exchanges/{vhost}/{name}",
                json={
                    "type": exchange.get('type', 'direct'),
                    "durable": exchange.get('durable', True),
                    "auto_delete": exchange.get('auto_delete', False),
                    "internal": exchange.get('internal', False),
                    "arguments": exchange.get('arguments', {})
                }
            )
            
            if response.status_code in [201, 204]:
                logger.info(f"Successfully created exchange: {name} in vhost {vhost}")
                return True
            else:
                logger.error(f"Failed to create exchange {name}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating exchange: {e}")
            return False
    
    def create_queue(self, queue: Dict[str, Any]) -> bool:
        """Create a queue in RabbitMQ."""
        try:
            vhost = queue['vhost']
            name = queue['name']
            
            response = self.session.put(
                f"{self.base_url}/queues/{vhost}/{name}",
                json={
                    "durable": queue.get('durable', True),
                    "auto_delete": queue.get('auto_delete', False),
                    "arguments": queue.get('arguments', {})
                }
            )
            
            if response.status_code in [201, 204]:
                logger.info(f"Successfully created queue: {name} in vhost {vhost}")
                return True
            else:
                logger.error(f"Failed to create queue {name}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating queue: {e}")
            return False
    
    def create_binding(self, binding: Dict[str, Any]) -> bool:
        """Create a binding between exchange and queue."""
        try:
            vhost = binding['vhost']
            source = binding['source']
            destination = binding['destination']
            destination_type = binding.get('destination_type', 'queue')
            routing_key = binding.get('routing_key', '')
            
            # For queue bindings
            if destination_type == 'queue':
                url = f"{self.base_url}/bindings/{vhost}/e/{source}/q/{destination}"
            else:
                # For exchange to exchange bindings
                url = f"{self.base_url}/bindings/{vhost}/e/{source}/e/{destination}"
            
            response = self.session.post(
                url,
                json={
                    "routing_key": routing_key,
                    "arguments": binding.get('arguments', {})
                }
            )
            
            if response.status_code in [201, 204]:
                logger.info(f"Successfully created binding: {source} -> {destination} (key: {routing_key})")
                return True
            else:
                logger.error(f"Failed to create binding: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating binding: {e}")
            return False
    
    def create_policy(self, policy: Dict[str, Any]) -> bool:
        """Create a policy in RabbitMQ."""
        try:
            vhost = policy['vhost']
            name = policy['name']
            
            response = self.session.put(
                f"{self.base_url}/policies/{vhost}/{name}",
                json={
                    "pattern": policy.get('pattern', '.*'),
                    "apply-to": policy.get('apply-to', 'all'),
                    "definition": policy.get('definition', {}),
                    "priority": policy.get('priority', 0)
                }
            )
            
            if response.status_code in [201, 204]:
                logger.info(f"Successfully created policy: {name} in vhost {vhost}")
                return True
            else:
                logger.error(f"Failed to create policy {name}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating policy: {e}")
            return False
    
    def configure_from_file(self, config_file: str) -> Dict[str, Any]:
        """Configure RabbitMQ from a definitions file."""
        try:
            # Load configuration
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            results = {
                "vhosts": {"success": 0, "failed": 0},
                "permissions": {"success": 0, "failed": 0},
                "exchanges": {"success": 0, "failed": 0},
                "queues": {"success": 0, "failed": 0},
                "bindings": {"success": 0, "failed": 0},
                "policies": {"success": 0, "failed": 0}
            }
            
            # Create vhosts
            for vhost in config.get('vhosts', []):
                if self.create_vhost(vhost):
                    results["vhosts"]["success"] += 1
                else:
                    results["vhosts"]["failed"] += 1
            
            # Set permissions
            for permission in config.get('permissions', []):
                if self.set_permissions(permission):
                    results["permissions"]["success"] += 1
                else:
                    results["permissions"]["failed"] += 1
            
            # Create exchanges
            for exchange in config.get('exchanges', []):
                if self.create_exchange(exchange):
                    results["exchanges"]["success"] += 1
                else:
                    results["exchanges"]["failed"] += 1
            
            # Create queues
            for queue in config.get('queues', []):
                if self.create_queue(queue):
                    results["queues"]["success"] += 1
                else:
                    results["queues"]["failed"] += 1
            
            # Create bindings
            for binding in config.get('bindings', []):
                if self.create_binding(binding):
                    results["bindings"]["success"] += 1
                else:
                    results["bindings"]["failed"] += 1
            
            # Create policies
            for policy in config.get('policies', []):
                if self.create_policy(policy):
                    results["policies"]["success"] += 1
                else:
                    results["policies"]["failed"] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
            return {}
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current RabbitMQ state."""
        try:
            state = {
                "vhosts": [],
                "exchanges": [],
                "queues": [],
                "bindings": []
            }
            
            # Get vhosts
            response = self.session.get(f"{self.base_url}/vhosts")
            if response.status_code == 200:
                state["vhosts"] = [v['name'] for v in response.json()]
            
            # Get exchanges
            response = self.session.get(f"{self.base_url}/exchanges")
            if response.status_code == 200:
                state["exchanges"] = len([e for e in response.json() if not e['name'].startswith('amq.')])
            
            # Get queues
            response = self.session.get(f"{self.base_url}/queues")
            if response.status_code == 200:
                state["queues"] = len(response.json())
            
            # Get bindings
            response = self.session.get(f"{self.base_url}/bindings")
            if response.status_code == 200:
                state["bindings"] = len([b for b in response.json() if b.get('source')])
            
            return state
            
        except Exception as e:
            logger.error(f"Error getting current state: {e}")
            return {}


def main():
    """Main function to configure RabbitMQ."""
    parser = argparse.ArgumentParser(description="Configure RabbitMQ for SutazAI service mesh")
    parser.add_argument("--rabbitmq-host", default="localhost", help="RabbitMQ host")
    parser.add_argument("--port", type=int, default=10042, help="RabbitMQ management port")
    parser.add_argument("--username", default="admin", help="RabbitMQ username")
    parser.add_argument("--password", default="adminpass", help="RabbitMQ password")
    parser.add_argument("--config", default="/opt/sutazaiapp/config/rabbitmq/definitions.json",
                        help="Path to RabbitMQ definitions file")
    
    args = parser.parse_args()
    
    # Create configurator instance
    configurator = RabbitMQConfigurator(
        args.rabbitmq_host, args.port, args.username, args.password
    )
    
    # Check RabbitMQ connectivity
    logger.info(f"Connecting to RabbitMQ at {args.rabbitmq_host}:{args.port}")
    if not configurator.check_rabbitmq_health():
        logger.error("RabbitMQ is not healthy or not reachable")
        return 1
    
    logger.info("RabbitMQ is healthy and accepting connections")
    
    # Get initial state
    initial_state = configurator.get_current_state()
    logger.info(f"Initial state - VHosts: {initial_state.get('vhosts', [])}")
    logger.info(f"Initial state - Exchanges: {initial_state.get('exchanges', 0)}")
    logger.info(f"Initial state - Queues: {initial_state.get('queues', 0)}")
    
    # Configure from file
    logger.info(f"Loading configuration from {args.config}")
    results = configurator.configure_from_file(args.config)
    
    # Report results
    if results:
        logger.info("\nConfiguration Summary:")
        for component, stats in results.items():
            logger.info(f"{component.capitalize()}: {stats['success']} success, {stats['failed']} failed")
        
        # Get final state
        final_state = configurator.get_current_state()
        logger.info(f"\nFinal state - VHosts: {final_state.get('vhosts', [])}")
        logger.info(f"Final state - Exchanges: {final_state.get('exchanges', 0)}")
        logger.info(f"Final state - Queues: {final_state.get('queues', 0)}")
        logger.info(f"Final state - Bindings: {final_state.get('bindings', 0)}")
        
        total_failed = sum(stats['failed'] for stats in results.values())
        return 0 if total_failed == 0 else 1
    else:
        logger.error("Configuration failed")
        return 1


if __name__ == "__main__":
    exit(main())