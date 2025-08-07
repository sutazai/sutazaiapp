#!/usr/bin/env python3
"""
Purpose: Optimize load balancing configuration for the SutazAI service mesh.
Usage: python optimize-load-balancing.py [--consul-host=localhost] [--kong-admin=http://localhost:10007]
Requirements: requests library, running Consul and Kong instances
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


class LoadBalancerOptimizer:
    """Optimize load balancing in the service mesh."""
    
    def __init__(self, consul_host: str = "localhost", consul_port: int = 10006,
                 kong_admin_url: str = "http://localhost:10007"):
        self.consul_url = f"http://{consul_host}:{consul_port}"
        self.kong_admin_url = kong_admin_url
        self.session = requests.Session()
        
        # Load balancing strategies by service type
        self.lb_strategies = {
            "ai": "least_connections",      # For AI/LLM services
            "database": "round_robin",      # For databases
            "cache": "consistent_hashing",  # For cache services
            "api": "weighted_round_robin",  # For API services
            "workflow": "least_requests",   # For workflow engines
            "monitoring": "round_robin"     # For monitoring services
        }
    
    def get_service_type(self, service_name: str, tags: List[str]) -> str:
        """Determine service type from name and tags."""
        # Check tags first
        if any(tag in ["ai", "llm", "inference"] for tag in tags):
            return "ai"
        elif any(tag in ["database", "db", "storage"] for tag in tags):
            return "database"
        elif any(tag in ["cache", "redis", "memcached"] for tag in tags):
            return "cache"
        elif any(tag in ["api", "rest", "graphql"] for tag in tags):
            return "api"
        elif any(tag in ["workflow", "automation"] for tag in tags):
            return "workflow"
        elif any(tag in ["monitoring", "metrics", "logging"] for tag in tags):
            return "monitoring"
        
        # Fallback to name analysis
        if "ollama" in service_name or "gpt" in service_name:
            return "ai"
        elif "postgres" in service_name or "neo4j" in service_name:
            return "database"
        elif "redis" in service_name:
            return "cache"
        elif "backend" in service_name or "api" in service_name:
            return "api"
        elif "flow" in service_name or "n8n" in service_name:
            return "workflow"
        elif "prometheus" in service_name or "grafana" in service_name:
            return "monitoring"
        
        return "api"  # Default
    
    def update_consul_service_weights(self, service_name: str, instances: List[Dict]) -> bool:
        """Update service weights in Consul based on health and performance."""
        try:
            for instance in instances:
                service_id = instance.get('ServiceID')
                
                # Calculate weight based on health checks and metadata
                weight = self.calculate_service_weight(instance)
                
                # Update service registration with new weight
                service_data = {
                    "ID": service_id,
                    "Name": service_name,
                    "Tags": instance.get('ServiceTags', []),
                    "Address": instance.get('ServiceAddress', instance.get('Address')),
                    "Port": instance.get('ServicePort'),
                    "Meta": instance.get('ServiceMeta', {}),
                    "Weights": {
                        "Passing": weight,
                        "Warning": max(1, weight // 2)
                    }
                }
                
                response = self.session.put(
                    f"{self.consul_url}/v1/agent/service/register",
                    json=service_data
                )
                
                if response.status_code == 200:
                    logger.info(f"Updated weight for {service_id}: {weight}")
                else:
                    logger.error(f"Failed to update weight for {service_id}")
                    
            return True
            
        except Exception as e:
            logger.error(f"Error updating Consul weights: {e}")
            return False
    
    def calculate_service_weight(self, instance: Dict) -> int:
        """Calculate optimal weight for a service instance."""
        base_weight = 10
        
        # Get instance metadata
        meta = instance.get('ServiceMeta', {})
        
        # Adjust based on CPU/memory if available
        if 'cpu_limit' in meta:
            cpu_factor = float(meta.get('cpu_limit', '1').rstrip('G'))
            base_weight = int(base_weight * cpu_factor)
        
        if 'memory_limit' in meta:
            mem_factor = float(meta.get('memory_limit', '1').rstrip('G')) / 2
            base_weight = int(base_weight * mem_factor)
        
        # Adjust based on version (newer versions get higher weight)
        version = meta.get('version', 'v1')
        if 'v40' in version:
            base_weight = int(base_weight * 1.2)
        
        # Cap weight between 1 and 100
        return max(1, min(100, base_weight))
    
    def configure_kong_upstream(self, service_name: str, strategy: str, 
                               targets: List[Dict]) -> bool:
        """Configure Kong upstream with load balancing strategy."""
        try:
            upstream_name = f"{service_name}-upstream"
            
            # Create or update upstream
            upstream_data = {
                "name": upstream_name,
                "algorithm": self.map_strategy_to_kong(strategy),
                "slots": 10000,
                "healthchecks": {
                    "active": {
                        "type": "http",
                        "http_path": "/health",
                        "healthy": {
                            "interval": 10,
                            "successes": 2
                        },
                        "unhealthy": {
                            "interval": 5,
                            "http_failures": 3,
                            "tcp_failures": 3,
                            "timeouts": 3
                        }
                    },
                    "passive": {
                        "healthy": {
                            "successes": 5
                        },
                        "unhealthy": {
                            "http_failures": 5,
                            "tcp_failures": 5,
                            "timeouts": 5
                        }
                    }
                }
            }
            
            # Try to create upstream
            response = self.session.post(
                f"{self.kong_admin_url}/upstreams",
                json=upstream_data
            )
            
            if response.status_code not in [201, 409]:
                logger.error(f"Failed to create upstream {upstream_name}: {response.text}")
                return False
            
            # Add targets to upstream
            for target in targets:
                target_data = {
                    "target": f"{target['address']}:{target['port']}",
                    "weight": target.get('weight', 100)
                }
                
                response = self.session.post(
                    f"{self.kong_admin_url}/upstreams/{upstream_name}/targets",
                    json=target_data
                )
                
                if response.status_code == 201:
                    logger.info(f"Added target {target_data['target']} to {upstream_name}")
                
            # Update service to use upstream
            response = self.session.patch(
                f"{self.kong_admin_url}/services/{service_name}-service",
                json={"host": upstream_name}
            )
            
            if response.status_code == 200:
                logger.info(f"Updated {service_name} to use upstream load balancing")
                return True
            else:
                logger.error(f"Failed to update service to use upstream: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error configuring Kong upstream: {e}")
            return False
    
    def map_strategy_to_kong(self, strategy: str) -> str:
        """Map our strategy names to Kong's algorithm names."""
        mapping = {
            "round_robin": "round-robin",
            "weighted_round_robin": "round-robin",  # Kong uses weights with round-robin
            "least_connections": "least-connections",
            "least_requests": "least-connections",
            "consistent_hashing": "consistent-hashing"
        }
        return mapping.get(strategy, "round-robin")
    
    def add_circuit_breaker(self, service_name: str) -> bool:
        """Add circuit breaker configuration to service."""
        try:
            # Configure circuit breaker plugin for the service
            cb_config = {
                "name": "circuit-breaker",
                "config": {
                    "error_threshold": 50,           # 50% error rate
                    "volume_threshold": 10,          # Minimum 10 requests
                    "timeout": 60,                   # 60 second timeout
                    "window_size": 60,               # 60 second window
                    "half_open_min_calls_in_window": 5,
                    "half_open_max_calls_in_window": 10
                }
            }
            
            response = self.session.post(
                f"{self.kong_admin_url}/services/{service_name}-service/plugins",
                json=cb_config
            )
            
            if response.status_code in [201, 409]:
                logger.info(f"Circuit breaker configured for {service_name}")
                return True
            else:
                logger.error(f"Failed to add circuit breaker: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding circuit breaker: {e}")
            return False
    
    def optimize_all_services(self) -> Dict[str, Any]:
        """Optimize load balancing for all registered services."""
        results = {
            "optimized": 0,
            "failed": 0,
            "services": []
        }
        
        try:
            # Get all services from Consul
            response = self.session.get(f"{self.consul_url}/v1/catalog/services")
            if response.status_code != 200:
                logger.error("Failed to fetch services from Consul")
                return results
            
            services = response.json()
            
            for service_name, tags in services.items():
                if service_name == "consul":
                    continue
                
                logger.info(f"\nOptimizing {service_name}...")
                
                # Get service instances
                response = self.session.get(
                    f"{self.consul_url}/v1/catalog/service/{service_name}"
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to fetch instances for {service_name}")
                    results["failed"] += 1
                    continue
                
                instances = response.json()
                
                # Determine service type and strategy
                service_type = self.get_service_type(service_name, tags)
                strategy = self.lb_strategies.get(service_type, "round_robin")
                
                logger.info(f"Service type: {service_type}, Strategy: {strategy}")
                
                # Update Consul weights
                self.update_consul_service_weights(service_name, instances)
                
                # Prepare targets for Kong
                targets = []
                for instance in instances:
                    targets.append({
                        "address": instance.get('ServiceAddress', instance.get('Address')),
                        "port": instance.get('ServicePort'),
                        "weight": self.calculate_service_weight(instance)
                    })
                
                # Configure Kong upstream
                if self.configure_kong_upstream(service_name, strategy, targets):
                    # Add circuit breaker for critical services
                    if service_type in ["ai", "api", "database"]:
                        self.add_circuit_breaker(service_name)
                    
                    results["optimized"] += 1
                    results["services"].append({
                        "name": service_name,
                        "type": service_type,
                        "strategy": strategy,
                        "instances": len(instances),
                        "targets": len(targets)
                    })
                else:
                    results["failed"] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error optimizing services: {e}")
            return results


def main():
    """Main function to optimize load balancing."""
    parser = argparse.ArgumentParser(description="Optimize SutazAI service mesh load balancing")
    parser.add_argument("--consul-host", default="localhost", help="Consul host")
    parser.add_argument("--consul-port", type=int, default=10006, help="Consul port")
    parser.add_argument("--kong-admin", default="http://localhost:10007", help="Kong Admin API URL")
    
    args = parser.parse_args()
    
    # Create optimizer instance
    optimizer = LoadBalancerOptimizer(args.consul_host, args.consul_port, args.kong_admin)
    
    logger.info("Starting SutazAI Load Balancing Optimization")
    logger.info("=" * 60)
    
    # Optimize all services
    results = optimizer.optimize_all_services()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Services Optimized: {results['optimized']}")
    logger.info(f"Failed: {results['failed']}")
    
    if results['services']:
        logger.info("\nOptimized Services:")
        for service in results['services']:
            logger.info(f"  - {service['name']}: {service['type']} "
                       f"({service['strategy']}) - {service['instances']} instances")
    
    # Save results
    with open("/opt/sutazaiapp/reports/load-balancing-optimization.json", 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("\nOptimization results saved to: /opt/sutazaiapp/reports/load-balancing-optimization.json")
    
    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    exit(main())