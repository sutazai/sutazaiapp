#!/usr/bin/env python3
"""
Purpose: Configure fault tolerance and circuit breakers for the SutazAI service mesh.
Usage: python configure-fault-tolerance.py [--kong-admin=http://localhost:10007]
Requirements: requests library, running Kong instance
"""

import json
import requests
import argparse
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FaultToleranceConfigurator:
    """Configure fault tolerance mechanisms in the service mesh."""
    
    def __init__(self, kong_admin_url: str = "http://localhost:10007"):
        self.kong_admin_url = kong_admin_url
        self.session = requests.Session()
        
        # Service criticality levels
        self.service_criticality = {
            "backend": "critical",
            "ollama": "critical",
            "postgres": "critical",
            "redis": "high",
            "neo4j": "high",
            "chromadb": "medium",
            "qdrant": "medium",
            "autogpt": "medium",
            "crewai": "medium",
            "prometheus": "low",
            "grafana": "low"
        }
    
    def get_criticality_config(self, criticality: str) -> Dict[str, Any]:
        """Get fault tolerance configuration based on criticality level."""
        configs = {
            "critical": {
                "retry": {
                    "max_retries": 5,
                    "retry_on": [429, 502, 503, 504],
                    "backoff_ms": [100, 500, 1000, 2000, 5000]
                },
                "timeout": {
                    "connect_timeout": 5000,
                    "send_timeout": 30000,
                    "read_timeout": 30000
                },
                "rate_limit": {
                    "second": 100,
                    "minute": 1000,
                    "hour": 10000
                },
                "request_termination": {
                    "status_code": 503,
                    "message": "Service temporarily unavailable. Please retry."
                }
            },
            "high": {
                "retry": {
                    "max_retries": 3,
                    "retry_on": [502, 503, 504],
                    "backoff_ms": [100, 500, 1000]
                },
                "timeout": {
                    "connect_timeout": 3000,
                    "send_timeout": 20000,
                    "read_timeout": 20000
                },
                "rate_limit": {
                    "second": 50,
                    "minute": 500,
                    "hour": 5000
                },
                "request_termination": {
                    "status_code": 503,
                    "message": "Service temporarily unavailable."
                }
            },
            "medium": {
                "retry": {
                    "max_retries": 2,
                    "retry_on": [502, 503],
                    "backoff_ms": [100, 500]
                },
                "timeout": {
                    "connect_timeout": 2000,
                    "send_timeout": 10000,
                    "read_timeout": 10000
                },
                "rate_limit": {
                    "second": 20,
                    "minute": 200
                },
                "request_termination": {
                    "status_code": 503,
                    "message": "Service unavailable."
                }
            },
            "low": {
                "retry": {
                    "max_retries": 1,
                    "retry_on": [503],
                    "backoff_ms": [100]
                },
                "timeout": {
                    "connect_timeout": 1000,
                    "send_timeout": 5000,
                    "read_timeout": 5000
                },
                "rate_limit": {
                    "second": 10,
                    "minute": 100
                },
                "request_termination": {
                    "status_code": 503,
                    "message": "Service unavailable."
                }
            }
        }
        return configs.get(criticality, configs["medium"])
    
    def configure_retry_policy(self, service_name: str, config: Dict[str, Any]) -> bool:
        """Configure retry policy for a service."""
        try:
            plugin_config = {
                "name": "retry",
                "config": {
                    "max_retries": config["max_retries"],
                    "retry_on": config["retry_on"],
                    "backoff_ms": config["backoff_ms"]
                }
            }
            
            response = self.session.post(
                f"{self.kong_admin_url}/services/{service_name}-service/plugins",
                json=plugin_config
            )
            
            if response.status_code in [201, 409]:
                logger.info(f"Retry policy configured for {service_name}: "
                           f"{config['max_retries']} retries")
                return True
            else:
                logger.error(f"Failed to configure retry policy: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error configuring retry policy: {e}")
            return False
    
    def configure_timeout_policy(self, service_name: str, config: Dict[str, Any]) -> bool:
        """Configure timeout settings for a service."""
        try:
            # Update service timeouts
            service_update = {
                "connect_timeout": config["connect_timeout"],
                "write_timeout": config["send_timeout"],
                "read_timeout": config["read_timeout"]
            }
            
            response = self.session.patch(
                f"{self.kong_admin_url}/services/{service_name}-service",
                json=service_update
            )
            
            if response.status_code == 200:
                logger.info(f"Timeout policy configured for {service_name}: "
                           f"connect={config['connect_timeout']}ms, "
                           f"read={config['read_timeout']}ms")
                return True
            else:
                logger.error(f"Failed to configure timeout policy: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error configuring timeout policy: {e}")
            return False
    
    def configure_rate_limiting(self, service_name: str, config: Dict[str, Any]) -> bool:
        """Update rate limiting configuration for better fault tolerance."""
        try:
            # First, remove existing rate limiting plugin if any
            plugins_response = self.session.get(
                f"{self.kong_admin_url}/services/{service_name}-service/plugins"
            )
            
            if plugins_response.status_code == 200:
                plugins = plugins_response.json().get('data', [])
                for plugin in plugins:
                    if plugin['name'] == 'rate-limiting':
                        self.session.delete(
                            f"{self.kong_admin_url}/plugins/{plugin['id']}"
                        )
            
            # Add new rate limiting with updated config
            plugin_config = {
                "name": "rate-limiting",
                "config": {
                    "second": config.get("second"),
                    "minute": config.get("minute"),
                    "hour": config.get("hour"),
                    "policy": "local",
                    "fault_tolerant": True,
                    "hide_client_headers": False,
                    "redis_ssl": False,
                    "redis_ssl_verify": False
                }
            }
            
            # Remove None values
            plugin_config["config"] = {k: v for k, v in plugin_config["config"].items() 
                                      if v is not None}
            
            response = self.session.post(
                f"{self.kong_admin_url}/services/{service_name}-service/plugins",
                json=plugin_config
            )
            
            if response.status_code in [201]:
                logger.info(f"Rate limiting configured for {service_name}")
                return True
            else:
                logger.error(f"Failed to configure rate limiting: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error configuring rate limiting: {e}")
            return False
    
    def configure_request_termination(self, service_name: str, config: Dict[str, Any]) -> bool:
        """Configure request termination for overload protection."""
        try:
            # This plugin terminates requests when service is overloaded
            plugin_config = {
                "name": "request-termination",
                "config": {
                    "status_code": config["status_code"],
                    "message": config["message"],
                    "content_type": "text/plain"
                },
                "enabled": False  # Disabled by default, enabled during incidents
            }
            
            response = self.session.post(
                f"{self.kong_admin_url}/services/{service_name}-service/plugins",
                json=plugin_config
            )
            
            if response.status_code in [201, 409]:
                logger.info(f"Request termination configured for {service_name} (disabled)")
                return True
            else:
                logger.error(f"Failed to configure request termination: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error configuring request termination: {e}")
            return False
    
    def configure_proxy_cache(self, service_name: str, criticality: str) -> bool:
        """Configure proxy caching for resilience."""
        try:
            # Only cache for less critical services
            if criticality in ["low", "medium"]:
                cache_ttl = 300 if criticality == "low" else 60
                
                plugin_config = {
                    "name": "proxy-cache",
                    "config": {
                        "cache_ttl": cache_ttl,
                        "strategy": "memory",
                        "memory": {
                            "dictionary_name": f"cache_{service_name}"
                        },
                        "request_method": ["GET", "HEAD"],
                        "response_code": [200, 301, 302],
                        "content_type": ["application/json", "text/plain", "text/html"]
                    }
                }
                
                response = self.session.post(
                    f"{self.kong_admin_url}/services/{service_name}-service/plugins",
                    json=plugin_config
                )
                
                if response.status_code in [201, 409]:
                    logger.info(f"Proxy cache configured for {service_name}: TTL={cache_ttl}s")
                    return True
                else:
                    logger.error(f"Failed to configure proxy cache: {response.text}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error configuring proxy cache: {e}")
            return False
    
    def configure_response_transformer(self, service_name: str) -> bool:
        """Add response headers for better observability."""
        try:
            plugin_config = {
                "name": "response-transformer",
                "config": {
                    "add": {
                        "headers": [
                            f"X-Service-Name:{service_name}",
                            "X-Kong-Proxy:true",
                            "X-Response-Time:$(kong_response_time)"
                        ]
                    }
                }
            }
            
            response = self.session.post(
                f"{self.kong_admin_url}/services/{service_name}-service/plugins",
                json=plugin_config
            )
            
            if response.status_code in [201, 409]:
                logger.info(f"Response transformer configured for {service_name}")
                return True
            else:
                # Log but don't fail - this is not critical
                logger.warning(f"Could not configure response transformer: {response.text}")
                return True
                
        except Exception as e:
            logger.warning(f"Error configuring response transformer: {e}")
            return True
    
    def configure_correlation_id(self, service_name: str) -> bool:
        """Configure correlation ID for request tracing."""
        try:
            plugin_config = {
                "name": "correlation-id",
                "config": {
                    "header_name": "X-Request-ID",
                    "generator": "uuid",
                    "echo_downstream": True
                }
            }
            
            response = self.session.post(
                f"{self.kong_admin_url}/services/{service_name}-service/plugins",
                json=plugin_config
            )
            
            if response.status_code in [201, 409]:
                logger.info(f"Correlation ID configured for {service_name}")
                return True
            else:
                logger.warning(f"Could not configure correlation ID: {response.text}")
                return True
                
        except Exception as e:
            logger.warning(f"Error configuring correlation ID: {e}")
            return True
    
    def get_all_services(self) -> List[str]:
        """Get all services registered in Kong."""
        try:
            response = self.session.get(f"{self.kong_admin_url}/services")
            if response.status_code == 200:
                services = response.json().get('data', [])
                return [s['name'].replace('-service', '') for s in services]
            return []
        except Exception as e:
            logger.error(f"Error fetching services: {e}")
            return []
    
    def configure_all_services(self) -> Dict[str, Any]:
        """Configure fault tolerance for all services."""
        results = {
            "configured": 0,
            "failed": 0,
            "services": []
        }
        
        services = self.get_all_services()
        
        for service_name in services:
            logger.info(f"\nConfiguring fault tolerance for {service_name}...")
            
            # Determine criticality
            criticality = self.service_criticality.get(service_name, "medium")
            config = self.get_criticality_config(criticality)
            
            logger.info(f"Service criticality: {criticality}")
            
            success = True
            
            # Configure retry policy
            if not self.configure_retry_policy(service_name, config["retry"]):
                success = False
            
            # Configure timeouts
            if not self.configure_timeout_policy(service_name, config["timeout"]):
                success = False
            
            # Configure rate limiting
            if not self.configure_rate_limiting(service_name, config["rate_limit"]):
                success = False
            
            # Configure request termination (for overload protection)
            if not self.configure_request_termination(service_name, config["request_termination"]):
                success = False
            
            # Configure proxy cache (for less critical services)
            if not self.configure_proxy_cache(service_name, criticality):
                success = False
            
            # Configure response transformer
            self.configure_response_transformer(service_name)
            
            # Configure correlation ID
            self.configure_correlation_id(service_name)
            
            if success:
                results["configured"] += 1
                results["services"].append({
                    "name": service_name,
                    "criticality": criticality,
                    "retry_enabled": True,
                    "timeout_configured": True,
                    "rate_limit_enabled": True,
                    "cache_enabled": criticality in ["low", "medium"]
                })
            else:
                results["failed"] += 1
        
        return results


def main():
    """Main function to configure fault tolerance."""
    parser = argparse.ArgumentParser(description="Configure fault tolerance for SutazAI service mesh")
    parser.add_argument("--kong-admin", default="http://localhost:10007", help="Kong Admin API URL")
    
    args = parser.parse_args()
    
    # Create configurator instance
    configurator = FaultToleranceConfigurator(args.kong_admin)
    
    logger.info("Starting SutazAI Fault Tolerance Configuration")
    logger.info("=" * 60)
    
    # Configure all services
    results = configurator.configure_all_services()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("CONFIGURATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Services Configured: {results['configured']}")
    logger.info(f"Failed: {results['failed']}")
    
    if results['services']:
        logger.info("\nConfigured Services:")
        for service in results['services']:
            features = []
            if service['retry_enabled']:
                features.append("retry")
            if service['timeout_configured']:
                features.append("timeout")
            if service['rate_limit_enabled']:
                features.append("rate-limit")
            if service['cache_enabled']:
                features.append("cache")
            
            logger.info(f"  - {service['name']} ({service['criticality']}): "
                       f"{', '.join(features)}")
    
    # Save results
    with open("/opt/sutazaiapp/reports/fault-tolerance-configuration.json", 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("\nConfiguration results saved to: /opt/sutazaiapp/reports/fault-tolerance-configuration.json")
    
    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    exit(main())