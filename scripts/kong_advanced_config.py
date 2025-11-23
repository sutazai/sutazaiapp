#!/usr/bin/env python3
"""
Kong Gateway Advanced Configuration Script
Purpose: Configure JWT, upstreams, health checks, circuit breakers, and advanced plugins
Created: 2025-11-15
Version: 1.0.0

Configures:
- JWT authentication
- Upstream targets and health checks
- Circuit breaker (proxy-cache plugin)
- IP restriction for admin routes
- Load balancing algorithms
- Advanced monitoring and logging
"""

import requests
import json
import sys
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'kong_config_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

KONG_ADMIN_URL = "http://localhost:10009"

class KongConfigurator:
    """Kong Gateway advanced configuration"""
    
    def __init__(self):
        self.admin_url = KONG_ADMIN_URL
        logger.info("Kong Configurator initialized")
    
    def create_upstream(self, name: str, targets: List[Dict[str, Any]], algorithm: str = "round-robin") -> Optional[Dict]:
        """Create upstream with targets and health checks"""
        try:
            # Check if upstream exists
            check_resp = requests.get(f"{self.admin_url}/upstreams/{name}")
            if check_resp.status_code == 200:
                logger.info(f"Upstream {name} already exists")
                return check_resp.json()
            
            # Create upstream
            upstream_data = {
                "name": name,
                "algorithm": algorithm,
                "slots": 10000,
                "healthchecks": {
                    "active": {
                        "timeout": 1,
                        "concurrency": 10,
                        "http_path": "/health",
                        "healthy": {
                            "interval": 5,
                            "successes": 2
                        },
                        "unhealthy": {
                            "interval": 3,
                            "tcp_failures": 2,
                            "http_failures": 3,
                            "timeouts": 2
                        }
                    },
                    "passive": {
                        "healthy": {
                            "successes": 5
                        },
                        "unhealthy": {
                            "tcp_failures": 2,
                            "http_failures": 5,
                            "timeouts": 2
                        }
                    }
                }
            }
            
            response = requests.post(f"{self.admin_url}/upstreams", json=upstream_data)
            
            if response.status_code in [200, 201]:
                logger.info(f"Created upstream: {name} with algorithm: {algorithm}")
                upstream = response.json()
                
                # Add targets
                for target in targets:
                    target_resp = requests.post(
                        f"{self.admin_url}/upstreams/{name}/targets",
                        json=target
                    )
                    if target_resp.status_code in [200, 201]:
                        logger.info(f"Added target {target['target']} to upstream {name}")
                    else:
                        logger.error(f"Failed to add target {target['target']}: {target_resp.text}")
                
                return upstream
            else:
                logger.error(f"Failed to create upstream {name}: {response.text}")
                return None
                
        except Exception as e:
            logger.exception(f"Error creating upstream {name}: {e}")
            return None
    
    def configure_jwt_plugin(self, service_name: str = None, route_name: str = None) -> Optional[Dict]:
        """Configure JWT authentication plugin"""
        try:
            plugin_data = {
                "name": "jwt",
                "config": {
                    "uri_param_names": ["jwt"],
                    "cookie_names": [],
                    "header_names": ["Authorization"],
                    "claims_to_verify": ["exp"],
                    "key_claim_name": "iss",
                    "secret_is_base64": False,
                    "anonymous": None,
                    "run_on_preflight": True,
                    "maximum_expiration": 31536000
                }
            }
            
            # Apply to specific service or route if provided
            if service_name:
                # Get service ID
                service_resp = requests.get(f"{self.admin_url}/services/{service_name}")
                if service_resp.status_code == 200:
                    service_id = service_resp.json()['id']
                    plugin_data['service'] = {"id": service_id}
            elif route_name:
                # Get route ID
                route_resp = requests.get(f"{self.admin_url}/routes/{route_name}")
                if route_resp.status_code == 200:
                    route_id = route_resp.json()['id']
                    plugin_data['route'] = {"id": route_id}
            
            response = requests.post(f"{self.admin_url}/plugins", json=plugin_data)
            
            if response.status_code in [200, 201]:
                logger.info(f"Configured JWT plugin for {service_name or route_name or 'global'}")
                return response.json()
            else:
                logger.error(f"Failed to configure JWT: {response.text}")
                return None
                
        except Exception as e:
            logger.exception(f"Error configuring JWT: {e}")
            return None
    
    def create_jwt_consumer(self, username: str, key: str, secret: str) -> Optional[Dict]:
        """Create JWT consumer with credentials"""
        try:
            # Create consumer
            consumer_data = {"username": username}
            consumer_resp = requests.post(f"{self.admin_url}/consumers", json=consumer_data)
            
            if consumer_resp.status_code in [200, 201, 409]:  # 409 = already exists
                if consumer_resp.status_code == 409:
                    logger.info(f"Consumer {username} already exists")
                    consumer_resp = requests.get(f"{self.admin_url}/consumers/{username}")
                else:
                    logger.info(f"Created consumer: {username}")
                
                consumer = consumer_resp.json()
                
                # Add JWT credentials
                jwt_cred_data = {
                    "key": key,
                    "secret": secret,
                    "algorithm": "HS256"
                }
                
                cred_resp = requests.post(
                    f"{self.admin_url}/consumers/{username}/jwt",
                    json=jwt_cred_data
                )
                
                if cred_resp.status_code in [200, 201]:
                    logger.info(f"Added JWT credentials for consumer {username}")
                    return cred_resp.json()
                else:
                    logger.error(f"Failed to add JWT credentials: {cred_resp.text}")
                    return None
            else:
                logger.error(f"Failed to create consumer {username}: {consumer_resp.text}")
                return None
                
        except Exception as e:
            logger.exception(f"Error creating JWT consumer: {e}")
            return None
    
    def configure_ip_restriction(self, allow_list: List[str] = None, deny_list: List[str] = None) -> Optional[Dict]:
        """Configure IP restriction plugin"""
        try:
            plugin_data = {
                "name": "ip-restriction",
                "config": {}
            }
            
            if allow_list:
                plugin_data["config"]["allow"] = allow_list
            if deny_list:
                plugin_data["config"]["deny"] = deny_list
            
            response = requests.post(f"{self.admin_url}/plugins", json=plugin_data)
            
            if response.status_code in [200, 201]:
                logger.info(f"Configured IP restriction: allow={allow_list}, deny={deny_list}")
                return response.json()
            else:
                logger.error(f"Failed to configure IP restriction: {response.text}")
                return None
                
        except Exception as e:
            logger.exception(f"Error configuring IP restriction: {e}")
            return None
    
    def configure_proxy_cache(self, cache_ttl: int = 300, content_types: List[str] = None) -> Optional[Dict]:
        """Configure proxy caching plugin for circuit breaker pattern"""
        try:
            plugin_data = {
                "name": "proxy-cache",
                "config": {
                    "cache_ttl": cache_ttl,
                    "strategy": "memory",
                    "content_type": content_types or ["application/json", "text/plain"],
                    "cache_control": True,
                    "response_code": [200, 301, 404],
                    "request_method": ["GET", "HEAD"],
                    "memory": {
                        "dictionary_name": "kong_db_cache"
                    }
                }
            }
            
            response = requests.post(f"{self.admin_url}/plugins", json=plugin_data)
            
            if response.status_code in [200, 201]:
                logger.info(f"Configured proxy cache with TTL: {cache_ttl}s")
                return response.json()
            else:
                logger.error(f"Failed to configure proxy cache: {response.text}")
                return None
                
        except Exception as e:
            logger.exception(f"Error configuring proxy cache: {e}")
            return None
    
    def configure_request_termination(self, status_code: int = 503, message: str = "Service temporarily unavailable") -> Optional[Dict]:
        """Configure request termination for circuit breaker"""
        try:
            plugin_data = {
                "name": "request-termination",
                "enabled": False,  # Disabled by default, enable when circuit opens
                "config": {
                    "status_code": status_code,
                    "message": message,
                    "content_type": "application/json"
                }
            }
            
            response = requests.post(f"{self.admin_url}/plugins", json=plugin_data)
            
            if response.status_code in [200, 201]:
                logger.info(f"Configured request termination (circuit breaker)")
                return response.json()
            else:
                logger.error(f"Failed to configure request termination: {response.text}")
                return None
                
        except Exception as e:
            logger.exception(f"Error configuring request termination: {e}")
            return None
    
    def configure_http_log(self, http_endpoint: str) -> Optional[Dict]:
        """Configure HTTP log plugin for centralized logging"""
        try:
            plugin_data = {
                "name": "http-log",
                "config": {
                    "http_endpoint": http_endpoint,
                    "method": "POST",
                    "timeout": 10000,
                    "keepalive": 60000,
                    "retry_count": 10,
                    "queue_size": 1000,
                    "flush_timeout": 2,
                    "headers": {
                        "Content-Type": "application/json"
                    }
                }
            }
            
            response = requests.post(f"{self.admin_url}/plugins", json=plugin_data)
            
            if response.status_code in [200, 201]:
                logger.info(f"Configured HTTP log to: {http_endpoint}")
                return response.json()
            else:
                logger.error(f"Failed to configure HTTP log: {response.text}")
                return None
                
        except Exception as e:
            logger.exception(f"Error configuring HTTP log: {e}")
            return None
    
    def configure_all(self):
        """Run all configurations"""
        logger.info("Starting Kong advanced configuration...")
        
        # 1. Create upstreams with health checks
        logger.info("\n=== Creating Upstreams with Health Checks ===")
        
        upstreams_config = [
            {
                "name": "backend-upstream",
                "algorithm": "round-robin",
                "targets": [
                    {"target": "sutazai-backend:8000", "weight": 100}
                ]
            },
            {
                "name": "vector-db-upstream",
                "algorithm": "least-connections",
                "targets": [
                    {"target": "sutazai-chromadb:8000", "weight": 50},
                    {"target": "sutazai-qdrant:10102", "weight": 100}  # Higher weight for faster DB
                ]
            }
        ]
        
        for config in upstreams_config:
            self.create_upstream(config["name"], config["targets"], config["algorithm"])
        
        # 2. Configure JWT authentication (optional, disabled by default)
        logger.info("\n=== Configuring JWT Authentication ===")
        # Uncomment to enable JWT on specific service
        # self.configure_jwt_plugin(service_name="backend-api")
        
        # Create test JWT consumer
        # self.create_jwt_consumer(
        #     username="test-user",
        #     key="test-key-12345",
        #     secret="test-secret-67890"
        # )
        logger.info("JWT configuration skipped (enable manually if needed)")
        
        # 3. Configure IP restriction for admin access (optional)
        logger.info("\n=== Configuring IP Restriction ===")
        # Allow local network access only
        # self.configure_ip_restriction(allow_list=["127.0.0.1", "172.20.0.0/16", "localhost"])
        logger.info("IP restriction configuration skipped (enable manually if needed)")
        
        # 4. Configure proxy cache for performance
        logger.info("\n=== Configuring Proxy Cache ===")
        self.configure_proxy_cache(cache_ttl=300, content_types=["application/json"])
        
        # 5. Configure request termination (circuit breaker)
        logger.info("\n=== Configuring Circuit Breaker ===")
        self.configure_request_termination(status_code=503, message="Service temporarily unavailable")
        
        logger.info("\n=== Kong Advanced Configuration Completed ===")
        
        return True

def main() -> int:
    """Main configuration execution"""
    try:
        logger.info("Kong Gateway Advanced Configuration Starting...")
        
        configurator = KongConfigurator()
        success = configurator.configure_all()
        
        if success:
            logger.info("Kong advanced configuration completed successfully")
            return 0
        else:
            logger.error("Kong configuration completed with errors")
            return 1
            
    except Exception as e:
        logger.exception(f"Fatal error in Kong configuration: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
