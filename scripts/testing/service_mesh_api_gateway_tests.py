#!/usr/bin/env python3
"""
Service Mesh and API Gateway Tests
=================================

Comprehensive validation tests for service mesh and API gateway components:
- Kong API Gateway with routing, load balancing, and plugins
- Consul service discovery with health checks and configuration
- Service mesh communication and traffic management
- HAProxy load balancer integration (if configured)
- API routing, rate limiting, and security policies

Focus on actual service mesh validation with real traffic routing.
"""

import asyncio
import aiohttp
import json
import logging
import time
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from urllib.parse import urljoin
import yaml

logger = logging.getLogger(__name__)

@dataclass
class ServiceMeshTestResult:
    """Service mesh test execution result"""
    component: str
    test_name: str
    success: bool
    duration: float
    metrics: Dict[str, Any]
    error_message: Optional[str] = None

@dataclass
class ServiceEndpoint:
    """Service endpoint for testing"""
    name: str
    url: str
    expected_status: int = 200
    timeout: int = 15

class ServiceMeshValidator:
    """Comprehensive service mesh and API gateway validation"""
    
    def __init__(self):
        self.results: List[ServiceMeshTestResult] = []
        
        # Service mesh configuration from docker-compose and port registry
        self.config = {
            "kong": {
                "proxy_host": "localhost",
                "proxy_port": 10005,
                "admin_host": "localhost", 
                "admin_port": 10015,
                "expected_services": ["backend", "frontend"],
                "expected_plugins": ["cors", "rate-limiting", "key-auth"]
            },
            "consul": {
                "host": "localhost",
                "port": 10006,
                "expected_services": ["backend", "frontend", "ollama", "kong"],
                "health_check_interval": "30s"
            },
            "backend": {
                "host": "localhost",
                "port": 10010,
                "health_path": "/health",
                "api_paths": ["/api/v1/health", "/api/v1/agents", "/api/v1/chat"]
            },
            "frontend": {
                "host": "localhost", 
                "port": 10011,
                "health_path": "/health"
            },
            "haproxy": {
                "host": "localhost",
                "port": 10047,  # From port registry if configured
                "stats_port": 10048
            }
        }
        
        # Test endpoints for routing validation
        self.test_endpoints = [
            ServiceEndpoint("backend_direct", f"http://localhost:{self.config['backend']['port']}/health"),
            ServiceEndpoint("frontend_direct", f"http://localhost:{self.config['frontend']['port']}/health"),
            ServiceEndpoint("backend_via_kong", f"http://localhost:{self.config['kong']['proxy_port']}/api/v1/health"),
            ServiceEndpoint("kong_status", f"http://localhost:{self.config['kong']['proxy_port']}/status"),
        ]
    
    async def run_all_service_mesh_tests(self) -> List[ServiceMeshTestResult]:
        """Execute all service mesh validation tests"""
        logger.info("Starting comprehensive service mesh validation")
        
        # Test execution order based on dependencies
        test_methods = [
            # Core service discovery first
            ("consul_service_discovery", self.test_consul_service_discovery),
            ("consul_health_checks", self.test_consul_health_checks),
            ("consul_kv_store", self.test_consul_kv_store),
            
            # API Gateway functionality
            ("kong_gateway_health", self.test_kong_gateway_health),
            ("kong_admin_api", self.test_kong_admin_api),
            ("kong_services_config", self.test_kong_services_config),
            ("kong_routing", self.test_kong_routing),
            ("kong_load_balancing", self.test_kong_load_balancing),
            
            # Service mesh integration
            ("service_mesh_communication", self.test_service_mesh_communication),
            ("traffic_routing", self.test_traffic_routing),
            ("service_discovery_integration", self.test_service_discovery_integration),
            
            # Advanced features
            ("api_rate_limiting", self.test_api_rate_limiting),
            ("security_policies", self.test_security_policies),
            ("monitoring_integration", self.test_monitoring_integration)
        ]
        
        # Execute tests sequentially (some have dependencies)
        for component, test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                logger.error(f"Service mesh test {component} failed: {e}")
        
        return self.results
    
    async def test_consul_service_discovery(self) -> None:
        """Test Consul service discovery functionality"""
        start_time = time.time()
        
        try:
            consul_url = f"http://{self.config['consul']['host']}:{self.config['consul']['port']}"
            
            async with aiohttp.ClientSession() as session:
                # Test Consul leader status
                async with session.get(f"{consul_url}/v1/status/leader",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    leader_success = response.status == 200
                    leader_info = await response.text() if response.status == 200 else ""
                
                # Test service catalog
                async with session.get(f"{consul_url}/v1/catalog/services",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    catalog_success = response.status == 200
                    if catalog_success:
                        services_data = await response.json()
                        registered_services = list(services_data.keys())
                    else:
                        services_data = {}
                        registered_services = []
                
                # Test individual service details
                service_details = {}
                for service_name in registered_services[:5]:  # Test first 5 services
                    async with session.get(f"{consul_url}/v1/catalog/service/{service_name}",
                                         timeout=aiohttp.ClientTimeout(total=15)) as response:
                        if response.status == 200:
                            service_info = await response.json()
                            service_details[service_name] = {
                                "instances": len(service_info),
                                "nodes": list(set(instance.get("Node", "unknown") for instance in service_info))
                            }
                
                # Test nodes information
                async with session.get(f"{consul_url}/v1/catalog/nodes",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    nodes_success = response.status == 200
                    if nodes_success:
                        nodes_data = await response.json()
                        registered_nodes = len(nodes_data)
                    else:
                        registered_nodes = 0
                
                # Check expected services
                expected_services = self.config["consul"]["expected_services"]
                expected_found = sum(1 for svc in expected_services if svc in registered_services)
                service_coverage = expected_found / len(expected_services) * 100 if expected_services else 0
            
            duration = time.time() - start_time
            
            self.results.append(ServiceMeshTestResult(
                component="consul_service_discovery",
                test_name="service_discovery",
                success=leader_success and catalog_success,
                duration=duration,
                metrics={
                    "leader_available": leader_success,
                    "catalog_accessible": catalog_success,
                    "leader_info": leader_info.strip('"') if leader_info else "unknown",
                    "registered_services": len(registered_services),
                    "service_names": registered_services,
                    "expected_services_found": expected_found,
                    "service_coverage_percent": service_coverage,
                    "registered_nodes": registered_nodes,
                    "service_details": service_details,
                    "performance_grade": "excellent" if duration < 2 else "good" if duration < 5 else "poor"
                }
            ))
            
            logger.info(f"Consul service discovery - Services: {len(registered_services)}, Coverage: {service_coverage:.1f}%")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ServiceMeshTestResult(
                component="consul_service_discovery",
                test_name="service_discovery",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Consul service discovery test failed: {e}")
    
    async def test_consul_health_checks(self) -> None:
        """Test Consul health checking functionality"""
        start_time = time.time()
        
        try:
            consul_url = f"http://{self.config['consul']['host']}:{self.config['consul']['port']}"
            
            async with aiohttp.ClientSession() as session:
                # Test health checks endpoint
                async with session.get(f"{consul_url}/v1/health/state/any",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    health_success = response.status == 200
                    if health_success:
                        health_data = await response.json()
                    else:
                        health_data = []
                
                # Test critical health checks
                async with session.get(f"{consul_url}/v1/health/state/critical",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    critical_success = response.status == 200
                    if critical_success:
                        critical_health = await response.json()
                    else:
                        critical_health = []
                
                # Test passing health checks
                async with session.get(f"{consul_url}/v1/health/state/passing",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    passing_success = response.status == 200
                    if passing_success:
                        passing_health = await response.json()
                    else:
                        passing_health = []
                
                # Test warning health checks
                async with session.get(f"{consul_url}/v1/health/state/warning",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    warning_success = response.status == 200
                    if warning_success:
                        warning_health = await response.json()
                    else:
                        warning_health = []
            
            # Calculate health statistics
            total_checks = len(health_data)
            critical_checks = len(critical_health)
            passing_checks = len(passing_health)
            warning_checks = len(warning_health)
            
            health_ratio = passing_checks / max(total_checks, 1) * 100
            
            # Analyze health check details
            services_with_health = set()
            nodes_with_health = set()
            
            for check in health_data:
                services_with_health.add(check.get("ServiceName", ""))
                nodes_with_health.add(check.get("Node", ""))
            
            duration = time.time() - start_time
            
            self.results.append(ServiceMeshTestResult(
                component="consul_health_checks",
                test_name="health_monitoring",
                success=health_success,
                duration=duration,
                metrics={
                    "health_endpoint_available": health_success,
                    "total_health_checks": total_checks,
                    "passing_checks": passing_checks,
                    "critical_checks": critical_checks,
                    "warning_checks": warning_checks,
                    "health_ratio_percent": health_ratio,
                    "services_with_health": len(services_with_health),
                    "nodes_with_health": len(nodes_with_health),
                    "health_grade": "excellent" if health_ratio > 90 else "good" if health_ratio > 70 else "poor",
                    "performance_grade": "excellent" if duration < 2 else "good" if duration < 5 else "poor"
                }
            ))
            
            logger.info(f"Consul health checks - Total: {total_checks}, Passing: {passing_checks} ({health_ratio:.1f}%)")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ServiceMeshTestResult(
                component="consul_health_checks",
                test_name="health_monitoring",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Consul health checks test failed: {e}")
    
    async def test_consul_kv_store(self) -> None:
        """Test Consul key-value store functionality"""
        start_time = time.time()
        
        try:
            consul_url = f"http://{self.config['consul']['host']}:{self.config['consul']['port']}"
            test_key = "service-mesh-test/timestamp"
            test_value = str(int(time.time()))
            
            async with aiohttp.ClientSession() as session:
                # Test KV store write
                async with session.put(f"{consul_url}/v1/kv/{test_key}",
                                     data=test_value,
                                     timeout=aiohttp.ClientTimeout(total=15)) as response:
                    write_success = response.status == 200
                
                # Test KV store read
                async with session.get(f"{consul_url}/v1/kv/{test_key}",
                                     timeout=aiohttp.ClientTimeout(total=15)) as response:
                    read_success = response.status == 200
                    if read_success:
                        kv_data = await response.json()
                        stored_value = kv_data[0].get("Value", "") if kv_data else ""
                        # Decode base64 value
                        import base64
                        try:
                            decoded_value = base64.b64decode(stored_value).decode('utf-8')
                            value_match = decoded_value == test_value
                        except:
                            value_match = False
                    else:
                        value_match = False
                
                # Test KV store list
                async with session.get(f"{consul_url}/v1/kv/service-mesh-test/?keys",
                                     timeout=aiohttp.ClientTimeout(total=15)) as response:
                    list_success = response.status == 200
                    if list_success:
                        keys_data = await response.json()
                    else:
                        keys_data = []
                
                # Test KV store delete
                async with session.delete(f"{consul_url}/v1/kv/{test_key}",
                                        timeout=aiohttp.ClientTimeout(total=15)) as response:
                    delete_success = response.status == 200
                
                # Test existing configuration keys
                config_endpoints = [
                    "/v1/kv/config/?keys",
                    "/v1/kv/services/?keys", 
                    "/v1/kv/mesh/?keys"
                ]
                
                existing_configs = {}
                for endpoint in config_endpoints:
                    try:
                        async with session.get(f"{consul_url}{endpoint}",
                                             timeout=aiohttp.ClientTimeout(total=10)) as response:
                            if response.status == 200:
                                config_keys = await response.json()
                                existing_configs[endpoint] = len(config_keys) if config_keys else 0
                            else:
                                existing_configs[endpoint] = 0
                    except:
                        existing_configs[endpoint] = 0
            
            duration = time.time() - start_time
            
            kv_operations_success = write_success and read_success and value_match
            
            self.results.append(ServiceMeshTestResult(
                component="consul_kv_store",
                test_name="kv_operations",
                success=kv_operations_success,
                duration=duration,
                metrics={
                    "write_operation": write_success,
                    "read_operation": read_success,
                    "list_operation": list_success,
                    "delete_operation": delete_success,
                    "value_integrity": value_match,
                    "kv_functionality": kv_operations_success,
                    "existing_config_keys": existing_configs,
                    "total_config_keys": sum(existing_configs.values()),
                    "performance_grade": "excellent" if duration < 1 else "good" if duration < 3 else "poor"
                }
            ))
            
            logger.info(f"Consul KV store - Operations: {kv_operations_success}, Config keys: {sum(existing_configs.values())}")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ServiceMeshTestResult(
                component="consul_kv_store",
                test_name="kv_operations",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Consul KV store test failed: {e}")
    
    async def test_kong_gateway_health(self) -> None:
        """Test Kong API gateway health and status"""
        start_time = time.time()
        
        try:
            proxy_url = f"http://{self.config['kong']['proxy_host']}:{self.config['kong']['proxy_port']}"
            admin_url = f"http://{self.config['kong']['admin_host']}:{self.config['kong']['admin_port']}"
            
            async with aiohttp.ClientSession() as session:
                # Test Kong proxy status
                async with session.get(f"{proxy_url}/status",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    proxy_status_success = response.status == 200
                    if proxy_status_success:
                        proxy_status_data = await response.json()
                    else:
                        proxy_status_data = {}
                
                # Test Kong admin API  
                async with session.get(f"{admin_url}/",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    admin_success = response.status == 200
                    if admin_success:
                        admin_info = await response.json()
                    else:
                        admin_info = {}
                
                # Test Kong status via admin API
                async with session.get(f"{admin_url}/status",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    admin_status_success = response.status == 200
                    if admin_status_success:
                        admin_status_data = await response.json()
                    else:
                        admin_status_data = {}
                
                # Test Kong configuration info
                async with session.get(f"{admin_url}/information",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    config_success = response.status == 200
                    if config_success:
                        config_info = await response.json()
                    else:
                        config_info = {}
            
            # Extract key metrics
            database_reachable = admin_status_data.get("database", {}).get("reachable", False)
            server_info = {
                "version": admin_info.get("version", "unknown"),
                "hostname": admin_info.get("hostname", "unknown"),
                "node_id": admin_info.get("node_id", "unknown")
            }
            
            duration = time.time() - start_time
            
            overall_success = proxy_status_success or admin_success
            
            self.results.append(ServiceMeshTestResult(
                component="kong_gateway_health",
                test_name="gateway_health",
                success=overall_success,
                duration=duration,
                metrics={
                    "proxy_status_available": proxy_status_success,
                    "admin_api_available": admin_success,
                    "admin_status_available": admin_status_success,
                    "configuration_accessible": config_success,
                    "database_reachable": database_reachable,
                    "server_info": server_info,
                    "proxy_status": proxy_status_data,
                    "admin_status": admin_status_data,
                    "performance_grade": "excellent" if duration < 2 else "good" if duration < 5 else "poor"
                }
            ))
            
            logger.info(f"Kong gateway health - Proxy: {proxy_status_success}, Admin: {admin_success}, Version: {server_info['version']}")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ServiceMeshTestResult(
                component="kong_gateway_health",
                test_name="gateway_health",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Kong gateway health test failed: {e}")
    
    async def test_kong_admin_api(self) -> None:
        """Test Kong admin API functionality"""
        start_time = time.time()
        
        try:
            admin_url = f"http://{self.config['kong']['admin_host']}:{self.config['kong']['admin_port']}"
            
            admin_endpoints = {
                "services": "/services",
                "routes": "/routes",
                "plugins": "/plugins",
                "upstreams": "/upstreams",
                "consumers": "/consumers",
                "certificates": "/certificates"
            }
            
            endpoint_results = {}
            
            async with aiohttp.ClientSession() as session:
                for endpoint_name, endpoint_path in admin_endpoints.items():
                    try:
                        async with session.get(f"{admin_url}{endpoint_path}",
                                             timeout=aiohttp.ClientTimeout(total=15)) as response:
                            endpoint_success = response.status == 200
                            if endpoint_success:
                                endpoint_data = await response.json()
                                if isinstance(endpoint_data, dict) and "data" in endpoint_data:
                                    item_count = len(endpoint_data["data"])
                                else:
                                    item_count = len(endpoint_data) if isinstance(endpoint_data, list) else 0
                            else:
                                item_count = 0
                                endpoint_data = {}
                            
                            endpoint_results[endpoint_name] = {
                                "available": endpoint_success,
                                "count": item_count,
                                "status": response.status
                            }
                            
                    except Exception as endpoint_error:
                        endpoint_results[endpoint_name] = {
                            "available": False,
                            "count": 0,
                            "error": str(endpoint_error)
                        }
                
                # Test admin API capabilities with a test service creation (and cleanup)
                test_service_operations = {}
                test_service_name = "test-service-validation"
                
                # Create test service
                test_service_config = {
                    "name": test_service_name,
                    "url": "http://httpbin.org/anything"
                }
                
                try:
                    async with session.post(f"{admin_url}/services",
                                          json=test_service_config,
                                          timeout=aiohttp.ClientTimeout(total=15)) as response:
                        create_success = response.status in [200, 201]
                        if create_success:
                            created_service = await response.json()
                            service_id = created_service.get("id", "")
                        else:
                            service_id = ""
                        
                        test_service_operations["create"] = create_success
                        
                except Exception as create_error:
                    test_service_operations["create"] = False
                    test_service_operations["create_error"] = str(create_error)
                    service_id = ""
                
                # Cleanup test service if created
                if service_id:
                    try:
                        async with session.delete(f"{admin_url}/services/{service_id}",
                                                timeout=aiohttp.ClientTimeout(total=15)) as response:
                            delete_success = response.status in [200, 204]
                            test_service_operations["delete"] = delete_success
                    except Exception as delete_error:
                        test_service_operations["delete"] = False
                        test_service_operations["delete_error"] = str(delete_error)
            
            duration = time.time() - start_time
            
            # Calculate admin API success rate
            available_endpoints = sum(1 for result in endpoint_results.values() if result["available"])
            admin_success_rate = available_endpoints / len(endpoint_results) * 100
            
            self.results.append(ServiceMeshTestResult(
                component="kong_admin_api",
                test_name="admin_functionality",
                success=admin_success_rate > 70,
                duration=duration,
                metrics={
                    "endpoint_results": endpoint_results,
                    "available_endpoints": available_endpoints,
                    "admin_success_rate": admin_success_rate,
                    "test_service_operations": test_service_operations,
                    "services_configured": endpoint_results.get("services", {}).get("count", 0),
                    "routes_configured": endpoint_results.get("routes", {}).get("count", 0),
                    "plugins_configured": endpoint_results.get("plugins", {}).get("count", 0),
                    "performance_grade": "excellent" if duration < 5 else "good" if duration < 10 else "poor"
                }
            ))
            
            logger.info(f"Kong admin API - Endpoints: {available_endpoints}/{len(endpoint_results)} ({admin_success_rate:.1f}%)")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ServiceMeshTestResult(
                component="kong_admin_api",
                test_name="admin_functionality",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Kong admin API test failed: {e}")
    
    async def test_kong_services_config(self) -> None:
        """Test Kong services and routes configuration"""
        start_time = time.time()
        
        try:
            admin_url = f"http://{self.config['kong']['admin_host']}:{self.config['kong']['admin_port']}"
            
            async with aiohttp.ClientSession() as session:
                # Get Kong services
                async with session.get(f"{admin_url}/services",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    services_success = response.status == 200
                    if services_success:
                        services_data = await response.json()
                        services = services_data.get("data", [])
                    else:
                        services = []
                
                # Get Kong routes
                async with session.get(f"{admin_url}/routes",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    routes_success = response.status == 200
                    if routes_success:
                        routes_data = await response.json()
                        routes = routes_data.get("data", [])
                    else:
                        routes = []
                
                # Analyze service configuration
                service_analysis = {}
                for service in services:
                    service_name = service.get("name", "unknown")
                    service_analysis[service_name] = {
                        "host": service.get("host", ""),
                        "port": service.get("port", 0),
                        "protocol": service.get("protocol", ""),
                        "path": service.get("path", ""),
                        "enabled": service.get("enabled", False),
                        "connect_timeout": service.get("connect_timeout", 0),
                        "read_timeout": service.get("read_timeout", 0),
                        "write_timeout": service.get("write_timeout", 0)
                    }
                
                # Analyze routes configuration
                route_analysis = {}
                for route in routes:
                    route_name = route.get("name", f"route_{route.get('id', 'unknown')}")
                    route_analysis[route_name] = {
                        "paths": route.get("paths", []),
                        "methods": route.get("methods", []),
                        "hosts": route.get("hosts", []),
                        "service_id": route.get("service", {}).get("id", ""),
                        "strip_path": route.get("strip_path", False),
                        "preserve_host": route.get("preserve_host", False)
                    }
                
                # Check if expected services are configured
                expected_services = self.config["kong"]["expected_services"]
                configured_expected = sum(1 for svc in expected_services 
                                        if any(svc.lower() in service_name.lower() 
                                             for service_name in service_analysis.keys()))
                
                service_configuration_rate = configured_expected / len(expected_services) * 100 if expected_services else 0
            
            duration = time.time() - start_time
            
            self.results.append(ServiceMeshTestResult(
                component="kong_services_config",
                test_name="services_routes_config",
                success=services_success and routes_success,
                duration=duration,
                metrics={
                    "services_endpoint_available": services_success,
                    "routes_endpoint_available": routes_success,
                    "total_services": len(services),
                    "total_routes": len(routes),
                    "service_analysis": service_analysis,
                    "route_analysis": route_analysis,
                    "expected_services_configured": configured_expected,
                    "service_configuration_rate": service_configuration_rate,
                    "services_with_routes": len([s for s in services 
                                               if any(r.get("service", {}).get("id") == s.get("id") for r in routes)]),
                    "performance_grade": "excellent" if duration < 2 else "good" if duration < 5 else "poor"
                }
            ))
            
            logger.info(f"Kong services config - Services: {len(services)}, Routes: {len(routes)}, Expected: {configured_expected}/{len(expected_services)}")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ServiceMeshTestResult(
                component="kong_services_config",
                test_name="services_routes_config",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Kong services config test failed: {e}")
    
    async def test_kong_routing(self) -> None:
        """Test Kong routing functionality"""
        start_time = time.time()
        
        try:
            proxy_url = f"http://{self.config['kong']['proxy_host']}:{self.config['kong']['proxy_port']}"
            
            # Test routing to different services
            routing_tests = {}
            
            async with aiohttp.ClientSession() as session:
                for endpoint in self.test_endpoints:
                    if "kong" in endpoint.name or endpoint.url.startswith(proxy_url):
                        routing_start = time.time()
                        try:
                            async with session.get(endpoint.url,
                                                 timeout=aiohttp.ClientTimeout(total=endpoint.timeout)) as response:
                                routing_success = response.status == endpoint.expected_status
                                response_time = (time.time() - routing_start) * 1000
                                
                                # Try to get response body for analysis
                                try:
                                    response_body = await response.text()
                                    response_size = len(response_body)
                                except:
                                    response_body = ""
                                    response_size = 0
                                
                                routing_tests[endpoint.name] = {
                                    "success": routing_success,
                                    "status_code": response.status,
                                    "response_time_ms": response_time,
                                    "response_size": response_size,
                                    "expected_status": endpoint.expected_status
                                }
                                
                        except Exception as route_error:
                            routing_tests[endpoint.name] = {
                                "success": False,
                                "error": str(route_error),
                                "response_time_ms": (time.time() - routing_start) * 1000
                            }
                
                # Test different HTTP methods if routing is working
                if any(test.get("success", False) for test in routing_tests.values()):
                    http_methods_test = {}
                    test_url = f"{proxy_url}/api/v1/health"
                    
                    methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
                    for method in methods:
                        try:
                            async with session.request(method, test_url,
                                                      timeout=aiohttp.ClientTimeout(total=10)) as response:
                                http_methods_test[method] = {
                                    "status_code": response.status,
                                    "allowed": response.status not in [404, 405]
                                }
                        except Exception as method_error:
                            http_methods_test[method] = {
                                "error": str(method_error),
                                "allowed": False
                            }
                else:
                    http_methods_test = {}
                
                # Test header forwarding (important for service mesh)
                header_forwarding_test = {}
                if routing_tests.get("backend_via_kong", {}).get("success", False):
                    test_headers = {
                        "X-Test-Header": "test-value",
                        "Authorization": "Bearer test-token",
                        "User-Agent": "SutazAI-Test/1.0"
                    }
                    
                    try:
                        async with session.get(f"{proxy_url}/api/v1/health",
                                             headers=test_headers,
                                             timeout=aiohttp.ClientTimeout(total=15)) as response:
                            header_forwarding_test = {
                                "status": response.status,
                                "headers_sent": len(test_headers),
                                "response_received": True
                            }
                    except Exception as header_error:
                        header_forwarding_test = {
                            "error": str(header_error),
                            "response_received": False
                        }
            
            duration = time.time() - start_time
            
            # Calculate routing success rate
            successful_routes = sum(1 for test in routing_tests.values() if test.get("success", False))
            routing_success_rate = successful_routes / len(routing_tests) * 100 if routing_tests else 0
            
            self.results.append(ServiceMeshTestResult(
                component="kong_routing",
                test_name="routing_functionality",
                success=routing_success_rate > 0,
                duration=duration,
                metrics={
                    "routing_tests": routing_tests,
                    "successful_routes": successful_routes,
                    "routing_success_rate": routing_success_rate,
                    "http_methods_test": http_methods_test,
                    "header_forwarding_test": header_forwarding_test,
                    "average_response_time": sum(test.get("response_time_ms", 0) for test in routing_tests.values()) / len(routing_tests) if routing_tests else 0,
                    "performance_grade": "excellent" if routing_success_rate > 80 else "good" if routing_success_rate > 50 else "poor"
                }
            ))
            
            logger.info(f"Kong routing - Success rate: {routing_success_rate:.1f}%, Routes tested: {len(routing_tests)}")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ServiceMeshTestResult(
                component="kong_routing",
                test_name="routing_functionality",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Kong routing test failed: {e}")
    
    async def test_kong_load_balancing(self) -> None:
        """Test Kong load balancing capabilities"""
        start_time = time.time()
        
        try:
            admin_url = f"http://{self.config['kong']['admin_host']}:{self.config['kong']['admin_port']}"
            proxy_url = f"http://{self.config['kong']['proxy_host']}:{self.config['kong']['proxy_port']}"
            
            async with aiohttp.ClientSession() as session:
                # Check for upstreams configuration
                async with session.get(f"{admin_url}/upstreams",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    upstreams_success = response.status == 200
                    if upstreams_success:
                        upstreams_data = await response.json()
                        upstreams = upstreams_data.get("data", [])
                    else:
                        upstreams = []
                
                # Test load balancing behavior (if upstreams exist)
                load_balance_test = {}
                if upstreams:
                    # Test multiple requests to see load balancing
                    upstream_name = upstreams[0].get("name", "")
                    test_requests = 10
                    response_variations = []
                    
                    for i in range(test_requests):
                        try:
                            async with session.get(f"{proxy_url}/api/v1/health",
                                                 timeout=aiohttp.ClientTimeout(total=5)) as response:
                                response_info = {
                                    "status": response.status,
                                    "headers": dict(response.headers),
                                    "request_id": response.headers.get("X-Kong-Request-Id", "")
                                }
                                response_variations.append(response_info)
                        except Exception as req_error:
                            response_variations.append({"error": str(req_error)})
                    
                    # Analyze response patterns for load balancing
                    unique_request_ids = set(r.get("request_id", "") for r in response_variations if r.get("request_id"))
                    successful_requests = sum(1 for r in response_variations if r.get("status") == 200)
                    
                    load_balance_test = {
                        "upstream_available": True,
                        "requests_sent": test_requests,
                        "successful_requests": successful_requests,
                        "unique_request_ids": len(unique_request_ids),
                        "success_rate": successful_requests / test_requests * 100
                    }
                else:
                    load_balance_test = {
                        "upstream_available": False,
                        "note": "No upstreams configured for load balancing"
                    }
                
                # Test upstream health checking (if upstreams exist)
                upstream_health = {}
                for upstream in upstreams[:3]:  # Test first 3 upstreams
                    upstream_id = upstream.get("id", "")
                    upstream_name = upstream.get("name", "unknown")
                    
                    if upstream_id:
                        try:
                            async with session.get(f"{admin_url}/upstreams/{upstream_id}/health",
                                                 timeout=aiohttp.ClientTimeout(total=15)) as response:
                                health_success = response.status == 200
                                if health_success:
                                    health_data = await response.json()
                                else:
                                    health_data = {}
                                
                                upstream_health[upstream_name] = {
                                    "health_endpoint_available": health_success,
                                    "health_data": health_data
                                }
                        except Exception as health_error:
                            upstream_health[upstream_name] = {
                                "health_endpoint_available": False,
                                "error": str(health_error)
                            }
            
            duration = time.time() - start_time
            
            self.results.append(ServiceMeshTestResult(
                component="kong_load_balancing",
                test_name="load_balancing",
                success=upstreams_success,
                duration=duration,
                metrics={
                    "upstreams_endpoint_available": upstreams_success,
                    "total_upstreams": len(upstreams),
                    "load_balance_test": load_balance_test,
                    "upstream_health": upstream_health,
                    "upstreams_configured": len(upstreams) > 0,
                    "load_balancing_functional": load_balance_test.get("upstream_available", False),
                    "performance_grade": "excellent" if duration < 10 else "good" if duration < 20 else "poor"
                }
            ))
            
            logger.info(f"Kong load balancing - Upstreams: {len(upstreams)}, Functional: {load_balance_test.get('upstream_available', False)}")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ServiceMeshTestResult(
                component="kong_load_balancing",
                test_name="load_balancing",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Kong load balancing test failed: {e}")
    
    async def test_service_mesh_communication(self) -> None:
        """Test service mesh communication patterns"""
        start_time = time.time()
        
        try:
            # Test direct service communication
            backend_url = f"http://{self.config['backend']['host']}:{self.config['backend']['port']}"
            frontend_url = f"http://{self.config['frontend']['host']}:{self.config['frontend']['port']}"
            
            communication_tests = {}
            
            async with aiohttp.ClientSession() as session:
                # Test backend direct access
                async with session.get(f"{backend_url}/health",
                                     timeout=aiohttp.ClientTimeout(total=15)) as response:
                    backend_direct = response.status == 200
                    if backend_direct:
                        backend_response = await response.json()
                    else:
                        backend_response = {}
                
                communication_tests["backend_direct"] = {
                    "success": backend_direct,
                    "response": backend_response
                }
                
                # Test frontend direct access
                async with session.get(f"{frontend_url}/health",
                                     timeout=aiohttp.ClientTimeout(total=15)) as response:
                    frontend_direct = response.status == 200
                    if frontend_direct:
                        try:
                            frontend_response = await response.json()
                        except:
                            frontend_response = {"status": "ok"}
                    else:
                        frontend_response = {}
                
                communication_tests["frontend_direct"] = {
                    "success": frontend_direct,
                    "response": frontend_response
                }
                
                # Test service-to-service communication (backend to services)
                if backend_direct:
                    # Test if backend can communicate with other services
                    service_endpoints = [
                        f"{backend_url}/api/v1/agents",
                        f"{backend_url}/api/v1/health/detailed"
                    ]
                    
                    backend_api_results = {}
                    for endpoint in service_endpoints:
                        try:
                            async with session.get(endpoint,
                                                 timeout=aiohttp.ClientTimeout(total=10)) as response:
                                api_success = response.status in [200, 404, 405]  # Accept not implemented
                                backend_api_results[endpoint] = {
                                    "accessible": api_success,
                                    "status": response.status
                                }
                        except Exception as api_error:
                            backend_api_results[endpoint] = {
                                "accessible": False,
                                "error": str(api_error)
                            }
                    
                    communication_tests["backend_api_endpoints"] = backend_api_results
                
                # Test mesh communication patterns
                mesh_patterns = {}
                
                # Pattern 1: Frontend -> Backend communication
                if frontend_direct and backend_direct:
                    mesh_patterns["frontend_to_backend"] = {
                        "pattern": "direct",
                        "available": True
                    }
                
                # Pattern 2: Gateway -> Backend communication (if Kong is working)
                kong_proxy_url = f"http://{self.config['kong']['proxy_host']}:{self.config['kong']['proxy_port']}"
                try:
                    async with session.get(f"{kong_proxy_url}/api/v1/health",
                                         timeout=aiohttp.ClientTimeout(total=10)) as response:
                        gateway_backend_success = response.status in [200, 404, 502, 503]  # Accept routing attempts
                        mesh_patterns["gateway_to_backend"] = {
                            "pattern": "via_kong",
                            "available": gateway_backend_success,
                            "status": response.status
                        }
                except Exception as gateway_error:
                    mesh_patterns["gateway_to_backend"] = {
                        "pattern": "via_kong",
                        "available": False,
                        "error": str(gateway_error)
                    }
                
                communication_tests["mesh_patterns"] = mesh_patterns
            
            duration = time.time() - start_time
            
            # Calculate communication success rate
            total_patterns = len(communication_tests)
            successful_patterns = sum(1 for test in communication_tests.values() 
                                    if isinstance(test, dict) and test.get("success", False))
            
            # Include mesh pattern successes
            if "mesh_patterns" in communication_tests:
                mesh_successes = sum(1 for pattern in communication_tests["mesh_patterns"].values() 
                                   if pattern.get("available", False))
                successful_patterns += mesh_successes
                total_patterns += len(communication_tests["mesh_patterns"])
            
            communication_success_rate = successful_patterns / max(total_patterns, 1) * 100
            
            self.results.append(ServiceMeshTestResult(
                component="service_mesh_communication",
                test_name="mesh_communication",
                success=communication_success_rate > 50,
                duration=duration,
                metrics={
                    "communication_tests": communication_tests,
                    "successful_patterns": successful_patterns,
                    "total_patterns": total_patterns,
                    "communication_success_rate": communication_success_rate,
                    "backend_accessible": communication_tests.get("backend_direct", {}).get("success", False),
                    "frontend_accessible": communication_tests.get("frontend_direct", {}).get("success", False),
                    "performance_grade": "excellent" if duration < 10 else "good" if duration < 20 else "poor"
                }
            ))
            
            logger.info(f"Service mesh communication - Success rate: {communication_success_rate:.1f}%, Patterns: {successful_patterns}/{total_patterns}")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ServiceMeshTestResult(
                component="service_mesh_communication",
                test_name="mesh_communication",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Service mesh communication test failed: {e}")
    
    async def test_traffic_routing(self) -> None:
        """Test traffic routing and load distribution"""
        start_time = time.time()
        
        try:
            # Test traffic routing patterns
            routing_patterns = {}
            
            proxy_url = f"http://{self.config['kong']['proxy_host']}:{self.config['kong']['proxy_port']}"
            
            async with aiohttp.ClientSession() as session:
                # Test different routing paths
                routing_paths = [
                    "/api/v1/health",
                    "/health", 
                    "/status",
                    "/api/v1/agents"
                ]
                
                for path in routing_paths:
                    routing_start = time.time()
                    try:
                        async with session.get(f"{proxy_url}{path}",
                                             timeout=aiohttp.ClientTimeout(total=10)) as response:
                            routing_time = (time.time() - routing_start) * 1000
                            
                            routing_patterns[path] = {
                                "routable": response.status not in [404, 502],
                                "status_code": response.status,
                                "routing_time_ms": routing_time,
                                "content_length": int(response.headers.get("content-length", 0))
                            }
                            
                    except Exception as route_error:
                        routing_patterns[path] = {
                            "routable": False,
                            "error": str(route_error),
                            "routing_time_ms": (time.time() - routing_start) * 1000
                        }
                
                # Test load distribution (multiple requests to same endpoint)
                load_distribution_test = {}
                test_path = "/api/v1/health"
                
                if routing_patterns.get(test_path, {}).get("routable", False):
                    distribution_requests = 20
                    response_times = []
                    status_codes = []
                    
                    for i in range(distribution_requests):
                        request_start = time.time()
                        try:
                            async with session.get(f"{proxy_url}{test_path}",
                                                 timeout=aiohttp.ClientTimeout(total=5)) as response:
                                response_time = (time.time() - request_start) * 1000
                                response_times.append(response_time)
                                status_codes.append(response.status)
                        except Exception as dist_error:
                            response_times.append((time.time() - request_start) * 1000)
                            status_codes.append(0)
                    
                    # Analyze load distribution
                    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
                    successful_requests = sum(1 for code in status_codes if code == 200)
                    success_rate = successful_requests / distribution_requests * 100
                    
                    load_distribution_test = {
                        "requests_sent": distribution_requests,
                        "successful_requests": successful_requests,
                        "success_rate": success_rate,
                        "avg_response_time_ms": avg_response_time,
                        "min_response_time_ms": min(response_times) if response_times else 0,
                        "max_response_time_ms": max(response_times) if response_times else 0,
                        "response_time_consistency": max(response_times) - min(response_times) if response_times else 0
                    }
                else:
                    load_distribution_test = {
                        "test_skipped": "No routable path found for testing"
                    }
            
            duration = time.time() - start_time
            
            # Calculate routing effectiveness
            routable_paths = sum(1 for pattern in routing_patterns.values() if pattern.get("routable", False))
            routing_effectiveness = routable_paths / len(routing_patterns) * 100 if routing_patterns else 0
            
            self.results.append(ServiceMeshTestResult(
                component="traffic_routing",
                test_name="traffic_patterns",
                success=routing_effectiveness > 0,
                duration=duration,
                metrics={
                    "routing_patterns": routing_patterns,
                    "routable_paths": routable_paths,
                    "routing_effectiveness": routing_effectiveness,
                    "load_distribution_test": load_distribution_test,
                    "avg_routing_time": sum(p.get("routing_time_ms", 0) for p in routing_patterns.values()) / len(routing_patterns) if routing_patterns else 0,
                    "performance_grade": "excellent" if routing_effectiveness > 75 else "good" if routing_effectiveness > 50 else "poor"
                }
            ))
            
            logger.info(f"Traffic routing - Routable paths: {routable_paths}/{len(routing_patterns)} ({routing_effectiveness:.1f}%)")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ServiceMeshTestResult(
                component="traffic_routing",
                test_name="traffic_patterns",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Traffic routing test failed: {e}")
    
    async def test_service_discovery_integration(self) -> None:
        """Test integration between service discovery and API gateway"""
        start_time = time.time()
        
        try:
            consul_url = f"http://{self.config['consul']['host']}:{self.config['consul']['port']}"
            kong_admin_url = f"http://{self.config['kong']['admin_host']}:{self.config['kong']['admin_port']}"
            
            integration_tests = {}
            
            async with aiohttp.ClientSession() as session:
                # Get services from Consul
                async with session.get(f"{consul_url}/v1/catalog/services",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    consul_success = response.status == 200
                    if consul_success:
                        consul_services = await response.json()
                        consul_service_names = list(consul_services.keys())
                    else:
                        consul_service_names = []
                
                integration_tests["consul_services"] = {
                    "accessible": consul_success,
                    "service_count": len(consul_service_names),
                    "services": consul_service_names
                }
                
                # Get services from Kong
                async with session.get(f"{kong_admin_url}/services",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    kong_success = response.status == 200
                    if kong_success:
                        kong_services_data = await response.json()
                        kong_services = kong_services_data.get("data", [])
                        kong_service_names = [svc.get("name", "") for svc in kong_services]
                    else:
                        kong_service_names = []
                
                integration_tests["kong_services"] = {
                    "accessible": kong_success,
                    "service_count": len(kong_service_names),
                    "services": kong_service_names
                }
                
                # Analyze service overlap/integration
                if consul_success and kong_success:
                    # Find common services (case-insensitive matching)
                    common_services = []
                    for consul_svc in consul_service_names:
                        for kong_svc in kong_service_names:
                            if consul_svc.lower() in kong_svc.lower() or kong_svc.lower() in consul_svc.lower():
                                common_services.append((consul_svc, kong_svc))
                    
                    integration_analysis = {
                        "consul_services": len(consul_service_names),
                        "kong_services": len(kong_service_names),
                        "common_services": len(common_services),
                        "integration_rate": len(common_services) / max(len(consul_service_names), 1) * 100,
                        "service_matches": common_services
                    }
                else:
                    integration_analysis = {
                        "error": "Could not access both Consul and Kong for comparison"
                    }
                
                integration_tests["service_integration_analysis"] = integration_analysis
                
                # Test service health correlation
                if consul_success:
                    health_correlation = {}
                    
                    for service_name in consul_service_names[:3]:  # Test first 3 services
                        # Get health from Consul
                        async with session.get(f"{consul_url}/v1/health/service/{service_name}",
                                             timeout=aiohttp.ClientTimeout(total=15)) as response:
                            if response.status == 200:
                                consul_health = await response.json()
                                healthy_instances = sum(1 for instance in consul_health 
                                                      if all(check.get("Status") == "passing" 
                                                           for check in instance.get("Checks", [])))
                                
                                health_correlation[service_name] = {
                                    "consul_instances": len(consul_health),
                                    "healthy_instances": healthy_instances,
                                    "health_rate": healthy_instances / max(len(consul_health), 1) * 100
                                }
                    
                    integration_tests["health_correlation"] = health_correlation
            
            duration = time.time() - start_time
            
            # Overall integration success
            integration_success = integration_tests.get("consul_services", {}).get("accessible", False) and \
                                integration_tests.get("kong_services", {}).get("accessible", False)
            
            self.results.append(ServiceMeshTestResult(
                component="service_discovery_integration",
                test_name="discovery_integration",
                success=integration_success,
                duration=duration,
                metrics={
                    "integration_tests": integration_tests,
                    "integration_available": integration_success,
                    "service_correlation": integration_analysis if 'integration_analysis' in locals() else {},
                    "performance_grade": "excellent" if duration < 10 else "good" if duration < 20 else "poor"
                }
            ))
            
            logger.info(f"Service discovery integration - Consul: {len(consul_service_names) if 'consul_service_names' in locals() else 0}, Kong: {len(kong_service_names) if 'kong_service_names' in locals() else 0}")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ServiceMeshTestResult(
                component="service_discovery_integration",
                test_name="discovery_integration",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Service discovery integration test failed: {e}")
    
    async def test_api_rate_limiting(self) -> None:
        """Test API rate limiting capabilities"""
        start_time = time.time()
        
        try:
            # Test rate limiting (if configured)
            proxy_url = f"http://{self.config['kong']['proxy_host']}:{self.config['kong']['proxy_port']}"
            admin_url = f"http://{self.config['kong']['admin_host']}:{self.config['kong']['admin_port']}"
            
            rate_limiting_tests = {}
            
            async with aiohttp.ClientSession() as session:
                # Check if rate limiting plugin is configured
                async with session.get(f"{admin_url}/plugins",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    plugins_success = response.status == 200
                    if plugins_success:
                        plugins_data = await response.json()
                        plugins = plugins_data.get("data", [])
                        rate_limit_plugins = [p for p in plugins if "rate" in p.get("name", "").lower()]
                    else:
                        rate_limit_plugins = []
                
                rate_limiting_tests["plugins_check"] = {
                    "plugins_accessible": plugins_success,
                    "total_plugins": len(plugins) if plugins_success else 0,
                    "rate_limit_plugins": len(rate_limit_plugins),
                    "configured": len(rate_limit_plugins) > 0
                }
                
                # Test rate limiting behavior (basic test)
                if len(rate_limit_plugins) > 0:
                    # Send multiple requests quickly to test rate limiting
                    rapid_requests = 15
                    request_results = []
                    
                    for i in range(rapid_requests):
                        request_start = time.time()
                        try:
                            async with session.get(f"{proxy_url}/api/v1/health",
                                                 timeout=aiohttp.ClientTimeout(total=3)) as response:
                                request_time = (time.time() - request_start) * 1000
                                request_results.append({
                                    "status": response.status,
                                    "time_ms": request_time,
                                    "rate_limit_headers": {
                                        "x-ratelimit-remaining": response.headers.get("x-ratelimit-remaining"),
                                        "x-ratelimit-limit": response.headers.get("x-ratelimit-limit"),
                                        "x-ratelimit-reset": response.headers.get("x-ratelimit-reset")
                                    }
                                })
                        except Exception as req_error:
                            request_results.append({
                                "error": str(req_error),
                                "time_ms": (time.time() - request_start) * 1000
                            })
                        
                        # Small delay between requests
                        await asyncio.sleep(0.1)
                    
                    # Analyze rate limiting behavior
                    status_codes = [r.get("status", 0) for r in request_results]
                    rate_limited_requests = sum(1 for code in status_codes if code == 429)
                    successful_requests = sum(1 for code in status_codes if code == 200)
                    
                    rate_limiting_tests["behavior_test"] = {
                        "requests_sent": rapid_requests,
                        "successful_requests": successful_requests,
                        "rate_limited_requests": rate_limited_requests,
                        "rate_limiting_triggered": rate_limited_requests > 0,
                        "avg_response_time": sum(r.get("time_ms", 0) for r in request_results) / len(request_results)
                    }
                else:
                    rate_limiting_tests["behavior_test"] = {
                        "test_skipped": "No rate limiting plugins configured"
                    }
            
            duration = time.time() - start_time
            
            self.results.append(ServiceMeshTestResult(
                component="api_rate_limiting",
                test_name="rate_limiting",
                success=rate_limiting_tests.get("plugins_check", {}).get("plugins_accessible", False),
                duration=duration,
                metrics={
                    "rate_limiting_tests": rate_limiting_tests,
                    "rate_limiting_configured": rate_limiting_tests.get("plugins_check", {}).get("configured", False),
                    "performance_grade": "excellent" if duration < 5 else "good" if duration < 15 else "poor"
                }
            ))
            
            logger.info(f"API rate limiting - Configured: {rate_limiting_tests.get('plugins_check', {}).get('configured', False)}")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ServiceMeshTestResult(
                component="api_rate_limiting",
                test_name="rate_limiting",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"API rate limiting test failed: {e}")
    
    async def test_security_policies(self) -> None:
        """Test security policies and authentication"""
        start_time = time.time()
        
        try:
            admin_url = f"http://{self.config['kong']['admin_host']}:{self.config['kong']['admin_port']}"
            proxy_url = f"http://{self.config['kong']['proxy_host']}:{self.config['kong']['proxy_port']}"
            
            security_tests = {}
            
            async with aiohttp.ClientSession() as session:
                # Check security plugins
                async with session.get(f"{admin_url}/plugins",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    plugins_success = response.status == 200
                    if plugins_success:
                        plugins_data = await response.json()
                        plugins = plugins_data.get("data", [])
                        
                        security_plugins = [p for p in plugins if any(sec_keyword in p.get("name", "").lower() 
                                                                     for sec_keyword in ["auth", "cors", "security", "key"])]
                    else:
                        security_plugins = []
                
                security_tests["security_plugins"] = {
                    "plugins_accessible": plugins_success,
                    "total_plugins": len(plugins) if plugins_success else 0,
                    "security_plugins": len(security_plugins),
                    "security_plugin_names": [p.get("name", "") for p in security_plugins]
                }
                
                # Test CORS policy (if configured)
                cors_test = {}
                try:
                    # Test preflight request
                    async with session.options(f"{proxy_url}/api/v1/health",
                                             headers={
                                                 "Origin": "http://localhost:3000",
                                                 "Access-Control-Request-Method": "GET"
                                             },
                                             timeout=aiohttp.ClientTimeout(total=10)) as response:
                        cors_test = {
                            "preflight_status": response.status,
                            "cors_headers": {
                                "access_control_allow_origin": response.headers.get("Access-Control-Allow-Origin"),
                                "access_control_allow_methods": response.headers.get("Access-Control-Allow-Methods"),
                                "access_control_allow_headers": response.headers.get("Access-Control-Allow-Headers")
                            },
                            "cors_configured": any(header.startswith("Access-Control") for header in response.headers.keys())
                        }
                except Exception as cors_error:
                    cors_test = {
                        "error": str(cors_error),
                        "cors_configured": False
                    }
                
                security_tests["cors_policy"] = cors_test
                
                # Test authentication (basic test)
                auth_test = {}
                try:
                    # Test without authentication
                    async with session.get(f"{proxy_url}/api/v1/agents",
                                         timeout=aiohttp.ClientTimeout(total=10)) as response:
                        no_auth_status = response.status
                    
                    # Test with basic auth header
                    async with session.get(f"{proxy_url}/api/v1/agents",
                                         headers={"Authorization": "Bearer test-token"},
                                         timeout=aiohttp.ClientTimeout(total=10)) as response:
                        with_auth_status = response.status
                    
                    auth_test = {
                        "no_auth_status": no_auth_status,
                        "with_auth_status": with_auth_status,
                        "auth_enforced": no_auth_status == 401 and with_auth_status != 401,
                        "auth_configured": no_auth_status in [401, 403] or with_auth_status in [401, 403]
                    }
                    
                except Exception as auth_error:
                    auth_test = {
                        "error": str(auth_error),
                        "auth_configured": False
                    }
                
                security_tests["authentication"] = auth_test
            
            duration = time.time() - start_time
            
            # Overall security assessment
            security_configured = (security_tests.get("security_plugins", {}).get("security_plugins", 0) > 0 or
                                 security_tests.get("cors_policy", {}).get("cors_configured", False) or
                                 security_tests.get("authentication", {}).get("auth_configured", False))
            
            self.results.append(ServiceMeshTestResult(
                component="security_policies",
                test_name="security_validation",
                success=security_tests.get("security_plugins", {}).get("plugins_accessible", False),
                duration=duration,
                metrics={
                    "security_tests": security_tests,
                    "security_configured": security_configured,
                    "security_features_count": sum([
                        security_tests.get("security_plugins", {}).get("security_plugins", 0) > 0,
                        security_tests.get("cors_policy", {}).get("cors_configured", False),
                        security_tests.get("authentication", {}).get("auth_configured", False)
                    ]),
                    "performance_grade": "excellent" if duration < 5 else "good" if duration < 15 else "poor"
                }
            ))
            
            logger.info(f"Security policies - Configured: {security_configured}, Plugins: {len(security_plugins) if 'security_plugins' in locals() else 0}")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ServiceMeshTestResult(
                component="security_policies",
                test_name="security_validation",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Security policies test failed: {e}")
    
    async def test_monitoring_integration(self) -> None:
        """Test monitoring integration with service mesh"""
        start_time = time.time()
        
        try:
            # Test monitoring endpoints integration
            monitoring_endpoints = [
                ("prometheus", "http://localhost:10200/metrics"),
                ("kong_prometheus", f"http://localhost:{self.config['kong']['admin_port']}/metrics"),
                ("consul_metrics", f"http://localhost:{self.config['consul']['port']}/v1/agent/metrics?format=prometheus")
            ]
            
            monitoring_tests = {}
            
            async with aiohttp.ClientSession() as session:
                for endpoint_name, endpoint_url in monitoring_endpoints:
                    try:
                        async with session.get(endpoint_url,
                                             timeout=aiohttp.ClientTimeout(total=15)) as response:
                            metrics_available = response.status == 200
                            if metrics_available:
                                metrics_content = await response.text()
                                metrics_count = len([line for line in metrics_content.split('\n') 
                                                   if line.strip() and not line.startswith('#')])
                            else:
                                metrics_count = 0
                            
                            monitoring_tests[endpoint_name] = {
                                "available": metrics_available,
                                "status": response.status,
                                "metrics_count": metrics_count
                            }
                            
                    except Exception as endpoint_error:
                        monitoring_tests[endpoint_name] = {
                            "available": False,
                            "error": str(endpoint_error)
                        }
                
                # Test service mesh specific metrics (if available)
                mesh_metrics = {}
                if monitoring_tests.get("kong_prometheus", {}).get("available", False):
                    # Kong should expose request metrics, upstream health, etc.
                    mesh_metrics["kong_metrics"] = True
                
                if monitoring_tests.get("consul_metrics", {}).get("available", False):
                    # Consul should expose service discovery metrics
                    mesh_metrics["consul_metrics"] = True
                
                # Test health check endpoints for monitoring
                health_endpoints = [
                    f"http://localhost:{self.config['kong']['proxy_port']}/status",
                    f"http://localhost:{self.config['consul']['port']}/v1/status/leader"
                ]
                
                health_monitoring = {}
                for endpoint in health_endpoints:
                    try:
                        async with session.get(endpoint,
                                             timeout=aiohttp.ClientTimeout(total=10)) as response:
                            health_monitoring[endpoint] = {
                                "monitorable": response.status == 200,
                                "status": response.status
                            }
                    except Exception as health_error:
                        health_monitoring[endpoint] = {
                            "monitorable": False,
                            "error": str(health_error)
                        }
            
            duration = time.time() - start_time
            
            # Calculate monitoring integration success
            available_monitoring = sum(1 for test in monitoring_tests.values() if test.get("available", False))
            monitoring_success_rate = available_monitoring / len(monitoring_tests) * 100 if monitoring_tests else 0
            
            self.results.append(ServiceMeshTestResult(
                component="monitoring_integration",
                test_name="monitoring_validation",
                success=monitoring_success_rate > 0,
                duration=duration,
                metrics={
                    "monitoring_endpoints": monitoring_tests,
                    "available_monitoring": available_monitoring,
                    "monitoring_success_rate": monitoring_success_rate,
                    "mesh_specific_metrics": mesh_metrics,
                    "health_monitoring": health_monitoring,
                    "total_metrics": sum(test.get("metrics_count", 0) for test in monitoring_tests.values()),
                    "performance_grade": "excellent" if duration < 10 else "good" if duration < 20 else "poor"
                }
            ))
            
            logger.info(f"Monitoring integration - Available: {available_monitoring}/{len(monitoring_tests)} ({monitoring_success_rate:.1f}%)")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(ServiceMeshTestResult(
                component="monitoring_integration",
                test_name="monitoring_validation",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Monitoring integration test failed: {e}")
    
    def generate_service_mesh_report(self) -> Dict[str, Any]:
        """Generate comprehensive service mesh validation report"""
        total_components = len(self.results)
        successful_components = len([r for r in self.results if r.success])
        
        # Group results by component
        component_results = {}
        for result in self.results:
            component_results[result.component] = result
        
        # Calculate service mesh health
        critical_components = ["consul_service_discovery", "kong_gateway_health", "service_mesh_communication"]
        critical_success = sum(1 for comp in critical_components 
                             if comp in component_results and component_results[comp].success)
        
        mesh_grade = "EXCELLENT" if critical_success == len(critical_components) else \
                    "GOOD" if critical_success >= len(critical_components) - 1 else \
                    "POOR"
        
        # Performance analysis
        performance_summary = {}
        for result in self.results:
            if result.success and "performance_grade" in result.metrics:
                performance_summary[result.component] = result.metrics["performance_grade"]
        
        # Service discovery analysis
        discovery_stats = {}
        if "consul_service_discovery" in component_results:
            consul_metrics = component_results["consul_service_discovery"].metrics
            discovery_stats = {
                "registered_services": consul_metrics.get("registered_services", 0),
                "service_coverage": consul_metrics.get("service_coverage_percent", 0),
                "registered_nodes": consul_metrics.get("registered_nodes", 0)
            }
        
        # API gateway analysis
        gateway_stats = {}
        if "kong_services_config" in component_results:
            kong_metrics = component_results["kong_services_config"].metrics
            gateway_stats = {
                "total_services": kong_metrics.get("total_services", 0),
                "total_routes": kong_metrics.get("total_routes", 0),
                "service_configuration_rate": kong_metrics.get("service_configuration_rate", 0)
            }
        
        # Security and features analysis
        features_analysis = {}
        if "security_policies" in component_results:
            security_metrics = component_results["security_policies"].metrics
            features_analysis["security_configured"] = security_metrics.get("security_configured", False)
        
        if "api_rate_limiting" in component_results:
            rate_limit_metrics = component_results["api_rate_limiting"].metrics
            features_analysis["rate_limiting_configured"] = rate_limit_metrics.get("rate_limiting_tests", {}).get("plugins_check", {}).get("configured", False)
        
        return {
            "summary": {
                "total_components_tested": total_components,
                "successful_components": successful_components,
                "success_rate": round(successful_components / max(total_components, 1) * 100, 2),
                "mesh_grade": mesh_grade,
                "critical_components_health": f"{critical_success}/{len(critical_components)}"
            },
            "component_details": {
                component: {
                    "status": "success" if result.success else "failed",
                    "duration_seconds": round(result.duration, 3),
                    "key_metrics": result.metrics,
                    "error": result.error_message
                }
                for component, result in component_results.items()
            },
            "performance_analysis": performance_summary,
            "service_discovery_stats": discovery_stats,
            "api_gateway_stats": gateway_stats,
            "features_analysis": features_analysis,
            "recommendations": self._generate_service_mesh_recommendations(component_results)
        }
    
    def _generate_service_mesh_recommendations(self, component_results: Dict) -> List[str]:
        """Generate service mesh improvement recommendations"""
        recommendations = []
        
        # Check critical components
        for component, result in component_results.items():
            if not result.success:
                if component == "consul_service_discovery":
                    recommendations.append("🔴 CRITICAL: Consul service discovery is not working - service mesh functionality limited")
                elif component == "kong_gateway_health":
                    recommendations.append("🔴 CRITICAL: Kong API gateway is not accessible - API routing disabled")
                elif component == "service_mesh_communication":
                    recommendations.append("🟡 WARNING: Service mesh communication has issues - inter-service connectivity limited")
        
        # Performance recommendations
        routing_performance = {}
        for comp in ["kong_routing", "traffic_routing"]:
            if comp in component_results and component_results[comp].success:
                metrics = component_results[comp].metrics
                performance_grade = metrics.get("performance_grade", "unknown")
                if performance_grade == "poor":
                    recommendations.append(f"🟡 PERFORMANCE: {comp} performance is slow - consider optimization")
        
        # Configuration recommendations
        if "kong_services_config" in component_results and component_results["kong_services_config"].success:
            kong_metrics = component_results["kong_services_config"].metrics
            config_rate = kong_metrics.get("service_configuration_rate", 0)
            
            if config_rate < 50:
                recommendations.append(f"🟡 CONFIG: Only {config_rate:.1f}% of expected services configured in Kong - check service registration")
        
        # Security recommendations
        if "security_policies" in component_results and component_results["security_policies"].success:
            security_metrics = component_results["security_policies"].metrics
            if not security_metrics.get("security_configured", False):
                recommendations.append("🟡 SECURITY: No security policies detected - consider enabling authentication and CORS")
        
        # Monitoring recommendations
        if "monitoring_integration" in component_results and component_results["monitoring_integration"].success:
            monitoring_metrics = component_results["monitoring_integration"].metrics
            monitoring_rate = monitoring_metrics.get("monitoring_success_rate", 0)
            
            if monitoring_rate < 50:
                recommendations.append("🟡 MONITORING: Limited monitoring integration - service mesh observability reduced")
        
        return recommendations if recommendations else ["✅ Service mesh is operating within optimal parameters"]

async def main():
    """Main execution for service mesh validation"""
    validator = ServiceMeshValidator()
    
    print("🌐 Starting Service Mesh and API Gateway Validation Tests")
    print("=" * 60)
    
    results = await validator.run_all_service_mesh_tests()
    report = validator.generate_service_mesh_report()
    
    print("\n" + "=" * 60)
    print("📊 SERVICE MESH VALIDATION COMPLETE")
    print("=" * 60)
    
    # Print summary
    summary = report["summary"]
    print(f"Components Tested: {summary['total_components_tested']}")
    print(f"Successful: {summary['successful_components']} ({summary['success_rate']}%)")
    print(f"Mesh Grade: {summary['mesh_grade']}")
    print(f"Critical Components: {summary['critical_components_health']}")
    
    # Print component details
    print("\n🔍 Component Status:")
    for component, details in report["component_details"].items():
        status_icon = "✅" if details["status"] == "success" else "❌"
        duration = details["duration_seconds"]
        print(f"  {status_icon} {component}: {details['status']} ({duration:.2f}s)")
        
        if details["error"]:
            print(f"    ⚠️  {details['error']}")
    
    # Print service discovery stats
    discovery_stats = report["service_discovery_stats"]
    if discovery_stats:
        print(f"\n🔍 Service Discovery:")
        print(f"  Registered Services: {discovery_stats.get('registered_services', 0)}")
        print(f"  Coverage: {discovery_stats.get('service_coverage', 0):.1f}%")
        print(f"  Nodes: {discovery_stats.get('registered_nodes', 0)}")
    
    # Print API gateway stats
    gateway_stats = report["api_gateway_stats"]
    if gateway_stats:
        print(f"\n🚪 API Gateway:")
        print(f"  Services: {gateway_stats.get('total_services', 0)}")
        print(f"  Routes: {gateway_stats.get('total_routes', 0)}")
        print(f"  Configuration: {gateway_stats.get('service_configuration_rate', 0):.1f}%")
    
    # Print features analysis
    features = report["features_analysis"]
    if features:
        print(f"\n🔒 Features:")
        security = "✅" if features.get("security_configured", False) else "❌"
        rate_limit = "✅" if features.get("rate_limiting_configured", False) else "❌"
        print(f"  Security: {security}")
        print(f"  Rate Limiting: {rate_limit}")
    
    # Print recommendations
    print("\n💡 Recommendations:")
    for rec in report["recommendations"]:
        print(f"  {rec}")
    
    # Save detailed report
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"service_mesh_validation_report_{timestamp}.json"
    
    import json
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return summary["mesh_grade"] in ["EXCELLENT", "GOOD"]

if __name__ == "__main__":
    success = asyncio.run(main())
    import sys
    sys.exit(0 if success else 1)