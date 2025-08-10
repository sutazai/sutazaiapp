#!/usr/bin/env python3
"""
Service Mesh Validation Script
Validates the complete service mesh integration
"""

import asyncio
import aiohttp
import json
import logging
import os
import sys
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Validation result for a test"""
    component: str
    test_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = None

class ServiceMeshValidator:
    """Validates service mesh components and integration"""
    
    def __init__(self):
        self.consul_url = os.getenv('CONSUL_URL', 'http://consul:8500')
        self.kong_admin_url = os.getenv('KONG_ADMIN_URL', 'http://kong:8001')
        self.kong_proxy_url = os.getenv('KONG_PROXY_URL', 'http://kong:8000')
        self.rabbitmq_url = os.getenv('RABBITMQ_URL', 'http://rabbitmq:15672')
        self.rabbitmq_user = os.getenv('RABBITMQ_USER', 'admin')
        self.rabbitmq_pass = os.getenv('RABBITMQ_PASS', 'adminpass')
        self.health_check_url = os.getenv('HEALTH_CHECK_URL', 'http://health-check-server:8080')
        
        self.results = []
        
    async def validate_consul(self) -> List[ValidationResult]:
        """Validate Consul service discovery"""
        logger.info("Validating Consul service discovery...")
        results = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test 1: Consul leader election
                try:
                    async with session.get(f"{self.consul_url}/v1/status/leader", timeout=10) as response:
                        if response.status == 200:
                            leader = await response.text()
                            results.append(ValidationResult(
                                "Consul", "Leader Election", True,
                                f"Leader elected: {leader.strip('\"')}"
                            ))
                        else:
                            results.append(ValidationResult(
                                "Consul", "Leader Election", False,
                                f"Failed to get leader: HTTP {response.status}"
                            ))
                except Exception as e:
                    results.append(ValidationResult(
                        "Consul", "Leader Election", False,
                        f"Connection failed: {str(e)}"
                    ))
                
                # Test 2: Service catalog
                try:
                    async with session.get(f"{self.consul_url}/v1/catalog/services", timeout=10) as response:
                        if response.status == 200:
                            services = await response.json()
                            results.append(ValidationResult(
                                "Consul", "Service Catalog", True,
                                f"Found {len(services)} services",
                                {"services": list(services.keys())}
                            ))
                        else:
                            results.append(ValidationResult(
                                "Consul", "Service Catalog", False,
                                f"Failed to get services: HTTP {response.status}"
                            ))
                except Exception as e:
                    results.append(ValidationResult(
                        "Consul", "Service Catalog", False,
                        f"Service catalog error: {str(e)}"
                    ))
                
                # Test 3: Health checks
                try:
                    async with session.get(f"{self.consul_url}/v1/health/state/any", timeout=10) as response:
                        if response.status == 200:
                            health_checks = await response.json()
                            passing_checks = [hc for hc in health_checks if hc.get('Status') == 'passing']
                            results.append(ValidationResult(
                                "Consul", "Health Checks", True,
                                f"Health checks: {len(passing_checks)}/{len(health_checks)} passing",
                                {"total_checks": len(health_checks), "passing_checks": len(passing_checks)}
                            ))
                        else:
                            results.append(ValidationResult(
                                "Consul", "Health Checks", False,
                                f"Failed to get health checks: HTTP {response.status}"
                            ))
                except Exception as e:
                    results.append(ValidationResult(
                        "Consul", "Health Checks", False,
                        f"Health check error: {str(e)}"
                    ))
        
        except Exception as e:
            results.append(ValidationResult(
                "Consul", "General", False,
                f"General error: {str(e)}"
            ))
        
        return results
    
    async def validate_kong(self) -> List[ValidationResult]:
        """Validate Kong API Gateway"""
        logger.info("Validating Kong API Gateway...")
        results = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test 1: Kong admin API
                try:
                    async with session.get(f"{self.kong_admin_url}/status", timeout=10) as response:
                        if response.status == 200:
                            status = await response.json()
                            results.append(ValidationResult(
                                "Kong", "Admin API", True,
                                f"Kong is running (DB: {status.get('database', {}).get('reachable', 'unknown')})",
                                status
                            ))
                        else:
                            results.append(ValidationResult(
                                "Kong", "Admin API", False,
                                f"Admin API failed: HTTP {response.status}"
                            ))
                except Exception as e:
                    results.append(ValidationResult(
                        "Kong", "Admin API", False,
                        f"Admin API error: {str(e)}"
                    ))
                
                # Test 2: Services configuration
                try:
                    async with session.get(f"{self.kong_admin_url}/services", timeout=10) as response:
                        if response.status == 200:
                            services_data = await response.json()
                            services = services_data.get('data', [])
                            results.append(ValidationResult(
                                "Kong", "Services", True,
                                f"Configured {len(services)} services",
                                {"service_count": len(services), "services": [s.get('name') for s in services]}
                            ))
                        else:
                            results.append(ValidationResult(
                                "Kong", "Services", False,
                                f"Failed to get services: HTTP {response.status}"
                            ))
                except Exception as e:
                    results.append(ValidationResult(
                        "Kong", "Services", False,
                        f"Services error: {str(e)}"
                    ))
                
                # Test 3: Routes configuration
                try:
                    async with session.get(f"{self.kong_admin_url}/routes", timeout=10) as response:
                        if response.status == 200:
                            routes_data = await response.json()
                            routes = routes_data.get('data', [])
                            results.append(ValidationResult(
                                "Kong", "Routes", True,
                                f"Configured {len(routes)} routes",
                                {"route_count": len(routes), "routes": [r.get('name') for r in routes]}
                            ))
                        else:
                            results.append(ValidationResult(
                                "Kong", "Routes", False,
                                f"Failed to get routes: HTTP {response.status}"
                            ))
                except Exception as e:
                    results.append(ValidationResult(
                        "Kong", "Routes", False,
                        f"Routes error: {str(e)}"
                    ))
                
                # Test 4: Proxy functionality
                try:
                    async with session.get(f"{self.kong_proxy_url}/", timeout=10) as response:
                        # Kong proxy should respond even if no route matches
                        if response.status in [200, 404, 502, 503]:
                            results.append(ValidationResult(
                                "Kong", "Proxy", True,
                                f"Proxy is responding (HTTP {response.status})"
                            ))
                        else:
                            results.append(ValidationResult(
                                "Kong", "Proxy", False,
                                f"Proxy unexpected response: HTTP {response.status}"
                            ))
                except Exception as e:
                    results.append(ValidationResult(
                        "Kong", "Proxy", False,
                        f"Proxy error: {str(e)}"
                    ))
        
        except Exception as e:
            results.append(ValidationResult(
                "Kong", "General", False,
                f"General error: {str(e)}"
            ))
        
        return results
    
    async def validate_rabbitmq(self) -> List[ValidationResult]:
        """Validate RabbitMQ messaging"""
        logger.info("Validating RabbitMQ messaging...")
        results = []
        
        try:
            auth = aiohttp.BasicAuth(self.rabbitmq_user, self.rabbitmq_pass)
            
            async with aiohttp.ClientSession() as session:
                # Test 1: RabbitMQ management API
                try:
                    async with session.get(f"{self.rabbitmq_url}/api/overview", auth=auth, timeout=10) as response:
                        if response.status == 200:
                            overview = await response.json()
                            results.append(ValidationResult(
                                "RabbitMQ", "Management API", True,
                                f"RabbitMQ node: {overview.get('node', 'unknown')}",
                                {"node": overview.get('node'), "version": overview.get('rabbitmq_version')}
                            ))
                        else:
                            results.append(ValidationResult(
                                "RabbitMQ", "Management API", False,
                                f"Management API failed: HTTP {response.status}"
                            ))
                except Exception as e:
                    results.append(ValidationResult(
                        "RabbitMQ", "Management API", False,
                        f"Management API error: {str(e)}"
                    ))
                
                # Test 2: Virtual hosts
                try:
                    async with session.get(f"{self.rabbitmq_url}/api/vhosts", auth=auth, timeout=10) as response:
                        if response.status == 200:
                            vhosts = await response.json()
                            vhost_names = [vh.get('name') for vh in vhosts]
                            results.append(ValidationResult(
                                "RabbitMQ", "Virtual Hosts", True,
                                f"Found {len(vhosts)} virtual hosts",
                                {"vhosts": vhost_names}
                            ))
                        else:
                            results.append(ValidationResult(
                                "RabbitMQ", "Virtual Hosts", False,
                                f"Failed to get vhosts: HTTP {response.status}"
                            ))
                except Exception as e:
                    results.append(ValidationResult(
                        "RabbitMQ", "Virtual Hosts", False,
                        f"Virtual hosts error: {str(e)}"
                    ))
                
                # Test 3: Queues
                try:
                    async with session.get(f"{self.rabbitmq_url}/api/queues", auth=auth, timeout=10) as response:
                        if response.status == 200:
                            queues = await response.json()
                            queue_names = [q.get('name') for q in queues]
                            results.append(ValidationResult(
                                "RabbitMQ", "Queues", True,
                                f"Found {len(queues)} queues",
                                {"queue_count": len(queues), "queues": queue_names[:10]}  # Show first 10
                            ))
                        else:
                            results.append(ValidationResult(
                                "RabbitMQ", "Queues", False,
                                f"Failed to get queues: HTTP {response.status}"
                            ))
                except Exception as e:
                    results.append(ValidationResult(
                        "RabbitMQ", "Queues", False,
                        f"Queues error: {str(e)}"
                    ))
                
                # Test 4: Exchanges
                try:
                    async with session.get(f"{self.rabbitmq_url}/api/exchanges", auth=auth, timeout=10) as response:
                        if response.status == 200:
                            exchanges = await response.json()
                            exchange_names = [e.get('name') for e in exchanges if e.get('name')]
                            results.append(ValidationResult(
                                "RabbitMQ", "Exchanges", True,
                                f"Found {len(exchange_names)} exchanges",
                                {"exchange_count": len(exchange_names), "exchanges": exchange_names[:10]}
                            ))
                        else:
                            results.append(ValidationResult(
                                "RabbitMQ", "Exchanges", False,
                                f"Failed to get exchanges: HTTP {response.status}"
                            ))
                except Exception as e:
                    results.append(ValidationResult(
                        "RabbitMQ", "Exchanges", False,
                        f"Exchanges error: {str(e)}"
                    ))
        
        except Exception as e:
            results.append(ValidationResult(
                "RabbitMQ", "General", False,
                f"General error: {str(e)}"
            ))
        
        return results
    
    async def validate_health_monitoring(self) -> List[ValidationResult]:
        """Validate health monitoring system"""
        logger.info("Validating health monitoring...")
        results = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test 1: Health check server
                try:
                    async with session.get(f"{self.health_check_url}/health", timeout=15) as response:
                        if response.status in [200, 503]:  # 503 is acceptable if some services are down
                            health_data = await response.json()
                            total_services = health_data.get('summary', {}).get('total_services', 0)
                            healthy_services = health_data.get('summary', {}).get('healthy_services', 0)
                            
                            results.append(ValidationResult(
                                "HealthMonitor", "Health Check API", True,
                                f"Health monitoring active: {healthy_services}/{total_services} services healthy",
                                health_data.get('summary', {})
                            ))
                        else:
                            results.append(ValidationResult(
                                "HealthMonitor", "Health Check API", False,
                                f"Health check API failed: HTTP {response.status}"
                            ))
                except Exception as e:
                    results.append(ValidationResult(
                        "HealthMonitor", "Health Check API", False,
                        f"Health check API error: {str(e)}"
                    ))
                
                # Test 2: Service list
                try:
                    async with session.get(f"{self.health_check_url}/services", timeout=10) as response:
                        if response.status == 200:
                            services_data = await response.json()
                            services = services_data.get('services', [])
                            results.append(ValidationResult(
                                "HealthMonitor", "Service Discovery", True,
                                f"Monitoring {len(services)} services",
                                {"services": services}
                            ))
                        else:
                            results.append(ValidationResult(
                                "HealthMonitor", "Service Discovery", False,
                                f"Failed to get service list: HTTP {response.status}"
                            ))
                except Exception as e:
                    results.append(ValidationResult(
                        "HealthMonitor", "Service Discovery", False,
                        f"Service discovery error: {str(e)}"
                    ))
                
                # Test 3: Metrics endpoint
                try:
                    async with session.get(f"{self.health_check_url}/metrics", timeout=10) as response:
                        if response.status == 200:
                            metrics_text = await response.text()
                            metric_lines = [line for line in metrics_text.split('\n') if line and not line.startswith('#')]
                            results.append(ValidationResult(
                                "HealthMonitor", "Metrics", True,
                                f"Generated {len(metric_lines)} metrics",
                                {"metric_count": len(metric_lines)}
                            ))
                        else:
                            results.append(ValidationResult(
                                "HealthMonitor", "Metrics", False,
                                f"Failed to get metrics: HTTP {response.status}"
                            ))
                except Exception as e:
                    results.append(ValidationResult(
                        "HealthMonitor", "Metrics", False,
                        f"Metrics error: {str(e)}"
                    ))
        
        except Exception as e:
            results.append(ValidationResult(
                "HealthMonitor", "General", False,
                f"General error: {str(e)}"
            ))
        
        return results
    
    async def validate_integration(self) -> List[ValidationResult]:
        """Validate end-to-end service mesh integration"""
        logger.info("Validating service mesh integration...")
        results = []
        
        try:
            # Test 1: Service discovery to Kong integration
            async with aiohttp.ClientSession() as session:
                # Get services from Consul
                consul_services = set()
                try:
                    async with session.get(f"{self.consul_url}/v1/catalog/services", timeout=10) as response:
                        if response.status == 200:
                            services = await response.json()
                            consul_services = set(services.keys())
                except Exception as e:
                    pass
                
                # Get services from Kong
                kong_services = set()
                try:
                    async with session.get(f"{self.kong_admin_url}/services", timeout=10) as response:
                        if response.status == 200:
                            services_data = await response.json()
                            services = services_data.get('data', [])
                            kong_services = set(s.get('name', '').replace('-service', '') for s in services)
                except Exception as e:
                    pass
                
                # Check overlap
                common_services = consul_services.intersection(kong_services)
                if common_services:
                    results.append(ValidationResult(
                        "Integration", "Consul-Kong", True,
                        f"Found {len(common_services)} services in both Consul and Kong",
                        {"common_services": list(common_services)}
                    ))
                else:
                    results.append(ValidationResult(
                        "Integration", "Consul-Kong", False,
                        "No common services found between Consul and Kong",
                        {"consul_services": list(consul_services), "kong_services": list(kong_services)}
                    ))
                
                # Test 2: Health monitoring integration
                try:
                    async with session.get(f"{self.health_check_url}/health", timeout=15) as response:
                        if response.status in [200, 503]:
                            health_data = await response.json()
                            monitored_services = set(health_data.get('services', {}).keys())
                            
                            # Check if health monitor knows about Consul services
                            common_monitored = consul_services.intersection(monitored_services)
                            if common_monitored:
                                results.append(ValidationResult(
                                    "Integration", "Health-Consul", True,
                                    f"Health monitor tracking {len(common_monitored)} Consul services",
                                    {"monitored_services": list(common_monitored)}
                                ))
                            else:
                                results.append(ValidationResult(
                                    "Integration", "Health-Consul", False,
                                    "Health monitor not tracking Consul services"
                                ))
                except Exception as e:
                    results.append(ValidationResult(
                        "Integration", "Health-Consul", False,
                        f"Health monitoring integration error: {str(e)}"
                    ))
        
        except Exception as e:
            results.append(ValidationResult(
                "Integration", "General", False,
                f"Integration validation error: {str(e)}"
            ))
        
        return results
    
    async def run_all_validations(self) -> Tuple[List[ValidationResult], bool]:
        """Run all validation tests"""
        logger.info("Starting complete service mesh validation...")
        
        all_results = []
        
        # Run all validation tests
        consul_results = await self.validate_consul()
        all_results.extend(consul_results)
        
        kong_results = await self.validate_kong()
        all_results.extend(kong_results)
        
        rabbitmq_results = await self.validate_rabbitmq()
        all_results.extend(rabbitmq_results)
        
        health_results = await self.validate_health_monitoring()
        all_results.extend(health_results)
        
        integration_results = await self.validate_integration()
        all_results.extend(integration_results)
        
        # Calculate overall success
        total_tests = len(all_results)
        passed_tests = sum(1 for result in all_results if result.passed)
        overall_success = passed_tests == total_tests
        
        logger.info(f"Validation completed: {passed_tests}/{total_tests} tests passed")
        
        return all_results, overall_success
    
    def print_results(self, results: List[ValidationResult], detailed: bool = False):
        """Print validation results"""
        print("\n" + "="*80)
        print("SERVICE MESH VALIDATION RESULTS")
        print("="*80)
        
        # Group results by component
        components = {}
        for result in results:
            if result.component not in components:
                components[result.component] = []
            components[result.component].append(result)
        
        for component, component_results in components.items():
            passed = sum(1 for r in component_results if r.passed)
            total = len(component_results)
            status = "PASS" if passed == total else "FAIL"
            
            print(f"\n{component}: {status} ({passed}/{total})")
            print("-" * 40)
            
            for result in component_results:
                status_icon = "✓" if result.passed else "✗"
                print(f"  {status_icon} {result.test_name}: {result.message}")
                
                if detailed and result.details:
                    print(f"    Details: {json.dumps(result.details, indent=2)}")
        
        # Overall summary
        total_tests = len(results)
        passed_tests = sum(1 for result in results if result.passed)
        overall_status = "PASS" if passed_tests == total_tests else "FAIL"
        
        print("\n" + "="*80)
        print(f"OVERALL STATUS: {overall_status} ({passed_tests}/{total_tests})")
        print("="*80)

async def main():
    """Main validation function"""
    parser = argparse.ArgumentParser(description='Validate SutazAI Service Mesh')
    parser.add_argument('--detailed', action='store_true', help='Show detailed results')
    parser.add_argument('--component', choices=['consul', 'kong', 'rabbitmq', 'health', 'integration'], 
                       help='Validate specific component only')
    args = parser.parse_args()
    
    validator = ServiceMeshValidator()
    
    try:
        if args.component:
            # Validate specific component
            if args.component == 'consul':
                results = await validator.validate_consul()
            elif args.component == 'kong':
                results = await validator.validate_kong()
            elif args.component == 'rabbitmq':
                results = await validator.validate_rabbitmq()
            elif args.component == 'health':
                results = await validator.validate_health_monitoring()
            elif args.component == 'integration':
                results = await validator.validate_integration()
            
            passed_tests = sum(1 for result in results if result.passed)
            overall_success = passed_tests == len(results)
        else:
            # Run all validations
            results, overall_success = await validator.run_all_validations()
        
        # Print results
        validator.print_results(results, detailed=args.detailed)
        
        # Exit with appropriate code
        sys.exit(0 if overall_success else 1)
        
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())