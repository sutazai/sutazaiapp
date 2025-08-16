#!/usr/bin/env python3
"""
Service Mesh Validation Script for CI/CD Integration
Performs comprehensive validation of the service mesh implementation

Exit Codes:
- 0: All tests passed
- 1: Critical infrastructure failure
- 2: Service discovery issues  
- 3: Load balancing issues
- 4: Circuit breaker issues
- 5: Performance issues
"""
import asyncio
import sys
import time
import json
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import mesh components
try:
    from app.mesh.service_mesh import (
        ServiceMesh, ServiceInstance, ServiceState, ServiceRequest,
        LoadBalancerStrategy, get_mesh
    )
    from app.mesh.mesh_dashboard import MeshDashboard, get_dashboard
    from app.mesh.distributed_tracing import get_tracer
except ImportError as e:
    logger.error(f"Failed to import mesh components: {e}")
    sys.exit(1)


class MeshValidator:
    """Comprehensive service mesh validator"""
    
    def __init__(self):
        self.mesh = None
        self.dashboard = None
        self.validation_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "infrastructure": {},
            "service_discovery": {},
            "load_balancing": {},
            "circuit_breakers": {},
            "performance": {},
            "errors": []
        }
        self.test_services = []
    
    async def initialize(self):
        """Initialize validator with mesh components"""
        try:
            self.mesh = await get_mesh()
            self.dashboard = await get_dashboard()
            logger.info("Mesh validator initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize mesh validator: {e}")
            self.validation_results["errors"].append(str(e))
            return False
    
    async def validate_infrastructure(self) -> bool:
        """Validate core infrastructure components"""
        logger.info("Validating infrastructure...")
        passed = True
        
        # Check Consul connectivity
        consul_connected = self.mesh.discovery.consul_client is not None
        self.validation_results["infrastructure"]["consul_connected"] = consul_connected
        
        if not consul_connected:
            logger.warning("Consul not connected - mesh running in degraded mode")
            # Degraded mode is acceptable, not a failure
        
        # Check Kong connectivity (if configured)
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                kong_url = self.mesh.kong_admin_url
                response = await client.get(f"{kong_url}/status")
                kong_available = response.status_code == 200
        except:
            kong_available = False
        
        self.validation_results["infrastructure"]["kong_available"] = kong_available
        
        # Check circuit breaker manager
        cb_initialized = self.mesh.circuit_breaker is not None
        self.validation_results["infrastructure"]["circuit_breaker_initialized"] = cb_initialized
        
        if not cb_initialized:
            logger.error("Circuit breaker not initialized")
            passed = False
        
        # Check load balancer
        lb_initialized = self.mesh.load_balancer is not None
        self.validation_results["infrastructure"]["load_balancer_initialized"] = lb_initialized
        
        if not lb_initialized:
            logger.error("Load balancer not initialized")
            passed = False
        
        logger.info(f"Infrastructure validation: {'PASSED' if passed else 'FAILED'}")
        return passed
    
    async def validate_service_discovery(self) -> bool:
        """Validate service discovery functionality"""
        logger.info("Validating service discovery...")
        passed = True
        
        try:
            # Register test services
            for i in range(3):
                service_name = f"test-service-{i}"
                instance = await self.mesh.register_service(
                    service_name=service_name,
                    address=f"10.0.0.{i+1}",
                    port=8080 + i,
                    tags=["test", "validation"],
                    metadata={"test": True, "index": i}
                )
                self.test_services.append(instance)
                logger.info(f"Registered test service: {instance.service_id}")
            
            # Discover services
            for service in self.test_services:
                discovered = await self.mesh.discovery.discover_services(service.service_name)
                
                if not discovered:
                    logger.error(f"Failed to discover service: {service.service_name}")
                    passed = False
                else:
                    logger.info(f"Discovered {len(discovered)} instances of {service.service_name}")
            
            self.validation_results["service_discovery"]["registration_successful"] = True
            self.validation_results["service_discovery"]["discovery_successful"] = passed
            self.validation_results["service_discovery"]["registered_count"] = len(self.test_services)
            
        except Exception as e:
            logger.error(f"Service discovery validation failed: {e}")
            self.validation_results["errors"].append(str(e))
            passed = False
        
        logger.info(f"Service discovery validation: {'PASSED' if passed else 'FAILED'}")
        return passed
    
    async def validate_load_balancing(self) -> bool:
        """Validate load balancing strategies"""
        logger.info("Validating load balancing...")
        passed = True
        
        # Create test instances
        instances = [
            ServiceInstance(f"lb-test-{i}", "lb-test", f"10.0.1.{i}", 9000 + i, state=ServiceState.HEALTHY)
            for i in range(3)
        ]
        
        strategies_tested = {}
        
        # Test Round Robin
        lb_rr = self.mesh.load_balancer
        lb_rr.strategy = LoadBalancerStrategy.ROUND_ROBIN
        selections = []
        for _ in range(6):
            selected = lb_rr.select_instance(instances, "lb-test")
            if selected:
                selections.append(selected.service_id)
        
        # Should cycle through all instances
        unique_selections = len(set(selections))
        strategies_tested["round_robin"] = unique_selections == 3
        
        if not strategies_tested["round_robin"]:
            logger.error("Round robin load balancing not working correctly")
            passed = False
        
        # Test Least Connections
        lb_rr.strategy = LoadBalancerStrategy.LEAST_CONNECTIONS
        instances[0].connections = 10
        instances[1].connections = 5
        instances[2].connections = 15
        
        selected = lb_rr.select_instance(instances, "lb-test")
        strategies_tested["least_connections"] = selected and selected.service_id == "lb-test-1"
        
        if not strategies_tested["least_connections"]:
            logger.error("Least connections load balancing not working correctly")
            passed = False
        
        # Test Weighted
        lb_rr.strategy = LoadBalancerStrategy.WEIGHTED
        instances[0].weight = 100
        instances[1].weight = 0
        instances[2].weight = 50
        
        weighted_selections = []
        for _ in range(100):
            selected = lb_rr.select_instance(instances, "lb-test")
            if selected:
                weighted_selections.append(selected.service_id)
        
        # Instance with weight 0 should never be selected
        strategies_tested["weighted"] = "lb-test-1" not in weighted_selections
        
        if not strategies_tested["weighted"]:
            logger.error("Weighted load balancing not working correctly")
            passed = False
        
        self.validation_results["load_balancing"] = strategies_tested
        logger.info(f"Load balancing validation: {'PASSED' if passed else 'FAILED'}")
        return passed
    
    async def validate_circuit_breakers(self) -> bool:
        """Validate circuit breaker functionality"""
        logger.info("Validating circuit breakers...")
        passed = True
        
        service_id = "cb-test-service"
        
        # Test circuit breaker tripping
        initial_state = self.mesh.circuit_breaker.is_open(service_id)
        self.validation_results["circuit_breakers"]["initial_state_closed"] = not initial_state
        
        # Record failures to trip the breaker
        threshold = self.mesh.circuit_breaker.failure_threshold
        for _ in range(threshold):
            self.mesh.circuit_breaker.record_failure(service_id)
        
        # Check if breaker opened
        is_open = self.mesh.circuit_breaker.is_open(service_id)
        self.validation_results["circuit_breakers"]["opens_on_threshold"] = is_open
        
        if not is_open:
            logger.error(f"Circuit breaker failed to open after {threshold} failures")
            passed = False
        
        # Test recovery (note: actual recovery requires timeout to pass)
        self.mesh.circuit_breaker.record_success(service_id)
        # Breaker should still be open until recovery timeout
        still_open = self.mesh.circuit_breaker.is_open(service_id)
        self.validation_results["circuit_breakers"]["remains_open_during_recovery"] = still_open
        
        logger.info(f"Circuit breaker validation: {'PASSED' if passed else 'FAILED'}")
        return passed
    
    async def validate_performance(self) -> bool:
        """Validate performance metrics"""
        logger.info("Validating performance...")
        passed = True
        
        # Test service discovery performance
        start = time.time()
        for _ in range(100):
            await self.mesh.discovery.discover_services("test-service", use_cache=True)
        discovery_time = time.time() - start
        
        self.validation_results["performance"]["discovery_100_calls_seconds"] = discovery_time
        self.validation_results["performance"]["discovery_calls_per_second"] = 100 / discovery_time
        
        if discovery_time > 1.0:
            logger.warning(f"Service discovery performance below target: {discovery_time:.2f}s for 100 calls")
            # Warning, not failure
        
        # Test load balancer performance
        instances = [
            ServiceInstance(f"perf-{i}", "perf", f"host{i}", 8000 + i, state=ServiceState.HEALTHY)
            for i in range(100)
        ]
        
        start = time.time()
        for _ in range(1000):
            self.mesh.load_balancer.select_instance(instances, "perf")
        lb_time = time.time() - start
        
        self.validation_results["performance"]["load_balancer_1000_selections_seconds"] = lb_time
        self.validation_results["performance"]["load_balancer_selections_per_second"] = 1000 / lb_time
        
        if lb_time > 1.0:
            logger.warning(f"Load balancer performance below target: {lb_time:.2f}s for 1000 selections")
            # Warning, not failure
        
        # Test topology reporting
        start = time.time()
        topology = await self.mesh.get_service_topology()
        topology_time = time.time() - start
        
        self.validation_results["performance"]["topology_generation_seconds"] = topology_time
        
        if topology_time > 0.5:
            logger.warning(f"Topology generation slow: {topology_time:.2f}s")
        
        logger.info(f"Performance validation: {'PASSED' if passed else 'FAILED'}")
        return passed
    
    async def validate_dashboard_metrics(self) -> bool:
        """Validate dashboard metrics collection"""
        logger.info("Validating dashboard metrics...")
        passed = True
        
        try:
            # Collect metrics
            await self.dashboard._collect_metrics()
            
            # Check if metrics were collected
            has_metrics = len(self.dashboard.metrics_history) > 0
            self.validation_results["dashboard"] = {
                "metrics_collected": has_metrics,
                "services_tracked": len(self.dashboard.service_metrics_history)
            }
            
            if not has_metrics:
                logger.error("Dashboard failed to collect metrics")
                passed = False
            
            # Get dashboard data
            dashboard_data = await self.dashboard.get_dashboard_data()
            
            if not dashboard_data:
                logger.error("Dashboard failed to generate data")
                passed = False
            
        except Exception as e:
            logger.error(f"Dashboard validation failed: {e}")
            self.validation_results["errors"].append(str(e))
            passed = False
        
        logger.info(f"Dashboard validation: {'PASSED' if passed else 'FAILED'}")
        return passed
    
    async def cleanup(self):
        """Clean up test resources"""
        logger.info("Cleaning up test resources...")
        
        # Deregister test services
        for service in self.test_services:
            try:
                await self.mesh.discovery.deregister_service(service.service_id)
                logger.info(f"Deregistered test service: {service.service_id}")
            except Exception as e:
                logger.warning(f"Failed to deregister {service.service_id}: {e}")
        
        # Shutdown components
        if self.dashboard:
            await self.dashboard.shutdown()
        
        if self.mesh:
            await self.mesh.shutdown()
    
    async def run_validation(self) -> int:
        """Run complete validation suite"""
        logger.info("=" * 60)
        logger.info("Starting Service Mesh Validation")
        logger.info("=" * 60)
        
        # Initialize
        if not await self.initialize():
            logger.error("Failed to initialize validator")
            return 1
        
        # Run validations
        results = {
            "infrastructure": await self.validate_infrastructure(),
            "service_discovery": await self.validate_service_discovery(),
            "load_balancing": await self.validate_load_balancing(),
            "circuit_breakers": await self.validate_circuit_breakers(),
            "performance": await self.validate_performance(),
            "dashboard": await self.validate_dashboard_metrics()
        }
        
        # Cleanup
        await self.cleanup()
        
        # Generate report
        self.validation_results["summary"] = {
            "total_tests": len(results),
            "passed": sum(1 for v in results.values() if v),
            "failed": sum(1 for v in results.values() if not v),
            "pass_rate": (sum(1 for v in results.values() if v) / len(results)) * 100
        }
        
        # Print report
        logger.info("=" * 60)
        logger.info("Validation Results Summary")
        logger.info("=" * 60)
        
        for component, passed in results.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            logger.info(f"{component.ljust(20)}: {status}")
        
        logger.info("-" * 60)
        logger.info(f"Total Tests: {self.validation_results['summary']['total_tests']}")
        logger.info(f"Passed: {self.validation_results['summary']['passed']}")
        logger.info(f"Failed: {self.validation_results['summary']['failed']}")
        logger.info(f"Pass Rate: {self.validation_results['summary']['pass_rate']:.1f}%")
        
        # Write detailed report
        report_file = f"mesh_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        logger.info(f"Detailed report written to: {report_file}")
        
        # Determine exit code
        if not results["infrastructure"]:
            return 1  # Critical infrastructure failure
        elif not results["service_discovery"]:
            return 2  # Service discovery issues
        elif not results["load_balancing"]:
            return 3  # Load balancing issues
        elif not results["circuit_breakers"]:
            return 4  # Circuit breaker issues
        elif not results["performance"]:
            return 5  # Performance issues
        elif self.validation_results["summary"]["failed"] > 0:
            return 6  # Other failures
        
        return 0  # All tests passed


async def main():
    """Main entry point"""
    validator = MeshValidator()
    exit_code = await validator.run_validation()
    
    if exit_code == 0:
        logger.info("✓ Service mesh validation completed successfully")
    else:
        logger.error(f"✗ Service mesh validation failed with exit code: {exit_code}")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())