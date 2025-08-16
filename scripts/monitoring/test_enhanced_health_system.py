#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
Comprehensive Test Suite for Enhanced Health Monitoring System
Tests all components of the ultra-enhanced health monitoring with circuit breakers
"""

import asyncio
import time
import json
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, '/opt/sutazaiapp/backend')

from app.core.health_monitoring import (
    get_health_monitoring_service, 
    ServiceStatus, 
    SystemStatus,
    HealthMonitoringService
)
from app.core.circuit_breaker_integration import (
    get_circuit_breaker_manager,
    SimpleCircuitBreaker,
    CircuitState
)

class HealthMonitoringTestSuite:
    """Comprehensive test suite for health monitoring system"""
    
    def __init__(self):
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            status = "✅ PASS"
        else:
            self.failed_tests += 1
            status = "❌ FAIL"
        
        result = {
            "test": test_name,
            "status": status,
            "passed": passed,
            "details": details,
            "timestamp": time.time()
        }
        self.test_results.append(result)
        logger.info(f"{status}: {test_name}")
        if details and not passed:
            logger.info(f"   Details: {details}")
    
    async def test_circuit_breaker_basic_functionality(self):
        """Test basic circuit breaker functionality"""
        try:
            # Create a simple circuit breaker
            breaker = SimpleCircuitBreaker(
                name="test_breaker",
                failure_threshold=3,
                recovery_timeout=5.0,
                timeout=1.0
            )
            
            # Test initial state
            assert breaker.state == CircuitState.CLOSED
            assert breaker.consecutive_failures == 0
            assert breaker.is_healthy == True
            
            # Test successful call
            async def success_func():
                return "success"
            
            result = await breaker.call(success_func)
            assert result == "success"
            assert breaker.successful_requests == 1
            
            # Test failing call
            async def fail_func():
                raise Exception("Test failure")
            
            # Cause failures to open circuit
            for i in range(3):
                try:
                    await breaker.call(fail_func)
                except Exception:
                    pass
            
            # Circuit should be open now
            assert breaker.state == CircuitState.OPEN
            assert breaker.is_healthy == False
            
            # Test that circuit rejects calls
            try:
                await breaker.call(success_func)
                self.log_result("Circuit Breaker Basic Functionality", False, "Circuit should reject calls when open")
                return
            except Exception:
                pass  # Expected
            
            self.log_result("Circuit Breaker Basic Functionality", True)
            
        except Exception as e:
            self.log_result("Circuit Breaker Basic Functionality", False, str(e))
    
    async def test_circuit_breaker_manager(self):
        """Test circuit breaker manager functionality"""
        try:
            manager = await get_circuit_breaker_manager()
            
            # Create breakers
            breaker1 = await manager.get_or_create_breaker("service1", failure_threshold=2)
            breaker2 = await manager.get_or_create_breaker("service2", failure_threshold=3)
            
            assert breaker1.name == "service1"
            assert breaker2.name == "service2"
            assert breaker1.failure_threshold == 2
            assert breaker2.failure_threshold == 3
            
            # Test getting existing breaker
            breaker1_again = await manager.get_or_create_breaker("service1")
            assert breaker1 is breaker1_again  # Should be same instance
            
            # Test stats
            stats = manager.get_all_stats()
            assert "service1" in stats
            assert "service2" in stats
            assert stats["service1"]["name"] == "service1"
            
            self.log_result("Circuit Breaker Manager", True)
            
        except Exception as e:
            self.log_result("Circuit Breaker Manager", False, str(e))
    
    async def test_health_monitoring_service_creation(self):
        """Test health monitoring service creation"""
        try:
            # Create health monitoring service
            health_monitor = await get_health_monitoring_service()
            
            assert isinstance(health_monitor, HealthMonitoringService)
            assert health_monitor._service_checkers is not None
            assert len(health_monitor._service_checkers) > 0
            
            # Check that default service checkers are registered
            expected_services = [
                'redis', 'database', 'ollama', 'task_queue',
                'vector_db_qdrant', 'vector_db_chromadb',
                'hardware_optimizer', 'ai_orchestrator'
            ]
            
            for service in expected_services:
                assert service in health_monitor._service_checkers
            
            self.log_result("Health Monitoring Service Creation", True)
            
        except Exception as e:
            self.log_result("Health Monitoring Service Creation", False, str(e))
    
    async def test_basic_health_check_performance(self):
        """Test basic health check performance (should be <100ms)"""
        try:
            health_monitor = await get_health_monitoring_service()
            
            # Warm up
            await health_monitor.get_basic_health()
            
            # Test performance
            start_time = time.time()
            health_result = await health_monitor.get_basic_health()
            end_time = time.time()
            
            response_time_ms = (end_time - start_time) * 1000
            
            # Validate response structure
            assert "status" in health_result
            assert "timestamp" in health_result
            assert "services" in health_result
            assert "response_time_ms" in health_result
            assert "check_type" in health_result
            
            # Validate performance (should be fast due to caching)
            performance_ok = response_time_ms < 100  # 100ms threshold
            
            self.log_result(
                "Basic Health Check Performance", 
                performance_ok,
                f"Response time: {response_time_ms:.1f}ms (target: <100ms)"
            )
            
        except Exception as e:
            self.log_result("Basic Health Check Performance", False, str(e))
    
    async def test_detailed_health_check_structure(self):
        """Test detailed health check response structure"""
        try:
            health_monitor = await get_health_monitoring_service()
            
            # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test circuit breakers for testing
            circuit_manager = await get_circuit_breaker_manager()
            redis_breaker = await circuit_manager.get_or_create_breaker("redis")
            health_monitor.register_circuit_breaker('redis', redis_breaker)
            
            health_report = await health_monitor.get_detailed_health()
            
            # Validate overall structure
            assert hasattr(health_report, 'overall_status')
            assert hasattr(health_report, 'timestamp')
            assert hasattr(health_report, 'services')
            assert hasattr(health_report, 'performance_metrics')
            assert hasattr(health_report, 'system_resources')
            assert hasattr(health_report, 'alerts')
            assert hasattr(health_report, 'recommendations')
            
            # Validate services structure
            assert isinstance(health_report.services, dict)
            
            # Check that services have expected fields
            for service_name, metrics in health_report.services.items():
                assert hasattr(metrics, 'name')
                assert hasattr(metrics, 'status')
                assert hasattr(metrics, 'response_time_ms')
                assert hasattr(metrics, 'last_check')
                assert isinstance(metrics.status, ServiceStatus)
                assert isinstance(metrics.response_time_ms, float)
            
            # Validate system resources
            if "error" not in health_report.system_resources:
                assert "cpu" in health_report.system_resources
                assert "memory" in health_report.system_resources
                assert "disk" in health_report.system_resources
            
            self.log_result("Detailed Health Check Structure", True)
            
        except Exception as e:
            self.log_result("Detailed Health Check Structure", False, str(e))
    
    async def test_prometheus_metrics_generation(self):
        """Test Prometheus metrics generation"""
        try:
            health_monitor = await get_health_monitoring_service()
            
            metrics_str = await health_monitor.get_prometheus_metrics()
            
            assert isinstance(metrics_str, str)
            assert len(metrics_str) > 0
            
            # Check for expected Prometheus format elements
            assert "# HELP" in metrics_str
            assert "# TYPE" in metrics_str
            assert "sutazai_service_health" in metrics_str
            assert "sutazai_system_health" in metrics_str
            
            # Should contain system metrics
            assert "sutazai_cpu_usage_percent" in metrics_str
            assert "sutazai_memory_usage_percent" in metrics_str
            
            # Validate basic Prometheus format
            lines = metrics_str.split('\n')
            metric_lines = [line for line in lines if not line.startswith('#') and line.strip()]
            
            for line in metric_lines:
                if line.strip():
                    # Basic format validation: metric_name{labels} value
                    assert ' ' in line, f"Invalid metric format: {line}"
                    parts = line.rsplit(' ', 1)
                    assert len(parts) == 2, f"Invalid metric format: {line}"
                    metric_part, value_part = parts
                    try:
                        float(value_part)  # Value should be numeric
                    except ValueError:
                        assert False, f"Non-numeric value in metric: {line}"
            
            self.log_result("Prometheus Metrics Generation", True)
            
        except Exception as e:
            self.log_result("Prometheus Metrics Generation", False, str(e))
    
    async def test_service_timeout_handling(self):
        """Test service timeout handling"""
        try:
            health_monitor = await get_health_monitoring_service()
            
            # Register a slow service checker
            async def slow_service_check():
                await asyncio.sleep(2.0)  # Longer than timeout
                return True
            
            health_monitor.register_service_checker("slow_service", slow_service_check)
            health_monitor._service_timeouts["slow_service"] = 0.5  # 500ms timeout
            
            start_time = time.time()
            metrics = await health_monitor._check_service_with_timeout(
                "slow_service", 
                slow_service_check, 
                timeout=0.5
            )
            end_time = time.time()
            
            # Should timeout quickly
            response_time = (end_time - start_time) * 1000
            assert response_time < 1000  # Should be much less than 2000ms
            assert metrics.status == ServiceStatus.TIMEOUT
            assert "timed out" in metrics.error_message.lower()
            
            self.log_result("Service Timeout Handling", True)
            
        except Exception as e:
            self.log_result("Service Timeout Handling", False, str(e))
    
    async def test_circuit_breaker_integration_with_health_check(self):
        """Test circuit breaker integration with health checks"""
        try:
            health_monitor = await get_health_monitoring_service()
            circuit_manager = await get_circuit_breaker_manager()
            
            # Create a circuit breaker for testing
            test_breaker = await circuit_manager.get_or_create_breaker(
                "test_integration_service",
                failure_threshold=2,
                recovery_timeout=5.0
            )
            
            # Register with health monitor
            health_monitor.register_circuit_breaker("test_integration_service", test_breaker)
            
            # Create a failing service checker
            call_count = 0
            async def failing_service():
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise Exception("Service failure")
                return True
            
            health_monitor.register_service_checker("test_integration_service", failing_service)
            
            # First few calls should fail and open the circuit
            metrics1 = await health_monitor._check_service_with_timeout(
                "test_integration_service",
                failing_service,
                timeout=1.0
            )
            
            metrics2 = await health_monitor._check_service_with_timeout(
                "test_integration_service", 
                failing_service,
                timeout=1.0
            )
            
            # Circuit should be open now
            metrics3 = await health_monitor._check_service_with_timeout(
                "test_integration_service",
                failing_service, 
                timeout=1.0
            )
            
            # Third call should show circuit breaker is open
            assert metrics3.status == ServiceStatus.CIRCUIT_OPEN
            assert metrics3.circuit_breaker_state == "open"
            
            self.log_result("Circuit Breaker Integration with Health Check", True)
            
        except Exception as e:
            self.log_result("Circuit Breaker Integration with Health Check", False, str(e))
    
    async def run_all_tests(self):
        """Run all tests"""
        logger.info("🚀 Starting Enhanced Health Monitoring System Test Suite")
        logger.info("=" * 70)
        
        test_methods = [
            self.test_circuit_breaker_basic_functionality,
            self.test_circuit_breaker_manager,
            self.test_health_monitoring_service_creation,
            self.test_basic_health_check_performance,
            self.test_detailed_health_check_structure,
            self.test_prometheus_metrics_generation,
            self.test_service_timeout_handling,
            self.test_circuit_breaker_integration_with_health_check,
        ]
        
        start_time = time.time()
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                test_name = test_method.__name__.replace('test_', '').replace('_', ' ').title()
                self.log_result(test_name, False, f"Test execution failed: {str(e)}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info("=" * 70)
        logger.info(f"🏁 Test Suite Completed in {total_time:.2f}s")
        logger.info(f"📊 Results: {self.passed_tests}/{self.total_tests} tests passed")
        
        if self.failed_tests > 0:
            logger.error(f"❌ {self.failed_tests} tests failed")
            logger.error("\n🔍 Failed Test Details:")
            for result in self.test_results:
                if not result['passed']:
                    logger.info(f"   • {result['test']}: {result['details']}")
        else:
            logger.info("🎉 All tests passed!")
        
        logger.info(f"\n📈 Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%")
        
        # Save detailed results
        results_file = "/opt/sutazaiapp/backend/health_monitoring_test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "summary": {
                    "total_tests": self.total_tests,
                    "passed_tests": self.passed_tests,
                    "failed_tests": self.failed_tests,
                    "success_rate": (self.passed_tests/self.total_tests)*100,
                    "total_time": total_time
                },
                "detailed_results": self.test_results
            }, f, indent=2)
        
        logger.info(f"📄 Detailed results saved to: {results_file}")
        
        return self.failed_tests == 0


async def main():
    """Main test runner"""
    test_suite = HealthMonitoringTestSuite()
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("\n✅ OVERALL: Enhanced Health Monitoring System is ready for production!")
        exit(0)
    else:
        logger.error("\n❌ OVERALL: Some tests failed - system needs attention before production use")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())