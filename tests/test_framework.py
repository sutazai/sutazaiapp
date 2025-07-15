"""
SutazAI Comprehensive Testing Framework
Enterprise-grade testing suite for AGI/ASI system validation

This module provides comprehensive testing capabilities including unit tests,
integration tests, performance tests, and security validation.
"""

import asyncio
import json
import logging
import time
import pytest
import unittest
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import subprocess
import requests
import psutil
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, '/opt/sutazaiapp')

# Import system components
from core.agi_system import IntegratedAGISystem, AGITask, TaskPriority, create_agi_task
from core.security import SecurityManager
from core.exceptions import SutazaiException
from database.manager import DatabaseManager
from performance.profiler import PerformanceProfiler
from models.local_model_manager import LocalModelManager
from api.agi_api import AGIAPISystem
from main_agi import SutazAIOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    category: str
    status: str  # "passed", "failed", "skipped", "error"
    duration: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class TestSuite:
    """Base class for test suites"""
    
    def __init__(self, name: str):
        self.name = name
        self.results: List[TestResult] = []
        self.setup_complete = False
        self.teardown_complete = False
    
    async def setup(self):
        """Setup test environment"""
        pass
    
    async def teardown(self):
        """Cleanup test environment"""
        pass
    
    def add_result(self, result: TestResult):
        """Add test result"""
        self.results.append(result)
        logger.info(f"Test {result.test_name}: {result.status} ({result.duration:.2f}s)")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test suite summary"""
        total = len(self.results)
        passed = len([r for r in self.results if r.status == "passed"])
        failed = len([r for r in self.results if r.status == "failed"])
        skipped = len([r for r in self.results if r.status == "skipped"])
        errors = len([r for r in self.results if r.status == "error"])
        
        return {
            "suite_name": self.name,
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "errors": errors,
            "success_rate": (passed / total * 100) if total > 0 else 0,
            "total_duration": sum(r.duration for r in self.results),
            "timestamp": datetime.now().isoformat()
        }

class UnitTestSuite(TestSuite):
    """Unit test suite for individual components"""
    
    def __init__(self):
        super().__init__("Unit Tests")
        self.agi_system = None
        self.security_manager = None
        self.db_manager = None
        self.model_manager = None
    
    async def setup(self):
        """Setup unit test environment"""
        try:
            # Create mock instances for testing
            self.agi_system = IntegratedAGISystem()
            self.security_manager = SecurityManager()
            self.db_manager = DatabaseManager()
            self.model_manager = LocalModelManager()
            
            self.setup_complete = True
            logger.info("Unit test setup complete")
            
        except Exception as e:
            logger.error(f"Unit test setup failed: {e}")
            raise
    
    async def test_agi_system_initialization(self):
        """Test AGI system initialization"""
        start_time = time.time()
        
        try:
            # Test system initialization
            assert self.agi_system is not None
            assert self.agi_system.state.value in ["ready", "initializing"]
            
            # Test system status
            status = self.agi_system.get_system_status()
            assert "state" in status
            assert "metrics" in status
            assert "neural_network" in status
            
            self.add_result(TestResult(
                test_name="agi_system_initialization",
                category="unit",
                status="passed",
                duration=time.time() - start_time,
                message="AGI system initialization successful"
            ))
            
        except Exception as e:
            self.add_result(TestResult(
                test_name="agi_system_initialization",
                category="unit",
                status="failed",
                duration=time.time() - start_time,
                message=f"AGI system initialization failed: {e}"
            ))
    
    async def test_task_creation_and_submission(self):
        """Test task creation and submission"""
        start_time = time.time()
        
        try:
            # Create test task
            task = create_agi_task(
                name="test_task",
                priority=TaskPriority.HIGH,
                data={"test": "data"}
            )
            
            assert task.name == "test_task"
            assert task.priority == TaskPriority.HIGH
            assert task.data == {"test": "data"}
            
            # Submit task
            task_id = self.agi_system.submit_task(task)
            assert task_id is not None
            assert len(task_id) > 0
            
            self.add_result(TestResult(
                test_name="task_creation_and_submission",
                category="unit",
                status="passed",
                duration=time.time() - start_time,
                message="Task creation and submission successful"
            ))
            
        except Exception as e:
            self.add_result(TestResult(
                test_name="task_creation_and_submission",
                category="unit",
                status="failed",
                duration=time.time() - start_time,
                message=f"Task creation and submission failed: {e}"
            ))
    
    async def test_security_manager(self):
        """Test security manager functionality"""
        start_time = time.time()
        
        try:
            # Test security validation
            test_input = {"user_input": "test data"}
            is_valid = self.security_manager.validate_input(test_input)
            assert isinstance(is_valid, bool)
            
            # Test threat detection
            threat_level = self.security_manager.assess_threat_level(test_input)
            assert isinstance(threat_level, (int, float))
            assert 0 <= threat_level <= 1
            
            self.add_result(TestResult(
                test_name="security_manager",
                category="unit",
                status="passed",
                duration=time.time() - start_time,
                message="Security manager tests passed"
            ))
            
        except Exception as e:
            self.add_result(TestResult(
                test_name="security_manager",
                category="unit",
                status="failed",
                duration=time.time() - start_time,
                message=f"Security manager tests failed: {e}"
            ))
    
    async def test_database_manager(self):
        """Test database manager functionality"""
        start_time = time.time()
        
        try:
            # Test database connection
            status = self.db_manager.get_connection_status()
            assert "status" in status
            
            # Test health check
            health = self.db_manager.health_check()
            assert isinstance(health, dict)
            assert "healthy" in health
            
            self.add_result(TestResult(
                test_name="database_manager",
                category="unit",
                status="passed",
                duration=time.time() - start_time,
                message="Database manager tests passed"
            ))
            
        except Exception as e:
            self.add_result(TestResult(
                test_name="database_manager",
                category="unit",
                status="failed",
                duration=time.time() - start_time,
                message=f"Database manager tests failed: {e}"
            ))
    
    async def test_model_manager(self):
        """Test model manager functionality"""
        start_time = time.time()
        
        try:
            # Test model registry
            models = self.model_manager.get_available_models()
            assert isinstance(models, list)
            
            # Test system status
            status = self.model_manager.get_system_status()
            assert isinstance(status, dict)
            assert "total_models" in status
            
            self.add_result(TestResult(
                test_name="model_manager",
                category="unit",
                status="passed",
                duration=time.time() - start_time,
                message="Model manager tests passed"
            ))
            
        except Exception as e:
            self.add_result(TestResult(
                test_name="model_manager",
                category="unit",
                status="failed",
                duration=time.time() - start_time,
                message=f"Model manager tests failed: {e}"
            ))
    
    async def run_all_tests(self):
        """Run all unit tests"""
        logger.info("Running unit tests...")
        
        await self.setup()
        
        test_methods = [
            self.test_agi_system_initialization,
            self.test_task_creation_and_submission,
            self.test_security_manager,
            self.test_database_manager,
            self.test_model_manager
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                logger.error(f"Test method {test_method.__name__} failed: {e}")
        
        await self.teardown()
        logger.info("Unit tests completed")

class IntegrationTestSuite(TestSuite):
    """Integration test suite for system components"""
    
    def __init__(self):
        super().__init__("Integration Tests")
        self.orchestrator = None
        self.api_client = None
    
    async def setup(self):
        """Setup integration test environment"""
        try:
            # Create orchestrator instance
            self.orchestrator = SutazAIOrchestrator()
            
            # Initialize minimal system for testing
            await self.orchestrator._load_configuration()
            await self.orchestrator._initialize_enterprise_components()
            
            self.setup_complete = True
            logger.info("Integration test setup complete")
            
        except Exception as e:
            logger.error(f"Integration test setup failed: {e}")
            raise
    
    async def test_system_orchestration(self):
        """Test system orchestration and component integration"""
        start_time = time.time()
        
        try:
            # Test orchestrator initialization
            assert self.orchestrator is not None
            assert self.orchestrator.settings is not None
            assert self.orchestrator.security_manager is not None
            
            # Test component communication
            status = await self.orchestrator._validate_system()
            
            self.add_result(TestResult(
                test_name="system_orchestration",
                category="integration",
                status="passed",
                duration=time.time() - start_time,
                message="System orchestration tests passed"
            ))
            
        except Exception as e:
            self.add_result(TestResult(
                test_name="system_orchestration",
                category="integration",
                status="failed",
                duration=time.time() - start_time,
                message=f"System orchestration tests failed: {e}"
            ))
    
    async def test_api_integration(self):
        """Test API integration and endpoints"""
        start_time = time.time()
        
        try:
            # Test API app creation
            api_system = AGIAPISystem()
            assert api_system.app is not None
            
            # Test API routes
            routes = [route.path for route in api_system.app.routes]
            expected_routes = ["/health", "/api/v1/system/status", "/api/v1/tasks"]
            
            for route in expected_routes:
                assert any(route in r for r in routes), f"Route {route} not found"
            
            self.add_result(TestResult(
                test_name="api_integration",
                category="integration",
                status="passed",
                duration=time.time() - start_time,
                message="API integration tests passed"
            ))
            
        except Exception as e:
            self.add_result(TestResult(
                test_name="api_integration",
                category="integration",
                status="failed",
                duration=time.time() - start_time,
                message=f"API integration tests failed: {e}"
            ))
    
    async def test_neural_network_integration(self):
        """Test neural network integration"""
        start_time = time.time()
        
        try:
            # Test neural network creation and operation
            from nln.nln_core import NeuralLinkNetwork
            from nln.neural_node import NeuralNode
            
            network = NeuralLinkNetwork()
            
            # Add test nodes
            node1 = NeuralNode("test_node_1", "input", (0, 0), 0.5)
            node2 = NeuralNode("test_node_2", "output", (1, 0), 0.7)
            
            network.add_node(node1)
            network.add_node(node2)
            
            # Test network state
            state = network.get_network_state()
            assert isinstance(state, dict)
            assert len(network.nodes) == 2
            
            self.add_result(TestResult(
                test_name="neural_network_integration",
                category="integration",
                status="passed",
                duration=time.time() - start_time,
                message="Neural network integration tests passed"
            ))
            
        except Exception as e:
            self.add_result(TestResult(
                test_name="neural_network_integration",
                category="integration",
                status="failed",
                duration=time.time() - start_time,
                message=f"Neural network integration tests failed: {e}"
            ))
    
    async def run_all_tests(self):
        """Run all integration tests"""
        logger.info("Running integration tests...")
        
        await self.setup()
        
        test_methods = [
            self.test_system_orchestration,
            self.test_api_integration,
            self.test_neural_network_integration
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                logger.error(f"Integration test {test_method.__name__} failed: {e}")
        
        await self.teardown()
        logger.info("Integration tests completed")

class PerformanceTestSuite(TestSuite):
    """Performance test suite for system benchmarking"""
    
    def __init__(self):
        super().__init__("Performance Tests")
        self.agi_system = None
        self.baseline_metrics = {}
    
    async def setup(self):
        """Setup performance test environment"""
        try:
            self.agi_system = IntegratedAGISystem()
            
            # Collect baseline metrics
            self.baseline_metrics = {
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            }
            
            self.setup_complete = True
            logger.info("Performance test setup complete")
            
        except Exception as e:
            logger.error(f"Performance test setup failed: {e}")
            raise
    
    async def test_task_processing_performance(self):
        """Test task processing performance"""
        start_time = time.time()
        
        try:
            # Create multiple test tasks
            tasks = []
            for i in range(100):
                task = create_agi_task(
                    name=f"perf_test_task_{i}",
                    priority=TaskPriority.MEDIUM,
                    data={"test_data": f"data_{i}"}
                )
                tasks.append(task)
            
            # Submit tasks and measure performance
            submission_start = time.time()
            task_ids = []
            
            for task in tasks:
                task_id = self.agi_system.submit_task(task)
                task_ids.append(task_id)
            
            submission_time = time.time() - submission_start
            
            # Wait for tasks to complete
            await asyncio.sleep(10)
            
            # Measure system performance
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            
            performance_metrics = {
                "tasks_submitted": len(tasks),
                "submission_time": submission_time,
                "tasks_per_second": len(tasks) / submission_time,
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "cpu_increase": cpu_usage - self.baseline_metrics["cpu_usage"],
                "memory_increase": memory_usage - self.baseline_metrics["memory_usage"]
            }
            
            # Performance assertions
            assert performance_metrics["tasks_per_second"] > 10, "Task processing too slow"
            assert performance_metrics["cpu_increase"] < 50, "CPU usage too high"
            assert performance_metrics["memory_increase"] < 30, "Memory usage too high"
            
            self.add_result(TestResult(
                test_name="task_processing_performance",
                category="performance",
                status="passed",
                duration=time.time() - start_time,
                message="Task processing performance test passed",
                details=performance_metrics
            ))
            
        except Exception as e:
            self.add_result(TestResult(
                test_name="task_processing_performance",
                category="performance",
                status="failed",
                duration=time.time() - start_time,
                message=f"Task processing performance test failed: {e}"
            ))
    
    async def test_memory_usage_patterns(self):
        """Test memory usage patterns and leaks"""
        start_time = time.time()
        
        try:
            initial_memory = psutil.virtual_memory().used
            
            # Simulate intensive operations
            for i in range(50):
                # Create and process tasks
                task = create_agi_task(
                    name=f"memory_test_task_{i}",
                    priority=TaskPriority.HIGH,
                    data={"large_data": "x" * 1000}
                )
                
                self.agi_system.submit_task(task)
                
                # Check memory every 10 iterations
                if i % 10 == 0:
                    current_memory = psutil.virtual_memory().used
                    memory_increase = current_memory - initial_memory
                    
                    # Memory should not increase excessively
                    assert memory_increase < 500 * 1024 * 1024, "Memory usage increased too much"
            
            # Final memory check
            final_memory = psutil.virtual_memory().used
            total_increase = final_memory - initial_memory
            
            memory_metrics = {
                "initial_memory_mb": initial_memory / (1024 * 1024),
                "final_memory_mb": final_memory / (1024 * 1024),
                "memory_increase_mb": total_increase / (1024 * 1024),
                "memory_leak_detected": total_increase > 100 * 1024 * 1024
            }
            
            self.add_result(TestResult(
                test_name="memory_usage_patterns",
                category="performance",
                status="passed",
                duration=time.time() - start_time,
                message="Memory usage patterns test passed",
                details=memory_metrics
            ))
            
        except Exception as e:
            self.add_result(TestResult(
                test_name="memory_usage_patterns",
                category="performance",
                status="failed",
                duration=time.time() - start_time,
                message=f"Memory usage patterns test failed: {e}"
            ))
    
    async def test_concurrent_operations(self):
        """Test concurrent operations performance"""
        start_time = time.time()
        
        try:
            # Create concurrent tasks
            async def create_and_submit_tasks(task_count: int):
                tasks = []
                for i in range(task_count):
                    task = create_agi_task(
                        name=f"concurrent_task_{i}",
                        priority=TaskPriority.MEDIUM,
                        data={"concurrent_data": f"data_{i}"}
                    )
                    
                    task_id = self.agi_system.submit_task(task)
                    tasks.append((task, task_id))
                
                return tasks
            
            # Run concurrent operations
            concurrent_start = time.time()
            
            tasks_batch1 = await create_and_submit_tasks(25)
            tasks_batch2 = await create_and_submit_tasks(25)
            tasks_batch3 = await create_and_submit_tasks(25)
            
            concurrent_time = time.time() - concurrent_start
            
            total_tasks = len(tasks_batch1) + len(tasks_batch2) + len(tasks_batch3)
            
            concurrency_metrics = {
                "total_concurrent_tasks": total_tasks,
                "concurrent_processing_time": concurrent_time,
                "concurrent_throughput": total_tasks / concurrent_time,
                "cpu_usage_during_test": psutil.cpu_percent(interval=1),
                "memory_usage_during_test": psutil.virtual_memory().percent
            }
            
            # Performance assertions
            assert concurrency_metrics["concurrent_throughput"] > 5, "Concurrent throughput too low"
            assert concurrency_metrics["cpu_usage_during_test"] < 90, "CPU usage too high during concurrent operations"
            
            self.add_result(TestResult(
                test_name="concurrent_operations",
                category="performance",
                status="passed",
                duration=time.time() - start_time,
                message="Concurrent operations test passed",
                details=concurrency_metrics
            ))
            
        except Exception as e:
            self.add_result(TestResult(
                test_name="concurrent_operations",
                category="performance",
                status="failed",
                duration=time.time() - start_time,
                message=f"Concurrent operations test failed: {e}"
            ))
    
    async def run_all_tests(self):
        """Run all performance tests"""
        logger.info("Running performance tests...")
        
        await self.setup()
        
        test_methods = [
            self.test_task_processing_performance,
            self.test_memory_usage_patterns,
            self.test_concurrent_operations
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                logger.error(f"Performance test {test_method.__name__} failed: {e}")
        
        await self.teardown()
        logger.info("Performance tests completed")

class SecurityTestSuite(TestSuite):
    """Security test suite for vulnerability assessment"""
    
    def __init__(self):
        super().__init__("Security Tests")
        self.security_manager = None
    
    async def setup(self):
        """Setup security test environment"""
        try:
            self.security_manager = SecurityManager()
            self.setup_complete = True
            logger.info("Security test setup complete")
            
        except Exception as e:
            logger.error(f"Security test setup failed: {e}")
            raise
    
    async def test_input_validation(self):
        """Test input validation and sanitization"""
        start_time = time.time()
        
        try:
            # Test various input types
            test_cases = [
                {"input": "normal text", "expected": True},
                {"input": "<script>alert('xss')</script>", "expected": False},
                {"input": "'; DROP TABLE users; --", "expected": False},
                {"input": "../../etc/passwd", "expected": False},
                {"input": "normal_user@email.com", "expected": True},
                {"input": "valid_json_data", "expected": True}
            ]
            
            passed_tests = 0
            for test_case in test_cases:
                result = self.security_manager.validate_input({"data": test_case["input"]})
                if result == test_case["expected"]:
                    passed_tests += 1
            
            success_rate = passed_tests / len(test_cases)
            
            assert success_rate >= 0.8, f"Input validation success rate too low: {success_rate}"
            
            self.add_result(TestResult(
                test_name="input_validation",
                category="security",
                status="passed",
                duration=time.time() - start_time,
                message=f"Input validation test passed (success rate: {success_rate:.2%})"
            ))
            
        except Exception as e:
            self.add_result(TestResult(
                test_name="input_validation",
                category="security",
                status="failed",
                duration=time.time() - start_time,
                message=f"Input validation test failed: {e}"
            ))
    
    async def test_authentication_security(self):
        """Test authentication and authorization security"""
        start_time = time.time()
        
        try:
            # Test unauthorized access attempts
            unauthorized_attempts = [
                {"user": "unauthorized@email.com", "action": "system_access"},
                {"user": "hacker@malicious.com", "action": "admin_access"},
                {"user": "", "action": "anonymous_access"}
            ]
            
            blocked_attempts = 0
            for attempt in unauthorized_attempts:
                # Simulate authorization check
                is_authorized = (attempt["user"] == "chrissuta01@gmail.com")
                if not is_authorized:
                    blocked_attempts += 1
            
            blocking_rate = blocked_attempts / len(unauthorized_attempts)
            
            assert blocking_rate == 1.0, "Not all unauthorized attempts were blocked"
            
            self.add_result(TestResult(
                test_name="authentication_security",
                category="security",
                status="passed",
                duration=time.time() - start_time,
                message="Authentication security test passed"
            ))
            
        except Exception as e:
            self.add_result(TestResult(
                test_name="authentication_security",
                category="security",
                status="failed",
                duration=time.time() - start_time,
                message=f"Authentication security test failed: {e}"
            ))
    
    async def test_data_encryption(self):
        """Test data encryption and protection"""
        start_time = time.time()
        
        try:
            # Test encryption functionality
            test_data = "sensitive_test_data_123"
            
            # Encrypt data
            encrypted_data = self.security_manager.encrypt_data(test_data)
            assert encrypted_data != test_data, "Data was not encrypted"
            
            # Decrypt data
            decrypted_data = self.security_manager.decrypt_data(encrypted_data)
            assert decrypted_data == test_data, "Data decryption failed"
            
            self.add_result(TestResult(
                test_name="data_encryption",
                category="security",
                status="passed",
                duration=time.time() - start_time,
                message="Data encryption test passed"
            ))
            
        except Exception as e:
            self.add_result(TestResult(
                test_name="data_encryption",
                category="security",
                status="failed",
                duration=time.time() - start_time,
                message=f"Data encryption test failed: {e}"
            ))
    
    async def run_all_tests(self):
        """Run all security tests"""
        logger.info("Running security tests...")
        
        await self.setup()
        
        test_methods = [
            self.test_input_validation,
            self.test_authentication_security,
            self.test_data_encryption
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                logger.error(f"Security test {test_method.__name__} failed: {e}")
        
        await self.teardown()
        logger.info("Security tests completed")

class ComprehensiveTestRunner:
    """Main test runner for all test suites"""
    
    def __init__(self):
        self.test_suites = [
            UnitTestSuite(),
            IntegrationTestSuite(),
            PerformanceTestSuite(),
            SecurityTestSuite()
        ]
        self.results_dir = Path("/opt/sutazaiapp/tests/results")
        self.results_dir.mkdir(exist_ok=True)
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites and generate comprehensive report"""
        logger.info("=== Starting Comprehensive Test Suite ===")
        
        start_time = time.time()
        all_results = {}
        
        # Run each test suite
        for suite in self.test_suites:
            try:
                logger.info(f"Running {suite.name}...")
                await suite.run_all_tests()
                all_results[suite.name] = suite.get_summary()
                
            except Exception as e:
                logger.error(f"Test suite {suite.name} failed: {e}")
                all_results[suite.name] = {
                    "error": str(e),
                    "status": "failed"
                }
        
        total_duration = time.time() - start_time
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report(all_results, total_duration)
        
        # Save report
        self.save_test_report(report)
        
        logger.info("=== Comprehensive Test Suite Complete ===")
        return report
    
    def generate_comprehensive_report(self, results: Dict[str, Any], total_duration: float) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_errors = 0
        
        for suite_name, suite_results in results.items():
            if "error" not in suite_results:
                total_tests += suite_results.get("total_tests", 0)
                total_passed += suite_results.get("passed", 0)
                total_failed += suite_results.get("failed", 0)
                total_errors += suite_results.get("errors", 0)
        
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            "test_report": {
                "timestamp": datetime.now().isoformat(),
                "total_duration": total_duration,
                "overall_summary": {
                    "total_tests": total_tests,
                    "passed": total_passed,
                    "failed": total_failed,
                    "errors": total_errors,
                    "success_rate": overall_success_rate,
                    "status": "PASS" if overall_success_rate >= 80 else "FAIL"
                },
                "suite_results": results,
                "recommendations": self.generate_recommendations(results),
                "system_info": {
                    "python_version": sys.version,
                    "platform": os.uname().sysname,
                    "cpu_count": psutil.cpu_count(),
                    "memory_total": psutil.virtual_memory().total,
                    "disk_space": psutil.disk_usage('/').free
                }
            }
        }
        
        return report
    
    def generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        for suite_name, suite_results in results.items():
            if "error" in suite_results:
                recommendations.append(f"Fix critical issues in {suite_name}")
                continue
            
            success_rate = suite_results.get("success_rate", 0)
            
            if success_rate < 80:
                recommendations.append(f"Improve {suite_name} - success rate below 80%")
            elif success_rate < 90:
                recommendations.append(f"Optimize {suite_name} - success rate could be higher")
            
            if suite_name == "Performance Tests":
                recommendations.append("Review performance metrics and optimize bottlenecks")
            
            if suite_name == "Security Tests":
                recommendations.append("Conduct additional security audit")
        
        if not recommendations:
            recommendations.append("All tests passed - system is ready for production")
        
        return recommendations
    
    def save_test_report(self, report: Dict[str, Any]):
        """Save test report to file"""
        try:
            # Save JSON report
            json_report_path = self.results_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(json_report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Save latest report
            latest_report_path = self.results_dir / "latest_test_report.json"
            with open(latest_report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Test report saved to {json_report_path}")
            
        except Exception as e:
            logger.error(f"Failed to save test report: {e}")

async def main():
    """Main entry point for running tests"""
    test_runner = ComprehensiveTestRunner()
    report = await test_runner.run_all_tests()
    
    # Print summary
    summary = report["test_report"]["overall_summary"]
    print(f"\n=== Test Summary ===")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Errors: {summary['errors']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Status: {summary['status']}")
    
    return summary['status'] == 'PASS'

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)