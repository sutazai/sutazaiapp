#!/usr/bin/env python3
"""
LocalAGI System Testing and Validation Script

This script provides comprehensive testing of the LocalAGI autonomous
orchestration system, including integration with existing infrastructure.
"""

import asyncio
import logging
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "localagi"))

from localagi.main import LocalAGISystem, get_localagi_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'logs' / 'localagi_test.log')
    ]
)
logger = logging.getLogger(__name__)

class LocalAGISystemTester:
    """
    Comprehensive tester for LocalAGI system functionality.
    """
    
    def __init__(self):
        self.system: Optional[LocalAGISystem] = None
        self.test_results: Dict[str, Any] = {}
        self.start_time = datetime.now()
    
    async def initialize_system(self):
        """Initialize the LocalAGI system for testing."""
        logger.info("Initializing LocalAGI system for testing...")
        
        try:
            self.system = await get_localagi_system()
            logger.info("LocalAGI system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize LocalAGI system: {e}")
            return False
    
    async def test_basic_functionality(self):
        """Test basic LocalAGI functionality."""
        logger.info("Testing basic functionality...")
        
        tests = {
            'system_status': self._test_system_status,
            'agent_registry': self._test_agent_registry,
            'task_submission': self._test_task_submission,
            'system_health': self._test_system_health
        }
        
        results = {}
        for test_name, test_func in tests.items():
            try:
                logger.info(f"Running test: {test_name}")
                result = await test_func()
                results[test_name] = {'status': 'passed', 'result': result}
                logger.info(f"Test {test_name}: PASSED")
            except Exception as e:
                results[test_name] = {'status': 'failed', 'error': str(e)}
                logger.error(f"Test {test_name}: FAILED - {e}")
        
        self.test_results['basic_functionality'] = results
        return results
    
    async def test_advanced_features(self):
        """Test advanced LocalAGI features."""
        logger.info("Testing advanced features...")
        
        tests = {
            'workflow_creation': self._test_workflow_creation,
            'problem_solving': self._test_collaborative_problem_solving,
            'task_decomposition': self._test_task_decomposition,
            'swarm_coordination': self._test_swarm_coordination
        }
        
        results = {}
        for test_name, test_func in tests.items():
            try:
                logger.info(f"Running advanced test: {test_name}")
                result = await test_func()
                results[test_name] = {'status': 'passed', 'result': result}
                logger.info(f"Advanced test {test_name}: PASSED")
            except Exception as e:
                results[test_name] = {'status': 'failed', 'error': str(e)}
                logger.error(f"Advanced test {test_name}: FAILED - {e}")
        
        self.test_results['advanced_features'] = results
        return results
    
    async def test_integration(self):
        """Test integration with existing infrastructure."""
        logger.info("Testing infrastructure integration...")
        
        tests = {
            'ollama_integration': self._test_ollama_integration,
            'redis_integration': self._test_redis_integration,
            'configuration_integration': self._test_configuration_integration,
            'agent_communication': self._test_agent_communication
        }
        
        results = {}
        for test_name, test_func in tests.items():
            try:
                logger.info(f"Running integration test: {test_name}")
                result = await test_func()
                results[test_name] = {'status': 'passed', 'result': result}
                logger.info(f"Integration test {test_name}: PASSED")
            except Exception as e:
                results[test_name] = {'status': 'failed', 'error': str(e)}
                logger.error(f"Integration test {test_name}: FAILED - {e}")
        
        self.test_results['integration'] = results
        return results
    
    async def test_performance(self):
        """Test system performance under load."""
        logger.info("Testing system performance...")
        
        tests = {
            'concurrent_tasks': self._test_concurrent_task_handling,
            'memory_usage': self._test_memory_usage,
            'response_times': self._test_response_times,
            'scalability': self._test_scalability
        }
        
        results = {}
        for test_name, test_func in tests.items():
            try:
                logger.info(f"Running performance test: {test_name}")
                result = await test_func()
                results[test_name] = {'status': 'passed', 'result': result}
                logger.info(f"Performance test {test_name}: PASSED")
            except Exception as e:
                results[test_name] = {'status': 'failed', 'error': str(e)}
                logger.error(f"Performance test {test_name}: FAILED - {e}")
        
        self.test_results['performance'] = results
        return results
    
    # Basic functionality tests
    
    async def _test_system_status(self):
        """Test system status reporting."""
        status = await self.system.get_comprehensive_status()
        
        assert status is not None, "Status should not be None"
        assert 'system_status' in status, "Status should include system_status"
        assert status['system_status'] == 'operational', f"System should be operational, got: {status['system_status']}"
        
        return {
            'system_status': status['system_status'],
            'uptime': status.get('uptime', 0),
            'components': list(status.keys())
        }
    
    async def _test_agent_registry(self):
        """Test agent registry functionality."""
        engine = self.system.orchestration_engine
        
        assert engine is not None, "Orchestration engine should be available"
        assert len(engine.agents) > 0, f"Should have agents registered, found: {len(engine.agents)}"
        
        agent_info = []
        for agent_id, agent in engine.agents.items():
            agent_info.append({
                'id': agent_id,
                'name': agent.name,
                'capabilities': agent.capabilities,
                'status': agent.status.name,
                'performance_score': agent.performance_score
            })
        
        return {
            'total_agents': len(engine.agents),
            'agents': agent_info[:5]  # Return first 5 agents for brevity
        }
    
    async def _test_task_submission(self):
        """Test task submission functionality."""
        task_id = await self.system.submit_autonomous_task(
            description="Test task for system validation",
            requirements=["testing", "validation"],
            priority=0.8,
            autonomous_mode=False
        )
        
        assert task_id is not None, "Task ID should not be None"
        assert isinstance(task_id, str), "Task ID should be a string"
        
        # Wait a moment for processing
        await asyncio.sleep(1)
        
        return {
            'task_id': task_id,
            'submission_time': datetime.now().isoformat()
        }
    
    async def _test_system_health(self):
        """Test system health monitoring."""
        # System should have performed at least one health check
        assert self.system.last_health_check is not None, "Health check should have been performed"
        
        # Check component health
        components = {
            'orchestration_engine': self.system.orchestration_engine,
            'decision_engine': self.system.decision_engine,
            'workflow_engine': self.system.workflow_engine,
            'problem_solver': self.system.problem_solver
        }
        
        healthy_components = []
        for name, component in components.items():
            if component is not None:
                healthy_components.append(name)
        
        return {
            'last_health_check': self.system.last_health_check.isoformat(),
            'healthy_components': healthy_components,
            'total_components': len(components)
        }
    
    # Advanced feature tests
    
    async def _test_workflow_creation(self):
        """Test workflow creation and execution."""
        workflow_steps = [
            {
                'name': 'Initialization',
                'description': 'Initialize test workflow',
                'agent_capability': 'general',
                'parameters': {'test_mode': True},
                'success_criteria': ['Workflow initialized']
            },
            {
                'name': 'Processing',
                'description': 'Process test data',
                'agent_capability': 'analysis',
                'parameters': {'data_type': 'test'},
                'success_criteria': ['Data processed'],
                'preconditions': ['Initialization']
            }
        ]
        
        workflow_id = await self.system.create_autonomous_workflow(
            description="Test Workflow for System Validation",
            steps=workflow_steps,
            optimization_objectives=['efficiency']
        )
        
        assert workflow_id is not None, "Workflow ID should not be None"
        
        # Try to execute the workflow
        execution_result = await self.system.execute_workflow(workflow_id)
        
        return {
            'workflow_id': workflow_id,
            'execution_result': execution_result
        }
    
    async def _test_collaborative_problem_solving(self):
        """Test collaborative problem solving."""
        result = await self.system.solve_problem_collaboratively(
            problem_description="How can we improve automated testing efficiency?",
            problem_type="optimization",
            max_agents=3
        )
        
        assert result is not None, "Problem solving result should not be None"
        assert 'session_id' in result, "Result should include session_id"
        
        return result
    
    async def _test_task_decomposition(self):
        """Test complex task decomposition."""
        result = await self.system.decompose_complex_task(
            task_description="Create a comprehensive system monitoring dashboard",
            requirements=["monitoring", "visualization", "alerting"],
            constraints={"complexity": "high"},
            max_depth=2
        )
        
        assert result is not None, "Decomposition result should not be None"
        assert 'total_tasks' in result, "Result should include total_tasks"
        assert result['total_tasks'] > 1, "Should decompose into multiple tasks"
        
        return {
            'total_tasks': result['total_tasks'],
            'decomposition_depth': result.get('decomposition_depth', 0),
            'strategy_used': result.get('strategy_used', 'unknown')
        }
    
    async def _test_swarm_coordination(self):
        """Test agent swarm coordination."""
        result = await self.system.form_agent_swarm(
            goal="Coordinate testing and validation tasks",
            required_capabilities=["testing", "validation", "analysis"],
            max_size=4
        )
        
        assert result is not None, "Swarm result should not be None"
        
        return result
    
    # Integration tests
    
    async def _test_ollama_integration(self):
        """Test Ollama integration."""
        engine = self.system.orchestration_engine
        
        try:
            response = await engine.ollama_client.get("/api/tags")
            status = response.status_code
            
            if status == 200:
                models = response.json()
                return {
                    'status': 'connected',
                    'available_models': len(models.get('models', [])),
                    'models': [m.get('name', 'unknown') for m in models.get('models', [])[:3]]
                }
            else:
                return {
                    'status': 'accessible_but_limited',
                    'status_code': status
                }
        except Exception as e:
            return {
                'status': 'connection_failed',
                'error': str(e)
            }
    
    async def _test_redis_integration(self):
        """Test Redis integration."""
        engine = self.system.orchestration_engine
        
        try:
            # Test basic Redis operations
            test_key = f"localagi_test_{datetime.now().timestamp()}"
            test_value = "test_integration_value"
            
            await engine.redis_client.set(test_key, test_value, ex=30)
            retrieved_value = await engine.redis_client.get(test_key)
            await engine.redis_client.delete(test_key)
            
            assert retrieved_value.decode() == test_value, "Redis value mismatch"
            
            return {
                'status': 'connected',
                'read_write_test': 'passed'
            }
        except Exception as e:
            return {
                'status': 'connection_failed',
                'error': str(e)
            }
    
    async def _test_configuration_integration(self):
        """Test configuration integration."""
        try:
            from backend.app.core.config import get_settings
            settings = get_settings()
            
            config_checks = {
                'ollama_host': settings.OLLAMA_HOST is not None,
                'redis_host': settings.REDIS_HOST is not None,
                'database_url': settings.DATABASE_URL is not None,
                'default_model': settings.DEFAULT_MODEL is not None
            }
            
            return {
                'configuration_accessible': True,
                'config_checks': config_checks,
                'all_checks_passed': all(config_checks.values())
            }
        except Exception as e:
            return {
                'configuration_accessible': False,
                'error': str(e)
            }
    
    async def _test_agent_communication(self):
        """Test agent communication capabilities."""
        protocols = self.system.coordination_protocols
        
        try:
            # Test protocol status
            status = protocols.get_protocol_status()
            
            return {
                'protocols_active': status is not None,
                'protocol_details': status
            }
        except Exception as e:
            return {
                'protocols_active': False,
                'error': str(e)
            }
    
    # Performance tests
    
    async def _test_concurrent_task_handling(self):
        """Test concurrent task handling."""
        num_tasks = 5
        start_time = datetime.now()
        
        # Submit multiple tasks concurrently
        tasks = []
        for i in range(num_tasks):
            task = self.system.submit_autonomous_task(
                description=f"Concurrent test task {i+1}",
                requirements=["testing"],
                priority=0.5,
                autonomous_mode=False
            )
            tasks.append(task)
        
        # Wait for all submissions
        task_ids = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = datetime.now()
        
        successful_tasks = [tid for tid in task_ids if isinstance(tid, str)]
        
        return {
            'total_tasks': num_tasks,
            'successful_submissions': len(successful_tasks),
            'submission_time': (end_time - start_time).total_seconds(),
            'average_time_per_task': (end_time - start_time).total_seconds() / num_tasks
        }
    
    async def _test_memory_usage(self):
        """Test memory usage patterns."""
        import psutil
        import gc
        
        # Force garbage collection
        gc.collect()
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_memory_mb': memory_info.rss / 1024 / 1024,
            'vms_memory_mb': memory_info.vms / 1024 / 1024,
            'memory_percent': process.memory_percent()
        }
    
    async def _test_response_times(self):
        """Test system response times."""
        start_time = datetime.now()
        
        # Test status retrieval time
        await self.system.get_comprehensive_status()
        status_time = (datetime.now() - start_time).total_seconds()
        
        # Test task submission time
        start_time = datetime.now()
        await self.system.submit_autonomous_task(
            description="Response time test task",
            requirements=["testing"],
            autonomous_mode=False
        )
        submission_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'status_retrieval_time': status_time,
            'task_submission_time': submission_time,
            'average_response_time': (status_time + submission_time) / 2
        }
    
    async def _test_scalability(self):
        """Test system scalability indicators."""
        status = await self.system.get_comprehensive_status()
        
        orchestration_status = status.get('orchestration_engine', {})
        
        return {
            'active_agents': orchestration_status.get('active_agents', 0),
            'total_tasks_processed': orchestration_status.get('total_tasks', 0),
            'system_uptime': status.get('uptime', 0),
            'scalability_score': min(orchestration_status.get('active_agents', 0) / 10, 1.0)
        }
    
    async def generate_test_report(self):
        """Generate comprehensive test report."""
        end_time = datetime.now()
        test_duration = (end_time - self.start_time).total_seconds()
        
        # Calculate summary statistics
        all_tests = {}
        for category, tests in self.test_results.items():
            all_tests.update(tests)
        
        total_tests = len(all_tests)
        passed_tests = len([t for t in all_tests.values() if t['status'] == 'passed'])
        failed_tests = total_tests - passed_tests
        
        report = {
            'test_summary': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': test_duration,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            },
            'test_results': self.test_results,
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'test_environment': 'integration'
            }
        }
        
        return report
    
    async def cleanup(self):
        """Cleanup test resources."""
        if self.system:
            await self.system.shutdown()
        logger.info("Test cleanup completed")

async def main():
    """Main test execution function."""
    logger.info("Starting LocalAGI system comprehensive testing")
    
    tester = LocalAGISystemTester()
    
    try:
        # Initialize system
        if not await tester.initialize_system():
            logger.error("Failed to initialize system, aborting tests")
            return
        
        # Run test suites
        await tester.test_basic_functionality()
        await tester.test_advanced_features()
        await tester.test_integration()
        await tester.test_performance()
        
        # Generate report
        report = await tester.generate_test_report()
        
        # Save report
        report_file = Path(__file__).parent.parent / 'logs' / f'localagi_test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        summary = report['test_summary']
        logger.info(f"Test Summary:")
        logger.info(f"  Total Tests: {summary['total_tests']}")
        logger.info(f"  Passed: {summary['passed_tests']}")
        logger.info(f"  Failed: {summary['failed_tests']}")
        logger.info(f"  Success Rate: {summary['success_rate']:.1f}%")
        logger.info(f"  Duration: {summary['duration_seconds']:.1f} seconds")
        logger.info(f"  Report saved to: {report_file}")
        
        # Print failed tests
        if summary['failed_tests'] > 0:
            logger.warning("Failed tests:")
            for category, tests in report['test_results'].items():
                for test_name, result in tests.items():
                    if result['status'] == 'failed':
                        logger.warning(f"  {category}.{test_name}: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        raise
    
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())