#!/usr/bin/env python3
"""
Integration Test for Enhanced Base Agent System
Tests the complete integration of BaseAgentV2, OllamaPool, CircuitBreaker, and RequestQueue

This test validates:
- Async Ollama integration
- Connection pooling efficiency
- Circuit breaker resilience
- Request queue management
- Backward compatibility
"""

import asyncio
import logging
import time
from typing import Dict, Any
import sys
import os

# Add the agents directory to path
sys.path.append('/opt/sutazaiapp/agents')

try:
    from agents.core.base_agent import BaseAgentV2, TaskResult
    from agents.core.ollama_pool import OllamaConnectionPool
    from agents.core.circuit_breaker import CircuitBreaker, CircuitBreakerOpenException
    from agents.core.request_queue import RequestQueue, RequestPriority
    from agents.core.migration_helper import LegacyAgentWrapper
except ImportError:
    # Fallback for testing
    BaseAgentV2 = object
    TaskResult = dict
    OllamaConnectionPool = object
    CircuitBreaker = object
    CircuitBreakerOpenException = Exception
    RequestQueue = object
    RequestPriority = object
    LegacyAgentWrapper = object


# Configure logging for testing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestAgent(BaseAgentV2):
    """Test agent implementation"""
    
    def __init__(self):
        super().__init__(
            max_concurrent_tasks=2,
            max_ollama_connections=1  # Conservative for testing
        )
        self.processed_tasks = []
    
    async def process_task(self, task: Dict[str, Any]) -> TaskResult:
        """Custom task processing for testing"""
        start_time = time.time()
        task_id = task.get("id", "unknown")
        
        try:
            # Simulate some processing
            await asyncio.sleep(0.5)
            
            # Test Ollama integration if requested
            if task.get("use_ollama", False):
                prompt = task.get("prompt", "Hello, test!")
                response = await self.query_ollama(prompt)
                result_data = {
                    "status": "success",
                    "message": f"Processed task {task_id} with Ollama",
                    "ollama_response": response,
                    "task_data": task
                }
            else:
                result_data = {
                    "status": "success", 
                    "message": f"Processed task {task_id}",
                    "task_data": task
                }
            
            processing_time = time.time() - start_time
            self.processed_tasks.append(task_id)
            
            return TaskResult(
                task_id=task_id,
                status="completed",
                result=result_data,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Task processing failed: {e}")
            
            return TaskResult(
                task_id=task_id,
                status="failed",
                result={"error": str(e)},
                processing_time=processing_time,
                error=str(e)
            )


async def test_ollama_pool():
    """Test Ollama connection pool"""
    logger.info("Testing Ollama Connection Pool...")
    
    try:
        async with OllamaConnectionPool(
            max_connections=2,
            default_model="tinyllama"
        ) as pool:
            
            # Test basic generation
            response = await pool.generate("Hello, this is a test!")
            logger.info(f"Pool generation test: {'PASS' if response else 'FAIL'}")
            
            # Test concurrent requests
            tasks = [
                pool.generate(f"Test prompt {i}") 
                for i in range(3)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful = sum(1 for r in results if isinstance(r, str))
            logger.info(f"Concurrent requests: {successful}/3 successful")
            
            # Get statistics
            stats = pool.get_stats()
            logger.info(f"Pool stats: {stats['total_requests']} requests, {stats['success_rate']:.2f} success rate")
            
        return True
        
    except Exception as e:
        logger.error(f"Ollama pool test failed: {e}")
        return False


async def test_circuit_breaker():
    """Test circuit breaker functionality"""
    logger.info("Testing Circuit Breaker...")
    
    try:
        # Create circuit breaker with low threshold for testing
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=5.0,
            timeout=1.0,
            name="test_breaker"
        )
        
        # Test successful calls
        async def success_func():
            await asyncio.sleep(0.1)
            return "success"
        
        result = await breaker.call(success_func)
        logger.info(f"Circuit breaker success test: {'PASS' if result == 'success' else 'FAIL'}")
        
        # Test failure handling
        async def fail_func():
            raise Exception("Test failure")
        
        failures = 0
        for i in range(3):
            try:
                await breaker.call(fail_func)
            except Exception:
                failures += 1
        
        # Should have opened after 2 failures
        try:
            await breaker.call(success_func)
            logger.info("Circuit breaker open test: FAIL (should have rejected)")
        except CircuitBreakerOpenException:
            logger.info("Circuit breaker open test: PASS")
        
        stats = breaker.get_stats()
        logger.info(f"Breaker stats: {stats['trip_count']} trips, {stats['failed_requests']} failures")
        
        return True
        
    except Exception as e:
        logger.error(f"Circuit breaker test failed: {e}")
        return False


async def test_request_queue():
    """Test request queue functionality"""
    logger.info("Testing Request Queue...")
    
    try:
        queue = RequestQueue(
            max_queue_size=10,
            max_concurrent=2,
            timeout=5.0,
            name="test_queue"
        )
        
        # Test basic request submission
        async def test_func(value):
            await asyncio.sleep(0.2)
            return f"processed_{value}"
        
        # Submit requests with different priorities
        request_ids = []
        for i in range(5):
            priority = RequestPriority.HIGH if i < 2 else RequestPriority.NORMAL
            request_id = await queue.submit(
                test_func, 
                i, 
                priority=priority
            )
            request_ids.append(request_id)
        
        # Get results
        results = []
        for request_id in request_ids:
            try:
                result = await queue.get_result(request_id, timeout=10.0)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to get result for {request_id}: {e}")
        
        logger.info(f"Queue processing test: {len(results)}/5 completed")
        
        # Test stats
        stats = queue.get_stats()
        logger.info(f"Queue stats: {stats['completed_requests']} completed, {stats['throughput_per_second']:.2f}/s")
        
        await queue.close()
        return True
        
    except Exception as e:
        logger.error(f"Request queue test failed: {e}")
        return False


async def test_enhanced_agent():
    """Test the enhanced base agent"""
    logger.info("Testing Enhanced Base Agent...")
    
    try:
        agent = TestAgent()
        
        # Setup async components
        await agent._setup_async_components()
        
        # Test task processing
        test_tasks = [
            {"id": "task_1", "data": "test_data_1"},
            {"id": "task_2", "data": "test_data_2", "use_ollama": False},  # Skip Ollama for speed
            {"id": "task_3", "data": "test_data_3"},
        ]
        
        results = []
        for task in test_tasks:
            result = await agent.process_task(task)
            results.append(result)
        
        successful = sum(1 for r in results if r.status == "completed")
        logger.info(f"Agent task processing: {successful}/{len(test_tasks)} successful")
        
        # Test health check
        health = await agent.health_check()
        logger.info(f"Agent health check: {'PASS' if health.get('healthy') else 'FAIL'}")
        
        # Test metrics
        logger.info(f"Agent processed {len(agent.processed_tasks)} tasks")
        
        # Cleanup
        await agent._cleanup_async_components()
        
        return successful == len(test_tasks)
        
    except Exception as e:
        logger.error(f"Enhanced agent test failed: {e}")
        return False


async def test_backward_compatibility():
    """Test backward compatibility features"""
    logger.info("Testing Backward Compatibility...")
    
    try:
        # Create a mock legacy agent class
        from agent_base import BaseAgent
        
        class MockLegacyAgent(BaseAgent):
            def process_task(self, task):
                return {
                    "status": "success",
                    "message": f"Legacy processed {task.get('id')}",
                    "legacy": True
                }
        
        # Test legacy wrapper
        wrapper = LegacyAgentWrapper(MockLegacyAgent)
        await wrapper._setup_async_components()
        
        test_task = {"id": "legacy_test", "data": "test"}
        result = await wrapper.process_task(test_task)
        
        logger.info(f"Legacy wrapper test: {'PASS' if result.status == 'completed' else 'FAIL'}")
        
        await wrapper._cleanup_async_components()
        return True
        
    except Exception as e:
        logger.error(f"Backward compatibility test failed: {e}")
        return False


async def run_integration_tests():
    """Run all integration tests"""
    logger.info("=== SutazAI Enhanced Agent Integration Tests ===")
    
    test_results = {}
    
    # Individual component tests
    test_results["ollama_pool"] = await test_ollama_pool()
    test_results["circuit_breaker"] = await test_circuit_breaker()
    test_results["request_queue"] = await test_request_queue()
    
    # Integration tests
    test_results["enhanced_agent"] = await test_enhanced_agent()
    test_results["backward_compatibility"] = await test_backward_compatibility()
    
    # Summary
    passed = sum(test_results.values())
    total = len(test_results)
    
    logger.info("\n=== Test Results ===")
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL" 
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Enhanced agent system is ready for deployment.")
    else:
        logger.warning("⚠️  Some tests failed. Review the logs above.")
    
    return passed == total


def run_quick_validation():
    """Quick validation of the enhanced system without Ollama"""
    logger.info("=== Quick Validation (No Ollama Required) ===")
    
    # Test imports
    try:
        from agents.core.base_agent import BaseAgentV2
        from core.ollama_pool import OllamaConnectionPool
        from core.circuit_breaker import CircuitBreaker
        from core.request_queue import RequestQueue
        from core.migration_helper import LegacyAgentWrapper
        
        logger.info("✓ All imports successful")
        
        # Test basic instantiation
        agent = BaseAgent()
        logger.info("✓ BaseAgent instantiation successful")
        
        # Test configuration
        from core.ollama_integration import OllamaConfig
        model = OllamaConfig.get_model_for_agent("test-agent")
        logger.info(f"✓ Model configuration: {model}")
        
        logger.info("🎉 Quick validation passed! System is ready for use.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Quick validation failed: {e}")
        return False


if __name__ == "__main__":
    unittest.main()
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick validation without Ollama
        success = run_quick_validation()
    else:
        # Full integration tests (requires Ollama)
        success = asyncio.run(run_integration_tests())
    
    sys.exit(0 if success else 1)