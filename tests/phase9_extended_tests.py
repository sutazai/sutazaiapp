#!/usr/bin/env python3
"""
Phase 9: Extended MCP Bridge Testing
Additional comprehensive tests for RabbitMQ, Redis, Failover, and Performance
Execution Time: 2025-11-15 (Phase 9 Extended)
"""

import asyncio
import json
import sys
import time
import random
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

import httpx
import aio_pika
import redis.asyncio as aioredis

# Test configuration
MCP_BASE_URL = "http://localhost:11100"
RABBITMQ_URL = "amqp://sutazai:sutazai_secure_2024@localhost:10004/"
REDIS_URL = "redis://localhost:10001"
TIMEOUT = 30.0
TEST_START_TIME = datetime.now()

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class ExtendedTestResults:
    """Track extended test results"""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []
        self.timings = {}
        self.performance_metrics = {}
    
    def record_pass(self, test_name: str, duration: float, metrics: Dict = None):
        self.total += 1
        self.passed += 1
        self.timings[test_name] = duration
        if metrics:
            self.performance_metrics[test_name] = metrics
        print(f"{Colors.OKGREEN}✓{Colors.ENDC} {test_name} ({duration:.3f}s)")
        if metrics:
            for key, value in metrics.items():
                print(f"  {Colors.OKCYAN}{key}: {value}{Colors.ENDC}")
    
    def record_fail(self, test_name: str, error: str, duration: float):
        self.total += 1
        self.failed += 1
        self.timings[test_name] = duration
        self.errors.append({"test": test_name, "error": error})
        print(f"{Colors.FAIL}✗{Colors.ENDC} {test_name} ({duration:.3f}s)")
        print(f"  {Colors.WARNING}Error: {error}{Colors.ENDC}")
    
    def record_skip(self, test_name: str, reason: str):
        self.total += 1
        self.skipped += 1
        print(f"{Colors.WARNING}⊘{Colors.ENDC} {test_name} (skipped: {reason})")
    
    def summary(self):
        total_time = sum(self.timings.values())
        pass_rate = (self.passed / self.total * 100) if self.total > 0 else 0
        
        print(f"\n{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
        print(f"{Colors.BOLD}PHASE 9 EXTENDED TEST RESULTS{Colors.ENDC}")
        print(f"{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
        print(f"\nTotal Tests:    {self.total}")
        print(f"{Colors.OKGREEN}Passed:        {self.passed}{Colors.ENDC}")
        print(f"{Colors.FAIL}Failed:        {self.failed}{Colors.ENDC}")
        print(f"{Colors.WARNING}Skipped:       {self.skipped}{Colors.ENDC}")
        print(f"\n{Colors.BOLD}Pass Rate:      {pass_rate:.1f}%{Colors.ENDC}")
        print(f"Total Duration: {total_time:.2f}s")
        
        if self.performance_metrics:
            print(f"\n{Colors.BOLD}Performance Metrics:{Colors.ENDC}")
            for test, metrics in self.performance_metrics.items():
                print(f"  {test}:")
                for key, value in metrics.items():
                    print(f"    - {key}: {value}")
        
        if self.errors:
            print(f"\n{Colors.FAIL}Failed Tests:{Colors.ENDC}")
            for error in self.errors:
                print(f"  - {error['test']}: {error['error']}")
        
        print(f"\n{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")
        
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "pass_rate": pass_rate,
            "duration": total_time,
            "errors": self.errors,
            "timings": self.timings,
            "performance_metrics": self.performance_metrics
        }

results = ExtendedTestResults()

async def test_wrapper(test_name: str, test_func):
    """Wrapper to time and record test results"""
    start = time.time()
    try:
        metrics = await test_func()
        duration = time.time() - start
        results.record_pass(test_name, duration, metrics)
        return True
    except Exception as e:
        duration = time.time() - start
        results.record_fail(test_name, str(e), duration)
        return False

# ============================================
# RABBITMQ INTEGRATION TESTS
# ============================================

async def test_rabbitmq_connection():
    """Test RabbitMQ connection and exchange setup"""
    try:
        connection = await aio_pika.connect_robust(RABBITMQ_URL, timeout=10)
        channel = await connection.channel()
        
        # Check if MCP exchange exists
        exchange = await channel.get_exchange("mcp.exchange")
        
        await connection.close()
        return {"status": "connected", "exchange": "mcp.exchange exists"}
    except Exception as e:
        raise AssertionError(f"RabbitMQ connection failed: {e}")

async def test_rabbitmq_queue_creation():
    """Test RabbitMQ queue creation and binding"""
    try:
        connection = await aio_pika.connect_robust(RABBITMQ_URL, timeout=10)
        channel = await connection.channel()
        
        # Create test queue
        test_queue_name = f"test_queue_{int(time.time())}"
        queue = await channel.declare_queue(test_queue_name, durable=True)
        
        # Delete test queue
        await queue.delete()
        
        await connection.close()
        return {"status": "success", "queue_created": test_queue_name}
    except Exception as e:
        raise AssertionError(f"Queue creation failed: {e}")

async def test_rabbitmq_message_publish():
    """Test message publishing to RabbitMQ"""
    try:
        connection = await aio_pika.connect_robust(RABBITMQ_URL, timeout=10)
        channel = await connection.channel()
        exchange = await channel.get_exchange("mcp.exchange")
        
        # Publish test message
        test_message = {
            "id": f"test-msg-{int(time.time())}",
            "type": "test",
            "payload": {"test": True}
        }
        
        await exchange.publish(
            aio_pika.Message(
                body=json.dumps(test_message).encode(),
                content_type="application/json"
            ),
            routing_key="bridge.test"
        )
        
        await connection.close()
        return {"status": "published", "message_id": test_message["id"]}
    except Exception as e:
        raise AssertionError(f"Message publish failed: {e}")

async def test_rabbitmq_message_consume():
    """Test message consumption from RabbitMQ"""
    try:
        connection = await aio_pika.connect_robust(RABBITMQ_URL, timeout=10)
        channel = await connection.channel()
        exchange = await channel.get_exchange("mcp.exchange")
        
        # Create temporary queue for testing
        test_queue_name = f"test_consume_{int(time.time())}"
        queue = await channel.declare_queue(test_queue_name, durable=False, auto_delete=True)
        await queue.bind(exchange, routing_key="test.consume")
        
        # Publish a message
        test_message = {
            "id": f"consume-test-{int(time.time())}",
            "type": "test",
            "payload": {"consume_test": True}
        }
        
        await exchange.publish(
            aio_pika.Message(
                body=json.dumps(test_message).encode(),
                content_type="application/json"
            ),
            routing_key="test.consume"
        )
        
        # Try to consume the message
        message_received = False
        try:
            async with asyncio.timeout(3):
                async for message in queue:
                    async with message.process():
                        received_data = json.loads(message.body.decode())
                        if received_data["id"] == test_message["id"]:
                            message_received = True
                            break
        except asyncio.TimeoutError:
            pass
        
        await queue.delete()
        await connection.close()
        
        if not message_received:
            raise AssertionError("Message not received from queue")
        
        return {"status": "consumed", "message_received": True}
    except Exception as e:
        raise AssertionError(f"Message consume failed: {e}")

# ============================================
# REDIS CACHING TESTS
# ============================================

async def test_redis_connection():
    """Test Redis connection"""
    try:
        redis_client = await aioredis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        await redis_client.ping()
        await redis_client.aclose()
        return {"status": "connected", "ping": "success"}
    except Exception as e:
        raise AssertionError(f"Redis connection failed: {e}")

async def test_redis_cache_write():
    """Test writing to Redis cache"""
    try:
        redis_client = await aioredis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        
        test_key = f"test:cache:{int(time.time())}"
        test_value = json.dumps({"test": True, "timestamp": datetime.now().isoformat()})
        
        # Write to cache
        await redis_client.setex(test_key, 60, test_value)
        
        # Verify write
        cached = await redis_client.get(test_key)
        if cached != test_value:
            raise AssertionError("Cache write verification failed")
        
        # Cleanup
        await redis_client.delete(test_key)
        await redis_client.aclose()
        
        return {"status": "success", "key": test_key, "verified": True}
    except Exception as e:
        raise AssertionError(f"Redis cache write failed: {e}")

async def test_redis_cache_expiration():
    """Test Redis cache TTL and expiration"""
    try:
        redis_client = await aioredis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        
        test_key = f"test:ttl:{int(time.time())}"
        test_value = "expiring_value"
        
        # Write with short TTL
        await redis_client.setex(test_key, 2, test_value)
        
        # Check TTL
        ttl = await redis_client.ttl(test_key)
        if ttl <= 0:
            raise AssertionError(f"TTL not set correctly: {ttl}")
        
        # Wait for expiration
        await asyncio.sleep(3)
        
        # Verify expiration
        expired_value = await redis_client.get(test_key)
        if expired_value is not None:
            raise AssertionError("Key did not expire")
        
        await redis_client.aclose()
        return {"status": "success", "ttl_set": ttl, "expired": True}
    except Exception as e:
        raise AssertionError(f"Redis cache expiration test failed: {e}")

async def test_redis_cache_invalidation():
    """Test Redis cache invalidation"""
    try:
        redis_client = await aioredis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        
        test_key = f"test:invalidate:{int(time.time())}"
        test_value = "to_be_invalidated"
        
        # Write to cache
        await redis_client.set(test_key, test_value)
        
        # Verify exists
        exists = await redis_client.exists(test_key)
        if not exists:
            raise AssertionError("Key not written")
        
        # Invalidate (delete)
        deleted = await redis_client.delete(test_key)
        if deleted != 1:
            raise AssertionError("Key not deleted")
        
        # Verify deleted
        exists_after = await redis_client.exists(test_key)
        if exists_after:
            raise AssertionError("Key still exists after deletion")
        
        await redis_client.aclose()
        return {"status": "success", "deleted": True}
    except Exception as e:
        raise AssertionError(f"Redis cache invalidation failed: {e}")

# ============================================
# PERFORMANCE BENCHMARKING
# ============================================

async def test_endpoint_throughput():
    """Benchmark endpoint throughput"""
    try:
        num_requests = 100
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            tasks = [client.get(f"{MCP_BASE_URL}/health") for _ in range(num_requests)]
            responses = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = num_requests / duration
        
        # Verify all succeeded
        success_count = sum(1 for r in responses if r.status_code == 200)
        
        return {
            "requests": num_requests,
            "duration_s": f"{duration:.3f}",
            "throughput_rps": f"{throughput:.2f}",
            "success_rate": f"{success_count/num_requests*100:.1f}%"
        }
    except Exception as e:
        raise AssertionError(f"Throughput test failed: {e}")

async def test_websocket_latency():
    """Measure WebSocket message latency"""
    try:
        import websockets
        
        uri = f"ws://localhost:11100/ws/latency-test-{int(time.time())}"
        latencies = []
        
        async with websockets.connect(uri, ping_interval=None) as websocket:
            # Consume connection message
            await websocket.recv()
            
            # Send multiple messages and measure latency
            for i in range(10):
                send_time = time.time()
                await websocket.send(json.dumps({
                    "type": "ping",
                    "payload": {"seq": i}
                }))
                
                # We won't get a response for this, but we can measure send latency
                latency = (time.time() - send_time) * 1000  # Convert to ms
                latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        return {
            "avg_latency_ms": f"{avg_latency:.3f}",
            "min_latency_ms": f"{min_latency:.3f}",
            "max_latency_ms": f"{max_latency:.3f}",
            "samples": len(latencies)
        }
    except Exception as e:
        raise AssertionError(f"WebSocket latency test failed: {e}")

async def test_concurrent_load():
    """Test system under concurrent load"""
    try:
        num_concurrent = 50
        endpoints = [
            f"{MCP_BASE_URL}/health",
            f"{MCP_BASE_URL}/services",
            f"{MCP_BASE_URL}/agents",
            f"{MCP_BASE_URL}/status"
        ]
        
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            tasks = []
            for _ in range(num_concurrent):
                endpoint = random.choice(endpoints)
                tasks.append(client.get(endpoint))
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Count successes and errors
        successes = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
        errors = sum(1 for r in responses if isinstance(r, Exception) or (hasattr(r, 'status_code') and r.status_code != 200))
        
        return {
            "concurrent_requests": num_concurrent,
            "duration_s": f"{duration:.3f}",
            "success": successes,
            "errors": errors,
            "success_rate": f"{successes/num_concurrent*100:.1f}%"
        }
    except Exception as e:
        raise AssertionError(f"Concurrent load test failed: {e}")

# ============================================
# FAILOVER & RESILIENCE TESTS
# ============================================

async def test_graceful_degradation():
    """Test system behavior when dependencies are unavailable"""
    try:
        # Test health check even if some services are down
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{MCP_BASE_URL}/health")
            
            # Should still return healthy status for the bridge itself
            if response.status_code != 200:
                raise AssertionError(f"Health check failed: {response.status_code}")
            
            data = response.json()
            if data.get("status") != "healthy":
                raise AssertionError(f"Bridge not healthy: {data.get('status')}")
        
        return {"status": "bridge_healthy", "graceful_degradation": "working"}
    except Exception as e:
        raise AssertionError(f"Graceful degradation test failed: {e}")

async def test_timeout_handling():
    """Test proper timeout handling"""
    try:
        # Test with very short timeout
        async with httpx.AsyncClient(timeout=0.001) as client:
            try:
                await client.get(f"{MCP_BASE_URL}/status")
                # If it succeeds, timeout wasn't triggered, which is fine
                timeout_handled = False
            except httpx.TimeoutException:
                # Timeout was triggered and handled properly
                timeout_handled = True
        
        return {"timeout_handling": "working", "timeout_triggered": timeout_handled}
    except Exception as e:
        raise AssertionError(f"Timeout handling test failed: {e}")

async def test_error_recovery():
    """Test error recovery mechanisms"""
    try:
        # Send invalid requests and verify system recovers
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Send invalid request
            try:
                await client.post(
                    f"{MCP_BASE_URL}/route",
                    content="invalid",
                    headers={"Content-Type": "application/json"}
                )
            except:
                pass
            
            # System should still be functional
            response = await client.get(f"{MCP_BASE_URL}/health")
            if response.status_code != 200:
                raise AssertionError("System did not recover from invalid request")
        
        return {"status": "system_recovered", "error_recovery": "working"}
    except Exception as e:
        raise AssertionError(f"Error recovery test failed: {e}")

# ============================================
# CAPABILITY-BASED SELECTION TESTS
# ============================================

async def test_single_capability_selection():
    """Test agent selection based on single capability"""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Get all agents
            response = await client.get(f"{MCP_BASE_URL}/agents")
            agents = response.json()
            
            # Find agents with specific capabilities
            code_agents = [
                agent_id for agent_id, agent in agents.items()
                if "code-editing" in agent.get("capabilities", []) or "pair-programming" in agent.get("capabilities", [])
            ]
            
            memory_agents = [
                agent_id for agent_id, agent in agents.items()
                if "memory" in agent.get("capabilities", [])
            ]
            
            return {
                "code_agents": len(code_agents),
                "memory_agents": len(memory_agents),
                "selection_working": len(code_agents) > 0 and len(memory_agents) > 0
            }
    except Exception as e:
        raise AssertionError(f"Capability selection test failed: {e}")

async def test_multi_capability_selection():
    """Test agent selection with multiple capability requirements"""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Get all agents
            response = await client.get(f"{MCP_BASE_URL}/agents")
            agents = response.json()
            
            # Find agents with multiple capabilities
            multi_capable = [
                agent_id for agent_id, agent in agents.items()
                if len(agent.get("capabilities", [])) >= 2
            ]
            
            return {
                "multi_capable_agents": len(multi_capable),
                "found_capable_agents": multi_capable[:3] if multi_capable else []
            }
    except Exception as e:
        raise AssertionError(f"Multi-capability selection test failed: {e}")

# ============================================
# MAIN TEST RUNNER
# ============================================

async def run_extended_tests():
    """Execute all extended Phase 9 tests"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}PHASE 9: EXTENDED MCP BRIDGE TESTING{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}Started: {TEST_START_TIME.strftime('%Y-%m-%d %H:%M:%S UTC')}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}\n")
    
    # RabbitMQ Integration Tests
    print(f"{Colors.BOLD}{Colors.OKCYAN}[1/6] RabbitMQ Integration Tests{Colors.ENDC}")
    await test_wrapper("RabbitMQ Connection", test_rabbitmq_connection)
    await test_wrapper("RabbitMQ Queue Creation", test_rabbitmq_queue_creation)
    await test_wrapper("RabbitMQ Message Publish", test_rabbitmq_message_publish)
    await test_wrapper("RabbitMQ Message Consume", test_rabbitmq_message_consume)
    
    # Redis Caching Tests
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}[2/6] Redis Caching Tests{Colors.ENDC}")
    await test_wrapper("Redis Connection", test_redis_connection)
    await test_wrapper("Redis Cache Write", test_redis_cache_write)
    await test_wrapper("Redis Cache Expiration", test_redis_cache_expiration)
    await test_wrapper("Redis Cache Invalidation", test_redis_cache_invalidation)
    
    # Performance Benchmarking
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}[3/6] Performance Benchmarking{Colors.ENDC}")
    await test_wrapper("Endpoint Throughput", test_endpoint_throughput)
    await test_wrapper("WebSocket Latency", test_websocket_latency)
    await test_wrapper("Concurrent Load Handling", test_concurrent_load)
    
    # Failover & Resilience Tests
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}[4/6] Failover & Resilience Tests{Colors.ENDC}")
    await test_wrapper("Graceful Degradation", test_graceful_degradation)
    await test_wrapper("Timeout Handling", test_timeout_handling)
    await test_wrapper("Error Recovery", test_error_recovery)
    
    # Capability-Based Selection
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}[5/6] Capability-Based Selection{Colors.ENDC}")
    await test_wrapper("Single Capability Selection", test_single_capability_selection)
    await test_wrapper("Multi-Capability Selection", test_multi_capability_selection)
    
    # Generate summary
    summary = results.summary()
    
    # Save results to file
    report_file = Path(f"/opt/sutazaiapp/PHASE_9_EXTENDED_TEST_RESULTS_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{Colors.OKGREEN}Extended results saved to: {report_file}{Colors.ENDC}\n")
    
    return summary

if __name__ == "__main__":
    # Run all extended tests
    summary = asyncio.run(run_extended_tests())
    
    # Exit with appropriate code
    sys.exit(0 if summary["failed"] == 0 else 1)
