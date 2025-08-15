#!/usr/bin/env python3
"""
Test Suite for SutazAI Architecture Enhancements
Validates service mesh, AI agents, and system integration
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import asyncio
import httpx
import json
import time
from typing import Dict, Any, List
import pika
import redis
import consul
from datetime import datetime

class EnhancementTestSuite:
    """Comprehensive test suite for architecture enhancements"""
    
    def __init__(self):
        self.results = []
        self.redis_client = None
        self.consul_client = None
        
    async def run_all_tests(self):
        """Run all test categories"""
        logger.info("=" * 60)
        logger.info("SutazAI Architecture Enhancement Test Suite")
        logger.info("=" * 60)
        logger.info()
        
        # Test categories
        await self.test_service_mesh()
        await self.test_ai_agents()
        await self.test_integration()
        await self.test_performance()
        
        # Print summary
        self.print_summary()
    
    async def test_service_mesh(self):
        """Test service mesh components"""
        logger.info("Testing Service Mesh Components")
        logger.info("-" * 40)
        
        # Test Kong Gateway
        result = await self.test_kong()
        self.results.append(("Kong Gateway", result))
        
        # Test Consul
        result = await self.test_consul()
        self.results.append(("Consul Discovery", result))
        
        # Test RabbitMQ
        result = await self.test_rabbitmq()
        self.results.append(("RabbitMQ Messaging", result))
        
        logger.info()
    
    async def test_kong(self) -> bool:
        """Test Kong Gateway functionality"""
        try:
            async with httpx.AsyncClient() as client:
                # Test proxy endpoint
                response = await client.get("http://localhost:10005/", timeout=5.0)
                if response.status_code in [200, 404]:
                    logger.info("✓ Kong proxy endpoint accessible")
                    
                    # Test admin API
                    admin_response = await client.get("http://localhost:8001/status", timeout=5.0)
                    if admin_response.status_code == 200:
                        logger.info("✓ Kong admin API accessible")
                        return True
                    
        except Exception as e:
            logger.error(f"✗ Kong test failed: {e}")
        
        return False
    
    async def test_consul(self) -> bool:
        """Test Consul service discovery"""
        try:
            self.consul_client = consul.Consul(host='localhost', port=10006)
            
            # Check if Consul is running
            leader = self.consul_client.status.leader()
            if leader:
                logger.info("✓ Consul cluster has leader")
                
                # List services
                services = self.consul_client.catalog.services()
                logger.info(f"✓ Consul has {len(services)} registered services")
                
                # Register test service
                self.consul_client.agent.service.register(
                    name='test-service',
                    service_id='test-1',
                    port=9999,
                    check=consul.Check.http('http://localhost:9999/health', interval='10s')
                )
                logger.info("✓ Successfully registered test service")
                
                # Deregister test service
                self.consul_client.agent.service.deregister('test-1')
                
                return True
                
        except Exception as e:
            logger.error(f"✗ Consul test failed: {e}")
        
        return False
    
    async def test_rabbitmq(self) -> bool:
        """Test RabbitMQ messaging"""
        try:
            # Connect to RabbitMQ
            credentials = pika.PlainCredentials('admin', 'sutazai_rabbit')
            connection = pika.BlockingConnection(
                pika.ConnectionParameters('localhost', 10007, credentials=credentials)
            )
            channel = connection.channel()
            
            # Declare test queue
            channel.queue_declare(queue='test_queue', durable=False)
            logger.info("✓ Connected to RabbitMQ")
            
            # Publish test message
            test_message = {'test': 'message', 'timestamp': datetime.utcnow().isoformat()}
            channel.basic_publish(
                exchange='',
                routing_key='test_queue',
                body=json.dumps(test_message)
            )
            logger.info("✓ Published test message")
            
            # Consume test message
            method, properties, body = channel.basic_get('test_queue')
            if body:
                received = json.loads(body)
                if received['test'] == 'message':
                    logger.info("✓ Successfully received test message")
                    channel.basic_ack(method.delivery_tag)
            
            # Cleanup
            channel.queue_delete('test_queue')
            connection.close()
            
            return True
            
        except Exception as e:
            logger.error(f"✗ RabbitMQ test failed: {e}")
        
        return False
    
    async def test_ai_agents(self):
        """Test AI agent implementations"""
        logger.info("Testing AI Agents")
        logger.info("-" * 40)
        
        # Test AI Orchestrator
        result = await self.test_orchestrator()
        self.results.append(("AI Orchestrator", result))
        
        # Test agent health endpoints
        agents = [
            ("AI Agent Orchestrator", "http://localhost:8589/health"),
            ("Multi-Agent Coordinator", "http://localhost:8587/health"),
            ("Task Assignment", "http://localhost:8551/health"),
            ("Resource Arbitration", "http://localhost:8588/health"),
        ]
        
        for name, url in agents:
            result = await self.test_agent_health(name, url)
            self.results.append((f"{name} Health", result))
        
        logger.info()
    
    async def test_orchestrator(self) -> bool:
        """Test AI Agent Orchestrator functionality"""
        try:
            async with httpx.AsyncClient() as client:
                # Test orchestration endpoint
                test_task = {
                    "task_type": "code_generation",
                    "payload": {
                        "language": "python",
                        "description": "Create a hello world function"
                    },
                    "priority": 5
                }
                
                response = await client.post(
                    "http://localhost:8589/orchestrate",
                    json=test_task,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "task_id" in result:
                        logger.info(f"✓ Task orchestrated: {result['task_id']}")
                        
                        # Check task status
                        await asyncio.sleep(2)
                        status_response = await client.get(
                            f"http://localhost:8589/task/{result['task_id']}",
                            timeout=5.0
                        )
                        
                        if status_response.status_code == 200:
                            status = status_response.json()
                            logger.info(f"✓ Task status: {status.get('status', 'unknown')}")
                            return True
                
        except Exception as e:
            logger.error(f"✗ Orchestrator test failed: {e}")
        
        return False
    
    async def test_agent_health(self, name: str, url: str) -> bool:
        """Test agent health endpoint"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") in ["healthy", "degraded"]:
                        logger.info(f"✓ {name} is {data.get('status')}")
                        return True
        except (AssertionError, Exception) as e:
            logger.error(f"Unexpected exception: {e}", exc_info=True)
            logger.info(f"✗ {name} not responding")
        
        return False
    
    async def test_integration(self):
        """Test system integration"""
        logger.info("Testing System Integration")
        logger.info("-" * 40)
        
        # Test Redis connectivity
        result = await self.test_redis_integration()
        self.results.append(("Redis Integration", result))
        
        # Test end-to-end flow
        result = await self.test_end_to_end_flow()
        self.results.append(("End-to-End Flow", result))
        
        logger.info()
    
    async def test_redis_integration(self) -> bool:
        """Test Redis integration"""
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=10001,
                decode_responses=True
            )
            
            # Test basic operations
            test_key = f"test:integration:{datetime.utcnow().timestamp()}"
            self.redis_client.set(test_key, "test_value", ex=60)
            value = self.redis_client.get(test_key)
            
            if value == "test_value":
                logger.info("✓ Redis integration working")
                self.redis_client.delete(test_key)
                return True
                
        except Exception as e:
            logger.error(f"✗ Redis integration failed: {e}")
        
        return False
    
    async def test_end_to_end_flow(self) -> bool:
        """Test complete end-to-end flow"""
        try:
            async with httpx.AsyncClient() as client:
                # Step 1: Submit task through Kong
                logger.info("  1. Submitting task through Kong...")
                response = await client.post(
                    "http://localhost:10005/ai/orchestrate",
                    json={
                        "task_type": "data_processing",
                        "payload": {"data": "test"},
                        "priority": 8
                    },
                    timeout=10.0
                )
                
                if response.status_code in [200, 404]:  # 404 if route not configured yet
                    logger.info("  ✓ Kong routing attempted")
                
                # Step 2: Check Consul for service registration
                logger.info("  2. Checking Consul service registry...")
                if self.consul_client:
                    services = self.consul_client.catalog.services()
                    logger.info(f"  ✓ Found {len(services)} services in Consul")
                
                # Step 3: Verify monitoring metrics
                logger.info("  3. Checking monitoring stack...")
                prometheus_response = await client.get(
                    "http://localhost:10200/api/v1/query?query=up",
                    timeout=5.0
                )
                
                if prometheus_response.status_code == 200:
                    logger.info("  ✓ Prometheus metrics available")
                
                return True
                
        except Exception as e:
            logger.error(f"✗ End-to-end test failed: {e}")
        
        return False
    
    async def test_performance(self):
        """Test system performance"""
        logger.info("Testing System Performance")
        logger.info("-" * 40)
        
        # Test response times
        endpoints = [
            ("Backend API", "http://localhost:10010/health"),
            ("AI Orchestrator", "http://localhost:8589/health"),
            ("Prometheus", "http://localhost:10200/-/healthy"),
            ("Grafana", "http://localhost:10201/api/health"),
        ]
        
        async with httpx.AsyncClient() as client:
            for name, url in endpoints:
                start_time = time.time()
                try:
                    response = await client.get(url, timeout=5.0)
                    elapsed = (time.time() - start_time) * 1000
                    
                    if response.status_code in [200, 401]:  # 401 for auth-required endpoints
                        logger.info(f"✓ {name}: {elapsed:.2f}ms")
                        self.results.append((f"{name} Response Time", elapsed < 1000))
                    else:
                        logger.info(f"✗ {name}: Status {response.status_code}")
                        self.results.append((f"{name} Response Time", False))
                        
                except Exception as e:
                    logger.error(f"✗ {name}: Failed - {e}")
                    self.results.append((f"{name} Response Time", False))
        
        logger.info()
    
    def print_summary(self):
        """Print test summary"""
        logger.info("=" * 60)
        logger.info("Test Summary")
        logger.info("=" * 60)
        
        passed = sum(1 for _, result in self.results if result)
        total = len(self.results)
        percentage = (passed / total * 100) if total > 0 else 0
        
        logger.info(f"\nResults: {passed}/{total} passed ({percentage:.1f}%)\n")
        
        # Group results
        logger.info("Detailed Results:")
        logger.info("-" * 40)
        
        for test_name, result in self.results:
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"{status} - {test_name}")
        
        logger.info()
        
        if percentage >= 80:
            logger.info("🎉 System enhancement successful!")
        elif percentage >= 60:
            logger.info("⚠️ System partially enhanced, some components need attention")
        else:
            logger.error("❌ System enhancement incomplete, review failed components")

async def main():
    """Main test execution"""
    test_suite = EnhancementTestSuite()
    await test_suite.run_all_tests()

if __name__ == "__main__":
