#!/usr/bin/env python3
"""
Live Agent Integration Test
Tests a real agent using the BaseAgentV2 framework with Ollama
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Add agents directory to path
sys.path.append('/opt/sutazaiapp/agents')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestAgent:
    """Simple test agent using BaseAgentV2"""
    
    def __init__(self):
        # Import BaseAgentV2
        from core.base_agent_v2 import BaseAgentV2
        from core.ollama_integration import OllamaIntegration
        
        self.agent = BaseAgentV2(
            config_path='/dev/null',  # Use defaults
            max_concurrent_tasks=1,
            max_ollama_connections=1
        )
        
        self.ollama = OllamaIntegration(
            base_url="http://localhost:11434",
            default_model="tinyllama"
        )
    
    async def test_basic_integration(self):
        """Test basic agent-Ollama integration"""
        logger.info("Testing live agent integration...")
        
        try:
            # Initialize async components
            await self.agent._setup_async_components()
            
            # Test Ollama query
            response = await self.agent.query_ollama(
                "Hello! Please respond with exactly: 'Agent integration successful'",
                model="tinyllama"
            )
            
            if response and "successful" in response.lower():
                logger.info(f"✓ Agent-Ollama integration working: {response.strip()}")
                return True
            else:
                logger.error(f"✗ Unexpected response: {response}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Integration test failed: {e}")
            return False
        finally:
            await self.agent._cleanup_async_components()
    
    async def test_direct_ollama(self):
        """Test direct Ollama integration"""
        logger.info("Testing direct Ollama integration...")
        
        try:
            async with self.ollama:
                response = await self.ollama.generate(
                    "Say 'Direct Ollama works' and nothing else",
                    model="tinyllama",
                    temperature=0.1,
                    max_tokens=10
                )
                
                if response and "works" in response.lower():
                    logger.info(f"✓ Direct Ollama integration working: {response.strip()}")
                    return True
                else:
                    logger.error(f"✗ Unexpected direct response: {response}")
                    return False
                    
        except Exception as e:
            logger.error(f"✗ Direct integration test failed: {e}")
            return False
    
    async def test_health_check(self):
        """Test agent health check"""
        logger.info("Testing agent health check...")
        
        try:
            await self.agent._setup_async_components()
            
            health = await self.agent.health_check()
            
            if health.get("healthy", False):
                logger.info("✓ Agent health check passed")
                logger.info(f"  - Ollama: {'healthy' if health.get('ollama_healthy') else 'unhealthy'}")
                logger.info(f"  - Backend: {'healthy' if health.get('backend_healthy') else 'unhealthy'}")
                logger.info(f"  - Model: {health.get('model', 'unknown')}")
                return True
            else:
                logger.error("✗ Agent health check failed")
                return False
                
        except Exception as e:
            logger.error(f"✗ Health check failed: {e}")
            return False
        finally:
            await self.agent._cleanup_async_components()

async def main():
    """Run live integration tests"""
    test_agent = TestAgent()
    
    logger.info("=" * 60)
    logger.info("LIVE AGENT INTEGRATION TEST")
    logger.info("=" * 60)
    
    tests = [
        ("Health Check", test_agent.test_health_check),
        ("Direct Ollama", test_agent.test_direct_ollama),
        ("Agent Integration", test_agent.test_basic_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name:<20}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 ALL TESTS PASSED! Ollama integration is working perfectly!")
        return 0
    elif passed > 0:
        logger.info("⚠️  Some tests passed. Integration is partially working.")
        return 1
    else:
        logger.error("❌ All tests failed. Integration needs fixing.")
        return 2

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)