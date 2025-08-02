#!/usr/bin/env python3
"""
Advanced Features Test Script for SutazAI
Demonstrates multi-model caching, request batching, GPU acceleration, and streaming
"""
import asyncio
import aiohttp
import json
import time
from typing import List, Dict, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedFeaturesDemo:
    """Demonstration of SutazAI's advanced performance features"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_streaming_chat(self):
        """Test real-time streaming chat functionality"""
        logger.info("🌊 Testing Streaming Chat")
        
        request_data = {
            "messages": [
                {"role": "user", "content": "Explain advanced computing in simple terms"}
            ],
            "model": "qwen2.5:3b",
            "temperature": 0.7,
            "stream": True
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/streaming/chat/stream",
                json=request_data
            ) as response:
                if response.status == 200:
                    logger.info("✅ Streaming started successfully")
                    
                    chunk_count = 0
                    start_time = time.time()
                    
                    async for line in response.content:
                        if line:
                            try:
                                line_str = line.decode().strip()
                                if line_str.startswith("data: "):
                                    data = json.loads(line_str[6:])
                                    
                                    if data["type"] == "start":
                                        logger.info(f"🚀 Stream started with model: {data['model']}")
                                    elif data["type"] == "chunk":
                                        chunk_count += 1
                                        content = data["content"]
                                        if content:
                                            print(f"  Chunk {chunk_count}: {content[:50]}..." if len(content) > 50 else f"  Chunk {chunk_count}: {content}")
                                    elif data["type"] == "complete":
                                        elapsed = time.time() - start_time
                                        logger.info(f"✅ Streaming completed in {elapsed:.2f}s with {chunk_count} chunks")
                                        break
                                    elif data["type"] == "error":
                                        logger.error(f"❌ Streaming error: {data['error']}")
                                        break
                                        
                            except json.JSONDecodeError:
                                continue
                else:
                    logger.error(f"❌ Streaming failed: HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ Error in streaming test: {e}")
    
    async def test_batch_processing(self):
        """Test batch processing for high-concurrency scenarios"""
        logger.info("📦 Testing Batch Processing")
        
        # Create a batch of requests
        batch_requests = [
            {"prompt": f"Write a {word} sentence about AI.", "temperature": 0.5}
            for word in ["short", "creative", "technical", "simple", "detailed"]
        ]
        
        request_data = {
            "requests": batch_requests,
            "model": "qwen2.5:3b",
            "batch_size": 3
        }
        
        try:
            start_time = time.time()
            
            async with self.session.post(
                f"{self.base_url}/api/v1/streaming/batch/process",
                json=request_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    elapsed = time.time() - start_time
                    
                    logger.info(f"✅ Batch processing completed in {elapsed:.2f}s")
                    logger.info(f"   Total requests: {result['total_requests']}")
                    logger.info(f"   Batch size: {result['batch_size']}")
                    logger.info(f"   Success rate: {result['success_rate']:.1%}")
                    
                    # Show some results
                    for i, res in enumerate(result['results'][:3]):
                        if res['success']:
                            response_preview = res['response'][:100] + "..." if len(res['response']) > 100 else res['response']
                            logger.info(f"   Result {i+1}: {response_preview}")
                        else:
                            logger.warning(f"   Result {i+1}: Failed - {res['error']}")
                else:
                    logger.error(f"❌ Batch processing failed: HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ Error in batch processing test: {e}")
    
    async def test_cache_management(self):
        """Test cache management and warming"""
        logger.info("🔥 Testing Cache Management")
        
        try:
            # Check cache status
            async with self.session.post(
                f"{self.base_url}/api/v1/streaming/cache/manage",
                json={"action": "status"}
            ) as response:
                if response.status == 200:
                    status = await response.json()
                    cache_status = status['cache_status']
                    
                    logger.info(f"✅ Cache Status:")
                    logger.info(f"   Total models: {cache_status['total_models']}")
                    logger.info(f"   Warmed models: {cache_status['warmed_models']}")
                    logger.info(f"   Cache hit rate: {cache_status['cache_hit_rate']:.1f}%")
                    logger.info(f"   Total cache size: {cache_status['total_cache_size_mb']:.1f} MB")
                    
                    # Test warmup
                    warmup_models = ["qwen2.5:3b", "codellama:7b"]
                    async with self.session.post(
                        f"{self.base_url}/api/v1/streaming/cache/manage",
                        json={"action": "warmup", "models": warmup_models}
                    ) as warmup_response:
                        if warmup_response.status == 200:
                            warmup_result = await warmup_response.json()
                            logger.info("🔥 Model warmup results:")
                            for result in warmup_result['results']:
                                status_icon = "✅" if result['success'] else "❌"
                                logger.info(f"   {status_icon} {result['model']}")
                        else:
                            logger.error(f"❌ Warmup failed: HTTP {warmup_response.status}")
                else:
                    logger.error(f"❌ Cache status check failed: HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ Error in cache management test: {e}")
    
    async def test_performance_metrics(self):
        """Test performance metrics collection"""
        logger.info("📊 Testing Performance Metrics")
        
        try:
            async with self.session.get(
                f"{self.base_url}/api/v1/streaming/performance/metrics"
            ) as response:
                if response.status == 200:
                    metrics = await response.json()
                    
                    logger.info("✅ Performance Metrics:")
                    
                    # Cache metrics
                    cache_status = metrics['cache_status']
                    logger.info(f"   Cache Hit Rate: {cache_status['cache_hit_rate']:.1f}%")
                    logger.info(f"   Cached Models: {cache_status['warmed_models']}/{cache_status['total_models']}")
                    
                    # System info
                    system_info = metrics['system_info']
                    logger.info(f"   GPU Available: {system_info['gpu_available']}")
                    logger.info(f"   Active Batch Processors: {system_info['active_batch_processors']}")
                    logger.info(f"   Pending Requests: {system_info['pending_batch_requests']}")
                    
                    # Performance stats
                    if metrics['performance_stats']:
                        logger.info("   Model Performance:")
                        for model, stats in metrics['performance_stats'].items():
                            logger.info(f"     {model}:")
                            logger.info(f"       Requests: {stats['total_requests']}")
                            logger.info(f"       Avg Latency: {stats['avg_latency']:.3f}s")
                            logger.info(f"       Cache Hits: {stats['cache_hits']}")
                else:
                    logger.error(f"❌ Performance metrics failed: HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ Error in performance metrics test: {e}")
    
    async def test_advanced_health_check(self):
        """Test advanced health check"""
        logger.info("🏥 Testing Advanced Health Check")
        
        try:
            async with self.session.get(
                f"{self.base_url}/api/v1/streaming/health/advanced"
            ) as response:
                if response.status == 200:
                    health = await response.json()
                    
                    status_icon = "✅" if health['status'] == "healthy" else "❌"
                    logger.info(f"{status_icon} Advanced Health Status: {health['status']}")
                    logger.info(f"   Service: {health['service_name']}")
                    logger.info(f"   Uptime: {health['uptime_seconds']:.0f}s")
                    logger.info(f"   GPU Available: {health['gpu_available']}")
                    logger.info(f"   Active Processors: {health['active_processors']}")
                    logger.info(f"   Pending Requests: {health['pending_requests']}")
                else:
                    logger.error(f"❌ Advanced health check failed: HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ Error in advanced health check: {e}")
    
    async def run_comprehensive_demo(self):
        """Run comprehensive demonstration of all advanced features"""
        logger.info("🚀 Starting Comprehensive Advanced Features Demo")
        logger.info("=" * 60)
        
        try:
            # Test 1: Advanced Health Check
            await self.test_advanced_health_check()
            await asyncio.sleep(2)
            
            # Test 2: Cache Management
            await self.test_cache_management()
            await asyncio.sleep(2)
            
            # Test 3: Performance Metrics
            await self.test_performance_metrics()
            await asyncio.sleep(2)
            
            # Test 4: Batch Processing
            await self.test_batch_processing()
            await asyncio.sleep(2)
            
            # Test 5: Streaming Chat
            await self.test_streaming_chat()
            
            logger.info("=" * 60)
            logger.info("🎉 Comprehensive demo completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")

async def main():
    """Main demonstration function"""
    logger.info("🔬 SutazAI Advanced Features Demonstration")
    logger.info("Testing enterprise-grade performance optimizations...")
    logger.info("")
    
    async with AdvancedFeaturesDemo() as demo:
        await demo.run_comprehensive_demo()

if __name__ == "__main__":
    asyncio.run(main()) 