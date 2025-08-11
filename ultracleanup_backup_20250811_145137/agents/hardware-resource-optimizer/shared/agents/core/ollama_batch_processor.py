#!/usr/bin/env python3
"""
Purpose: Batch processing and caching system for Ollama optimization
Usage: Improves throughput by batching requests and caching responses
Requirements: asyncio, redis, httpx, ollama
"""

import asyncio
import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from datetime import datetime, timedelta
import redis
import httpx
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logger = logging.getLogger('ollama-batch-processor')

@dataclass
class BatchRequest:
    """Individual request in a batch"""
    id: str
    model: str
    prompt: str
    options: Dict[str, Any]
    timestamp: datetime
    priority: int = 1
    callback: Optional[callable] = None

@dataclass
class BatchResult:
    """Result of a batch request"""
    request_id: str
    response: str
    processing_time: float
    from_cache: bool
    error: Optional[str] = None

class OllamaBatchProcessor:
    """Optimized batch processing system for Ollama requests"""
    
    def __init__(self, 
                 redis_host: str = "localhost",
                 redis_port: int = 10001,
                 ollama_url: str = "http://localhost:10104",
                 max_batch_size: int = 16,
                 batch_timeout: float = 0.1,
                 cache_ttl: int = 3600):
        
        self.ollama_url = ollama_url
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.cache_ttl = cache_ttl
        
        # Redis for caching
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=1)
            self.redis_client.ping()
            self.cache_enabled = True
            logger.info("Redis cache enabled")
        except Exception as e:
            logger.warning(f"Redis cache disabled: {e}")
            self.cache_enabled = False
        
        # HTTP client for Ollama
        self.client = httpx.AsyncClient(timeout=120.0)
        
        # Batch processing queues
        self.request_queues: Dict[str, deque] = defaultdict(deque)
        self.processing_batches: Dict[str, List[BatchRequest]] = {}
        self.result_cache: Dict[str, BatchResult] = {}
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'cached_responses': 0,
            'batch_processed': 0,
            'average_batch_size': 0.0,
            'cache_hit_rate': 0.0
        }
        
        # Background processing
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.running = False
        
    async def start(self):
        """Start the batch processing system"""
        logger.info("Starting Ollama Batch Processor")
        self.running = True
        
        # Start batch processors for each model
        models = ['tinyllama', 'tinyllama.2:3b', 'tinyllama']
        for model in models:
            self.processing_tasks[model] = asyncio.create_task(self._process_model_queue(model))
        
        logger.info(f"Started batch processors for {len(models)} models")
    
    async def stop(self):
        """Stop the batch processing system"""
        logger.info("Stopping Ollama Batch Processor")
        self.running = False
        
        # Cancel all processing tasks
        for task in self.processing_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks.values(), return_exceptions=True)
        
        await self.client.aclose()
        logger.info("Batch processor stopped")
    
    def _generate_cache_key(self, model: str, prompt: str, options: Dict[str, Any]) -> str:
        """Generate cache key for request"""
        cache_data = {
            'model': model,
            'prompt': prompt,
            'options': options
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()[:32]
    
    async def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if available"""
        if not self.cache_enabled:
            return None
        
        try:
            cached = self.redis_client.get(f"ollama_cache:{cache_key}")
            if cached:
                self.stats['cached_responses'] += 1
                return cached.decode('utf-8')
        except Exception as e:
            logger.error(f"Cache read error: {e}")
        
        return None
    
    async def _cache_response(self, cache_key: str, response: str):
        """Cache response for future use"""
        if not self.cache_enabled:
            return
        
        try:
            self.redis_client.setex(
                f"ollama_cache:{cache_key}",
                self.cache_ttl,
                response
            )
        except Exception as e:
            logger.error(f"Cache write error: {e}")
    
    async def submit_request(self, 
                           model: str,
                           prompt: str,
                           options: Optional[Dict[str, Any]] = None,
                           priority: int = 1) -> str:
        """Submit a request for batch processing"""
        
        if options is None:
            options = {}
        
        # Generate request ID and cache key
        request_id = f"{int(time.time() * 1000)}_{len(self.request_queues[model])}"
        cache_key = self._generate_cache_key(model, prompt, options)
        
        # Check cache first
        cached_response = await self._get_cached_response(cache_key)
        if cached_response:
            logger.debug(f"Cache hit for request {request_id}")
            return cached_response
        
        # Create batch request
        batch_request = BatchRequest(
            id=request_id,
            model=model,
            prompt=prompt,
            options=options,
            timestamp=datetime.now(),
            priority=priority
        )
        
        # Add to appropriate queue
        self.request_queues[model].append(batch_request)
        self.stats['total_requests'] += 1
        
        # Wait for result
        return await self._wait_for_result(request_id)
    
    async def _wait_for_result(self, request_id: str, timeout: float = 120.0) -> str:
        """Wait for batch processing result"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if request_id in self.result_cache:
                result = self.result_cache.pop(request_id)
                if result.error:
                    raise Exception(result.error)
                return result.response
            
            await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
        
        raise TimeoutError(f"Request {request_id} timed out")
    
    async def _process_model_queue(self, model: str):
        """Process requests for a specific model"""
        logger.info(f"Started batch processor for model: {model}")
        
        while self.running:
            try:
                # Wait for requests or timeout
                await asyncio.sleep(self.batch_timeout)
                
                # Collect batch
                batch = self._collect_batch(model)
                
                if batch:
                    await self._process_batch(model, batch)
            
            except Exception as e:
                logger.error(f"Error processing {model} queue: {e}")
                await asyncio.sleep(1)  # Prevent rapid error loops
    
    def _collect_batch(self, model: str) -> List[BatchRequest]:
        """Collect requests for batch processing"""
        queue = self.request_queues[model]
        batch = []
        
        # Collect up to max_batch_size requests
        while queue and len(batch) < self.max_batch_size:
            batch.append(queue.popleft())
        
        # Sort by priority (higher priority first)
        if batch:
            batch.sort(key=lambda r: r.priority, reverse=True)
        
        return batch
    
    async def _process_batch(self, model: str, batch: List[BatchRequest]):
        """Process a batch of requests"""
        start_time = time.time()
        logger.debug(f"Processing batch of {len(batch)} requests for {model}")
        
        # Group requests by similar options for better batching
        option_groups = defaultdict(list)
        for request in batch:
            options_key = json.dumps(request.options, sort_keys=True)
            option_groups[options_key].append(request)
        
        # Process each group
        tasks = []
        for options_key, group_requests in option_groups.items():
            task = asyncio.create_task(self._process_request_group(model, group_requests))
            tasks.append(task)
        
        # Wait for all groups to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        processing_time = time.time() - start_time
        self.stats['batch_processed'] += 1
        
        # Update average batch size
        total_batches = self.stats['batch_processed']
        self.stats['average_batch_size'] = (
            (self.stats['average_batch_size'] * (total_batches - 1) + len(batch)) / total_batches
        )
        
        logger.debug(f"Batch processed in {processing_time:.2f}s")
    
    async def _process_request_group(self, model: str, requests: List[BatchRequest]):
        """Process a group of requests with similar options"""
        # For now, process each request individually
        # Future optimization: true batch processing at Ollama level
        
        for request in requests:
            try:
                await self._process_single_request(request)
            except Exception as e:
                logger.error(f"Error processing request {request.id}: {e}")
                # Store error result
                self.result_cache[request.id] = BatchResult(
                    request_id=request.id,
                    response="",
                    processing_time=0.0,
                    from_cache=False,
                    error=str(e)
                )
    
    async def _process_single_request(self, request: BatchRequest):
        """Process a single request"""
        start_time = time.time()
        
        try:
            # Make request to Ollama
            payload = {
                "model": request.model,
                "prompt": request.prompt,
                "stream": False,
                "options": request.options
            }
            
            response = await self.client.post(
                f"{self.ollama_url}/api/generate",
                json=payload
            )
            
            if response.status_code == 200:
                result_data = response.json()
                response_text = result_data.get('response', '')
                
                processing_time = time.time() - start_time
                
                # Cache the response
                cache_key = self._generate_cache_key(request.model, request.prompt, request.options)
                await self._cache_response(cache_key, response_text)
                
                # Store result
                self.result_cache[request.id] = BatchResult(
                    request_id=request.id,
                    response=response_text,
                    processing_time=processing_time,
                    from_cache=False
                )
                
                logger.debug(f"Request {request.id} completed in {processing_time:.2f}s")
            
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
        
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Request {request.id} failed: {e}")
            
            self.result_cache[request.id] = BatchResult(
                request_id=request.id,
                response="",
                processing_time=processing_time,
                from_cache=False,
                error=str(e)
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        # Calculate cache hit rate
        if self.stats['total_requests'] > 0:
            self.stats['cache_hit_rate'] = self.stats['cached_responses'] / self.stats['total_requests']
        
        return {
            **self.stats,
            'queue_lengths': {model: len(queue) for model, queue in self.request_queues.items()},
            'active_batches': len(self.processing_batches),
            'cached_results': len(self.result_cache)
        }
    
    async def warm_cache(self, common_prompts: List[Tuple[str, str]]):
        """Warm up cache with common prompts"""
        logger.info(f"Warming cache with {len(common_prompts)} common prompts")
        
        tasks = []
        for model, prompt in common_prompts:
            task = asyncio.create_task(self.submit_request(model, prompt))
            tasks.append(task)
        
        # Execute all cache warming requests
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Cache warming completed")
    
    async def clear_cache(self):
        """Clear all cached responses"""
        if self.cache_enabled:
            try:
                # Get all cache keys
                keys = self.redis_client.keys("ollama_cache:*")
                if keys:
                    self.redis_client.delete(*keys)
                    logger.info(f"Cleared {len(keys)} cached responses")
                else:
                    logger.info("No cached responses to clear")
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
        
        # Clear local result cache
        self.result_cache.clear()
        logger.info("Local result cache cleared")

class OllamaCacheManager:
    """Advanced cache management for Ollama responses"""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 10001):
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=2)
            self.redis_client.ping()
            self.enabled = True
            logger.info("Cache manager initialized")
        except Exception as e:
            logger.warning(f"Cache manager disabled: {e}")
            self.enabled = False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.enabled:
            return {'enabled': False}
        
        try:
            info = self.redis_client.info('memory')
            keyspace = self.redis_client.info('keyspace')
            
            return {
                'enabled': True,
                'memory_used': info.get('used_memory_human', 'Unknown'),
                'total_keys': sum(db.get('keys', 0) for db in keyspace.values()),
                'hit_rate': self._calculate_hit_rate(),
                'cache_keys': len(self.redis_client.keys("ollama_cache:*"))
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'enabled': True, 'error': str(e)}
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        try:
            stats = self.redis_client.info('stats')
            hits = stats.get('keyspace_hits', 0)
            misses = stats.get('keyspace_misses', 0)
            
            if hits + misses > 0:
                return hits / (hits + misses)
        except Exception:
            pass
        
        return 0.0
    
    async def optimize_cache(self):
        """Optimize cache by removing least used entries"""
        if not self.enabled:
            return
        
        try:
            # Get all cache keys
            keys = self.redis_client.keys("ollama_cache:*")
            
            if len(keys) > 10000:  # If too many keys, clean up
                # Remove keys with shortest TTL (least recently used)
                key_ttls = [(key, self.redis_client.ttl(key)) for key in keys]
                key_ttls.sort(key=lambda x: x[1])  # Sort by TTL
                
                # Remove bottom 20%
                keys_to_remove = [key for key, _ in key_ttls[:len(keys_ttls)//5]]
                if keys_to_remove:
                    self.redis_client.delete(*keys_to_remove)
                    logger.info(f"Removed {len(keys_to_remove)} least used cache entries")
        
        except Exception as e:
            logger.error(f"Error optimizing cache: {e}")

# CLI interface for batch processor
async def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ollama Batch Processor')
    parser.add_argument('--start', action='store_true', help='Start batch processor')
    parser.add_argument('--test', action='store_true', help='Run test requests')
    parser.add_argument('--warm-cache', action='store_true', help='Warm up cache')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cache')
    
    args = parser.parse_args()
    
    processor = OllamaBatchProcessor()
    
    try:
        if args.start:
            await processor.start()
            logger.info("Batch processor started. Press Ctrl+C to stop.")
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                await processor.stop()
        
        elif args.test:
            await processor.start()
            
            # Run test requests
            test_prompts = [
                "What is artificial intelligence?",
                "Explain machine learning in simple terms.",
                "What are the benefits of using AI?"
            ]
            
            logger.info("Running test requests...")
            start_time = time.time()
            
            tasks = []
            for i, prompt in enumerate(test_prompts):
                task = asyncio.create_task(processor.submit_request('tinyllama', prompt))
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            logger.info(f"Processed {len(responses)} requests in {total_time:.2f}s")
            
            await processor.stop()
        
        elif args.warm_cache:
            await processor.start()
            
            common_prompts = [
                ('tinyllama', 'Hello, how are you?'),
                ('tinyllama', 'What is the weather like?'),
                ('tinyllama', 'Explain artificial intelligence.'),
                ('tinyllama', 'What is machine learning?'),
                ('tinyllama', 'How does deep learning work?')
            ]
            
            await processor.warm_cache(common_prompts)
            await processor.stop()
        
        elif args.stats:
            stats = processor.get_stats()
            print(json.dumps(stats, indent=2))
            
            cache_manager = OllamaCacheManager()
            cache_stats = cache_manager.get_cache_stats()
            print("\nCache Statistics:")
            print(json.dumps(cache_stats, indent=2))
        
        elif args.clear_cache:
            await processor.clear_cache()
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import sys
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())