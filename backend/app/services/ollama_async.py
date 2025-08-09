"""
Non-blocking Async Ollama Service
Implements proper async patterns for LLM operations without blocking the event loop
"""

import asyncio
import json
import logging
import hashlib
from typing import Dict, Any, Optional, List, AsyncGenerator
from datetime import datetime, timedelta
import pickle
from concurrent.futures import ThreadPoolExecutor

from app.core.connection_pool import get_pool_manager

logger = logging.getLogger(__name__)


class OllamaAsyncService:
    """High-performance async wrapper for Ollama operations"""
    
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._generation_cache = {}
        self._cache_ttl = 3600  # 1 hour cache
        self._max_cache_size = 1000
        self._request_queue = asyncio.Queue(maxsize=100)
        self._processing = False
        self._stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'avg_response_time': 0
        }
        
    async def initialize(self):
        """Initialize the service and start background workers"""
        self._processing = True
        # Start background processing workers
        for i in range(3):  # 3 concurrent workers
            asyncio.create_task(self._process_queue_worker(i))
        logger.info("Ollama async service initialized with 3 workers")
        
    def _get_cache_key(self, model: str, prompt: str, options: Dict[str, Any]) -> str:
        """Generate cache key for prompts"""
        key_data = f"{model}:{prompt}:{json.dumps(options, sort_keys=True)}"
        return hashlib.sha256(key_data.encode()).hexdigest()
        
    def _is_cache_valid(self, cached_entry: Dict[str, Any]) -> bool:
        """Check if cached entry is still valid"""
        if 'timestamp' not in cached_entry:
            return False
        age = (datetime.now() - cached_entry['timestamp']).total_seconds()
        return age < self._cache_ttl
        
    async def generate(
        self,
        prompt: str,
        model: str = "tinyllama",
        options: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Generate text using Ollama with caching and non-blocking execution"""
        
        self._stats['total_requests'] += 1
        start_time = asyncio.get_event_loop().time()
        
        # Default options for performance
        if options is None:
            options = {
                'num_predict': 150,
                'temperature': 0.7,
                'top_k': 40,
                'top_p': 0.9,
                'num_ctx': 2048  # Reduced context for faster processing
            }
            
        # Check cache first
        if use_cache and not stream:
            cache_key = self._get_cache_key(model, prompt, options)
            if cache_key in self._generation_cache:
                cached = self._generation_cache[cache_key]
                if self._is_cache_valid(cached):
                    self._stats['cache_hits'] += 1
                    logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                    return cached['response']
                    
        self._stats['cache_misses'] += 1
        
        try:
            # Get connection pool manager
            pool_manager = await get_pool_manager()
            
            # Use the pooled Ollama client
            async with pool_manager.get_http_client('ollama') as client:
                # Prepare request
                request_data = {
                    'model': model,
                    'prompt': prompt,
                    'stream': stream,
                    'options': options
                }
                
                # Make async request
                response = await client.post(
                    '/api/generate',
                    json=request_data
                )
                
                if response.status_code != 200:
                    raise Exception(f"Ollama error: {response.status_code} - {response.text}")
                    
                result = response.json()
                
                # Cache the result if not streaming
                if use_cache and not stream:
                    self._cache_result(cache_key, result)
                    
                # Update stats
                elapsed = asyncio.get_event_loop().time() - start_time
                self._update_avg_response_time(elapsed)
                
                return result
                
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Ollama generation error: {e}")
            
            # Return fallback response
            return {
                'response': f"Error generating response: {str(e)}",
                'model': model,
                'done': True,
                'error': str(e)
            }
            
    async def generate_streaming(
        self,
        prompt: str,
        model: str = "tinyllama",
        options: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Stream generation for real-time responses"""
        
        if options is None:
            options = {
                'num_predict': 150,
                'temperature': 0.7
            }
            
        try:
            pool_manager = await get_pool_manager()
            
            async with pool_manager.get_http_client('ollama') as client:
                request_data = {
                    'model': model,
                    'prompt': prompt,
                    'stream': True,
                    'options': options
                }
                
                # Stream the response
                async with client.stream('POST', '/api/generate', json=request_data) as response:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if 'response' in data:
                                    yield data['response']
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            yield f"Error: {str(e)}"
            
    async def batch_generate(
        self,
        prompts: List[str],
        model: str = "tinyllama",
        options: Optional[Dict[str, Any]] = None,
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """Process multiple prompts concurrently with rate limiting"""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(prompt: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.generate(prompt, model, options)
                
        # Process all prompts concurrently
        tasks = [process_single(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch generation error for prompt {i}: {result}")
                processed_results.append({
                    'response': f"Error: {str(result)}",
                    'error': True
                })
            else:
                processed_results.append(result)
                
        return processed_results
        
    async def generate_with_retry(
        self,
        prompt: str,
        model: str = "tinyllama",
        options: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Dict[str, Any]:
        """Generate with automatic retry on failure"""
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return await self.generate(prompt, model, options, use_cache=attempt == 0)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    
        # All retries failed
        logger.error(f"All retries failed for prompt: {prompt[:50]}... Error: {last_error}")
        return {
            'response': f"Failed after {max_retries} attempts: {str(last_error)}",
            'error': True
        }
        
    def _cache_result(self, key: str, result: Dict[str, Any]):
        """Cache generation result"""
        # Implement LRU eviction if cache is full
        if len(self._generation_cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_key = min(
                self._generation_cache.keys(),
                key=lambda k: self._generation_cache[k].get('timestamp', datetime.min)
            )
            del self._generation_cache[oldest_key]
            
        self._generation_cache[key] = {
            'response': result,
            'timestamp': datetime.now()
        }
        
    def _update_avg_response_time(self, elapsed: float):
        """Update average response time"""
        current_avg = self._stats['avg_response_time']
        total_requests = self._stats['total_requests']
        self._stats['avg_response_time'] = (
            (current_avg * (total_requests - 1) + elapsed) / total_requests
        )
        
    async def _process_queue_worker(self, worker_id: int):
        """Background worker for processing queued requests"""
        logger.info(f"Queue worker {worker_id} started")
        
        while self._processing:
            try:
                # Get request from queue
                request = await asyncio.wait_for(
                    self._request_queue.get(),
                    timeout=1.0
                )
                
                # Process the request
                result = await self.generate(
                    request['prompt'],
                    request.get('model', 'tinyllama'),
                    request.get('options')
                )
                
                # Store result if callback provided
                if 'callback' in request:
                    await request['callback'](result)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Queue worker {worker_id} error: {e}")
                
    async def queue_generation(
        self,
        prompt: str,
        model: str = "tinyllama",
        options: Optional[Dict[str, Any]] = None,
        callback: Optional[Any] = None
    ) -> bool:
        """Queue a generation request for background processing"""
        
        try:
            request = {
                'prompt': prompt,
                'model': model,
                'options': options,
                'callback': callback
            }
            
            # Try to add to queue (non-blocking)
            await asyncio.wait_for(
                self._request_queue.put(request),
                timeout=0.1
            )
            return True
            
        except asyncio.TimeoutError:
            logger.warning("Generation queue is full")
            return False
            
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        cache_hit_rate = 0
        if self._stats['total_requests'] > 0:
            cache_hit_rate = self._stats['cache_hits'] / self._stats['total_requests']
            
        return {
            **self._stats,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self._generation_cache),
            'queue_size': self._request_queue.qsize()
        }
        
    def clear_cache(self):
        """Clear the generation cache"""
        self._generation_cache.clear()
        logger.info("Ollama generation cache cleared")
        
    async def warmup(self, num_requests: int = 5):
        """Warmup the service with test requests"""
        logger.info(f"Warming up Ollama service with {num_requests} requests")
        
        test_prompts = [
            "Hello",
            "Test",
            "System check",
            "What is 1+1?",
            "Ready"
        ]
        
        for i in range(min(num_requests, len(test_prompts))):
            await self.generate(test_prompts[i], use_cache=False)
            
        logger.info("Ollama service warmup complete")
        
    async def shutdown(self):
        """Gracefully shutdown the service"""
        self._processing = False
        self._executor.shutdown(wait=True)
        logger.info("Ollama async service shutdown complete")


# Global service instance
_ollama_service: Optional[OllamaAsyncService] = None


async def get_ollama_service() -> OllamaAsyncService:
    """Get or create the global Ollama service"""
    global _ollama_service
    
    if _ollama_service is None:
        _ollama_service = OllamaAsyncService()
        await _ollama_service.initialize()
        # Warmup with a few requests
        await _ollama_service.warmup(3)
        
    return _ollama_service