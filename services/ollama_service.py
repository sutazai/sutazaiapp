"""
Optimized Ollama Service for SutazAI
Provides efficient LLM integration with caching, batching, and TinyLlama optimizations
"""

from typing import Dict, Any, Optional, List, Callable
import httpx
from pydantic import BaseModel, Field
import asyncio
from functools import lru_cache
import logging
import hashlib
import json
import redis.asyncio as redis
from datetime import datetime, timedelta
import backoff

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerationRequest(BaseModel):
    """Request model for text generation"""
    prompt: str
    max_tokens: int = Field(default=256, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    cache_key: Optional[str] = None
    cache_ttl: int = Field(default=300, ge=0)
    
class GenerationResponse(BaseModel):
    """Response model for text generation"""
    text: str
    tokens_used: int
    cached: bool = False
    generation_time: float
    model: str = "tinyllama"

class OllamaService:
    """
    Optimized Ollama integration service with:
    - Connection pooling
    - Response caching
    - Request batching
    - TinyLlama-specific optimizations
    - Automatic retry with exponential backoff
    """
    
    # TinyLlama optimal settings
    TINYLLAMA_CONFIG = {
        "context_window": 2048,
        "optimal_batch_size": 4,
        "max_concurrent": 4,
        "num_ctx": 2048,
        "num_batch": 128,
        "num_gpu": 0,  # CPU only
        "repeat_penalty": 1.1,
        "stop": ["\n\n", "###", "</s>"]
    }
    
    def __init__(
        self, 
        base_url: str = "http://sutazai-ollama:11434",
        redis_url: str = "redis://redis:6379/0",
        enable_cache: bool = True,
        enable_batching: bool = True
    ):
        self.base_url = base_url
        self.enable_cache = enable_cache
        self.enable_batching = enable_batching
        
        # HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30.0
            )
        )
        
        # Redis for caching (lazy initialization)
        self._redis_client = None
        self.redis_url = redis_url
        
        # Request batching
        self.batch_queue: List[tuple] = []
        self.batch_lock = asyncio.Lock()
        self.batch_event = asyncio.Event()
        
        # Metrics
        self.total_requests = 0
        self.cache_hits = 0
        self.total_tokens = 0
        
    async def _get_redis(self) -> redis.Redis:
        """Lazy Redis client initialization"""
        if self._redis_client is None and self.enable_cache:
            try:
                self._redis_client = await redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                await self._redis_client.ping()
                logger.info("Redis cache connected successfully")
            except Exception as e:
                logger.warning(f"Redis connection failed, caching disabled: {e}")
                self.enable_cache = False
                
        return self._redis_client
        
    def _optimize_prompt(self, prompt: str, max_tokens: int) -> str:
        """
        Optimize prompt for TinyLlama's capabilities
        TinyLlama works best with clear, structured prompts
        """
        # Truncate if too long (reserve space for response)
        max_prompt_length = self.TINYLLAMA_CONFIG["context_window"] - max_tokens - 100
        
        if len(prompt) > max_prompt_length:
            prompt = prompt[:max_prompt_length] + "..."
            logger.warning(f"Prompt truncated to fit context window")
            
        # Add instruction format that TinyLlama responds well to
        if not prompt.startswith("###"):
            prompt = f"### Instruction:\n{prompt}\n\n### Response:"
            
        return prompt
        
    def _create_cache_key(self, prompt: str, params: Dict[str, Any]) -> str:
        """Create deterministic cache key from prompt and parameters"""
        cache_data = {
            "prompt": prompt,
            "max_tokens": params.get("max_tokens", 256),
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 0.9)
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return f"ollama:gen:{hashlib.md5(cache_str.encode()).hexdigest()}"
        
    async def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached response if available"""
        if not self.enable_cache:
            return None
            
        try:
            redis_client = await self._get_redis()
            if redis_client:
                cached = await redis_client.get(cache_key)
                if cached:
                    self.cache_hits += 1
                    return json.loads(cached)
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            
        return None
        
    async def _cache_response(
        self, 
        cache_key: str, 
        response: Dict[str, Any], 
        ttl: int = 300
    ):
        """Cache response with TTL"""
        if not self.enable_cache:
            return
            
        try:
            redis_client = await self._get_redis()
            if redis_client:
                await redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(response)
                )
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
            
    @backoff.on_exception(
        backoff.expo,
        (httpx.HTTPError, httpx.ConnectError),
        max_tries=3,
        max_time=30
    )
    async def _call_ollama(
        self, 
        prompt: str, 
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call Ollama API with retry logic"""
        
        payload = {
            "model": "tinyllama",
            "prompt": prompt,
            "stream": False,
            "options": {
                **self.TINYLLAMA_CONFIG,
                **options,
                "num_predict": options.get("max_tokens", 256)
            }
        }
        
        response = await self.client.post(
            f"{self.base_url}/api/generate",
            json=payload
        )
        response.raise_for_status()
        
        return response.json()
        
    async def generate(
        self,
        request: GenerationRequest
    ) -> GenerationResponse:
        """
        Generate text with caching and optimization
        
        Args:
            request: Generation request parameters
            
        Returns:
            GenerationResponse with generated text and metadata
        """
        
        start_time = asyncio.get_event_loop().time()
        self.total_requests += 1
        
        # Check cache first
        cache_key = request.cache_key or self._create_cache_key(
            request.prompt,
            request.dict()
        )
        
        cached_response = await self._get_cached_response(cache_key)
        if cached_response:
            return GenerationResponse(
                text=cached_response["text"],
                tokens_used=cached_response["tokens_used"],
                cached=True,
                generation_time=0.0,
                model="tinyllama"
            )
            
        # Optimize prompt for TinyLlama
        optimized_prompt = self._optimize_prompt(request.prompt, request.max_tokens)
        
        # Prepare options
        options = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens
        }
        
        # Call Ollama
        try:
            result = await self._call_ollama(optimized_prompt, options)
            
            # Extract response
            generated_text = result.get("response", "")
            tokens_used = result.get("eval_count", 0) + result.get("prompt_eval_count", 0)
            
            self.total_tokens += tokens_used
            
            # Prepare response
            response_data = {
                "text": generated_text,
                "tokens_used": tokens_used
            }
            
            # Cache response
            await self._cache_response(cache_key, response_data, request.cache_ttl)
            
            generation_time = asyncio.get_event_loop().time() - start_time
            
            return GenerationResponse(
                text=generated_text,
                tokens_used=tokens_used,
                cached=False,
                generation_time=generation_time,
                model="tinyllama"
            )
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
            
    async def generate_batch(
        self,
        requests: List[GenerationRequest]
    ) -> List[GenerationResponse]:
        """
        Generate text for multiple requests efficiently
        
        Args:
            requests: List of generation requests
            
        Returns:
            List of generation responses
        """
        
        # Process requests in parallel with concurrency limit
        semaphore = asyncio.Semaphore(self.TINYLLAMA_CONFIG["max_concurrent"])
        
        async def process_with_limit(req):
            async with semaphore:
                return await self.generate(req)
                
        tasks = [process_with_limit(req) for req in requests]
        return await asyncio.gather(*tasks)
        
    async def analyze_text(
        self,
        text: str,
        task: str = "summarize",
        max_tokens: int = 256
    ) -> str:
        """
        Analyze text for specific tasks
        
        Args:
            text: Text to analyze
            task: Analysis task (summarize, classify, extract, etc.)
            max_tokens: Maximum tokens for response
            
        Returns:
            Analysis result
        """
        
        task_prompts = {
            "summarize": "Summarize the following text in 2-3 sentences:\n\n{text}",
            "classify": "Classify the following text into a category:\n\n{text}",
            "extract": "Extract key information from the following text:\n\n{text}",
            "sentiment": "Analyze the sentiment of the following text (positive/negative/neutral):\n\n{text}",
            "keywords": "Extract 5 keywords from the following text:\n\n{text}"
        }
        
        prompt_template = task_prompts.get(task, task_prompts["summarize"])
        prompt = prompt_template.format(text=text[:1500])  # Limit input text
        
        request = GenerationRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.3,  # Lower temperature for analysis tasks
            cache_ttl=600  # Longer cache for analysis
        )
        
        response = await self.generate(request)
        return response.text
        
    async def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        
        cache_hit_rate = (
            self.cache_hits / self.total_requests * 100
            if self.total_requests > 0
            else 0
        )
        
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": f"{cache_hit_rate:.2f}%",
            "total_tokens_used": self.total_tokens,
            "average_tokens_per_request": (
                self.total_tokens / self.total_requests
                if self.total_requests > 0
                else 0
            ),
            "cache_enabled": self.enable_cache,
            "batching_enabled": self.enable_batching,
            "model": "tinyllama",
            "context_window": self.TINYLLAMA_CONFIG["context_window"]
        }
        
    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        
        health = {
            "service": "ollama_service",
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        # Check Ollama connection
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            
            health["checks"]["ollama"] = {
                "status": "healthy",
                "models": [m.get("name") for m in models]
            }
        except Exception as e:
            health["status"] = "degraded"
            health["checks"]["ollama"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            
        # Check Redis connection
        if self.enable_cache:
            try:
                redis_client = await self._get_redis()
                if redis_client:
                    await redis_client.ping()
                    health["checks"]["redis"] = {"status": "healthy"}
                else:
                    health["checks"]["redis"] = {"status": "unavailable"}
            except Exception as e:
                health["checks"]["redis"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                
        # Add metrics
        health["metrics"] = await self.get_metrics()
        
        return health
        
    async def close(self):
        """Close connections gracefully"""
        await self.client.aclose()
        if self._redis_client:
            await self._redis_client.close()

# Example usage for testing
async def example_usage():
    """Example usage of OllamaService"""
    
    service = OllamaService()
    
    try:
        # Single generation
        request = GenerationRequest(
            prompt="What is Docker?",
            max_tokens=100,
            temperature=0.7
        )
        
        response = await service.generate(request)
        print(f"Generated: {response.text}")
        print(f"Tokens used: {response.tokens_used}")
        print(f"Cached: {response.cached}")
        
        # Batch generation
        requests = [
            GenerationRequest(prompt="What is Python?", max_tokens=50),
            GenerationRequest(prompt="What is AI?", max_tokens=50),
            GenerationRequest(prompt="What is Docker?", max_tokens=50)
        ]
        
        responses = await service.generate_batch(requests)
        for i, resp in enumerate(responses):
            print(f"Request {i+1}: {resp.text[:100]}...")
            
        # Text analysis
        sample_text = "Docker is a platform for developing, shipping, and running applications in containers."
        summary = await service.analyze_text(sample_text, task="summarize")
        print(f"Summary: {summary}")
        
        # Get metrics
        metrics = await service.get_metrics()
        print(f"Metrics: {json.dumps(metrics, indent=2)}")
        
        # Health check
        health = await service.health_check()
        print(f"Health: {json.dumps(health, indent=2)}")
        
    finally:
        await service.close()

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())