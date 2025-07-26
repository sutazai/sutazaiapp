#!/usr/bin/env python3
"""
Backend timeout and performance fixes for SutazAI
Addresses the HTTPConnectionPool timeout issues
"""

import asyncio
import aiohttp
import time
import logging
from functools import wraps
import requests
from typing import Dict, Any, Optional
import signal
import threading

# Import our logging system
from enhanced_logging_system import info, debug, warning, error, log_exception, log_api_calls

class TimeoutHandler:
    """Handles timeouts and connection issues"""
    
    @staticmethod
    def timeout_with_fallback(timeout_seconds: int = 30):
        """Decorator to add timeout handling with fallback"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
                except asyncio.TimeoutError:
                    error(f"Timeout in {func.__name__} after {timeout_seconds}s", category="api")
                    return {"error": f"Operation timed out after {timeout_seconds} seconds", "timeout": True}
                except Exception as e:
                    log_exception(e, context=f"Function {func.__name__}", category="api")
                    return {"error": str(e), "exception": True}
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
                
                # Set timeout signal
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
                
                try:
                    result = func(*args, **kwargs)
                    signal.alarm(0)  # Cancel alarm
                    return result
                except TimeoutError as e:
                    error(f"Timeout in {func.__name__}: {str(e)}", category="api")
                    return {"error": str(e), "timeout": True}
                except Exception as e:
                    signal.alarm(0)  # Cancel alarm
                    log_exception(e, context=f"Function {func.__name__}", category="api")
                    return {"error": str(e), "exception": True}
                finally:
                    signal.signal(signal.SIGALRM, old_handler)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator

class EnhancedAPIClient:
    """Enhanced API client with connection pooling and retry logic"""
    
    def __init__(self):
        self.session = None
        self.connection_pool_size = 10
        self.max_retries = 3
        self.base_timeout = 30
        self.ollama_url = "http://localhost:11434"
        
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=self.connection_pool_size,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(
            total=90,  # Total timeout increased to 90s
            connect=10,  # Connection timeout
            sock_read=60   # Socket read timeout
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": "SutazAI-Client/1.0"}
        )
        
        info("Enhanced API client session started", category="api")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            info("Enhanced API client session closed", category="api")
    
    @timeout_with_fallback(90)  # 90 second timeout for chat operations
    async def chat_completion(self, message: str, model: str = "llama3.2:1b") -> Dict[str, Any]:
        """Chat completion with enhanced error handling"""
        
        debug(f"Starting chat completion with {model}", category="api", model=model)
        
        payload = {
            "model": model,
            "prompt": message,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000,
                "stop": []
            }
        }
        
        url = f"{self.ollama_url}/api/generate"
        
        for attempt in range(self.max_retries):
            try:
                debug(f"Chat attempt {attempt + 1}/{self.max_retries}", category="api")
                
                async with self.session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        info(
                            f"Chat completion successful with {model}",
                            category="api",
                            model=model,
                            attempt=attempt + 1
                        )
                        
                        return {
                            "success": True,
                            "response": result.get("response", ""),
                            "model": model,
                            "done": result.get("done", True),
                            "context": result.get("context", []),
                            "total_duration": result.get("total_duration", 0),
                            "load_duration": result.get("load_duration", 0),
                            "prompt_eval_count": result.get("prompt_eval_count", 0),
                            "eval_count": result.get("eval_count", 0)
                        }
                    else:
                        error_text = await response.text()
                        error(
                            f"Chat API error: {response.status} - {error_text}",
                            category="api",
                            status_code=response.status
                        )
                        
                        if attempt == self.max_retries - 1:
                            return {
                                "success": False,
                                "error": f"API error: {response.status} - {error_text}",
                                "status_code": response.status
                            }
                
            except asyncio.TimeoutError:
                warning(f"Chat timeout on attempt {attempt + 1}", category="api")
                if attempt == self.max_retries - 1:
                    return {
                        "success": False,
                        "error": "Request timed out after multiple attempts",
                        "timeout": True
                    }
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
                
            except Exception as e:
                log_exception(e, context=f"Chat attempt {attempt + 1}", category="api")
                if attempt == self.max_retries - 1:
                    return {
                        "success": False,
                        "error": f"Connection error: {str(e)}",
                        "exception": True
                    }
                
                await asyncio.sleep(1)
        
        return {
            "success": False,
            "error": "All retry attempts failed",
            "max_retries_exceeded": True
        }
    
    @timeout_with_fallback(30)
    async def health_check(self) -> Dict[str, Any]:
        """Health check with timeout"""
        
        try:
            async with self.session.get(f"{self.ollama_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("models", [])
                    
                    info(f"Health check successful - {len(models)} models available", category="api")
                    
                    return {
                        "success": True,
                        "status": "healthy",
                        "models_count": len(models),
                        "models": [model.get("name", "unknown") for model in models[:5]]  # First 5 models
                    }
                else:
                    error(f"Health check failed: {response.status}", category="api")
                    return {
                        "success": False,
                        "error": f"Health check failed: {response.status}",
                        "status_code": response.status
                    }
                    
        except Exception as e:
            log_exception(e, context="Health check", category="api")
            return {
                "success": False,
                "error": f"Health check exception: {str(e)}",
                "exception": True
            }

# Create a global client instance
api_client = None

async def get_api_client() -> EnhancedAPIClient:
    """Get or create API client instance"""
    global api_client
    if api_client is None:
        api_client = EnhancedAPIClient()
    return api_client

# Sync wrapper functions for non-async code
def sync_chat_completion(message: str, model: str = "llama3.2:1b") -> Dict[str, Any]:
    """Synchronous wrapper for chat completion"""
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    async def _async_chat():
        async with EnhancedAPIClient() as client:
            return await client.chat_completion(message, model)
    
    try:
        return loop.run_until_complete(_async_chat())
    except Exception as e:
        log_exception(e, context="Sync chat completion", category="api")
        return {
            "success": False,
            "error": f"Async execution error: {str(e)}",
            "exception": True
        }

def sync_health_check() -> Dict[str, Any]:
    """Synchronous wrapper for health check"""
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    async def _async_health():
        async with EnhancedAPIClient() as client:
            return await client.health_check()
    
    try:
        return loop.run_until_complete(_async_health())
    except Exception as e:
        log_exception(e, context="Sync health check", category="api")
        return {
            "success": False,
            "error": f"Health check error: {str(e)}",
            "exception": True
        }

# Connection pool warming
def warm_connection_pool():
    """Warm up the connection pool"""
    
    def _warm_pool():
        try:
            info("Warming up connection pool", category="system")
            
            # Make a simple health check to warm the pool
            result = sync_health_check()
            
            if result["success"]:
                info("Connection pool warmed successfully", category="system")
            else:
                warning(f"Connection pool warming failed: {result.get('error')}", category="system")
                
        except Exception as e:
            log_exception(e, context="Connection pool warming", category="system")
    
    # Run in background thread
    threading.Thread(target=_warm_pool, daemon=True).start()

# Initialize on import
warm_connection_pool()

info("Backend timeout fixes initialized", category="system")