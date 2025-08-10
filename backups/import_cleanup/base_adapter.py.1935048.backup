"""
Base adapter class for all external AI service integrations
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import aiohttp
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ServiceAdapter(ABC):
    """Base class for all service adapters"""
    
    def __init__(self, service_name: str, config: Dict[str, Any]):
        self.service_name = service_name
        self.config = config
        self.base_url = config.get('base_url', '')
        self.api_key = config.get('api_key', '')
        self.health_check_endpoint = config.get('health_check_endpoint', '/health')
        self.timeout = config.get('timeout', 30)
        self.retry_count = config.get('retry_count', 3)
        self.session: Optional[aiohttp.ClientSession] = None
        self._is_healthy = False
        self._last_health_check = None
        
    async def __aenter__(self):
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
        
    async def connect(self):
        """Initialize connection to the service"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        logger.info(f"Connected to {self.service_name}")
        
    async def disconnect(self):
        """Close connection to the service"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info(f"Disconnected from {self.service_name}")
        
    async def health_check(self) -> Dict[str, Any]:
        """Check if the service is healthy"""
        try:
            if self.base_url and self.health_check_endpoint:
                url = f"{self.base_url}{self.health_check_endpoint}"
                async with self.session.get(url) as response:
                    self._is_healthy = response.status == 200
                    self._last_health_check = datetime.utcnow()
                    return {
                        'service': self.service_name,
                        'healthy': self._is_healthy,
                        'status_code': response.status,
                        'timestamp': self._last_health_check.isoformat()
                    }
            else:
                # For services without HTTP endpoints
                self._is_healthy = await self._custom_health_check()
                self._last_health_check = datetime.utcnow()
                return {
                    'service': self.service_name,
                    'healthy': self._is_healthy,
                    'timestamp': self._last_health_check.isoformat()
                }
        except Exception as e:
            logger.error(f"Health check failed for {self.service_name}: {str(e)}")
            self._is_healthy = False
            return {
                'service': self.service_name,
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
            
    async def _custom_health_check(self) -> bool:
        """Override for custom health check logic"""
        return True
        
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""
        url = f"{self.base_url}{endpoint}"
        headers = kwargs.pop('headers', {})
        
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"
            
        for attempt in range(self.retry_count):
            try:
                async with self.session.request(method, url, headers=headers, **kwargs) as response:
                    if response.status >= 200 and response.status < 300:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.warning(f"Request failed (attempt {attempt + 1}): {error_text}")
                        
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"Request error (attempt {attempt + 1}): {str(e)}")
                
            if attempt < self.retry_count - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
        raise Exception(f"Failed to complete request after {self.retry_count} attempts")
        
    @abstractmethod
    async def initialize(self):
        """Initialize the service (create indices, load models, etc.)"""
        pass
        
    @abstractmethod
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get service capabilities and configuration"""
        pass
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        return {
            'service': self.service_name,
            'healthy': self._is_healthy,
            'last_health_check': self._last_health_check.isoformat() if self._last_health_check else None,
            'config': {
                'base_url': self.base_url,
                'timeout': self.timeout,
                'retry_count': self.retry_count
            }
        }