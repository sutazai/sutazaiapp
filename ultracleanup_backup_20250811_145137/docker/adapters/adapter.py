#!/usr/bin/env python3
"""
Universal Service Adapter for SutazAI External Integration
Provides proxy, monitoring, and protocol translation capabilities
"""

import asyncio
import aiohttp
from aiohttp import web
import yaml
import os
import time
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Metrics
request_count = Counter('adapter_requests_total', 'Total requests', ['method', 'status'])
request_duration = Histogram('adapter_request_duration_seconds', 'Request duration')
active_connections = Gauge('adapter_active_connections', 'Active connections')
target_health = Gauge('adapter_target_health', 'Target service health status')

class ServiceAdapter:
    """Universal adapter for external service integration"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = self._load_config(config_path)
        self.target_host = os.getenv('TARGET_HOST', self.config.get('target_host', 'localhost'))
        self.target_port = int(os.getenv('TARGET_PORT', self.config.get('target_port', 80)))
        self.adapter_port = int(os.getenv('ADAPTER_PORT', self.config.get('adapter_port', 8080)))
        self.session = None
        self.app = web.Application()
        self._setup_routes()
        self.health_status = {'healthy': False, 'last_check': None}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("Config file not found, using defaults", path=config_path)
            return {}
    
    def _setup_routes(self):
        """Setup HTTP routes"""
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/metrics', self.metrics)
        self.app.router.add_get('/status', self.status)
        self.app.router.add_route('*', '/{path:.*}', self.proxy_request)
    
    async def start(self):
        """Start the adapter"""
        logger.info("Starting service adapter", 
                   target=f"{self.target_host}:{self.target_port}",
                   adapter_port=self.adapter_port)
        
        # Create session
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Start health check loop
        asyncio.create_task(self._health_check_loop())
        
        # Start the web server
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.adapter_port)
        await site.start()
        
        logger.info("Service adapter started successfully")
    
    async def stop(self):
        """Stop the adapter"""
        if self.session:
            await self.session.close()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def proxy_request(self, request):
        """Proxy requests to target service"""
        path = request.match_info.get('path', '')
        target_url = f"http://{self.target_host}:{self.target_port}/{path}"
        
        # Add query string
        if request.query_string:
            target_url += f"?{request.query_string}"
        
        logger.debug("Proxying request", 
                    method=request.method,
                    path=path,
                    target_url=target_url)
        
        start_time = time.time()
        active_connections.inc()
        
        try:
            # Forward headers
            headers = dict(request.headers)
            headers['X-Forwarded-For'] = request.remote
            headers['X-Forwarded-Host'] = request.host
            headers['X-Forwarded-Proto'] = request.scheme
            headers['X-SutazAI-Adapter'] = 'universal-adapter'
            
            # Make the request
            async with self.session.request(
                method=request.method,
                url=target_url,
                headers=headers,
                data=await request.read(),
                allow_redirects=False
            ) as response:
                # Copy response
                body = await response.read()
                
                # Create response
                resp = web.Response(
                    body=body,
                    status=response.status,
                    headers=response.headers
                )
                
                # Record metrics
                duration = time.time() - start_time
                request_duration.observe(duration)
                request_count.labels(method=request.method, status=response.status).inc()
                
                logger.info("Request proxied successfully",
                           method=request.method,
                           path=path,
                           status=response.status,
                           duration=duration)
                
                return resp
                
        except aiohttp.ClientError as e:
            logger.error("Proxy error", error=str(e), target_url=target_url)
            request_count.labels(method=request.method, status=502).inc()
            return web.Response(text=f"Bad Gateway: {str(e)}", status=502)
            
        except Exception as e:
            logger.error("Unexpected error", error=str(e), exc_info=True)
            request_count.labels(method=request.method, status=500).inc()
            return web.Response(text="Internal Server Error", status=500)
            
        finally:
            active_connections.dec()
    
    async def health_check(self, request):
        """Health check endpoint"""
        health_data = {
            'status': 'healthy' if self.health_status['healthy'] else 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'target': {
                'host': self.target_host,
                'port': self.target_port,
                'reachable': self.health_status['healthy'],
                'last_check': self.health_status['last_check']
            },
            'adapter': {
                'version': '1.0.0',
                'uptime': time.time() - self.start_time if hasattr(self, 'start_time') else 0
            }
        }
        
        status_code = 200 if self.health_status['healthy'] else 503
        return web.json_response(health_data, status=status_code)
    
    async def metrics(self, request):
        """Prometheus metrics endpoint"""
        metrics_data = generate_latest()
        return web.Response(text=metrics_data.decode('utf-8'),
                          content_type='text/plain; version=0.0.4')
    
    async def status(self, request):
        """Detailed status endpoint"""
        status_data = {
            'adapter': {
                'type': 'universal',
                'version': '1.0.0',
                'config': {
                    'target_host': self.target_host,
                    'target_port': self.target_port,
                    'adapter_port': self.adapter_port
                }
            },
            'target_service': {
                'healthy': self.health_status['healthy'],
                'last_check': self.health_status['last_check'],
                'type': self.config.get('service_type', 'unknown')
            },
            'statistics': {
                'active_connections': active_connections._value.get(),
                'total_requests': sum(request_count._metrics.values())
            }
        }
        
        return web.json_response(status_data)
    
    async def _health_check_loop(self):
        """Periodic health check of target service"""
        while True:
            try:
                async with self.session.get(
                    f"http://{self.target_host}:{self.target_port}/",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    self.health_status['healthy'] = response.status < 500
                    self.health_status['last_check'] = datetime.utcnow().isoformat()
                    target_health.set(1 if self.health_status['healthy'] else 0)
                    
            except Exception as e:
                logger.warning("Health check failed", error=str(e))
                self.health_status['healthy'] = False
                self.health_status['last_check'] = datetime.utcnow().isoformat()
                target_health.set(0)
            
            await asyncio.sleep(30)  # Check every 30 seconds

async def main():
    """Main entry point"""
    # Set start time
    adapter = ServiceAdapter()
    adapter.start_time = time.time()
    
    # Start adapter
    await adapter.start()
    
    # Keep running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down adapter")
    finally:
        await adapter.stop()

if __name__ == '__main__':
    asyncio.run(main())