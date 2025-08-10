#!/usr/bin/env python3
"""
Circuit Breaker Service for SutazAI System
Implements circuit breaker pattern to prevent cascading failures
"""

import os
import time
import json
import logging
import asyncio
import aiohttp
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from aiohttp import web
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/circuit_breaker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerMetrics:
    name: str
    state: CircuitBreakerState
    failure_count: int
    success_count: int
    last_failure_time: Optional[datetime]
    last_success_time: Optional[datetime]
    total_requests: int
    failed_requests: int
    success_rate: float
    avg_response_time: float

class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: int = 60, 
                 half_open_max_calls: int = 3):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.total_requests = 0
        self.failed_requests = 0
        self.response_times = []
        self.half_open_calls = 0
        
        self._lock = threading.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function call through the circuit breaker"""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                else:
                    logger.warning(f"Circuit breaker {self.name} is OPEN, rejecting call")
                    raise CircuitBreakerOpenException(f"Circuit breaker {self.name} is OPEN")
            
            if self.state == CircuitBreakerState.HALF_OPEN and self.half_open_calls >= self.half_open_max_calls:
                logger.warning(f"Circuit breaker {self.name} HALF_OPEN limit reached")
                raise CircuitBreakerOpenException(f"Circuit breaker {self.name} HALF_OPEN limit reached")

        # Execute the function
        start_time = time.time()
        self.total_requests += 1
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            response_time = time.time() - start_time
            self._record_success(response_time)
            return result
            
        except Exception as e:
            # Record failure
            response_time = time.time() - start_time
            self._record_failure(response_time)
            raise e

    def _record_success(self, response_time: float):
        """Record a successful call"""
        with self._lock:
            self.success_count += 1
            self.last_success_time = datetime.now()
            self.response_times.append(response_time)
            
            # Keep only last 100 response times
            if len(self.response_times) > 100:
                self.response_times = self.response_times[-100:]
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.half_open_calls += 1
                if self.half_open_calls >= self.half_open_max_calls:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    logger.info(f"Circuit breaker {self.name} transitioning to CLOSED")
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0  # Reset failure count on success

    def _record_failure(self, response_time: float):
        """Record a failed call"""
        with self._lock:
            self.failure_count += 1
            self.failed_requests += 1
            self.last_failure_time = datetime.now()
            self.response_times.append(response_time)
            
            # Keep only last 100 response times
            if len(self.response_times) > 100:
                self.response_times = self.response_times[-100:]
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker {self.name} transitioning to OPEN due to failure in HALF_OPEN")
            elif self.state == CircuitBreakerState.CLOSED and self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker {self.name} transitioning to OPEN due to {self.failure_count} failures")

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset"""
        if self.last_failure_time is None:
            return False
        
        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.recovery_timeout

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get current circuit breaker metrics"""
        with self._lock:
            success_rate = (self.success_count / max(self.total_requests, 1)) * 100
            avg_response_time = sum(self.response_times) / max(len(self.response_times), 1)
            
            return CircuitBreakerMetrics(
                name=self.name,
                state=self.state,
                failure_count=self.failure_count,
                success_count=self.success_count,
                last_failure_time=self.last_failure_time,
                last_success_time=self.last_success_time,
                total_requests=self.total_requests,
                failed_requests=self.failed_requests,
                success_rate=success_rate,
                avg_response_time=avg_response_time
            )

    def reset(self):
        """Manually reset the circuit breaker"""
        with self._lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0
            logger.info(f"Circuit breaker {self.name} manually reset")

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class CircuitBreakerService:
    def __init__(self):
        self.failure_threshold = int(os.getenv('FAILURE_THRESHOLD', '5'))
        self.recovery_timeout = int(os.getenv('RECOVERY_TIMEOUT', '60'))
        self.half_open_max_calls = int(os.getenv('HALF_OPEN_MAX_CALLS', '3'))
        
        # Service endpoints to protect
        self.service_endpoints = {
            'postgres': os.getenv('POSTGRES_URL'),
            'redis': os.getenv('REDIS_URL'),
            'neo4j': os.getenv('NEO4J_URL'),
            'ollama': os.getenv('OLLAMA_URL')
        }
        
        # Create circuit breakers for each service
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        for service_name in self.service_endpoints.keys():
            self.circuit_breakers[service_name] = CircuitBreaker(
                name=service_name,
                failure_threshold=self.failure_threshold,
                recovery_timeout=self.recovery_timeout,
                half_open_max_calls=self.half_open_max_calls
            )
        
        self.app = web.Application()
        self.setup_routes()

    def setup_routes(self):
        """Setup HTTP API routes"""
        self.app.router.add_get('/health', self.health_handler)
        self.app.router.add_get('/metrics', self.metrics_handler)
        self.app.router.add_get('/status', self.status_handler)
        self.app.router.add_post('/reset/{service}', self.reset_handler)
        self.app.router.add_post('/test/{service}', self.test_handler)

    async def health_handler(self, request):
        """Health check endpoint"""
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'circuit_breakers': len(self.circuit_breakers)
        })

    async def metrics_handler(self, request):
        """Return metrics for all circuit breakers"""
        metrics = {}
        for name, cb in self.circuit_breakers.items():
            metrics[name] = asdict(cb.get_metrics())
            # Convert datetime objects to strings
            if metrics[name]['last_failure_time']:
                metrics[name]['last_failure_time'] = metrics[name]['last_failure_time'].isoformat()
            if metrics[name]['last_success_time']:
                metrics[name]['last_success_time'] = metrics[name]['last_success_time'].isoformat()
            # Convert enum to string
            metrics[name]['state'] = metrics[name]['state'].value
        
        return web.json_response({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })

    async def status_handler(self, request):
        """Return status summary for all circuit breakers"""
        status = {}
        for name, cb in self.circuit_breakers.items():
            status[name] = {
                'state': cb.state.value,
                'failure_count': cb.failure_count,
                'success_rate': (cb.success_count / max(cb.total_requests, 1)) * 100
            }
        
        return web.json_response({
            'timestamp': datetime.now().isoformat(),
            'status': status
        })

    async def reset_handler(self, request):
        """Reset a specific circuit breaker"""
        service = request.match_info['service']
        
        if service not in self.circuit_breakers:
            return web.json_response({'error': f'Service {service} not found'}, status=404)
        
        self.circuit_breakers[service].reset()
        
        return web.json_response({
            'message': f'Circuit breaker for {service} reset successfully',
            'timestamp': datetime.now().isoformat()
        })

    async def test_handler(self, request):
        """Test a service through its circuit breaker"""
        service = request.match_info['service']
        
        if service not in self.circuit_breakers:
            return web.json_response({'error': f'Service {service} not found'}, status=404)
        
        try:
            result = await self.test_service_connection(service)
            return web.json_response({
                'service': service,
                'result': result,
                'timestamp': datetime.now().isoformat()
            })
        except CircuitBreakerOpenException as e:
            return web.json_response({
                'service': service,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, status=503)
        except Exception as e:
            return web.json_response({
                'service': service,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, status=500)

    async def test_service_connection(self, service_name: str):
        """Test connection to a service through circuit breaker"""
        cb = self.circuit_breakers[service_name]
        endpoint = self.service_endpoints[service_name]
        
        async def test_connection():
            if service_name == 'postgres':
                # Test postgres connection
                import asyncpg
                conn = await asyncpg.connect(endpoint)
                await conn.execute('SELECT 1')
                await conn.close()
                return 'postgres_healthy'
            
            elif service_name == 'redis':
                # Test redis connection
                import aioredis
                redis = aioredis.from_url(endpoint)
                await redis.ping()
                await redis.close()
                return 'redis_healthy'
            
            elif service_name == 'neo4j':
                # Test neo4j connection
                from neo4j import AsyncGraphDatabase
                driver = AsyncGraphDatabase.driver(endpoint.split('@')[0], auth=('neo4j', endpoint.split('@')[1]))
                async with driver.session() as session:
                    await session.run('RETURN 1')
                await driver.close()
                return 'neo4j_healthy'
            
            elif service_name == 'ollama':
                # Test ollama connection
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{endpoint}/api/version") as response:
                        if response.status == 200:
                            return 'ollama_healthy'
                        else:
                            raise Exception(f"Ollama returned status {response.status}")
            
            else:
                raise Exception(f"Unknown service: {service_name}")
        
        return await cb.call(test_connection)

    async def monitor_services(self):
        """Background task to continuously monitor services"""
        while True:
            try:
                for service_name in self.service_endpoints.keys():
                    try:
                        await self.test_service_connection(service_name)
                        logger.debug(f"Service {service_name} is healthy")
                    except CircuitBreakerOpenException:
                        logger.debug(f"Circuit breaker for {service_name} is open")
                    except Exception as e:
                        logger.warning(f"Service {service_name} health check failed: {e}")
                
                # Save metrics
                await self.save_metrics()
                
                # Wait before next check
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)

    async def save_metrics(self):
        """Save circuit breaker metrics to file"""
        try:
            metrics = {}
            for name, cb in self.circuit_breakers.items():
                metrics[name] = asdict(cb.get_metrics())
                # Convert datetime objects to strings
                if metrics[name]['last_failure_time']:
                    metrics[name]['last_failure_time'] = metrics[name]['last_failure_time'].isoformat()
                if metrics[name]['last_success_time']:
                    metrics[name]['last_success_time'] = metrics[name]['last_success_time'].isoformat()
                # Convert enum to string
                metrics[name]['state'] = metrics[name]['state'].value
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'circuit_breakers': metrics
            }
            
            with open('/app/logs/circuit_breaker_metrics.json', 'w') as f:
                json.dump(report, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

def signal_handler(signum, frame):
    logger.info("Received shutdown signal, stopping circuit breaker service")
    sys.exit(0)

async def main():
    service = CircuitBreakerService()
    
    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start background monitoring
    monitor_task = asyncio.create_task(service.monitor_services())
    
    # Start web server
    runner = web.AppRunner(service.app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()
    
    logger.info("Circuit breaker service started on port 8080")
    
    try:
        await monitor_task
    except KeyboardInterrupt:
        logger.info("Shutting down circuit breaker service")
    finally:
        await runner.cleanup()

if __name__ == '__main__':
    asyncio.run(main())