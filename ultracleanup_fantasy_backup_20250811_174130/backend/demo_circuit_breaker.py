#!/usr/bin/env python3
"""
Demonstration of Circuit Breaker usage in service calls
Shows how the circuit breaker protects against cascading failures
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.connection_pool import get_pool_manager
from app.core.circuit_breaker import CircuitBreakerError
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CircuitBreakerDemo:
    def __init__(self):
        self.pool_manager = None
        self.results = []
    
    async def setup(self):
        """Initialize the connection pool manager"""
        self.pool_manager = await get_pool_manager()
        logger.info("Connection pool manager initialized")
    
    async def demo_ollama_with_circuit_breaker(self):
        """Demonstrate Ollama service calls with circuit breaker protection"""
        logger.info("=" * 60)
        logger.info("DEMO: Ollama Service with Circuit Breaker")
        logger.info("=" * 60)
        
        # Simulate multiple requests to Ollama
        for i in range(10):
            try:
                logger.info(f"\nAttempt {i+1}: Calling Ollama service...")
                
                # Use the circuit breaker protected HTTP request
                response = await self.pool_manager.make_http_request(
                    service='ollama',
                    method='GET',
                    url='/api/tags'
                )
                
                logger.info(f"✓ Success: Got response with status {response.status_code}")
                self.results.append({"attempt": i+1, "service": "ollama", "status": "success"})
                
            except CircuitBreakerError as e:
                logger.warning(f"⚠️ Circuit breaker OPEN: {e}")
                self.results.append({"attempt": i+1, "service": "ollama", "status": "circuit_open"})
                
                # Wait a bit before next attempt
                logger.info("Waiting 5 seconds before retry...")
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"✗ Request failed: {e}")
                self.results.append({"attempt": i+1, "service": "ollama", "status": "failed"})
            
            # Small delay between requests
            await asyncio.sleep(1)
        
        # Check circuit breaker status
        cb_status = self.pool_manager.get_circuit_breaker_status()
        logger.info(f"\nCircuit Breaker Status after demo:")
        logger.info(json.dumps(cb_status, indent=2))
    
    async def demo_redis_with_circuit_breaker(self):
        """Demonstrate Redis operations with circuit breaker protection"""
        logger.info("\n" + "=" * 60)
        logger.info("DEMO: Redis Operations with Circuit Breaker")
        logger.info("=" * 60)
        
        test_key = "circuit_breaker_test"
        test_value = "demo_value"
        
        # Test SET operation
        try:
            logger.info(f"Setting key '{test_key}' in Redis...")
            result = await self.pool_manager.execute_redis_command(
                'set', test_key, test_value
            )
            logger.info(f"✓ SET successful: {result}")
            self.results.append({"operation": "redis_set", "status": "success"})
        except CircuitBreakerError as e:
            logger.warning(f"⚠️ Circuit breaker OPEN for Redis: {e}")
            self.results.append({"operation": "redis_set", "status": "circuit_open"})
        except Exception as e:
            logger.error(f"✗ Redis SET failed: {e}")
            self.results.append({"operation": "redis_set", "status": "failed"})
        
        await asyncio.sleep(1)
        
        # Test GET operation
        try:
            logger.info(f"Getting key '{test_key}' from Redis...")
            result = await self.pool_manager.execute_redis_command('get', test_key)
            logger.info(f"✓ GET successful: {result}")
            self.results.append({"operation": "redis_get", "status": "success"})
        except CircuitBreakerError as e:
            logger.warning(f"⚠️ Circuit breaker OPEN for Redis: {e}")
            self.results.append({"operation": "redis_get", "status": "circuit_open"})
        except Exception as e:
            logger.error(f"✗ Redis GET failed: {e}")
            self.results.append({"operation": "redis_get", "status": "failed"})
        
        # Clean up
        try:
            await self.pool_manager.execute_redis_command('delete', test_key)
            logger.info("✓ Test key cleaned up")
        except:
            pass
    
    async def demo_database_with_circuit_breaker(self):
        """Demonstrate database queries with circuit breaker protection"""
        logger.info("\n" + "=" * 60)
        logger.info("DEMO: Database Queries with Circuit Breaker")
        logger.info("=" * 60)
        
        # Test simple query
        try:
            logger.info("Executing database health check query...")
            result = await self.pool_manager.execute_db_query(
                "SELECT 1 as health_check", 
                fetch_one=True
            )
            logger.info(f"✓ Database query successful: {dict(result)}")
            self.results.append({"operation": "db_query", "status": "success"})
        except CircuitBreakerError as e:
            logger.warning(f"⚠️ Circuit breaker OPEN for database: {e}")
            self.results.append({"operation": "db_query", "status": "circuit_open"})
        except Exception as e:
            logger.error(f"✗ Database query failed: {e}")
            self.results.append({"operation": "db_query", "status": "failed"})
        
        await asyncio.sleep(1)
        
        # Test table query
        try:
            logger.info("Querying database tables...")
            result = await self.pool_manager.execute_db_query(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public' LIMIT 5"
            )
            logger.info(f"✓ Found {len(result)} tables")
            for row in result:
                logger.info(f"  - {row['tablename']}")
            self.results.append({"operation": "db_tables", "status": "success"})
        except CircuitBreakerError as e:
            logger.warning(f"⚠️ Circuit breaker OPEN for database: {e}")
            self.results.append({"operation": "db_tables", "status": "circuit_open"})
        except Exception as e:
            logger.error(f"✗ Table query failed: {e}")
            self.results.append({"operation": "db_tables", "status": "failed"})
    
    async def demo_circuit_breaker_recovery(self):
        """Demonstrate circuit breaker recovery after failures"""
        logger.info("\n" + "=" * 60)
        logger.info("DEMO: Circuit Breaker Recovery Mechanism")
        logger.info("=" * 60)
        
        # First, let's check the current state
        initial_status = self.pool_manager.get_circuit_breaker_status()
        logger.info("Initial circuit breaker states:")
        for service, data in initial_status.get('breakers', {}).items():
            logger.info(f"  {service}: {data['state']}")
        
        # If any circuit is open, demonstrate recovery
        for service, data in initial_status.get('breakers', {}).items():
            if data['state'] == 'open':
                logger.info(f"\n'{service}' circuit is OPEN. Waiting for recovery timeout...")
                logger.info("Circuit will transition to HALF_OPEN after 30 seconds")
                logger.info("Then a test request will determine if service has recovered")
                
                # In a real scenario, we would wait 30 seconds
                # For demo, we'll just show the concept
                logger.info("(In production, this would wait 30 seconds)")
                
        # Reset a circuit breaker manually for demo
        logger.info("\nManually resetting 'ollama' circuit breaker for demonstration...")
        self.pool_manager.reset_circuit_breaker('ollama')
        
        # Check status after reset
        final_status = self.pool_manager.get_circuit_breaker_status()
        logger.info("\nCircuit breaker states after reset:")
        for service, data in final_status.get('breakers', {}).items():
            logger.info(f"  {service}: {data['state']}")
    
    async def run_demo(self):
        """Run all demonstrations"""
        await self.setup()
        
        logger.info("\n" + "=" * 60)
        logger.info("CIRCUIT BREAKER DEMONSTRATION")
        logger.info("Showing resilient service communication patterns")
        logger.info("=" * 60)
        
        # Run demonstrations
        await self.demo_redis_with_circuit_breaker()
        await self.demo_database_with_circuit_breaker()
        await self.demo_ollama_with_circuit_breaker()
        await self.demo_circuit_breaker_recovery()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("DEMONSTRATION SUMMARY")
        logger.info("=" * 60)
        
        success_count = sum(1 for r in self.results if r.get('status') == 'success')
        circuit_open_count = sum(1 for r in self.results if r.get('status') == 'circuit_open')
        failed_count = sum(1 for r in self.results if r.get('status') == 'failed')
        
        logger.info(f"Total operations: {len(self.results)}")
        logger.info(f"  ✓ Successful: {success_count}")
        logger.info(f"  ⚠️ Circuit breaker protected: {circuit_open_count}")
        logger.info(f"  ✗ Failed: {failed_count}")
        
        # Get final metrics
        final_metrics = self.pool_manager.get_circuit_breaker_status()
        logger.info("\nFinal Circuit Breaker Metrics:")
        logger.info(json.dumps(final_metrics['global'], indent=2))
        
        # Save results
        with open("circuit_breaker_demo_results.json", "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": self.results,
                "final_metrics": final_metrics
            }, f, indent=2)
        
        logger.info("\nDemo results saved to circuit_breaker_demo_results.json")
        
        # Clean up
        if self.pool_manager:
            await self.pool_manager.close()


async def main():
    """Main entry point"""
    demo = CircuitBreakerDemo()
    try:
        await demo.run_demo()
    except KeyboardInterrupt:
        logger.info("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())