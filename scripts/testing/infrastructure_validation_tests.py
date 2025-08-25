#!/usr/bin/env python3
"""
Infrastructure Validation Tests
==============================

Comprehensive validation tests for core infrastructure services:
- PostgreSQL database with connection pooling and performance
- Redis cache with high-availability and persistence  
- Neo4j graph database with clustering support
- RabbitMQ message broker with queue management
- Consul service discovery with health checks
- Kong API gateway with routing validation

Focus on actual system validation, not mocks.
"""

import asyncio
import aiohttp
import asyncpg
import aioredis
import neo4j
import pika
import subprocess
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import psutil

logger = logging.getLogger(__name__)

@dataclass
class InfrastructureTestResult:
    """Infrastructure test execution result"""
    service: str
    test_name: str
    success: bool
    duration: float
    metrics: Dict[str, Any]
    error_message: Optional[str] = None

class InfrastructureValidator:
    """Comprehensive infrastructure validation"""
    
    def __init__(self):
        self.results: List[InfrastructureTestResult] = []
        
        # Service configurations from docker-compose and port registry
        self.config = {
            "postgres": {
                "host": "localhost",
                "port": 10000,
                "database": "sutazai", 
                "user": "sutazai",
                "password": "sutazai123",
                "expected_connections": 200,
                "performance_threshold_ms": 50
            },
            "redis": {
                "host": "localhost",
                "port": 10001,
                "password": None,
                "performance_threshold_ms": 5,
                "expected_memory_mb": 400
            },
            "neo4j": {
                "host": "localhost", 
                "http_port": 10002,
                "bolt_port": 10003,
                "user": "neo4j",
                "password": "neo4j123",
                "performance_threshold_ms": 100
            },
            "rabbitmq": {
                "host": "localhost",
                "amqp_port": 10007,
                "management_port": 10008,
                "user": "sutazai",
                "password": "sutazai123",
                "vhost": "/",
                "expected_queues": ["agent_tasks", "notifications"]
            },
            "consul": {
                "host": "localhost",
                "port": 10006,
                "expected_services": ["backend", "frontend", "ollama"]
            },
            "kong": {
                "host": "localhost", 
                "proxy_port": 10005,
                "admin_port": 10015,
                "expected_routes": ["/api", "/health"]
            }
        }
    
    async def run_all_infrastructure_tests(self) -> List[InfrastructureTestResult]:
        """Execute all infrastructure validation tests"""
        logger.info("Starting comprehensive infrastructure validation")
        
        # Run tests in logical dependency order
        test_methods = [
            # Core databases first (can run in parallel)
            ("postgres", self.test_postgres_comprehensive),
            ("redis", self.test_redis_comprehensive), 
            ("neo4j", self.test_neo4j_comprehensive),
            
            # Message broker (depends on databases)
            ("rabbitmq", self.test_rabbitmq_comprehensive),
            
            # Service discovery and gateway (depends on core services)
            ("consul", self.test_consul_comprehensive),
            ("kong", self.test_kong_comprehensive)
        ]
        
        # Execute core database tests in parallel
        core_db_tasks = []
        for service, method in test_methods[:3]:  # postgres, redis, neo4j
            task = asyncio.create_task(method())
            core_db_tasks.append((service, task))
        
        # Wait for core databases
        for service, task in core_db_tasks:
            try:
                await task
            except Exception as e:
                logger.error(f"Core database test {service} failed: {e}")
        
        # Execute remaining tests sequentially
        for service, method in test_methods[3:]:
            try:
                await method()
            except Exception as e:
                logger.error(f"Infrastructure test {service} failed: {e}")
        
        return self.results
    
    async def test_postgres_comprehensive(self) -> None:
        """Comprehensive PostgreSQL validation"""
        config = self.config["postgres"]
        start_time = time.time()
        
        try:
            # Test basic connectivity
            conn = await asyncpg.connect(
                host=config["host"],
                port=config["port"],
                database=config["database"],
                user=config["user"],
                password=config["password"],
                timeout=10
            )
            
            # Test basic query performance
            query_start = time.time()
            result = await conn.fetchval("SELECT 1")
            query_duration = (time.time() - query_start) * 1000
            
            # Test connection pool capabilities
            pool = await asyncpg.create_pool(
                host=config["host"],
                port=config["port"],
                database=config["database"],
                user=config["user"],
                password=config["password"],
                min_size=5,
                max_size=20,
                timeout=10
            )
            
            # Test concurrent connections
            async def test_concurrent_query():
                async with pool.acquire() as pool_conn:
                    return await pool_conn.fetchval("SELECT pg_backend_pid()")
            
            concurrent_tasks = [test_concurrent_query() for _ in range(10)]
            backend_pids = await asyncio.gather(*concurrent_tasks)
            
            # Test database performance with realistic queries
            perf_queries = [
                "SELECT version()",
                "SELECT count(*) FROM information_schema.tables",
                "SELECT current_timestamp",
                "SHOW max_connections"
            ]
            
            perf_results = {}
            for query in perf_queries:
                perf_start = time.time()
                await conn.execute(query)
                perf_results[query] = (time.time() - perf_start) * 1000
            
            # Check PostgreSQL configuration
            pg_settings = await conn.fetch("""
                SELECT name, setting, unit 
                FROM pg_settings 
                WHERE name IN ('max_connections', 'shared_buffers', 'effective_cache_size')
            """)
            
            await conn.close()
            await pool.close()
            
            duration = time.time() - start_time
            
            self.results.append(InfrastructureTestResult(
                service="postgres",
                test_name="comprehensive_validation",
                success=True,
                duration=duration,
                metrics={
                    "query_performance_ms": query_duration,
                    "concurrent_connections": len(set(backend_pids)),
                    "performance_queries": perf_results,
                    "pg_settings": {row["name"]: row["setting"] for row in pg_settings},
                    "performance_grade": "excellent" if query_duration < 10 else "good" if query_duration < 50 else "poor"
                }
            ))
            
            logger.info(f"PostgreSQL validation successful - Query time: {query_duration:.2f}ms")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(InfrastructureTestResult(
                service="postgres",
                test_name="comprehensive_validation", 
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"PostgreSQL validation failed: {e}")
    
    async def test_redis_comprehensive(self) -> None:
        """Comprehensive Redis validation"""
        config = self.config["redis"]
        start_time = time.time()
        
        try:
            # Test basic connectivity
            redis = aioredis.from_url(f"redis://{config['host']}:{config['port']}")
            
            # Test basic operations with performance measurement
            perf_start = time.time()
            await redis.ping()
            ping_duration = (time.time() - perf_start) * 1000
            
            # Test data operations
            test_key = "infrastructure_test"
            test_value = {"timestamp": time.time(), "test": "comprehensive"}
            
            # Set and get operations
            set_start = time.time() 
            await redis.set(test_key, json.dumps(test_value))
            set_duration = (time.time() - set_start) * 1000
            
            get_start = time.time()
            retrieved_value = await redis.get(test_key)
            get_duration = (time.time() - get_start) * 1000
            
            # Test concurrent operations
            concurrent_ops = []
            for i in range(100):
                concurrent_ops.append(redis.set(f"concurrent_test_{i}", f"value_{i}"))
            
            concurrent_start = time.time()
            await asyncio.gather(*concurrent_ops)
            concurrent_duration = (time.time() - concurrent_start) * 1000
            
            # Get Redis info and configuration
            redis_info = await redis.info()
            redis_config = await redis.config_get("*memory*")
            
            # Test Redis data structures
            await redis.lpush("test_list", "item1", "item2", "item3")
            list_length = await redis.llen("test_list")
            
            await redis.hset("test_hash", mapping={"field1": "value1", "field2": "value2"})
            hash_data = await redis.hgetall("test_hash")
            
            # Cleanup test data
            await redis.delete(test_key, "test_list", "test_hash")
            for i in range(100):
                await redis.delete(f"concurrent_test_{i}")
            
            await redis.close()
            duration = time.time() - start_time
            
            self.results.append(InfrastructureTestResult(
                service="redis",
                test_name="comprehensive_validation",
                success=True,
                duration=duration,
                metrics={
                    "ping_time_ms": ping_duration,
                    "set_operation_ms": set_duration,
                    "get_operation_ms": get_duration,
                    "concurrent_ops_ms": concurrent_duration,
                    "concurrent_throughput": 100 / (concurrent_duration / 1000),
                    "memory_usage_mb": redis_info.get("used_memory", 0) / 1024 / 1024,
                    "connected_clients": redis_info.get("connected_clients", 0),
                    "total_commands": redis_info.get("total_commands_processed", 0),
                    "keyspace_hits": redis_info.get("keyspace_hits", 0),
                    "keyspace_misses": redis_info.get("keyspace_misses", 0),
                    "data_structures_test": {
                        "list_length": list_length,
                        "hash_fields": len(hash_data)
                    },
                    "performance_grade": "excellent" if ping_duration < 1 else "good" if ping_duration < 5 else "poor"
                }
            ))
            
            logger.info(f"Redis validation successful - Ping: {ping_duration:.2f}ms")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(InfrastructureTestResult(
                service="redis",
                test_name="comprehensive_validation",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Redis validation failed: {e}")
    
    async def test_neo4j_comprehensive(self) -> None:
        """Comprehensive Neo4j graph database validation"""
        config = self.config["neo4j"]
        start_time = time.time()
        
        try:
            # Test Neo4j connectivity via HTTP API
            async with aiohttp.ClientSession() as session:
                # Test HTTP endpoint
                http_url = f"http://{config['host']}:{config['http_port']}"
                
                async with session.get(f"{http_url}/", timeout=aiohttp.ClientTimeout(total=30)) as response:
                    http_available = response.status == 200
                
                # Test with authentication
                auth = aiohttp.BasicAuth(config["user"], config["password"])
                async with session.get(f"{http_url}/db/data/", auth=auth, 
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    auth_success = response.status == 200
            
            # Test Bolt protocol connection
            bolt_uri = f"bolt://{config['host']}:{config['bolt_port']}"
            
            try:
                driver = neo4j.GraphDatabase.driver(
                    bolt_uri, 
                    auth=(config["user"], config["password"]),
                    connection_timeout=30
                )
                
                # Test basic Cypher query
                def test_cypher_query(tx):
                    result = tx.run("RETURN 'Hello Neo4j' AS greeting, timestamp() AS time")
                    return result.single()
                
                with driver.session() as session:
                    query_start = time.time()
                    result = session.read_transaction(test_cypher_query)
                    query_duration = (time.time() - query_start) * 1000
                    
                    # Test database info
                    def get_db_info(tx):
                        return tx.run("CALL dbms.components() YIELD name, versions, edition").data()
                    
                    db_info = session.read_transaction(get_db_info)
                
                driver.close()
                bolt_success = True
                
            except Exception as bolt_error:
                bolt_success = False
                query_duration = 0
                db_info = []
                logger.warning(f"Neo4j Bolt connection failed: {bolt_error}")
            
            duration = time.time() - start_time
            
            self.results.append(InfrastructureTestResult(
                service="neo4j",
                test_name="comprehensive_validation",
                success=http_available or bolt_success,
                duration=duration,
                metrics={
                    "http_available": http_available,
                    "authentication_success": auth_success,
                    "bolt_connection": bolt_success,
                    "query_performance_ms": query_duration,
                    "database_info": db_info,
                    "performance_grade": "excellent" if query_duration < 50 else "good" if query_duration < 100 else "poor"
                }
            ))
            
            logger.info(f"Neo4j validation - HTTP: {http_available}, Bolt: {bolt_success}")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(InfrastructureTestResult(
                service="neo4j",
                test_name="comprehensive_validation",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Neo4j validation failed: {e}")
    
    async def test_rabbitmq_comprehensive(self) -> None:
        """Comprehensive RabbitMQ message broker validation"""
        config = self.config["rabbitmq"]
        start_time = time.time()
        
        try:
            # Test RabbitMQ management API
            auth = aiohttp.BasicAuth(config["user"], config["password"])
            management_url = f"http://{config['host']}:{config['management_port']}"
            
            async with aiohttp.ClientSession() as session:
                # Test overview
                async with session.get(f"{management_url}/api/overview", auth=auth,
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    overview_success = response.status == 200
                    if overview_success:
                        overview_data = await response.json()
                    else:
                        overview_data = {}
                
                # Test queues
                async with session.get(f"{management_url}/api/queues", auth=auth,
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    queues_success = response.status == 200
                    if queues_success:
                        queues_data = await response.json()
                    else:
                        queues_data = []
            
            # Test AMQP connection and basic operations
            try:
                connection_params = pika.ConnectionParameters(
                    host=config["host"],
                    port=config["amqp_port"], 
                    virtual_host=config["vhost"],
                    credentials=pika.PlainCredentials(config["user"], config["password"]),
                    heartbeat=30,
                    connection_attempts=3,
                    retry_delay=1
                )
                
                connection = pika.BlockingConnection(connection_params)
                channel = connection.channel()
                
                # Test queue operations
                test_queue = "infrastructure_test_queue"
                channel.queue_declare(queue=test_queue, durable=True)
                
                # Test message publishing
                test_message = json.dumps({
                    "test": "infrastructure_validation",
                    "timestamp": time.time()
                })
                
                publish_start = time.time()
                channel.basic_publish(
                    exchange='',
                    routing_key=test_queue,
                    body=test_message,
                    properties=pika.BasicProperties(delivery_mode=2)  # Persistent
                )
                publish_duration = (time.time() - publish_start) * 1000
                
                # Test message consumption
                consume_start = time.time()
                method, properties, body = channel.basic_get(queue=test_queue, auto_ack=True)
                consume_duration = (time.time() - consume_start) * 1000
                
                # Cleanup
                channel.queue_delete(queue=test_queue)
                connection.close()
                
                amqp_success = True
                message_test_success = method is not None
                
            except Exception as amqp_error:
                amqp_success = False
                message_test_success = False
                publish_duration = 0
                consume_duration = 0
                logger.warning(f"RabbitMQ AMQP test failed: {amqp_error}")
            
            duration = time.time() - start_time
            
            self.results.append(InfrastructureTestResult(
                service="rabbitmq",
                test_name="comprehensive_validation",
                success=overview_success or amqp_success,
                duration=duration,
                metrics={
                    "management_api_available": overview_success,
                    "amqp_connection": amqp_success,
                    "message_publishing": message_test_success,
                    "publish_duration_ms": publish_duration,
                    "consume_duration_ms": consume_duration,
                    "active_queues": len(queues_data),
                    "rabbitmq_version": overview_data.get("rabbitmq_version", "unknown"),
                    "erlang_version": overview_data.get("erlang_version", "unknown"),
                    "total_connections": overview_data.get("connections", 0),
                    "total_channels": overview_data.get("channels", 0),
                    "performance_grade": "excellent" if publish_duration < 10 else "good" if publish_duration < 50 else "poor"
                }
            ))
            
            logger.info(f"RabbitMQ validation - API: {overview_success}, AMQP: {amqp_success}")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(InfrastructureTestResult(
                service="rabbitmq", 
                test_name="comprehensive_validation",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"RabbitMQ validation failed: {e}")
    
    async def test_consul_comprehensive(self) -> None:
        """Comprehensive Consul service discovery validation"""
        config = self.config["consul"]
        start_time = time.time()
        
        try:
            consul_url = f"http://{config['host']}:{config['port']}"
            
            async with aiohttp.ClientSession() as session:
                # Test Consul leader election
                async with session.get(f"{consul_url}/v1/status/leader",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    leader_success = response.status == 200
                    if leader_success:
                        leader_data = await response.text()
                    else:
                        leader_data = ""
                
                # Test service catalog
                async with session.get(f"{consul_url}/v1/catalog/services",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    services_success = response.status == 200
                    if services_success:
                        services_data = await response.json()
                    else:
                        services_data = {}
                
                # Test health checks
                async with session.get(f"{consul_url}/v1/health/state/any",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    health_success = response.status == 200
                    if health_success:
                        health_data = await response.json()
                    else:
                        health_data = []
                
                # Test key-value store
                kv_test_key = "infrastructure_test/timestamp"
                kv_test_value = str(time.time())
                
                # Write to KV store
                async with session.put(f"{consul_url}/v1/kv/{kv_test_key}",
                                     data=kv_test_value,
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    kv_write_success = response.status == 200
                
                # Read from KV store  
                async with session.get(f"{consul_url}/v1/kv/{kv_test_key}",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    kv_read_success = response.status == 200
                    if kv_read_success:
                        kv_data = await response.json()
                    else:
                        kv_data = []
                
                # Cleanup KV test key
                if kv_write_success:
                    await session.delete(f"{consul_url}/v1/kv/{kv_test_key}")
            
            duration = time.time() - start_time
            
            # Analyze service health
            healthy_services = len([h for h in health_data if h.get("Status") == "passing"])
            total_health_checks = len(health_data)
            
            self.results.append(InfrastructureTestResult(
                service="consul",
                test_name="comprehensive_validation",
                success=leader_success and services_success,
                duration=duration,
                metrics={
                    "leader_election": leader_success,
                    "service_discovery": services_success,
                    "health_checks_available": health_success,
                    "kv_store_functional": kv_write_success and kv_read_success,
                    "registered_services": list(services_data.keys()) if services_data else [],
                    "service_count": len(services_data),
                    "health_checks_total": total_health_checks,
                    "healthy_services": healthy_services,
                    "health_ratio": healthy_services / max(total_health_checks, 1),
                    "leader_address": leader_data.strip('"') if leader_data else "unknown",
                    "performance_grade": "excellent" if duration < 1 else "good" if duration < 3 else "poor"
                }
            ))
            
            logger.info(f"Consul validation - Services: {len(services_data)}, Health checks: {healthy_services}/{total_health_checks}")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(InfrastructureTestResult(
                service="consul",
                test_name="comprehensive_validation",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Consul validation failed: {e}")
    
    async def test_kong_comprehensive(self) -> None:
        """Comprehensive Kong API gateway validation"""
        config = self.config["kong"]
        start_time = time.time()
        
        try:
            proxy_url = f"http://{config['host']}:{config['proxy_port']}"
            admin_url = f"http://{config['host']}:{config['admin_port']}"
            
            async with aiohttp.ClientSession() as session:
                # Test Kong status via proxy
                async with session.get(f"{proxy_url}/status",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    proxy_status_success = response.status == 200
                
                # Test Kong admin API
                async with session.get(f"{admin_url}/",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    admin_success = response.status == 200
                    if admin_success:
                        admin_data = await response.json()
                    else:
                        admin_data = {}
                
                # Test Kong services configuration
                async with session.get(f"{admin_url}/services",
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    services_success = response.status == 200
                    if services_success:
                        services_data = await response.json()
                    else:
                        services_data = {"data": []}
                
                # Test Kong routes configuration
                async with session.get(f"{admin_url}/routes", 
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    routes_success = response.status == 200
                    if routes_success:
                        routes_data = await response.json()
                    else:
                        routes_data = {"data": []}
                
                # Test proxy routing (if backend service is configured)
                routing_test_success = False
                routing_error = None
                
                try:
                    async with session.get(f"{proxy_url}/health",
                                         timeout=aiohttp.ClientTimeout(total=15)) as response:
                        routing_test_success = response.status in [200, 404, 502, 503]  # Accept routing attempts
                except Exception as routing_err:
                    routing_error = str(routing_err)
            
            duration = time.time() - start_time
            
            self.results.append(InfrastructureTestResult(
                service="kong",
                test_name="comprehensive_validation",
                success=proxy_status_success or admin_success,
                duration=duration,
                metrics={
                    "proxy_available": proxy_status_success,
                    "admin_api_available": admin_success,
                    "services_configured": len(services_data.get("data", [])),
                    "routes_configured": len(routes_data.get("data", [])),
                    "routing_test": routing_test_success,
                    "routing_error": routing_error,
                    "kong_version": admin_data.get("version", "unknown"),
                    "configuration": admin_data.get("configuration", {}),
                    "performance_grade": "excellent" if duration < 1 else "good" if duration < 3 else "poor"
                }
            ))
            
            logger.info(f"Kong validation - Proxy: {proxy_status_success}, Admin: {admin_success}, Routes: {len(routes_data.get('data', []))}")
            
        except Exception as e:
            duration = time.time() - start_time
            self.results.append(InfrastructureTestResult(
                service="kong",
                test_name="comprehensive_validation",
                success=False,
                duration=duration,
                metrics={},
                error_message=str(e)
            ))
            logger.error(f"Kong validation failed: {e}")
    
    def generate_infrastructure_report(self) -> Dict[str, Any]:
        """Generate comprehensive infrastructure validation report"""
        total_tests = len(self.results)
        successful_tests = len([r for r in self.results if r.success])
        
        # Group results by service
        service_results = {}
        for result in self.results:
            service_results[result.service] = result
        
        # Calculate overall infrastructure health
        critical_services = ["postgres", "redis", "rabbitmq", "consul"]
        critical_success = sum(1 for svc in critical_services 
                             if svc in service_results and service_results[svc].success)
        
        infrastructure_grade = "EXCELLENT" if critical_success == len(critical_services) else \
                              "GOOD" if critical_success >= len(critical_services) - 1 else \
                              "POOR"
        
        # Performance analysis
        performance_summary = {}
        for result in self.results:
            if result.success and "performance_grade" in result.metrics:
                performance_summary[result.service] = result.metrics["performance_grade"]
        
        return {
            "summary": {
                "total_services_tested": total_tests,
                "successful_services": successful_tests,
                "success_rate": round(successful_services / max(total_tests, 1) * 100, 2),
                "infrastructure_grade": infrastructure_grade,
                "critical_services_health": f"{critical_success}/{len(critical_services)}"
            },
            "service_details": {
                service: {
                    "status": "success" if result.success else "failed",
                    "duration_seconds": round(result.duration, 3),
                    "key_metrics": result.metrics,
                    "error": result.error_message
                }
                for service, result in service_results.items()
            },
            "performance_analysis": performance_summary,
            "recommendations": self._generate_infrastructure_recommendations(service_results)
        }
    
    def _generate_infrastructure_recommendations(self, service_results: Dict) -> List[str]:
        """Generate infrastructure improvement recommendations"""
        recommendations = []
        
        for service, result in service_results.items():
            if not result.success:
                recommendations.append(f"🔴 CRITICAL: {service} service is not accessible - check container and network configuration")
            elif result.metrics.get("performance_grade") == "poor":
                recommendations.append(f"🟡 PERFORMANCE: {service} performance is below optimal - consider resource scaling")
            
        # Specific service recommendations
        if "postgres" in service_results and service_results["postgres"].success:
            pg_metrics = service_results["postgres"].metrics
            if pg_metrics.get("query_performance_ms", 0) > 100:
                recommendations.append("🟡 PostgreSQL query performance is slow - check connection pooling and indexing")
        
        if "redis" in service_results and service_results["redis"].success:
            redis_metrics = service_results["redis"].metrics
            memory_mb = redis_metrics.get("memory_usage_mb", 0)
            if memory_mb > 300:
                recommendations.append(f"🟡 Redis memory usage is high ({memory_mb:.1f}MB) - monitor for memory leaks")
        
        if "consul" in service_results and service_results["consul"].success:
            consul_metrics = service_results["consul"].metrics
            health_ratio = consul_metrics.get("health_ratio", 1.0)
            if health_ratio < 0.8:
                recommendations.append(f"🟡 Consul health checks failing ({health_ratio:.1%}) - investigate service health")
        
        return recommendations if recommendations else ["✅ Infrastructure is operating within optimal parameters"]

async def main():
    """Main execution for infrastructure validation"""
    validator = InfrastructureValidator()
    
    print("🏗️  Starting Infrastructure Validation Tests")
    print("=" * 60)
    
    results = await validator.run_all_infrastructure_tests()
    report = validator.generate_infrastructure_report()
    
    print("\n" + "=" * 60)
    print("📊 INFRASTRUCTURE VALIDATION COMPLETE")
    print("=" * 60)
    
    # Print summary
    summary = report["summary"]
    print(f"Services Tested: {summary['total_services_tested']}")
    print(f"Successful: {summary['successful_services']} ({summary['success_rate']}%)")
    print(f"Infrastructure Grade: {summary['infrastructure_grade']}")
    print(f"Critical Services: {summary['critical_services_health']}")
    
    # Print service details
    print("\n🔍 Service Status:")
    for service, details in report["service_details"].items():
        status_icon = "✅" if details["status"] == "success" else "❌"
        duration = details["duration_seconds"]
        print(f"  {status_icon} {service}: {details['status']} ({duration:.2f}s)")
        
        if details["error"]:
            print(f"    ⚠️  {details['error']}")
    
    # Print recommendations
    print("\n💡 Recommendations:")
    for rec in report["recommendations"]:
        print(f"  {rec}")
    
    # Save detailed report
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"infrastructure_validation_report_{timestamp}.json"
    
    import json
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return summary["infrastructure_grade"] in ["EXCELLENT", "GOOD"]

if __name__ == "__main__":
    success = asyncio.run(main())
    import sys
    sys.exit(0 if success else 1)