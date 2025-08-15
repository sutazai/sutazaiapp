#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
ULTRA Performance Profiler - System-wide performance analysis
Measures response times, memory usage, and identifies bottlenecks
"""

import asyncio
import time
import psutil
import aiohttp
import json
from datetime import datetime
from typing import Dict, List, Tuple
import statistics
import tracemalloc
import redis
import asyncpg
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class UltraPerformanceProfiler:
    """High-precision performance profiling for production systems"""
    
    def __init__(self):
        self.base_url = "http://localhost:10010"
        self.frontend_url = "http://localhost:10011"
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "api_response_times": {},
            "memory_usage": {},
            "database_performance": {},
            "cache_performance": {},
            "bottlenecks": [],
            "recommendations": []
        }
        
    async def profile_api_endpoints(self) -> Dict:
        """Profile all API endpoints for response time"""
        endpoints = [
            ("GET", "/health", None),
            ("GET", "/api/v1/models/", None),
            ("POST", "/api/v1/chat/", {"message": "test", "model": "tinyllama"}),
            ("GET", "/metrics", None),
            ("POST", "/api/v1/mesh/enqueue", {"task": "test_task", "priority": 1}),
            ("GET", "/api/v1/mesh/results", None),
            ("GET", "/api/v1/agents/", None),
            ("GET", "/api/v1/tasks/", None),
        ]
        
        async with aiohttp.ClientSession() as session:
            for method, endpoint, data in endpoints:
                times = []
                for _ in range(10):  # 10 samples per endpoint
                    start = time.perf_counter()
                    try:
                        if method == "GET":
                            async with session.get(f"{self.base_url}{endpoint}") as resp:
                                await resp.text()
                        else:
                            async with session.post(f"{self.base_url}{endpoint}", json=data) as resp:
                                await resp.text()
                        elapsed = (time.perf_counter() - start) * 1000  # ms
                        times.append(elapsed)
                    except Exception as e:
                        logger.error(f"Error profiling {endpoint}: {e}")
                        times.append(999999)  # Flag as error
                
                if times:
                    self.results["api_response_times"][endpoint] = {
                        "min": min(times),
                        "max": max(times),
                        "avg": statistics.mean(times),
                        "p50": statistics.median(times),
                        "p95": np.percentile(times, 95) if len(times) > 1 else times[0],
                        "p99": np.percentile(times, 99) if len(times) > 1 else times[0],
                        "samples": len(times)
                    }
        
        return self.results["api_response_times"]
    
    def profile_memory_usage(self) -> Dict:
        """Profile system and process memory usage"""
        process = psutil.Process()
        
        # System-wide memory
        vm = psutil.virtual_memory()
        self.results["memory_usage"]["system"] = {
            "total_gb": vm.total / (1024**3),
            "used_gb": vm.used / (1024**3),
            "available_gb": vm.available / (1024**3),
            "percent": vm.percent
        }
        
        # Process memory
        mem_info = process.memory_info()
        self.results["memory_usage"]["backend_process"] = {
            "rss_mb": mem_info.rss / (1024**2),
            "vms_mb": mem_info.vms / (1024**2),
            "percent": process.memory_percent()
        }
        
        # Docker containers memory
        try:
            containers = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                if 'docker' in proc.info['name']:
                    containers.append({
                        "name": proc.info['name'],
                        "memory_mb": proc.info['memory_info'].rss / (1024**2)
                    })
            self.results["memory_usage"]["containers"] = containers[:10]  # Top 10
        except:
            pass
        
        return self.results["memory_usage"]
    
    async def profile_database(self) -> Dict:
        """Profile PostgreSQL performance"""
        conn = None
        try:
            conn = await asyncpg.connect(
                host='localhost',
                port=10000,
                user='sutazai',
                password='sutazai_secure_2024',
                database='sutazai_db'
            )
            
            # Test query performance
            queries = [
                ("SELECT 1", "ping"),
                ("SELECT COUNT(*) FROM pg_tables", "count_tables"),
                ("SELECT * FROM pg_stat_activity", "active_connections"),
                ("SELECT * FROM pg_stat_database WHERE datname = 'sutazai_db'", "db_stats")
            ]
            
            for query, name in queries:
                times = []
                for _ in range(5):
                    start = time.perf_counter()
                    await conn.fetch(query)
                    elapsed = (time.perf_counter() - start) * 1000
                    times.append(elapsed)
                
                self.results["database_performance"][name] = {
                    "avg_ms": statistics.mean(times),
                    "min_ms": min(times),
                    "max_ms": max(times)
                }
            
            # Get connection pool stats
            stats = await conn.fetchone(
                "SELECT count(*) as connections FROM pg_stat_activity WHERE datname = 'sutazai_db'"
            )
            self.results["database_performance"]["active_connections"] = stats['connections']
            
        except Exception as e:
            self.results["database_performance"]["error"] = str(e)
        finally:
            if conn:
                await conn.close()
        
        return self.results["database_performance"]
    
    def profile_redis_cache(self) -> Dict:
        """Profile Redis cache performance"""
        try:
            r = redis.Redis(host='localhost', port=10001, decode_responses=True)
            
            # Test operations
            operations = []
            
            # SET performance
            times = []
            for i in range(100):
                key = f"perf_test_{i}"
                value = "x" * 1024  # 1KB payload
                start = time.perf_counter()
                r.set(key, value)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
            
            self.results["cache_performance"]["set_1kb"] = {
                "avg_ms": statistics.mean(times),
                "p95_ms": np.percentile(times, 95)
            }
            
            # GET performance
            times = []
            for i in range(100):
                key = f"perf_test_{i}"
                start = time.perf_counter()
                r.get(key)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
            
            self.results["cache_performance"]["get_1kb"] = {
                "avg_ms": statistics.mean(times),
                "p95_ms": np.percentile(times, 95)
            }
            
            # Cleanup
            for i in range(100):
                r.delete(f"perf_test_{i}")
            
            # Cache stats
            info = r.info('stats')
            self.results["cache_performance"]["stats"] = {
                "total_connections": info.get('total_connections_received', 0),
                "commands_processed": info.get('total_commands_processed', 0),
                "keyspace_hits": info.get('keyspace_hits', 0),
                "keyspace_misses": info.get('keyspace_misses', 0),
                "hit_rate": info.get('keyspace_hits', 0) / max(1, info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0))
            }
            
        except Exception as e:
            self.results["cache_performance"]["error"] = str(e)
        
        return self.results["cache_performance"]
    
    def identify_bottlenecks(self):
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # API Response Time Bottlenecks
        for endpoint, metrics in self.results["api_response_times"].items():
            if metrics["p95"] > 50:  # >50ms P95
                bottlenecks.append({
                    "type": "api_response",
                    "severity": "high" if metrics["p95"] > 100 else "medium",
                    "endpoint": endpoint,
                    "p95_ms": metrics["p95"],
                    "target_ms": 50
                })
        
        # Memory Bottlenecks
        if self.results["memory_usage"]["system"]["percent"] > 80:
            bottlenecks.append({
                "type": "memory",
                "severity": "high",
                "usage_percent": self.results["memory_usage"]["system"]["percent"],
                "target_percent": 70
            })
        
        # Database Bottlenecks
        db_perf = self.results.get("database_performance", {})
        if db_perf.get("active_connections", 0) > 50:
            bottlenecks.append({
                "type": "database_connections",
                "severity": "medium",
                "connections": db_perf["active_connections"],
                "target": 30
            })
        
        # Cache Hit Rate
        cache_stats = self.results.get("cache_performance", {}).get("stats", {})
        if cache_stats.get("hit_rate", 0) < 0.8:
            bottlenecks.append({
                "type": "cache_hit_rate",
                "severity": "medium",
                "hit_rate": cache_stats.get("hit_rate", 0),
                "target": 0.9
            })
        
        self.results["bottlenecks"] = bottlenecks
        return bottlenecks
    
    def generate_recommendations(self):
        """Generate optimization recommendations"""
        recommendations = []
        
        for bottleneck in self.results["bottlenecks"]:
            if bottleneck["type"] == "api_response":
                recommendations.append({
                    "area": "API Performance",
                    "issue": f"Endpoint {bottleneck['endpoint']} has P95 of {bottleneck['p95_ms']:.1f}ms",
                    "action": "Implement Redis caching for this endpoint",
                    "impact": "High",
                    "effort": "Low"
                })
            elif bottleneck["type"] == "memory":
                recommendations.append({
                    "area": "Memory Usage",
                    "issue": f"System memory at {bottleneck['usage_percent']:.1f}%",
                    "action": "Implement memory pooling and optimize container limits",
                    "impact": "High",
                    "effort": "Medium"
                })
            elif bottleneck["type"] == "database_connections":
                recommendations.append({
                    "area": "Database",
                    "issue": f"High connection count: {bottleneck['connections']}",
                    "action": "Implement connection pooling with pgbouncer",
                    "impact": "Medium",
                    "effort": "Low"
                })
            elif bottleneck["type"] == "cache_hit_rate":
                recommendations.append({
                    "area": "Caching",
                    "issue": f"Low cache hit rate: {bottleneck['hit_rate']:.2%}",
                    "action": "Implement aggressive caching strategy for hot data",
                    "impact": "High",
                    "effort": "Low"
                })
        
        self.results["recommendations"] = recommendations
        return recommendations
    
    async def run_full_profile(self):
        """Run complete performance profile"""
        logger.info("🚀 ULTRA Performance Profiler Starting...")
        logger.info("=" * 60)
        
        # Profile all components
        logger.info("\n📊 Profiling API Endpoints...")
        await self.profile_api_endpoints()
        
        logger.info("💾 Profiling Memory Usage...")
        self.profile_memory_usage()
        
        logger.info("🗄️ Profiling Database...")
        await self.profile_database()
        
        logger.info("⚡ Profiling Redis Cache...")
        self.profile_redis_cache()
        
        logger.info("🔍 Identifying Bottlenecks...")
        self.identify_bottlenecks()
        
        logger.info("💡 Generating Recommendations...")
        self.generate_recommendations()
        
        # Save results
        with open('/opt/sutazaiapp/reports/performance_profile.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Print summary
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print performance summary"""
        logger.info("\n" + "=" * 60)
        logger.info("📈 PERFORMANCE PROFILE SUMMARY")
        logger.info("=" * 60)
        
        # API Performance
        logger.info("\n🌐 API Response Times (P95):")
        for endpoint, metrics in self.results["api_response_times"].items():
            status = "✅" if metrics["p95"] < 50 else "⚠️" if metrics["p95"] < 100 else "❌"
            logger.info(f"  {status} {endpoint}: {metrics['p95']:.1f}ms")
        
        # Memory Usage
        mem = self.results["memory_usage"]["system"]
        logger.info(f"\n💾 Memory Usage: {mem['used_gb']:.2f}GB / {mem['total_gb']:.2f}GB ({mem['percent']:.1f}%)")
        
        # Bottlenecks
        if self.results["bottlenecks"]:
            logger.info(f"\n⚠️ Bottlenecks Found: {len(self.results['bottlenecks'])}")
            for b in self.results["bottlenecks"]:
                logger.info(f"  - {b['type']}: Severity {b['severity'].upper()}")
        
        # Recommendations
        if self.results["recommendations"]:
            logger.info(f"\n💡 Top Recommendations:")
            for r in self.results["recommendations"][:3]:
                logger.info(f"  - {r['area']}: {r['action']}")
        
        logger.info("\n✅ Full report saved to: /opt/sutazaiapp/reports/performance_profile.json")

async def main():
    profiler = UltraPerformanceProfiler()
    await profiler.run_full_profile()

if __name__ == "__main__":
    asyncio.run(main())