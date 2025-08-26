#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
ULTRAPERFORMANCE Baseline Testing Suite
Measures ALL performance metrics for complete system optimization
"""

import asyncio
import time
import psutil
import aiohttp
import redis.asyncio as redis
import asyncpg
from typing import Dict, List, Tuple
import json
from datetime import datetime
import statistics
import concurrent.futures
from dataclasses import dataclass, asdict

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    endpoint: str
    method: str
    min_response_time: float
    max_response_time: float
    avg_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    cache_hit_rate: float
    db_query_time: float
    timestamp: str

class UltraPerformanceAnalyzer:
    """ULTRAPERFORMANCE analysis engine"""
    
    def __init__(self):
        self.base_url = "http://localhost:10010"
        self.redis_client = None
        self.pg_conn = None
        self.metrics = []
        self.cache_stats = {"hits": 0, "misses": 0}
        
    async def setup(self):
        """Initialize connections"""
        redis_url = os.getenv("REDIS_URL", "redis://localhost:10001")
        self.redis_client = await redis.from_url(redis_url)
        self.pg_conn = await asyncpg.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "10000")),
            database=os.getenv("POSTGRES_DB", "sutazai"),
            user=os.getenv("POSTGRES_USER", "sutazai"),
            password=os.getenv("POSTGRES_PASSWORD", "sutazai")
        )
        
    async def cleanup(self):
        """Cleanup connections"""
        if self.redis_client:
            await self.redis_client.close()
        if self.pg_conn:
            await self.pg_conn.close()
            
    async def measure_endpoint(self, endpoint: str, method: str = "GET", 
                              payload: dict = None, iterations: int = 100) -> PerformanceMetrics:
        """Measure endpoint performance with multiple iterations"""
        url = f"{self.base_url}{endpoint}"
        response_times = []
        errors = 0
        
        # CPU and memory baseline
        process = psutil.Process()
        cpu_baseline = process.cpu_percent(interval=0.1)
        mem_baseline = process.memory_info().rss / 1024 / 1024  # MB
        
        logger.info(f"Testing {method} {endpoint} with {iterations} iterations...")
        
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            
            for i in range(iterations):
                try:
                    req_start = time.time()
                    
                    if method == "GET":
                        async with session.get(url) as response:
                            await response.text()
                            status = response.status
                    elif method == "POST":
                        async with session.post(url, json=payload or {}) as response:
                            await response.text()
                            status = response.status
                            
                    req_time = (time.time() - req_start) * 1000  # ms
                    response_times.append(req_time)
                    
                    if status >= 400:
                        errors += 1
                        
                except Exception as e:
                    errors += 1
                    logger.error(f"Error on iteration {i}: {e}")
                    
            total_time = time.time() - start_time
            
        # Calculate metrics
        sorted_times = sorted(response_times) if response_times else [0]
        
        metrics = PerformanceMetrics(
            endpoint=endpoint,
            method=method,
            min_response_time=min(sorted_times) if sorted_times else 0,
            max_response_time=max(sorted_times) if sorted_times else 0,
            avg_response_time=statistics.mean(sorted_times) if sorted_times else 0,
            median_response_time=statistics.median(sorted_times) if sorted_times else 0,
            p95_response_time=sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0,
            p99_response_time=sorted_times[int(len(sorted_times) * 0.99)] if sorted_times else 0,
            requests_per_second=iterations / total_time if total_time > 0 else 0,
            error_rate=(errors / iterations) * 100,
            cpu_usage=process.cpu_percent() - cpu_baseline,
            memory_usage=(process.memory_info().rss / 1024 / 1024) - mem_baseline,
            cache_hit_rate=await self.measure_cache_hit_rate(),
            db_query_time=await self.measure_db_query_time(),
            timestamp=datetime.now().isoformat()
        )
        
        self.metrics.append(metrics)
        return metrics
        
    async def measure_cache_hit_rate(self) -> float:
        """Measure Redis cache hit rate"""
        try:
            info = await self.redis_client.info("stats")
            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            total = hits + misses
            return (hits / total * 100) if total > 0 else 0
        except:
            return 0.0
            
    async def measure_db_query_time(self) -> float:
        """Measure average database query time"""
        try:
            start = time.time()
            await self.pg_conn.fetch("SELECT 1")
            return (time.time() - start) * 1000  # ms
        except:
            return 0.0
            
    async def load_test_concurrent(self, endpoint: str, concurrent_users: int = 50,
                                  duration_seconds: int = 30) -> Dict:
        """Concurrent load testing"""
        logger.info(f"\nLoad testing {endpoint} with {concurrent_users} concurrent users for {duration_seconds}s...")
        
        results = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "throughput": 0
        }
        
        async def worker():
            url = f"{self.base_url}{endpoint}"
            local_results = {"success": 0, "fail": 0, "times": []}
            
            async with aiohttp.ClientSession() as session:
                end_time = time.time() + duration_seconds
                
                while time.time() < end_time:
                    try:
                        start = time.time()
                        async with session.get(url) as response:
                            await response.text()
                            elapsed = (time.time() - start) * 1000
                            
                            if response.status < 400:
                                local_results["success"] += 1
                            else:
                                local_results["fail"] += 1
                                
                            local_results["times"].append(elapsed)
                    except:
                        local_results["fail"] += 1
                        
            return local_results
            
        # Run concurrent workers
        start_time = time.time()
        tasks = [worker() for _ in range(concurrent_users)]
        worker_results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Aggregate results
        for wr in worker_results:
            results["successful_requests"] += wr["success"]
            results["failed_requests"] += wr["fail"]
            results["response_times"].extend(wr["times"])
            
        results["total_requests"] = results["successful_requests"] + results["failed_requests"]
        results["throughput"] = results["total_requests"] / total_time if total_time > 0 else 0
        
        if results["response_times"]:
            sorted_times = sorted(results["response_times"])
            results["avg_response_time"] = statistics.mean(sorted_times)
            results["p95_response_time"] = sorted_times[int(len(sorted_times) * 0.95)]
            results["p99_response_time"] = sorted_times[int(len(sorted_times) * 0.99)]
        
        return results
        
    async def analyze_database_performance(self) -> Dict:
        """Analyze database query performance"""
        logger.info("\nAnalyzing database performance...")
        
        analysis = {
            "slow_queries": [],
            "index_usage": {},
            "connection_stats": {},
            "table_sizes": {}
        }
        
        try:
            # Get slow queries
            slow_queries = await self.pg_conn.fetch("""
                SELECT query, calls, mean_exec_time, max_exec_time
                FROM pg_stat_statements
                WHERE mean_exec_time > 10
                ORDER BY mean_exec_time DESC
                LIMIT 10
            """)
            analysis["slow_queries"] = [dict(q) for q in slow_queries]
        except:
            pass
            
        try:
            # Get table sizes
            table_sizes = await self.pg_conn.fetch("""
                SELECT 
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
                FROM pg_tables
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
            """)
            analysis["table_sizes"] = [dict(t) for t in table_sizes]
        except:
            pass
            
        try:
            # Get index usage
            index_usage = await self.pg_conn.fetch("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan,
                    idx_tup_read,
                    idx_tup_fetch
                FROM pg_stat_user_indexes
                ORDER BY idx_scan DESC
            """)
            analysis["index_usage"] = [dict(i) for i in index_usage]
        except:
            pass
            
        return analysis
        
    async def generate_optimization_report(self) -> Dict:
        """Generate comprehensive optimization recommendations"""
        logger.info("\nGenerating optimization report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics_summary": {},
            "bottlenecks": [],
            "optimizations": [],
            "performance_score": 0
        }
        
        if self.metrics:
            # Calculate summary statistics
            avg_response = statistics.mean([m.avg_response_time for m in self.metrics])
            avg_rps = statistics.mean([m.requests_per_second for m in self.metrics])
            avg_error_rate = statistics.mean([m.error_rate for m in self.metrics])
            avg_cache_hit = statistics.mean([m.cache_hit_rate for m in self.metrics])
            
            report["metrics_summary"] = {
                "avg_response_time_ms": round(avg_response, 2),
                "avg_requests_per_second": round(avg_rps, 2),
                "avg_error_rate_percent": round(avg_error_rate, 2),
                "avg_cache_hit_rate_percent": round(avg_cache_hit, 2)
            }
            
            # Identify bottlenecks
            if avg_response > 100:
                report["bottlenecks"].append({
                    "issue": "High average response time",
                    "current": f"{avg_response:.2f}ms",
                    "target": "< 100ms",
                    "impact": "HIGH"
                })
                
            if avg_cache_hit < 80:
                report["bottlenecks"].append({
                    "issue": "Low cache hit rate",
                    "current": f"{avg_cache_hit:.2f}%",
                    "target": "> 80%",
                    "impact": "MEDIUM"
                })
                
            if avg_error_rate > 1:
                report["bottlenecks"].append({
                    "issue": "High error rate",
                    "current": f"{avg_error_rate:.2f}%",
                    "target": "< 1%",
                    "impact": "CRITICAL"
                })
                
            # Generate optimizations
            if avg_response > 100:
                report["optimizations"].append({
                    "area": "Response Time",
                    "recommendation": "Implement database query optimization and connection pooling",
                    "expected_improvement": "50-70% reduction in response time",
                    "priority": "HIGH"
                })
                
            if avg_cache_hit < 80:
                report["optimizations"].append({
                    "area": "Caching",
                    "recommendation": "Implement aggressive Redis caching strategy",
                    "expected_improvement": "80%+ cache hit rate",
                    "priority": "HIGH"
                })
                
            # Calculate performance score
            score = 100
            score -= min(50, avg_response / 10)  # Penalize high response time
            score -= min(20, (100 - avg_cache_hit) / 5)  # Penalize low cache hit
            score -= min(30, avg_error_rate * 10)  # Penalize errors
            report["performance_score"] = max(0, round(score))
            
        return report
        
    async def run_complete_analysis(self):
        """Run complete ULTRAPERFORMANCE analysis"""
        logger.info("=" * 80)
        logger.info("ULTRAPERFORMANCE Analysis Starting...")
        logger.info("=" * 80)
        
        await self.setup()
        
        try:
            # Test critical endpoints
            critical_endpoints = [
                ("/health", "GET"),
                ("/api/v1/chat/", "POST"),
                ("/api/v1/models/", "GET"),
                ("/api/v1/mesh/results", "GET"),
                ("/metrics", "GET")
            ]
            
            for endpoint, method in critical_endpoints:
                payload = {"message": "test", "model": "tinyllama"} if method == "POST" else None
                metrics = await self.measure_endpoint(endpoint, method, payload)
                logger.info(f"\n{endpoint}:")
                logger.info(f"  Avg Response: {metrics.avg_response_time:.2f}ms")
                logger.info(f"  P95 Response: {metrics.p95_response_time:.2f}ms")
                logger.info(f"  RPS: {metrics.requests_per_second:.2f}")
                logger.error(f"  Error Rate: {metrics.error_rate:.2f}%")
                
            # Load testing
            load_results = await self.load_test_concurrent("/health", 50, 10)
            logger.info(f"\nLoad Test Results:")
            logger.info(f"  Total Requests: {load_results['total_requests']}")
            logger.info(f"  Throughput: {load_results['throughput']:.2f} req/s")
            logger.info(f"  Avg Response: {load_results.get('avg_response_time', 0):.2f}ms")
            
            # Database analysis
            db_analysis = await self.analyze_database_performance()
            
            # Generate report
            report = await self.generate_optimization_report()
            report["database_analysis"] = db_analysis
            report["load_test_results"] = load_results
            
            # Save report
            report_file = f"/opt/sutazaiapp/reports/performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2, default=str)
                
            logger.info(f"\n{'=' * 80}")
            logger.info(f"PERFORMANCE SCORE: {report['performance_score']}/100")
            logger.info(f"Report saved to: {report_file}")
            logger.info(f"{'=' * 80}")
            
            return report
            
        finally:
            await self.cleanup()

async def main():
    analyzer = UltraPerformanceAnalyzer()
    report = await analyzer.run_complete_analysis()
    
    # Print key findings
    logger.info("\nKEY FINDINGS:")
    for bottleneck in report.get("bottlenecks", []):
        logger.info(f"  ⚠️  {bottleneck['issue']}: {bottleneck['current']} (target: {bottleneck['target']})")
        
    logger.info("\nRECOMMENDED OPTIMIZATIONS:")
    for opt in report.get("optimizations", []):
        logger.info(f"  ✅ {opt['area']}: {opt['recommendation']}")
        logger.info(f"     Expected: {opt['expected_improvement']}")

if __name__ == "__main__":
    asyncio.run(main())