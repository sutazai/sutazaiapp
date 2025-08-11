#!/usr/bin/env python3
"""
Quick Performance Test - Measure current system performance
"""

import time
import requests
import json
import statistics
from datetime import datetime
import concurrent.futures
import subprocess

def test_endpoint(url, method="GET", payload=None, iterations=50):
    """Test endpoint performance"""
    response_times = []
    errors = 0
    
    for i in range(iterations):
        try:
            start = time.time()
            
            if method == "GET":
                response = requests.get(url, timeout=5)
            elif method == "POST":
                response = requests.post(url, json=payload or {}, timeout=5)
                
            elapsed = (time.time() - start) * 1000  # ms
            response_times.append(elapsed)
            
            if response.status_code >= 400:
                errors += 1
        except Exception as e:
            errors += 1
            print(f"Error: {e}")
            
    if response_times:
        return {
            "url": url,
            "method": method,
            "iterations": iterations,
            "avg_response_ms": statistics.mean(response_times),
            "min_response_ms": min(response_times),
            "max_response_ms": max(response_times),
            "p95_response_ms": sorted(response_times)[int(len(response_times) * 0.95)],
            "error_rate": (errors / iterations) * 100
        }
    else:
        return {"url": url, "error": "All requests failed"}

def check_redis_performance():
    """Check Redis cache performance"""
    try:
        # Check Redis stats
        result = subprocess.run(
            ["docker", "exec", "sutazai-redis", "redis-cli", "INFO", "STATS"],
            capture_output=True, text=True
        )
        
        stats = {}
        for line in result.stdout.split('\n'):
            if 'keyspace_hits' in line:
                stats['hits'] = int(line.split(':')[1])
            elif 'keyspace_misses' in line:
                stats['misses'] = int(line.split(':')[1])
                
        total = stats.get('hits', 0) + stats.get('misses', 0)
        hit_rate = (stats.get('hits', 0) / total * 100) if total > 0 else 0
        
        return {
            "cache_hits": stats.get('hits', 0),
            "cache_misses": stats.get('misses', 0),
            "cache_hit_rate": round(hit_rate, 2)
        }
    except Exception as e:
        return {"error": str(e)}

def check_database_performance():
    """Check database query performance"""
    try:
        # Simple query performance test
        cmd = """docker exec sutazai-postgres psql -U sutazai -c "EXPLAIN ANALYZE SELECT 1;" """
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Extract execution time
        for line in result.stdout.split('\n'):
            if 'Execution Time' in line:
                exec_time = float(line.split(':')[1].strip().replace(' ms', ''))
                return {"query_time_ms": exec_time}
                
        return {"query_time_ms": "unknown"}
    except Exception as e:
        return {"error": str(e)}

def concurrent_load_test(url, users=20, duration=10):
    """Run concurrent load test"""
    print(f"\nLoad testing {url} with {users} concurrent users for {duration}s...")
    
    def worker():
        results = []
        end_time = time.time() + duration
        
        while time.time() < end_time:
            try:
                start = time.time()
                response = requests.get(url, timeout=5)
                elapsed = (time.time() - start) * 1000
                results.append({
                    "time": elapsed,
                    "status": response.status_code
                })
            except:
                results.append({"time": 5000, "status": 500})
                
        return results
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=users) as executor:
        futures = [executor.submit(worker) for _ in range(users)]
        all_results = []
        
        for future in concurrent.futures.as_completed(futures):
            all_results.extend(future.result())
            
    # Calculate statistics
    response_times = [r["time"] for r in all_results]
    success_count = sum(1 for r in all_results if r["status"] < 400)
    
    return {
        "total_requests": len(all_results),
        "successful_requests": success_count,
        "avg_response_ms": statistics.mean(response_times) if response_times else 0,
        "max_response_ms": max(response_times) if response_times else 0,
        "requests_per_second": len(all_results) / duration,
        "success_rate": (success_count / len(all_results) * 100) if all_results else 0
    }

def main():
    print("=" * 80)
    print("ULTRAPERFORMANCE QUICK ANALYSIS")
    print("=" * 80)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "endpoints": [],
        "redis": {},
        "database": {},
        "load_test": {},
        "bottlenecks": [],
        "optimizations": []
    }
    
    # Test endpoints
    endpoints = [
        ("http://localhost:10010/health", "GET"),
        ("http://localhost:10010/api/v1/models/", "GET"),
        ("http://localhost:11110/health", "GET"),  # Hardware optimizer
        ("http://localhost:8589/health", "GET"),   # AI Agent Orchestrator
        ("http://localhost:8090/health", "GET"),   # Ollama Integration
    ]
    
    print("\n1. ENDPOINT PERFORMANCE TESTS")
    print("-" * 40)
    
    for url, method in endpoints:
        print(f"Testing {url}...")
        result = test_endpoint(url, method)
        results["endpoints"].append(result)
        
        if "avg_response_ms" in result:
            print(f"  Avg: {result['avg_response_ms']:.2f}ms")
            print(f"  P95: {result['p95_response_ms']:.2f}ms")
            print(f"  Errors: {result['error_rate']:.2f}%")
            
            # Identify bottlenecks
            if result['avg_response_ms'] > 100:
                results["bottlenecks"].append({
                    "type": "slow_endpoint",
                    "endpoint": url,
                    "current": f"{result['avg_response_ms']:.2f}ms",
                    "target": "< 100ms"
                })
    
    print("\n2. REDIS CACHE PERFORMANCE")
    print("-" * 40)
    redis_stats = check_redis_performance()
    results["redis"] = redis_stats
    print(f"Cache Hit Rate: {redis_stats.get('cache_hit_rate', 'N/A')}%")
    
    if redis_stats.get('cache_hit_rate', 0) < 80:
        results["bottlenecks"].append({
            "type": "low_cache_hit_rate",
            "current": f"{redis_stats.get('cache_hit_rate', 0)}%",
            "target": "> 80%"
        })
        results["optimizations"].append({
            "area": "Redis Caching",
            "action": "Implement aggressive caching strategy",
            "expected_improvement": "80%+ cache hit rate"
        })
    
    print("\n3. DATABASE PERFORMANCE")
    print("-" * 40)
    db_stats = check_database_performance()
    results["database"] = db_stats
    print(f"Query Time: {db_stats.get('query_time_ms', 'N/A')}ms")
    
    print("\n4. LOAD TEST (Backend API)")
    print("-" * 40)
    load_results = concurrent_load_test("http://localhost:10010/health", users=30, duration=10)
    results["load_test"] = load_results
    print(f"Total Requests: {load_results['total_requests']}")
    print(f"Requests/Second: {load_results['requests_per_second']:.2f}")
    print(f"Avg Response: {load_results['avg_response_ms']:.2f}ms")
    print(f"Success Rate: {load_results['success_rate']:.2f}%")
    
    # Calculate performance score
    score = 100
    avg_response = statistics.mean([e.get('avg_response_ms', 0) for e in results['endpoints'] if 'avg_response_ms' in e])
    
    if avg_response > 100:
        score -= min(30, avg_response / 10)
    if redis_stats.get('cache_hit_rate', 0) < 80:
        score -= 20
    if load_results['success_rate'] < 99:
        score -= 20
        
    results["performance_score"] = max(0, round(score))
    
    # Generate optimizations
    if avg_response > 100:
        results["optimizations"].append({
            "area": "Response Time",
            "action": "Implement connection pooling and query optimization",
            "expected_improvement": "50-70% faster response times"
        })
        
    if load_results['requests_per_second'] < 100:
        results["optimizations"].append({
            "area": "Throughput",
            "action": "Scale horizontally and optimize backend code",
            "expected_improvement": "2-3x throughput increase"
        })
    
    # Save results
    report_file = f"/opt/sutazaiapp/reports/performance_quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 80)
    print(f"PERFORMANCE SCORE: {results['performance_score']}/100")
    print("=" * 80)
    
    print("\nBOTTLENECKS IDENTIFIED:")
    for bottleneck in results["bottlenecks"]:
        print(f"  ⚠️  {bottleneck['type']}: {bottleneck.get('current', 'N/A')} (target: {bottleneck.get('target', 'N/A')})")
        
    print("\nRECOMMENDED OPTIMIZATIONS:")
    for opt in results["optimizations"]:
        print(f"  ✅ {opt['area']}: {opt['action']}")
        print(f"     Expected: {opt['expected_improvement']}")
        
    print(f"\nFull report saved to: {report_file}")
    
    return results

if __name__ == "__main__":
    main()