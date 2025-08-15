#!/usr/bin/env python3
"""
API Health Monitor - Real-time monitoring of critical API endpoints
"""
import asyncio
import httpx
import time
import sys
from datetime import datetime
from typing import Dict, List, Optional

class APIHealthMonitor:
    """Monitor API endpoints for health and performance"""
    
    def __init__(self, base_url: str = "http://localhost:10010"):
        self.base_url = base_url
        self.endpoints = [
            ("/health", "GET", None, "Core health check"),
            ("/metrics", "GET", None, "Prometheus metrics"),
            ("/api/v1/agents", "GET", None, "Agent listing"),
            ("/api/v1/status", "GET", None, "System status"),
            ("/api/v1/chat", "POST", {"message": "test", "model": "tinyllama"}, "Chat endpoint"),
            ("/docs", "GET", None, "API documentation"),
        ]
        self.stats = {
            "total_checks": 0,
            "failures": 0,
            "timeouts": 0,
            "success_rate": 0.0
        }
        
    async def check_endpoint(self, path: str, method: str = "GET", 
                            data: Optional[Dict] = None, 
                            description: str = "") -> Dict:
        """Check single endpoint health"""
        url = f"{self.base_url}{path}"
        
        try:
            start = time.time()
            async with httpx.AsyncClient() as client:
                if method == "GET":
                    response = await client.get(url, timeout=5.0)
                elif method == "POST":
                    response = await client.post(url, json=data, timeout=5.0)
                else:
                    raise ValueError(f"Unsupported method: {method}")
                    
                elapsed = (time.time() - start) * 1000
                
                return {
                    "path": path,
                    "description": description,
                    "method": method,
                    "status": response.status_code,
                    "response_time_ms": round(elapsed, 2),
                    "healthy": 200 <= response.status_code < 300,
                    "size_bytes": len(response.content)
                }
        except httpx.TimeoutException:
            self.stats["timeouts"] += 1
            return {
                "path": path,
                "description": description,
                "method": method,
                "status": 0,
                "error": "TIMEOUT (5s)",
                "healthy": False
            }
        except Exception as e:
            self.stats["failures"] += 1
            return {
                "path": path,
                "description": description,
                "method": method,
                "status": 0,
                "error": str(e)[:50],
                "healthy": False
            }
    
    def print_status(self, results: List[Dict]):
        """Print formatted status report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        healthy = sum(1 for r in results if r["healthy"])
        total = len(results)
        
        # Update stats
        self.stats["total_checks"] += total
        self.stats["success_rate"] = (healthy / total * 100) if total > 0 else 0
        
        # Clear screen for clean output
        print("\033[2J\033[H")  # Clear screen and move cursor to top
        
        print("=" * 80)
        print(f"SutazAI API Health Monitor - {timestamp}")
        print("=" * 80)
        print(f"\n📊 Overall Health: {healthy}/{total} endpoints healthy ({self.stats['success_rate']:.1f}%)")
        print(f"📈 Total Checks: {self.stats['total_checks']} | Timeouts: {self.stats['timeouts']} | Failures: {self.stats['failures']}")
        print("\n" + "-" * 80)
        print(f"{'Status':<8} {'Method':<7} {'Time(ms)':<10} {'Path':<30} {'Description':<25}")
        print("-" * 80)
        
        # Sort by health status (unhealthy first)
        results.sort(key=lambda x: (x["healthy"], x.get("response_time_ms", 9999)))
        
        for result in results:
            status_icon = "✅" if result["healthy"] else "❌"
            method = result["method"]
            path = result["path"][:28]
            desc = result["description"][:23]
            
            if result["healthy"]:
                time_str = f"{result['response_time_ms']:.1f}ms"
                print(f"{status_icon:<8} {method:<7} {time_str:<10} {path:<30} {desc:<25}")
            else:
                error = result.get("error", f"HTTP {result['status']}")[:20]
                print(f"{status_icon:<8} {method:<7} {'ERROR':<10} {path:<30} {error:<25}")
        
        print("-" * 80)
        
        # Performance summary
        healthy_results = [r for r in results if r["healthy"]]
        if healthy_results:
            avg_time = sum(r["response_time_ms"] for r in healthy_results) / len(healthy_results)
            min_time = min(r["response_time_ms"] for r in healthy_results)
            max_time = max(r["response_time_ms"] for r in healthy_results)
            
            print(f"\n⚡ Performance: Avg: {avg_time:.1f}ms | Min: {min_time:.1f}ms | Max: {max_time:.1f}ms")
        
        # Alerts
        if self.stats["timeouts"] > 5:
            print("\n⚠️  WARNING: High timeout rate detected!")
        if self.stats["success_rate"] < 50:
            print("\n🚨 CRITICAL: API health below 50%!")
            
    async def monitor_continuous(self, interval: int = 10):
        """Run continuous monitoring"""
        print("Starting API Health Monitor...")
        print(f"Monitoring {self.base_url}")
        print(f"Check interval: {interval} seconds")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Check all endpoints in parallel
                tasks = [
                    self.check_endpoint(path, method, data, desc)
                    for path, method, data, desc in self.endpoints
                ]
                results = await asyncio.gather(*tasks)
                
                # Display results
                self.print_status(results)
                
                # Check for critical issues
                if all(not r["healthy"] for r in results):
                    print("\n🔴 CRITICAL: ALL ENDPOINTS DOWN! Backend may be crashed.")
                    print("   Run: docker logs sutazai-backend --tail 50")
                    print("   Or:  docker restart sutazai-backend")
                
                # Wait for next check
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            print(f"Final stats: {self.stats}")
            
    async def quick_check(self):
        """Run a single health check"""
        tasks = [
            self.check_endpoint(path, method, data, desc)
            for path, method, data, desc in self.endpoints
        ]
        results = await asyncio.gather(*tasks)
        self.print_status(results)
        
        # Return exit code based on health
        healthy = sum(1 for r in results if r["healthy"])
        return 0 if healthy > len(results) / 2 else 1

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor SutazAI API health")
    parser.add_argument("--url", default="http://localhost:10010", 
                       help="Base URL of the API (default: http://localhost:10010)")
    parser.add_argument("--interval", type=int, default=10,
                       help="Check interval in seconds (default: 10)")
    parser.add_argument("--once", action="store_true",
                       help="Run single check and exit")
    
    args = parser.parse_args()
    
    monitor = APIHealthMonitor(args.url)
    
    if args.once:
        exit_code = asyncio.run(monitor.quick_check())
        sys.exit(exit_code)
    else:
        try:
            asyncio.run(monitor.monitor_continuous(args.interval))
        except KeyboardInterrupt:
            sys.exit(0)

if __name__ == "__main__":
    main()