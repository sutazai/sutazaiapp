#!/usr/bin/env python3
"""
ULTRATEST Graduated Load Testing Suite
Progressive load testing with automatic scaling
"""

import asyncio
import aiohttp
import time
import json
import sys
from typing import Dict, List
import statistics
import traceback

class GraduatedLoadTest:
    """Progressive load testing to find optimal performance"""
    
    def __init__(self):
        self.results = []
        self.critical_endpoints = [
            {"name": "Backend Health", "url": "http://localhost:10010/health", "critical": True},
            {"name": "Frontend", "url": "http://localhost:10011/", "critical": True},
            {"name": "Ollama Integration", "url": "http://localhost:8090/health", "critical": True},
            {"name": "AI Orchestrator", "url": "http://localhost:8589/health", "critical": True},
            {"name": "Hardware Optimizer", "url": "http://localhost:11110/health", "critical": True},
        ]
        
        self.all_endpoints = [
            {"name": "Backend Health", "url": "http://localhost:10010/health"},
            {"name": "Frontend", "url": "http://localhost:10011/"},
            {"name": "Ollama Tags", "url": "http://localhost:10104/api/tags"},
            {"name": "Ollama Integration", "url": "http://localhost:8090/health"},
            {"name": "AI Orchestrator", "url": "http://localhost:8589/health"},
            {"name": "Hardware Optimizer", "url": "http://localhost:11110/health"},
            {"name": "Resource Arbitration", "url": "http://localhost:8588/health"},
            {"name": "Task Assignment", "url": "http://localhost:8551/health"},
            {"name": "Neo4j", "url": "http://localhost:10002/"},
            {"name": "Qdrant", "url": "http://localhost:10101/healthz"},
            {"name": "ChromaDB", "url": "http://localhost:10100/api/v1/heartbeat"},
            {"name": "FAISS", "url": "http://localhost:10103/health"},
            {"name": "Prometheus", "url": "http://localhost:10200/-/healthy"},
            {"name": "Grafana", "url": "http://localhost:10201/api/health"},
            {"name": "Loki", "url": "http://localhost:10202/ready"},
            {"name": "AlertManager", "url": "http://localhost:10203/-/healthy"},
        ]
    
    async def test_single_request(self, session: aiohttp.ClientSession, endpoint: Dict) -> Dict:
        """Test single request to endpoint"""
        start_time = time.time()
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with session.get(endpoint["url"], timeout=timeout) as response:
                response_time = time.time() - start_time
                return {
                    "name": endpoint["name"],
                    "success": response.status == 200,
                    "status_code": response.status,
                    "response_time": response_time,
                    "error": None
                }
        except Exception as e:
            return {
                "name": endpoint["name"],
                "success": False,
                "status_code": -1,
                "response_time": time.time() - start_time,
                "error": str(e)
            }
    
    async def test_load_level(self, endpoints: List[Dict], concurrent_users: int) -> Dict:
        """Test specific load level"""
        print(f"🔄 Testing load level: {concurrent_users} concurrent users...")
        
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create tasks for concurrent requests
            tasks = []
            for i in range(concurrent_users):
                endpoint = endpoints[i % len(endpoints)]
                tasks.append(self.test_single_request(session, endpoint))
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # Process results
            successful_results = []
            failed_results = []
            
            for result in results:
                if isinstance(result, Exception):
                    failed_results.append({"error": str(result)})
                elif result["success"]:
                    successful_results.append(result)
                else:
                    failed_results.append(result)
            
            success_rate = len(successful_results) / len(results) * 100
            avg_response_time = statistics.mean([r["response_time"] for r in successful_results]) if successful_results else 0
            throughput = len(results) / total_time
            
            return {
                "concurrent_users": concurrent_users,
                "total_requests": len(results),
                "successful_requests": len(successful_results),
                "failed_requests": len(failed_results),
                "success_rate": success_rate,
                "avg_response_time": avg_response_time,
                "throughput": throughput,
                "total_time": total_time,
                "sample_failures": failed_results[:3]
            }
    
    async def run_graduated_test(self) -> List[Dict]:
        """Run graduated load test"""
        print("🚀 Starting ULTRATEST Graduated Load Testing")
        print("=" * 60)
        
        # Test load levels: 1, 5, 10, 20, 50, 100, 150 users
        load_levels = [1, 5, 10, 20, 50, 100, 150]
        results = []
        
        for load_level in load_levels:
            try:
                # Use critical endpoints for high load, all endpoints for low load
                endpoints = self.critical_endpoints if load_level > 20 else self.all_endpoints
                
                result = await self.test_load_level(endpoints, load_level)
                results.append(result)
                
                print(f"✅ Load Level {load_level}: {result['success_rate']:.1f}% success, "
                      f"{result['avg_response_time']:.3f}s avg, {result['throughput']:.1f} req/s")
                
                # Stop if success rate drops below 80%
                if result['success_rate'] < 80:
                    print(f"⚠️  Success rate dropped below 80% at {load_level} users - stopping progression")
                    break
                
                # Wait between tests to let system recover
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"❌ Failed at load level {load_level}: {e}")
                break
        
        return results
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze load test results"""
        if not results:
            return {"error": "No results to analyze"}
        
        # Find optimal load level
        optimal_result = None
        max_throughput = 0
        
        for result in results:
            if result['success_rate'] >= 95 and result['throughput'] > max_throughput:
                max_throughput = result['throughput']
                optimal_result = result
        
        # Find breaking point
        breaking_point = None
        for result in results:
            if result['success_rate'] < 90:
                breaking_point = result
                break
        
        # Calculate system capacity
        max_successful_load = max(r['concurrent_users'] for r in results if r['success_rate'] >= 95)
        
        return {
            "total_tests": len(results),
            "max_successful_load": max_successful_load,
            "optimal_performance": optimal_result,
            "breaking_point": breaking_point,
            "system_grade": self.calculate_system_grade(results),
            "detailed_results": results
        }
    
    def calculate_system_grade(self, results: List[Dict]) -> str:
        """Calculate system performance grade"""
        if not results:
            return "F (No Data)"
        
        max_load = max(r['concurrent_users'] for r in results if r['success_rate'] >= 95)
        avg_response_time = statistics.mean([r['avg_response_time'] for r in results if r['success_rate'] >= 95])
        
        if max_load >= 100 and avg_response_time <= 1.0:
            return "A+ (Excellent - Production Ready)"
        elif max_load >= 50 and avg_response_time <= 2.0:
            return "A (Very Good - Ready for Production)"
        elif max_load >= 20 and avg_response_time <= 3.0:
            return "B+ (Good - Minor Optimizations Needed)"
        elif max_load >= 10 and avg_response_time <= 5.0:
            return "B (Satisfactory - Optimization Required)"
        elif max_load >= 5:
            return "C (Needs Improvement)"
        else:
            return "F (Critical Issues - Not Production Ready)"
    
    def print_final_report(self, analysis: Dict):
        """Print comprehensive final report"""
        print("\n" + "=" * 80)
        print("🏆 ULTRATEST GRADUATED LOAD TEST - FINAL REPORT")
        print("=" * 80)
        
        if "error" in analysis:
            print(f"❌ Error: {analysis['error']}")
            return
        
        print(f"📊 Tests Completed: {analysis['total_tests']}")
        print(f"🎯 Maximum Stable Load: {analysis['max_successful_load']} concurrent users")
        print(f"📈 System Grade: {analysis['system_grade']}")
        
        if analysis['optimal_performance']:
            opt = analysis['optimal_performance']
            print(f"\n🚀 OPTIMAL PERFORMANCE:")
            print(f"   • Concurrent Users: {opt['concurrent_users']}")
            print(f"   • Success Rate: {opt['success_rate']:.1f}%")
            print(f"   • Average Response Time: {opt['avg_response_time']:.3f}s")
            print(f"   • Throughput: {opt['throughput']:.1f} requests/second")
        
        if analysis['breaking_point']:
            bp = analysis['breaking_point']
            print(f"\n⚠️  BREAKING POINT:")
            print(f"   • Load Level: {bp['concurrent_users']} users")
            print(f"   • Success Rate: {bp['success_rate']:.1f}%")
            print(f"   • Response Time: {bp['avg_response_time']:.3f}s")
        
        print(f"\n📋 DETAILED RESULTS:")
        for result in analysis['detailed_results']:
            status = "✅" if result['success_rate'] >= 95 else "⚠️" if result['success_rate'] >= 80 else "❌"
            print(f"   {status} {result['concurrent_users']:3d} users: "
                  f"{result['success_rate']:5.1f}% success, "
                  f"{result['avg_response_time']:5.3f}s avg, "
                  f"{result['throughput']:5.1f} req/s")
        
        # Final determination
        print("\n" + "=" * 80)
        if analysis['max_successful_load'] >= 100:
            print("🏆 ULTRATEST FINAL RESULT: SYSTEM READY FOR PRODUCTION")
            print("   ✅ Can handle 100+ concurrent users")
            print("   ✅ Meets enterprise performance standards")
        elif analysis['max_successful_load'] >= 50:
            print("✅ ULTRATEST FINAL RESULT: SYSTEM READY WITH MONITORING")
            print("   ⚠️  Monitor performance under high load")
            print("   ✅ Suitable for most production workloads")
        elif analysis['max_successful_load'] >= 20:
            print("⚠️  ULTRATEST FINAL RESULT: OPTIMIZATION NEEDED")
            print("   🔧 Performance tuning recommended")
            print("   ⚠️  Limited concurrent user capacity")
        else:
            print("🚨 ULTRATEST FINAL RESULT: CRITICAL PERFORMANCE ISSUES")
            print("   🚨 System not ready for production")
            print("   🔧 Immediate optimization required")
        print("=" * 80)

async def main():
    """Main execution function"""
    tester = GraduatedLoadTest()
    
    try:
        # Wait for system to be ready
        print("⏳ Waiting for system to stabilize...")
        await asyncio.sleep(5)
        
        results = await tester.run_graduated_test()
        analysis = tester.analyze_results(results)
        
        # Save report
        timestamp = int(time.time())
        report_file = f"/opt/sutazaiapp/tests/ultratest_graduated_report_{timestamp}.json"
        with open(report_file, "w") as f:
            json.dump(analysis, f, indent=2)
        
        tester.print_final_report(analysis)
        print(f"\n📄 Detailed report saved: {report_file}")
        
        # Return appropriate exit code
        if analysis.get('max_successful_load', 0) >= 50:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Needs improvement
            
    except Exception as e:
        print(f"🚨 ULTRATEST CRITICAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    asyncio.run(main())