#!/usr/bin/env python3
"""
MCP Server Latency and Throughput Testing

Detailed performance testing focusing on message latency and throughput
for working MCP servers.
"""

import asyncio
import json
import logging
import statistics
import time
from test_mcp_servers import MCPServer, MCPServerProcess, MCPProtocolTester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LatencyTester:
    def __init__(self):
        self.protocol_tester = MCPProtocolTester()
    
    async def test_server_latency(self, server_name, command, args):
        """Test latency for a specific server"""
        server = MCPServer(
            name=server_name,
            command=command,
            args=args,
            server_type="stdio",
            description=f"Latency test for {server_name}"
        )
        
        server_process = MCPServerProcess(server)
        latencies = []
        errors = 0
        
        try:
            # Start server
            if not await server_process.start():
                return {"error": "Failed to start server"}
            
            # Initialize
            init_request = self.protocol_tester.create_initialize_request()
            init_response = await server_process.send_message(init_request, timeout=5)
            
            if not init_response or self.protocol_tester.is_error_response(init_response):
                # Even if init fails, try ping tests
                logger.warning(f"{server_name}: Init failed, trying ping anyway")
            
            # Run latency tests
            for i in range(20):  # 20 ping tests
                start_time = time.perf_counter()
                
                ping_request = self.protocol_tester.create_ping_request(request_id=i+100)
                response = await server_process.send_message(ping_request, timeout=3)
                
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                
                if response and not self.protocol_tester.is_error_response(response):
                    latencies.append(latency_ms)
                else:
                    errors += 1
                    # Still record the time it took to fail
                    latencies.append(latency_ms)
                
                # Small delay between tests
                await asyncio.sleep(0.01)
            
            # Calculate statistics
            if latencies:
                return {
                    "server_name": server_name,
                    "latencies": latencies,
                    "avg_latency": statistics.mean(latencies),
                    "min_latency": min(latencies),
                    "max_latency": max(latencies),
                    "p50_latency": statistics.median(latencies),
                    "p95_latency": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 5 else max(latencies),
                    "success_rate": (len(latencies) - errors) / len(latencies) * 100,
                    "total_tests": len(latencies),
                    "errors": errors
                }
            else:
                return {"error": "No latency measurements collected"}
                
        except Exception as e:
            return {"error": f"Test failed: {str(e)}"}
        finally:
            server_process.stop()
    
    async def test_throughput(self, server_name, command, args):
        """Test message throughput"""
        server = MCPServer(
            name=server_name,
            command=command,
            args=args,
            server_type="stdio", 
            description=f"Throughput test for {server_name}"
        )
        
        server_process = MCPServerProcess(server)
        
        try:
            if not await server_process.start():
                return {"error": "Failed to start server"}
            
            # Initialize
            init_request = self.protocol_tester.create_initialize_request()
            await server_process.send_message(init_request, timeout=5)
            
            # Throughput test - send messages concurrently
            batch_size = 10
            start_time = time.perf_counter()
            
            # Create batch of ping requests
            tasks = []
            for i in range(batch_size):
                ping_request = self.protocol_tester.create_ping_request(request_id=i+200)
                task = asyncio.create_task(server_process.send_message(ping_request, timeout=5))
                tasks.append(task)
            
            # Execute all requests
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # Count successful responses
            successful = 0
            for response in responses:
                if not isinstance(response, Exception) and response and not self.protocol_tester.is_error_response(response):
                    successful += 1
            
            throughput = successful / duration if duration > 0 else 0
            
            return {
                "server_name": server_name,
                "batch_size": batch_size,
                "successful_messages": successful,
                "total_messages": batch_size,
                "duration_seconds": duration,
                "throughput_msg_per_sec": throughput,
                "success_rate": successful / batch_size * 100
            }
            
        except Exception as e:
            return {"error": f"Throughput test failed: {str(e)}"}
        finally:
            server_process.stop()

async def main():
    """Run latency and throughput tests"""
    print("⚡ MCP Server Latency & Throughput Testing")
    print("=" * 50)
    
    # Test configuration for working servers
    test_servers = [
        ("extended-memory", "/opt/sutazaiapp/scripts/mcp/wrappers/extended-memory.sh", []),
        ("files", "/opt/sutazaiapp/scripts/mcp/wrappers/files.sh", []),
        ("context7", "/opt/sutazaiapp/scripts/mcp/wrappers/context7.sh", []),
        ("http_fetch", "/opt/sutazaiapp/scripts/mcp/wrappers/http_fetch.sh", []),
        ("sequentialthinking", "/opt/sutazaiapp/scripts/mcp/wrappers/sequentialthinking.sh", []),
    ]
    
    tester = LatencyTester()
    
    print("\n📊 Latency Testing")
    print("-" * 20)
    
    latency_results = []
    for server_name, command, args in test_servers:
        print(f"Testing {server_name}...")
        result = await tester.test_server_latency(server_name, command, args)
        latency_results.append(result)
        
        if "error" not in result:
            print(f"  ✅ Avg: {result['avg_latency']:.1f}ms, P95: {result['p95_latency']:.1f}ms, Success: {result['success_rate']:.0f}%")
        else:
            print(f"  ❌ Error: {result['error']}")
    
    print(f"\n🚀 Throughput Testing")
    print("-" * 22)
    
    throughput_results = []
    for server_name, command, args in test_servers:
        print(f"Testing {server_name}...")
        result = await tester.test_throughput(server_name, command, args)
        throughput_results.append(result)
        
        if "error" not in result:
            print(f"  ✅ Throughput: {result['throughput_msg_per_sec']:.1f} msg/sec, Success: {result['success_rate']:.0f}%")
        else:
            print(f"  ❌ Error: {result['error']}")
    
    # Generate summary
    print(f"\n📈 Performance Summary")
    print("-" * 25)
    
    # Best latency
    valid_latency = [r for r in latency_results if "error" not in r]
    if valid_latency:
        best_latency = min(valid_latency, key=lambda x: x['avg_latency'])
        print(f"🥇 Best Latency: {best_latency['server_name']} ({best_latency['avg_latency']:.1f}ms avg)")
        
        # Best throughput  
        valid_throughput = [r for r in throughput_results if "error" not in r]
        if valid_throughput:
            best_throughput = max(valid_throughput, key=lambda x: x['throughput_msg_per_sec'])
            print(f"🥇 Best Throughput: {best_throughput['server_name']} ({best_throughput['throughput_msg_per_sec']:.1f} msg/sec)")
        
        # Reliability ranking
        reliable_servers = sorted(valid_latency, key=lambda x: x['success_rate'], reverse=True)
        print(f"🥇 Most Reliable: {reliable_servers[0]['server_name']} ({reliable_servers[0]['success_rate']:.0f}% success)")
    
    # Save detailed results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_data = {
        "timestamp": time.time(),
        "date": time.strftime('%Y-%m-%d %H:%M:%S'),
        "latency_results": latency_results,
        "throughput_results": throughput_results
    }
    
    import os
    os.makedirs("/opt/sutazaiapp/mcp-manager/benchmark_results", exist_ok=True)
    
    with open(f"/opt/sutazaiapp/mcp-manager/benchmark_results/latency_test_{timestamp}.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 Detailed results saved to latency_test_{timestamp}.json")

if __name__ == "__main__":
    asyncio.run(main())