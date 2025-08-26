#!/usr/bin/env python3
"""
Focused MCP Server Performance Benchmarking

Quick performance testing focused on working MCP servers with reduced test scope.
Tests startup time, basic message latency, and identifies bottlenecks.

Author: Claude Code Performance Benchmarker
Created: 2025-08-26
"""

import asyncio
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import our existing test infrastructure
from test_mcp_servers import (
    MCPServer, MCPServerProcess, MCPProtocolTester,
    load_mcp_servers
)

# Focused test configuration
FOCUSED_ITERATIONS = 20
QUICK_TIMEOUT = 10  # seconds
LATENCY_TEST_COUNT = 50

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QuickMetrics:
    """Quick performance metrics"""
    startup_time_ms: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    success_rate: float = 0.0
    error_count: int = 0
    working: bool = False


@dataclass
class QuickBenchmarkResult:
    """Quick benchmark result"""
    server_name: str
    metrics: QuickMetrics
    error_messages: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class FocusedBenchmarker:
    """Focused MCP server benchmarking"""
    
    def __init__(self):
        self.protocol_tester = MCPProtocolTester()
    
    async def quick_startup_test(self, server: MCPServer) -> QuickBenchmarkResult:
        """Quick startup performance test"""
        logger.info(f"⚡ Quick test for {server.name}")
        
        startup_times = []
        errors = []
        
        # Test startup 5 times
        for i in range(5):
            server_process = MCPServerProcess(server)
            start_time = time.perf_counter()
            
            try:
                started = await server_process.start()
                startup_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
                
                if started:
                    startup_times.append(startup_time)
                    
                    # Quick initialization test
                    init_request = self.protocol_tester.create_initialize_request()
                    response = await server_process.send_message(init_request, timeout=5)
                    
                    if not response or self.protocol_tester.is_error_response(response):
                        errors.append(f"Initialization failed on attempt {i+1}")
                else:
                    errors.append(f"Startup failed on attempt {i+1}")
                    
            except Exception as e:
                errors.append(f"Error in attempt {i+1}: {str(e)}")
            finally:
                server_process.stop()
                await asyncio.sleep(0.2)  # Brief pause
        
        if not startup_times:
            return QuickBenchmarkResult(
                server_name=server.name,
                metrics=QuickMetrics(working=False, error_count=len(errors)),
                error_messages=errors
            )
        
        # Quick latency test on working server
        latencies = []
        if startup_times:
            server_process = MCPServerProcess(server)
            try:
                if await server_process.start():
                    # Initialize
                    init_request = self.protocol_tester.create_initialize_request()
                    init_response = await server_process.send_message(init_request, timeout=5)
                    
                    if init_response and not self.protocol_tester.is_error_response(init_response):
                        # Quick latency measurements (10 pings)
                        for i in range(10):
                            start_time = time.perf_counter()
                            ping_request = self.protocol_tester.create_ping_request(request_id=i+100)
                            response = await server_process.send_message(ping_request, timeout=3)
                            latency_ms = (time.perf_counter() - start_time) * 1000
                            
                            if response and not self.protocol_tester.is_error_response(response):
                                latencies.append(latency_ms)
                            else:
                                errors.append(f"Ping {i+1} failed")
            except Exception as e:
                errors.append(f"Latency test error: {str(e)}")
            finally:
                server_process.stop()
        
        # Calculate metrics
        avg_startup = statistics.mean(startup_times) if startup_times else 0
        avg_latency = statistics.mean(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        success_rate = len(latencies) / 10 * 100 if latencies else 0
        
        metrics = QuickMetrics(
            startup_time_ms=avg_startup,
            avg_latency_ms=avg_latency,
            max_latency_ms=max_latency,
            success_rate=success_rate,
            error_count=len(errors),
            working=len(startup_times) > 0 and success_rate > 50
        )
        
        return QuickBenchmarkResult(
            server_name=server.name,
            metrics=metrics,
            error_messages=errors[:3]  # First 3 errors only
        )
    
    async def run_focused_benchmarks(self, servers: List[MCPServer], max_servers: int = 8) -> List[QuickBenchmarkResult]:
        """Run focused benchmarks on selected servers"""
        logger.info(f"🎯 Running focused benchmarks on up to {max_servers} servers")
        
        # Prioritize known working servers (wrapper scripts that passed health checks)
        working_servers = [s for s in servers if s.command.endswith('.sh')][:max_servers//2]
        other_servers = [s for s in servers if not s.command.endswith('.sh')][:max_servers//2]
        selected_servers = working_servers + other_servers
        
        results = []
        for i, server in enumerate(selected_servers):
            logger.info(f"🧪 Testing {i+1}/{len(selected_servers)}: {server.name}")
            
            try:
                result = await self.quick_startup_test(server)
                results.append(result)
                
                if result.metrics.working:
                    logger.info(f"✅ {server.name} - Working (startup: {result.metrics.startup_time_ms:.0f}ms)")
                else:
                    logger.info(f"❌ {server.name} - Not responding")
                    
            except Exception as e:
                logger.error(f"❌ Error testing {server.name}: {str(e)}")
                error_result = QuickBenchmarkResult(
                    server_name=server.name,
                    metrics=QuickMetrics(working=False, error_count=1),
                    error_messages=[f"Test failed: {str(e)}"]
                )
                results.append(error_result)
            
            await asyncio.sleep(1)  # Brief pause between servers
        
        return results


def generate_focused_report(results: List[QuickBenchmarkResult]) -> str:
    """Generate focused performance report"""
    report = []
    report.append("🎯 Focused MCP Server Performance Report")
    report.append("=" * 50)
    report.append(f"📊 Tested {len(results)} servers")
    report.append(f"🕒 Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Working servers
    working_servers = [r for r in results if r.metrics.working]
    report.append(f"✅ Working Servers: {len(working_servers)}/{len(results)}")
    report.append("-" * 30)
    
    if working_servers:
        # Sort by startup time (fastest first)
        working_servers.sort(key=lambda x: x.metrics.startup_time_ms)
        
        for result in working_servers:
            report.append(f"  {result.server_name}:")
            report.append(f"    Startup: {result.metrics.startup_time_ms:.0f}ms")
            report.append(f"    Latency: {result.metrics.avg_latency_ms:.1f}ms avg, {result.metrics.max_latency_ms:.1f}ms max")
            report.append(f"    Success Rate: {result.metrics.success_rate:.0f}%")
            if result.metrics.error_count > 0:
                report.append(f"    Errors: {result.metrics.error_count}")
            report.append("")
    
    # Non-working servers
    non_working = [r for r in results if not r.metrics.working]
    if non_working:
        report.append(f"❌ Non-Working Servers: {len(non_working)}")
        report.append("-" * 25)
        for result in non_working:
            report.append(f"  {result.server_name}: {result.metrics.error_count} errors")
            if result.error_messages:
                report.append(f"    Error: {result.error_messages[0]}")
        report.append("")
    
    # Performance Rankings
    if working_servers:
        report.append("🏆 Performance Rankings")
        report.append("-" * 20)
        
        # Fastest startup
        fastest_startup = min(working_servers, key=lambda x: x.metrics.startup_time_ms)
        report.append(f"⚡ Fastest Startup: {fastest_startup.server_name} ({fastest_startup.metrics.startup_time_ms:.0f}ms)")
        
        # Lowest latency
        lowest_latency = min(working_servers, key=lambda x: x.metrics.avg_latency_ms)
        report.append(f"🚀 Lowest Latency: {lowest_latency.server_name} ({lowest_latency.metrics.avg_latency_ms:.1f}ms)")
        
        # Most reliable
        most_reliable = max(working_servers, key=lambda x: x.metrics.success_rate)
        report.append(f"🎯 Most Reliable: {most_reliable.server_name} ({most_reliable.metrics.success_rate:.0f}%)")
        report.append("")
    
    # Recommendations
    report.append("💡 Quick Recommendations")
    report.append("-" * 22)
    
    if len(working_servers) < len(results) * 0.5:
        report.append("• Many servers are not responding - check server configurations")
    
    slow_servers = [r for r in working_servers if r.metrics.startup_time_ms > 2000]
    if slow_servers:
        report.append(f"• Slow startup detected in: {', '.join(r.server_name for r in slow_servers)}")
    
    high_latency = [r for r in working_servers if r.metrics.avg_latency_ms > 100]
    if high_latency:
        report.append(f"• High latency detected in: {', '.join(r.server_name for r in high_latency)}")
    
    unreliable = [r for r in working_servers if r.metrics.success_rate < 80]
    if unreliable:
        report.append(f"• Reliability issues in: {', '.join(r.server_name for r in unreliable)}")
    
    if len(working_servers) > 5:
        report.append("• Consider load balancing across multiple working servers")
    
    return "\n".join(report)


async def main():
    """Main focused benchmarking function"""
    logger.info("🎯 MCP Server Focused Performance Benchmarking")
    logger.info("=" * 55)
    
    # Load servers
    servers = load_mcp_servers()
    logger.info(f"📋 Found {len(servers)} MCP servers")
    
    if not servers:
        logger.error("❌ No MCP servers found!")
        return
    
    # Run focused benchmarks
    benchmarker = FocusedBenchmarker()
    results = await benchmarker.run_focused_benchmarks(servers, max_servers=8)
    
    if not results:
        logger.error("❌ No results collected!")
        return
    
    # Generate and display report
    report = generate_focused_report(results)
    print("\n" + report)
    
    # Save results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_dir = "/opt/sutazaiapp/mcp-manager/benchmark_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save JSON results
    json_data = {
        'timestamp': time.time(),
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': [
            {
                'server_name': r.server_name,
                'working': r.metrics.working,
                'startup_time_ms': r.metrics.startup_time_ms,
                'avg_latency_ms': r.metrics.avg_latency_ms,
                'max_latency_ms': r.metrics.max_latency_ms,
                'success_rate': r.metrics.success_rate,
                'error_count': r.metrics.error_count,
                'error_messages': r.error_messages
            }
            for r in results
        ]
    }
    
    json_filename = f"{results_dir}/focused_benchmark_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Save text report
    report_filename = f"{results_dir}/focused_report_{timestamp}.txt"
    with open(report_filename, 'w') as f:
        f.write(report)
    
    logger.info(f"📄 Results saved to: {results_dir}")
    
    # Summary
    working_count = len([r for r in results if r.metrics.working])
    logger.info(f"✅ Benchmark complete: {working_count}/{len(results)} servers working")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Benchmarking interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")