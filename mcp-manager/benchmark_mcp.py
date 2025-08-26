#!/usr/bin/env python3
"""
MCP Server Performance Benchmarking Suite

Comprehensive performance testing for Model Context Protocol (MCP) servers.
Tests throughput, latency, memory usage, and concurrent connection handling.

Based on the official MCP Python SDK patterns and performance best practices.
Measures compliance with Python SDK standards from:
https://github.com/modelcontextprotocol/python-sdk

Author: Claude Code Performance Benchmarker
Created: 2025-08-26
"""

import asyncio
import gc
import json
import logging
import os
import psutil
import statistics
import subprocess
import sys
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import patch

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Import our existing test infrastructure
from test_mcp_servers import (
    MCPServer, MCPServerProcess, MCPProtocolTester,
    load_mcp_servers, TEST_TIMEOUT, MCP_MESSAGE_TIMEOUT
)

# Performance test configuration
BENCHMARK_TIMEOUT = 60  # seconds
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
CONCURRENT_CONNECTIONS = [1, 2, 4, 8, 16]
MESSAGE_BATCH_SIZES = [1, 5, 10, 25, 50]
MEMORY_SAMPLE_INTERVAL = 0.1  # seconds
LATENCY_PERCENTILES = [50, 75, 90, 95, 99, 99.9]

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance measurement results"""
    throughput: float = 0.0  # messages per second
    avg_latency: float = 0.0  # milliseconds
    latency_percentiles: Dict[float, float] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)  # MB
    cpu_usage: float = 0.0  # percentage
    error_rate: float = 0.0  # percentage
    connection_time: float = 0.0  # seconds to establish connection
    peak_memory: float = 0.0  # MB
    memory_leak_rate: float = 0.0  # MB/hour


@dataclass
class BenchmarkResult:
    """Complete benchmark results for a server"""
    server_name: str
    test_type: str
    metrics: PerformanceMetrics
    raw_data: Dict[str, Any] = field(default_factory=dict)
    error_messages: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class SystemMonitor:
    """System resource monitoring utilities"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.memory_samples = []
        self.cpu_samples = []
        self.monitoring_active = False
    
    def start_monitoring(self):
        """Start background system monitoring"""
        self.monitoring_active = True
        self.memory_samples = []
        self.cpu_samples = []
        
    def stop_monitoring(self):
        """Stop background system monitoring"""
        self.monitoring_active = False
    
    def sample_resources(self):
        """Take a resource usage sample"""
        if not self.monitoring_active:
            return
            
        memory_info = self.process.memory_info()
        cpu_percent = self.process.cpu_percent()
        
        self.memory_samples.append(memory_info.rss / 1024 / 1024)  # MB
        self.cpu_samples.append(cpu_percent)
    
    def get_stats(self) -> Dict[str, float]:
        """Get monitoring statistics"""
        if not self.memory_samples or not self.cpu_samples:
            return {}
            
        return {
            'avg_memory_mb': statistics.mean(self.memory_samples),
            'peak_memory_mb': max(self.memory_samples),
            'avg_cpu_percent': statistics.mean(self.cpu_samples),
            'peak_cpu_percent': max(self.cpu_samples),
            'memory_std': statistics.stdev(self.memory_samples) if len(self.memory_samples) > 1 else 0,
        }


class MemoryProfiler:
    """Memory profiling for detecting leaks"""
    
    def __init__(self):
        self.snapshots = []
        self.baseline_memory = 0
    
    def start_profiling(self):
        """Start memory profiling"""
        tracemalloc.start()
        self.baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.snapshots = []
    
    def take_snapshot(self):
        """Take a memory snapshot"""
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            self.snapshots.append((time.time(), snapshot, current_memory))
    
    def stop_profiling(self) -> Dict[str, float]:
        """Stop profiling and analyze for leaks"""
        if not tracemalloc.is_tracing():
            return {}
            
        tracemalloc.stop()
        
        if len(self.snapshots) < 2:
            return {'memory_leak_rate_mb_per_hour': 0.0}
        
        # Calculate memory growth rate
        start_time, _, start_memory = self.snapshots[0]
        end_time, _, end_memory = self.snapshots[-1]
        
        duration_hours = (end_time - start_time) / 3600
        memory_growth = end_memory - start_memory
        
        leak_rate = memory_growth / duration_hours if duration_hours > 0 else 0
        
        return {
            'memory_leak_rate_mb_per_hour': leak_rate,
            'total_memory_growth_mb': memory_growth,
            'duration_seconds': end_time - start_time
        }


class MCPPerformanceBenchmarker:
    """Main benchmarking engine for MCP servers"""
    
    def __init__(self):
        self.protocol_tester = MCPProtocolTester()
        self.system_monitor = SystemMonitor()
        self.memory_profiler = MemoryProfiler()
        self.results = []
    
    async def benchmark_server_startup(self, server: MCPServer) -> BenchmarkResult:
        """Benchmark server startup time and resource usage"""
        logger.info(f"🚀 Benchmarking startup performance for {server.name}")
        
        startup_times = []
        startup_errors = []
        
        # Warm up - discard first few measurements
        for i in range(WARMUP_ITERATIONS):
            server_process = MCPServerProcess(server)
            start_time = time.time()
            
            try:
                started = await server_process.start()
                startup_time = time.time() - start_time
                
                if started:
                    startup_times.append(startup_time * 1000)  # Convert to ms
                else:
                    startup_errors.append(f"Startup failed in warmup iteration {i}")
                    
            except Exception as e:
                startup_errors.append(f"Startup error in warmup {i}: {str(e)}")
            finally:
                server_process.stop()
                await asyncio.sleep(0.1)  # Brief pause between attempts
        
        # Actual measurements
        measurement_times = []
        for i in range(20):  # 20 startup measurements
            server_process = MCPServerProcess(server)
            start_time = time.time()
            
            try:
                started = await server_process.start()
                startup_time = time.time() - start_time
                
                if started:
                    measurement_times.append(startup_time * 1000)  # Convert to ms
                else:
                    startup_errors.append(f"Startup failed in measurement {i}")
                    
            except Exception as e:
                startup_errors.append(f"Startup error in measurement {i}: {str(e)}")
            finally:
                server_process.stop()
                await asyncio.sleep(0.1)
        
        if not measurement_times:
            return BenchmarkResult(
                server_name=server.name,
                test_type="startup",
                metrics=PerformanceMetrics(),
                error_messages=startup_errors
            )
        
        # Calculate statistics
        avg_startup = statistics.mean(measurement_times)
        startup_percentiles = {}
        for p in LATENCY_PERCENTILES:
            if len(measurement_times) > 1:
                startup_percentiles[p] = statistics.quantiles(measurement_times, n=100)[int(p)-1]
        
        metrics = PerformanceMetrics(
            avg_latency=avg_startup,
            latency_percentiles=startup_percentiles,
            connection_time=avg_startup / 1000,  # Convert back to seconds
            error_rate=len(startup_errors) / (len(measurement_times) + len(startup_errors)) * 100
        )
        
        return BenchmarkResult(
            server_name=server.name,
            test_type="startup",
            metrics=metrics,
            raw_data={
                'startup_times_ms': measurement_times,
                'successful_startups': len(measurement_times),
                'failed_startups': len(startup_errors)
            },
            error_messages=startup_errors
        )
    
    async def benchmark_message_latency(self, server: MCPServer) -> BenchmarkResult:
        """Benchmark message latency and response times"""
        logger.info(f"⚡ Benchmarking message latency for {server.name}")
        
        server_process = MCPServerProcess(server)
        
        try:
            if not await server_process.start():
                return BenchmarkResult(
                    server_name=server.name,
                    test_type="latency",
                    metrics=PerformanceMetrics(),
                    error_messages=["Failed to start server for latency test"]
                )
            
            # Initialize server
            init_request = self.protocol_tester.create_initialize_request()
            init_response = await server_process.send_message(init_request, timeout=10)
            
            if not init_response or self.protocol_tester.is_error_response(init_response):
                return BenchmarkResult(
                    server_name=server.name,
                    test_type="latency",
                    metrics=PerformanceMetrics(),
                    error_messages=["Failed to initialize server for latency test"]
                )
            
            # Warmup phase
            for i in range(WARMUP_ITERATIONS):
                ping_request = self.protocol_tester.create_ping_request(request_id=i+1000)
                await server_process.send_message(ping_request, timeout=5)
            
            # Measurement phase
            latencies = []
            errors = []
            
            self.system_monitor.start_monitoring()
            
            for i in range(BENCHMARK_ITERATIONS):
                start_time = time.perf_counter()
                
                ping_request = self.protocol_tester.create_ping_request(request_id=i+2000)
                response = await server_process.send_message(ping_request, timeout=5)
                
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                
                if response and not self.protocol_tester.is_error_response(response):
                    latencies.append(latency_ms)
                else:
                    errors.append(f"Message {i}: No response or error")
                
                self.system_monitor.sample_resources()
                
                # Small delay between messages to prevent overwhelming
                if i % 10 == 0:
                    await asyncio.sleep(0.001)
            
            self.system_monitor.stop_monitoring()
            system_stats = self.system_monitor.get_stats()
            
            if not latencies:
                return BenchmarkResult(
                    server_name=server.name,
                    test_type="latency",
                    metrics=PerformanceMetrics(),
                    error_messages=errors
                )
            
            # Calculate latency statistics
            avg_latency = statistics.mean(latencies)
            latency_percentiles = {}
            
            if len(latencies) > 1:
                sorted_latencies = sorted(latencies)
                for p in LATENCY_PERCENTILES:
                    index = int((p / 100) * len(sorted_latencies))
                    if index >= len(sorted_latencies):
                        index = len(sorted_latencies) - 1
                    latency_percentiles[p] = sorted_latencies[index]
            
            error_rate = len(errors) / (len(latencies) + len(errors)) * 100 if (len(latencies) + len(errors)) > 0 else 0
            
            metrics = PerformanceMetrics(
                avg_latency=avg_latency,
                latency_percentiles=latency_percentiles,
                memory_usage=system_stats,
                cpu_usage=system_stats.get('avg_cpu_percent', 0),
                error_rate=error_rate
            )
            
            return BenchmarkResult(
                server_name=server.name,
                test_type="latency",
                metrics=metrics,
                raw_data={
                    'latencies_ms': latencies,
                    'system_stats': system_stats,
                    'successful_messages': len(latencies),
                    'failed_messages': len(errors)
                },
                error_messages=errors[:10]  # Include first 10 errors
            )
            
        finally:
            server_process.stop()
    
    async def benchmark_throughput(self, server: MCPServer) -> BenchmarkResult:
        """Benchmark message throughput under sustained load"""
        logger.info(f"📊 Benchmarking throughput for {server.name}")
        
        server_process = MCPServerProcess(server)
        
        try:
            if not await server_process.start():
                return BenchmarkResult(
                    server_name=server.name,
                    test_type="throughput",
                    metrics=PerformanceMetrics(),
                    error_messages=["Failed to start server for throughput test"]
                )
            
            # Initialize server
            init_request = self.protocol_tester.create_initialize_request()
            init_response = await server_process.send_message(init_request, timeout=10)
            
            if not init_response or self.protocol_tester.is_error_response(init_response):
                return BenchmarkResult(
                    server_name=server.name,
                    test_type="throughput",
                    metrics=PerformanceMetrics(),
                    error_messages=["Failed to initialize server for throughput test"]
                )
            
            throughput_results = []
            
            for batch_size in MESSAGE_BATCH_SIZES:
                logger.info(f"  Testing batch size: {batch_size}")
                
                # Prepare batch of messages
                messages = []
                for i in range(batch_size):
                    ping_request = self.protocol_tester.create_ping_request(request_id=i+3000)
                    messages.append(ping_request)
                
                self.system_monitor.start_monitoring()
                start_time = time.perf_counter()
                
                # Send batch concurrently
                tasks = []
                for msg in messages:
                    task = asyncio.create_task(server_process.send_message(msg, timeout=10))
                    tasks.append(task)
                
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                end_time = time.perf_counter()
                self.system_monitor.stop_monitoring()
                
                # Analyze results
                successful_responses = 0
                errors = 0
                
                for response in responses:
                    if isinstance(response, Exception):
                        errors += 1
                    elif response and not self.protocol_tester.is_error_response(response):
                        successful_responses += 1
                    else:
                        errors += 1
                
                duration = end_time - start_time
                throughput = successful_responses / duration if duration > 0 else 0
                error_rate = errors / len(responses) * 100 if responses else 0
                
                system_stats = self.system_monitor.get_stats()
                
                throughput_results.append({
                    'batch_size': batch_size,
                    'throughput_msg_per_sec': throughput,
                    'duration_seconds': duration,
                    'successful_messages': successful_responses,
                    'failed_messages': errors,
                    'error_rate_percent': error_rate,
                    'system_stats': system_stats
                })
                
                # Brief pause between batch tests
                await asyncio.sleep(1.0)
            
            if not throughput_results:
                return BenchmarkResult(
                    server_name=server.name,
                    test_type="throughput",
                    metrics=PerformanceMetrics(),
                    error_messages=["No throughput results collected"]
                )
            
            # Find best throughput result
            best_result = max(throughput_results, key=lambda x: x['throughput_msg_per_sec'])
            avg_throughput = statistics.mean([r['throughput_msg_per_sec'] for r in throughput_results])
            
            metrics = PerformanceMetrics(
                throughput=best_result['throughput_msg_per_sec'],
                error_rate=best_result['error_rate_percent'],
                memory_usage=best_result['system_stats'],
                cpu_usage=best_result['system_stats'].get('avg_cpu_percent', 0)
            )
            
            return BenchmarkResult(
                server_name=server.name,
                test_type="throughput",
                metrics=metrics,
                raw_data={
                    'throughput_results': throughput_results,
                    'best_throughput': best_result['throughput_msg_per_sec'],
                    'average_throughput': avg_throughput,
                    'best_batch_size': best_result['batch_size']
                }
            )
            
        finally:
            server_process.stop()
    
    async def benchmark_concurrent_connections(self, server: MCPServer) -> BenchmarkResult:
        """Benchmark concurrent connection handling"""
        logger.info(f"🔄 Benchmarking concurrent connections for {server.name}")
        
        concurrent_results = []
        errors = []
        
        for connection_count in CONCURRENT_CONNECTIONS:
            logger.info(f"  Testing {connection_count} concurrent connections")
            
            # Start multiple server instances
            processes = []
            start_time = time.perf_counter()
            
            try:
                # Start all connections
                for i in range(connection_count):
                    process = MCPServerProcess(server)
                    if await process.start():
                        processes.append(process)
                    else:
                        errors.append(f"Failed to start connection {i+1}/{connection_count}")
                
                connection_setup_time = time.perf_counter() - start_time
                
                if not processes:
                    errors.append(f"No connections established for {connection_count} concurrent test")
                    continue
                
                # Initialize all connections
                init_tasks = []
                for i, process in enumerate(processes):
                    init_request = self.protocol_tester.create_initialize_request(request_id=i+4000)
                    task = asyncio.create_task(process.send_message(init_request, timeout=10))
                    init_tasks.append(task)
                
                init_responses = await asyncio.gather(*init_tasks, return_exceptions=True)
                
                # Count successful initializations
                successful_inits = 0
                for response in init_responses:
                    if not isinstance(response, Exception) and response and not self.protocol_tester.is_error_response(response):
                        successful_inits += 1
                
                # Send concurrent ping messages
                self.system_monitor.start_monitoring()
                ping_start = time.perf_counter()
                
                ping_tasks = []
                for i, process in enumerate(processes):
                    ping_request = self.protocol_tester.create_ping_request(request_id=i+5000)
                    task = asyncio.create_task(process.send_message(ping_request, timeout=15))
                    ping_tasks.append(task)
                
                ping_responses = await asyncio.gather(*ping_tasks, return_exceptions=True)
                
                ping_duration = time.perf_counter() - ping_start
                self.system_monitor.stop_monitoring()
                
                # Analyze ping responses
                successful_pings = 0
                for response in ping_responses:
                    if not isinstance(response, Exception) and response and not self.protocol_tester.is_error_response(response):
                        successful_pings += 1
                
                system_stats = self.system_monitor.get_stats()
                
                concurrent_results.append({
                    'connection_count': connection_count,
                    'established_connections': len(processes),
                    'successful_initializations': successful_inits,
                    'successful_pings': successful_pings,
                    'connection_setup_time': connection_setup_time,
                    'ping_duration': ping_duration,
                    'success_rate': successful_pings / connection_count * 100,
                    'system_stats': system_stats
                })
                
            finally:
                # Clean up all connections
                for process in processes:
                    process.stop()
                
                # Brief pause between concurrent tests
                await asyncio.sleep(2.0)
        
        if not concurrent_results:
            return BenchmarkResult(
                server_name=server.name,
                test_type="concurrent",
                metrics=PerformanceMetrics(),
                error_messages=errors
            )
        
        # Find best concurrent performance
        best_result = max(concurrent_results, key=lambda x: x['success_rate'])
        max_connections = max([r['established_connections'] for r in concurrent_results])
        
        metrics = PerformanceMetrics(
            throughput=best_result['successful_pings'] / best_result['ping_duration'] if best_result['ping_duration'] > 0 else 0,
            error_rate=100 - best_result['success_rate'],
            connection_time=best_result['connection_setup_time'],
            memory_usage=best_result['system_stats'],
            cpu_usage=best_result['system_stats'].get('avg_cpu_percent', 0)
        )
        
        return BenchmarkResult(
            server_name=server.name,
            test_type="concurrent",
            metrics=metrics,
            raw_data={
                'concurrent_results': concurrent_results,
                'max_concurrent_connections': max_connections,
                'best_connection_count': best_result['connection_count'],
                'best_success_rate': best_result['success_rate']
            },
            error_messages=errors[:10]
        )
    
    async def benchmark_memory_usage(self, server: MCPServer) -> BenchmarkResult:
        """Benchmark memory usage and detect memory leaks"""
        logger.info(f"💾 Benchmarking memory usage for {server.name}")
        
        server_process = MCPServerProcess(server)
        
        try:
            self.memory_profiler.start_profiling()
            
            if not await server_process.start():
                return BenchmarkResult(
                    server_name=server.name,
                    test_type="memory",
                    metrics=PerformanceMetrics(),
                    error_messages=["Failed to start server for memory test"]
                )
            
            # Initialize server
            init_request = self.protocol_tester.create_initialize_request()
            init_response = await server_process.send_message(init_request, timeout=10)
            
            if not init_response or self.protocol_tester.is_error_response(init_response):
                return BenchmarkResult(
                    server_name=server.name,
                    test_type="memory",
                    metrics=PerformanceMetrics(),
                    error_messages=["Failed to initialize server for memory test"]
                )
            
            self.memory_profiler.take_snapshot()  # Baseline
            
            # Run sustained load test to detect memory leaks
            test_duration = 30  # seconds
            messages_sent = 0
            errors = []
            
            end_time = time.time() + test_duration
            
            while time.time() < end_time:
                # Send batch of messages
                for i in range(10):  # 10 messages per batch
                    ping_request = self.protocol_tester.create_ping_request(request_id=messages_sent+6000)
                    response = await server_process.send_message(ping_request, timeout=5)
                    
                    if not response or self.protocol_tester.is_error_response(response):
                        errors.append(f"Message {messages_sent}: No response or error")
                    
                    messages_sent += 1
                
                # Take memory snapshot every few seconds
                if messages_sent % 50 == 0:
                    self.memory_profiler.take_snapshot()
                    gc.collect()  # Force garbage collection
                
                await asyncio.sleep(0.1)
            
            self.memory_profiler.take_snapshot()  # Final snapshot
            leak_analysis = self.memory_profiler.stop_profiling()
            
            error_rate = len(errors) / messages_sent * 100 if messages_sent > 0 else 0
            
            metrics = PerformanceMetrics(
                throughput=messages_sent / test_duration,
                error_rate=error_rate,
                memory_leak_rate=leak_analysis.get('memory_leak_rate_mb_per_hour', 0),
                peak_memory=leak_analysis.get('total_memory_growth_mb', 0)
            )
            
            return BenchmarkResult(
                server_name=server.name,
                test_type="memory",
                metrics=metrics,
                raw_data={
                    'messages_sent': messages_sent,
                    'test_duration': test_duration,
                    'leak_analysis': leak_analysis,
                    'error_count': len(errors)
                },
                error_messages=errors[:5]  # First 5 errors
            )
            
        finally:
            server_process.stop()
    
    async def run_comprehensive_benchmark(self, servers: List[MCPServer], max_servers: int = 5) -> List[BenchmarkResult]:
        """Run comprehensive performance benchmarks on selected servers"""
        logger.info("🎯 Starting comprehensive MCP server performance benchmarks")
        logger.info(f"Testing up to {max_servers} servers with multiple performance tests")
        
        selected_servers = servers[:max_servers]  # Limit number of servers to avoid overwhelming
        all_results = []
        
        for i, server in enumerate(selected_servers):
            if not server.enabled:
                logger.info(f"⏭️  Skipping disabled server: {server.name}")
                continue
                
            logger.info(f"\n{'='*60}")
            logger.info(f"🧪 Testing server {i+1}/{len(selected_servers)}: {server.name}")
            logger.info(f"{'='*60}")
            
            try:
                # Test 1: Startup Performance
                startup_result = await self.benchmark_server_startup(server)
                all_results.append(startup_result)
                await asyncio.sleep(1)
                
                # Test 2: Message Latency
                latency_result = await self.benchmark_message_latency(server)
                all_results.append(latency_result)
                await asyncio.sleep(1)
                
                # Test 3: Throughput
                throughput_result = await self.benchmark_throughput(server)
                all_results.append(throughput_result)
                await asyncio.sleep(2)
                
                # Test 4: Concurrent Connections
                concurrent_result = await self.benchmark_concurrent_connections(server)
                all_results.append(concurrent_result)
                await asyncio.sleep(2)
                
                # Test 5: Memory Usage & Leak Detection
                memory_result = await self.benchmark_memory_usage(server)
                all_results.append(memory_result)
                
                logger.info(f"✅ Completed benchmarking {server.name}")
                
            except Exception as e:
                logger.error(f"❌ Error benchmarking {server.name}: {str(e)}")
                error_result = BenchmarkResult(
                    server_name=server.name,
                    test_type="error",
                    metrics=PerformanceMetrics(),
                    error_messages=[f"Benchmark failed: {str(e)}"]
                )
                all_results.append(error_result)
            
            # Pause between servers to allow cleanup
            await asyncio.sleep(3)
        
        logger.info(f"\n🎉 Benchmark suite completed! Generated {len(all_results)} results")
        return all_results


class BenchmarkAnalyzer:
    """Analyze and report benchmark results"""
    
    def __init__(self, results: List[BenchmarkResult]):
        self.results = results
        self.servers = list(set(r.server_name for r in results))
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report"""
        report = []
        report.append("🏆 MCP Server Performance Benchmark Report")
        report.append("=" * 60)
        report.append(f"📊 Tested {len(self.servers)} servers with {len(self.results)} total benchmarks")
        report.append(f"🕒 Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Server Performance Overview
        report.append("📈 Performance Overview")
        report.append("-" * 30)
        
        server_summaries = {}
        for server_name in self.servers:
            server_results = [r for r in self.results if r.server_name == server_name]
            server_summaries[server_name] = self._analyze_server_performance(server_results)
        
        # Rank servers by overall performance
        ranked_servers = sorted(
            server_summaries.items(),
            key=lambda x: x[1]['overall_score'],
            reverse=True
        )
        
        for i, (server_name, summary) in enumerate(ranked_servers):
            report.append(f"{i+1:2d}. {server_name}")
            report.append(f"    Overall Score: {summary['overall_score']:.1f}/100")
            report.append(f"    Best Metrics: {', '.join(summary['strengths'])}")
            if summary['weaknesses']:
                report.append(f"    Areas for Improvement: {', '.join(summary['weaknesses'])}")
            report.append("")
        
        # Detailed Performance Analysis
        report.append("🔍 Detailed Performance Analysis")
        report.append("-" * 40)
        
        # Startup Performance
        startup_results = [r for r in self.results if r.test_type == "startup"]
        if startup_results:
            report.append("⚡ Startup Performance:")
            startup_times = [(r.server_name, r.metrics.avg_latency) for r in startup_results if r.metrics.avg_latency > 0]
            startup_times.sort(key=lambda x: x[1])  # Sort by startup time
            
            for server_name, startup_time in startup_times[:5]:
                report.append(f"    {server_name}: {startup_time:.1f}ms")
            report.append("")
        
        # Throughput Performance
        throughput_results = [r for r in self.results if r.test_type == "throughput"]
        if throughput_results:
            report.append("🚀 Throughput Performance:")
            throughput_rates = [(r.server_name, r.metrics.throughput) for r in throughput_results if r.metrics.throughput > 0]
            throughput_rates.sort(key=lambda x: x[1], reverse=True)  # Sort by throughput
            
            for server_name, throughput in throughput_rates[:5]:
                report.append(f"    {server_name}: {throughput:.1f} msg/sec")
            report.append("")
        
        # Latency Performance
        latency_results = [r for r in self.results if r.test_type == "latency"]
        if latency_results:
            report.append("⏱️  Latency Performance (P95):")
            latency_p95 = [(r.server_name, r.metrics.latency_percentiles.get(95, r.metrics.avg_latency)) 
                          for r in latency_results if r.metrics.avg_latency > 0]
            latency_p95.sort(key=lambda x: x[1])  # Sort by latency (lower is better)
            
            for server_name, p95_latency in latency_p95[:5]:
                report.append(f"    {server_name}: {p95_latency:.1f}ms")
            report.append("")
        
        # Memory Usage
        memory_results = [r for r in self.results if r.test_type == "memory"]
        if memory_results:
            report.append("💾 Memory Usage:")
            for result in memory_results:
                if result.metrics.memory_leak_rate != 0:
                    status = "⚠️ " if result.metrics.memory_leak_rate > 10 else "✅"
                    report.append(f"    {result.server_name}: {status} {result.metrics.memory_leak_rate:.2f} MB/hour leak rate")
            report.append("")
        
        # Error Rates
        error_results = [r for r in self.results if r.metrics.error_rate > 0]
        if error_results:
            report.append("⚠️  Error Rates:")
            for result in error_results:
                report.append(f"    {result.server_name} ({result.test_type}): {result.metrics.error_rate:.1f}%")
            report.append("")
        
        # Recommendations
        report.append("💡 Performance Recommendations")
        report.append("-" * 35)
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            report.append(f"• {rec}")
        
        return "\n".join(report)
    
    def _analyze_server_performance(self, server_results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze performance for a single server"""
        scores = {}
        strengths = []
        weaknesses = []
        
        for result in server_results:
            if result.test_type == "startup" and result.metrics.avg_latency > 0:
                # Lower startup time is better (score inversely)
                startup_score = max(0, 100 - (result.metrics.avg_latency / 10))  # 10ms = 100 points
                scores['startup'] = startup_score
                if startup_score > 80:
                    strengths.append("Fast startup")
                elif startup_score < 50:
                    weaknesses.append("Slow startup")
            
            elif result.test_type == "throughput" and result.metrics.throughput > 0:
                # Higher throughput is better
                throughput_score = min(100, result.metrics.throughput * 2)  # 50 msg/sec = 100 points
                scores['throughput'] = throughput_score
                if throughput_score > 80:
                    strengths.append("High throughput")
                elif throughput_score < 30:
                    weaknesses.append("Low throughput")
            
            elif result.test_type == "latency" and result.metrics.avg_latency > 0:
                # Lower latency is better
                latency_score = max(0, 100 - (result.metrics.avg_latency / 2))  # 2ms = 100 points
                scores['latency'] = latency_score
                if latency_score > 80:
                    strengths.append("Low latency")
                elif latency_score < 50:
                    weaknesses.append("High latency")
            
            elif result.test_type == "memory":
                # Lower memory leak rate is better
                leak_rate = abs(result.metrics.memory_leak_rate)
                memory_score = max(0, 100 - leak_rate)  # 1 MB/hour = 99 points
                scores['memory'] = memory_score
                if memory_score > 90:
                    strengths.append("No memory leaks")
                elif memory_score < 70:
                    weaknesses.append("Memory leaks detected")
        
        # Calculate overall score (weighted average)
        weights = {'startup': 0.2, 'throughput': 0.3, 'latency': 0.3, 'memory': 0.2}
        overall_score = sum(scores.get(metric, 0) * weight for metric, weight in weights.items())
        
        return {
            'overall_score': overall_score,
            'individual_scores': scores,
            'strengths': strengths[:3],  # Top 3 strengths
            'weaknesses': weaknesses[:2]  # Top 2 weaknesses
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations based on results"""
        recommendations = []
        
        # Analyze common issues
        high_error_servers = [r.server_name for r in self.results if r.metrics.error_rate > 10]
        if high_error_servers:
            recommendations.append(f"Investigate error handling in: {', '.join(set(high_error_servers))}")
        
        slow_startup_servers = [r.server_name for r in self.results 
                               if r.test_type == "startup" and r.metrics.avg_latency > 1000]
        if slow_startup_servers:
            recommendations.append(f"Optimize startup time for: {', '.join(set(slow_startup_servers))}")
        
        memory_leak_servers = [r.server_name for r in self.results 
                              if r.test_type == "memory" and abs(r.metrics.memory_leak_rate) > 5]
        if memory_leak_servers:
            recommendations.append(f"Address memory leaks in: {', '.join(set(memory_leak_servers))}")
        
        low_throughput_servers = [r.server_name for r in self.results 
                                 if r.test_type == "throughput" and r.metrics.throughput < 10]
        if low_throughput_servers:
            recommendations.append(f"Improve message processing throughput in: {', '.join(set(low_throughput_servers))}")
        
        # General recommendations
        recommendations.extend([
            "Consider implementing connection pooling for high-concurrency scenarios",
            "Monitor memory usage patterns in production environments",
            "Implement proper error handling and graceful degradation",
            "Use async/await patterns for better concurrency handling",
            "Consider message batching to improve throughput"
        ])
        
        return recommendations[:8]  # Return top 8 recommendations
    
    def export_results_json(self, filename: str) -> None:
        """Export detailed results to JSON file"""
        export_data = {
            'benchmark_metadata': {
                'timestamp': time.time(),
                'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'servers_tested': len(self.servers),
                'total_tests': len(self.results)
            },
            'results': []
        }
        
        for result in self.results:
            result_data = {
                'server_name': result.server_name,
                'test_type': result.test_type,
                'timestamp': result.timestamp,
                'metrics': {
                    'throughput': result.metrics.throughput,
                    'avg_latency': result.metrics.avg_latency,
                    'latency_percentiles': result.metrics.latency_percentiles,
                    'memory_usage': result.metrics.memory_usage,
                    'cpu_usage': result.metrics.cpu_usage,
                    'error_rate': result.metrics.error_rate,
                    'connection_time': result.metrics.connection_time,
                    'peak_memory': result.metrics.peak_memory,
                    'memory_leak_rate': result.metrics.memory_leak_rate
                },
                'raw_data': result.raw_data,
                'error_messages': result.error_messages
            }
            export_data['results'].append(result_data)
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"📄 Detailed results exported to {filename}")


def create_performance_visualizations(results: List[BenchmarkResult], output_dir: str = "/opt/sutazaiapp/mcp-manager/benchmark_results"):
    """Create performance visualization charts"""
    if not VISUALIZATION_AVAILABLE:
        logger.warning("📊 Matplotlib/pandas not available - skipping visualizations")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Throughput comparison chart
    throughput_results = [r for r in results if r.test_type == "throughput" and r.metrics.throughput > 0]
    if throughput_results:
        plt.figure(figsize=(12, 6))
        servers = [r.server_name for r in throughput_results]
        throughputs = [r.metrics.throughput for r in throughput_results]
        
        plt.bar(servers, throughputs, color='skyblue', alpha=0.7)
        plt.title('MCP Server Throughput Comparison')
        plt.xlabel('Server')
        plt.ylabel('Messages per Second')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/throughput_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Latency comparison chart
    latency_results = [r for r in results if r.test_type == "latency" and r.metrics.avg_latency > 0]
    if latency_results:
        plt.figure(figsize=(12, 6))
        servers = [r.server_name for r in latency_results]
        avg_latencies = [r.metrics.avg_latency for r in latency_results]
        p95_latencies = [r.metrics.latency_percentiles.get(95, r.metrics.avg_latency) for r in latency_results]
        
        x = range(len(servers))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], avg_latencies, width, label='Average Latency', color='lightcoral', alpha=0.7)
        plt.bar([i + width/2 for i in x], p95_latencies, width, label='P95 Latency', color='lightblue', alpha=0.7)
        
        plt.title('MCP Server Latency Comparison')
        plt.xlabel('Server')
        plt.ylabel('Latency (ms)')
        plt.xticks(x, servers, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/latency_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"📊 Performance charts saved to {output_dir}/")


async def main():
    """Main benchmarking function"""
    logger.info("🎯 MCP Server Performance Benchmarking Suite")
    logger.info("=" * 60)
    
    # Load MCP servers
    servers = load_mcp_servers()
    logger.info(f"📋 Found {len(servers)} MCP servers in configuration")
    
    if not servers:
        logger.error("❌ No MCP servers found! Please check configuration files.")
        return
    
    # Create benchmarker
    benchmarker = MCPPerformanceBenchmarker()
    
    # Run comprehensive benchmarks
    max_servers = min(len(servers), 5)  # Limit to 5 servers to avoid overwhelming system
    logger.info(f"🧪 Running benchmarks on first {max_servers} servers")
    
    try:
        results = await benchmarker.run_comprehensive_benchmark(servers, max_servers)
        
        if not results:
            logger.error("❌ No benchmark results collected!")
            return
        
        # Analyze results
        analyzer = BenchmarkAnalyzer(results)
        
        # Generate and display summary report
        summary_report = analyzer.generate_summary_report()
        print("\n" + summary_report)
        
        # Export detailed results
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        results_dir = "/opt/sutazaiapp/mcp-manager/benchmark_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Export JSON results
        json_filename = f"{results_dir}/mcp_benchmark_{timestamp}.json"
        analyzer.export_results_json(json_filename)
        
        # Export text report
        report_filename = f"{results_dir}/mcp_benchmark_report_{timestamp}.txt"
        with open(report_filename, 'w') as f:
            f.write(summary_report)
        logger.info(f"📄 Summary report saved to {report_filename}")
        
        # Create visualizations
        create_performance_visualizations(results, results_dir)
        
        logger.info(f"\n✅ Benchmarking completed successfully!")
        logger.info(f"📁 Results saved to: {results_dir}")
        
        # Print key findings
        print("\n🔑 Key Findings:")
        working_servers = len(set(r.server_name for r in results if r.metrics.throughput > 0 or r.metrics.avg_latency > 0))
        print(f"   • {working_servers}/{len(set(r.server_name for r in results))} servers are responding to benchmarks")
        
        best_throughput = max((r.metrics.throughput for r in results if r.metrics.throughput > 0), default=0)
        if best_throughput > 0:
            best_server = next(r.server_name for r in results if r.metrics.throughput == best_throughput)
            print(f"   • Best throughput: {best_throughput:.1f} msg/sec ({best_server})")
        
        fastest_startup = min((r.metrics.avg_latency for r in results if r.test_type == "startup" and r.metrics.avg_latency > 0), default=0)
        if fastest_startup > 0:
            fastest_server = next(r.server_name for r in results if r.test_type == "startup" and r.metrics.avg_latency == fastest_startup)
            print(f"   • Fastest startup: {fastest_startup:.1f}ms ({fastest_server})")
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Benchmarking interrupted by user")
    except Exception as e:
        logger.error(f"❌ Benchmarking failed with error: {str(e)}")
        logger.debug(f"Full error: {repr(e)}")


if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("MCP Server Performance Benchmarking Suite")
            print("")
            print("Usage:")
            print("  python benchmark_mcp.py                    # Run full benchmark suite")
            print("  python benchmark_mcp.py --servers N       # Limit to N servers")
            print("  python benchmark_mcp.py --help            # Show this help")
            print("")
            print("Tests performed:")
            print("  • Server startup performance")
            print("  • Message latency (ping/pong)")
            print("  • Throughput under load")
            print("  • Concurrent connection handling")
            print("  • Memory usage and leak detection")
            print("")
            print("Results are saved to: /opt/sutazaiapp/mcp-manager/benchmark_results/")
            sys.exit(0)
        elif sys.argv[1] == "--servers" and len(sys.argv) > 2:
            try:
                max_servers = int(sys.argv[2])
                logger.info(f"🎯 Limiting benchmark to {max_servers} servers")
            except ValueError:
                logger.error("❌ Invalid number of servers specified")
                sys.exit(1)
    
    # Run the benchmark
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Benchmarking interrupted")
        sys.exit(1)