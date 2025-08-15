#!/usr/bin/env python3
"""
MCP Performance Testing Suite

Comprehensive performance and load testing for MCP server automation system.
Validates system performance under various load conditions, measures response times,
throughput, resource utilization, and scalability characteristics.

Test Coverage:
- Download performance and throughput
- Concurrent operation performance
- System resource utilization
- Memory and CPU usage monitoring
- Network performance and bandwidth
- Database operation performance
- Scalability and load testing
- Performance regression detection

Author: Claude AI Assistant (senior-automated-tester)
Created: 2025-08-15 UTC
Version: 1.0.0
"""

import pytest
import asyncio
import time
import psutil
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import AsyncMock, Mock, patch
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from conftest import TestEnvironment, TestMCPServer

# Import automation modules
from config import MCPAutomationConfig, UpdateMode, LogLevel
from mcp_update_manager import MCPUpdateManager
from version_manager import MCPVersionManager
from download_manager import MCPDownloadManager


@dataclass
class PerformanceMetrics:
    """Performance metrics collection structure."""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    throughput: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    network_bytes: Optional[int] = None
    error: Optional[str] = None


@dataclass
class LoadTestResult:
    """Load test result aggregation."""
    test_name: str
    total_operations: int
    successful_operations: int
    failed_operations: int
    duration_seconds: float
    operations_per_second: float
    average_response_time: float
    min_response_time: float
    max_response_time: float
    percentile_95: float
    percentile_99: float
    errors: List[str] = field(default_factory=list)


class PerformanceProfiler:
    """Performance profiler for capturing system metrics."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.start_time: Optional[float] = None
        self.process = psutil.Process()
    
    def start_operation(self, operation_name: str) -> str:
        """Start profiling an operation."""
        self.start_time = time.time()
        return operation_name
    
    def end_operation(
        self,
        operation_name: str,
        success: bool = True,
        error: Optional[str] = None,
        throughput: Optional[float] = None,
        network_bytes: Optional[int] = None
    ) -> PerformanceMetrics:
        """End profiling and record metrics."""
        end_time = time.time()
        duration = end_time - (self.start_time or end_time)
        
        # Capture system metrics
        memory_info = self.process.memory_info()
        cpu_percent = self.process.cpu_percent()
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=self.start_time or end_time,
            end_time=end_time,
            duration=duration,
            success=success,
            throughput=throughput,
            memory_usage_mb=memory_info.rss / (1024 * 1024),
            cpu_usage_percent=cpu_percent,
            network_bytes=network_bytes,
            error=error
        )
        
        self.metrics.append(metrics)
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics:
            return {}
        
        durations = [m.duration for m in self.metrics if m.success]
        memory_usage = [m.memory_usage_mb for m in self.metrics if m.memory_usage_mb]
        cpu_usage = [m.cpu_usage_percent for m in self.metrics if m.cpu_usage_percent]
        
        return {
            "total_operations": len(self.metrics),
            "successful_operations": sum(1 for m in self.metrics if m.success),
            "failed_operations": sum(1 for m in self.metrics if not m.success),
            "average_duration": statistics.mean(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "duration_stddev": statistics.stdev(durations) if len(durations) > 1 else 0,
            "average_memory_mb": statistics.mean(memory_usage) if memory_usage else 0,
            "peak_memory_mb": max(memory_usage) if memory_usage else 0,
            "average_cpu_percent": statistics.mean(cpu_usage) if cpu_usage else 0,
            "peak_cpu_percent": max(cpu_usage) if cpu_usage else 0
        }


class TestMCPDownloadPerformance:
    """Test suite for MCP download performance and throughput."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_download_performance(
        self,
        test_environment: TestEnvironment,
        mock_process_runner,
        performance_monitor
    ):
        """Test performance of single MCP server download."""
        config = test_environment.config
        download_manager = MCPDownloadManager(config)
        profiler = PerformanceProfiler()
        
        server_name = "files"
        package_name = config.mcp_servers[server_name]["package"]
        version = "1.2.3"
        
        with patch.object(download_manager, '_run_command', side_effect=mock_process_runner):
            profiler.start_operation("single_download")
            
            result = await download_manager.download_package(
                package_name,
                version,
                config.get_staging_path(server_name)
            )
            
            metrics = profiler.end_operation(
                "single_download",
                success=result.get("checksum_verified", True),
                throughput=result.get("size_bytes", 0) / result.get("download_time", 1),
                network_bytes=result.get("size_bytes", 0)
            )
            
            # Performance assertions
            assert metrics.duration < 30.0  # Should complete within 30 seconds
            assert metrics.success is True
            assert metrics.memory_usage_mb < 200  # Should not exceed 200MB memory
            
            # Verify download metrics
            assert result["size_bytes"] > 0
            assert result["download_time"] > 0
            assert result["checksum_verified"] is True
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_download_performance(
        self,
        test_environment: TestEnvironment,
        mock_process_runner
    ):
        """Test performance of concurrent MCP server downloads."""
        config = test_environment.config
        download_manager = MCPDownloadManager(config)
        profiler = PerformanceProfiler()
        
        # Select multiple servers for concurrent download
        server_names = list(config.get_all_servers())[:3]
        
        with patch.object(download_manager, '_run_command', side_effect=mock_process_runner):
            profiler.start_operation("concurrent_downloads")
            
            # Start concurrent downloads
            download_tasks = []
            for server_name in server_names:
                package_name = config.mcp_servers[server_name]["package"]
                version = "1.0.0"
                
                task = asyncio.create_task(
                    download_manager.download_package(
                        package_name,
                        version,
                        config.get_staging_path(server_name)
                    )
                )
                download_tasks.append((server_name, task))
            
            # Wait for all downloads to complete
            results = {}
            for server_name, task in download_tasks:
                try:
                    result = await task
                    results[server_name] = result
                except Exception as e:
                    results[server_name] = {"error": str(e), "success": False}
            
            metrics = profiler.end_operation(
                "concurrent_downloads",
                success=all(r.get("checksum_verified", False) for r in results.values()),
                throughput=len(server_names) / metrics.duration if hasattr(metrics, 'duration') else 0
            )
            
            # Performance assertions
            assert len(results) == len(server_names)
            assert metrics.duration < 60.0  # Concurrent should be faster than sequential
            
            # Verify all downloads succeeded
            for server_name, result in results.items():
                if "error" not in result:
                    assert result["checksum_verified"] is True
                    assert result["size_bytes"] > 0
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_download_throughput_scaling(
        self,
        test_environment: TestEnvironment,
        mock_process_runner
    ):
        """Test download throughput scaling with increasing concurrency."""
        config = test_environment.config
        download_manager = MCPDownloadManager(config)
        
        # Test different concurrency levels
        concurrency_levels = [1, 2, 4]
        throughput_results = {}
        
        with patch.object(download_manager, '_run_command', side_effect=mock_process_runner):
            for concurrency in concurrency_levels:
                profiler = PerformanceProfiler()
                profiler.start_operation(f"throughput_test_{concurrency}")
                
                # Create test servers for concurrency level
                test_servers = []
                for i in range(concurrency):
                    server_name = f"test-server-{i}"
                    config.mcp_servers[server_name] = {
                        "package": f"@test/server-{i}",
                        "wrapper": f"test-server-{i}.sh"
                    }
                    test_servers.append(server_name)
                
                # Start concurrent downloads
                download_tasks = []
                for server_name in test_servers:
                    package_name = config.mcp_servers[server_name]["package"]
                    task = asyncio.create_task(
                        download_manager.download_package(
                            package_name,
                            "1.0.0",
                            config.get_staging_path(server_name)
                        )
                    )
                    download_tasks.append(task)
                
                # Measure completion time
                start_time = time.time()
                results = await asyncio.gather(*download_tasks, return_exceptions=True)
                end_time = time.time()
                
                duration = end_time - start_time
                successful_downloads = sum(1 for r in results if not isinstance(r, Exception))
                throughput = successful_downloads / duration if duration > 0 else 0
                
                throughput_results[concurrency] = {
                    "duration": duration,
                    "successful_downloads": successful_downloads,
                    "throughput": throughput
                }
                
                profiler.end_operation(
                    f"throughput_test_{concurrency}",
                    success=successful_downloads == concurrency,
                    throughput=throughput
                )
            
            # Verify throughput scaling
            assert len(throughput_results) == len(concurrency_levels)
            
            # Throughput should generally increase with concurrency (up to a point)
            throughputs = [throughput_results[c]["throughput"] for c in concurrency_levels]
            assert throughputs[1] >= throughputs[0]  # 2 concurrent >= 1 concurrent


class TestMCPSystemPerformance:
    """Test suite for overall system performance under load."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_system_load_performance(
        self,
        test_environment: TestEnvironment,
        mock_health_checker,
        mock_process_runner
    ):
        """Test system performance under high load conditions."""
        config = test_environment.config
        update_manager = MCPUpdateManager(config)
        profiler = PerformanceProfiler()
        
        # Increase system load parameters
        num_operations = 10
        concurrent_servers = 5
        
        with patch.object(update_manager, '_run_health_check', side_effect=mock_health_checker), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=mock_process_runner):
            
            profiler.start_operation("system_load_test")
            
            # Create multiple concurrent operations
            operations = []
            
            # Add server updates
            for i in range(concurrent_servers):
                server_name = f"load-test-server-{i}"
                config.mcp_servers[server_name] = {
                    "package": f"@test/load-server-{i}",
                    "wrapper": f"load-test-server-{i}.sh"
                }
                
                operation = asyncio.create_task(
                    update_manager.update_server(server_name, target_version="1.0.0")
                )
                operations.append(("update", server_name, operation))
            
            # Add health checks
            for i in range(num_operations - concurrent_servers):
                server_name = list(config.get_all_servers())[i % len(config.get_all_servers())]
                operation = asyncio.create_task(
                    update_manager._run_health_check(server_name, timeout=30)
                )
                operations.append(("health", server_name, operation))
            
            # Execute all operations concurrently
            start_time = time.time()
            results = []
            
            for op_type, server_name, task in operations:
                try:
                    result = await task
                    results.append({
                        "type": op_type,
                        "server": server_name,
                        "success": True,
                        "result": result
                    })
                except Exception as e:
                    results.append({
                        "type": op_type,
                        "server": server_name,
                        "success": False,
                        "error": str(e)
                    })
            
            end_time = time.time()
            
            metrics = profiler.end_operation(
                "system_load_test",
                success=all(r["success"] for r in results),
                throughput=len(operations) / (end_time - start_time)
            )
            
            # Performance assertions
            assert len(results) == num_operations
            assert metrics.duration < 120.0  # Should complete within 2 minutes
            assert metrics.memory_usage_mb < 500  # Should not exceed 500MB
            
            # Verify operation success rates
            successful_operations = sum(1 for r in results if r["success"])
            success_rate = successful_operations / len(results)
            assert success_rate >= 0.8  # At least 80% success rate under load
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_sustained_load_performance(
        self,
        test_environment: TestEnvironment,
        mock_health_checker,
        mock_process_runner
    ):
        """Test system performance under sustained load over time."""
        config = test_environment.config
        update_manager = MCPUpdateManager(config)
        
        # Sustained load parameters
        duration_seconds = 30  # 30 second sustained test
        operations_per_second = 2
        
        performance_samples = []
        
        with patch.object(update_manager, '_run_health_check', side_effect=mock_health_checker), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=mock_process_runner):
            
            start_time = time.time()
            operation_count = 0
            
            while time.time() - start_time < duration_seconds:
                sample_start = time.time()
                
                # Perform batch of operations
                batch_operations = []
                for _ in range(operations_per_second):
                    server_name = list(config.get_all_servers())[operation_count % len(config.get_all_servers())]
                    operation = asyncio.create_task(
                        update_manager._run_health_check(server_name, timeout=10)
                    )
                    batch_operations.append(operation)
                    operation_count += 1
                
                # Wait for batch completion
                try:
                    batch_results = await asyncio.gather(*batch_operations, return_exceptions=True)
                    successful_ops = sum(1 for r in batch_results if not isinstance(r, Exception))
                except Exception:
                    successful_ops = 0
                
                sample_end = time.time()
                sample_duration = sample_end - sample_start
                
                # Record performance sample
                performance_samples.append({
                    "timestamp": sample_end,
                    "duration": sample_duration,
                    "operations": len(batch_operations),
                    "successful": successful_ops,
                    "throughput": successful_ops / sample_duration if sample_duration > 0 else 0
                })
                
                # Wait for next sample interval
                sleep_time = max(0, 1.0 - sample_duration)  # Try to maintain 1 second intervals
                await asyncio.sleep(sleep_time)
            
            # Analyze sustained performance
            total_duration = time.time() - start_time
            total_operations = sum(s["operations"] for s in performance_samples)
            total_successful = sum(s["successful"] for s in performance_samples)
            
            average_throughput = total_successful / total_duration if total_duration > 0 else 0
            throughputs = [s["throughput"] for s in performance_samples]
            throughput_variance = statistics.stdev(throughputs) if len(throughputs) > 1 else 0
            
            # Performance assertions
            assert total_operations >= duration_seconds * operations_per_second * 0.8  # At least 80% of expected operations
            assert total_successful / total_operations >= 0.9  # At least 90% success rate
            assert average_throughput >= operations_per_second * 0.8  # At least 80% of target throughput
            assert throughput_variance < average_throughput * 0.5  # Throughput should be stable
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_performance(
        self,
        test_environment: TestEnvironment,
        mock_health_checker,
        mock_process_runner
    ):
        """Test memory usage patterns and potential memory leaks."""
        config = test_environment.config
        update_manager = MCPUpdateManager(config)
        
        memory_samples = []
        process = psutil.Process()
        
        with patch.object(update_manager, '_run_health_check', side_effect=mock_health_checker), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=mock_process_runner):
            
            # Baseline memory measurement
            baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_samples.append({"operation": "baseline", "memory_mb": baseline_memory})
            
            # Perform operations and monitor memory
            operations = [
                ("health_check", lambda: update_manager._run_health_check("files", timeout=30)),
                ("update_server", lambda: update_manager.update_server("files", target_version="1.0.0")),
                ("system_status", lambda: update_manager.get_system_status()),
            ]
            
            for i in range(3):  # Repeat operations to detect memory leaks
                for op_name, op_func in operations:
                    # Record memory before operation
                    pre_memory = process.memory_info().rss / (1024 * 1024)
                    
                    # Perform operation
                    await op_func()
                    
                    # Record memory after operation
                    post_memory = process.memory_info().rss / (1024 * 1024)
                    
                    memory_samples.append({
                        "operation": f"{op_name}_pre_{i}",
                        "memory_mb": pre_memory
                    })
                    memory_samples.append({
                        "operation": f"{op_name}_post_{i}",
                        "memory_mb": post_memory
                    })
                    
                    # Small delay to allow garbage collection
                    await asyncio.sleep(0.1)
            
            # Final memory measurement
            final_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_samples.append({"operation": "final", "memory_mb": final_memory})
            
            # Memory usage analysis
            memory_values = [s["memory_mb"] for s in memory_samples]
            max_memory = max(memory_values)
            memory_growth = final_memory - baseline_memory
            
            # Memory performance assertions
            assert max_memory < 300  # Should not exceed 300MB during operations
            assert memory_growth < 50  # Should not grow more than 50MB over baseline
            
            # Check for significant memory leaks
            # Final memory should not be dramatically higher than baseline
            memory_leak_threshold = baseline_memory * 1.5  # 50% increase threshold
            assert final_memory < memory_leak_threshold, f"Potential memory leak: {final_memory}MB vs baseline {baseline_memory}MB"


class TestMCPPerformanceRegression:
    """Test suite for performance regression detection."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_performance_baseline_establishment(
        self,
        test_environment: TestEnvironment,
        mock_health_checker,
        mock_process_runner,
        tmp_path: Path
    ):
        """Establish performance baselines for regression testing."""
        config = test_environment.config
        update_manager = MCPUpdateManager(config)
        
        baseline_file = tmp_path / "performance_baseline.json"
        
        with patch.object(update_manager, '_run_health_check', side_effect=mock_health_checker), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=mock_process_runner):
            
            # Measure baseline performance for key operations
            baselines = {}
            
            # Health check baseline
            health_times = []
            for _ in range(5):
                start_time = time.time()
                await update_manager._run_health_check("files", timeout=30)
                end_time = time.time()
                health_times.append(end_time - start_time)
            
            baselines["health_check"] = {
                "average_time": statistics.mean(health_times),
                "max_time": max(health_times),
                "min_time": min(health_times),
                "stddev": statistics.stdev(health_times) if len(health_times) > 1 else 0
            }
            
            # Update baseline
            update_times = []
            for i in range(3):
                server_name = f"baseline-server-{i}"
                config.mcp_servers[server_name] = {
                    "package": f"@test/baseline-{i}",
                    "wrapper": f"baseline-server-{i}.sh"
                }
                
                start_time = time.time()
                await update_manager.update_server(server_name, target_version="1.0.0")
                end_time = time.time()
                update_times.append(end_time - start_time)
            
            baselines["server_update"] = {
                "average_time": statistics.mean(update_times),
                "max_time": max(update_times),
                "min_time": min(update_times),
                "stddev": statistics.stdev(update_times) if len(update_times) > 1 else 0
            }
            
            # System status baseline
            status_times = []
            for _ in range(5):
                start_time = time.time()
                await update_manager.get_system_status()
                end_time = time.time()
                status_times.append(end_time - start_time)
            
            baselines["system_status"] = {
                "average_time": statistics.mean(status_times),
                "max_time": max(status_times),
                "min_time": min(status_times),
                "stddev": statistics.stdev(status_times) if len(status_times) > 1 else 0
            }
            
            # Save baseline data
            baseline_data = {
                "timestamp": time.time(),
                "version": "1.0.0",
                "baselines": baselines
            }
            
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f, indent=2)
            
            # Verify baseline establishment
            assert baseline_file.exists()
            assert len(baselines) == 3
            
            for operation, baseline in baselines.items():
                assert baseline["average_time"] > 0
                assert baseline["max_time"] >= baseline["average_time"]
                assert baseline["min_time"] <= baseline["average_time"]
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_performance_regression_detection(
        self,
        test_environment: TestEnvironment,
        mock_health_checker,
        mock_process_runner,
        tmp_path: Path
    ):
        """Test detection of performance regressions against baseline."""
        config = test_environment.config
        update_manager = MCPUpdateManager(config)
        
        # Create mock baseline data
        baseline_data = {
            "timestamp": time.time() - 86400,  # 1 day ago
            "version": "0.9.0",
            "baselines": {
                "health_check": {
                    "average_time": 2.0,
                    "max_time": 3.0,
                    "min_time": 1.5,
                    "stddev": 0.5
                },
                "server_update": {
                    "average_time": 10.0,
                    "max_time": 15.0,
                    "min_time": 8.0,
                    "stddev": 2.0
                },
                "system_status": {
                    "average_time": 1.0,
                    "max_time": 1.5,
                    "min_time": 0.8,
                    "stddev": 0.2
                }
            }
        }
        
        baseline_file = tmp_path / "baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        
        # Mock slower performance to simulate regression
        slow_health_call_count = 0
        async def slow_health_check(name: str, timeout: int = 30):
            nonlocal slow_health_call_count
            slow_health_call_count += 1
            
            # Simulate performance regression (3x slower)
            await asyncio.sleep(0.5)  # Simulate slow operation
            
            return {
                "server_name": name,
                "healthy": True,
                "response_time": 6.0,  # 3x slower than baseline
                "version": "1.0.0",
                "memory_usage_mb": 50
            }
        
        with patch.object(update_manager, '_run_health_check', side_effect=slow_health_check), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=mock_process_runner):
            
            # Measure current performance
            current_times = []
            for _ in range(3):
                start_time = time.time()
                await update_manager._run_health_check("files", timeout=30)
                end_time = time.time()
                current_times.append(end_time - start_time)
            
            current_average = statistics.mean(current_times)
            baseline_average = baseline_data["baselines"]["health_check"]["average_time"]
            
            # Calculate performance regression
            performance_ratio = current_average / baseline_average
            regression_threshold = 2.0  # 100% slower is considered regression
            
            # Detect regression
            is_regression = performance_ratio > regression_threshold
            
            regression_report = {
                "operation": "health_check",
                "baseline_time": baseline_average,
                "current_time": current_average,
                "performance_ratio": performance_ratio,
                "is_regression": is_regression,
                "regression_threshold": regression_threshold,
                "regression_percentage": (performance_ratio - 1) * 100
            }
            
            # Verify regression detection
            assert is_regression is True  # Should detect the simulated regression
            assert performance_ratio > regression_threshold
            assert regression_report["regression_percentage"] > 100  # More than 100% slower
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_performance_trend_analysis(
        self,
        test_environment: TestEnvironment,
        mock_health_checker,
        mock_process_runner,
        tmp_path: Path
    ):
        """Test performance trend analysis over multiple test runs."""
        config = test_environment.config
        update_manager = MCPUpdateManager(config)
        
        # Simulate historical performance data
        historical_data = []
        base_time = time.time() - 604800  # 1 week ago
        
        # Generate 7 days of mock performance data with slight degradation trend
        for day in range(7):
            timestamp = base_time + (day * 86400)  # 1 day intervals
            degradation_factor = 1 + (day * 0.05)  # 5% degradation per day
            
            historical_data.append({
                "timestamp": timestamp,
                "health_check_time": 2.0 * degradation_factor,
                "update_time": 10.0 * degradation_factor,
                "memory_usage": 50 + (day * 2)  # Slight memory increase
            })
        
        # Save historical data
        history_file = tmp_path / "performance_history.json"
        with open(history_file, 'w') as f:
            json.dump(historical_data, f, indent=2)
        
        with patch.object(update_manager, '_run_health_check', side_effect=mock_health_checker), \
             patch.object(update_manager.download_manager, '_run_command', side_effect=mock_process_runner):
            
            # Collect current performance data
            current_health_times = []
            for _ in range(3):
                start_time = time.time()
                await update_manager._run_health_check("files", timeout=30)
                end_time = time.time()
                current_health_times.append(end_time - start_time)
            
            current_average = statistics.mean(current_health_times)
            
            # Analyze performance trend
            health_times = [d["health_check_time"] for d in historical_data]
            
            # Calculate trend (linear regression slope)
            x_values = list(range(len(health_times)))
            y_values = health_times
            
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)
            
            # Linear regression slope calculation
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            trend_analysis = {
                "historical_points": len(historical_data),
                "trend_slope": slope,
                "trend_direction": "degrading" if slope > 0 else "improving",
                "current_performance": current_average,
                "historical_average": statistics.mean(health_times),
                "performance_variance": statistics.stdev(health_times) if len(health_times) > 1 else 0
            }
            
            # Verify trend analysis
            assert trend_analysis["historical_points"] == 7
            assert slope > 0  # Should detect degrading trend
            assert trend_analysis["trend_direction"] == "degrading"
            assert trend_analysis["performance_variance"] > 0