"""
AI-Powered Performance Testing Suite for SutazAI System
Implements comprehensive performance benchmarks and quality validation
"""

import asyncio
import time
import json
import logging
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import psutil
import numpy as np
from transformers import pipeline
import pytest
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

@dataclass
class PerformanceResult:
    """Performance test result data structure"""
    test_name: str
    component: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    error_message: Optional[str] = None
    metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

@dataclass
class BenchmarkThresholds:
    """Performance benchmark thresholds"""
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_acceptable_ms: float
    success_rate: float
    throughput_rps: float

class AIPerformanceOracle:
    """AI-powered performance analysis and prediction"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.historical_data = []
        self.performance_model = None
        
    async def initialize_ai_models(self):
        """Initialize AI models for performance analysis"""
        try:
            # Initialize lightweight performance analysis model
            self.performance_model = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                device=-1  # CPU
            )
            logger.info("AI performance models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
            
    def analyze_performance_patterns(self, results: List[PerformanceResult]) -> Dict[str, Any]:
        """Analyze performance patterns using AI"""
        if not results:
            return {"analysis": "No data available", "recommendations": []}
            
        # Convert results to numerical features
        features = []
        for result in results:
            feature_vector = [
                result.duration,
                1.0 if result.success else 0.0,
                len(result.error_message or ""),
                result.metrics.get("cpu_usage", 0),
                result.metrics.get("memory_usage", 0),
                result.metrics.get("response_size", 0)
            ]
            features.append(feature_vector)
            
        if len(features) < 10:
            return {"analysis": "Insufficient data for AI analysis", "recommendations": []}
            
        # Normalize features
        features_normalized = self.scaler.fit_transform(features)
        
        # Detect anomalies
        anomalies = self.anomaly_detector.fit_predict(features_normalized)
        anomaly_indices = [i for i, a in enumerate(anomalies) if a == -1]
        
        # Calculate performance trends
        durations = [r.duration for r in results]
        success_rates = [1 if r.success else 0 for r in results]
        
        analysis = {
            "total_tests": len(results),
            "success_rate": statistics.mean(success_rates),
            "avg_duration": statistics.mean(durations),
            "p50_duration": statistics.median(durations),
            "p95_duration": np.percentile(durations, 95),
            "p99_duration": np.percentile(durations, 99),
            "anomalies_detected": len(anomaly_indices),
            "anomaly_percentage": len(anomaly_indices) / len(results) * 100,
            "performance_trend": self._calculate_trend(durations),
            "recommendations": self._generate_ai_recommendations(results, anomaly_indices)
        }
        
        return analysis
        
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate performance trend"""
        if len(values) < 5:
            return "insufficient_data"
            
        # Simple linear trend calculation
        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 0.01:
            return "stable"
        elif slope > 0:
            return "degrading"  
        else:
            return "improving"
            
    def _generate_ai_recommendations(self, results: List[PerformanceResult], anomaly_indices: List[int]) -> List[str]:
        """Generate AI-powered performance recommendations"""
        recommendations = []
        
        # Analyze error patterns
        errors = [r.error_message for r in results if r.error_message]
        if errors:
            error_analysis = self._analyze_error_patterns(errors)
            recommendations.extend(error_analysis)
            
        # Analyze performance patterns
        slow_tests = [r for r in results if r.duration > 5000]  # > 5 seconds
        if len(slow_tests) > len(results) * 0.1:  # > 10% slow
            recommendations.append("Optimize slow operations - consider caching or async processing")
            
        # Memory usage analysis
        high_memory_tests = [r for r in results if r.metrics.get("memory_usage", 0) > 80]
        if high_memory_tests:
            recommendations.append("High memory usage detected - implement memory optimization")
            
        # CPU usage analysis
        high_cpu_tests = [r for r in results if r.metrics.get("cpu_usage", 0) > 85]
        if high_cpu_tests:
            recommendations.append("High CPU usage detected - consider load balancing or optimization")
            
        # Anomaly-based recommendations
        if len(anomaly_indices) > 0:
            recommendations.append(f"Performance anomalies detected in {len(anomaly_indices)} tests - investigate outliers")
            
        return recommendations[:10]  # Limit to top 10 recommendations
        
    def _analyze_error_patterns(self, errors: List[str]) -> List[str]:
        """Analyze error patterns for recommendations"""
        recommendations = []
        
        # Common error pattern analysis
        timeout_errors = sum(1 for e in errors if "timeout" in e.lower())
        if timeout_errors > len(errors) * 0.3:
            recommendations.append("High timeout rate - increase timeout thresholds or optimize operations")
            
        connection_errors = sum(1 for e in errors if "connection" in e.lower())
        if connection_errors > len(errors) * 0.2:
            recommendations.append("Connection issues detected - check network configuration and connection pooling")
            
        memory_errors = sum(1 for e in errors if "memory" in e.lower() or "oom" in e.lower())
        if memory_errors > 0:
            recommendations.append("Memory-related errors detected - increase memory allocation or implement garbage collection")
            
        return recommendations

class PerformanceTestSuite:
    """Comprehensive AI-powered performance testing suite"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.ai_oracle = AIPerformanceOracle()
        self.results: List[PerformanceResult] = []
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Benchmark thresholds based on requirements document
        self.thresholds = {
            "/health": BenchmarkThresholds(10, 25, 50, 100, 99.95, 1000),
            "/api/v1/system/status": BenchmarkThresholds(50, 150, 300, 500, 99.9, 500),
            "/api/v1/agents/": BenchmarkThresholds(75, 200, 400, 1000, 99.5, 200),
            "/api/v1/models/list": BenchmarkThresholds(100, 250, 500, 1000, 99.0, 100),
            "/api/v1/coordinator/think": BenchmarkThresholds(500, 2000, 5000, 10000, 95.0, 50),
            "/api/v1/vectors/search": BenchmarkThresholds(100, 300, 600, 1500, 98.0, 500),
        }
        
    async def initialize(self):
        """Initialize the performance test suite"""
        await self.ai_oracle.initialize_ai_models()
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=20)
        )
        logger.info("Performance test suite initialized")
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            
    async def run_comprehensive_suite(self) -> Dict[str, Any]:
        """Run the complete performance test suite"""
        logger.info("Starting comprehensive performance test suite")
        
        test_results = {
            "start_time": datetime.now().isoformat(),
            "tests": {},
            "summary": {},
            "ai_analysis": {},
            "recommendations": []
        }
        
        try:
            # Run different test categories
            test_results["tests"]["api_latency"] = await self._test_api_latency()
            test_results["tests"]["load_testing"] = await self._test_load_performance()
            test_results["tests"]["stress_testing"] = await self._test_stress_limits()
            test_results["tests"]["endurance_testing"] = await self._test_endurance()
            test_results["tests"]["agent_performance"] = await self._test_agent_performance()
            test_results["tests"]["model_inference"] = await self._test_model_inference()
            test_results["tests"]["database_performance"] = await self._test_database_performance()
            test_results["tests"]["resource_utilization"] = await self._test_resource_utilization()
            
            # Generate AI analysis
            test_results["ai_analysis"] = self.ai_oracle.analyze_performance_patterns(self.results)
            test_results["summary"] = self._generate_test_summary()
            test_results["recommendations"] = test_results["ai_analysis"].get("recommendations", [])
            
        except Exception as e:
            logger.error(f"Performance test suite failed: {e}")
            test_results["error"] = str(e)
        finally:
            test_results["end_time"] = datetime.now().isoformat()
            test_results["duration"] = (
                datetime.fromisoformat(test_results["end_time"]) - 
                datetime.fromisoformat(test_results["start_time"])
            ).total_seconds()
            
        logger.info("Performance test suite completed")
        return test_results
        
    async def _test_api_latency(self) -> Dict[str, Any]:
        """Test API endpoint latency against benchmarks"""
        logger.info("Testing API latency performance")
        latency_results = {}
        
        for endpoint, threshold in self.thresholds.items():
            results = []
            
            # Test each endpoint multiple times
            for i in range(50):  # 50 requests per endpoint
                start_time = time.time()
                success = False
                error_msg = None
                
                try:
                    if endpoint == "/api/v1/coordinator/think":
                        # Special handling for coordinator endpoint
                        payload = {
                            "input_data": {"text": f"Performance test query {i}"},
                            "reasoning_type": "deductive"
                        }
                        response = await self._make_request("POST", endpoint, json=payload)
                    else:
                        response = await self._make_request("GET", endpoint)
                        
                    success = 200 <= response.status < 300
                    if not success:
                        error_msg = f"HTTP {response.status}"
                        
                except Exception as e:
                    error_msg = str(e)
                    
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                
                result = PerformanceResult(
                    test_name=f"latency_{endpoint}",
                    component="api",
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration_ms,
                    success=success,
                    error_message=error_msg,
                    metrics={
                        "cpu_usage": psutil.cpu_percent(),
                        "memory_usage": psutil.virtual_memory().percent
                    }
                )
                
                results.append(result)
                self.results.append(result)
                
                # Small delay between requests
                await asyncio.sleep(0.1)
                
            # Analyze results for this endpoint
            durations = [r.duration for r in results]
            success_rate = sum(1 for r in results if r.success) / len(results) * 100
            
            latency_results[endpoint] = {
                "p50": np.percentile(durations, 50),
                "p95": np.percentile(durations, 95),
                "p99": np.percentile(durations, 99),
                "max": max(durations),
                "min": min(durations),
                "avg": statistics.mean(durations),
                "success_rate": success_rate,
                "threshold_compliance": {
                    "p50": np.percentile(durations, 50) <= threshold.p50_ms,
                    "p95": np.percentile(durations, 95) <= threshold.p95_ms,
                    "p99": np.percentile(durations, 99) <= threshold.p99_ms,
                    "success_rate": success_rate >= threshold.success_rate
                }
            }
            
        return latency_results
        
    async def _test_load_performance(self) -> Dict[str, Any]:
        """Test system performance under load"""
        logger.info("Testing load performance")
        
        # Gradually increase load
        load_levels = [10, 25, 50, 100, 200]  # Concurrent requests
        load_results = {}
        
        for load_level in load_levels:
            logger.info(f"Testing load level: {load_level} concurrent requests")
            
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(load_level)
            
            async def make_load_request():
                async with semaphore:
                    start_time = time.time()
                    try:
                        response = await self._make_request("GET", "/api/v1/system/status")
                        success = 200 <= response.status < 300
                        error_msg = None if success else f"HTTP {response.status}"
                    except Exception as e:
                        success = False
                        error_msg = str(e)
                    
                    end_time = time.time()
                    return PerformanceResult(
                        test_name=f"load_test_{load_level}",
                        component="api",
                        start_time=start_time,
                        end_time=end_time,
                        duration=(end_time - start_time) * 1000,
                        success=success,
                        error_message=error_msg,
                        metrics={
                            "cpu_usage": psutil.cpu_percent(),
                            "memory_usage": psutil.virtual_memory().percent,
                            "concurrent_level": load_level
                        }
                    )
            
            # Execute load test
            start_time = time.time()
            results = await asyncio.gather(*[make_load_request() for _ in range(load_level * 10)])
            end_time = time.time()
            
            # Calculate metrics
            durations = [r.duration for r in results]
            success_rate = sum(1 for r in results if r.success) / len(results) * 100
            throughput = len(results) / (end_time - start_time)
            
            load_results[load_level] = {
                "requests": len(results),
                "duration": end_time - start_time,
                "throughput_rps": throughput,
                "avg_response_time": statistics.mean(durations),
                "p95_response_time": np.percentile(durations, 95),
                "success_rate": success_rate,
                "errors": len([r for r in results if not r.success])
            }
            
            self.results.extend(results)
            
            # Brief pause between load levels
            await asyncio.sleep(2)
            
        return load_results
        
    async def _test_stress_limits(self) -> Dict[str, Any]:
        """Test system behavior at stress limits"""
        logger.info("Testing stress limits")
        
        stress_results = {
            "max_concurrent_users": 0,
            "breaking_point": None,
            "recovery_time": None,
            "degradation_pattern": []
        }
        
        # Gradually increase stress until breaking point
        concurrent_levels = [100, 250, 500, 750, 1000, 1500, 2000]
        
        for level in concurrent_levels:
            logger.info(f"Testing stress level: {level} concurrent requests")
            
            # Monitor system before stress test
            cpu_before = psutil.cpu_percent(interval=1)
            memory_before = psutil.virtual_memory().percent
            
            # Apply stress
            semaphore = asyncio.Semaphore(level)
            successful_requests = 0
            failed_requests = 0
            
            async def stress_request():
                nonlocal successful_requests, failed_requests
                async with semaphore:
                    try:
                        response = await self._make_request("GET", "/health", timeout=5)
                        if 200 <= response.status < 300:
                            successful_requests += 1
                        else:
                            failed_requests += 1
                    except:
                        failed_requests += 1
            
            start_time = time.time()
            await asyncio.gather(*[stress_request() for _ in range(level)], return_exceptions=True)
            end_time = time.time()
            
            # Monitor system after stress test
            cpu_after = psutil.cpu_percent(interval=1)
            memory_after = psutil.virtual_memory().percent
            
            success_rate = successful_requests / (successful_requests + failed_requests) * 100
            
            stress_results["degradation_pattern"].append({
                "level": level,
                "success_rate": success_rate,
                "response_time": end_time - start_time,
                "cpu_usage": cpu_after,
                "memory_usage": memory_after,
                "cpu_delta": cpu_after - cpu_before,
                "memory_delta": memory_after - memory_before
            })
            
            # Check if we've reached breaking point (success rate < 80%)
            if success_rate < 80:
                stress_results["breaking_point"] = level
                break
            else:
                stress_results["max_concurrent_users"] = level
                
            await asyncio.sleep(5)  # Recovery time between tests
            
        return stress_results
        
    async def _test_endurance(self) -> Dict[str, Any]:
        """Test system endurance over time"""
        logger.info("Testing system endurance (5-minute sustained load)")
        
        endurance_duration = 300  # 5 minutes
        request_interval = 1  # 1 request per second
        
        start_time = time.time()
        end_time = start_time + endurance_duration
        results = []
        
        while time.time() < end_time:
            request_start = time.time()
            try:
                response = await self._make_request("GET", "/api/v1/system/status")
                success = 200 <= response.status < 300
                error_msg = None if success else f"HTTP {response.status}"
            except Exception as e:
                success = False
                error_msg = str(e)
                
            request_end = time.time()
            
            result = PerformanceResult(
                test_name="endurance_test",
                component="api",
                start_time=request_start,
                end_time=request_end,
                duration=(request_end - request_start) * 1000,
                success=success,
                error_message=error_msg,
                metrics={
                    "cpu_usage": psutil.cpu_percent(),
                    "memory_usage": psutil.virtual_memory().percent,
                    "elapsed_time": time.time() - start_time
                }
            )
            
            results.append(result)
            
            # Wait for next request interval
            next_request = request_start + request_interval
            wait_time = max(0, next_request - time.time())
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                
        # Analyze endurance results
        durations = [r.duration for r in results]
        memory_usage = [r.metrics["memory_usage"] for r in results]
        cpu_usage = [r.metrics["cpu_usage"] for r in results]
        success_rates = [1 if r.success else 0 for r in results]
        
        self.results.extend(results)
        
        return {
            "duration": endurance_duration,
            "total_requests": len(results),
            "avg_response_time": statistics.mean(durations),
            "response_time_trend": self.ai_oracle._calculate_trend(durations),
            "memory_usage_trend": self.ai_oracle._calculate_trend(memory_usage),
            "cpu_usage_trend": self.ai_oracle._calculate_trend(cpu_usage),
            "overall_success_rate": statistics.mean(success_rates) * 100,
            "memory_leak_detected": max(memory_usage) - min(memory_usage) > 20,
            "performance_degradation": max(durations) > min(durations) * 2
        }
        
    async def _test_agent_performance(self) -> Dict[str, Any]:
        """Test agent-specific performance"""
        logger.info("Testing agent performance")
        
        agent_results = {}
        
        # Test agent list endpoint
        start_time = time.time()
        try:
            response = await self._make_request("GET", "/api/v1/agents/")
            success = 200 <= response.status < 300
            if success:
                data = await response.json()
                agent_count = data.get("active_count", 0)
            else:
                agent_count = 0
        except Exception as e:
            success = False
            agent_count = 0
            
        end_time = time.time()
        
        agent_results["agent_list"] = {
            "success": success,
            "response_time": (end_time - start_time) * 1000,
            "active_agents": agent_count
        }
        
        # Test workflow execution (if available)
        try:
            workflow_payload = {
                "directory": "/opt/sutazaiapp/backend",
                "output_format": "json"
            }
            
            start_time = time.time()
            response = await self._make_request(
                "POST", 
                "/api/v1/agents/workflows/code-improvement",
                json=workflow_payload
            )
            end_time = time.time()
            
            success = 200 <= response.status < 300
            
            agent_results["workflow_execution"] = {
                "success": success,
                "initiation_time": (end_time - start_time) * 1000,
                "status": "initiated" if success else "failed"
            }
            
        except Exception as e:
            agent_results["workflow_execution"] = {
                "success": False,
                "error": str(e)
            }
            
        return agent_results
        
    async def _test_model_inference(self) -> Dict[str, Any]:
        """Test AI model inference performance"""
        logger.info("Testing model inference performance")
        
        # Test different model endpoints if available
        model_results = {}
        
        # Test model list
        try:
            start_time = time.time()
            response = await self._make_request("GET", "/api/v1/models/")
            end_time = time.time()
            
            success = 200 <= response.status < 300
            model_results["model_list"] = {
                "success": success,
                "response_time": (end_time - start_time) * 1000
            }
            
        except Exception as e:
            model_results["model_list"] = {
                "success": False,
                "error": str(e)
            }
            
        # Test coordinator thinking (if available)
        try:
            think_payload = {
                "input_data": {"text": "Performance test: What is 2+2?"},
                "reasoning_type": "deductive"
            }
            
            start_time = time.time()
            response = await self._make_request(
                "POST",
                "/api/v1/coordinator/think",
                json=think_payload
            )
            end_time = time.time()
            
            success = 200 <= response.status < 300
            model_results["coordinator_inference"] = {
                "success": success,
                "inference_time": (end_time - start_time) * 1000
            }
            
        except Exception as e:
            model_results["coordinator_inference"] = {
                "success": False,
                "error": str(e)
            }
            
        return model_results
        
    async def _test_database_performance(self) -> Dict[str, Any]:
        """Test database performance through API calls"""
        logger.info("Testing database performance")
        
        # Test multiple rapid requests to simulate database load
        db_results = {}
        
        # Rapid sequential requests
        sequential_times = []
        for i in range(20):
            start_time = time.time()
            try:
                response = await self._make_request("GET", "/api/v1/system/status")
                success = 200 <= response.status < 300
            except:
                success = False
            end_time = time.time()
            
            if success:
                sequential_times.append((end_time - start_time) * 1000)
                
        # Concurrent requests  
        concurrent_results = []
        
        async def concurrent_db_request():
            start_time = time.time()
            try:
                response = await self._make_request("GET", "/api/v1/system/status")
                success = 200 <= response.status < 300
            except:
                success = False
            end_time = time.time()
            return (end_time - start_time) * 1000 if success else None
            
        concurrent_times = await asyncio.gather(*[
            concurrent_db_request() for _ in range(10)
        ])
        concurrent_times = [t for t in concurrent_times if t is not None]
        
        db_results = {
            "sequential_avg": statistics.mean(sequential_times) if sequential_times else 0,
            "concurrent_avg": statistics.mean(concurrent_times) if concurrent_times else 0,
            "sequential_p95": np.percentile(sequential_times, 95) if sequential_times else 0,
            "concurrent_p95": np.percentile(concurrent_times, 95) if concurrent_times else 0,
            "sequential_success_rate": len(sequential_times) / 20 * 100,
            "concurrent_success_rate": len(concurrent_times) / 10 * 100
        }
        
        return db_results
        
    async def _test_resource_utilization(self) -> Dict[str, Any]:
        """Test resource utilization under different loads"""
        logger.info("Testing resource utilization")
        
        # Monitor resources during different load levels
        resource_results = {}
        
        # Baseline resource usage
        baseline_cpu = psutil.cpu_percent(interval=1)
        baseline_memory = psutil.virtual_memory().percent
        baseline_disk = psutil.disk_usage('/').percent
        
        resource_results["baseline"] = {
            "cpu_percent": baseline_cpu,
            "memory_percent": baseline_memory,
            "disk_percent": baseline_disk
        }
        
        # Load test with resource monitoring
        load_levels = [10, 50, 100]
        
        for load_level in load_levels:
            # Apply load
            async def resource_load_request():
                try:
                    response = await self._make_request("GET", "/api/v1/system/status")
                    return response.status == 200
                except:
                    return False
                    
            start_time = time.time()
            
            # Monitor resources during load
            resource_samples = []
            
            async def monitor_resources():
                for _ in range(10):  # 10 samples over load period
                    resource_samples.append({
                        "cpu": psutil.cpu_percent(),
                        "memory": psutil.virtual_memory().percent,
                        "disk": psutil.disk_usage('/').percent,
                        "timestamp": time.time()
                    })
                    await asyncio.sleep(0.5)
                    
            # Run load and monitoring concurrently
            monitor_task = asyncio.create_task(monitor_resources())
            load_results = await asyncio.gather(*[
                resource_load_request() for _ in range(load_level)
            ])
            await monitor_task
            
            end_time = time.time()
            
            # Calculate resource statistics
            cpu_values = [s["cpu"] for s in resource_samples]
            memory_values = [s["memory"] for s in resource_samples]
            
            resource_results[f"load_level_{load_level}"] = {
                "duration": end_time - start_time,
                "success_rate": sum(load_results) / len(load_results) * 100,
                "avg_cpu": statistics.mean(cpu_values),
                "max_cpu": max(cpu_values),
                "avg_memory": statistics.mean(memory_values),
                "max_memory": max(memory_values),
                "cpu_increase": max(cpu_values) - baseline_cpu,
                "memory_increase": max(memory_values) - baseline_memory
            }
            
            await asyncio.sleep(2)  # Recovery between tests
            
        return resource_results
        
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> aiohttp.ClientResponse:
        """Make HTTP request with proper error handling"""
        url = f"{self.base_url}{endpoint}"
        timeout = kwargs.pop("timeout", 10)
        
        try:
            async with self.session.request(
                method, 
                url, 
                timeout=aiohttp.ClientTimeout(total=timeout),
                **kwargs
            ) as response:
                # Read response body to ensure complete request
                await response.read()
                return response
        except Exception as e:
            logger.debug(f"Request failed {method} {url}: {e}")
            raise
            
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        if not self.results:
            return {"error": "No test results available"}
            
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        
        durations = [r.duration for r in self.results]
        
        summary = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": successful_tests / total_tests * 100,
            "avg_duration": statistics.mean(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "p50_duration": np.percentile(durations, 50),
            "p95_duration": np.percentile(durations, 95),
            "p99_duration": np.percentile(durations, 99),
            "test_components": list(set(r.component for r in self.results)),
            "benchmark_compliance": self._check_benchmark_compliance()
        }
        
        return summary
        
    def _check_benchmark_compliance(self) -> Dict[str, bool]:
        """Check if results meet benchmark requirements"""
        compliance = {}
        
        # Group results by test endpoint
        endpoint_results = {}
        for result in self.results:
            if result.test_name.startswith("latency_"):
                endpoint = result.test_name.replace("latency_", "")
                if endpoint not in endpoint_results:
                    endpoint_results[endpoint] = []
                endpoint_results[endpoint].append(result)
                
        # Check compliance for each endpoint
        for endpoint, results in endpoint_results.items():
            if endpoint in self.thresholds:
                threshold = self.thresholds[endpoint]
                durations = [r.duration for r in results]
                success_rate = sum(1 for r in results if r.success) / len(results) * 100
                
                compliance[endpoint] = {
                    "p50_compliant": np.percentile(durations, 50) <= threshold.p50_ms,
                    "p95_compliant": np.percentile(durations, 95) <= threshold.p95_ms,
                    "p99_compliant": np.percentile(durations, 99) <= threshold.p99_ms,
                    "success_rate_compliant": success_rate >= threshold.success_rate,
                    "overall_compliant": all([
                        np.percentile(durations, 50) <= threshold.p50_ms,
                        np.percentile(durations, 95) <= threshold.p95_ms,
                        np.percentile(durations, 99) <= threshold.p99_ms,
                        success_rate >= threshold.success_rate
                    ])
                }
                
        return compliance
        
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed performance report"""
        report = f"""
# SutazAI Performance Test Report

**Generated**: {datetime.now().isoformat()}
**Test Duration**: {results.get('duration', 0):.2f} seconds

## Executive Summary

{self._generate_executive_summary(results)}

## Detailed Results

### API Latency Performance
{self._format_latency_results(results.get('tests', {}).get('api_latency', {}))}

### Load Testing Results  
{self._format_load_results(results.get('tests', {}).get('load_testing', {}))}

### Stress Testing Results
{self._format_stress_results(results.get('tests', {}).get('stress_testing', {}))}

### AI Performance Analysis
{self._format_ai_analysis(results.get('ai_analysis', {}))}

## Recommendations
{self._format_recommendations(results.get('recommendations', []))}

## Benchmark Compliance
{self._format_compliance_status(results.get('summary', {}).get('benchmark_compliance', {}))}

---
*Report generated by SutazAI Testing QA Validator*
        """
        
        return report.strip()
        
    def _generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary"""
        summary = results.get('summary', {})
        ai_analysis = results.get('ai_analysis', {})
        
        success_rate = summary.get('success_rate', 0)
        avg_duration = summary.get('avg_duration', 0)
        total_tests = summary.get('total_tests', 0)
        
        performance_grade = "A" if success_rate > 95 else "B" if success_rate > 85 else "C"
        
        return f"""
**Overall Performance Grade**: {performance_grade}
**Success Rate**: {success_rate:.1f}%
**Average Response Time**: {avg_duration:.1f}ms
**Total Tests Executed**: {total_tests}
**Anomalies Detected**: {ai_analysis.get('anomalies_detected', 0)}
**Performance Trend**: {ai_analysis.get('performance_trend', 'unknown').title()}
        """
        
    def _format_latency_results(self, latency_results: Dict[str, Any]) -> str:
        """Format latency test results"""
        if not latency_results:
            return "No latency test results available."
            
        output = []
        for endpoint, results in latency_results.items():
            compliance = results.get('threshold_compliance', {})
            status = "✅ PASS" if all(compliance.values()) else "❌ FAIL"
            
            output.append(f"""
**{endpoint}** {status}
- P50: {results['p50']:.1f}ms (Threshold: {self.thresholds.get(endpoint, BenchmarkThresholds(0,0,0,0,0,0)).p50_ms}ms)
- P95: {results['p95']:.1f}ms (Threshold: {self.thresholds.get(endpoint, BenchmarkThresholds(0,0,0,0,0,0)).p95_ms}ms)
- P99: {results['p99']:.1f}ms (Threshold: {self.thresholds.get(endpoint, BenchmarkThresholds(0,0,0,0,0,0)).p99_ms}ms)
- Success Rate: {results['success_rate']:.1f}%
            """)
            
        return "\n".join(output)
        
    def _format_load_results(self, load_results: Dict[str, Any]) -> str:
        """Format load test results"""
        if not load_results:
            return "No load test results available."
            
        output = []
        for level, results in load_results.items():
            throughput_status = "✅" if results['throughput_rps'] > 50 else "⚠️"
            latency_status = "✅" if results['p95_response_time'] < 1000 else "⚠️"
            
            output.append(f"""
**Load Level {level}** 
- Throughput: {results['throughput_rps']:.1f} req/s {throughput_status}
- P95 Response Time: {results['p95_response_time']:.1f}ms {latency_status}
- Success Rate: {results['success_rate']:.1f}%
            """)
            
        return "\n".join(output)
        
    def _format_stress_results(self, stress_results: Dict[str, Any]) -> str:
        """Format stress test results"""
        if not stress_results:
            return "No stress test results available."
            
        breaking_point = stress_results.get('breaking_point', 'Not reached')
        max_users = stress_results.get('max_concurrent_users', 0)
        
        return f"""
**Maximum Concurrent Users**: {max_users}
**Breaking Point**: {breaking_point}
**System Resilience**: {"High" if breaking_point == 'Not reached' or breaking_point > 1000 else "Medium" if breaking_point > 500 else "Low"}
        """
        
    def _format_ai_analysis(self, ai_analysis: Dict[str, Any]) -> str:
        """Format AI analysis results"""
        if not ai_analysis:
            return "No AI analysis available."
            
        return f"""
**Performance Trend**: {ai_analysis.get('performance_trend', 'unknown').title()}
**Anomaly Detection**: {ai_analysis.get('anomalies_detected', 0)} anomalies ({ai_analysis.get('anomaly_percentage', 0):.1f}%)
**Success Rate**: {ai_analysis.get('success_rate', 0):.1f}%
**Average Duration**: {ai_analysis.get('avg_duration', 0):.1f}ms
**P95 Duration**: {ai_analysis.get('p95_duration', 0):.1f}ms
        """
        
    def _format_recommendations(self, recommendations: List[str]) -> str:
        """Format recommendations"""
        if not recommendations:
            return "No specific recommendations at this time."
            
        return "\n".join(f"- {rec}" for rec in recommendations)
        
    def _format_compliance_status(self, compliance: Dict[str, Any]) -> str:
        """Format benchmark compliance status"""
        if not compliance:
            return "No compliance data available."
            
        output = []
        for endpoint, status in compliance.items():
            overall_status = "✅ COMPLIANT" if status.get('overall_compliant', False) else "❌ NON-COMPLIANT"
            output.append(f"**{endpoint}**: {overall_status}")
            
        return "\n".join(output)

# Test execution functions for pytest integration
@pytest.mark.asyncio
@pytest.mark.performance
async def test_api_performance():
    """Pytest wrapper for API performance testing"""
    suite = PerformanceTestSuite()
    await suite.initialize()
    
    try:
        results = await suite._test_api_latency()
        
        # Assert that critical endpoints meet thresholds
        for endpoint, result in results.items():
            if endpoint in suite.thresholds:
                threshold = suite.thresholds[endpoint]
                
                assert result['p95'] <= threshold.p95_ms, f"{endpoint} P95 latency {result['p95']}ms exceeds threshold {threshold.p95_ms}ms"
                assert result['success_rate'] >= threshold.success_rate, f"{endpoint} success rate {result['success_rate']}% below threshold {threshold.success_rate}%"
                
    finally:
        await suite.cleanup()

@pytest.mark.asyncio
@pytest.mark.load
async def test_load_performance():
    """Pytest wrapper for load testing"""
    suite = PerformanceTestSuite()
    await suite.initialize()
    
    try:
        results = await suite._test_load_performance()
        
        # Assert that system handles expected load
        for level, result in results.items():
            if level <= 100:  # Expected normal load
                assert result['success_rate'] >= 95, f"Success rate {result['success_rate']}% too low at load level {level}"
                assert result['p95_response_time'] <= 2000, f"P95 response time {result['p95_response_time']}ms too high at load level {level}"
                
    finally:
        await suite.cleanup()

if __name__ == "__main__":
    async def main():
        """Main execution function"""
        suite = PerformanceTestSuite()
        await suite.initialize()
        
        try:
            logger.info("Starting SutazAI Performance Test Suite")
            results = await suite.run_comprehensive_suite()
            
            # Save results to file
            results_file = f"/opt/sutazaiapp/data/performance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
                
            # Generate and save report
            report = suite.generate_performance_report(results)
            report_file = results_file.replace('.json', '_report.md')
            with open(report_file, 'w') as f:
                f.write(report)
                
            logger.info(f"Performance test completed. Results saved to {results_file}")
            logger.info(f"Report saved to {report_file}")
            
            print("\n" + "="*50)
            print("PERFORMANCE TEST SUMMARY")
            print("="*50)
            print(f"Success Rate: {results.get('summary', {}).get('success_rate', 0):.1f}%")
            print(f"Average Duration: {results.get('summary', {}).get('avg_duration', 0):.1f}ms")
            print(f"Total Tests: {results.get('summary', {}).get('total_tests', 0)}")
            print(f"Anomalies: {results.get('ai_analysis', {}).get('anomalies_detected', 0)}")
            print(f"Performance Trend: {results.get('ai_analysis', {}).get('performance_trend', 'unknown').title()}")
            
            if results.get('recommendations'):
                print("\nTop Recommendations:")
                for i, rec in enumerate(results['recommendations'][:3], 1):
                    print(f"{i}. {rec}")
                    
        except Exception as e:
            logger.error(f"Performance test suite failed: {e}")
            return 1
            
        finally:
            await suite.cleanup()
            
        return 0
    
    # Run the test suite
    import sys
    sys.exit(asyncio.run(main()))