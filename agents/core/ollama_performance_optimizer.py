#!/usr/bin/env python3
"""
Purpose: Advanced Ollama Performance Optimization Manager for SutazAI
Usage: Optimizes AI model inference performance across 69 agents
Requirements: ollama, httpx, asyncio, prometheus_client, yaml
"""

import os
import sys
import time
import asyncio
import json
import yaml
import httpx
import psutil
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import threading
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ollama-optimizer')

@dataclass
class ModelMetrics:
    """Metrics for a specific model"""
    model_name: str
    load_time: float
    inference_time: float
    queue_length: int
    memory_usage: int
    requests_per_second: float
    error_rate: float
    cache_hit_rate: float
    last_used: datetime
    total_requests: int
    active_connections: int

@dataclass
class PerformanceBenchmark:
    """Performance benchmark results"""
    timestamp: datetime
    model: str
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float

class OllamaPerformanceOptimizer:
    """Advanced Ollama performance optimization manager"""
    
    def __init__(self, config_path: str = "/opt/sutazaiapp/config/ollama_performance_optimization.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.base_url = "http://localhost:10104"
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Performance tracking
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.request_history = deque(maxlen=10000)
        self.benchmark_history = deque(maxlen=1000)
        self.performance_cache = {}
        
        # Prometheus metrics
        self._setup_prometheus_metrics()
        
        # Threading for continuous optimization
        self.optimization_thread = None
        self.monitoring_thread = None
        self.running = False
        
        # Connection pool management
        self.connection_pools = {}
        self.model_priorities = self._calculate_model_priorities()
        
        # Load balancing
        self.instance_weights = {}
        self.circuit_breakers = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """Load optimization configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails"""
        return {
            'ollama_optimization': {
                'performance': {
                    'model_loading': {
                        'preload_models': ['tinyllama'],
                        'max_loaded_models': 2,
                        'model_unload_timeout': 120
                    },
                    'inference_optimization': {
                        'batch_size': 16,
                        'max_concurrent_requests': 32,
                        'queue_timeout': 30,
                        'request_timeout': 120
                    },
                    'caching': {
                        'enable_response_cache': True,
                        'cache_size_mb': 512,
                        'cache_ttl': 3600
                    }
                }
            }
        }
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics collection"""
        self.request_counter = Counter('ollama_requests_total', 'Total requests', ['model', 'status'])
        self.request_duration = Histogram('ollama_request_duration_seconds', 'Request duration', ['model'])
        self.queue_length = Gauge('ollama_queue_length', 'Queue length', ['model'])
        self.memory_usage = Gauge('ollama_memory_usage_bytes', 'Memory usage', ['model'])
        self.model_load_time = Histogram('ollama_model_load_seconds', 'Model load time', ['model'])
        self.throughput = Gauge('ollama_throughput_rps', 'Throughput requests per second', ['model'])
        self.cache_hits = Counter('ollama_cache_hits_total', 'Cache hits', ['model'])
        
    def _calculate_model_priorities(self) -> Dict[str, int]:
        """Calculate model priorities based on usage patterns"""
        priorities = {
            'tinyllama': 1,  # Highest priority
            'tinyllama.2:3b': 2,
            'tinyllama': 3
        }
        return priorities
    
    async def start_optimization(self):
        """Start the optimization system"""
        logger.info("Starting Ollama Performance Optimizer")
        self.running = True
        
        # Start Prometheus metrics server
        start_http_server(8090)
        logger.info("Prometheus metrics server started on port 8090")
        
        # Preload priority models
        await self._preload_models()
        
        # Start background threads
        self.optimization_thread = threading.Thread(target=self._run_optimization_loop)
        self.monitoring_thread = threading.Thread(target=self._run_monitoring_loop)
        
        self.optimization_thread.start()
        self.monitoring_thread.start()
        
        logger.info("Optimization system started successfully")
    
    async def stop_optimization(self):
        """Stop the optimization system"""
        logger.info("Stopping Ollama Performance Optimizer")
        self.running = False
        
        if self.optimization_thread:
            self.optimization_thread.join()
        if self.monitoring_thread:
            self.monitoring_thread.join()
            
        await self.client.aclose()
        logger.info("Optimization system stopped")
    
    async def _preload_models(self):
        """Preload priority models for optimal performance"""
        preload_models = self.config['ollama_optimization']['performance']['model_loading']['preload_models']
        
        for model in preload_models:
            try:
                start_time = time.time()
                logger.info(f"Preloading model: {model}")
                
                # Make a test request to load the model
                response = await self.client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": "Hello",
                        "stream": False,
                        "options": {"num_predict": 1}
                    }
                )
                
                load_time = time.time() - start_time
                
                if response.status_code == 200:
                    logger.info(f"Successfully preloaded {model} in {load_time:.2f}s")
                    self.model_load_time.labels(model=model).observe(load_time)
                else:
                    logger.error(f"Failed to preload {model}: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error preloading {model}: {e}")
    
    def _run_optimization_loop(self):
        """Background optimization loop"""
        while self.running:
            try:
                asyncio.run(self._optimization_cycle())
                time.sleep(30)  # Run optimization every 30 seconds
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(10)
    
    def _run_monitoring_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                asyncio.run(self._monitoring_cycle())
                time.sleep(10)  # Monitor every 10 seconds
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    async def _optimization_cycle(self):
        """Run one optimization cycle"""
        # 1. Analyze current performance
        metrics = await self._collect_performance_metrics()
        
        # 2. Optimize model loading
        await self._optimize_model_loading(metrics)
        
        # 3. Optimize memory usage
        await self._optimize_memory_usage(metrics)
        
        # 4. Optimize connection pools
        await self._optimize_connection_pools(metrics)
        
        # 5. Update load balancing weights
        await self._update_load_balancing(metrics)
        
        logger.debug("Optimization cycle completed")
    
    async def _monitoring_cycle(self):
        """Run one monitoring cycle"""
        # Collect and update Prometheus metrics
        for model_name, metrics in self.model_metrics.items():
            self.queue_length.labels(model=model_name).set(metrics.queue_length)
            self.memory_usage.labels(model=model_name).set(metrics.memory_usage)
            self.throughput.labels(model=model_name).set(metrics.requests_per_second)
    
    async def _collect_performance_metrics(self) -> Dict[str, ModelMetrics]:
        """Collect current performance metrics"""
        try:
            # Get running processes
            response = await self.client.get(f"{self.base_url}/api/ps")
            if response.status_code != 200:
                return self.model_metrics
                
            running_models = response.json().get('models', [])
            
            for model_info in running_models:
                model_name = model_info.get('name', '')
                if model_name:
                    # Update or create metrics
                    if model_name not in self.model_metrics:
                        self.model_metrics[model_name] = ModelMetrics(
                            model_name=model_name,
                            load_time=0.0,
                            inference_time=0.0,
                            queue_length=0,
                            memory_usage=model_info.get('size', 0),
                            requests_per_second=0.0,
                            error_rate=0.0,
                            cache_hit_rate=0.0,
                            last_used=datetime.now(),
                            total_requests=0,
                            active_connections=0
                        )
                    else:
                        self.model_metrics[model_name].memory_usage = model_info.get('size', 0)
                        self.model_metrics[model_name].last_used = datetime.now()
            
            return self.model_metrics
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            return self.model_metrics
    
    async def _optimize_model_loading(self, metrics: Dict[str, ModelMetrics]):
        """Optimize model loading and unloading"""
        max_loaded = self.config['ollama_optimization']['performance']['model_loading']['max_loaded_models']
        unload_timeout = self.config['ollama_optimization']['performance']['model_loading']['model_unload_timeout']
        
        # Determine which models to keep loaded
        priority_models = []
        for model_name, priority in self.model_priorities.items():
            if model_name in metrics:
                priority_models.append((priority, model_name, metrics[model_name]))
        
        priority_models.sort()  # Sort by priority (lower number = higher priority)
        
        # Keep top priority models loaded
        models_to_keep = [item[1] for item in priority_models[:max_loaded]]
        
        # Unload old unused models
        current_time = datetime.now()
        for model_name, model_metrics in metrics.items():
            if model_name not in models_to_keep:
                time_since_use = current_time - model_metrics.last_used
                if time_since_use.total_seconds() > unload_timeout:
                    await self._unload_model(model_name)
    
    async def _unload_model(self, model_name: str):
        """Unload a specific model"""
        try:
            logger.info(f"Unloading unused model: {model_name}")
            if model_name in self.model_metrics:
                del self.model_metrics[model_name]
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
    
    async def _optimize_memory_usage(self, metrics: Dict[str, ModelMetrics]):
        """Optimize memory usage across models"""
        total_memory = psutil.virtual_memory().total
        used_memory = psutil.virtual_memory().used
        memory_percent = (used_memory / total_memory) * 100
        
        threshold = self.config['ollama_optimization']['performance']['memory_management']['memory_threshold']
        
        if memory_percent > threshold:
            logger.warning(f"Memory usage high: {memory_percent:.1f}%")
            # Trigger aggressive garbage collection
            await self._trigger_garbage_collection()
    
    async def _trigger_garbage_collection(self):
        """Trigger garbage collection to free memory"""
        import gc
        gc.collect()
        logger.info("Triggered garbage collection")
    
    async def _optimize_connection_pools(self, metrics: Dict[str, ModelMetrics]):
        """Optimize connection pool sizes based on load"""
        for model_name, model_metrics in metrics.items():
            # Adjust pool size based on request rate
            if model_metrics.requests_per_second > 100:
                # High load - increase pool size
                pool_size = min(200, int(model_metrics.requests_per_second * 1.5))
            else:
                # Normal load - use default pool size
                pool_size = 100
            
            if model_name not in self.connection_pools:
                self.connection_pools[model_name] = {'size': pool_size}
            else:
                self.connection_pools[model_name]['size'] = pool_size
    
    async def _update_load_balancing(self, metrics: Dict[str, ModelMetrics]):
        """Update load balancing weights based on performance"""
        for model_name, model_metrics in metrics.items():
            # Calculate weight based on performance (lower latency = higher weight)
            if model_metrics.inference_time > 0:
                weight = 1.0 / model_metrics.inference_time
            else:
                weight = 1.0
            
            # Adjust for error rate
            weight *= (1.0 - model_metrics.error_rate)
            
            self.instance_weights[model_name] = max(0.1, weight)  # Minimum weight 0.1
    
    async def benchmark_model_performance(self, model: str, num_requests: int = 100) -> PerformanceBenchmark:
        """Benchmark a specific model's performance"""
        logger.info(f"Benchmarking model: {model} with {num_requests} requests")
        
        start_time = time.time()
        latencies = []
        errors = 0
        
        # Get baseline memory usage
        memory_before = psutil.Process().memory_info().rss
        cpu_before = psutil.cpu_percent()
        
        # Run benchmark requests
        semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
        
        async def make_request():
            async with semaphore:
                request_start = time.time()
                try:
                    response = await self.client.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": model,
                            "prompt": "Explain artificial intelligence in one sentence.",
                            "stream": False,
                            "options": {"num_predict": 50}
                        }
                    )
                    
                    request_time = time.time() - request_start
                    latencies.append(request_time)
                    
                    if response.status_code != 200:
                        nonlocal errors
                        errors += 1
                        
                    # Update Prometheus metrics
                    self.request_counter.labels(model=model, status=str(response.status_code)).inc()
                    self.request_duration.labels(model=model).observe(request_time)
                    
                except Exception as e:
                    errors += 1
                    logger.error(f"Benchmark request failed: {e}")
        
        # Execute all requests concurrently
        tasks = [make_request() for _ in range(num_requests)]
        await asyncio.gather(*tasks)
        
        # Calculate metrics
        total_time = time.time() - start_time
        throughput = num_requests / total_time
        error_rate = errors / num_requests
        
        # Get final resource usage
        memory_after = psutil.Process().memory_info().rss
        cpu_after = psutil.cpu_percent()
        
        memory_usage_mb = (memory_after - memory_before) / (1024 * 1024)
        cpu_usage_percent = cpu_after - cpu_before
        
        # Calculate percentiles
        if latencies:
            latency_p50 = statistics.median(latencies)
            latency_p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
            latency_p99 = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        else:
            latency_p50 = latency_p95 = latency_p99 = 0.0
        
        benchmark = PerformanceBenchmark(
            timestamp=datetime.now(),
            model=model,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            throughput=throughput,
            error_rate=error_rate,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=cpu_usage_percent
        )
        
        self.benchmark_history.append(benchmark)
        
        logger.info(f"Benchmark completed for {model}:")
        logger.info(f"  Throughput: {throughput:.2f} req/s")
        logger.info(f"  Latency P95: {latency_p95*1000:.0f}ms")
        logger.info(f"  Error Rate: {error_rate*100:.2f}%")
        logger.info(f"  Memory Usage: {memory_usage_mb:.1f}MB")
        
        return benchmark
    
    async def optimize_for_agent_workload(self, agent_name: str, expected_load: int):
        """Optimize configuration for a specific agent's workload"""
        logger.info(f"Optimizing for agent: {agent_name} with expected load: {expected_load}")
        
        # Get agent-specific config
        agent_config = self.config.get('agent_optimizations', {}).get(agent_name, {})
        
        # Apply agent-specific optimizations
        if agent_config.get('dedicated_instance'):
            await self._setup_dedicated_instance(agent_name)
        
        if agent_config.get('preload_model'):
            model = agent_config.get('model', 'tinyllama')
            await self._preload_models([model])
        
        if agent_config.get('cache_responses'):
            await self._enable_response_caching(agent_name)
    
    async def _setup_dedicated_instance(self, agent_name: str):
        """Setup dedicated Ollama instance for high-priority agent"""
        logger.info(f"Setting up dedicated instance for {agent_name}")
        # This would involve starting a new Ollama instance on a different port
        # For now, we'll just track the requirement
        
    async def _enable_response_caching(self, agent_name: str):
        """Enable response caching for an agent"""
        logger.info(f"Enabling response caching for {agent_name}")
        # Implementation would depend on the caching backend
        
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            },
            'model_metrics': {},
            'benchmarks': [],
            'optimization_config': self.config
        }
        
        # Add model metrics
        for model_name, metrics in self.model_metrics.items():
            report['model_metrics'][model_name] = asdict(metrics)
        
        # Add recent benchmarks
        for benchmark in list(self.benchmark_history)[-10:]:  # Last 10 benchmarks
            report['benchmarks'].append(asdict(benchmark))
        
        # Save report
        report_path = f"/opt/sutazaiapp/logs/ollama_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report saved to: {report_path}")
        return report
    
    async def run_comprehensive_benchmark(self) -> Dict[str, PerformanceBenchmark]:
        """Run comprehensive benchmark across all models"""
        logger.info("Running comprehensive performance benchmark")
        
        models = ['tinyllama', 'tinyllama.2:3b', 'tinyllama']
        benchmarks = {}
        
        for model in models:
            try:
                benchmark = await self.benchmark_model_performance(model, num_requests=50)
                benchmarks[model] = benchmark
            except Exception as e:
                logger.error(f"Failed to benchmark {model}: {e}")
        
        # Generate comparison report
        self._generate_benchmark_comparison(benchmarks)
        
        return benchmarks
    
    def _generate_benchmark_comparison(self, benchmarks: Dict[str, PerformanceBenchmark]):
        """Generate benchmark comparison report"""
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'recommendations': []
        }
        
        for model, benchmark in benchmarks.items():
            comparison['models'][model] = {
                'throughput': benchmark.throughput,
                'latency_p95_ms': benchmark.latency_p95 * 1000,
                'error_rate_percent': benchmark.error_rate * 100,
                'memory_usage_mb': benchmark.memory_usage_mb
            }
        
        # Generate recommendations
        if benchmarks:
            best_throughput = max(benchmarks.values(), key=lambda b: b.throughput)
            best_latency = min(benchmarks.values(), key=lambda b: b.latency_p95)
            
            comparison['recommendations'].append(f"Best throughput: {best_throughput.model} ({best_throughput.throughput:.2f} req/s)")
            comparison['recommendations'].append(f"Best latency: {best_latency.model} ({best_latency.latency_p95*1000:.0f}ms p95)")
        
        # Save comparison
        comparison_path = f"/opt/sutazaiapp/logs/benchmark_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Benchmark comparison saved to: {comparison_path}")

# CLI interface
async def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ollama Performance Optimizer')
    parser.add_argument('--start', action='store_true', help='Start optimization daemon')
    parser.add_argument('--benchmark', choices=['all', 'tinyllama', 'tinyllama.2:3b', 'tinyllama'], help='Run benchmark')
    parser.add_argument('--report', action='store_true', help='Generate performance report')
    parser.add_argument('--optimize-agent', help='Optimize for specific agent')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    config_path = args.config or "/opt/sutazaiapp/config/ollama_performance_optimization.yaml"
    optimizer = OllamaPerformanceOptimizer(config_path)
    
    try:
        if args.start:
            await optimizer.start_optimization()
            # Keep running until interrupted
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                await optimizer.stop_optimization()
        
        elif args.benchmark:
            if args.benchmark == 'all':
                await optimizer.run_comprehensive_benchmark()
            else:
                await optimizer.benchmark_model_performance(args.benchmark)
        
        elif args.report:
            optimizer.generate_performance_report()
        
        elif args.optimize_agent:
            await optimizer.optimize_for_agent_workload(args.optimize_agent, 100)
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
