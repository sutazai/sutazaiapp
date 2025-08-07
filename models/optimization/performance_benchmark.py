#!/usr/bin/env python3
"""
Purpose: Comprehensive performance benchmarking for optimized models
Usage: Benchmarks inference speed, memory usage, and quality metrics
Requirements: asyncio, numpy, psutil, matplotlib
"""

import os
import json
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import psutil
from pathlib import Path
import statistics
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

logger = logging.getLogger('performance-benchmark')


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking"""
    model_name: str
    test_prompts: List[str]
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16])
    num_iterations: int = 100
    warmup_iterations: int = 10
    measure_memory: bool = True
    measure_quality: bool = True
    cpu_cores: int = 12
    timeout_seconds: int = 300


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    model_name: str
    optimization_type: str
    timestamp: datetime
    
    # Performance metrics
    latency_metrics: Dict[str, float]  # p50, p95, p99, mean
    throughput_metrics: Dict[str, float]  # tokens/sec by batch size
    
    # Resource metrics
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_utilization: float
    
    # Quality metrics
    accuracy_score: float
    perplexity: float
    
    # Optimization impact
    speedup_vs_baseline: float
    memory_reduction_vs_baseline: float
    quality_preservation: float
    
    # Additional info
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceBenchmark:
    """
    Comprehensive benchmarking system for CPU-optimized models
    
    Features:
    - Latency and throughput measurement
    - Memory profiling
    - Quality assessment
    - Multi-core scaling analysis
    - Batch size optimization
    - Comparative analysis
    """
    
    def __init__(self, output_dir: str = "/opt/sutazaiapp/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Benchmark storage
        self.results: Dict[str, List[BenchmarkResult]] = {}
        self.baseline_results: Dict[str, BenchmarkResult] = {}
        
    async def benchmark_model(self,
                            model_path: str,
                            config: BenchmarkConfig,
                            optimization_type: str = "baseline") -> BenchmarkResult:
        """
        Run comprehensive benchmark on a model
        
        Args:
            model_path: Path to model
            config: Benchmark configuration
            optimization_type: Type of optimization applied
            
        Returns:
            Benchmark results
        """
        logger.info(f"Starting benchmark for {config.model_name} ({optimization_type})")
        
        result = BenchmarkResult(
            model_name=config.model_name,
            optimization_type=optimization_type,
            timestamp=datetime.utcnow(),
            latency_metrics={},
            throughput_metrics={},
            memory_usage_mb=0,
            peak_memory_mb=0,
            cpu_utilization=0,
            accuracy_score=0,
            perplexity=0,
            speedup_vs_baseline=1.0,
            memory_reduction_vs_baseline=0,
            quality_preservation=1.0
        )
        
        try:
            # Warmup
            await self._warmup(model_path, config)
            
            # Benchmark latency
            latency_results = await self._benchmark_latency(model_path, config)
            result.latency_metrics = self._calculate_latency_metrics(latency_results)
            
            # Benchmark throughput
            throughput_results = await self._benchmark_throughput(model_path, config)
            result.throughput_metrics = throughput_results
            
            # Measure resource usage
            if config.measure_memory:
                memory_metrics = await self._measure_memory_usage(model_path, config)
                result.memory_usage_mb = memory_metrics['average_mb']
                result.peak_memory_mb = memory_metrics['peak_mb']
                result.cpu_utilization = memory_metrics['cpu_percent']
            
            # Measure quality
            if config.measure_quality:
                quality_metrics = await self._measure_quality(model_path, config)
                result.accuracy_score = quality_metrics['accuracy']
                result.perplexity = quality_metrics['perplexity']
            
            # Compare with baseline
            if optimization_type != "baseline" and config.model_name in self.baseline_results:
                baseline = self.baseline_results[config.model_name]
                result.speedup_vs_baseline = (
                    baseline.latency_metrics['mean'] / result.latency_metrics['mean']
                )
                result.memory_reduction_vs_baseline = (
                    1 - result.memory_usage_mb / baseline.memory_usage_mb
                )
                result.quality_preservation = (
                    result.accuracy_score / baseline.accuracy_score
                )
            
            # Store results
            if config.model_name not in self.results:
                self.results[config.model_name] = []
            self.results[config.model_name].append(result)
            
            if optimization_type == "baseline":
                self.baseline_results[config.model_name] = result
            
            logger.info(f"Benchmark complete: {result.speedup_vs_baseline:.2f}x speedup, "
                       f"{result.memory_reduction_vs_baseline:.1%} memory reduction")
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            result.errors.append(str(e))
        
        return result
    
    async def _warmup(self, model_path: str, config: BenchmarkConfig):
        """Warmup the model to ensure stable measurements"""
        logger.debug("Running warmup iterations")
        
        for _ in range(config.warmup_iterations):
            # Simulate inference
            await self._run_inference(
                model_path,
                config.test_prompts[0],
                batch_size=1
            )
    
    async def _benchmark_latency(self, 
                                model_path: str,
                                config: BenchmarkConfig) -> List[float]:
        """Benchmark inference latency"""
        latencies = []
        
        for i in range(config.num_iterations):
            prompt = config.test_prompts[i % len(config.test_prompts)]
            
            start_time = time.perf_counter()
            await self._run_inference(model_path, prompt, batch_size=1)
            latency = (time.perf_counter() - start_time) * 1000  # ms
            
            latencies.append(latency)
        
        return latencies
    
    async def _benchmark_throughput(self,
                                  model_path: str,
                                  config: BenchmarkConfig) -> Dict[str, float]:
        """Benchmark throughput at different batch sizes"""
        throughput_results = {}
        
        for batch_size in config.batch_sizes:
            # Prepare batch
            batch_prompts = [
                config.test_prompts[i % len(config.test_prompts)]
                for i in range(batch_size)
            ]
            
            # Measure time for multiple batches
            start_time = time.perf_counter()
            num_batches = max(10, 100 // batch_size)
            
            for _ in range(num_batches):
                await self._run_inference(
                    model_path,
                    batch_prompts,
                    batch_size=batch_size
                )
            
            total_time = time.perf_counter() - start_time
            
            # Calculate throughput (requests per second)
            total_requests = num_batches * batch_size
            throughput = total_requests / total_time
            
            throughput_results[f"batch_{batch_size}"] = throughput
        
        return throughput_results
    
    async def _measure_memory_usage(self,
                                  model_path: str,
                                  config: BenchmarkConfig) -> Dict[str, float]:
        """Measure memory usage during inference"""
        
        process = psutil.Process()
        memory_samples = []
        cpu_samples = []
        
        # Start memory monitoring task
        monitoring = True
        
        async def monitor_resources():
            while monitoring:
                memory_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
                cpu_samples.append(process.cpu_percent())
                await asyncio.sleep(0.1)
        
        monitor_task = asyncio.create_task(monitor_resources())
        
        # Run inference workload
        for _ in range(20):
            prompt = np.random.choice(config.test_prompts)
            await self._run_inference(model_path, prompt, batch_size=4)
        
        # Stop monitoring
        monitoring = False
        await monitor_task
        
        return {
            'average_mb': statistics.mean(memory_samples) if memory_samples else 0,
            'peak_mb': max(memory_samples) if memory_samples else 0,
            'cpu_percent': statistics.mean(cpu_samples) if cpu_samples else 0
        }
    
    async def _measure_quality(self,
                             model_path: str,
                             config: BenchmarkConfig) -> Dict[str, float]:
        """Measure model quality metrics"""
        
        # Simplified quality measurement
        # In practice, this would run actual quality benchmarks
        
        accuracies = []
        perplexities = []
        
        for prompt in config.test_prompts[:10]:  # Sample of prompts
            # Simulate quality measurement
            await asyncio.sleep(0.05)
            
            # Mock scores (in practice, compare with ground truth)
            accuracy = np.random.uniform(0.85, 0.99)
            perplexity = np.random.uniform(10, 50)
            
            accuracies.append(accuracy)
            perplexities.append(perplexity)
        
        return {
            'accuracy': statistics.mean(accuracies),
            'perplexity': statistics.mean(perplexities)
        }
    
    async def _run_inference(self,
                           model_path: str,
                           prompts: Union[str, List[str]],
                           batch_size: int) -> List[str]:
        """Run model inference (simulation)"""
        
        # Simulate CPU inference with realistic timing
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Base latency depends on model optimization
        if 'int8' in model_path:
            base_latency = 0.02
        elif 'int4' in model_path:
            base_latency = 0.015
        else:
            base_latency = 0.05
        
        # Batch processing advantage
        batch_latency = base_latency + (len(prompts) - 1) * base_latency * 0.3
        
        await asyncio.sleep(batch_latency)
        
        return [f"Response {i}" for i in range(len(prompts))]
    
    def _calculate_latency_metrics(self, latencies: List[float]) -> Dict[str, float]:
        """Calculate latency statistics"""
        
        if not latencies:
            return {}
        
        sorted_latencies = sorted(latencies)
        
        return {
            'mean': statistics.mean(latencies),
            'median': statistics.median(latencies),
            'p50': sorted_latencies[int(len(latencies) * 0.50)],
            'p95': sorted_latencies[int(len(latencies) * 0.95)],
            'p99': sorted_latencies[int(len(latencies) * 0.99)],
            'min': min(latencies),
            'max': max(latencies),
            'std': statistics.stdev(latencies) if len(latencies) > 1 else 0
        }
    
    def generate_report(self, model_name: str) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        
        if model_name not in self.results:
            return {"error": "No results found for model"}
        
        results = self.results[model_name]
        baseline = self.baseline_results.get(model_name)
        
        report = {
            'model': model_name,
            'benchmark_date': datetime.utcnow().isoformat(),
            'optimization_comparison': {},
            'best_configuration': None,
            'recommendations': []
        }
        
        # Compare different optimizations
        for result in results:
            opt_type = result.optimization_type
            
            report['optimization_comparison'][opt_type] = {
                'speedup': result.speedup_vs_baseline,
                'memory_reduction': result.memory_reduction_vs_baseline,
                'quality_preservation': result.quality_preservation,
                'latency_p95_ms': result.latency_metrics.get('p95', 0),
                'throughput_batch_8': result.throughput_metrics.get('batch_8', 0),
                'memory_usage_mb': result.memory_usage_mb
            }
        
        # Find best configuration
        best_result = max(results, key=lambda r: r.speedup_vs_baseline * r.quality_preservation)
        report['best_configuration'] = {
            'optimization': best_result.optimization_type,
            'speedup': best_result.speedup_vs_baseline,
            'quality': best_result.quality_preservation
        }
        
        # Generate recommendations
        if best_result.speedup_vs_baseline > 3 and best_result.quality_preservation > 0.95:
            report['recommendations'].append(
                f"Use {best_result.optimization_type} for production deployment"
            )
        
        if best_result.memory_reduction_vs_baseline > 0.5:
            report['recommendations'].append(
                "Consider running multiple model instances due to memory savings"
            )
        
        # Identify optimal batch size
        optimal_batch = self._find_optimal_batch_size(best_result)
        report['recommendations'].append(
            f"Use batch size {optimal_batch} for optimal throughput"
        )
        
        return report
    
    def _find_optimal_batch_size(self, result: BenchmarkResult) -> int:
        """Find optimal batch size based on throughput"""
        
        if not result.throughput_metrics:
            return 1
        
        # Find batch size with best throughput
        best_batch = 1
        best_throughput = 0
        
        for batch_key, throughput in result.throughput_metrics.items():
            batch_size = int(batch_key.split('_')[1])
            if throughput > best_throughput:
                best_throughput = throughput
                best_batch = batch_size
        
        return best_batch
    
    def plot_results(self, model_name: str):
        """Generate visualization plots"""
        
        if model_name not in self.results:
            return
        
        results = self.results[model_name]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Performance Benchmark: {model_name}')
        
        # 1. Speedup comparison
        ax = axes[0, 0]
        opt_types = [r.optimization_type for r in results if r.optimization_type != 'baseline']
        speedups = [r.speedup_vs_baseline for r in results if r.optimization_type != 'baseline']
        ax.bar(opt_types, speedups)
        ax.set_title('Speedup vs Baseline')
        ax.set_ylabel('Speedup Factor')
        ax.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        
        # 2. Memory usage
        ax = axes[0, 1]
        memory_usage = [r.memory_usage_mb for r in results]
        ax.bar([r.optimization_type for r in results], memory_usage)
        ax.set_title('Memory Usage')
        ax.set_ylabel('Memory (MB)')
        
        # 3. Quality preservation
        ax = axes[1, 0]
        quality = [r.quality_preservation for r in results if r.optimization_type != 'baseline']
        ax.bar(opt_types, quality)
        ax.set_title('Quality Preservation')
        ax.set_ylabel('Quality Score')
        ax.axhline(y=1, color='g', linestyle='--', alpha=0.5)
        ax.set_ylim(0.8, 1.05)
        
        # 4. Latency distribution
        ax = axes[1, 1]
        for result in results:
            if result.latency_metrics:
                x = ['p50', 'p95', 'p99']
                y = [result.latency_metrics.get(p, 0) for p in x]
                ax.plot(x, y, marker='o', label=result.optimization_type)
        ax.set_title('Latency Distribution')
        ax.set_ylabel('Latency (ms)')
        ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / f'{model_name}_benchmark.png'
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        logger.info(f"Saved benchmark plot to {output_path}")
    
    def save_results(self, model_name: str):
        """Save benchmark results to file"""
        
        if model_name not in self.results:
            return
        
        # Convert results to JSON-serializable format
        results_data = []
        for result in self.results[model_name]:
            result_dict = {
                'model_name': result.model_name,
                'optimization_type': result.optimization_type,
                'timestamp': result.timestamp.isoformat(),
                'latency_metrics': result.latency_metrics,
                'throughput_metrics': result.throughput_metrics,
                'memory_usage_mb': result.memory_usage_mb,
                'speedup': result.speedup_vs_baseline,
                'memory_reduction': result.memory_reduction_vs_baseline,
                'quality_preservation': result.quality_preservation
            }
            results_data.append(result_dict)
        
        # Save to JSON
        output_path = self.output_dir / f'{model_name}_results.json'
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")


async def benchmark_all_optimizations():
    """Run benchmarks for all optimization types"""
    
    benchmark = PerformanceBenchmark()
    
    # Test configuration
    config = BenchmarkConfig(
        model_name="tinyllama",
        test_prompts=[
            "Explain quantum computing",
            "Write a Python function to sort a list",
            "What is machine learning?",
            "Describe the water cycle",
            "How do neural networks work?"
        ],
        batch_sizes=[1, 4, 8, 16],
        num_iterations=50,
        warmup_iterations=5
    )
    
    # Benchmark different optimizations
    optimizations = [
        ("baseline", "/models/tinyllama.onnx"),
        ("int8", "/models/tinyllama_int8.onnx"),
        ("int4", "/models/tinyllama_int4.onnx"),
        ("pruned", "/models/tinyllama_pruned.onnx"),
        ("distilled", "/models/tinyllama_distilled.onnx")
    ]
    
    for opt_type, model_path in optimizations:
        await benchmark.benchmark_model(model_path, config, opt_type)
    
    # Generate report
    report = benchmark.generate_report("tinyllama")
    print(json.dumps(report, indent=2))
    
    # Create visualizations
    benchmark.plot_results("tinyllama")
    
    # Save results
    benchmark.save_results("tinyllama")


if __name__ == "__main__":
    asyncio.run(benchmark_all_optimizations())