"""
Comprehensive Performance Benchmarking Suite for SutazAI
Implements advanced benchmarking, profiling, and performance analysis
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json
import os
from pathlib import Path
import asyncio
import aiohttp
import sqlite3
import time
import hashlib
from collections import defaultdict, deque
import statistics
import psutil
import threading
import multiprocessing
import resource
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logger = logging.getLogger(__name__)

class BenchmarkType(Enum):
    """Types of performance benchmarks"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_UTILIZATION = "cpu_utilization"
    ACCURACY = "accuracy"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    SCALABILITY = "scalability"
    STRESS_TEST = "stress_test"
    ENDURANCE = "endurance"

class LoadPattern(Enum):
    """Load patterns for benchmarking"""
    CONSTANT = "constant"
    RAMP_UP = "ramp_up"
    SPIKE = "spike"
    BURST = "burst"
    SINUSOIDAL = "sinusoidal"
    REALISTIC = "realistic"

@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarks"""
    name: str
    benchmark_types: List[BenchmarkType] = field(default_factory=lambda: [BenchmarkType.LATENCY, BenchmarkType.THROUGHPUT])
    
    # Test parameters
    duration_seconds: float = 60.0
    warmup_seconds: float = 10.0
    cooldown_seconds: float = 5.0
    
    # Load configuration
    load_pattern: LoadPattern = LoadPattern.CONSTANT
    max_concurrent_requests: int = 10
    requests_per_second: float = 5.0
    ramp_up_duration: float = 30.0
    
    # Test data
    test_prompts: List[str] = field(default_factory=list)
    prompt_categories: Dict[str, List[str]] = field(default_factory=dict)
    
    # Models to benchmark
    models: List[str] = field(default_factory=lambda: ["tinyllama"])
    model_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Resource monitoring
    monitor_system_resources: bool = True
    resource_sample_interval: float = 1.0
    
    # Output configuration
    output_directory: str = "benchmark_results"
    generate_plots: bool = True
    detailed_logging: bool = True
    
    # Quality thresholds
    max_acceptable_latency: float = 5.0
    min_acceptable_throughput: float = 1.0
    max_memory_usage_mb: float = 4096.0

@dataclass
class BenchmarkResult:
    """Single benchmark measurement result"""
    timestamp: float
    model_name: str
    prompt: str
    response: str
    
    # Performance metrics
    latency: float
    tokens_generated: int
    tokens_per_second: float
    
    # Quality metrics
    response_quality: float
    confidence_score: float
    
    # Resource usage
    memory_usage_mb: float
    cpu_usage_percent: float
    
    # Context information
    request_id: str
    thread_id: int
    process_id: int
    
    # Additional metrics
    additional_metrics: Dict[str, float] = field(default_factory=dict)

class SystemMonitor:
    """Monitors system resources during benchmarking"""
    
    def __init__(self, sample_interval: float = 1.0):
        self.sample_interval = sample_interval
        self.monitoring = False
        self.monitor_thread = None
        self.resource_data = []
        self.lock = threading.Lock()
    
    def start_monitoring(self):
        """Start system resource monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Started system resource monitoring")
    
    def stop_monitoring(self):
        """Stop system resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped system resource monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                network_io = psutil.net_io_counters()
                
                # Get process-specific metrics
                process = psutil.Process()
                process_memory = process.memory_info()
                process_cpu = process.cpu_percent()
                
                resource_sample = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_mb': memory.used / (1024 * 1024),
                    'memory_available_mb': memory.available / (1024 * 1024),
                    'process_memory_mb': process_memory.rss / (1024 * 1024),
                    'process_cpu_percent': process_cpu,
                    'disk_read_mb': disk_io.read_bytes / (1024 * 1024) if disk_io else 0,
                    'disk_write_mb': disk_io.write_bytes / (1024 * 1024) if disk_io else 0,
                    'network_sent_mb': network_io.bytes_sent / (1024 * 1024) if network_io else 0,
                    'network_recv_mb': network_io.bytes_recv / (1024 * 1024) if network_io else 0
                }
                
                with self.lock:
                    self.resource_data.append(resource_sample)
                    
                    # Limit data size
                    if len(self.resource_data) > 10000:
                        self.resource_data = self.resource_data[-5000:]
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.sample_interval)
    
    def get_resource_data(self) -> List[Dict[str, float]]:
        """Get collected resource data"""
        with self.lock:
            return self.resource_data.copy()
    
    def get_resource_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics of resource usage"""
        with self.lock:
            if not self.resource_data:
                return {}
            
            data = self.resource_data
            
            summary = {}
            metrics = ['cpu_percent', 'memory_percent', 'process_memory_mb', 'process_cpu_percent']
            
            for metric in metrics:
                values = [sample[metric] for sample in data if metric in sample]
                if values:
                    summary[metric] = {
                        'mean': np.mean(values),
                        'max': np.max(values),
                        'min': np.min(values),
                        'std': np.std(values),
                        'median': np.median(values)
                    }
            
            return summary

class LoadGenerator:
    """Generates load patterns for benchmarking"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.current_rps = 0.0
        self.start_time = None
    
    def get_request_rate(self, elapsed_time: float) -> float:
        """Get current request rate based on load pattern"""
        if self.config.load_pattern == LoadPattern.CONSTANT:
            return self.config.requests_per_second
        
        elif self.config.load_pattern == LoadPattern.RAMP_UP:
            if elapsed_time < self.config.ramp_up_duration:
                progress = elapsed_time / self.config.ramp_up_duration
                return self.config.requests_per_second * progress
            else:
                return self.config.requests_per_second
        
        elif self.config.load_pattern == LoadPattern.SPIKE:
            # Spike at 1/3 through the test
            spike_time = self.config.duration_seconds / 3
            spike_duration = 10.0  # 10 second spike
            
            if spike_time <= elapsed_time <= spike_time + spike_duration:
                return self.config.requests_per_second * 3  # 3x normal rate
            else:
                return self.config.requests_per_second
        
        elif self.config.load_pattern == LoadPattern.BURST:
            # Bursts every 30 seconds
            cycle_time = elapsed_time % 30
            if cycle_time < 5:  # 5 second burst every 30 seconds
                return self.config.requests_per_second * 2
            else:
                return self.config.requests_per_second * 0.5
        
        elif self.config.load_pattern == LoadPattern.SINUSOIDAL:
            # Sinusoidal load pattern
            period = 60.0  # 1 minute period
            amplitude = self.config.requests_per_second * 0.5
            baseline = self.config.requests_per_second
            
            return baseline + amplitude * np.sin(2 * np.pi * elapsed_time / period)
        
        elif self.config.load_pattern == LoadPattern.REALISTIC:
            # Realistic pattern with varying load
            hour_of_day = (elapsed_time / 3600) % 24
            
            # Peak hours: 9-11 AM and 2-4 PM
            if 9 <= hour_of_day <= 11 or 14 <= hour_of_day <= 16:
                return self.config.requests_per_second * 1.5
            # Low hours: 10 PM - 6 AM
            elif hour_of_day >= 22 or hour_of_day <= 6:
                return self.config.requests_per_second * 0.3
            else:
                return self.config.requests_per_second
        
        else:
            return self.config.requests_per_second
    
    def should_send_request(self, elapsed_time: float, last_request_time: float) -> bool:
        """Determine if a request should be sent based on load pattern"""
        target_rps = self.get_request_rate(elapsed_time)
        
        if target_rps <= 0:
            return False
        
        # Calculate time since last request
        time_since_last = elapsed_time - last_request_time
        
        # Expected interval between requests
        expected_interval = 1.0 / target_rps
        
        # Add some randomness to avoid perfect timing
        jitter = np.random.exponential(0.1)  # Exponential jitter
        
        return time_since_last >= (expected_interval - jitter)

class ModelBenchmarker:
    """Benchmarks individual models"""
    
    def __init__(self, model_name: str, ollama_host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.session = None
        self.request_count = 0
        self.error_count = 0
    
    async def initialize(self):
        """Initialize the benchmarker"""
        self.session = aiohttp.ClientSession()
        
        # Warm up the model
        await self._warmup()
        logger.info(f"Initialized benchmarker for {self.model_name}")
    
    async def _warmup(self):
        """Warm up the model"""
        try:
            await self.generate("Warmup prompt", max_tokens=10)
        except Exception as e:
            logger.warning(f"Warmup failed for {self.model_name}: {e}")
    
    async def generate(self, prompt: str, max_tokens: int = 512, 
                      temperature: float = 0.7, **kwargs) -> BenchmarkResult:
        """Generate response and collect performance metrics"""
        request_id = f"{self.model_name}_{self.request_count}_{int(time.time()*1000)}"
        self.request_count += 1
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        start_cpu = psutil.cpu_percent()
        
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "num_ctx": 2048,
                **kwargs
            }
        }
        
        try:
            async with self.session.post(
                f"{self.ollama_host}/api/generate",
                json=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                end_cpu = psutil.cpu_percent()
                
                latency = end_time - start_time
                memory_usage = end_memory - start_memory
                cpu_usage = end_cpu - start_cpu
                
                if response.status == 200:
                    result = await response.json()
                    response_text = result.get('response', '')
                    
                    # Calculate tokens and quality metrics
                    tokens_generated = len(response_text.split())
                    tokens_per_second = tokens_generated / latency if latency > 0 else 0
                    
                    # Simple quality assessment
                    response_quality = self._assess_response_quality(prompt, response_text)
                    confidence_score = min(1.0, max(0.0, 1.0 - latency / 10.0))  # Simple confidence based on speed
                    
                    return BenchmarkResult(
                        timestamp=start_time,
                        model_name=self.model_name,
                        prompt=prompt,
                        response=response_text,
                        latency=latency,
                        tokens_generated=tokens_generated,
                        tokens_per_second=tokens_per_second,
                        response_quality=response_quality,
                        confidence_score=confidence_score,
                        memory_usage_mb=memory_usage,
                        cpu_usage_percent=cpu_usage,
                        request_id=request_id,
                        thread_id=threading.get_ident(),
                        process_id=os.getpid(),
                        additional_metrics={
                            'total_duration': result.get('total_duration', 0),
                            'load_duration': result.get('load_duration', 0),
                            'prompt_eval_count': result.get('prompt_eval_count', 0),
                            'eval_count': result.get('eval_count', 0)
                        }
                    )
                else:
                    self.error_count += 1
                    logger.error(f"Generation failed for {self.model_name}: {response.status}")
                    
                    return BenchmarkResult(
                        timestamp=start_time,
                        model_name=self.model_name,
                        prompt=prompt,
                        response="",
                        latency=latency,
                        tokens_generated=0,
                        tokens_per_second=0,
                        response_quality=0.0,
                        confidence_score=0.0,
                        memory_usage_mb=memory_usage,
                        cpu_usage_percent=cpu_usage,
                        request_id=request_id,
                        thread_id=threading.get_ident(),
                        process_id=os.getpid(),
                        additional_metrics={'error': f'HTTP {response.status}'}
                    )
                    
        except Exception as e:
            self.error_count += 1
            end_time = time.time()
            latency = end_time - start_time
            
            logger.error(f"Error benchmarking {self.model_name}: {e}")
            
            return BenchmarkResult(
                timestamp=start_time,
                model_name=self.model_name,
                prompt=prompt,
                response="",
                latency=latency,
                tokens_generated=0,
                tokens_per_second=0,
                response_quality=0.0,
                confidence_score=0.0,
                memory_usage_mb=0,
                cpu_usage_percent=0,
                request_id=request_id,
                thread_id=threading.get_ident(),
                process_id=os.getpid(),
                additional_metrics={'error': str(e)}
            )
    
    def _assess_response_quality(self, prompt: str, response: str) -> float:
        """Simple response quality assessment"""
        if not response or len(response.strip()) == 0:
            return 0.0
        
        quality_score = 0.0
        
        # Length factor (reasonable response length)
        response_length = len(response.split())
        if 10 <= response_length <= 200:
            quality_score += 0.3
        elif response_length > 5:
            quality_score += 0.1
        
        # Content indicators
        if '.' in response:  # Contains sentences
            quality_score += 0.2
        
        if response.count(' ') > 5:  # Multiple words
            quality_score += 0.2
        
        # Avoid obvious errors
        error_indicators = ['error', 'sorry', "i don't", "unclear", "cannot", "unable"]
        if not any(indicator in response.lower() for indicator in error_indicators):
            quality_score += 0.3
        
        return min(1.0, quality_score)
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()

class BenchmarkSuite:
    """Main benchmarking suite orchestrator"""
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig("Default Benchmark")
        self.benchmarkers = {}
        self.system_monitor = SystemMonitor(self.config.resource_sample_interval)
        self.load_generator = LoadGenerator(self.config)
        self.results = []
        self.db_path = f"{self.config.output_directory}/benchmark_results.db"
        
        # Create output directory
        Path(self.config.output_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for storing results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id TEXT PRIMARY KEY,
                name TEXT,
                config TEXT,
                start_time REAL,
                end_time REAL,
                status TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id TEXT PRIMARY KEY,
                run_id TEXT,
                timestamp REAL,
                model_name TEXT,
                prompt TEXT,
                response TEXT,
                latency REAL,
                tokens_generated INTEGER,
                tokens_per_second REAL,
                response_quality REAL,
                confidence_score REAL,
                memory_usage_mb REAL,
                cpu_usage_percent REAL,
                request_id TEXT,
                thread_id INTEGER,
                process_id INTEGER,
                additional_metrics TEXT,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id TEXT PRIMARY KEY,
                run_id TEXT,
                timestamp REAL,
                cpu_percent REAL,
                memory_percent REAL,
                memory_used_mb REAL,
                process_memory_mb REAL,
                process_cpu_percent REAL,
                FOREIGN KEY (run_id) REFERENCES benchmark_runs (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def initialize(self):
        """Initialize the benchmark suite"""
        logger.info("Initializing benchmark suite...")
        
        # Initialize benchmarkers for each model
        for model_name in self.config.models:
            benchmarker = ModelBenchmarker(model_name)
            await benchmarker.initialize()
            self.benchmarkers[model_name] = benchmarker
        
        # Prepare test prompts if not provided
        if not self.config.test_prompts:
            self.config.test_prompts = self._generate_default_prompts()
        
        logger.info(f"Benchmark suite initialized with {len(self.benchmarkers)} models")
    
    def _generate_default_prompts(self) -> List[str]:
        """Generate default test prompts"""
        return [
            "What is machine learning?",
            "Explain quantum computing in simple terms.",
            "Write a Python function to sort a list.",
            "Describe the benefits of renewable energy.",
            "How does a neural network work?",
            "What are the principles of good software design?",
            "Explain the concept of recursion.",
            "What is the greenhouse effect?",
            "How do you implement a binary search?",
            "Describe the process of photosynthesis.",
            "What is artificial intelligence?",
            "Explain the difference between HTTP and HTTPS.",
            "How do databases work?",
            "What is cloud computing?",
            "Describe the water cycle.",
            "How do you debug a program?",
            "What is cryptocurrency?",
            "Explain the concept of entropy.",
            "How does the internet work?",
            "What is the theory of relativity?"
        ]
    
    async def run_benchmark(self, run_id: str = None) -> Dict[str, Any]:
        """Run the complete benchmark suite"""
        run_id = run_id or f"benchmark_{int(time.time())}"
        start_time = time.time()
        
        logger.info(f"Starting benchmark run: {run_id}")
        
        # Record benchmark run in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO benchmark_runs (id, name, config, start_time, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (run_id, self.config.name, json.dumps(self.config.__dict__, default=str), start_time, 'running'))
        conn.commit()
        conn.close()
        
        try:
            # Start system monitoring
            if self.config.monitor_system_resources:
                self.system_monitor.start_monitoring()
            
            # Warmup phase
            if self.config.warmup_seconds > 0:
                logger.info(f"Warmup phase: {self.config.warmup_seconds} seconds")
                await self._run_warmup()
            
            # Main benchmark phase
            logger.info(f"Main benchmark phase: {self.config.duration_seconds} seconds")
            benchmark_results = await self._run_main_benchmark(run_id)
            
            # Cooldown phase
            if self.config.cooldown_seconds > 0:
                logger.info(f"Cooldown phase: {self.config.cooldown_seconds} seconds")
                await asyncio.sleep(self.config.cooldown_seconds)
            
            # Stop monitoring
            if self.config.monitor_system_resources:
                self.system_monitor.stop_monitoring()
            
            # Store system metrics
            await self._store_system_metrics(run_id)
            
            # Analyze results
            analysis = self._analyze_results(benchmark_results)
            
            # Generate reports
            if self.config.generate_plots:
                await self._generate_plots(run_id, benchmark_results)
            
            # Update run status
            end_time = time.time()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE benchmark_runs 
                SET end_time = ?, status = ?
                WHERE id = ?
            ''', (end_time, 'completed', run_id))
            conn.commit()
            conn.close()
            
            logger.info(f"Benchmark run completed: {run_id}")
            
            return {
                'run_id': run_id,
                'duration': end_time - start_time,
                'results': benchmark_results,
                'analysis': analysis,
                'system_metrics': self.system_monitor.get_resource_summary()
            }
            
        except Exception as e:
            logger.error(f"Benchmark run failed: {e}")
            
            # Update run status
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE benchmark_runs 
                SET status = ?
                WHERE id = ?
            ''', ('failed', run_id))
            conn.commit()
            conn.close()
            
            raise
    
    async def _run_warmup(self):
        """Run warmup phase"""
        warmup_prompts = self.config.test_prompts[:3]  # Use first 3 prompts for warmup
        
        tasks = []
        for model_name, benchmarker in self.benchmarkers.items():
            for prompt in warmup_prompts:
                task = benchmarker.generate(prompt, max_tokens=50, temperature=0.5)
                tasks.append(task)
        
        # Execute warmup requests
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Small delay after warmup
        await asyncio.sleep(2)
    
    async def _run_main_benchmark(self, run_id: str) -> List[BenchmarkResult]:
        """Run main benchmark phase"""
        results = []
        start_time = time.time()
        last_request_time = 0.0
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # Track active tasks
        active_tasks = set()
        
        while time.time() - start_time < self.config.duration_seconds:
            elapsed_time = time.time() - start_time
            
            # Check if we should send a new request
            if self.load_generator.should_send_request(elapsed_time, last_request_time):
                # Select random model and prompt
                model_name = np.random.choice(self.config.models)
                prompt = np.random.choice(self.config.test_prompts)
                
                # Create benchmark task
                task = asyncio.create_task(
                    self._run_single_benchmark(semaphore, model_name, prompt, run_id)
                )
                active_tasks.add(task)
                last_request_time = elapsed_time
            
            # Collect completed tasks
            done_tasks = {task for task in active_tasks if task.done()}
            for task in done_tasks:
                try:
                    result = await task
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Benchmark task failed: {e}")
                
                active_tasks.remove(task)
            
            # Small delay to prevent busy waiting
            await asyncio.sleep(0.1)
        
        # Wait for remaining tasks to complete
        if active_tasks:
            remaining_results = await asyncio.gather(*active_tasks, return_exceptions=True)
            for result in remaining_results:
                if isinstance(result, BenchmarkResult):
                    results.append(result)
        
        return results
    
    async def _run_single_benchmark(self, semaphore: asyncio.Semaphore, 
                                  model_name: str, prompt: str, run_id: str) -> Optional[BenchmarkResult]:
        """Run a single benchmark request"""
        async with semaphore:
            try:
                benchmarker = self.benchmarkers[model_name]
                
                # Get model-specific configuration
                model_config = self.config.model_configs.get(model_name, {})
                
                result = await benchmarker.generate(
                    prompt,
                    max_tokens=model_config.get('max_tokens', 256),
                    temperature=model_config.get('temperature', 0.7),
                    **model_config.get('additional_params', {})
                )
                
                # Store result in database
                await self._store_result(run_id, result)
                
                return result
                
            except Exception as e:
                logger.error(f"Single benchmark failed: {e}")
                return None
    
    async def _store_result(self, run_id: str, result: BenchmarkResult):
        """Store benchmark result in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        result_id = f"{result.request_id}_{int(result.timestamp * 1000)}"
        
        cursor.execute('''
            INSERT INTO benchmark_results (
                id, run_id, timestamp, model_name, prompt, response,
                latency, tokens_generated, tokens_per_second,
                response_quality, confidence_score, memory_usage_mb,
                cpu_usage_percent, request_id, thread_id, process_id,
                additional_metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result_id, run_id, result.timestamp, result.model_name,
            result.prompt, result.response, result.latency,
            result.tokens_generated, result.tokens_per_second,
            result.response_quality, result.confidence_score,
            result.memory_usage_mb, result.cpu_usage_percent,
            result.request_id, result.thread_id, result.process_id,
            json.dumps(result.additional_metrics)
        ))
        
        conn.commit()
        conn.close()
    
    async def _store_system_metrics(self, run_id: str):
        """Store system metrics in database"""
        resource_data = self.system_monitor.get_resource_data()
        
        if not resource_data:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for sample in resource_data:
            metric_id = f"{run_id}_{int(sample['timestamp'] * 1000)}"
            
            cursor.execute('''
                INSERT OR IGNORE INTO system_metrics (
                    id, run_id, timestamp, cpu_percent, memory_percent,
                    memory_used_mb, process_memory_mb, process_cpu_percent
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metric_id, run_id, sample['timestamp'],
                sample.get('cpu_percent', 0),
                sample.get('memory_percent', 0),
                sample.get('memory_used_mb', 0),
                sample.get('process_memory_mb', 0),
                sample.get('process_cpu_percent', 0)
            ))
        
        conn.commit()
        conn.close()
    
    def _analyze_results(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze benchmark results"""
        if not results:
            return {"error": "No results to analyze"}
        
        # Group results by model
        model_results = defaultdict(list)
        for result in results:
            model_results[result.model_name].append(result)
        
        analysis = {
            "overall": self._analyze_overall_performance(results),
            "by_model": {},
            "comparisons": {},
            "recommendations": []
        }
        
        # Analyze each model
        for model_name, model_results_list in model_results.items():
            analysis["by_model"][model_name] = self._analyze_model_performance(model_results_list)
        
        # Model comparisons
        if len(model_results) > 1:
            analysis["comparisons"] = self._compare_models(model_results)
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _analyze_overall_performance(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze overall performance across all models"""
        latencies = [r.latency for r in results]
        throughputs = [r.tokens_per_second for r in results]
        qualities = [r.response_quality for r in results]
        
        return {
            "total_requests": len(results),
            "latency": {
                "mean": np.mean(latencies),
                "median": np.median(latencies),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
                "min": np.min(latencies),
                "max": np.max(latencies),
                "std": np.std(latencies)
            },
            "throughput": {
                "mean": np.mean(throughputs),
                "median": np.median(throughputs),
                "max": np.max(throughputs),
                "total_tokens": sum(r.tokens_generated for r in results)
            },
            "quality": {
                "mean": np.mean(qualities),
                "median": np.median(qualities),
                "min": np.min(qualities),
                "max": np.max(qualities)
            },
            "error_rate": sum(1 for r in results if r.response_quality == 0) / len(results)
        }
    
    def _analyze_model_performance(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze performance for a specific model"""
        if not results:
            return {"error": "No results for model"}
        
        latencies = [r.latency for r in results]
        throughputs = [r.tokens_per_second for r in results]
        qualities = [r.response_quality for r in results]
        memory_usage = [r.memory_usage_mb for r in results]
        
        return {
            "request_count": len(results),
            "latency_stats": {
                "mean": np.mean(latencies),
                "median": np.median(latencies),
                "p95": np.percentile(latencies, 95),
                "std": np.std(latencies)
            },
            "throughput_stats": {
                "mean": np.mean(throughputs),
                "median": np.median(throughputs),
                "max": np.max(throughputs)
            },
            "quality_stats": {
                "mean": np.mean(qualities),
                "std": np.std(qualities)
            },
            "resource_usage": {
                "avg_memory_mb": np.mean(memory_usage),
                "max_memory_mb": np.max(memory_usage)
            },
            "error_rate": sum(1 for r in results if r.response_quality == 0) / len(results)
        }
    
    def _compare_models(self, model_results: Dict[str, List[BenchmarkResult]]) -> Dict[str, Any]:
        """Compare performance between models"""
        comparisons = {}
        
        models = list(model_results.keys())
        
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model_a = models[i]
                model_b = models[j]
                
                results_a = model_results[model_a]
                results_b = model_results[model_b]
                
                latencies_a = [r.latency for r in results_a]
                latencies_b = [r.latency for r in results_b]
                
                throughputs_a = [r.tokens_per_second for r in results_a]
                throughputs_b = [r.tokens_per_second for r in results_b]
                
                comparison_key = f"{model_a}_vs_{model_b}"
                
                comparisons[comparison_key] = {
                    "latency_comparison": {
                        "model_a_mean": np.mean(latencies_a),
                        "model_b_mean": np.mean(latencies_b),
                        "improvement": (np.mean(latencies_a) - np.mean(latencies_b)) / np.mean(latencies_a)
                    },
                    "throughput_comparison": {
                        "model_a_mean": np.mean(throughputs_a),
                        "model_b_mean": np.mean(throughputs_b),
                        "improvement": (np.mean(throughputs_b) - np.mean(throughputs_a)) / np.mean(throughputs_a)
                    }
                }
        
        return comparisons
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        overall = analysis.get("overall", {})
        
        # Latency recommendations
        if overall.get("latency", {}).get("mean", 0) > self.config.max_acceptable_latency:
            recommendations.append(f"Average latency ({overall['latency']['mean']:.2f}s) exceeds threshold")
        
        # Throughput recommendations
        if overall.get("throughput", {}).get("mean", 0) < self.config.min_acceptable_throughput:
            recommendations.append(f"Average throughput is below acceptable threshold")
        
        # Error rate recommendations
        error_rate = overall.get("error_rate", 0)
        if error_rate > 0.05:  # 5% error rate threshold
            recommendations.append(f"High error rate detected: {error_rate:.1%}")
        
        # Model-specific recommendations
        by_model = analysis.get("by_model", {})
        for model_name, model_stats in by_model.items():
            model_error_rate = model_stats.get("error_rate", 0)
            if model_error_rate > 0.1:  # 10% error rate for individual models
                recommendations.append(f"Model {model_name} has high error rate: {model_error_rate:.1%}")
        
        return recommendations
    
    async def _generate_plots(self, run_id: str, results: List[BenchmarkResult]):
        """Generate performance plots"""
        if not results:
            return
        
        try:
            # Set up plotting style
            plt.style.use('seaborn-v0_8')
            
            # Create plots directory
            plot_dir = Path(self.config.output_directory) / "plots" / run_id
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # Latency distribution plot
            self._plot_latency_distribution(results, plot_dir)
            
            # Throughput over time plot
            self._plot_throughput_over_time(results, plot_dir)
            
            # Model comparison plot
            self._plot_model_comparison(results, plot_dir)
            
            # System resource usage plot
            self._plot_system_resources(run_id, plot_dir)
            
            logger.info(f"Generated plots in {plot_dir}")
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
    
    def _plot_latency_distribution(self, results: List[BenchmarkResult], plot_dir: Path):
        """Plot latency distribution"""
        plt.figure(figsize=(12, 6))
        
        # Group by model
        model_latencies = defaultdict(list)
        for result in results:
            model_latencies[result.model_name].append(result.latency)
        
        # Plot histograms
        for model_name, latencies in model_latencies.items():
            plt.hist(latencies, alpha=0.7, label=model_name, bins=50)
        
        plt.xlabel('Latency (seconds)')
        plt.ylabel('Frequency')
        plt.title('Latency Distribution by Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'latency_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_throughput_over_time(self, results: List[BenchmarkResult], plot_dir: Path):
        """Plot throughput over time"""
        plt.figure(figsize=(14, 8))
        
        # Sort results by timestamp
        results_sorted = sorted(results, key=lambda x: x.timestamp)
        
        # Calculate moving average throughput
        window_size = 10
        
        model_data = defaultdict(lambda: {'times': [], 'throughputs': []})
        
        for result in results_sorted:
            model_data[result.model_name]['times'].append(result.timestamp)
            model_data[result.model_name]['throughputs'].append(result.tokens_per_second)
        
        # Plot for each model
        for model_name, data in model_data.items():
            times = np.array(data['times'])
            throughputs = np.array(data['throughputs'])
            
            # Normalize time to start from 0
            times = times - times[0]
            
            # Calculate moving average
            if len(throughputs) >= window_size:
                moving_avg = np.convolve(throughputs, np.ones(window_size)/window_size, mode='valid')
                moving_times = times[window_size-1:]
                plt.plot(moving_times, moving_avg, label=f'{model_name} (moving avg)', linewidth=2)
            
            # Plot raw data with transparency
            plt.scatter(times, throughputs, alpha=0.3, s=10)
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Throughput (tokens/second)')
        plt.title('Throughput Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'throughput_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_comparison(self, results: List[BenchmarkResult], plot_dir: Path):
        """Plot model comparison"""
        model_stats = defaultdict(lambda: {'latencies': [], 'throughputs': [], 'qualities': []})
        
        for result in results:
            model_stats[result.model_name]['latencies'].append(result.latency)
            model_stats[result.model_name]['throughputs'].append(result.tokens_per_second)
            model_stats[result.model_name]['qualities'].append(result.response_quality)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(model_stats.keys())
        
        # Latency comparison
        latency_means = [np.mean(model_stats[model]['latencies']) for model in models]
        latency_stds = [np.std(model_stats[model]['latencies']) for model in models]
        
        axes[0, 0].bar(models, latency_means, yerr=latency_stds, capsize=5)
        axes[0, 0].set_title('Average Latency by Model')
        axes[0, 0].set_ylabel('Latency (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Throughput comparison
        throughput_means = [np.mean(model_stats[model]['throughputs']) for model in models]
        throughput_stds = [np.std(model_stats[model]['throughputs']) for model in models]
        
        axes[0, 1].bar(models, throughput_means, yerr=throughput_stds, capsize=5)
        axes[0, 1].set_title('Average Throughput by Model')
        axes[0, 1].set_ylabel('Throughput (tokens/second)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Quality comparison
        quality_means = [np.mean(model_stats[model]['qualities']) for model in models]
        quality_stds = [np.std(model_stats[model]['qualities']) for model in models]
        
        axes[1, 0].bar(models, quality_means, yerr=quality_stds, capsize=5)
        axes[1, 0].set_title('Average Response Quality by Model')
        axes[1, 0].set_ylabel('Quality Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Latency vs Throughput scatter
        for model in models:
            latencies = model_stats[model]['latencies']
            throughputs = model_stats[model]['throughputs']
            axes[1, 1].scatter(latencies, throughputs, label=model, alpha=0.6)
        
        axes[1, 1].set_xlabel('Latency (seconds)')
        axes[1, 1].set_ylabel('Throughput (tokens/second)')
        axes[1, 1].set_title('Latency vs Throughput')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_system_resources(self, run_id: str, plot_dir: Path):
        """Plot system resource usage"""
        resource_data = self.system_monitor.get_resource_data()
        
        if not resource_data:
            return
        
        timestamps = [sample['timestamp'] for sample in resource_data]
        cpu_usage = [sample.get('cpu_percent', 0) for sample in resource_data]
        memory_usage = [sample.get('memory_percent', 0) for sample in resource_data]
        process_memory = [sample.get('process_memory_mb', 0) for sample in resource_data]
        
        # Normalize timestamps
        start_time = min(timestamps)
        timestamps = [(t - start_time) for t in timestamps]
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # CPU Usage
        axes[0].plot(timestamps, cpu_usage, label='System CPU %', linewidth=2)
        axes[0].set_ylabel('CPU Usage (%)')
        axes[0].set_title('System Resource Usage Over Time')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Memory Usage
        axes[1].plot(timestamps, memory_usage, label='System Memory %', linewidth=2, color='orange')
        axes[1].plot(timestamps, process_memory, label='Process Memory (MB)', linewidth=2, color='red')
        axes[1].set_xlabel('Time (seconds)')
        axes[1].set_ylabel('Memory Usage')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'system_resources.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, run_id: str) -> str:
        """Generate comprehensive benchmark report"""
        try:
            # Load results from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get run info
            cursor.execute('SELECT * FROM benchmark_runs WHERE id = ?', (run_id,))
            run_info = cursor.fetchone()
            
            if not run_info:
                return f"Benchmark run {run_id} not found"
            
            # Get results
            cursor.execute('SELECT * FROM benchmark_results WHERE run_id = ?', (run_id,))
            result_rows = cursor.fetchall()
            
            conn.close()
            
            if not result_rows:
                return f"No results found for benchmark run {run_id}"
            
            # Convert to BenchmarkResult objects
            results = []
            for row in result_rows:
                # Map database columns to BenchmarkResult fields
                additional_metrics = json.loads(row[16]) if row[16] else {}
                
                result = BenchmarkResult(
                    timestamp=row[2],
                    model_name=row[3],
                    prompt=row[4],
                    response=row[5],
                    latency=row[6],
                    tokens_generated=row[7],
                    tokens_per_second=row[8],
                    response_quality=row[9],
                    confidence_score=row[10],
                    memory_usage_mb=row[11],
                    cpu_usage_percent=row[12],
                    request_id=row[13],
                    thread_id=row[14],
                    process_id=row[15],
                    additional_metrics=additional_metrics
                )
                results.append(result)
            
            # Analyze results
            analysis = self._analyze_results(results)
            
            # Generate report
            report = []
            report.append(f"# Performance Benchmark Report")
            report.append(f"Run ID: {run_id}")
            report.append(f"Benchmark: {run_info[1]}")
            report.append(f"Started: {time.ctime(run_info[3])}")
            report.append("=" * 60)
            
            # Overall performance
            overall = analysis.get('overall', {})
            report.append(f"\n## Overall Performance")
            report.append(f"Total Requests: {overall.get('total_requests', 0)}")
            report.append(f"Error Rate: {overall.get('error_rate', 0):.1%}")
            
            if 'latency' in overall:
                latency = overall['latency']
                report.append(f"\n### Latency Statistics")
                report.append(f"Mean: {latency['mean']:.3f}s")
                report.append(f"Median: {latency['median']:.3f}s")
                report.append(f"95th Percentile: {latency['p95']:.3f}s")
                report.append(f"99th Percentile: {latency['p99']:.3f}s")
            
            if 'throughput' in overall:
                throughput = overall['throughput']
                report.append(f"\n### Throughput Statistics")
                report.append(f"Mean: {throughput['mean']:.2f} tokens/second")
                report.append(f"Total Tokens: {throughput.get('total_tokens', 0)}")
            
            # Model-specific performance
            by_model = analysis.get('by_model', {})
            if by_model:
                report.append(f"\n## Performance by Model")
                
                for model_name, model_stats in by_model.items():
                    report.append(f"\n### {model_name}")
                    report.append(f"Requests: {model_stats.get('request_count', 0)}")
                    report.append(f"Error Rate: {model_stats.get('error_rate', 0):.1%}")
                    
                    latency_stats = model_stats.get('latency_stats', {})
                    if latency_stats:
                        report.append(f"Average Latency: {latency_stats.get('mean', 0):.3f}s")
                        report.append(f"95th Percentile Latency: {latency_stats.get('p95', 0):.3f}s")
                    
                    throughput_stats = model_stats.get('throughput_stats', {})
                    if throughput_stats:
                        report.append(f"Average Throughput: {throughput_stats.get('mean', 0):.2f} tokens/s")
            
            # Recommendations
            recommendations = analysis.get('recommendations', [])
            if recommendations:
                report.append(f"\n## Recommendations")
                for i, rec in enumerate(recommendations, 1):
                    report.append(f"{i}. {rec}")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"Error generating report: {str(e)}"
    
    async def cleanup(self):
        """Cleanup benchmark suite resources"""
        for benchmarker in self.benchmarkers.values():
            await benchmarker.cleanup()
        
        self.system_monitor.stop_monitoring()

# Factory functions
def create_latency_benchmark(models: List[str] = None) -> BenchmarkConfig:
    """Create a latency-focused benchmark configuration"""
    return BenchmarkConfig(
        name="Latency Benchmark",
        benchmark_types=[BenchmarkType.LATENCY],
        models=models or ["tinyllama"],
        duration_seconds=300,  # 5 minutes
        requests_per_second=2.0,
        max_concurrent_requests=5
    )

def create_throughput_benchmark(models: List[str] = None) -> BenchmarkConfig:
    """Create a throughput-focused benchmark configuration"""
    return BenchmarkConfig(
        name="Throughput Benchmark",
        benchmark_types=[BenchmarkType.THROUGHPUT],
        models=models or ["tinyllama"],
        duration_seconds=600,  # 10 minutes
        requests_per_second=10.0,
        max_concurrent_requests=20,
        load_pattern=LoadPattern.RAMP_UP
    )

def create_stress_test_benchmark(models: List[str] = None) -> BenchmarkConfig:
    """Create a stress test benchmark configuration"""
    return BenchmarkConfig(
        name="Stress Test Benchmark",
        benchmark_types=[BenchmarkType.STRESS_TEST, BenchmarkType.MEMORY_USAGE],
        models=models or ["tinyllama"],
        duration_seconds=1800,  # 30 minutes
        requests_per_second=20.0,
        max_concurrent_requests=50,
        load_pattern=LoadPattern.SPIKE
    )

# Example usage
async def example_benchmarking():
    """Example benchmarking usage"""
    # Create benchmark configuration
    config = create_latency_benchmark(["tinyllama", "qwen2.5-coder:7b"])
    config.duration_seconds = 60  # 1 minute for demo
    
    # Create and initialize benchmark suite
    benchmark_suite = BenchmarkSuite(config)
    await benchmark_suite.initialize()
    
    # Run benchmark
    results = await benchmark_suite.run_benchmark("demo_benchmark")
    
    # Generate report
    report = benchmark_suite.generate_report("demo_benchmark")
    
    print("Benchmark Results:")
    print(json.dumps(results['analysis'], indent=2, default=str))
    print("\nBenchmark Report:")
    print(report)
    
    # Cleanup
    await benchmark_suite.cleanup()
    
    return results

if __name__ == "__main__":
    # Run example
    async def main():
        results = await example_benchmarking()
        return results
    
    # asyncio.run(main())