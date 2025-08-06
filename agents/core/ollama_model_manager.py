#!/usr/bin/env python3
"""
Purpose: Model versioning and comprehensive benchmarking system for Ollama
Usage: Manages model versions and runs detailed performance benchmarks
Requirements: ollama, httpx, psutil, sqlite3, prometheus_client
"""

import os
import sys
import json
import time
import sqlite3
import asyncio
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import httpx
import psutil
from concurrent.futures import ThreadPoolExecutor
import statistics
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, push_to_gateway

# Configure logging
logger = logging.getLogger('ollama-model-manager')

@dataclass
class ModelVersion:
    """Model version information"""
    name: str
    version: str
    digest: str
    size_bytes: int
    parameter_count: str
    quantization: str
    family: str
    created_at: datetime
    performance_score: Optional[float] = None
    benchmark_results: Optional[Dict[str, Any]] = None

@dataclass
class BenchmarkResult:
    """Comprehensive benchmark result"""
    model: str
    version: str
    timestamp: datetime
    
    # Latency metrics (in seconds)
    latency_mean: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_max: float
    
    # Throughput metrics
    throughput_rps: float
    tokens_per_second: float
    
    # Quality metrics
    response_quality_score: float
    coherence_score: float
    relevance_score: float
    
    # Resource usage
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: float
    
    # Error metrics
    error_rate: float
    timeout_rate: float
    
    # Context handling
    max_context_handled: int
    context_efficiency: float
    
    # Overall performance score
    performance_score: float

class OllamaModelManager:
    """Comprehensive model management and benchmarking system"""
    
    def __init__(self, 
                 ollama_url: str = "http://localhost:11434",
                 db_path: str = "/opt/sutazaiapp/data/model_management.db"):
        
        self.ollama_url = ollama_url
        self.db_path = db_path
        self.client = httpx.AsyncClient(timeout=300.0)
        
        # Initialize database
        self._init_database()
        
        # Model tracking
        self.models: Dict[str, ModelVersion] = {}
        self.benchmark_history: List[BenchmarkResult] = []
        
        # Benchmark configurations
        self.benchmark_configs = {
            'quick': {
                'num_requests': 20,
                'concurrent_requests': 5,
                'test_prompts': 5
            },
            'standard': {
                'num_requests': 100,
                'concurrent_requests': 10,
                'test_prompts': 20
            },
            'comprehensive': {
                'num_requests': 500,
                'concurrent_requests': 20,
                'test_prompts': 50
            }
        }
        
        # Prometheus metrics
        self.registry = CollectorRegistry()
        self._setup_metrics()
    
    def _init_database(self):
        """Initialize SQLite database for model management"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    digest TEXT UNIQUE NOT NULL,
                    size_bytes INTEGER,
                    parameter_count TEXT,
                    quantization TEXT,
                    family TEXT,
                    created_at TIMESTAMP,
                    performance_score REAL,
                    benchmark_results TEXT,
                    UNIQUE(name, version)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT NOT NULL,
                    version TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    latency_mean REAL,
                    latency_p50 REAL,
                    latency_p95 REAL,
                    latency_p99 REAL,
                    latency_max REAL,
                    throughput_rps REAL,
                    tokens_per_second REAL,
                    response_quality_score REAL,
                    coherence_score REAL,
                    relevance_score REAL,
                    memory_usage_mb REAL,
                    cpu_usage_percent REAL,
                    gpu_usage_percent REAL,
                    error_rate REAL,
                    timeout_rate REAL,
                    max_context_handled INTEGER,
                    context_efficiency REAL,
                    performance_score REAL
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_benchmark_model_timestamp 
                ON benchmark_results(model, timestamp)
            ''')
    
    def _setup_metrics(self):
        """Setup Prometheus metrics"""
        self.model_performance_score = Gauge(
            'ollama_model_performance_score', 
            'Model performance score',
            ['model', 'version'],
            registry=self.registry
        )
        
        self.model_latency = Histogram(
            'ollama_model_latency_seconds',
            'Model response latency',
            ['model', 'version'],
            registry=self.registry
        )
        
        self.model_throughput = Gauge(
            'ollama_model_throughput_rps',
            'Model throughput in requests per second',
            ['model', 'version'],
            registry=self.registry
        )
        
        self.model_memory_usage = Gauge(
            'ollama_model_memory_mb',
            'Model memory usage in MB',
            ['model', 'version'],
            registry=self.registry
        )
    
    async def discover_models(self) -> Dict[str, ModelVersion]:
        """Discover all available models and their versions"""
        logger.info("Discovering available models")
        
        try:
            response = await self.client.get(f"{self.ollama_url}/api/tags")
            if response.status_code != 200:
                logger.error(f"Failed to get models: {response.status_code}")
                return {}
            
            models_data = response.json().get('models', [])
            discovered_models = {}
            
            for model_info in models_data:
                name = model_info.get('name', '')
                digest = model_info.get('digest', '')
                size_bytes = model_info.get('size', 0)
                modified_at = model_info.get('modified_at', '')
                
                details = model_info.get('details', {})
                parameter_count = details.get('parameter_size', 'Unknown')
                quantization = details.get('quantization_level', 'Unknown')
                family = details.get('family', 'Unknown')
                
                # Parse version from name
                if ':' in name:
                    model_name, version = name.split(':', 1)
                else:
                    model_name, version = name, 'latest'
                
                # Parse creation time
                try:
                    created_at = datetime.fromisoformat(modified_at.replace('Z', '+00:00'))
                except:
                    created_at = datetime.now()
                
                model_version = ModelVersion(
                    name=model_name,
                    version=version,
                    digest=digest,
                    size_bytes=size_bytes,
                    parameter_count=parameter_count,
                    quantization=quantization,
                    family=family,
                    created_at=created_at
                )
                
                discovered_models[name] = model_version
                self.models[name] = model_version
                
                # Save to database
                self._save_model_version(model_version)
            
            logger.info(f"Discovered {len(discovered_models)} models")
            return discovered_models
        
        except Exception as e:
            logger.error(f"Error discovering models: {e}")
            return {}
    
    def _save_model_version(self, model_version: ModelVersion):
        """Save model version to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO model_versions 
                (name, version, digest, size_bytes, parameter_count, quantization, 
                 family, created_at, performance_score, benchmark_results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_version.name,
                model_version.version,
                model_version.digest,
                model_version.size_bytes,
                model_version.parameter_count,
                model_version.quantization,
                model_version.family,
                model_version.created_at,
                model_version.performance_score,
                json.dumps(model_version.benchmark_results) if model_version.benchmark_results else None
            ))
    
    async def run_comprehensive_benchmark(self, 
                                        model: str, 
                                        benchmark_type: str = 'standard') -> BenchmarkResult:
        """Run comprehensive benchmark on a model"""
        logger.info(f"Running {benchmark_type} benchmark for {model}")
        
        config = self.benchmark_configs[benchmark_type]
        start_time = time.time()
        
        # Prepare test data
        test_prompts = self._generate_benchmark_prompts(config['test_prompts'])
        
        # Initialize metrics tracking
        latencies = []
        response_lengths = []
        errors = 0
        timeouts = 0
        quality_scores = []
        coherence_scores = []
        relevance_scores = []
        
        # Resource monitoring
        initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        cpu_samples = []
        
        # Run benchmark
        semaphore = asyncio.Semaphore(config['concurrent_requests'])
        
        async def benchmark_request(prompt_info: Dict[str, Any]):
            async with semaphore:
                request_start = time.time()
                
                try:
                    response = await self.client.post(
                        f"{self.ollama_url}/api/generate",
                        json={
                            "model": model,
                            "prompt": prompt_info['prompt'],
                            "stream": False,
                            "options": {
                                "temperature": 0.7,
                                "max_tokens": prompt_info.get('max_tokens', 256)
                            }
                        }
                    )
                    
                    request_time = time.time() - request_start
                    
                    if response.status_code == 200:
                        result = response.json()
                        response_text = result.get('response', '')
                        
                        latencies.append(request_time)
                        response_lengths.append(len(response_text.split()))
                        
                        # Evaluate response quality
                        quality_score = self._evaluate_response_quality(
                            prompt_info['prompt'], response_text, prompt_info.get('expected_type', 'general')
                        )
                        quality_scores.append(quality_score)
                        
                        coherence_score = self._evaluate_coherence(response_text)
                        coherence_scores.append(coherence_score)
                        
                        relevance_score = self._evaluate_relevance(prompt_info['prompt'], response_text)
                        relevance_scores.append(relevance_score)
                        
                    elif response.status_code == 408:  # Timeout
                        timeouts += 1
                    else:
                        errors += 1
                
                except asyncio.TimeoutError:
                    timeouts += 1
                except Exception as e:
                    logger.error(f"Benchmark request error: {e}")
                    errors += 1
                
                # Sample CPU usage
                cpu_samples.append(psutil.cpu_percent())
        
        # Execute all benchmark requests
        tasks = []
        for _ in range(config['num_requests']):
            prompt_info = test_prompts[_ % len(test_prompts)]
            task = asyncio.create_task(benchmark_request(prompt_info))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Calculate final metrics
        total_time = time.time() - start_time
        final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        memory_usage = final_memory - initial_memory
        
        # Calculate statistics
        if latencies:
            latency_mean = statistics.mean(latencies)
            latency_p50 = statistics.median(latencies)
            latency_p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else latency_mean
            latency_p99 = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else latency_mean
            latency_max = max(latencies)
            
            throughput_rps = len(latencies) / total_time
            
            # Estimate tokens per second
            if response_lengths:
                avg_response_length = statistics.mean(response_lengths)
                tokens_per_second = throughput_rps * avg_response_length
            else:
                tokens_per_second = 0
        else:
            latency_mean = latency_p50 = latency_p95 = latency_p99 = latency_max = 0
            throughput_rps = tokens_per_second = 0
        
        # Quality metrics
        response_quality_score = statistics.mean(quality_scores) if quality_scores else 0
        coherence_score = statistics.mean(coherence_scores) if coherence_scores else 0
        relevance_score = statistics.mean(relevance_scores) if relevance_scores else 0
        
        # Error rates
        total_requests = config['num_requests']
        error_rate = errors / total_requests
        timeout_rate = timeouts / total_requests
        
        # Resource usage
        cpu_usage_percent = statistics.mean(cpu_samples) if cpu_samples else 0
        
        # Context metrics (simplified)
        max_context_handled = 2048  # Default context window
        context_efficiency = 0.8 if latency_mean > 0 else 0
        
        # Calculate overall performance score
        performance_score = self._calculate_performance_score(
            throughput_rps, latency_p95, error_rate, response_quality_score
        )
        
        # Create benchmark result
        benchmark_result = BenchmarkResult(
            model=model,
            version=self.models.get(model, ModelVersion('', '', '', 0, '', '', '', datetime.now())).version,
            timestamp=datetime.now(),
            latency_mean=latency_mean,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            latency_max=latency_max,
            throughput_rps=throughput_rps,
            tokens_per_second=tokens_per_second,
            response_quality_score=response_quality_score,
            coherence_score=coherence_score,
            relevance_score=relevance_score,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage_percent,
            gpu_usage_percent=0.0,  # Not available in current setup
            error_rate=error_rate,
            timeout_rate=timeout_rate,
            max_context_handled=max_context_handled,
            context_efficiency=context_efficiency,
            performance_score=performance_score
        )
        
        # Save to database
        self._save_benchmark_result(benchmark_result)
        
        # Update Prometheus metrics
        self._update_prometheus_metrics(benchmark_result)
        
        logger.info(f"Benchmark completed for {model}:")
        logger.info(f"  Performance Score: {performance_score:.2f}")
        logger.info(f"  Throughput: {throughput_rps:.2f} req/s")
        logger.info(f"  Latency P95: {latency_p95*1000:.0f}ms")
        logger.info(f"  Error Rate: {error_rate*100:.1f}%")
        logger.info(f"  Quality Score: {response_quality_score:.2f}")
        
        return benchmark_result
    
    def _generate_benchmark_prompts(self, num_prompts: int) -> List[Dict[str, Any]]:
        """Generate diverse benchmark prompts"""
        prompt_templates = [
            {
                'prompt': 'Explain the concept of artificial intelligence in simple terms.',
                'expected_type': 'explanation',
                'max_tokens': 200
            },
            {
                'prompt': 'Write a short story about a robot discovering emotions.',
                'expected_type': 'creative',
                'max_tokens': 300
            },
            {
                'prompt': 'List the steps to implement a binary search algorithm.',
                'expected_type': 'technical',
                'max_tokens': 250
            },
            {
                'prompt': 'Compare the advantages and disadvantages of renewable energy.',
                'expected_type': 'analytical',
                'max_tokens': 300
            },
            {
                'prompt': 'Summarize the main causes of climate change.',
                'expected_type': 'factual',
                'max_tokens': 150
            },
            {
                'prompt': 'Describe the process of photosynthesis.',
                'expected_type': 'scientific',
                'max_tokens': 200
            },
            {
                'prompt': 'What are the key principles of good software design?',
                'expected_type': 'technical',
                'max_tokens': 250
            },
            {
                'prompt': 'Explain the importance of data privacy in the digital age.',
                'expected_type': 'ethical',
                'max_tokens': 200
            }
        ]
        
        prompts = []
        for i in range(num_prompts):
            template = prompt_templates[i % len(prompt_templates)]
            prompts.append(template.copy())
        
        return prompts
    
    def _evaluate_response_quality(self, prompt: str, response: str, expected_type: str) -> float:
        """Evaluate response quality (simplified heuristic)"""
        if not response or len(response.strip()) < 10:
            return 0.0
        
        score = 0.5  # Base score
        
        # Length appropriateness
        response_length = len(response.split())
        if 50 <= response_length <= 500:
            score += 0.2
        
        # Contains relevant keywords based on type
        keywords = {
            'explanation': ['because', 'means', 'refers to', 'definition'],
            'creative': ['story', 'character', 'narrative', 'plot'],
            'technical': ['step', 'algorithm', 'process', 'method'],
            'analytical': ['advantage', 'disadvantage', 'compare', 'contrast'],
            'factual': ['fact', 'evidence', 'data', 'research'],
            'scientific': ['process', 'mechanism', 'theory', 'study'],
            'ethical': ['important', 'should', 'consider', 'responsibility']
        }
        
        type_keywords = keywords.get(expected_type, [])
        keyword_matches = sum(1 for kw in type_keywords if kw.lower() in response.lower())
        score += min(0.2, keyword_matches * 0.05)
        
        # Grammar and coherence (simplified)
        sentences = response.split('.')
        if len(sentences) > 1:
            score += 0.1
        
        return min(1.0, score)
    
    def _evaluate_coherence(self, response: str) -> float:
        """Evaluate response coherence (simplified)"""
        if not response:
            return 0.0
        
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if len(sentences) <= 1:
            return 0.5
        
        # Check for repeated words or phrases
        words = response.lower().split()
        unique_words = set(words)
        diversity_ratio = len(unique_words) / len(words) if words else 0
        
        # Check sentence length variety
        sentence_lengths = [len(s.split()) for s in sentences]
        if sentence_lengths:
            length_variance = statistics.variance(sentence_lengths) if len(sentence_lengths) > 1 else 0
            coherence_score = min(1.0, diversity_ratio + (length_variance / 100))
        else:
            coherence_score = 0.0
        
        return coherence_score
    
    def _evaluate_relevance(self, prompt: str, response: str) -> float:
        """Evaluate response relevance to prompt"""
        if not response or not prompt:
            return 0.0
        
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        # Calculate word overlap
        common_words = prompt_words.intersection(response_words)
        relevance_score = len(common_words) / len(prompt_words) if prompt_words else 0
        
        return min(1.0, relevance_score * 2)  # Scale up the score
    
    def _calculate_performance_score(self, 
                                   throughput: float, 
                                   latency_p95: float, 
                                   error_rate: float, 
                                   quality_score: float) -> float:
        """Calculate overall performance score"""
        # Normalize metrics (0-1 scale)
        throughput_score = min(1.0, throughput / 100)  # Max 100 RPS = 1.0
        latency_score = max(0.0, 1.0 - (latency_p95 / 10))  # 10s latency = 0, 0s = 1.0
        error_score = max(0.0, 1.0 - (error_rate * 10))  # 10% error = 0, 0% = 1.0
        quality_normalized = quality_score  # Already 0-1
        
        # Weighted average
        weights = {'throughput': 0.3, 'latency': 0.3, 'error': 0.2, 'quality': 0.2}
        
        performance_score = (
            weights['throughput'] * throughput_score +
            weights['latency'] * latency_score +
            weights['error'] * error_score +
            weights['quality'] * quality_normalized
        )
        
        return performance_score
    
    def _save_benchmark_result(self, result: BenchmarkResult):
        """Save benchmark result to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO benchmark_results 
                (model, version, timestamp, latency_mean, latency_p50, latency_p95, 
                 latency_p99, latency_max, throughput_rps, tokens_per_second, 
                 response_quality_score, coherence_score, relevance_score, 
                 memory_usage_mb, cpu_usage_percent, gpu_usage_percent, 
                 error_rate, timeout_rate, max_context_handled, context_efficiency, 
                 performance_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.model, result.version, result.timestamp,
                result.latency_mean, result.latency_p50, result.latency_p95,
                result.latency_p99, result.latency_max, result.throughput_rps,
                result.tokens_per_second, result.response_quality_score,
                result.coherence_score, result.relevance_score,
                result.memory_usage_mb, result.cpu_usage_percent,
                result.gpu_usage_percent, result.error_rate, result.timeout_rate,
                result.max_context_handled, result.context_efficiency,
                result.performance_score
            ))
    
    def _update_prometheus_metrics(self, result: BenchmarkResult):
        """Update Prometheus metrics"""
        self.model_performance_score.labels(
            model=result.model, 
            version=result.version
        ).set(result.performance_score)
        
        self.model_throughput.labels(
            model=result.model, 
            version=result.version
        ).set(result.throughput_rps)
        
        self.model_memory_usage.labels(
            model=result.model, 
            version=result.version
        ).set(result.memory_usage_mb)
    
    def get_model_performance_history(self, model: str, days: int = 30) -> List[BenchmarkResult]:
        """Get performance history for a model"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM benchmark_results 
                WHERE model = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            ''', (model, cutoff_date))
            
            results = []
            for row in cursor.fetchall():
                # Convert row to BenchmarkResult (simplified)
                result = BenchmarkResult(
                    model=row[1], version=row[2], timestamp=datetime.fromisoformat(row[3]),
                    latency_mean=row[4], latency_p50=row[5], latency_p95=row[6],
                    latency_p99=row[7], latency_max=row[8], throughput_rps=row[9],
                    tokens_per_second=row[10], response_quality_score=row[11],
                    coherence_score=row[12], relevance_score=row[13],
                    memory_usage_mb=row[14], cpu_usage_percent=row[15],
                    gpu_usage_percent=row[16], error_rate=row[17],
                    timeout_rate=row[18], max_context_handled=row[19],
                    context_efficiency=row[20], performance_score=row[21]
                )
                results.append(result)
            
            return results
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'benchmarks': [],
            'recommendations': [],
            'summary': {}
        }
        
        # Get recent benchmarks for all models
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT model, AVG(performance_score) as avg_score, 
                       AVG(throughput_rps) as avg_throughput,
                       AVG(latency_p95) as avg_latency,
                       AVG(error_rate) as avg_error_rate,
                       COUNT(*) as benchmark_count
                FROM benchmark_results 
                WHERE timestamp >= datetime('now', '-7 days')
                GROUP BY model
                ORDER BY avg_score DESC
            ''')
            
            model_summaries = {}
            for row in cursor.fetchall():
                model, avg_score, avg_throughput, avg_latency, avg_error_rate, count = row
                model_summaries[model] = {
                    'performance_score': avg_score,
                    'throughput_rps': avg_throughput,
                    'latency_p95_ms': avg_latency * 1000,
                    'error_rate_percent': avg_error_rate * 100,
                    'benchmark_count': count
                }
            
            report['models'] = model_summaries
        
        # Generate recommendations
        if model_summaries:
            best_model = max(model_summaries.keys(), key=lambda m: model_summaries[m]['performance_score'])
            fastest_model = max(model_summaries.keys(), key=lambda m: model_summaries[m]['throughput_rps'])
            
            report['recommendations'] = [
                f"Best overall performance: {best_model} (score: {model_summaries[best_model]['performance_score']:.2f})",
                f"Highest throughput: {fastest_model} ({model_summaries[fastest_model]['throughput_rps']:.1f} req/s)",
            ]
            
            # Add specific recommendations
            for model, stats in model_summaries.items():
                if stats['error_rate_percent'] > 5:
                    report['recommendations'].append(f"High error rate for {model}: {stats['error_rate_percent']:.1f}% - investigate")
                if stats['latency_p95_ms'] > 5000:
                    report['recommendations'].append(f"High latency for {model}: {stats['latency_p95_ms']:.0f}ms - consider optimization")
        
        # Save report
        report_path = f"/opt/sutazaiapp/logs/model_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved to: {report_path}")
        return report
    
    async def run_all_model_benchmarks(self, benchmark_type: str = 'standard') -> Dict[str, BenchmarkResult]:
        """Run benchmarks on all available models"""
        logger.info(f"Running {benchmark_type} benchmarks on all models")
        
        # Discover models first
        await self.discover_models()
        
        results = {}
        for model_name in self.models.keys():
            try:
                logger.info(f"Benchmarking {model_name}...")
                result = await self.run_comprehensive_benchmark(model_name, benchmark_type)
                results[model_name] = result
                
                # Brief pause between models to prevent overload
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Failed to benchmark {model_name}: {e}")
        
        # Generate final report
        self.generate_performance_report()
        
        logger.info(f"Completed benchmarks for {len(results)} models")
        return results

# CLI interface
async def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ollama Model Manager and Benchmarking System')
    parser.add_argument('--discover', action='store_true', help='Discover available models')
    parser.add_argument('--benchmark', choices=['gpt-oss', 'gpt-oss.2:3b', 'gpt-oss-r1:8b', 'all'], help='Run benchmark')
    parser.add_argument('--benchmark-type', choices=['quick', 'standard', 'comprehensive'], default='standard', help='Benchmark type')
    parser.add_argument('--report', action='store_true', help='Generate performance report')
    parser.add_argument('--history', help='Show performance history for model')
    parser.add_argument('--days', type=int, default=30, help='Days of history to show')
    
    args = parser.parse_args()
    
    manager = OllamaModelManager()
    
    try:
        if args.discover:
            models = await manager.discover_models()
            print(f"Discovered {len(models)} models:")
            for name, model in models.items():
                print(f"  {name}: {model.parameter_count} parameters, {model.size_bytes/(1024*1024):.0f}MB")
        
        elif args.benchmark:
            if args.benchmark == 'all':
                results = await manager.run_all_model_benchmarks(args.benchmark_type)
                print(f"Completed benchmarks for {len(results)} models")
            else:
                result = await manager.run_comprehensive_benchmark(args.benchmark, args.benchmark_type)
                print(f"Benchmark completed with performance score: {result.performance_score:.2f}")
        
        elif args.report:
            report = manager.generate_performance_report()
            print(json.dumps(report, indent=2))
        
        elif args.history:
            history = manager.get_model_performance_history(args.history, args.days)
            print(f"Performance history for {args.history} (last {args.days} days):")
            for result in history[:10]:  # Show last 10 benchmarks
                print(f"  {result.timestamp}: Score {result.performance_score:.2f}, "
                      f"Throughput {result.throughput_rps:.1f} req/s")
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    
    finally:
        await manager.client.aclose()

if __name__ == '__main__':
    asyncio.run(main())