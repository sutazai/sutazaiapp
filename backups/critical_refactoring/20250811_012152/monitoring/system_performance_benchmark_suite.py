#!/usr/bin/env python3
"""
SutazAI System Performance Benchmark Suite
==========================================

Comprehensive benchmarking framework for the entire SutazAI ecosystem including:
- 90+ AI agents across all categories
- Advanced AGI orchestration layer
- Scalable architecture components
- Service mesh infrastructure
- Self-healing container systems
- Knowledge graph systems
- Multi-modal fusion capabilities
- Federated learning systems
"""

import asyncio
import json
import time
import psutil
import docker
import requests
import subprocess
import logging
import sqlite3
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prometheus_client.parser import text_string_to_metric_families

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Individual benchmark result data structure"""
    component: str
    category: str
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class SystemSnapshot:
    """System resource snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_bytes_sent: float
    network_bytes_recv: float
    active_containers: int
    running_agents: int

class AgentPerformanceBenchmark:
    """Individual agent performance benchmarking"""
    
    def __init__(self, agent_name: str, agent_port: int, agent_type: str):
        self.agent_name = agent_name
        self.agent_port = agent_port
        self.agent_type = agent_type
        self.base_url = f"http://localhost:{agent_port}"
    
    async def health_check(self) -> Tuple[bool, float]:
        """Test agent health endpoint response time"""
        start_time = time.time()
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response_time = time.time() - start_time
            return response.status_code == 200, response_time
        except Exception as e:
            logger.warning(f"Health check failed for {self.agent_name}: {e}")
            return False, time.time() - start_time
    
    async def load_test(self, concurrent_requests: int = 10, duration: int = 30) -> Dict[str, float]:
        """Load test the agent with concurrent requests"""
        results = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'min_response_time': float('inf'),
            'max_response_time': 0.0,
            'requests_per_second': 0.0
        }
        
        response_times = []
        start_time = time.time()
        
        async def make_request():
            try:
                start = time.time()
                response = requests.get(f"{self.base_url}/health", timeout=10)
                end = time.time()
                response_time = end - start
                response_times.append(response_time)
                return response.status_code == 200
            except Exception as e:
                logger.warning(f"Exception caught, returning: {e}")
                return False
        
        # Run load test for specified duration
        while time.time() - start_time < duration:
            tasks = [make_request() for _ in range(concurrent_requests)]
            results_batch = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results_batch:
                results['total_requests'] += 1
                if result is True:
                    results['successful_requests'] += 1
                else:
                    results['failed_requests'] += 1
            
            await asyncio.sleep(0.1)  # Brief pause between batches
        
        # Calculate statistics
        if response_times:
            results['avg_response_time'] = statistics.mean(response_times)
            results['min_response_time'] = min(response_times)
            results['max_response_time'] = max(response_times)
        
        total_time = time.time() - start_time
        results['requests_per_second'] = results['total_requests'] / total_time if total_time > 0 else 0
        
        return results

class SystemResourceMonitor:
    """System resource monitoring and benchmarking"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.baseline_metrics = {}
    
    def get_system_snapshot(self) -> SystemSnapshot:
        """Capture current system resource state"""
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        # Count active containers
        active_containers = len([c for c in self.docker_client.containers.list() if c.status == 'running'])
        
        # Estimate running agents (containers with 'agent' in name)
        running_agents = len([c for c in self.docker_client.containers.list() 
                            if c.status == 'running' and 'agent' in c.name.lower()])
        
        return SystemSnapshot(
            timestamp=datetime.now(),
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            disk_io_read_mb=disk_io.read_bytes / (1024**2) if disk_io else 0,
            disk_io_write_mb=disk_io.write_bytes / (1024**2) if disk_io else 0,
            network_bytes_sent=network_io.bytes_sent if network_io else 0,
            network_bytes_recv=network_io.bytes_recv if network_io else 0,
            active_containers=active_containers,
            running_agents=running_agents
        )
    
    def get_container_stats(self, container_name: str) -> Dict[str, Any]:
        """Get detailed container resource usage"""
        try:
            container = self.docker_client.containers.get(container_name)
            stats = container.stats(stream=False)
            
            # Calculate CPU percentage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
            cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0.0
            
            # Memory usage
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_percent = (memory_usage / memory_limit) * 100.0 if memory_limit > 0 else 0.0
            
            return {
                'cpu_percent': cpu_percent,
                'memory_usage_mb': memory_usage / (1024**2),
                'memory_limit_mb': memory_limit / (1024**2),
                'memory_percent': memory_percent,
                'status': container.status,
                'restart_count': container.attrs['RestartCount']
            }
        except Exception as e:
            logger.error(f"Failed to get stats for container {container_name}: {e}")
            return {}

class AGIOrchestrationBenchmark:
    """Benchmark AGI orchestration layer performance"""
    
    def __init__(self):
        self.orchestration_endpoints = [
            'http://localhost:8080/agi/status',
            'http://localhost:8080/agi/coordination',
            'http://localhost:8080/agi/collective-intelligence'
        ]
    
    async def test_orchestration_latency(self) -> Dict[str, float]:
        """Test AGI orchestration response times"""
        results = {}
        
        for endpoint in self.orchestration_endpoints:
            try:
                start_time = time.time()
                response = requests.get(endpoint, timeout=10)
                latency = time.time() - start_time
                results[endpoint.split('/')[-1]] = latency
            except Exception as e:
                logger.warning(f"AGI endpoint {endpoint} failed: {e}")
                results[endpoint.split('/')[-1]] = float('inf')
        
        return results
    
    async def test_multi_agent_coordination(self) -> Dict[str, Any]:
        """Test multi-agent coordination performance"""
        # Simulate coordination task
        coordination_metrics = {
            'task_distribution_time': 0.0,
            'agent_response_time': 0.0,
            'consensus_time': 0.0,
            'total_coordination_time': 0.0
        }
        
        start_time = time.time()
        
        # Simulate task distribution
        await asyncio.sleep(0.1)  # Simulated distribution delay
        coordination_metrics['task_distribution_time'] = time.time() - start_time
        
        # Simulate agent responses
        agent_start = time.time()
        await asyncio.sleep(0.2)  # Simulated agent processing
        coordination_metrics['agent_response_time'] = time.time() - agent_start
        
        # Simulate consensus building
        consensus_start = time.time()
        await asyncio.sleep(0.1)  # Simulated consensus delay
        coordination_metrics['consensus_time'] = time.time() - consensus_start
        
        coordination_metrics['total_coordination_time'] = time.time() - start_time
        
        return coordination_metrics

class ServiceMeshBenchmark:
    """Benchmark service mesh performance (Consul, Kong, RabbitMQ)"""
    
    def __init__(self):
        self.consul_url = "http://localhost:8500"
        self.kong_url = "http://localhost:8001"
        self.rabbitmq_url = "http://localhost:15672"
    
    async def test_service_discovery(self) -> Dict[str, float]:
        """Test service discovery performance"""
        results = {}
        
        # Test Consul service discovery
        try:
            start_time = time.time()
            response = requests.get(f"{self.consul_url}/v1/catalog/services", timeout=5)
            results['consul_discovery_time'] = time.time() - start_time
            results['consul_services_count'] = len(response.json()) if response.status_code == 200 else 0
        except Exception as e:
            logger.warning(f"Consul service discovery failed: {e}")
            results['consul_discovery_time'] = float('inf')
            results['consul_services_count'] = 0
        
        return results
    
    async def test_api_gateway_performance(self) -> Dict[str, float]:
        """Test Kong API gateway performance"""
        results = {}
        
        try:
            start_time = time.time()
            response = requests.get(f"{self.kong_url}/status", timeout=5)
            results['kong_status_time'] = time.time() - start_time
            results['kong_healthy'] = response.status_code == 200
        except Exception as e:
            logger.warning(f"Kong API gateway test failed: {e}")
            results['kong_status_time'] = float('inf')
            results['kong_healthy'] = False
        
        return results
    
    async def test_message_queue_performance(self) -> Dict[str, float]:
        """Test RabbitMQ message queue performance"""
        results = {}
        
        try:
            start_time = time.time()
            response = requests.get(f"{self.rabbitmq_url}/api/overview", 
                                  auth=('guest', 'guest'), timeout=5)
            results['rabbitmq_api_time'] = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                results['message_queue_count'] = data.get('queue_totals', {}).get('messages', 0)
                results['connection_count'] = data.get('object_totals', {}).get('connections', 0)
            else:
                results['message_queue_count'] = 0
                results['connection_count'] = 0
                
        except Exception as e:
            logger.warning(f"RabbitMQ test failed: {e}")
            results['rabbitmq_api_time'] = float('inf')
            results['message_queue_count'] = 0
            results['connection_count'] = 0
        
        return results

class PerformanceForecastingModel:
    """Performance forecasting using time series analysis"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for metrics storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                component TEXT,
                category TEXT,
                metric_name TEXT,
                value REAL,
                unit TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                cpu_percent REAL,
                memory_percent REAL,
                memory_used_gb REAL,
                memory_available_gb REAL,
                disk_io_read_mb REAL,
                disk_io_write_mb REAL,
                network_bytes_sent REAL,
                network_bytes_recv REAL,
                active_containers INTEGER,
                running_agents INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_benchmark_result(self, result: BenchmarkResult):
        """Store benchmark result in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO benchmark_results 
            (timestamp, component, category, metric_name, value, unit, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.timestamp,
            result.component,
            result.category,
            result.metric_name,
            result.value,
            result.unit,
            json.dumps(result.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def store_system_snapshot(self, snapshot: SystemSnapshot):
        """Store system snapshot in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO system_snapshots 
            (timestamp, cpu_percent, memory_percent, memory_used_gb, memory_available_gb,
             disk_io_read_mb, disk_io_write_mb, network_bytes_sent, network_bytes_recv,
             active_containers, running_agents)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            snapshot.timestamp,
            snapshot.cpu_percent,
            snapshot.memory_percent,
            snapshot.memory_used_gb,
            snapshot.memory_available_gb,
            snapshot.disk_io_read_mb,
            snapshot.disk_io_write_mb,
            snapshot.network_bytes_sent,
            snapshot.network_bytes_recv,
            snapshot.active_containers,
            snapshot.running_agents
        ))
        
        conn.commit()
        conn.close()
    
    def predict_resource_usage(self, hours_ahead: int = 24) -> Dict[str, float]:
        """Predict resource usage using simple trend analysis"""
        conn = sqlite3.connect(self.db_path)
        
        # Get recent data (last 7 days)
        df = pd.read_sql_query('''
            SELECT * FROM system_snapshots 
            WHERE timestamp > datetime('now', '-7 days')
            ORDER BY timestamp
        ''', conn)
        
        conn.close()
        
        if df.empty:
            return {}
        
        # Simple linear trend prediction
        predictions = {}
        
        for column in ['cpu_percent', 'memory_percent', 'active_containers', 'running_agents']:
            if column in df.columns and len(df[column]) > 1:
                # Calculate trend
                x = np.arange(len(df[column]))
                y = df[column].values
                
                # Linear regression
                z = np.polyfit(x, y, 1)
                
                # Predict future value
                future_x = len(df[column]) + (hours_ahead * 12)  # Assuming 12 samples per hour
                predicted_value = z[0] * future_x + z[1]
                
                predictions[f'{column}_predicted_{hours_ahead}h'] = max(0, predicted_value)
        
        return predictions

class SystemPerformanceBenchmarkSuite:
    """Main benchmark suite orchestrator"""
    
    def __init__(self, config_path: str = "/opt/sutazaiapp/config/benchmark_config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.resource_monitor = SystemResourceMonitor()
        self.agi_benchmark = AGIOrchestrationBenchmark()
        self.service_mesh_benchmark = ServiceMeshBenchmark()
        self.forecasting_model = PerformanceForecastingModel("/opt/sutazaiapp/data/performance_metrics.db")
        self.agent_benchmarks = self.init_agent_benchmarks()
    
    def load_config(self) -> Dict[str, Any]:
        """Load benchmark configuration"""
        default_config = {
            'agents': self.discover_agents(),
            'benchmark_duration': 300,  # 5 minutes
            'concurrent_requests': 10,
            'sampling_interval': 30,  # seconds
            'report_output_dir': '/opt/sutazaiapp/reports/performance',
            'sla_thresholds': {
                'agent_response_time_ms': 1000,
                'cpu_utilization_percent': 80,
                'memory_utilization_percent': 85,
                'error_rate_percent': 5
            }
        }
        
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
        except Exception as e:
            logger.warning(f"Failed to load config from {self.config_path}: {e}")
        
        return default_config
    
    def discover_agents(self) -> List[Dict[str, Any]]:
        """Discover active agents from registry and containers"""
        agents = []
        
        # Load agent registry
        try:
            with open('/opt/sutazaiapp/agents/agent_registry.json', 'r') as f:
                registry = json.load(f)
                
            # Get running containers
            docker_client = docker.from_env()
            running_containers = {c.name: c for c in docker_client.containers.list() if c.status == 'running'}
            
            port_counter = 8001  # Starting port for agents
            
            for agent_name, agent_info in registry.get('agents', {}).items():
                # Try to find corresponding container
                container_name = f"sutazaiapp-{agent_name}"
                if container_name in running_containers:
                    agents.append({
                        'name': agent_name,
                        'type': agent_info.get('category', 'unknown'),
                        'port': port_counter,
                        'container': container_name,
                        'capabilities': agent_info.get('capabilities', [])
                    })
                    port_counter += 1
                    
        except Exception as e:
            logger.error(f"Failed to discover agents: {e}")
        
        return agents
    
    def init_agent_benchmarks(self) -> List[AgentPerformanceBenchmark]:
        """Initialize agent benchmark instances"""
        benchmarks = []
        
        for agent_config in self.config['agents']:
            benchmark = AgentPerformanceBenchmark(
                agent_name=agent_config['name'],
                agent_port=agent_config['port'],
                agent_type=agent_config['type']
            )
            benchmarks.append(benchmark)
        
        return benchmarks
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        logger.info("Starting comprehensive system performance benchmark")
        
        benchmark_results = {
            'start_time': datetime.now().isoformat(),
            'system_info': self.get_system_info(),
            'baseline_snapshot': asdict(self.resource_monitor.get_system_snapshot()),
            'agent_performance': {},
            'agi_orchestration': {},
            'service_mesh': {},
            'resource_utilization': [],
            'performance_forecast': {},
            'sla_compliance': {},
            'recommendations': []
        }
        
        # 1. Baseline system snapshot
        baseline = self.resource_monitor.get_system_snapshot()
        self.forecasting_model.store_system_snapshot(baseline)
        
        # 2. Agent performance benchmarks
        logger.info("Benchmarking individual agents...")
        agent_results = await self.benchmark_all_agents()
        benchmark_results['agent_performance'] = agent_results
        
        # 3. AGI orchestration benchmarks
        logger.info("Benchmarking AGI orchestration layer...")
        agi_results = await self.benchmark_agi_orchestration()
        benchmark_results['agi_orchestration'] = agi_results
        
        # 4. Service mesh benchmarks
        logger.info("Benchmarking service mesh infrastructure...")
        service_mesh_results = await self.benchmark_service_mesh()
        benchmark_results['service_mesh'] = service_mesh_results
        
        # 5. Resource utilization monitoring
        logger.info("Monitoring resource utilization during load...")
        resource_results = await self.monitor_resource_utilization()
        benchmark_results['resource_utilization'] = resource_results
        
        # 6. Performance forecasting
        logger.info("Generating performance forecasts...")
        forecast_results = self.generate_performance_forecast()
        benchmark_results['performance_forecast'] = forecast_results
        
        # 7. SLA compliance analysis
        logger.info("Analyzing SLA compliance...")
        sla_results = self.analyze_sla_compliance(benchmark_results)
        benchmark_results['sla_compliance'] = sla_results
        
        # 8. Generate recommendations
        logger.info("Generating optimization recommendations...")
        recommendations = self.generate_recommendations(benchmark_results)
        benchmark_results['recommendations'] = recommendations
        
        benchmark_results['end_time'] = datetime.now().isoformat()
        benchmark_results['total_duration'] = (
            datetime.fromisoformat(benchmark_results['end_time']) - 
            datetime.fromisoformat(benchmark_results['start_time'])
        ).total_seconds()
        
        logger.info("Comprehensive benchmark completed")
        return benchmark_results
    
    async def benchmark_all_agents(self) -> Dict[str, Any]:
        """Benchmark all discovered agents"""
        results = {}
        
        # Test in batches to avoid overwhelming the system
        batch_size = 10
        
        for i in range(0, len(self.agent_benchmarks), batch_size):
            batch = self.agent_benchmarks[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[self.benchmark_single_agent(agent) for agent in batch],
                return_exceptions=True
            )
            
            for agent, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Agent {agent.agent_name} benchmark failed: {result}")
                    results[agent.agent_name] = {'error': str(result)}
                else:
                    results[agent.agent_name] = result
            
            # Brief pause between batches
            await asyncio.sleep(2)
        
        return results
    
    async def benchmark_single_agent(self, agent: AgentPerformanceBenchmark) -> Dict[str, Any]:
        """Benchmark a single agent"""
        result = {
            'agent_type': agent.agent_type,
            'health_check': {},
            'load_test': {},
            'resource_usage': {}
        }
        
        # Health check
        is_healthy, response_time = await agent.health_check()
        result['health_check'] = {
            'healthy': is_healthy,
            'response_time_ms': response_time * 1000
        }
        
        # Store benchmark result
        health_result = BenchmarkResult(
            component=agent.agent_name,
            category='agent',
            metric_name='health_response_time_ms',
            value=response_time * 1000,
            unit='milliseconds',
            timestamp=datetime.now(),
            metadata={'agent_type': agent.agent_type, 'healthy': is_healthy}
        )
        self.forecasting_model.store_benchmark_result(health_result)
        
        # Load test (only if healthy)
        if is_healthy:
            load_results = await agent.load_test(
                concurrent_requests=self.config['concurrent_requests'],
                duration=60  # 1 minute load test per agent
            )
            result['load_test'] = load_results
            
            # Store load test results
            for metric, value in load_results.items():
                if isinstance(value, (int, float)):
                    load_result = BenchmarkResult(
                        component=agent.agent_name,
                        category='agent',
                        metric_name=f'load_test_{metric}',
                        value=value,
                        unit='various',
                        timestamp=datetime.now(),
                        metadata={'agent_type': agent.agent_type}
                    )
                    self.forecasting_model.store_benchmark_result(load_result)
        
        # Resource usage
        container_stats = self.resource_monitor.get_container_stats(f"sutazaiapp-{agent.agent_name}")
        result['resource_usage'] = container_stats
        
        return result
    
    async def benchmark_agi_orchestration(self) -> Dict[str, Any]:
        """Benchmark AGI orchestration layer"""
        results = {}
        
        # Test orchestration latency
        orchestration_latency = await self.agi_benchmark.test_orchestration_latency()
        results['orchestration_latency'] = orchestration_latency
        
        # Test multi-agent coordination
        coordination_metrics = await self.agi_benchmark.test_multi_agent_coordination()
        results['coordination_metrics'] = coordination_metrics
        
        return results
    
    async def benchmark_service_mesh(self) -> Dict[str, Any]:
        """Benchmark service mesh components"""
        results = {}
        
        # Service discovery performance
        discovery_results = await self.service_mesh_benchmark.test_service_discovery()
        results['service_discovery'] = discovery_results
        
        # API gateway performance
        gateway_results = await self.service_mesh_benchmark.test_api_gateway_performance()
        results['api_gateway'] = gateway_results
        
        # Message queue performance
        queue_results = await self.service_mesh_benchmark.test_message_queue_performance()
        results['message_queue'] = queue_results
        
        return results
    
    async def monitor_resource_utilization(self) -> List[Dict[str, Any]]:
        """Monitor system resource utilization during benchmarks"""
        snapshots = []
        duration = 300  # 5 minutes
        interval = 30   # 30 seconds
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            snapshot = self.resource_monitor.get_system_snapshot()
            snapshots.append(asdict(snapshot))
            self.forecasting_model.store_system_snapshot(snapshot)
            await asyncio.sleep(interval)
        
        return snapshots
    
    def generate_performance_forecast(self) -> Dict[str, Any]:
        """Generate performance forecasts"""
        return {
            '24h_forecast': self.forecasting_model.predict_resource_usage(24),
            '7d_forecast': self.forecasting_model.predict_resource_usage(168),
            '30d_forecast': self.forecasting_model.predict_resource_usage(720)
        }
    
    def analyze_sla_compliance(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze SLA compliance based on benchmark results"""
        sla_compliance = {
            'overall_compliance': True,
            'violations': [],
            'compliance_score': 0.0,
            'agent_compliance': {}
        }
        
        thresholds = self.config['sla_thresholds']
        violations = []
        
        # Check agent response times
        for agent_name, agent_data in benchmark_results.get('agent_performance', {}).items():
            if 'error' in agent_data:
                violations.append(f"Agent {agent_name} failed completely")
                continue
                
            health_data = agent_data.get('health_check', {})
            response_time = health_data.get('response_time_ms', 0)
            
            if response_time > thresholds['agent_response_time_ms']:
                violations.append(
                    f"Agent {agent_name} response time {response_time:.2f}ms exceeds SLA {thresholds['agent_response_time_ms']}ms"
                )
            
            # Check error rates from load tests
            load_data = agent_data.get('load_test', {})
            if load_data:
                total_requests = load_data.get('total_requests', 0)
                failed_requests = load_data.get('failed_requests', 0)
                error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
                
                if error_rate > thresholds['error_rate_percent']:
                    violations.append(
                        f"Agent {agent_name} error rate {error_rate:.2f}% exceeds SLA {thresholds['error_rate_percent']}%"
                    )
        
        # Check system resource utilization
        resource_data = benchmark_results.get('resource_utilization', [])
        if resource_data:
            max_cpu = max([r.get('cpu_percent', 0) for r in resource_data])
            max_memory = max([r.get('memory_percent', 0) for r in resource_data])
            
            if max_cpu > thresholds['cpu_utilization_percent']:
                violations.append(
                    f"CPU utilization {max_cpu:.2f}% exceeds SLA {thresholds['cpu_utilization_percent']}%"
                )
            
            if max_memory > thresholds['memory_utilization_percent']:
                violations.append(
                    f"Memory utilization {max_memory:.2f}% exceeds SLA {thresholds['memory_utilization_percent']}%"
                )
        
        sla_compliance['violations'] = violations
        sla_compliance['overall_compliance'] = len(violations) == 0
        
        # Calculate compliance score
        total_checks = len(self.config['agents']) * 2 + 2  # 2 checks per agent + 2 system checks
        failed_checks = len(violations)
        sla_compliance['compliance_score'] = ((total_checks - failed_checks) / total_checks) * 100
        
        return sla_compliance
    
    def generate_recommendations(self, benchmark_results: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on benchmark results"""
        recommendations = []
        
        # Analyze agent performance
        agent_performance = benchmark_results.get('agent_performance', {})
        slow_agents = []
        failed_agents = []
        
        for agent_name, agent_data in agent_performance.items():
            if 'error' in agent_data:
                failed_agents.append(agent_name)
                continue
            
            health_data = agent_data.get('health_check', {})
            response_time = health_data.get('response_time_ms', 0)
            
            if response_time > 2000:  # >2 seconds is slow
                slow_agents.append((agent_name, response_time))
        
        if failed_agents:
            recommendations.append(
                f"CRITICAL: {len(failed_agents)} agents are failing: {', '.join(failed_agents[:5])}. "
                "Investigate container health and resource allocation."
            )
        
        if slow_agents:
            recommendations.append(
                f"PERFORMANCE: {len(slow_agents)} agents have slow response times. "
                f"Consider optimizing: {', '.join([a[0] for a in slow_agents[:3]])}"
            )
        
        # Analyze resource utilization
        resource_data = benchmark_results.get('resource_utilization', [])
        if resource_data:
            avg_cpu = statistics.mean([r.get('cpu_percent', 0) for r in resource_data])
            avg_memory = statistics.mean([r.get('memory_percent', 0) for r in resource_data])
            
            if avg_cpu > 70:
                recommendations.append(
                    f"CPU utilization averaging {avg_cpu:.1f}% is high. "
                    "Consider scaling horizontally or optimizing agent algorithms."
                )
            
            if avg_memory > 80:
                recommendations.append(
                    f"Memory utilization averaging {avg_memory:.1f}% is high. "
                    "Consider increasing memory limits or implementing memory optimization."
                )
        
        # AGI orchestration recommendations
        agi_data = benchmark_results.get('agi_orchestration', {})
        coord_metrics = agi_data.get('coordination_metrics', {})
        total_coord_time = coord_metrics.get('total_coordination_time', 0)
        
        if total_coord_time > 1.0:  # >1 second for coordination
            recommendations.append(
                "AGI coordination taking >1 second. Consider optimizing consensus algorithms "
                "or reducing agent communication overhead."
            )
        
        # Service mesh recommendations
        service_mesh_data = benchmark_results.get('service_mesh', {})
        consul_time = service_mesh_data.get('service_discovery', {}).get('consul_discovery_time', 0)
        
        if consul_time > 0.5:  # >500ms for service discovery
            recommendations.append(
                "Service discovery latency is high. Consider Consul performance tuning "
                "or implementing service discovery caching."
            )
        
        # Capacity planning recommendations
        forecast_data = benchmark_results.get('performance_forecast', {})
        if forecast_data:
            for forecast_period, predictions in forecast_data.items():
                cpu_predicted = predictions.get('cpu_percent_predicted_24h', 0)
                memory_predicted = predictions.get('memory_percent_predicted_24h', 0)
                
                if cpu_predicted > 90:
                    recommendations.append(
                        f"Predicted CPU utilization will exceed 90% within {forecast_period}. "
                        "Plan for capacity scaling or optimization."
                    )
                
                if memory_predicted > 95:
                    recommendations.append(
                        f"Predicted memory utilization will exceed 95% within {forecast_period}. "
                        "Plan for memory scaling or cleanup."
                    )
        
        if not recommendations:
            recommendations.append("System performance is optimal. No immediate action required.")
        
        return recommendations
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'docker_version': self.get_docker_version(),
            'agent_count': len(self.config['agents'])
        }
    
    def get_docker_version(self) -> str:
        """Get Docker version"""
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            return result.stdout.strip()
        except Exception as e:
            logger.warning(f"Exception caught, returning: {e}")
            return "unknown"
    
    async def generate_report(self, benchmark_results: Dict[str, Any]):
        """Generate comprehensive benchmark report"""
        report_dir = Path(self.config['report_output_dir'])
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON report
        json_report_path = report_dir / f'benchmark_report_{timestamp}.json'
        with open(json_report_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        # Generate visualizations
        await self.generate_visualizations(benchmark_results, report_dir, timestamp)
        
        # Markdown report
        markdown_report_path = report_dir / f'benchmark_report_{timestamp}.md'
        await self.generate_markdown_report(benchmark_results, markdown_report_path)
        
        logger.info(f"Reports generated in {report_dir}")
        
        return {
            'json_report': str(json_report_path),
            'markdown_report': str(markdown_report_path),
            'visualizations_dir': str(report_dir)
        }
    
    async def generate_visualizations(self, benchmark_results: Dict[str, Any], 
                                    report_dir: Path, timestamp: str):
        """Generate performance visualization charts"""
        plt.style.use('seaborn-v0_8')
        
        # Agent Response Time Distribution
        agent_response_times = []
        agent_names = []
        
        for agent_name, agent_data in benchmark_results.get('agent_performance', {}).items():
            if 'error' not in agent_data:
                response_time = agent_data.get('health_check', {}).get('response_time_ms', 0)
                if response_time > 0:
                    agent_response_times.append(response_time)
                    agent_names.append(agent_name[:15])  # Truncate long names
        
        if agent_response_times:
            plt.figure(figsize=(12, 8))
            plt.barh(agent_names[:20], agent_response_times[:20])  # Top 20 agents
            plt.xlabel('Response Time (ms)')
            plt.title('Agent Response Time Benchmark')
            plt.tight_layout()
            plt.savefig(report_dir / f'agent_response_times_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Resource Utilization Over Time
        resource_data = benchmark_results.get('resource_utilization', [])
        if resource_data:
            timestamps = [datetime.fromisoformat(r['timestamp']) for r in resource_data]
            cpu_data = [r['cpu_percent'] for r in resource_data]
            memory_data = [r['memory_percent'] for r in resource_data]
            
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, cpu_data, label='CPU %', linewidth=2)
            plt.plot(timestamps, memory_data, label='Memory %', linewidth=2)
            plt.xlabel('Time')
            plt.ylabel('Utilization %')
            plt.title('System Resource Utilization During Benchmark')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(report_dir / f'resource_utilization_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # SLA Compliance Score
        sla_data = benchmark_results.get('sla_compliance', {})
        compliance_score = sla_data.get('compliance_score', 0)
        
        plt.figure(figsize=(8, 8))
        colors = ['#ff4444' if compliance_score < 80 else '#44ff44' if compliance_score > 95 else '#ffaa44']
        plt.pie([compliance_score, 100 - compliance_score], 
                labels=[f'Compliant ({compliance_score:.1f}%)', f'Non-compliant ({100-compliance_score:.1f}%)'],
                colors=colors + ['#cccccc'],
                startangle=90)
        plt.title('SLA Compliance Score')
        plt.savefig(report_dir / f'sla_compliance_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    async def generate_markdown_report(self, benchmark_results: Dict[str, Any], 
                                     report_path: Path):
        """Generate detailed markdown report"""
        report_content = f"""# SutazAI System Performance Benchmark Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Duration:** {benchmark_results.get('total_duration', 0):.1f} seconds  
**System:** {benchmark_results.get('system_info', {}).get('platform', 'Unknown')}

## Executive Summary

### System Overview
- **Total Agents:** {len(benchmark_results.get('agent_performance', {}))}
- **CPU Cores:** {benchmark_results.get('system_info', {}).get('cpu_count', 'Unknown')}
- **Total Memory:** {benchmark_results.get('system_info', {}).get('memory_total_gb', 0):.1f} GB
- **Active Containers:** {benchmark_results.get('baseline_snapshot', {}).get('active_containers', 0)}

### Performance Highlights
"""
        
        # SLA Compliance
        sla_data = benchmark_results.get('sla_compliance', {})
        compliance_score = sla_data.get('compliance_score', 0)
        violations = sla_data.get('violations', [])
        
        report_content += f"""
### SLA Compliance: {compliance_score:.1f}%
{'✅ **PASSING**' if compliance_score > 90 else '⚠️ **WARNING**' if compliance_score > 70 else '❌ **FAILING**'}

"""
        
        if violations:
            report_content += "**Violations:**\n"
            for violation in violations[:10]:  # Top 10 violations
                report_content += f"- {violation}\n"
        
        # Agent Performance Summary
        agent_perf = benchmark_results.get('agent_performance', {})
        healthy_agents = sum(1 for data in agent_perf.values() 
                           if 'error' not in data and data.get('health_check', {}).get('healthy', False))
        
        report_content += f"""
## Agent Performance Analysis

### Health Status
- **Healthy Agents:** {healthy_agents}/{len(agent_perf)}
- **Failed Agents:** {len(agent_perf) - healthy_agents}

### Response Time Statistics
"""
        
        response_times = []
        for agent_data in agent_perf.values():
            if 'error' not in agent_data:
                rt = agent_data.get('health_check', {}).get('response_time_ms', 0)
                if rt > 0:
                    response_times.append(rt)
        
        if response_times:
            report_content += f"""
- **Average Response Time:** {statistics.mean(response_times):.2f}ms
- **Median Response Time:** {statistics.median(response_times):.2f}ms
- **95th Percentile:** {np.percentile(response_times, 95):.2f}ms
- **Slowest Agent:** {max(response_times):.2f}ms
"""
        
        # Resource Utilization
        resource_data = benchmark_results.get('resource_utilization', [])
        if resource_data:
            avg_cpu = statistics.mean([r.get('cpu_percent', 0) for r in resource_data])
            avg_memory = statistics.mean([r.get('memory_percent', 0) for r in resource_data])
            max_cpu = max([r.get('cpu_percent', 0) for r in resource_data])
            max_memory = max([r.get('memory_percent', 0) for r in resource_data])
            
            report_content += f"""
## Resource Utilization

### CPU Usage
- **Average:** {avg_cpu:.1f}%
- **Peak:** {max_cpu:.1f}%

### Memory Usage
- **Average:** {avg_memory:.1f}%
- **Peak:** {max_memory:.1f}%
"""
        
        # Recommendations
        recommendations = benchmark_results.get('recommendations', [])
        if recommendations:
            report_content += "\n## Optimization Recommendations\n\n"
            for i, rec in enumerate(recommendations, 1):
                report_content += f"{i}. {rec}\n"
        
        # Performance Forecast
        forecast_data = benchmark_results.get('performance_forecast', {})
        if forecast_data:
            report_content += "\n## Performance Forecast\n\n"
            for period, predictions in forecast_data.items():
                report_content += f"### {period.replace('_', ' ').title()}\n"
                for metric, value in predictions.items():
                    if 'cpu' in metric.lower():
                        report_content += f"- **CPU:** {value:.1f}%\n"
                    elif 'memory' in metric.lower():
                        report_content += f"- **Memory:** {value:.1f}%\n"
                    elif 'container' in metric.lower():
                        report_content += f"- **Containers:** {value:.0f}\n"
                    elif 'agent' in metric.lower():
                        report_content += f"- **Agents:** {value:.0f}\n"
        
        # Detailed Agent Results
        report_content += "\n## Detailed Agent Results\n\n"
        for agent_name, agent_data in sorted(agent_perf.items()):
            if 'error' in agent_data:
                report_content += f"### ❌ {agent_name} (FAILED)\n"
                report_content += f"**Error:** {agent_data['error']}\n\n"
            else:
                health_data = agent_data.get('health_check', {})
                load_data = agent_data.get('load_test', {})
                resource_data = agent_data.get('resource_usage', {})
                
                status = "✅" if health_data.get('healthy', False) else "❌"
                report_content += f"### {status} {agent_name}\n"
                
                if health_data:
                    report_content += f"- **Response Time:** {health_data.get('response_time_ms', 0):.2f}ms\n"
                
                if load_data:
                    rps = load_data.get('requests_per_second', 0)
                    avg_rt = load_data.get('avg_response_time', 0) * 1000
                    success_rate = (load_data.get('successful_requests', 0) / 
                                  max(load_data.get('total_requests', 1), 1)) * 100
                    
                    report_content += f"- **Throughput:** {rps:.1f} req/sec\n"
                    report_content += f"- **Avg Load Response:** {avg_rt:.2f}ms\n"
                    report_content += f"- **Success Rate:** {success_rate:.1f}%\n"
                
                if resource_data:
                    cpu = resource_data.get('cpu_percent', 0)
                    memory_mb = resource_data.get('memory_usage_mb', 0)
                    report_content += f"- **CPU Usage:** {cpu:.1f}%\n"
                    report_content += f"- **Memory Usage:** {memory_mb:.1f}MB\n"
                
                report_content += "\n"
        
        # Footer
        report_content += f"""
---
*Report generated by SutazAI Performance Benchmark Suite v1.0*  
*Timestamp: {datetime.now().isoformat()}*
"""
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(report_content)

# Main benchmark execution
async def main():
    """Main benchmark execution function"""
    benchmark_suite = SystemPerformanceBenchmarkSuite()
    
    try:
        # Run comprehensive benchmark
        results = await benchmark_suite.run_comprehensive_benchmark()
        
        # Generate reports
        report_info = await benchmark_suite.generate_report(results)
        
        print("\n" + "="*80)
        print("SUTAZAI SYSTEM PERFORMANCE BENCHMARK COMPLETED")
        print("="*80)
        print(f"Total Duration: {results['total_duration']:.1f} seconds")
        print(f"Agents Tested: {len(results['agent_performance'])}")
        print(f"SLA Compliance: {results['sla_compliance']['compliance_score']:.1f}%")
        print(f"JSON Report: {report_info['json_report']}")
        print(f"Markdown Report: {report_info['markdown_report']}")
        
        # Print summary recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print("\nKEY RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"{i}. {rec}")
        
        print("="*80)
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    # Import required packages
    try:
        import platform
    except ImportError:
        import sys
        sys.path.append('/opt/sutazaiapp')
        import platform
    
    # Run benchmark
    asyncio.run(main())