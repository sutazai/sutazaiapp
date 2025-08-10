#!/usr/bin/env python3
"""
SutazAI Performance Baseline Creation Script
Creates comprehensive performance baselines and monitoring thresholds
"""

import json
import time
import psutil
import docker
import requests
from typing import Dict, List, Any
import logging
import os
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('baseline-creator')

class PerformanceBaseline:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.baseline_db = '/opt/sutazaiapp/data/performance_baseline.db'
        self.baseline_file = '/opt/sutazaiapp/monitoring/performance_baseline.json'
        self.setup_database()
        
    def setup_database(self):
        """Setup SQLite database for performance tracking"""
        os.makedirs(os.path.dirname(self.baseline_db), exist_ok=True)
        
        conn = sqlite3.connect(self.baseline_db)
        cursor = conn.cursor()
        
        # System metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                timestamp TEXT PRIMARY KEY,
                cpu_percent REAL,
                memory_percent REAL,
                memory_available_gb REAL,
                load_average REAL,
                disk_usage_percent REAL,
                container_count INTEGER
            )
        ''')
        
        # Container metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS container_metrics (
                timestamp TEXT,
                container_name TEXT,
                status TEXT,
                health_status TEXT,
                cpu_percent REAL,
                memory_usage_mb REAL,
                memory_limit_mb REAL,
                network_rx_mb REAL,
                network_tx_mb REAL,
                PRIMARY KEY (timestamp, container_name)
            )
        ''')
        
        # Agent performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_performance (
                timestamp TEXT,
                agent_name TEXT,
                response_time_ms REAL,
                requests_per_minute REAL,
                success_rate REAL,
                error_count INTEGER,
                PRIMARY KEY (timestamp, agent_name)
            )
        ''')
        
        # Performance baselines table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_baselines (
                metric_name TEXT PRIMARY KEY,
                baseline_value REAL,
                threshold_warning REAL,
                threshold_critical REAL,
                measurement_unit TEXT,
                last_updated TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system performance metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        load_avg = psutil.getloadavg()[0]
        
        containers = self.docker_client.containers.list(all=True)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'load_average': load_avg,
            'disk_usage_percent': disk.percent,
            'container_count': len(containers),
            'memory_total_gb': memory.total / (1024**3),
            'cpu_count': psutil.cpu_count()
        }
    
    def collect_container_metrics(self) -> List[Dict[str, Any]]:
        """Collect metrics for all containers"""
        metrics = []
        containers = self.docker_client.containers.list(all=True)
        
        for container in containers:
            try:
                if container.status == 'running':
                    stats = container.stats(stream=False)
                    
                    # Calculate CPU percentage
                    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                               stats['precpu_stats']['cpu_usage']['total_usage']
                    system_cpu_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                      stats['precpu_stats']['system_cpu_usage']
                    cpu_percent = (cpu_delta / system_cpu_delta) * 100.0 if system_cpu_delta > 0 else 0
                    
                    # Get memory usage
                    memory_usage = stats['memory_stats'].get('usage', 0) / (1024*1024)  # MB
                    memory_limit = stats['memory_stats'].get('limit', 0) / (1024*1024)  # MB
                    
                    # Get network I/O
                    networks = stats.get('networks', {})
                    rx_bytes = sum(net.get('rx_bytes', 0) for net in networks.values())
                    tx_bytes = sum(net.get('tx_bytes', 0) for net in networks.values())
                    
                    health_status = container.attrs.get('State', {}).get('Health', {}).get('Status', 'no-health-check')
                else:
                    cpu_percent = 0
                    memory_usage = 0
                    memory_limit = 0
                    rx_bytes = 0
                    tx_bytes = 0
                    health_status = 'stopped'
                
                metrics.append({
                    'timestamp': datetime.now().isoformat(),
                    'container_name': container.name,
                    'status': container.status,
                    'health_status': health_status,
                    'cpu_percent': cpu_percent,
                    'memory_usage_mb': memory_usage,
                    'memory_limit_mb': memory_limit,
                    'network_rx_mb': rx_bytes / (1024*1024),
                    'network_tx_mb': tx_bytes / (1024*1024)
                })
                
            except Exception as e:
                logger.warning(f"Error collecting metrics for {container.name}: {e}")
                
        return metrics
    
    def test_agent_performance(self) -> List[Dict[str, Any]]:
        """Test AI agent response times and performance"""
        agent_metrics = []
        
        # Test agents with health endpoints
        agent_ports = range(11045, 11069)  # Phase 3 agent ports
        
        for port in agent_ports:
            try:
                start_time = time.time()
                response = requests.get(f'http://localhost:{port}/health', timeout=10)
                response_time = (time.time() - start_time) * 1000  # milliseconds
                
                success = response.status_code == 200
                agent_name = f"agent-port-{port}"
                
                agent_metrics.append({
                    'timestamp': datetime.now().isoformat(),
                    'agent_name': agent_name,
                    'response_time_ms': response_time,
                    'success_rate': 1.0 if success else 0.0,
                    'status_code': response.status_code
                })
                
            except requests.RequestException as e:
                agent_metrics.append({
                    'timestamp': datetime.now().isoformat(),
                    'agent_name': f"agent-port-{port}",
                    'response_time_ms': 0,
                    'success_rate': 0.0,
                    'error': str(e)
                })
                
        return agent_metrics
    
    def test_ollama_performance(self) -> Dict[str, Any]:
        """Test Ollama inference performance"""
        try:
            # Test simple inference
            start_time = time.time()
            response = requests.post(
                'http://localhost:10104/api/generate',
                json={
                    'model': 'tinyllama:latest',
                    'prompt': 'Hello, world!',
                    'stream': False
                },
                timeout=30
            )
            inference_time = (time.time() - start_time) * 1000
            
            success = response.status_code == 200
            
            return {
                'timestamp': datetime.now().isoformat(),
                'inference_time_ms': inference_time,
                'success': success,
                'model': 'tinyllama:latest'
            }
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'inference_time_ms': 0,
                'success': False,
                'error': str(e)
            }
    
    def calculate_baselines(self, sample_duration_minutes: int = 30) -> Dict[str, Any]:
        """Calculate performance baselines over a sampling period"""
        logger.info(f"Collecting baseline data for {sample_duration_minutes} minutes...")
        
        system_samples = []
        container_samples = []
        agent_samples = []
        ollama_samples = []
        
        sample_interval = 30  # seconds
        total_samples = (sample_duration_minutes * 60) // sample_interval
        
        for i in range(total_samples):
            logger.info(f"Collecting sample {i+1}/{total_samples}")
            
            # Collect system metrics
            system_metrics = self.collect_system_metrics()
            system_samples.append(system_metrics)
            
            # Collect container metrics
            container_metrics = self.collect_container_metrics()
            container_samples.extend(container_metrics)
            
            # Test agent performance (every 5th sample to avoid overload)
            if i % 5 == 0:
                agent_metrics = self.test_agent_performance()
                agent_samples.extend(agent_metrics)
                
                ollama_metrics = self.test_ollama_performance()
                ollama_samples.append(ollama_metrics)
            
            # Wait for next sample
            if i < total_samples - 1:
                time.sleep(sample_interval)
        
        # Calculate baseline statistics
        baselines = self.analyze_samples(system_samples, container_samples, agent_samples, ollama_samples)
        
        # Save to database and file
        self.save_baselines(baselines)
        
        return baselines
    
    def analyze_samples(self, system_samples, container_samples, agent_samples, ollama_samples) -> Dict[str, Any]:
        """Analyze collected samples to create baselines"""
        import statistics
        
        baselines = {
            'creation_time': datetime.now().isoformat(),
            'sample_count': len(system_samples),
            'system_baselines': {},
            'container_baselines': {},
            'agent_baselines': {},
            'ollama_baselines': {},
            'thresholds': {}
        }
        
        # System baselines
        if system_samples:
            cpu_values = [s['cpu_percent'] for s in system_samples]
            memory_values = [s['memory_percent'] for s in system_samples]
            load_values = [s['load_average'] for s in system_samples]
            
            baselines['system_baselines'] = {
                'cpu_percent': {
                    'mean': statistics.mean(cpu_values),
                    'median': statistics.median(cpu_values),
                    'std_dev': statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0,
                    'min': min(cpu_values),
                    'max': max(cpu_values)
                },
                'memory_percent': {
                    'mean': statistics.mean(memory_values),
                    'median': statistics.median(memory_values),
                    'std_dev': statistics.stdev(memory_values) if len(memory_values) > 1 else 0,
                    'min': min(memory_values),
                    'max': max(memory_values)
                },
                'load_average': {
                    'mean': statistics.mean(load_values),
                    'median': statistics.median(load_values),
                    'std_dev': statistics.stdev(load_values) if len(load_values) > 1 else 0,
                    'min': min(load_values),
                    'max': max(load_values)
                }
            }
        
        # Container health baselines
        if container_samples:
            sutazai_containers = [c for c in container_samples if 'sutazai-' in c['container_name']]
            
            if sutazai_containers:
                healthy_count = sum(1 for c in sutazai_containers if c['health_status'] == 'healthy')
                total_count = len(sutazai_containers)
                health_rate = (healthy_count / total_count * 100) if total_count > 0 else 0
                
                memory_usage = [c['memory_usage_mb'] for c in sutazai_containers if c['memory_usage_mb'] > 0]
                cpu_usage = [c['cpu_percent'] for c in sutazai_containers if c['cpu_percent'] > 0]
                
                baselines['container_baselines'] = {
                    'health_rate_percent': health_rate,
                    'total_containers': total_count,
                    'memory_usage_mb': {
                        'mean': statistics.mean(memory_usage) if memory_usage else 0,
                        'median': statistics.median(memory_usage) if memory_usage else 0,
                        'max': max(memory_usage) if memory_usage else 0
                    },
                    'cpu_percent': {
                        'mean': statistics.mean(cpu_usage) if cpu_usage else 0,
                        'median': statistics.median(cpu_usage) if cpu_usage else 0,
                        'max': max(cpu_usage) if cpu_usage else 0
                    }
                }
        
        # Agent performance baselines
        if agent_samples:
            successful_agents = [a for a in agent_samples if a.get('success_rate', 0) > 0]
            response_times = [a['response_time_ms'] for a in successful_agents]
            
            if response_times:
                baselines['agent_baselines'] = {
                    'response_time_ms': {
                        'mean': statistics.mean(response_times),
                        'median': statistics.median(response_times),
                        'p95': sorted(response_times)[int(len(response_times) * 0.95)],
                        'max': max(response_times)
                    },
                    'success_rate': len(successful_agents) / len(agent_samples) if agent_samples else 0
                }
        
        # Ollama performance baselines
        if ollama_samples:
            successful_ollama = [o for o in ollama_samples if o.get('success', False)]
            inference_times = [o['inference_time_ms'] for o in successful_ollama]
            
            if inference_times:
                baselines['ollama_baselines'] = {
                    'inference_time_ms': {
                        'mean': statistics.mean(inference_times),
                        'median': statistics.median(inference_times),
                        'p95': sorted(inference_times)[int(len(inference_times) * 0.95)],
                        'max': max(inference_times)
                    },
                    'success_rate': len(successful_ollama) / len(ollama_samples) if ollama_samples else 0
                }
        
        # Calculate alert thresholds
        baselines['thresholds'] = self.calculate_thresholds(baselines)
        
        return baselines
    
    def calculate_thresholds(self, baselines: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate warning and critical thresholds based on baselines"""
        thresholds = {}
        
        # System thresholds
        if 'system_baselines' in baselines:
            sys_base = baselines['system_baselines']
            
            thresholds['system'] = {
                'cpu_percent': {
                    'warning': min(70, sys_base.get('cpu_percent', {}).get('mean', 50) + 20),
                    'critical': min(85, sys_base.get('cpu_percent', {}).get('mean', 50) + 35)
                },
                'memory_percent': {
                    'warning': min(80, sys_base.get('memory_percent', {}).get('mean', 40) + 30),
                    'critical': min(90, sys_base.get('memory_percent', {}).get('mean', 40) + 45)
                },
                'load_average': {
                    'warning': sys_base.get('load_average', {}).get('mean', 2) + 2,
                    'critical': sys_base.get('load_average', {}).get('mean', 2) + 4
                }
            }
        
        # Container thresholds
        if 'container_baselines' in baselines:
            thresholds['container'] = {
                'health_rate_percent': {
                    'warning': 80,
                    'critical': 70
                },
                'memory_usage_mb': {
                    'warning': baselines['container_baselines'].get('memory_usage_mb', {}).get('mean', 256) * 2,
                    'critical': baselines['container_baselines'].get('memory_usage_mb', {}).get('mean', 256) * 3
                }
            }
        
        # Agent thresholds
        if 'agent_baselines' in baselines:
            agent_base = baselines['agent_baselines']
            thresholds['agent'] = {
                'response_time_ms': {
                    'warning': agent_base.get('response_time_ms', {}).get('mean', 1000) * 3,
                    'critical': agent_base.get('response_time_ms', {}).get('mean', 1000) * 5
                },
                'success_rate': {
                    'warning': 0.9,
                    'critical': 0.8
                }
            }
        
        # Ollama thresholds  
        if 'ollama_baselines' in baselines:
            ollama_base = baselines['ollama_baselines']
            thresholds['ollama'] = {
                'inference_time_ms': {
                    'warning': ollama_base.get('inference_time_ms', {}).get('mean', 2000) * 2,
                    'critical': ollama_base.get('inference_time_ms', {}).get('mean', 2000) * 4
                }
            }
        
        return thresholds
    
    def save_baselines(self, baselines: Dict[str, Any]):
        """Save baselines to database and JSON file"""
        # Save to JSON file
        os.makedirs(os.path.dirname(self.baseline_file), exist_ok=True)
        with open(self.baseline_file, 'w') as f:
            json.dump(baselines, f, indent=2)
        
        # Save to database
        conn = sqlite3.connect(self.baseline_db)
        cursor = conn.cursor()
        
        # Save key metrics as baselines
        baseline_entries = [
            ('system_cpu_percent', baselines.get('system_baselines', {}).get('cpu_percent', {}).get('mean', 0), 
             baselines.get('thresholds', {}).get('system', {}).get('cpu_percent', {}).get('warning', 70),
             baselines.get('thresholds', {}).get('system', {}).get('cpu_percent', {}).get('critical', 85), '%'),
            ('system_memory_percent', baselines.get('system_baselines', {}).get('memory_percent', {}).get('mean', 0),
             baselines.get('thresholds', {}).get('system', {}).get('memory_percent', {}).get('warning', 80),
             baselines.get('thresholds', {}).get('system', {}).get('memory_percent', {}).get('critical', 90), '%'),
            ('container_health_rate', baselines.get('container_baselines', {}).get('health_rate_percent', 0),
             baselines.get('thresholds', {}).get('container', {}).get('health_rate_percent', {}).get('warning', 80),
             baselines.get('thresholds', {}).get('container', {}).get('health_rate_percent', {}).get('critical', 70), '%'),
            ('agent_response_time', baselines.get('agent_baselines', {}).get('response_time_ms', {}).get('mean', 0),
             baselines.get('thresholds', {}).get('agent', {}).get('response_time_ms', {}).get('warning', 3000),
             baselines.get('thresholds', {}).get('agent', {}).get('response_time_ms', {}).get('critical', 5000), 'ms')
        ]
        
        for entry in baseline_entries:
            cursor.execute('''
                INSERT OR REPLACE INTO performance_baselines 
                (metric_name, baseline_value, threshold_warning, threshold_critical, measurement_unit, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (*entry, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Baselines saved to {self.baseline_file} and database")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SutazAI Performance Baseline Creator')
    parser.add_argument('--duration', type=int, default=30,
                       help='Sampling duration in minutes (default: 30)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick baseline (5 minutes)')
    
    args = parser.parse_args()
    
    duration = 5 if args.quick else args.duration
    
    baseline_creator = PerformanceBaseline()
    baselines = baseline_creator.calculate_baselines(duration)
    
    print("\n=== Performance Baseline Summary ===")
    print(f"Baseline created: {baselines['creation_time']}")
    print(f"Sample count: {baselines['sample_count']}")
    
    if 'system_baselines' in baselines:
        sys_base = baselines['system_baselines']
        print(f"\nSystem Performance:")
        print(f"  CPU: {sys_base.get('cpu_percent', {}).get('mean', 0):.1f}% average")
        print(f"  Memory: {sys_base.get('memory_percent', {}).get('mean', 0):.1f}% average")
        print(f"  Load: {sys_base.get('load_average', {}).get('mean', 0):.2f} average")
    
    if 'container_baselines' in baselines:
        cont_base = baselines['container_baselines']
        print(f"\nContainer Health:")
        print(f"  Health Rate: {cont_base.get('health_rate_percent', 0):.1f}%")
        print(f"  Total Containers: {cont_base.get('total_containers', 0)}")
    
    if 'agent_baselines' in baselines:
        agent_base = baselines['agent_baselines']
        print(f"\nAgent Performance:")
        print(f"  Response Time: {agent_base.get('response_time_ms', {}).get('mean', 0):.0f}ms average")
        print(f"  Success Rate: {agent_base.get('success_rate', 0):.1%}")
    
    print(f"\nBaseline saved to: {baseline_creator.baseline_file}")

if __name__ == '__main__':
    main()