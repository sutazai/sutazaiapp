#!/usr/bin/env python3
"""
SutazAI System Benchmarking Tool
Comprehensive performance benchmarking and system testing
"""

import time
import sys
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List
import concurrent.futures
import statistics
import psutil
import threading

# Add current directory to Python path
sys.path.insert(0, '/opt/sutazaiapp')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemBenchmark:
    """Comprehensive system benchmarking suite"""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.baseline_metrics = {}
        
    def print_benchmark_header(self):
        """Print benchmark header"""
        header = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                       SutazAI System Benchmark Suite                        ║
║                                                                              ║
║                    Performance Testing & Validation                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(header)
        print(f"🚀 Starting benchmark at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def collect_baseline_metrics(self):
        """Collect baseline system metrics"""
        logger.info("📊 Collecting baseline system metrics...")
        
        self.baseline_metrics = {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024**3),
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"💻 System Specs:")
        print(f"   • CPU Cores: {self.baseline_metrics['cpu_count']}")
        print(f"   • Total Memory: {self.baseline_metrics['memory_total']:.2f} GB")
        print(f"   • CPU Usage: {self.baseline_metrics['cpu_usage']:.1f}%")
        print(f"   • Memory Usage: {self.baseline_metrics['memory_usage']:.1f}%")
        print(f"   • Disk Usage: {self.baseline_metrics['disk_usage']:.1f}%")
        print()
    
    def benchmark_agi_system(self):
        """Benchmark AGI system performance"""
        logger.info("🧠 Benchmarking AGI System...")
        
        try:
            from core.agi_system import IntegratedAGISystem, create_agi_task, TaskPriority
            
            # Initialize AGI system
            start_time = time.time()
            agi_system = IntegratedAGISystem()
            init_time = time.time() - start_time
            
            # Test task creation speed
            task_creation_times = []
            for i in range(100):
                start = time.time()
                task = create_agi_task(f'benchmark_task_{i}', TaskPriority.MEDIUM, {'test': i})
                task_creation_times.append(time.time() - start)
            
            # Test task submission speed
            task_submission_times = []
            for i in range(50):
                task = create_agi_task(f'submit_task_{i}', TaskPriority.HIGH, {'test': i})
                start = time.time()
                agi_system.submit_task(task)
                task_submission_times.append(time.time() - start)
            
            # Test system status retrieval speed
            status_times = []
            for i in range(20):
                start = time.time()
                status = agi_system.get_system_status()
                status_times.append(time.time() - start)
            
            self.results['agi_system'] = {
                'initialization_time': init_time,
                'task_creation': {
                    'avg_time': statistics.mean(task_creation_times),
                    'min_time': min(task_creation_times),
                    'max_time': max(task_creation_times),
                    'tasks_per_second': 1.0 / statistics.mean(task_creation_times)
                },
                'task_submission': {
                    'avg_time': statistics.mean(task_submission_times),
                    'min_time': min(task_submission_times),
                    'max_time': max(task_submission_times),
                    'tasks_per_second': 1.0 / statistics.mean(task_submission_times)
                },
                'status_retrieval': {
                    'avg_time': statistics.mean(status_times),
                    'min_time': min(status_times),
                    'max_time': max(status_times)
                }
            }
            
            print(f"🧠 AGI System Benchmark Results:")
            print(f"   • Initialization: {init_time:.3f}s")
            print(f"   • Task Creation: {self.results['agi_system']['task_creation']['tasks_per_second']:.1f} tasks/sec")
            print(f"   • Task Submission: {self.results['agi_system']['task_submission']['tasks_per_second']:.1f} tasks/sec")
            print(f"   • Status Retrieval: {self.results['agi_system']['status_retrieval']['avg_time']:.3f}s avg")
            print()
            
        except Exception as e:
            logger.error(f"AGI system benchmark failed: {e}")
            self.results['agi_system'] = {'error': str(e)}
    
    def benchmark_neural_network(self):
        """Benchmark neural network performance"""
        logger.info("🔗 Benchmarking Neural Network...")
        
        try:
            from nln.nln_core import NeuralLinkNetwork
            from nln.neural_node import NeuralNode
            from nln.neural_link import NeuralLink
            
            # Test network creation speed
            network_creation_times = []
            for i in range(10):
                start = time.time()
                network = NeuralLinkNetwork()
                
                # Add nodes
                for j in range(20):
                    node = NeuralNode(f'node_{j}', 'processing', (j, 0), 0.5)
                    network.add_node(node)
                
                # Add links
                for j in range(19):
                    link = NeuralLink(f'node_{j}', f'node_{j+1}', 0.5, 'excitatory')
                    network.add_link(link)
                
                network_creation_times.append(time.time() - start)
            
            # Test processing speed
            network = NeuralLinkNetwork()
            for i in range(10):
                node = NeuralNode(f'test_node_{i}', 'processing', (i, 0), 0.5)
                network.add_node(node)
            
            processing_times = []
            for i in range(100):
                start = time.time()
                result = network.process_input([0.5, 0.3, 0.8, 0.2, 0.7])
                processing_times.append(time.time() - start)
            
            self.results['neural_network'] = {
                'network_creation': {
                    'avg_time': statistics.mean(network_creation_times),
                    'min_time': min(network_creation_times),
                    'max_time': max(network_creation_times)
                },
                'processing': {
                    'avg_time': statistics.mean(processing_times),
                    'min_time': min(processing_times),
                    'max_time': max(processing_times),
                    'ops_per_second': 1.0 / statistics.mean(processing_times)
                }
            }
            
            print(f"🔗 Neural Network Benchmark Results:")
            print(f"   • Network Creation: {self.results['neural_network']['network_creation']['avg_time']:.3f}s avg")
            print(f"   • Processing Speed: {self.results['neural_network']['processing']['ops_per_second']:.1f} ops/sec")
            print(f"   • Processing Time: {self.results['neural_network']['processing']['avg_time']:.6f}s avg")
            print()
            
        except Exception as e:
            logger.error(f"Neural network benchmark failed: {e}")
            self.results['neural_network'] = {'error': str(e)}
    
    def benchmark_security_system(self):
        """Benchmark security system performance"""
        logger.info("🔐 Benchmarking Security System...")
        
        try:
            from core.security import SecurityManager
            
            security = SecurityManager()
            
            # Test input validation speed
            validation_times = []
            test_inputs = [
                {'safe': 'data'},
                {'user': 'test@example.com'},
                {'query': 'SELECT * FROM users'},
                {'script': '<script>alert("test")</script>'},
                {'path': '../../../etc/passwd'}
            ]
            
            for i in range(200):
                test_input = test_inputs[i % len(test_inputs)]
                start = time.time()
                security.validate_input(test_input)
                validation_times.append(time.time() - start)
            
            # Test threat assessment speed
            threat_times = []
            for i in range(100):
                test_input = test_inputs[i % len(test_inputs)]
                start = time.time()
                security.assess_threat_level(test_input)
                threat_times.append(time.time() - start)
            
            # Test encryption/decryption speed
            encryption_times = []
            decryption_times = []
            test_data = "This is test data for encryption benchmarking"
            
            for i in range(100):
                start = time.time()
                encrypted = security.encrypt_data(test_data)
                encryption_times.append(time.time() - start)
                
                start = time.time()
                decrypted = security.decrypt_data(encrypted)
                decryption_times.append(time.time() - start)
            
            self.results['security'] = {
                'validation': {
                    'avg_time': statistics.mean(validation_times),
                    'validations_per_second': 1.0 / statistics.mean(validation_times)
                },
                'threat_assessment': {
                    'avg_time': statistics.mean(threat_times),
                    'assessments_per_second': 1.0 / statistics.mean(threat_times)
                },
                'encryption': {
                    'avg_time': statistics.mean(encryption_times),
                    'encryptions_per_second': 1.0 / statistics.mean(encryption_times)
                },
                'decryption': {
                    'avg_time': statistics.mean(decryption_times),
                    'decryptions_per_second': 1.0 / statistics.mean(decryption_times)
                }
            }
            
            print(f"🔐 Security System Benchmark Results:")
            print(f"   • Input Validation: {self.results['security']['validation']['validations_per_second']:.1f} validations/sec")
            print(f"   • Threat Assessment: {self.results['security']['threat_assessment']['assessments_per_second']:.1f} assessments/sec")
            print(f"   • Encryption: {self.results['security']['encryption']['encryptions_per_second']:.1f} encryptions/sec")
            print(f"   • Decryption: {self.results['security']['decryption']['decryptions_per_second']:.1f} decryptions/sec")
            print()
            
        except Exception as e:
            logger.error(f"Security benchmark failed: {e}")
            self.results['security'] = {'error': str(e)}
    
    def benchmark_concurrent_operations(self):
        """Benchmark concurrent operations"""
        logger.info("⚡ Benchmarking Concurrent Operations...")
        
        try:
            from core.agi_system import IntegratedAGISystem, create_agi_task, TaskPriority
            
            agi_system = IntegratedAGISystem()
            
            def submit_tasks(count):
                """Submit multiple tasks"""
                times = []
                for i in range(count):
                    task = create_agi_task(f'concurrent_task_{i}', TaskPriority.MEDIUM, {'test': i})
                    start = time.time()
                    agi_system.submit_task(task)
                    times.append(time.time() - start)
                return times
            
            # Test concurrent task submission
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(submit_tasks, 10) for _ in range(10)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            total_time = time.time() - start_time
            
            # Flatten results
            all_times = []
            for result in results:
                all_times.extend(result)
            
            total_tasks = len(all_times)
            
            self.results['concurrent'] = {
                'total_tasks': total_tasks,
                'total_time': total_time,
                'tasks_per_second': total_tasks / total_time,
                'avg_task_time': statistics.mean(all_times),
                'max_task_time': max(all_times),
                'min_task_time': min(all_times)
            }
            
            print(f"⚡ Concurrent Operations Benchmark Results:")
            print(f"   • Total Tasks: {total_tasks}")
            print(f"   • Total Time: {total_time:.3f}s")
            print(f"   • Throughput: {self.results['concurrent']['tasks_per_second']:.1f} tasks/sec")
            print(f"   • Avg Task Time: {self.results['concurrent']['avg_task_time']:.3f}s")
            print()
            
        except Exception as e:
            logger.error(f"Concurrent operations benchmark failed: {e}")
            self.results['concurrent'] = {'error': str(e)}
    
    def stress_test_system(self):
        """Perform system stress test"""
        logger.info("🔥 Performing System Stress Test...")
        
        try:
            from core.agi_system import IntegratedAGISystem, create_agi_task, TaskPriority
            
            agi_system = IntegratedAGISystem()
            
            # Monitor system resources during stress test
            initial_cpu = psutil.cpu_percent()
            initial_memory = psutil.virtual_memory().percent
            
            start_time = time.time()
            
            # Create high load
            def stress_worker():
                for i in range(50):
                    task = create_agi_task(f'stress_task_{i}', TaskPriority.HIGH, {'stress': True})
                    agi_system.submit_task(task)
                    time.sleep(0.01)  # Small delay
            
            # Run stress test with multiple threads
            stress_threads = []
            for i in range(5):
                thread = threading.Thread(target=stress_worker)
                thread.start()
                stress_threads.append(thread)
            
            # Monitor resources during stress
            max_cpu = 0
            max_memory = 0
            
            for thread in stress_threads:
                thread.join()
                max_cpu = max(max_cpu, psutil.cpu_percent())
                max_memory = max(max_memory, psutil.virtual_memory().percent)
            
            stress_time = time.time() - start_time
            
            final_cpu = psutil.cpu_percent()
            final_memory = psutil.virtual_memory().percent
            
            self.results['stress_test'] = {
                'duration': stress_time,
                'initial_cpu': initial_cpu,
                'final_cpu': final_cpu,
                'max_cpu': max_cpu,
                'initial_memory': initial_memory,
                'final_memory': final_memory,
                'max_memory': max_memory,
                'cpu_increase': max_cpu - initial_cpu,
                'memory_increase': max_memory - initial_memory
            }
            
            print(f"🔥 Stress Test Results:")
            print(f"   • Duration: {stress_time:.3f}s")
            print(f"   • CPU Usage: {initial_cpu:.1f}% → {max_cpu:.1f}% (max)")
            print(f"   • Memory Usage: {initial_memory:.1f}% → {max_memory:.1f}% (max)")
            print(f"   • CPU Increase: {self.results['stress_test']['cpu_increase']:.1f}%")
            print(f"   • Memory Increase: {self.results['stress_test']['memory_increase']:.1f}%")
            print()
            
        except Exception as e:
            logger.error(f"Stress test failed: {e}")
            self.results['stress_test'] = {'error': str(e)}
    
    def calculate_performance_score(self):
        """Calculate overall performance score"""
        logger.info("📊 Calculating Performance Score...")
        
        score = 100.0
        
        # AGI System scoring
        if 'agi_system' in self.results and 'error' not in self.results['agi_system']:
            agi_score = 0
            if self.results['agi_system']['initialization_time'] < 1.0:
                agi_score += 25
            if self.results['agi_system']['task_creation']['tasks_per_second'] > 1000:
                agi_score += 25
            if self.results['agi_system']['task_submission']['tasks_per_second'] > 100:
                agi_score += 25
            if self.results['agi_system']['status_retrieval']['avg_time'] < 0.01:
                agi_score += 25
            
            score *= (agi_score / 100)
        
        # Neural Network scoring
        if 'neural_network' in self.results and 'error' not in self.results['neural_network']:
            neural_score = 0
            if self.results['neural_network']['processing']['ops_per_second'] > 1000:
                neural_score += 50
            if self.results['neural_network']['processing']['avg_time'] < 0.001:
                neural_score += 50
            
            score *= (neural_score / 100)
        
        # Security scoring
        if 'security' in self.results and 'error' not in self.results['security']:
            security_score = 0
            if self.results['security']['validation']['validations_per_second'] > 10000:
                security_score += 25
            if self.results['security']['threat_assessment']['assessments_per_second'] > 5000:
                security_score += 25
            if self.results['security']['encryption']['encryptions_per_second'] > 1000:
                security_score += 25
            if self.results['security']['decryption']['decryptions_per_second'] > 1000:
                security_score += 25
            
            score *= (security_score / 100)
        
        # Concurrent operations scoring
        if 'concurrent' in self.results and 'error' not in self.results['concurrent']:
            concurrent_score = 0
            if self.results['concurrent']['tasks_per_second'] > 50:
                concurrent_score += 100
            elif self.results['concurrent']['tasks_per_second'] > 20:
                concurrent_score += 75
            elif self.results['concurrent']['tasks_per_second'] > 10:
                concurrent_score += 50
            else:
                concurrent_score += 25
            
            score *= (concurrent_score / 100)
        
        self.results['performance_score'] = {
            'overall_score': score,
            'grade': 'A' if score >= 80 else 'B' if score >= 60 else 'C' if score >= 40 else 'D',
            'category': 'Excellent' if score >= 90 else 'Good' if score >= 70 else 'Fair' if score >= 50 else 'Poor'
        }
        
        print(f"📊 Performance Score: {score:.1f}/100 (Grade: {self.results['performance_score']['grade']})")
        print(f"   Category: {self.results['performance_score']['category']}")
        print()
    
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        logger.info("📋 Generating Benchmark Report...")
        
        total_time = time.time() - self.start_time
        
        report = {
            'benchmark_info': {
                'timestamp': datetime.now().isoformat(),
                'total_duration': total_time,
                'system_specs': self.baseline_metrics
            },
            'results': self.results,
            'summary': {
                'total_tests': len([k for k in self.results.keys() if k != 'performance_score']),
                'passed_tests': len([k for k, v in self.results.items() if k != 'performance_score' and 'error' not in v]),
                'failed_tests': len([k for k, v in self.results.items() if k != 'performance_score' and 'error' in v]),
                'performance_score': self.results.get('performance_score', {}).get('overall_score', 0)
            }
        }
        
        # Save report
        report_file = '/opt/sutazaiapp/benchmark_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Benchmark Report saved to: {report_file}")
        print(f"⏱️  Total benchmark time: {total_time:.3f}s")
        print()
        
        return report
    
    def run_benchmark(self):
        """Run complete benchmark suite"""
        self.start_time = time.time()
        
        self.print_benchmark_header()
        self.collect_baseline_metrics()
        
        # Run all benchmarks
        self.benchmark_agi_system()
        self.benchmark_neural_network()
        self.benchmark_security_system()
        self.benchmark_concurrent_operations()
        self.stress_test_system()
        
        # Calculate performance score
        self.calculate_performance_score()
        
        # Generate report
        report = self.generate_report()
        
        # Print summary
        print("="*80)
        print("🎯 BENCHMARK COMPLETE")
        print("="*80)
        print(f"Performance Score: {report['summary']['performance_score']:.1f}/100")
        print(f"Tests Passed: {report['summary']['passed_tests']}/{report['summary']['total_tests']}")
        print(f"Total Time: {report['benchmark_info']['total_duration']:.3f}s")
        print("="*80)
        
        return report

def main():
    """Main benchmark entry point"""
    benchmark = SystemBenchmark()
    benchmark.run_benchmark()

if __name__ == "__main__":
    main()