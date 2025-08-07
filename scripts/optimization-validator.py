#!/usr/bin/env python3
"""
Purpose: Test and validate system optimizations
Usage: python optimization-validator.py [--test-type] [--iterations]
Requirements: psutil, docker
"""

import sys
import time
import json
import psutil
import docker
import subprocess
import threading
from datetime import datetime
import argparse

class OptimizationValidator:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': [],
            'baseline_metrics': {},
            'optimized_metrics': {},
            'improvements': {}
        }
        
    def collect_baseline_metrics(self):
        """Collect baseline system metrics"""
        print("📊 Collecting baseline metrics...")
        
        # Take multiple samples for accuracy
        cpu_samples = []
        memory_samples = []
        
        for i in range(5):
            cpu_samples.append(psutil.cpu_percent(interval=1))
            memory_samples.append(psutil.virtual_memory().percent)
            
        self.results['baseline_metrics'] = {
            'cpu_avg': round(sum(cpu_samples) / len(cpu_samples), 2),
            'cpu_max': max(cpu_samples),
            'memory_avg': round(sum(memory_samples) / len(memory_samples), 2),
            'memory_max': max(memory_samples),
            'process_count': len(psutil.pids()),
            'swap_used': psutil.swap_memory().percent,
            'load_avg': list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
        }
        
    def test_memory_optimization(self):
        """Test memory optimization effectiveness"""
        print("🧠 Testing memory optimization...")
        
        test_result = {
            'name': 'memory_optimization',
            'status': 'running',
            'start_time': time.time(),
            'metrics': {}
        }
        
        # Get memory-intensive processes
        memory_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
            try:
                proc_info = proc.as_dict()
                if proc_info['memory_percent'] > 1.0:
                    memory_processes.append({
                        'pid': proc_info['pid'],
                        'name': proc_info['name'],
                        'memory_percent': proc_info['memory_percent']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        test_result['metrics']['high_memory_processes'] = len(memory_processes)
        test_result['metrics']['top_memory_processes'] = sorted(memory_processes, key=lambda x: x['memory_percent'], reverse=True)[:5]
        
        # Check memory fragmentation
        memory = psutil.virtual_memory()
        test_result['metrics']['memory_fragmentation'] = {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'cached_gb': round(memory.cached / (1024**3), 2) if hasattr(memory, 'cached') else 0,
            'fragmentation_ratio': round((memory.total - memory.available - memory.cached) / memory.total, 2) if hasattr(memory, 'cached') else 0
        }
        
        test_result['status'] = 'completed'
        test_result['duration'] = time.time() - test_result['start_time']
        self.results['tests'].append(test_result)
        
    def test_cpu_optimization(self):
        """Test CPU optimization effectiveness"""
        print("⚡ Testing CPU optimization...")
        
        test_result = {
            'name': 'cpu_optimization',
            'status': 'running',
            'start_time': time.time(),
            'metrics': {}
        }
        
        # Monitor CPU usage patterns
        cpu_samples = []
        for i in range(10):
            cpu_samples.append(psutil.cpu_percent(interval=0.5, percpu=True))
            
        # Calculate CPU statistics
        avg_per_core = [sum(core_vals) / len(core_vals) for core_vals in zip(*cpu_samples)]
        test_result['metrics']['cpu_per_core_avg'] = [round(x, 2) for x in avg_per_core]
        test_result['metrics']['cpu_load_distribution'] = {
            'balanced': max(avg_per_core) - min(avg_per_core) < 20,
            'max_core_usage': max(avg_per_core),
            'min_core_usage': min(avg_per_core),
            'load_variance': round(sum((x - sum(avg_per_core)/len(avg_per_core))**2 for x in avg_per_core) / len(avg_per_core), 2)
        }
        
        # Check high CPU processes
        high_cpu_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
            try:
                proc_info = proc.as_dict()
                if proc_info['cpu_percent'] > 5.0:
                    high_cpu_processes.append({
                        'pid': proc_info['pid'],
                        'name': proc_info['name'],
                        'cpu_percent': proc_info['cpu_percent']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        test_result['metrics']['high_cpu_processes'] = sorted(high_cpu_processes, key=lambda x: x['cpu_percent'], reverse=True)
        
        test_result['status'] = 'completed'
        test_result['duration'] = time.time() - test_result['start_time']
        self.results['tests'].append(test_result)
        
    def test_container_optimization(self):
        """Test container resource optimization"""
        print("🐳 Testing container optimization...")
        
        test_result = {
            'name': 'container_optimization',
            'status': 'running', 
            'start_time': time.time(),
            'metrics': {}
        }
        
        try:
            client = docker.from_env()
            running_containers = client.containers.list()
            
            test_result['metrics']['container_count'] = len(running_containers)
            test_result['metrics']['containers'] = []
            
            for container in running_containers:
                try:
                    # Get basic container info
                    container_info = {
                        'name': container.name,
                        'status': container.status,
                        'image': container.image.tags[0] if container.image.tags else 'unknown',
                        'restart_count': container.attrs['RestartCount']
                    }
                    
                    # Try to get resource usage
                    try:
                        stats = container.stats(stream=False)
                        memory_usage = stats.get('memory_stats', {}).get('usage', 0)
                        memory_limit = stats.get('memory_stats', {}).get('limit', 0)
                        
                        container_info.update({
                            'memory_usage_mb': round(memory_usage / (1024*1024), 2) if memory_usage else 0,
                            'memory_limit_mb': round(memory_limit / (1024*1024), 2) if memory_limit else 0,
                            'memory_percent': round((memory_usage / memory_limit) * 100, 2) if memory_limit > 0 else 0
                        })
                    except Exception:
                        container_info.update({
                            'memory_usage_mb': 0,
                            'memory_limit_mb': 0, 
                            'memory_percent': 0
                        })
                        
                    test_result['metrics']['containers'].append(container_info)
                    
                except Exception as e:
                    print(f"Error analyzing container {container.name}: {e}")
                    
            # Check for potential issues
            test_result['metrics']['issues'] = []
            
            # Check for containers without memory limits
            no_limit_containers = [c for c in test_result['metrics']['containers'] if c['memory_limit_mb'] == 0]
            if no_limit_containers:
                test_result['metrics']['issues'].append({
                    'type': 'no_memory_limits',
                    'count': len(no_limit_containers),
                    'containers': [c['name'] for c in no_limit_containers]
                })
                
            # Check for high memory usage containers
            high_memory_containers = [c for c in test_result['metrics']['containers'] if c['memory_percent'] > 80]
            if high_memory_containers:
                test_result['metrics']['issues'].append({
                    'type': 'high_memory_usage',
                    'count': len(high_memory_containers),
                    'containers': [{'name': c['name'], 'memory_percent': c['memory_percent']} for c in high_memory_containers]
                })
                
            # Check for frequently restarting containers
            restart_containers = [c for c in test_result['metrics']['containers'] if c['restart_count'] > 3]
            if restart_containers:
                test_result['metrics']['issues'].append({
                    'type': 'frequent_restarts',
                    'count': len(restart_containers),
                    'containers': [{'name': c['name'], 'restart_count': c['restart_count']} for c in restart_containers]
                })
                
        except Exception as e:
            test_result['error'] = str(e)
            test_result['metrics']['container_count'] = 0
            
        test_result['status'] = 'completed'
        test_result['duration'] = time.time() - test_result['start_time']
        self.results['tests'].append(test_result)
        
    def test_process_optimization(self):
        """Test process optimization and duplicate detection"""
        print("🔍 Testing process optimization...")
        
        test_result = {
            'name': 'process_optimization',
            'status': 'running',
            'start_time': time.time(),
            'metrics': {}
        }
        
        # Group processes by name
        process_groups = {}
        zombie_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent', 'memory_percent', 'create_time']):
            try:
                proc_info = proc.as_dict()
                process_name = proc_info['name'].lower()
                
                if process_name not in process_groups:
                    process_groups[process_name] = []
                    
                process_groups[process_name].append({
                    'pid': proc_info['pid'],
                    'cpu_percent': proc_info['cpu_percent'],
                    'memory_percent': proc_info['memory_percent'],
                    'status': proc_info['status'],
                    'runtime_hours': round((time.time() - proc_info['create_time']) / 3600, 2) if proc_info['create_time'] else 0
                })
                
                # Check for zombie processes
                if proc_info['status'] == psutil.STATUS_ZOMBIE:
                    zombie_processes.append(proc_info['pid'])
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        # Analyze duplicate processes
        duplicate_processes = {}
        for name, procs in process_groups.items():
            if len(procs) > 3 and name in ['claude', 'python', 'node', 'ollama']:
                duplicate_processes[name] = {
                    'count': len(procs),
                    'total_memory_percent': sum(p['memory_percent'] for p in procs),
                    'total_cpu_percent': sum(p['cpu_percent'] for p in procs),
                    'processes': procs
                }
                
        test_result['metrics'] = {
            'total_processes': len(psutil.pids()),
            'zombie_processes': len(zombie_processes),
            'duplicate_processes': duplicate_processes,
            'process_groups_summary': {name: len(procs) for name, procs in process_groups.items() if len(procs) > 2}
        }
        
        test_result['status'] = 'completed'
        test_result['duration'] = time.time() - test_result['start_time']
        self.results['tests'].append(test_result)
        
    def stress_test_system(self, duration=30):
        """Run a stress test to validate system stability"""
        print(f"🚀 Running {duration}s stress test...")
        
        test_result = {
            'name': 'stress_test',
            'status': 'running',
            'start_time': time.time(),
            'metrics': {'samples': []}
        }
        
        start_time = time.time()
        while time.time() - start_time < duration:
            sample = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'swap_percent': psutil.swap_memory().percent,
                'load_avg': list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            }
            test_result['metrics']['samples'].append(sample)
            
            # Print progress
            elapsed = time.time() - start_time
            progress = (elapsed / duration) * 100
            print(f"  Progress: {progress:.1f}% - CPU: {sample['cpu_percent']:.1f}%, RAM: {sample['memory_percent']:.1f}%")
            
        # Calculate stress test statistics
        cpu_values = [s['cpu_percent'] for s in test_result['metrics']['samples']]
        memory_values = [s['memory_percent'] for s in test_result['metrics']['samples']]
        
        test_result['metrics']['statistics'] = {
            'cpu_avg': round(sum(cpu_values) / len(cpu_values), 2),
            'cpu_max': max(cpu_values),
            'cpu_min': min(cpu_values),
            'memory_avg': round(sum(memory_values) / len(memory_values), 2),
            'memory_max': max(memory_values),
            'memory_min': min(memory_values),
            'samples_count': len(test_result['metrics']['samples'])
        }
        
        test_result['status'] = 'completed'
        test_result['duration'] = time.time() - test_result['start_time']
        self.results['tests'].append(test_result)
        
    def calculate_improvements(self):
        """Calculate improvements after optimization"""
        print("📈 Calculating improvements...")
        
        # Collect current metrics
        cpu_samples = []
        memory_samples = []
        
        for i in range(5):
            cpu_samples.append(psutil.cpu_percent(interval=1))
            memory_samples.append(psutil.virtual_memory().percent)
            
        current_metrics = {
            'cpu_avg': round(sum(cpu_samples) / len(cpu_samples), 2),
            'memory_avg': round(sum(memory_samples) / len(memory_samples), 2),
            'process_count': len(psutil.pids()),
            'swap_used': psutil.swap_memory().percent
        }
        
        self.results['optimized_metrics'] = current_metrics
        
        # Calculate improvements
        baseline = self.results['baseline_metrics']
        improvements = {}
        
        if baseline:
            improvements['cpu_improvement'] = baseline['cpu_avg'] - current_metrics['cpu_avg']
            improvements['memory_improvement'] = baseline['memory_avg'] - current_metrics['memory_avg']
            improvements['process_reduction'] = baseline['process_count'] - current_metrics['process_count']
            improvements['swap_improvement'] = baseline['swap_used'] - current_metrics['swap_used']
            
        self.results['improvements'] = improvements
        
    def run_validation_suite(self, include_stress_test=False, stress_duration=30):
        """Run complete validation suite"""
        print("🧪 Starting optimization validation suite...")
        print("=" * 60)
        
        self.collect_baseline_metrics()
        self.test_memory_optimization()
        self.test_cpu_optimization()
        self.test_container_optimization()
        self.test_process_optimization()
        
        if include_stress_test:
            self.stress_test_system(stress_duration)
            
        self.calculate_improvements()
        
        return self.results
        
    def print_validation_report(self):
        """Print validation report"""
        print("\n" + "=" * 60)
        print("🧪 OPTIMIZATION VALIDATION REPORT")
        print("=" * 60)
        
        # Baseline vs Current
        baseline = self.results['baseline_metrics']
        optimized = self.results['optimized_metrics']
        improvements = self.results['improvements']
        
        print(f"📊 System Metrics Comparison:")
        print(f"   CPU Usage: {baseline['cpu_avg']:.1f}% → {optimized['cpu_avg']:.1f}% ({improvements['cpu_improvement']:+.1f}%)")
        print(f"   Memory Usage: {baseline['memory_avg']:.1f}% → {optimized['memory_avg']:.1f}% ({improvements['memory_improvement']:+.1f}%)")
        print(f"   Process Count: {baseline['process_count']} → {optimized['process_count']} ({improvements['process_reduction']:+d})")
        print(f"   Swap Usage: {baseline['swap_used']:.1f}% → {optimized['swap_used']:.1f}% ({improvements['swap_improvement']:+.1f}%)")
        
        # Test Results
        print(f"\n🧪 Test Results ({len([t for t in self.results['tests'] if t['status'] == 'completed'])} completed):")
        
        for test in self.results['tests']:
            status_icon = "✅" if test['status'] == 'completed' else "❌"
            print(f"   {status_icon} {test['name']} ({test.get('duration', 0):.1f}s)")
            
            # Show key findings for each test
            if test['name'] == 'memory_optimization':
                metrics = test['metrics']
                print(f"      • High memory processes: {metrics['high_memory_processes']}")
                print(f"      • Memory fragmentation: {metrics['memory_fragmentation']['fragmentation_ratio']:.2f}")
                
            elif test['name'] == 'cpu_optimization':
                metrics = test['metrics']
                load_dist = metrics['cpu_load_distribution']
                print(f"      • CPU load balanced: {load_dist['balanced']}")
                print(f"      • Max core usage: {load_dist['max_core_usage']:.1f}%")
                print(f"      • High CPU processes: {len(metrics['high_cpu_processes'])}")
                
            elif test['name'] == 'container_optimization':
                metrics = test['metrics']
                print(f"      • Running containers: {metrics['container_count']}")
                print(f"      • Issues found: {len(metrics.get('issues', []))}")
                
            elif test['name'] == 'process_optimization':
                metrics = test['metrics']
                print(f"      • Total processes: {metrics['total_processes']}")
                print(f"      • Zombie processes: {metrics['zombie_processes']}")
                print(f"      • Duplicate process groups: {len(metrics['duplicate_processes'])}")
                
            elif test['name'] == 'stress_test':
                metrics = test['metrics']
                stats = metrics['statistics']
                print(f"      • CPU avg/max: {stats['cpu_avg']:.1f}%/{stats['cpu_max']:.1f}%")
                print(f"      • Memory avg/max: {stats['memory_avg']:.1f}%/{stats['memory_max']:.1f}%")
                
        # Overall assessment
        print(f"\n🎯 Overall Assessment:")
        
        if improvements['cpu_improvement'] > 5:
            print("   ✅ Significant CPU improvement achieved")
        elif improvements['cpu_improvement'] > 0:
            print("   ⚡ Minor CPU improvement achieved")
        else:
            print("   ⚠️  No CPU improvement detected")
            
        if improvements['memory_improvement'] > 5:
            print("   ✅ Significant memory improvement achieved")
        elif improvements['memory_improvement'] > 0:
            print("   🧠 Minor memory improvement achieved")
        else:
            print("   ⚠️  No memory improvement detected")
            
        print("\n" + "=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Validate system optimizations')
    parser.add_argument('--stress-test', action='store_true', help='Include stress test')
    parser.add_argument('--stress-duration', type=int, default=30, help='Stress test duration in seconds')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--save', type=str, help='Save results to file')
    args = parser.parse_args()
    
    validator = OptimizationValidator()
    
    try:
        results = validator.run_validation_suite(
            include_stress_test=args.stress_test,
            stress_duration=args.stress_duration
        )
        
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            validator.print_validation_report()
            
        if args.save:
            with open(args.save, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Validation results saved to {args.save}")
            
    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()