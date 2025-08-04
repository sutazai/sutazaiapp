#!/usr/bin/env python3
"""
Purpose: Comprehensive system resource analysis and optimization validation
Usage: python system-resource-analyzer.py [--detailed] [--optimize]
Requirements: psutil, docker (python package)
"""

import sys
import time
import json
import psutil
import docker
import subprocess
from datetime import datetime
from collections import defaultdict
import argparse

class SystemResourceAnalyzer:
    def __init__(self):
        self.analysis_data = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {},
            'resource_usage': {},
            'processes': [],
            'containers': [],
            'issues': [],
            'recommendations': []
        }
        
    def analyze_system_info(self):
        """Collect basic system information"""
        print("📊 Collecting system information...")
        
        # Basic system info
        self.analysis_data['system_info'] = {
            'cpu_count': psutil.cpu_count(logical=True),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'swap_total_gb': round(psutil.swap_memory().total / (1024**3), 2),
            'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            'load_average': list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
        }
        
    def analyze_resource_usage(self):
        """Analyze current resource usage"""
        print("🔍 Analyzing resource usage...")
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        cpu_avg = sum(cpu_percent) / len(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        self.analysis_data['resource_usage'] = {
            'cpu': {
                'average_percent': round(cpu_avg, 2),
                'per_core_percent': [round(x, 2) for x in cpu_percent],
                'load_1min': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else None
            },
            'memory': {
                'total_gb': round(memory.total / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'percent_used': memory.percent,
                'cached_gb': round(memory.cached / (1024**3), 2) if hasattr(memory, 'cached') else None,
                'buffers_gb': round(memory.buffers / (1024**3), 2) if hasattr(memory, 'buffers') else None
            },
            'swap': {
                'total_gb': round(swap.total / (1024**3), 2),
                'used_gb': round(swap.used / (1024**3), 2),
                'percent_used': swap.percent
            },
            'disk': {
                'total_gb': round(disk.total / (1024**3), 2),
                'used_gb': round(disk.used / (1024**3), 2),
                'free_gb': round(disk.free / (1024**3), 2),
                'percent_used': round((disk.used / disk.total) * 100, 2)
            }
        }
        
    def analyze_processes(self):
        """Analyze all running processes"""
        print("🔍 Analyzing running processes...")
        
        processes = []
        process_groups = defaultdict(list)
        
        for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent', 'memory_info', 'create_time', 'status', 'cmdline']):
            try:
                proc_info = proc.as_dict()
                
                # Group similar processes
                process_name = proc_info['name'].lower()
                process_groups[process_name].append(proc_info)
                
                # Only include high resource processes or important ones
                if (proc_info['cpu_percent'] > 1.0 or 
                    proc_info['memory_percent'] > 1.0 or
                    process_name in ['claude', 'ollama', 'python', 'docker', 'postgres', 'redis', 'neo4j']):
                    
                    processes.append({
                        'pid': proc_info['pid'],
                        'name': proc_info['name'],
                        'username': proc_info['username'],
                        'cpu_percent': round(proc_info['cpu_percent'], 2),
                        'memory_percent': round(proc_info['memory_percent'], 2),
                        'memory_mb': round(proc_info['memory_info'].rss / (1024*1024), 2) if proc_info['memory_info'] else 0,
                        'status': proc_info['status'],
                        'runtime_hours': round((time.time() - proc_info['create_time']) / 3600, 2) if proc_info['create_time'] else 0,
                        'cmdline': ' '.join(proc_info['cmdline'][:3]) if proc_info['cmdline'] else ''
                    })
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        # Sort by resource usage
        processes.sort(key=lambda x: (x['cpu_percent'] + x['memory_percent']), reverse=True)
        self.analysis_data['processes'] = processes[:50]  # Top 50 processes
        
        # Analyze process groups for duplicates
        self._analyze_process_groups(process_groups)
        
    def _analyze_process_groups(self, process_groups):
        """Analyze process groups for potential issues"""
        print("🔍 Analyzing process groups for duplicates...")
        
        for process_name, procs in process_groups.items():
            if len(procs) > 5:  # More than 5 of the same process
                if process_name in ['claude', 'python', 'node']:
                    self.analysis_data['issues'].append({
                        'type': 'duplicate_processes',
                        'severity': 'high',
                        'description': f"Found {len(procs)} {process_name} processes - potential resource waste",
                        'processes': [{'pid': p['pid'], 'cpu': p['cpu_percent'], 'memory': p['memory_percent']} for p in procs[:10]]
                    })
                    
    def analyze_containers(self):
        """Analyze Docker containers"""
        print("🐳 Analyzing Docker containers...")
        
        try:
            client = docker.from_env()
            containers = []
            
            for container in client.containers.list(all=True):
                try:
                    stats = container.stats(stream=False)
                    
                    # Calculate CPU usage
                    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
                    system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
                    cpu_percent = (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100.0 if system_delta > 0 else 0
                    
                    # Calculate memory usage
                    memory_usage = stats['memory_stats'].get('usage', 0)
                    memory_limit = stats['memory_stats'].get('limit', 0)
                    memory_percent = (memory_usage / memory_limit) * 100.0 if memory_limit > 0 else 0
                    
                    container_info = {
                        'name': container.name,
                        'status': container.status,
                        'image': container.image.tags[0] if container.image.tags else 'unknown',
                        'cpu_percent': round(cpu_percent, 2),
                        'memory_mb': round(memory_usage / (1024*1024), 2),
                        'memory_percent': round(memory_percent, 2),
                        'memory_limit_mb': round(memory_limit / (1024*1024), 2) if memory_limit > 0 else 0,
                        'created': container.attrs['Created'][:19],
                        'restart_count': container.attrs['RestartCount']
                    }
                    
                    containers.append(container_info)
                    
                    # Check for potential issues
                    if memory_percent > 80:
                        self.analysis_data['issues'].append({
                            'type': 'high_memory_container',
                            'severity': 'high',
                            'description': f"Container {container.name} using {memory_percent:.1f}% of allocated memory",
                            'container': container.name
                        })
                        
                    if container.attrs['RestartCount'] > 5:
                        self.analysis_data['issues'].append({
                            'type': 'frequent_restarts',
                            'severity': 'medium', 
                            'description': f"Container {container.name} has restarted {container.attrs['RestartCount']} times",
                            'container': container.name
                        })
                        
                except Exception as e:
                    print(f"Error analyzing container {container.name}: {e}")
                    continue
                    
            self.analysis_data['containers'] = sorted(containers, key=lambda x: x['memory_percent'], reverse=True)
            
        except Exception as e:
            print(f"Error connecting to Docker: {e}")
            self.analysis_data['containers'] = []
            
    def detect_resource_issues(self):
        """Detect resource-related issues"""
        print("⚠️  Detecting resource issues...")
        
        usage = self.analysis_data['resource_usage']
        
        # High CPU usage
        if usage['cpu']['average_percent'] > 80:
            self.analysis_data['issues'].append({
                'type': 'high_cpu_usage',
                'severity': 'high',
                'description': f"Average CPU usage is {usage['cpu']['average_percent']}%",
                'value': usage['cpu']['average_percent']
            })
            
        # High memory usage
        if usage['memory']['percent_used'] > 80:
            self.analysis_data['issues'].append({
                'type': 'high_memory_usage', 
                'severity': 'high',
                'description': f"Memory usage is {usage['memory']['percent_used']}%",
                'value': usage['memory']['percent_used']
            })
            
        # Swap usage
        if usage['swap']['percent_used'] > 10:
            self.analysis_data['issues'].append({
                'type': 'swap_usage',
                'severity': 'medium',
                'description': f"Swap usage is {usage['swap']['percent_used']}%",
                'value': usage['swap']['percent_used']
            })
            
        # High disk usage
        if usage['disk']['percent_used'] > 85:
            self.analysis_data['issues'].append({
                'type': 'high_disk_usage',
                'severity': 'medium',
                'description': f"Disk usage is {usage['disk']['percent_used']}%",
                'value': usage['disk']['percent_used']
            })
            
    def generate_recommendations(self):
        """Generate optimization recommendations"""
        print("💡 Generating recommendations...")
        
        usage = self.analysis_data['resource_usage']
        
        # Memory recommendations
        if usage['memory']['percent_used'] > 70:
            self.analysis_data['recommendations'].append({
                'category': 'memory',
                'priority': 'high',
                'action': 'Consider reducing the number of Claude processes running simultaneously',
                'impact': 'Could free up 2-4GB of memory'
            })
            
        # CPU recommendations  
        if usage['cpu']['average_percent'] > 60:
            # Check for Ollama runner consuming high CPU
            ollama_processes = [p for p in self.analysis_data['processes'] if 'ollama runner' in p['cmdline']]
            if ollama_processes and ollama_processes[0]['cpu_percent'] > 100:
                self.analysis_data['recommendations'].append({
                    'category': 'cpu',
                    'priority': 'high',
                    'action': 'Ollama runner is consuming excessive CPU. Consider limiting model context size or switching to a smaller model',
                    'impact': 'Could reduce CPU usage by 50-80%'
                })
                
        # Container recommendations
        high_memory_containers = [c for c in self.analysis_data['containers'] if c['memory_percent'] > 60]
        if high_memory_containers:
            self.analysis_data['recommendations'].append({
                'category': 'containers',
                'priority': 'medium',
                'action': f"Containers with high memory usage: {', '.join([c['name'] for c in high_memory_containers[:3]])}. Consider setting memory limits.",
                'impact': 'Better resource isolation and prevention of memory leaks'
            })
            
        # Process consolidation
        claude_processes = [p for p in self.analysis_data['processes'] if p['name'] == 'claude']
        if len(claude_processes) > 3:
            self.analysis_data['recommendations'].append({
                'category': 'processes',
                'priority': 'medium',
                'action': f"Found {len(claude_processes)} Claude processes. Consider consolidating work into fewer instances.",
                'impact': f'Could free up {len(claude_processes) * 0.4:.1f}GB of memory'
            })
            
    def run_analysis(self, detailed=False):
        """Run complete system analysis"""
        print("🚀 Starting comprehensive system analysis...")
        print("=" * 60)
        
        self.analyze_system_info()
        self.analyze_resource_usage()
        self.analyze_processes()
        self.analyze_containers()
        self.detect_resource_issues()
        self.generate_recommendations()
        
        return self.analysis_data
        
    def print_summary(self):
        """Print analysis summary"""
        print("\n" + "=" * 60)
        print("📊 SYSTEM RESOURCE ANALYSIS SUMMARY")
        print("=" * 60)
        
        # System info
        sys_info = self.analysis_data['system_info']
        print(f"🖥️  System: {sys_info['cpu_count']} cores, {sys_info['memory_total_gb']}GB RAM, {sys_info['swap_total_gb']}GB swap")
        
        # Resource usage
        usage = self.analysis_data['resource_usage']
        print(f"📈 CPU: {usage['cpu']['average_percent']}% avg")
        print(f"🧠 Memory: {usage['memory']['used_gb']:.1f}GB/{usage['memory']['total_gb']}GB ({usage['memory']['percent_used']:.1f}%)")
        print(f"💾 Swap: {usage['swap']['used_gb']:.1f}GB/{usage['swap']['total_gb']}GB ({usage['swap']['percent_used']:.1f}%)")
        print(f"💿 Disk: {usage['disk']['used_gb']:.1f}GB/{usage['disk']['total_gb']}GB ({usage['disk']['percent_used']:.1f}%)")
        
        # Top processes
        print(f"\n🔝 Top resource consumers:")
        for i, proc in enumerate(self.analysis_data['processes'][:5]):
            print(f"  {i+1}. {proc['name']} (PID {proc['pid']}): {proc['cpu_percent']}% CPU, {proc['memory_percent']:.1f}% RAM")
            
        # Containers
        print(f"\n🐳 Container summary: {len([c for c in self.analysis_data['containers'] if c['status'] == 'running'])} running")
        for container in self.analysis_data['containers'][:3]:
            if container['status'] == 'running':
                print(f"  • {container['name']}: {container['cpu_percent']}% CPU, {container['memory_mb']:.0f}MB RAM")
        
        # Issues
        if self.analysis_data['issues']:
            print(f"\n⚠️  Issues found ({len(self.analysis_data['issues'])}):")
            for issue in self.analysis_data['issues']:
                severity_icon = "🔴" if issue['severity'] == 'high' else "🟡"
                print(f"  {severity_icon} {issue['description']}")
                
        # Recommendations
        if self.analysis_data['recommendations']:
            print(f"\n💡 Recommendations ({len(self.analysis_data['recommendations'])}):")
            for rec in self.analysis_data['recommendations']:
                priority_icon = "🔥" if rec['priority'] == 'high' else "📋"
                print(f"  {priority_icon} [{rec['category'].upper()}] {rec['action']}")
                print(f"     Impact: {rec['impact']}")
                
        print("\n" + "=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Comprehensive system resource analysis')
    parser.add_argument('--detailed', action='store_true', help='Show detailed analysis')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--save', type=str, help='Save results to file')
    args = parser.parse_args()
    
    analyzer = SystemResourceAnalyzer()
    
    try:
        analysis_data = analyzer.run_analysis(detailed=args.detailed)
        
        if args.json:
            print(json.dumps(analysis_data, indent=2))
        else:
            analyzer.print_summary()
            
        if args.save:
            with open(args.save, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            print(f"\n💾 Analysis saved to {args.save}")
            
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()