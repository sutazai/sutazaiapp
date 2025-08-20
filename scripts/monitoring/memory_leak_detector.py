#!/usr/bin/env python3
"""
Memory Leak Detection and Performance Analysis Tool
Performs comprehensive memory leak detection and resource efficiency analysis
"""

import json
import subprocess
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import sys
import os

class MemoryLeakDetector:
    """Comprehensive memory leak detection and performance analysis"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.container_history: Dict[str, List[Dict]] = {}
        self.memory_trends: Dict[str, Dict] = {}
        
    def run_command(self, command: str) -> str:
        """Execute shell command and return output"""
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return result.stdout.strip()
        except Exception as e:
            print(f"Error executing command: {e}")
            return ""
    
    def get_container_stats(self) -> List[Dict[str, Any]]:
        """Get current container statistics"""
        cmd = "docker stats --no-stream --format json"
        output = self.run_command(cmd)
        
        containers = []
        for line in output.split('\n'):
            if line:
                try:
                    container = json.loads(line)
                    # Parse memory usage
                    mem_usage = container.get('MemUsage', '0MiB / 0MiB')
                    current, limit = mem_usage.split(' / ')
                    
                    # Convert to MB
                    current_mb = self.convert_to_mb(current)
                    limit_mb = self.convert_to_mb(limit) if limit != '0B' else 0
                    
                    containers.append({
                        'name': container.get('Name', ''),
                        'container_id': container.get('Container', ''),
                        'cpu_percent': float(container.get('CPUPerc', '0%').rstrip('%')),
                        'memory_current_mb': current_mb,
                        'memory_limit_mb': limit_mb,
                        'memory_percent': float(container.get('MemPerc', '0%').rstrip('%')),
                        'net_io': container.get('NetIO', ''),
                        'block_io': container.get('BlockIO', ''),
                        'pids': int(container.get('PIDs', '0'))
                    })
                except (json.JSONDecodeError, ValueError) as e:
                    continue
                    
        return containers
    
    def convert_to_mb(self, size_str: str) -> float:
        """Convert size string to MB"""
        size_str = size_str.strip()
        if 'GiB' in size_str:
            return float(size_str.replace('GiB', '')) * 1024
        elif 'MiB' in size_str:
            return float(size_str.replace('MiB', ''))
        elif 'KiB' in size_str:
            return float(size_str.replace('KiB', '')) / 1024
        elif 'B' in size_str:
            return float(size_str.replace('B', '')) / (1024 * 1024)
        return 0
    
    def get_container_runtime_info(self, container_name: str) -> Dict[str, Any]:
        """Get detailed container runtime information"""
        cmd = f"docker inspect {container_name}"
        output = self.run_command(cmd)
        
        if output:
            try:
                info = json.loads(output)[0]
                state = info.get('State', {})
                host_config = info.get('HostConfig', {})
                
                # Calculate uptime
                started_at = state.get('StartedAt', '')
                if started_at:
                    start_time = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                    uptime_hours = (datetime.now(start_time.tzinfo) - start_time).total_seconds() / 3600
                else:
                    uptime_hours = 0
                
                return {
                    'status': state.get('Status', 'unknown'),
                    'running': state.get('Running', False),
                    'restart_count': info.get('RestartCount', 0),
                    'uptime_hours': uptime_hours,
                    'memory_limit': host_config.get('Memory', 0),
                    'memory_swap': host_config.get('MemorySwap', 0),
                    'cpu_shares': host_config.get('CpuShares', 0),
                    'oom_killed': state.get('OOMKilled', False)
                }
            except (json.JSONDecodeError, IndexError, KeyError):
                pass
        
        return {}
    
    def detect_memory_leaks(self, history_window: int = 5) -> Dict[str, Dict]:
        """Detect memory leaks based on historical data"""
        leaks = {}
        
        for container_name, history in self.container_history.items():
            if len(history) < history_window:
                continue
                
            # Calculate memory growth rate
            recent_history = history[-history_window:]
            first_mem = recent_history[0]['memory_current_mb']
            last_mem = recent_history[-1]['memory_current_mb']
            
            if first_mem > 0:
                growth_rate = ((last_mem - first_mem) / first_mem) * 100
                avg_memory = sum(h['memory_current_mb'] for h in recent_history) / len(recent_history)
                
                # Detect leak patterns
                is_growing = all(recent_history[i]['memory_current_mb'] >= recent_history[i-1]['memory_current_mb'] 
                                for i in range(1, len(recent_history)))
                
                if growth_rate > 10 or is_growing:
                    runtime_info = self.get_container_runtime_info(container_name)
                    
                    leaks[container_name] = {
                        'growth_rate_percent': growth_rate,
                        'initial_memory_mb': first_mem,
                        'current_memory_mb': last_mem,
                        'average_memory_mb': avg_memory,
                        'is_monotonic_growth': is_growing,
                        'uptime_hours': runtime_info.get('uptime_hours', 0),
                        'restart_count': runtime_info.get('restart_count', 0),
                        'oom_killed': runtime_info.get('oom_killed', False),
                        'memory_per_hour': (last_mem - first_mem) / max(runtime_info.get('uptime_hours', 1), 0.1)
                    }
        
        return leaks
    
    def analyze_resource_efficiency(self, containers: List[Dict]) -> Dict[str, Any]:
        """Analyze resource efficiency and identify optimization opportunities"""
        efficiency_analysis = {
            'over_allocated': [],
            'under_utilized': [],
            'no_limits': [],
            'high_cpu': [],
            'zombie_containers': []
        }
        
        for container in containers:
            name = container['name']
            mem_current = container['memory_current_mb']
            mem_limit = container['memory_limit_mb']
            mem_percent = container['memory_percent']
            cpu_percent = container['cpu_percent']
            pids = container['pids']
            
            # Check for over-allocation (limit much higher than usage)
            if mem_limit > 0 and mem_current > 0:
                utilization = (mem_current / mem_limit) * 100
                if utilization < 10:  # Less than 10% utilization
                    efficiency_analysis['over_allocated'].append({
                        'name': name,
                        'current_mb': mem_current,
                        'limit_mb': mem_limit,
                        'utilization_percent': utilization,
                        'potential_savings_mb': mem_limit - (mem_current * 2)  # Allow 2x headroom
                    })
            
            # Check for containers without limits
            elif mem_limit == 0:
                efficiency_analysis['no_limits'].append({
                    'name': name,
                    'current_mb': mem_current,
                    'recommendation': f"Set limit to {int(mem_current * 2)}MB"
                })
            
            # Check for high CPU usage
            if cpu_percent > 80:
                efficiency_analysis['high_cpu'].append({
                    'name': name,
                    'cpu_percent': cpu_percent,
                    'memory_mb': mem_current
                })
            
            # Check for zombie containers (low activity)
            if cpu_percent < 0.1 and mem_current < 50 and pids < 5:
                efficiency_analysis['zombie_containers'].append({
                    'name': name,
                    'cpu_percent': cpu_percent,
                    'memory_mb': mem_current,
                    'pids': pids
                })
        
        return efficiency_analysis
    
    def monitor_memory_trends(self, duration_seconds: int = 60, interval_seconds: int = 10):
        """Monitor memory trends over time"""
        print(f"Starting memory trend monitoring for {duration_seconds} seconds...")
        print(f"Sampling every {interval_seconds} seconds\n")
        
        samples = duration_seconds // interval_seconds
        
        for i in range(samples):
            containers = self.get_container_stats()
            
            # Update history
            for container in containers:
                name = container['name']
                if name not in self.container_history:
                    self.container_history[name] = []
                
                self.container_history[name].append({
                    'timestamp': datetime.now().isoformat(),
                    'memory_current_mb': container['memory_current_mb'],
                    'memory_percent': container['memory_percent'],
                    'cpu_percent': container['cpu_percent']
                })
            
            # Detect leaks after collecting enough samples
            if i >= 4:  # Need at least 5 samples
                leaks = self.detect_memory_leaks()
                if leaks:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Potential Memory Leaks Detected:")
                    for container_name, leak_info in leaks.items():
                        print(f"  - {container_name}:")
                        print(f"    Growth Rate: {leak_info['growth_rate_percent']:.1f}%")
                        print(f"    Memory Growth: {leak_info['initial_memory_mb']:.1f}MB -> {leak_info['current_memory_mb']:.1f}MB")
                        print(f"    Growth/Hour: {leak_info['memory_per_hour']:.2f}MB/hr")
                        if leak_info['oom_killed']:
                            print(f"    WARNING: Previously OOM killed!")
            
            if i < samples - 1:
                time.sleep(interval_seconds)
        
        return self.generate_final_report(containers)
    
    def generate_final_report(self, current_containers: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive performance analysis report"""
        
        # Get system memory info
        system_memory = self.run_command("free -m | grep Mem | awk '{print $2,$3,$4,$6}'").split()
        if len(system_memory) >= 4:
            total_mb = int(system_memory[0])
            used_mb = int(system_memory[1])
            free_mb = int(system_memory[2])
            available_mb = int(system_memory[3])
        else:
            total_mb = used_mb = free_mb = available_mb = 0
        
        # Analyze efficiency
        efficiency = self.analyze_resource_efficiency(current_containers)
        
        # Detect memory leaks
        memory_leaks = self.detect_memory_leaks()
        
        # Calculate total container memory
        total_container_memory = sum(c['memory_current_mb'] for c in current_containers)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
            'system_memory': {
                'total_mb': total_mb,
                'used_mb': used_mb,
                'free_mb': free_mb,
                'available_mb': available_mb,
                'usage_percent': (used_mb / total_mb * 100) if total_mb > 0 else 0
            },
            'container_summary': {
                'total_containers': len(current_containers),
                'total_memory_mb': total_container_memory,
                'containers_with_leaks': len(memory_leaks),
                'over_allocated_containers': len(efficiency['over_allocated']),
                'containers_without_limits': len(efficiency['no_limits'])
            },
            'memory_leaks': memory_leaks,
            'efficiency_analysis': efficiency,
            'top_memory_consumers': sorted(
                [{'name': c['name'], 'memory_mb': c['memory_current_mb']} for c in current_containers],
                key=lambda x: x['memory_mb'],
                reverse=True
            )[:10],
            'optimization_potential': {
                'total_savings_mb': sum(c['potential_savings_mb'] for c in efficiency['over_allocated']),
                'recommendations': self.generate_recommendations(efficiency, memory_leaks)
            }
        }
        
        return report
    
    def generate_recommendations(self, efficiency: Dict, leaks: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Memory leak recommendations
        if leaks:
            for container_name, leak_info in leaks.items():
                if leak_info['memory_per_hour'] > 10:
                    recommendations.append(f"CRITICAL: Restart {container_name} - memory growing at {leak_info['memory_per_hour']:.1f}MB/hour")
                elif leak_info['is_monotonic_growth']:
                    recommendations.append(f"WARNING: Monitor {container_name} - consistent memory growth detected")
        
        # Over-allocation recommendations
        for container in efficiency['over_allocated']:
            if container['potential_savings_mb'] > 100:
                recommendations.append(
                    f"OPTIMIZE: Reduce {container['name']} memory limit from {container['limit_mb']:.0f}MB to {container['current_mb'] * 2:.0f}MB"
                )
        
        # No limits recommendations
        for container in efficiency['no_limits']:
            recommendations.append(f"SECURITY: Add memory limit to {container['name']}: {container['recommendation']}")
        
        # Zombie container recommendations
        for container in efficiency['zombie_containers']:
            recommendations.append(f"CLEANUP: Consider removing inactive container {container['name']}")
        
        return recommendations

def main():
    """Main execution function"""
    print("=" * 80)
    print("MEMORY LEAK DETECTION AND PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    detector = MemoryLeakDetector()
    
    # Quick analysis (1 minute monitoring)
    report = detector.monitor_memory_trends(duration_seconds=60, interval_seconds=10)
    
    # Print final report
    print("\n" + "=" * 80)
    print("FINAL PERFORMANCE ANALYSIS REPORT")
    print("=" * 80)
    
    print(f"\nSystem Memory Status:")
    print(f"  Total: {report['system_memory']['total_mb']}MB")
    print(f"  Used: {report['system_memory']['used_mb']}MB ({report['system_memory']['usage_percent']:.1f}%)")
    print(f"  Available: {report['system_memory']['available_mb']}MB")
    
    print(f"\nContainer Summary:")
    print(f"  Total Containers: {report['container_summary']['total_containers']}")
    print(f"  Total Container Memory: {report['container_summary']['total_memory_mb']:.1f}MB")
    print(f"  Containers with Leaks: {report['container_summary']['containers_with_leaks']}")
    print(f"  Over-allocated Containers: {report['container_summary']['over_allocated_containers']}")
    
    if report['memory_leaks']:
        print(f"\nMemory Leak Detection Results:")
        for container, leak in report['memory_leaks'].items():
            print(f"  {container}:")
            print(f"    - Growth Rate: {leak['growth_rate_percent']:.1f}%")
            print(f"    - Memory/Hour: {leak['memory_per_hour']:.2f}MB/hr")
            print(f"    - Current: {leak['current_memory_mb']:.1f}MB")
    
    print(f"\nTop 5 Memory Consumers:")
    for i, consumer in enumerate(report['top_memory_consumers'][:5], 1):
        print(f"  {i}. {consumer['name']}: {consumer['memory_mb']:.1f}MB")
    
    print(f"\nOptimization Potential:")
    print(f"  Total Potential Savings: {report['optimization_potential']['total_savings_mb']:.0f}MB")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(report['optimization_potential']['recommendations'][:10], 1):
        print(f"  {i}. {rec}")
    
    # Save detailed report to file
    report_file = f"/opt/sutazaiapp/performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    return 0 if not report['memory_leaks'] else 1

if __name__ == "__main__":
    sys.exit(main())