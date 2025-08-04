#!/usr/bin/env python3
"""
Purpose: Profile memory usage of hygiene monitoring system
Usage: python profile-hygiene-memory.py [--duration 300]
Requirements: tracemalloc, psutil, matplotlib
"""

import tracemalloc
import psutil
import time
import json
import asyncio
import subprocess
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class HygieneMemoryProfiler:
    def __init__(self, duration=300):
        self.duration = duration
        self.memory_samples = []
        self.snapshots = []
        self.process_data = {}
        
    def find_hygiene_processes(self):
        """Find all hygiene-related processes"""
        processes = {}
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
            try:
                info = proc.info
                cmdline = ' '.join(info['cmdline'] or [])
                
                if 'hygiene' in cmdline or ('monitoring' in cmdline and 'backend' in cmdline):
                    processes[info['pid']] = {
                        'name': info['name'],
                        'cmdline': cmdline[:100],
                        'process': proc
                    }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return processes
    
    def profile_process_memory(self, proc):
        """Profile memory usage of a specific process"""
        try:
            memory_info = proc.memory_info()
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': proc.memory_percent(),
                'num_threads': proc.num_threads(),
                'num_fds': proc.num_fds() if hasattr(proc, 'num_fds') else 0
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
    
    def profile_docker_containers(self):
        """Get memory stats from Docker containers"""
        try:
            result = subprocess.run([
                'docker', 'stats', '--no-stream', '--format', 'json'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                stats = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        data = json.loads(line)
                        if 'hygiene' in data.get('Name', ''):
                            stats.append({
                                'name': data['Name'],
                                'memory': data.get('MemUsage', ''),
                                'percent': data.get('MemPerc', '')
                            })
                return stats
        except Exception as e:
            print(f"Error getting Docker stats: {e}")
        return []
    
    async def monitor_memory(self):
        """Monitor memory usage over time"""
        print(f"Starting memory profiling for {self.duration} seconds...")
        
        # Start tracemalloc for Python memory tracking
        tracemalloc.start()
        
        start_time = time.time()
        sample_interval = 5  # seconds
        
        while time.time() - start_time < self.duration:
            timestamp = datetime.now()
            
            # Find and profile processes
            processes = self.find_hygiene_processes()
            process_stats = {}
            
            for pid, proc_info in processes.items():
                stats = self.profile_process_memory(proc_info['process'])
                if stats:
                    process_stats[pid] = {
                        **proc_info,
                        **stats
                    }
            
            # Get Docker stats
            docker_stats = self.profile_docker_containers()
            
            # Take a tracemalloc snapshot
            snapshot = tracemalloc.take_snapshot()
            
            # Store sample
            sample = {
                'timestamp': timestamp,
                'processes': process_stats,
                'docker': docker_stats,
                'system': {
                    'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                    'available_memory_gb': psutil.virtual_memory().available / (1024**3),
                    'percent_used': psutil.virtual_memory().percent
                }
            }
            
            self.memory_samples.append(sample)
            self.snapshots.append(snapshot)
            
            # Print current status
            total_hygiene_memory = sum(p['rss_mb'] for p in process_stats.values())
            print(f"\r[{timestamp.strftime('%H:%M:%S')}] "
                  f"Hygiene processes: {len(process_stats)}, "
                  f"Total memory: {total_hygiene_memory:.1f}MB", end='', flush=True)
            
            await asyncio.sleep(sample_interval)
        
        print("\nProfiling complete!")
        
    def analyze_results(self):
        """Analyze the collected memory data"""
        print("\n=== Memory Usage Analysis ===\n")
        
        if not self.memory_samples:
            print("No data collected!")
            return
        
        # Analyze process memory trends
        process_trends = defaultdict(list)
        
        for sample in self.memory_samples:
            for pid, stats in sample['processes'].items():
                process_trends[stats['cmdline']].append({
                    'timestamp': sample['timestamp'],
                    'memory_mb': stats['rss_mb'],
                    'threads': stats['num_threads'],
                    'fds': stats['num_fds']
                })
        
        # Find memory growth
        print("Memory Growth Analysis:")
        for cmdline, trends in process_trends.items():
            if len(trends) > 1:
                start_memory = trends[0]['memory_mb']
                end_memory = trends[-1]['memory_mb']
                growth = end_memory - start_memory
                growth_percent = (growth / start_memory) * 100 if start_memory > 0 else 0
                
                print(f"\nProcess: {cmdline[:60]}...")
                print(f"  Start memory: {start_memory:.1f}MB")
                print(f"  End memory: {end_memory:.1f}MB")
                print(f"  Growth: {growth:.1f}MB ({growth_percent:.1f}%)")
                
                # Check for file descriptor leaks
                start_fds = trends[0]['fds']
                end_fds = trends[-1]['fds']
                if end_fds > start_fds:
                    print(f"  ⚠️  File descriptors increased: {start_fds} -> {end_fds}")
                
                # Check for thread leaks
                start_threads = trends[0]['threads']
                end_threads = trends[-1]['threads']
                if end_threads > start_threads:
                    print(f"  ⚠️  Threads increased: {start_threads} -> {end_threads}")
        
        # Analyze tracemalloc snapshots
        if len(self.snapshots) > 1:
            print("\n\nPython Memory Allocation Top Differences:")
            
            # Compare first and last snapshots
            first_snapshot = self.snapshots[0]
            last_snapshot = self.snapshots[-1]
            
            top_stats = last_snapshot.compare_to(first_snapshot, 'lineno')
            
            for stat in top_stats[:10]:
                print(f"\n{stat}")
        
        # Generate report
        self.generate_report()
        
    def generate_report(self):
        """Generate a detailed memory profiling report"""
        report_path = Path("/opt/sutazaiapp/reports/hygiene_memory_profile.json")
        report_path.parent.mkdir(exist_ok=True)
        
        # Prepare data for JSON serialization
        report_data = {
            'profiling_duration': self.duration,
            'start_time': self.memory_samples[0]['timestamp'].isoformat() if self.memory_samples else None,
            'end_time': self.memory_samples[-1]['timestamp'].isoformat() if self.memory_samples else None,
            'samples': []
        }
        
        for sample in self.memory_samples:
            sample_data = {
                'timestamp': sample['timestamp'].isoformat(),
                'processes': sample['processes'],
                'docker': sample['docker'],
                'system': sample['system']
            }
            report_data['samples'].append(sample_data)
        
        # Calculate summary statistics
        if self.memory_samples:
            all_memory_values = []
            for sample in self.memory_samples:
                total_memory = sum(p['rss_mb'] for p in sample['processes'].values())
                all_memory_values.append(total_memory)
            
            report_data['summary'] = {
                'min_memory_mb': min(all_memory_values),
                'max_memory_mb': max(all_memory_values),
                'avg_memory_mb': sum(all_memory_values) / len(all_memory_values),
                'memory_growth_mb': all_memory_values[-1] - all_memory_values[0]
            }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_path}")
        
        # Generate visualization
        self.generate_visualization()
        
    def generate_visualization(self):
        """Generate memory usage visualization"""
        if not self.memory_samples:
            return
        
        # Extract data for plotting
        timestamps = []
        memory_by_process = defaultdict(list)
        
        for sample in self.memory_samples:
            timestamps.append(sample['timestamp'])
            
            for pid, stats in sample['processes'].items():
                key = stats['cmdline'][:40] + '...' if len(stats['cmdline']) > 40 else stats['cmdline']
                memory_by_process[key].append(stats['rss_mb'])
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        for process, memory_values in memory_by_process.items():
            plt.plot(timestamps, memory_values, marker='o', label=process)
        
        plt.xlabel('Time')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Hygiene Monitoring System Memory Usage Over Time')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gcf().autofmt_xdate()
        
        # Save plot
        plot_path = Path("/opt/sutazaiapp/reports/hygiene_memory_profile.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {plot_path}")
        
        plt.close()

async def main():
    parser = argparse.ArgumentParser(description='Profile hygiene monitoring memory usage')
    parser.add_argument('--duration', type=int, default=300, help='Profiling duration in seconds')
    args = parser.parse_args()
    
    profiler = HygieneMemoryProfiler(duration=args.duration)
    
    # Run profiling
    await profiler.monitor_memory()
    
    # Analyze results
    profiler.analyze_results()

if __name__ == '__main__':
    asyncio.run(main())