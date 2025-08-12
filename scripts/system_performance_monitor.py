#!/usr/bin/env python3
"""
REAL-TIME SYSTEM PERFORMANCE MONITOR
Date: August 12, 2025
Author: System Optimization and Reorganization Specialist

Monitors system performance with MCP container awareness
"""

import os
import sys
import time
import json
import psutil
import docker
from datetime import datetime
from typing import Dict, List
import signal

class SystemPerformanceMonitor:
    def __init__(self, monitor_duration=300):  # 5 minutes default
        self.docker_client = docker.from_env()
        self.monitor_duration = monitor_duration
        self.start_time = datetime.now()
        self.monitoring = True
        self.metrics_history = []
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\nReceived signal {signum}. Shutting down monitoring...")
        self.monitoring = False
        
    def get_detailed_metrics(self) -> Dict:
        """Get comprehensive system metrics"""
        try:
            # Basic system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            load_avg = os.getloadavg()
            
            # Process metrics
            process_count = len(psutil.pids())
            
            # File descriptors (with timeout)
            fd_count = 0
            try:
                import subprocess
                result = subprocess.run(['lsof'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    fd_count = len([line for line in result.stdout.split('\n') if line.strip()])
            except:
                pass  # FD count will remain 0 if unable to get
            
            # Docker metrics
            containers = self.docker_client.containers.list(all=True)
            running_containers = len([c for c in containers if c.status == 'running'])
            
            # Categorize containers
            mcp_containers = 0
            sutazai_containers = 0
            
            mcp_patterns = ['crystaldba/postgres-mcp', 'mcp/duckduckgo', 'mcp/fetch', 'mcp/sequentialthinking']
            
            for container in containers:
                if container.status == 'running':
                    image_name = container.image.tags[0] if container.image.tags else str(container.image.id)
                    
                    if any(pattern in image_name for pattern in mcp_patterns):
                        mcp_containers += 1
                    elif 'sutazai' in container.name.lower():
                        sutazai_containers += 1
            
            # Context switches
            context_switches = 0
            try:
                with open('/proc/stat', 'r') as f:
                    for line in f:
                        if line.startswith('ctxt'):
                            context_switches = int(line.split()[1])
                            break
            except:
                pass
            
            # Network connections
            network_connections = len(psutil.net_connections())
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'memory_percent': memory.percent,
                'disk_used_percent': disk.percent,
                'load_average_1m': load_avg[0],
                'load_average_5m': load_avg[1], 
                'load_average_15m': load_avg[2],
                'process_count': process_count,
                'file_descriptors': fd_count,
                'running_containers': running_containers,
                'mcp_containers': mcp_containers,
                'sutazai_containers': sutazai_containers,
                'context_switches': context_switches,
                'network_connections': network_connections
            }
            
        except Exception as e:
            print(f"Error getting metrics: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def get_top_processes(self, limit=5) -> List[Dict]:
        """Get top processes by CPU and memory"""
        try:
            processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'cmdline']):
                try:
                    proc_info = proc.info
                    if proc_info['cpu_percent'] > 0.1 or proc_info['memory_percent'] > 0.5:
                        # Truncate command line for safety
                        cmdline = ' '.join(proc_info['cmdline'] or [])[:80]
                        
                        processes.append({
                            'pid': proc_info['pid'],
                            'name': proc_info['name'],
                            'cpu_percent': proc_info['cpu_percent'],
                            'memory_percent': proc_info['memory_percent'],
                            'cmdline': cmdline
                        })
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Sort by CPU usage
            processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            return processes[:limit]
            
        except Exception as e:
            return [{'error': str(e)}]
    
    def display_metrics(self, metrics: Dict, top_processes: List[Dict]):
        """Display metrics in a readable format"""
        os.system('clear')  # Clear screen
        
        print("🔍 ULTRA SYSTEM PERFORMANCE MONITOR")
        print("=" * 70)
        
        if 'error' in metrics:
            print(f"❌ Error: {metrics['error']}")
            return
        
        # System overview
        print(f"⏰ Time: {metrics['timestamp']}")
        print(f"🏃 Uptime: {datetime.now() - self.start_time}")
        print()
        
        # Performance metrics
        print("📊 PERFORMANCE METRICS")
        print("-" * 30)
        print(f"CPU Usage:        {metrics['cpu_percent']:6.1f}%")
        print(f"Memory Usage:     {metrics['memory_percent']:6.1f}% ({metrics['memory_used_gb']:.1f}GB used)")
        print(f"Disk Usage:       {metrics['disk_used_percent']:6.1f}%")
        print(f"Load Average:     {metrics['load_average_1m']:.2f}, {metrics['load_average_5m']:.2f}, {metrics['load_average_15m']:.2f}")
        print()
        
        # Process and resource metrics
        print("🔢 RESOURCE METRICS")
        print("-" * 30)
        print(f"Processes:        {metrics['process_count']:>6}")
        print(f"File Descriptors: {metrics['file_descriptors']:>6}")
        print(f"Network Conns:    {metrics['network_connections']:>6}")
        print(f"Context Switches: {metrics['context_switches']:>6}")
        print()
        
        # Container metrics
        print("🐳 CONTAINER METRICS")
        print("-" * 30)
        print(f"Total Running:    {metrics['running_containers']:>6}")
        print(f"MCP Containers:   {metrics['mcp_containers']:>6} (PRESERVED)")
        print(f"SutazAI:          {metrics['sutazai_containers']:>6}")
        print()
        
        # Health indicators
        print("🚦 HEALTH INDICATORS")
        print("-" * 30)
        
        # CPU health
        cpu_status = "🟢 Good"
        if metrics['cpu_percent'] > 80:
            cpu_status = "🔴 Critical"
        elif metrics['cpu_percent'] > 60:
            cpu_status = "🟡 Warning"
            
        print(f"CPU Health:       {cpu_status}")
        
        # Memory health
        memory_status = "🟢 Good"
        if metrics['memory_percent'] > 85:
            memory_status = "🔴 Critical"
        elif metrics['memory_percent'] > 70:
            memory_status = "🟡 Warning"
            
        print(f"Memory Health:    {memory_status}")
        
        # Load health
        load_status = "🟢 Good"
        if metrics['load_average_1m'] > 4.0:
            load_status = "🔴 Critical"
        elif metrics['load_average_1m'] > 2.0:
            load_status = "🟡 Warning"
            
        print(f"Load Health:      {load_status}")
        
        # File descriptor health
        fd_status = "🟢 Good"
        if metrics['file_descriptors'] > 100000:
            fd_status = "🔴 Critical"
        elif metrics['file_descriptors'] > 50000:
            fd_status = "🟡 Warning"
            
        print(f"FD Health:        {fd_status}")
        print()
        
        # Top processes
        if top_processes and not top_processes[0].get('error'):
            print("🔝 TOP PROCESSES")
            print("-" * 70)
            print(f"{'PID':<8} {'CPU%':<6} {'MEM%':<6} {'NAME':<12} {'COMMAND'}")
            print("-" * 70)
            
            for proc in top_processes:
                print(f"{proc['pid']:<8} {proc['cpu_percent']:<6.1f} {proc['memory_percent']:<6.1f} {proc['name']:<12} {proc['cmdline'][:30]}")
        
        print()
        print("Press Ctrl+C to stop monitoring...")
    
    def save_metrics_history(self):
        """Save metrics history to file"""
        try:
            log_file = f'/opt/sutazaiapp/logs/system_metrics_{self.start_time.strftime("%Y%m%d_%H%M%S")}.json'
            
            report = {
                'monitoring_session': {
                    'start_time': self.start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
                    'metrics_count': len(self.metrics_history)
                },
                'metrics_history': self.metrics_history
            }
            
            with open(log_file, 'w') as f:
                json.dump(report, f, indent=2)
                
            print(f"\n📄 Metrics saved to: {log_file}")
            
        except Exception as e:
            print(f"Error saving metrics: {e}")
    
    def monitor(self):
        """Main monitoring loop"""
        print("🔍 Starting Ultra System Performance Monitor...")
        print(f"📊 Monitoring for {self.monitor_duration} seconds (or until Ctrl+C)")
        print("🔧 MCP containers are being preserved during optimization")
        print()
        
        end_time = datetime.now().timestamp() + self.monitor_duration
        
        while self.monitoring and datetime.now().timestamp() < end_time:
            try:
                # Get current metrics
                metrics = self.get_detailed_metrics()
                top_processes = self.get_top_processes(5)
                
                # Store for history
                self.metrics_history.append({
                    'metrics': metrics,
                    'top_processes': top_processes
                })
                
                # Display
                self.display_metrics(metrics, top_processes)
                
                # Wait before next update
                time.sleep(5)  # Update every 5 seconds
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(5)
        
        # Save final report
        self.save_metrics_history()
        print("\n✅ Monitoring session completed.")

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultra System Performance Monitor')
    parser.add_argument('--duration', '-d', type=int, default=300, 
                       help='Monitoring duration in seconds (default: 300)')
    parser.add_argument('--continuous', '-c', action='store_true',
                       help='Monitor continuously until interrupted')
    
    args = parser.parse_args()
    
    # Set duration
    duration = float('inf') if args.continuous else args.duration
    
    try:
        monitor = SystemPerformanceMonitor(duration)
        monitor.monitor()
        return 0
    except KeyboardInterrupt:
        print("\n⚠️  Monitoring interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())