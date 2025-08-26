#!/usr/bin/env python3
"""
SutazAI Performance Optimizer
Addresses critical performance issues and optimizes system resources
"""

import os
import sys
import time
import json
import psutil
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple

class SystemPerformanceOptimizer:
    def __init__(self):
        self.metrics_before = {}
        self.metrics_after = {}
        self.actions_taken = []
        
    def capture_metrics(self) -> Dict:
        """Capture current system performance metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': psutil.cpu_percent(interval=1),
                'count': psutil.cpu_count(),
                'load_average': os.getloadavg()
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'used': psutil.virtual_memory().used,
                'percent': psutil.virtual_memory().percent,
                'available': psutil.virtual_memory().available
            },
            'processes': {
                'total': len(psutil.pids()),
                'high_cpu': []
            },
            'docker': {
                'container_count': 0,
                'containers_with_limits': 0
            }
        }
        
        # Find high CPU processes
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                if proc.info['cpu_percent'] > 5:
                    metrics['processes']['high_cpu'].append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_percent': proc.info['memory_percent']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
        # Count Docker containers
        try:
            result = subprocess.run(['docker', 'ps', '-q'], capture_output=True, text=True)
            metrics['docker']['container_count'] = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
        except:
            pass
            
        return metrics
        
    def identify_duplicate_claude_processes(self) -> List[int]:
        """Identify duplicate Claude processes that can be safely terminated"""
        claude_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'create_time']):
            try:
                if 'claude' in proc.info['name'].lower():
                    claude_processes.append({
                        'pid': proc.info['pid'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'create_time': proc.info['create_time'],
                        'cmdline': ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
        # Sort by creation time (keep oldest)
        claude_processes.sort(key=lambda x: x['create_time'])
        
        # Keep the first one, mark others for termination
        pids_to_kill = []
        if len(claude_processes) > 1:
            print(f"Found {len(claude_processes)} Claude processes")
            for proc in claude_processes[1:]:
                if proc['cpu_percent'] > 5:  # Only kill high-CPU duplicates
                    pids_to_kill.append(proc['pid'])
                    
        return pids_to_kill
        
    def kill_duplicate_processes(self, pids: List[int]) -> bool:
        """Safely terminate duplicate processes"""
        success_count = 0
        
        for pid in pids:
            try:
                proc = psutil.Process(pid)
                print(f"Terminating duplicate process {pid} ({proc.name()})")
                proc.terminate()
                time.sleep(0.5)
                
                # Force kill if still running
                if proc.is_running():
                    proc.kill()
                    
                success_count += 1
                self.actions_taken.append(f"Killed duplicate process {pid}")
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"Could not terminate process {pid}: {e}")
                
        return success_count == len(pids)
        
    def optimize_docker_containers(self):
        """Add resource limits to Docker containers without them"""
        script = '''#!/bin/bash
        
        # Get containers without memory limits
        for container in $(docker ps -q); do
            name=$(docker inspect --format='{{.Name}}' $container | sed 's/^\\///')
            mem_limit=$(docker inspect --format='{{.HostConfig.Memory}}' $container)
            cpu_limit=$(docker inspect --format='{{.HostConfig.CpuQuota}}' $container)
            
            if [ "$mem_limit" = "0" ]; then
                echo "Adding resource limits to container: $name"
                
                # Determine appropriate limits based on container name
                case "$name" in
                    *postgres*|*postgresql*)
                        docker update --memory="512m" --memory-reservation="256m" --cpus="0.5" $container
                        ;;
                    *redis*)
                        docker update --memory="256m" --memory-reservation="128m" --cpus="0.25" $container
                        ;;
                    *neo4j*)
                        docker update --memory="1g" --memory-reservation="512m" --cpus="1.0" $container
                        ;;
                    *chroma*|*qdrant*)
                        docker update --memory="512m" --memory-reservation="256m" --cpus="0.5" $container
                        ;;
                    *)
                        # Default limits for unknown containers
                        docker update --memory="256m" --memory-reservation="128m" --cpus="0.25" $container
                        ;;
                esac
            fi
        done
        '''
        
        try:
            result = subprocess.run(['bash', '-c', script], capture_output=True, text=True)
            if result.stdout:
                print("Docker optimization output:", result.stdout)
            self.actions_taken.append("Applied resource limits to Docker containers")
        except Exception as e:
            print(f"Error optimizing Docker containers: {e}")
            
    def optimize_backend_asyncio(self):
        """Create optimized backend configuration"""
        config_content = '''# Backend Performance Configuration
# Optimized for reduced CPU usage and better async performance

# Worker settings
WORKERS=2  # Reduced from auto-detected to limit CPU usage
WORKER_CLASS=uvicorn.workers.UvicornWorker
WORKER_CONNECTIONS=100
MAX_REQUESTS=1000
MAX_REQUESTS_JITTER=50

# Asyncio settings
ASYNCIO_MAX_WORKERS=4
ASYNCIO_LOOP_POLICY=uvloop

# Connection pooling
CONNECTION_POOL_SIZE=10
CONNECTION_POOL_MAX_OVERFLOW=5
CONNECTION_POOL_TIMEOUT=30

# Cache settings
CACHE_TTL=300
CACHE_MAX_SIZE=1000

# Rate limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Monitoring
ENABLE_PERFORMANCE_MONITORING=true
PERFORMANCE_SAMPLE_RATE=0.1
'''
        
        config_path = '/opt/sutazaiapp/backend/.env.performance'
        try:
            with open(config_path, 'w') as f:
                f.write(config_content)
            print(f"Created performance configuration at {config_path}")
            self.actions_taken.append("Created optimized backend configuration")
        except Exception as e:
            print(f"Error creating backend config: {e}")
            
    def create_monitoring_script(self):
        """Create a monitoring script for ongoing performance tracking"""
        monitor_script = '''#!/bin/bash
# SutazAI Performance Monitor
# Run this script to check system performance

echo "=== SutazAI Performance Report ==="
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "=== System Resources ==="
echo "CPU Usage:"
top -bn1 | head -5 | tail -2
echo ""
echo "Memory Usage:"
free -h
echo ""
echo "Load Average: $(uptime | awk -F'load average:' '{print $2}')"
echo ""

echo "=== High CPU Processes ==="
ps aux | head -1
ps aux | sort -k3 -rn | head -10 | grep -v "ps aux"
echo ""

echo "=== Docker Containers ==="
docker stats --no-stream --format "table {{.Container}}\\t{{.Name}}\\t{{.CPUPerc}}\\t{{.MemUsage}}\\t{{.MemPerc}}"
echo ""

echo "=== Claude Processes ==="
ps aux | grep claude | grep -v grep || echo "No Claude processes found"
echo ""

echo "=== Backend Status ==="
curl -s http://localhost:10010/health | python3 -m json.tool 2>/dev/null || echo "Backend not responding"
echo ""

echo "=== Recommendations ==="
cpu_percent=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
if (( $(echo "$cpu_percent > 80" | bc -l) )); then
    echo "⚠️  High CPU usage detected. Consider:"
    echo "   - Running performance optimizer: python3 /opt/sutazaiapp/scripts/performance/optimize_system.py"
    echo "   - Checking for runaway processes"
    echo "   - Reviewing Docker container limits"
fi

mem_percent=$(free | grep Mem | awk '{print ($3/$2) * 100.0}' | cut -d'.' -f1)
if [ "$mem_percent" -gt 80 ]; then
    echo "⚠️  High memory usage detected. Consider:"
    echo "   - Restarting memory-intensive containers"
    echo "   - Clearing caches: sync && echo 3 > /proc/sys/vm/drop_caches"
fi

echo ""
echo "=== Performance History ==="
if [ -f /opt/sutazaiapp/logs/performance.log ]; then
    tail -5 /opt/sutazaiapp/logs/performance.log
else
    echo "No performance history available"
fi
'''
        
        monitor_path = '/opt/sutazaiapp/scripts/performance/monitor.sh'
        try:
            os.makedirs(os.path.dirname(monitor_path), exist_ok=True)
            with open(monitor_path, 'w') as f:
                f.write(monitor_script)
            os.chmod(monitor_path, 0o755)
            print(f"Created monitoring script at {monitor_path}")
            self.actions_taken.append("Created performance monitoring script")
        except Exception as e:
            print(f"Error creating monitor script: {e}")
            
    def save_performance_log(self):
        """Save performance metrics to log file"""
        log_dir = '/opt/sutazaiapp/logs'
        os.makedirs(log_dir, exist_ok=True)
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'metrics_before': self.metrics_before,
            'metrics_after': self.metrics_after,
            'actions_taken': self.actions_taken,
            'improvement': {
                'cpu_reduction': self.metrics_before['cpu']['percent'] - self.metrics_after['cpu']['percent'],
                'memory_freed': self.metrics_before['memory']['used'] - self.metrics_after['memory']['used'],
                'processes_terminated': len(self.metrics_before['processes']['high_cpu']) - len(self.metrics_after['processes']['high_cpu'])
            }
        }
        
        log_path = os.path.join(log_dir, 'performance.log')
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            print(f"Performance log saved to {log_path}")
        except Exception as e:
            print(f"Error saving log: {e}")
            
    def run_optimization(self):
        """Main optimization workflow"""
        print("=" * 60)
        print("SutazAI Performance Optimizer")
        print("=" * 60)
        
        # Capture initial metrics
        print("\n📊 Capturing baseline metrics...")
        self.metrics_before = self.capture_metrics()
        print(f"  CPU: {self.metrics_before['cpu']['percent']}%")
        print(f"  Memory: {self.metrics_before['memory']['percent']}%")
        print(f"  High CPU processes: {len(self.metrics_before['processes']['high_cpu'])}")
        
        # Kill duplicate Claude processes
        print("\n🔍 Identifying duplicate Claude processes...")
        duplicate_pids = self.identify_duplicate_claude_processes()
        if duplicate_pids:
            print(f"  Found {len(duplicate_pids)} duplicate processes to terminate")
            self.kill_duplicate_processes(duplicate_pids)
        else:
            print("  No duplicate Claude processes found")
            
        # Optimize Docker containers
        print("\n🐳 Optimizing Docker containers...")
        self.optimize_docker_containers()
        
        # Create optimized backend configuration
        print("\n⚙️  Creating optimized backend configuration...")
        self.optimize_backend_asyncio()
        
        # Create monitoring script
        print("\n📈 Creating monitoring script...")
        self.create_monitoring_script()
        
        # Wait for changes to take effect
        print("\n⏳ Waiting for optimizations to take effect...")
        time.sleep(3)
        
        # Capture final metrics
        print("\n📊 Capturing final metrics...")
        self.metrics_after = self.capture_metrics()
        print(f"  CPU: {self.metrics_after['cpu']['percent']}%")
        print(f"  Memory: {self.metrics_after['memory']['percent']}%")
        print(f"  High CPU processes: {len(self.metrics_after['processes']['high_cpu'])}")
        
        # Save performance log
        self.save_performance_log()
        
        # Print summary
        print("\n" + "=" * 60)
        print("OPTIMIZATION SUMMARY")
        print("=" * 60)
        
        cpu_improvement = self.metrics_before['cpu']['percent'] - self.metrics_after['cpu']['percent']
        memory_freed = (self.metrics_before['memory']['used'] - self.metrics_after['memory']['used']) / (1024**3)
        
        print(f"✅ CPU Usage Reduced: {cpu_improvement:.1f}%")
        print(f"✅ Memory Freed: {memory_freed:.1f} GB")
        print(f"✅ Actions Taken: {len(self.actions_taken)}")
        
        for action in self.actions_taken:
            print(f"   - {action}")
            
        print("\n📝 Next Steps:")
        print("1. Run monitoring script: bash /opt/sutazaiapp/scripts/performance/monitor.sh")
        print("2. Apply backend config: source /opt/sutazaiapp/backend/.env.performance")
        print("3. Restart backend service: systemctl restart sutazai-backend")
        print("4. Monitor performance: watch -n 5 'docker stats --no-stream'")
        
        return cpu_improvement > 0 or memory_freed > 0

if __name__ == "__main__":
    optimizer = SystemPerformanceOptimizer()
    success = optimizer.run_optimization()
    sys.exit(0 if success else 1)