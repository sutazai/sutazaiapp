#!/usr/bin/env python3
"""
SutazAI Performance Monitoring Script
Real-time system performance tracking and alerting
"""

import time
import json
import psutil
import docker
from datetime import datetime

def monitor_performance():
    """Monitor system performance continuously"""
    client = docker.from_env()
    
    while True:
        try:
            # Collect metrics
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'system_cpu': psutil.cpu_percent(interval=1),
                'system_memory': psutil.virtual_memory()._asdict(),
                'agent_stats': []
            }
            
            # Get agent statistics
            containers = client.containers.list()
            agent_containers = [c for c in containers if 'agent' in c.name]
            
            for container in agent_containers:
                try:
                    stats = container.stats(stream=False)
                    metrics['agent_stats'].append({
                        'name': container.name,
                        'status': container.status,
                        'cpu_percent': calculate_cpu_percent(stats),
                        'memory_usage': stats['memory_stats'].get('usage', 0)
                    })
                except Exception as e:
                    print(f"Failed to get stats for {container.name}: {e}")
            
            # Save metrics
            with open('/opt/sutazaiapp/logs/performance_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"[{datetime.now()}] Monitoring {len(agent_containers)} agents")
            time.sleep(60)  # Monitor every minute
            
        except KeyboardInterrupt:
            print("Monitoring stopped")
            break
        except Exception as e:
            print(f"Monitoring error: {e}")
            time.sleep(60)

def calculate_cpu_percent(stats):
    """Calculate CPU percentage from container stats"""
    try:
        cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] -                    stats['precpu_stats']['cpu_usage']['total_usage']
        system_delta = stats['cpu_stats']['system_cpu_usage'] -                       stats['precpu_stats']['system_cpu_usage']
        
        if cpu_delta > 0 and system_delta > 0:
            return (cpu_delta / system_delta) * 100.0
    except (KeyError, ZeroDivisionError):
        pass
    return 0.0

if __name__ == "__main__":
    monitor_performance()
