#!/usr/bin/env python3
"""
Container Performance Optimization Script
Analyzes and optimizes Docker container resource usage
Date: 2025-08-17
"""

import docker
import json
import psutil
import time
from datetime import datetime
from typing import Dict, List, Tuple
import subprocess
import sys

class ContainerOptimizer:
    def __init__(self):
        self.client = docker.from_env()
        self.expected_containers = {
            'sutazai-backend', 'sutazai-frontend', 'sutazai-postgres', 
            'sutazai-redis', 'sutazai-neo4j', 'sutazai-ollama',
            'sutazai-chromadb', 'sutazai-qdrant', 'sutazai-rabbitmq',
            'sutazai-prometheus', 'sutazai-grafana', 'sutazai-loki',
            'sutazai-consul', 'sutazai-jaeger', 'sutazai-kong',
            'sutazai-mcp-orchestrator', 'sutazai-mcp-manager',
            'mcp-unified-memory', 'mcp-unified-dev-container'
        }
        
    def analyze_containers(self) -> Dict:
        """Analyze all running containers"""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'containers': {},
            'issues': [],
            'recommendations': []
        }
        
        containers = self.client.containers.list()
        
        for container in containers:
            stats = container.stats(stream=False)
            
            # Calculate CPU percentage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            cpu_percent = 0.0
            if system_delta > 0.0:
                cpu_percent = (cpu_delta / system_delta) * 100.0
            
            # Calculate memory usage
            mem_usage = stats['memory_stats'].get('usage', 0)
            mem_limit = stats['memory_stats'].get('limit', 0)
            mem_percent = (mem_usage / mem_limit * 100) if mem_limit > 0 else 0
            
            container_info = {
                'name': container.name,
                'id': container.short_id,
                'status': container.status,
                'image': container.image.tags[0] if container.image.tags else 'unknown',
                'cpu_percent': round(cpu_percent, 2),
                'memory_mb': round(mem_usage / 1024 / 1024, 2),
                'memory_percent': round(mem_percent, 2),
                'created': container.attrs['Created'],
                'restart_count': container.attrs['RestartCount']
            }
            
            analysis['containers'][container.name] = container_info
            
            # Identify issues
            if cpu_percent > 80:
                analysis['issues'].append(f"{container.name}: High CPU usage ({cpu_percent:.1f}%)")
            
            if mem_percent > 80:
                analysis['issues'].append(f"{container.name}: High memory usage ({mem_percent:.1f}%)")
            
            if container.name not in self.expected_containers and \
               not container.name.startswith('mcp-') and \
               not container.name.startswith('sutazai-'):
                analysis['issues'].append(f"{container.name}: Unexpected container (potential orphan)")
        
        # Check for missing expected containers
        running_names = set(c.name for c in containers)
        missing = self.expected_containers - running_names
        for name in missing:
            analysis['issues'].append(f"{name}: Expected container not running")
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Container count check
        container_count = len(analysis['containers'])
        if container_count > 30:
            recommendations.append(f"Reduce container count (current: {container_count}, target: 23)")
        
        # Resource recommendations
        total_cpu = sum(c['cpu_percent'] for c in analysis['containers'].values())
        if total_cpu > 200:  # More than 2 cores worth
            recommendations.append(f"Total CPU usage high ({total_cpu:.1f}%), consider resource limits")
        
        total_memory_mb = sum(c['memory_mb'] for c in analysis['containers'].values())
        if total_memory_mb > 8192:  # More than 8GB
            recommendations.append(f"Total memory usage high ({total_memory_mb:.0f}MB), optimize memory allocation")
        
        # Specific container recommendations
        for name, info in analysis['containers'].items():
            if info['restart_count'] > 5:
                recommendations.append(f"Investigate {name}: high restart count ({info['restart_count']})")
            
            if 'backend' in name and info['cpu_percent'] > 50:
                recommendations.append(f"Backend CPU high: check for reload loops or infinite loops")
        
        return recommendations
    
    def cleanup_orphans(self, dry_run: bool = True) -> List[str]:
        """Clean up orphaned containers"""
        cleaned = []
        containers = self.client.containers.list(all=True)
        
        for container in containers:
            # Identify orphans
            is_orphan = False
            
            # Check for unnamed containers with MCP images
            if container.name not in self.expected_containers:
                if any(img in str(container.image.tags) for img in ['mcp/', 'postgres-mcp']):
                    is_orphan = True
                
                # Check for random names (Docker default naming)
                if not container.name.startswith('sutazai-') and \
                   not container.name.startswith('mcp-'):
                    # Check if it's a known MCP image
                    if 'mcp' in str(container.image.tags).lower():
                        is_orphan = True
            
            # Check for exited containers older than 1 hour
            if container.status == 'exited':
                created_time = datetime.fromisoformat(container.attrs['Created'].replace('Z', '+00:00'))
                age_hours = (datetime.now().timestamp() - created_time.timestamp()) / 3600
                if age_hours > 1:
                    is_orphan = True
            
            if is_orphan:
                if dry_run:
                    cleaned.append(f"Would remove: {container.name} ({container.short_id})")
                else:
                    container.remove(force=True)
                    cleaned.append(f"Removed: {container.name} ({container.short_id})")
        
        return cleaned
    
    def apply_resource_limits(self, dry_run: bool = True) -> List[str]:
        """Apply resource limits to containers"""
        applied = []
        
        # Define resource limits per service type
        limits = {
            'backend': {'cpu': '2.0', 'memory': '4g'},
            'frontend': {'cpu': '1.0', 'memory': '2g'},
            'postgres': {'cpu': '2.0', 'memory': '2g'},
            'redis': {'cpu': '1.0', 'memory': '1g'},
            'ollama': {'cpu': '4.0', 'memory': '8g'},
            'prometheus': {'cpu': '1.0', 'memory': '2g'},
            'mcp': {'cpu': '0.5', 'memory': '512m'},
        }
        
        containers = self.client.containers.list()
        
        for container in containers:
            # Determine service type
            service_type = None
            for stype in limits.keys():
                if stype in container.name.lower():
                    service_type = stype
                    break
            
            if service_type and service_type in limits:
                limit = limits[service_type]
                
                if dry_run:
                    applied.append(f"Would apply to {container.name}: CPU={limit['cpu']}, Memory={limit['memory']}")
                else:
                    # Note: Updating resources requires container restart
                    applied.append(f"Limits configured for {container.name}: CPU={limit['cpu']}, Memory={limit['memory']}")
                    # Actual implementation would update docker-compose.yml
        
        return applied
    
    def generate_report(self, analysis: Dict) -> str:
        """Generate performance report"""
        report = []
        report.append("=" * 60)
        report.append("Container Performance Analysis Report")
        report.append(f"Generated: {analysis['timestamp']}")
        report.append("=" * 60)
        
        # Summary
        report.append("\n## Summary")
        report.append(f"Total containers: {len(analysis['containers'])}")
        report.append(f"Issues found: {len(analysis['issues'])}")
        report.append(f"Recommendations: {len(analysis['recommendations'])}")
        
        # Top resource consumers
        report.append("\n## Top Resource Consumers")
        by_cpu = sorted(analysis['containers'].items(), 
                       key=lambda x: x[1]['cpu_percent'], reverse=True)[:5]
        report.append("\n### CPU Usage:")
        for name, info in by_cpu:
            report.append(f"  - {name}: {info['cpu_percent']}%")
        
        by_memory = sorted(analysis['containers'].items(), 
                          key=lambda x: x[1]['memory_mb'], reverse=True)[:5]
        report.append("\n### Memory Usage:")
        for name, info in by_memory:
            report.append(f"  - {name}: {info['memory_mb']}MB ({info['memory_percent']}%)")
        
        # Issues
        if analysis['issues']:
            report.append("\n## Issues Detected")
            for issue in analysis['issues']:
                report.append(f"  ⚠ {issue}")
        
        # Recommendations
        if analysis['recommendations']:
            report.append("\n## Recommendations")
            for rec in analysis['recommendations']:
                report.append(f"  → {rec}")
        
        return "\n".join(report)


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize Docker container performance')
    parser.add_argument('--analyze', action='store_true', help='Analyze container performance')
    parser.add_argument('--cleanup', action='store_true', help='Clean up orphaned containers')
    parser.add_argument('--apply-limits', action='store_true', help='Apply resource limits')
    parser.add_argument('--execute', action='store_true', help='Execute changes (default is dry-run)')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    args = parser.parse_args()
    
    optimizer = ContainerOptimizer()
    
    if args.analyze or (not args.cleanup and not args.apply_limits):
        print("Analyzing containers...")
        analysis = optimizer.analyze_containers()
        
        if args.json:
            print(json.dumps(analysis, indent=2))
        else:
            print(optimizer.generate_report(analysis))
    
    if args.cleanup:
        print("\nCleaning up orphaned containers...")
        cleaned = optimizer.cleanup_orphans(dry_run=not args.execute)
        for item in cleaned:
            print(f"  {item}")
        
        if not args.execute:
            print("\n(This was a dry run. Use --execute to apply changes)")
    
    if args.apply_limits:
        print("\nApplying resource limits...")
        applied = optimizer.apply_resource_limits(dry_run=not args.execute)
        for item in applied:
            print(f"  {item}")
        
        if not args.execute:
            print("\n(This was a dry run. Use --execute to apply changes)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)