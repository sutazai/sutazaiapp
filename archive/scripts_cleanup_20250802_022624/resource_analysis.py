#!/usr/bin/env python3
"""
SutazAI Resource Usage Analysis Tool
Analyzes current and projected resource requirements
"""

import os
import psutil
import docker
import yaml
import json
from datetime import datetime
from collections import defaultdict
import subprocess

class ResourceAnalyzer:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.compose_path = "/opt/sutazaiapp/config/docker/docker-compose.yml"
        self.services = self._load_compose_config()
        
    def _load_compose_config(self):
        """Load and parse docker-compose.yml"""
        with open(self.compose_path, 'r') as f:
            return yaml.safe_load(f)
    
    def analyze_current_usage(self):
        """Analyze current system resource usage"""
        # System info
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk = psutil.disk_usage('/')
        
        # Docker stats
        containers = self.docker_client.containers.list()
        container_stats = []
        
        for container in containers:
            try:
                stats = container.stats(stream=False)
                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                           stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                              stats['precpu_stats']['system_cpu_usage']
                
                if system_delta > 0:
                    cpu_percent_container = (cpu_delta / system_delta) * cpu_count * 100
                else:
                    cpu_percent_container = 0
                
                memory_usage = stats['memory_stats']['usage'] / (1024**3)  # GB
                memory_limit = stats['memory_stats'].get('limit', 0) / (1024**3)
                
                container_stats.append({
                    'name': container.name,
                    'cpu_percent': round(cpu_percent_container, 2),
                    'memory_gb': round(memory_usage, 2),
                    'memory_limit_gb': round(memory_limit, 2),
                    'status': container.status
                })
            except Exception as e:
                print(f"Error getting stats for {container.name}: {e}")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu_cores': cpu_count,
                'cpu_percent': cpu_percent,
                'memory_total_gb': round(memory.total / (1024**3), 2),
                'memory_used_gb': round(memory.used / (1024**3), 2),
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'memory_percent': memory.percent,
                'swap_total_gb': round(swap.total / (1024**3), 2),
                'swap_used_gb': round(swap.used / (1024**3), 2),
                'swap_percent': swap.percent,
                'disk_total_gb': round(disk.total / (1024**3), 2),
                'disk_used_gb': round(disk.used / (1024**3), 2),
                'disk_free_gb': round(disk.free / (1024**3), 2),
                'disk_percent': disk.percent
            },
            'containers': container_stats
        }
    
    def calculate_service_requirements(self):
        """Calculate theoretical resource requirements from docker-compose.yml"""
        requirements = {
            'total_cpu_cores': 0,
            'total_memory_gb': 0,
            'total_gpu_memory_gb': 0,
            'services': {},
            'by_category': defaultdict(lambda: {'count': 0, 'cpu': 0, 'memory_gb': 0})
        }
        
        for service_name, service_config in self.services.get('services', {}).items():
            deploy = service_config.get('deploy', {})
            resources = deploy.get('resources', {})
            limits = resources.get('limits', {})
            reservations = resources.get('reservations', {})
            
            # Parse CPU limits
            cpu_limit = limits.get('cpus', '1')
            cpu_reservation = reservations.get('cpus', '0.5')
            
            # Parse memory limits
            memory_limit = limits.get('memory', '512M')
            memory_reservation = reservations.get('memory', '256M')
            
            # Convert memory to GB
            def parse_memory(mem_str):
                if isinstance(mem_str, (int, float)):
                    return mem_str
                mem_str = str(mem_str).upper()
                if 'G' in mem_str:
                    return float(mem_str.replace('G', ''))
                elif 'M' in mem_str:
                    return float(mem_str.replace('M', '')) / 1024
                return 0.5  # Default 512MB
            
            cpu_float = float(str(cpu_limit).replace("'", ""))
            memory_gb = parse_memory(memory_limit)
            
            requirements['services'][service_name] = {
                'cpu_limit': cpu_float,
                'cpu_reservation': float(str(cpu_reservation).replace("'", "")),
                'memory_limit_gb': memory_gb,
                'memory_reservation_gb': parse_memory(memory_reservation)
            }
            
            requirements['total_cpu_cores'] += cpu_float
            requirements['total_memory_gb'] += memory_gb
            
            # Categorize services
            if 'ollama' in service_name or 'model' in service_name:
                category = 'ai_models'
            elif 'agent' in service_name or 'gpt' in service_name or 'ai' in service_name:
                category = 'ai_agents'
            elif 'postgres' in service_name or 'redis' in service_name or 'neo4j' in service_name:
                category = 'databases'
            elif 'chroma' in service_name or 'qdrant' in service_name or 'faiss' in service_name:
                category = 'vector_stores'
            elif 'prometheus' in service_name or 'grafana' in service_name or 'loki' in service_name:
                category = 'monitoring'
            else:
                category = 'other'
            
            requirements['by_category'][category]['count'] += 1
            requirements['by_category'][category]['cpu'] += cpu_float
            requirements['by_category'][category]['memory_gb'] += memory_gb
        
        return requirements
    
    def identify_bottlenecks(self, current, requirements):
        """Identify resource bottlenecks"""
        bottlenecks = []
        
        # CPU bottleneck
        cpu_utilization = (requirements['total_cpu_cores'] / current['system']['cpu_cores']) * 100
        if cpu_utilization > 80:
            bottlenecks.append({
                'type': 'CPU',
                'severity': 'HIGH' if cpu_utilization > 100 else 'MEDIUM',
                'current_cores': current['system']['cpu_cores'],
                'required_cores': requirements['total_cpu_cores'],
                'utilization_percent': round(cpu_utilization, 2),
                'recommendation': f"Need {max(16, int(requirements['total_cpu_cores'] * 1.3))} CPU cores for optimal performance"
            })
        
        # Memory bottleneck
        memory_utilization = (requirements['total_memory_gb'] / current['system']['memory_total_gb']) * 100
        if memory_utilization > 80:
            bottlenecks.append({
                'type': 'Memory',
                'severity': 'CRITICAL' if memory_utilization > 100 else 'HIGH',
                'current_gb': current['system']['memory_total_gb'],
                'required_gb': requirements['total_memory_gb'],
                'utilization_percent': round(memory_utilization, 2),
                'recommendation': f"Need {max(32, int(requirements['total_memory_gb'] * 1.3))}GB RAM for safe operation"
            })
        
        # Disk bottleneck
        if current['system']['disk_percent'] > 80:
            bottlenecks.append({
                'type': 'Disk',
                'severity': 'MEDIUM',
                'current_used_gb': current['system']['disk_used_gb'],
                'current_free_gb': current['system']['disk_free_gb'],
                'utilization_percent': current['system']['disk_percent'],
                'recommendation': "Free up disk space or expand storage"
            })
        
        return bottlenecks
    
    def calculate_minimum_requirements(self):
        """Calculate minimum requirements for core automation functionality"""
        core_services = [
            'postgres', 'redis', 'neo4j', 'ollama', 'backend', 'frontend',
            'chromadb', 'qdrant', 'jarvis-agi', 'agent-message-bus', 'agent-registry'
        ]
        
        min_requirements = {
            'cpu_cores': 0,
            'memory_gb': 0,
            'disk_gb': 50,  # Base requirement
            'services': {}
        }
        
        for service_name, service_config in self.services.get('services', {}).items():
            if any(core in service_name for core in core_services):
                service_req = self.calculate_service_requirements()['services'].get(service_name, {})
                if service_req:
                    min_requirements['cpu_cores'] += service_req.get('cpu_reservation', 0.5)
                    min_requirements['memory_gb'] += service_req.get('memory_reservation_gb', 0.5)
                    min_requirements['services'][service_name] = service_req
        
        # Add overhead
        min_requirements['cpu_cores'] = round(min_requirements['cpu_cores'] * 1.2, 1)
        min_requirements['memory_gb'] = round(min_requirements['memory_gb'] * 1.2, 1)
        
        return min_requirements
    
    def generate_optimization_recommendations(self, current, requirements, bottlenecks):
        """Generate specific optimization recommendations"""
        recommendations = []
        
        # Memory optimizations
        if any(b['type'] == 'Memory' for b in bottlenecks):
            recommendations.extend([
                {
                    'priority': 'HIGH',
                    'category': 'Memory',
                    'action': 'Enable swap file',
                    'command': 'sudo fallocate -l 16G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile',
                    'impact': 'Provides 16GB additional virtual memory'
                },
                {
                    'priority': 'HIGH',
                    'category': 'Memory',
                    'action': 'Use lightweight models',
                    'details': 'Switch to tinyllama, qwen2.5:0.5b, or smollm:360m models',
                    'impact': 'Reduce memory usage by 70-80%'
                },
                {
                    'priority': 'MEDIUM',
                    'category': 'Memory',
                    'action': 'Limit concurrent services',
                    'details': 'Run only essential services: postgres, redis, ollama, backend, frontend',
                    'impact': 'Save 8-10GB RAM'
                }
            ])
        
        # CPU optimizations
        if any(b['type'] == 'CPU' for b in bottlenecks):
            recommendations.extend([
                {
                    'priority': 'HIGH',
                    'category': 'CPU',
                    'action': 'Reduce CPU limits',
                    'details': 'Lower CPU limits for non-critical services in docker-compose.yml',
                    'impact': 'Better CPU scheduling and reduced contention'
                },
                {
                    'priority': 'MEDIUM',
                    'category': 'CPU',
                    'action': 'Enable CPU throttling',
                    'command': 'docker update --cpus="0.5" $(docker ps -q)',
                    'impact': 'Prevent any single container from monopolizing CPU'
                }
            ])
        
        # General optimizations
        recommendations.extend([
            {
                'priority': 'HIGH',
                'category': 'Docker',
                'action': 'Prune unused Docker resources',
                'command': 'docker system prune -a --volumes -f',
                'impact': 'Free up disk space and reduce overhead'
            },
            {
                'priority': 'MEDIUM',
                'category': 'Models',
                'action': 'Implement model quantization',
                'details': 'Use GGUF quantized models (Q4_0 or Q4_K_M)',
                'impact': 'Reduce model size by 60-70% with minimal quality loss'
            },
            {
                'priority': 'LOW',
                'category': 'Monitoring',
                'action': 'Disable non-essential monitoring',
                'details': 'Temporarily disable Prometheus, Grafana, Loki to save resources',
                'impact': 'Save 2-3GB RAM and 2 CPU cores'
            }
        ])
        
        return recommendations
    
    def generate_report(self):
        """Generate comprehensive resource analysis report"""
        print("🔍 SutazAI Resource Usage Analysis\n")
        print("=" * 80)
        
        # Current usage
        current = self.analyze_current_usage()
        print("\n📊 CURRENT SYSTEM RESOURCES")
        print("-" * 40)
        print(f"CPU: {current['system']['cpu_cores']} cores @ {current['system']['cpu_percent']}% utilization")
        print(f"Memory: {current['system']['memory_used_gb']}/{current['system']['memory_total_gb']}GB ({current['system']['memory_percent']}%)")
        print(f"Swap: {current['system']['swap_used_gb']}/{current['system']['swap_total_gb']}GB ({current['system']['swap_percent']}%)")
        print(f"Disk: {current['system']['disk_used_gb']}/{current['system']['disk_total_gb']}GB ({current['system']['disk_percent']}%)")
        
        # Docker containers
        print("\n🐳 DOCKER CONTAINER USAGE")
        print("-" * 40)
        if current['containers']:
            print(f"{'Container':<40} {'CPU%':<8} {'Memory':<12} {'Status'}")
            print("-" * 70)
            for container in current['containers']:
                print(f"{container['name']:<40} {container['cpu_percent']:<8} "
                      f"{container['memory_gb']:.1f}/{container['memory_limit_gb']:.1f}GB  {container['status']}")
        else:
            print("No running containers found")
        
        # Service requirements
        requirements = self.calculate_service_requirements()
        print("\n📋 SERVICE REQUIREMENTS (from docker-compose.yml)")
        print("-" * 40)
        print(f"Total Services Defined: {len(requirements['services'])}")
        print(f"Total CPU Cores Required: {requirements['total_cpu_cores']:.1f}")
        print(f"Total Memory Required: {requirements['total_memory_gb']:.1f}GB")
        
        print("\n📁 REQUIREMENTS BY CATEGORY")
        print("-" * 40)
        for category, stats in sorted(requirements['by_category'].items()):
            print(f"{category:<15} Services: {stats['count']:<3} CPU: {stats['cpu']:.1f} cores  Memory: {stats['memory_gb']:.1f}GB")
        
        # Bottlenecks
        bottlenecks = self.identify_bottlenecks(current, requirements)
        if bottlenecks:
            print("\n🚨 RESOURCE BOTTLENECKS")
            print("-" * 40)
            for bottleneck in bottlenecks:
                print(f"\n[{bottleneck['severity']}] {bottleneck['type']} Bottleneck")
                for key, value in bottleneck.items():
                    if key not in ['type', 'severity']:
                        print(f"  {key}: {value}")
        
        # Minimum requirements
        min_req = self.calculate_minimum_requirements()
        print("\n💡 MINIMUM REQUIREMENTS FOR CORE automation")
        print("-" * 40)
        print(f"CPU Cores: {min_req['cpu_cores']}")
        print(f"Memory: {min_req['memory_gb']}GB")
        print(f"Disk: {min_req['disk_gb']}GB")
        print(f"Core Services: {', '.join(min_req['services'].keys())}")
        
        # Optimization recommendations
        recommendations = self.generate_optimization_recommendations(current, requirements, bottlenecks)
        print("\n🔧 OPTIMIZATION RECOMMENDATIONS")
        print("-" * 40)
        
        for priority in ['HIGH', 'MEDIUM', 'LOW']:
            priority_recs = [r for r in recommendations if r['priority'] == priority]
            if priority_recs:
                print(f"\n{priority} Priority:")
                for rec in priority_recs:
                    print(f"\n  • {rec['action']} [{rec['category']}]")
                    if 'details' in rec:
                        print(f"    Details: {rec['details']}")
                    if 'command' in rec:
                        print(f"    Command: {rec['command']}")
                    print(f"    Impact: {rec['impact']}")
        
        # Save report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'current_usage': current,
            'service_requirements': requirements,
            'bottlenecks': bottlenecks,
            'minimum_requirements': min_req,
            'recommendations': recommendations
        }
        
        report_path = '/opt/sutazaiapp/resource_analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Full report saved to: {report_path}")
        print("=" * 80)

if __name__ == "__main__":
    analyzer = ResourceAnalyzer()
    analyzer.generate_report()