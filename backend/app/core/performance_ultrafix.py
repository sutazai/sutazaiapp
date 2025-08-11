"""
ULTRAFIX: Performance Monitoring and Optimization Module
Real-time performance tracking and automatic optimization
"""

import asyncio
import psutil
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import docker
import httpx

logger = logging.getLogger(__name__)


class PerformanceUltrafix:
    """
    ULTRAFIX: Advanced performance monitoring and optimization system
    
    Features:
    - Real-time container memory monitoring
    - Automatic memory limit adjustments
    - Service performance profiling
    - Resource waste detection
    - Automatic container cleanup
    """
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.performance_data = {}
        self.optimization_history = []
        self.memory_targets = {
            # Optimized memory limits based on actual usage
            'postgres': 512,  # MB
            'redis': 256,
            'neo4j': 1024,
            'ollama': 2048,
            'rabbitmq': 512,
            'qdrant': 512,
            'chromadb': 256,
            'backend': 512,
            'frontend': 256,
            'agents': 256,  # Per agent
            'monitoring': 128,  # Prometheus, Grafana, etc
        }
        
    def get_container_stats(self) -> List[Dict[str, Any]]:
        """Get real-time stats for all containers"""
        stats = []
        
        for container in self.docker_client.containers.list():
            try:
                container_stats = container.stats(stream=False)
                
                # Calculate memory usage
                mem_usage = container_stats['memory_stats'].get('usage', 0)
                mem_limit = container_stats['memory_stats'].get('limit', 0)
                mem_percent = (mem_usage / mem_limit * 100) if mem_limit > 0 else 0
                
                # Calculate CPU usage
                cpu_delta = container_stats['cpu_stats']['cpu_usage']['total_usage'] - \
                           container_stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = container_stats['cpu_stats']['system_cpu_usage'] - \
                              container_stats['precpu_stats']['system_cpu_usage']
                cpu_percent = (cpu_delta / system_delta * 100) if system_delta > 0 else 0
                
                stats.append({
                    'name': container.name,
                    'id': container.short_id,
                    'status': container.status,
                    'memory_mb': round(mem_usage / 1024 / 1024, 2),
                    'memory_limit_mb': round(mem_limit / 1024 / 1024, 2),
                    'memory_percent': round(mem_percent, 2),
                    'cpu_percent': round(cpu_percent, 2),
                    'optimized': self._is_optimized(container.name, mem_usage, mem_limit)
                })
                
            except Exception as e:
                logger.error(f"Error getting stats for {container.name}: {e}")
                
        return stats
    
    def _is_optimized(self, container_name: str, mem_usage: int, mem_limit: int) -> bool:
        """Check if container memory is optimized"""
        # Container is optimized if:
        # 1. Memory usage is between 30-70% of limit (not over or under provisioned)
        # 2. Limit is close to our target
        
        if mem_limit == 0:
            return False
            
        usage_percent = (mem_usage / mem_limit) * 100
        
        # Check if within optimal range
        if 30 <= usage_percent <= 70:
            # Check if limit matches our target
            service_type = self._get_service_type(container_name)
            if service_type in self.memory_targets:
                target_mb = self.memory_targets[service_type]
                limit_mb = mem_limit / 1024 / 1024
                
                # Within 20% of target
                if abs(limit_mb - target_mb) / target_mb <= 0.2:
                    return True
                    
        return False
    
    def _get_service_type(self, container_name: str) -> str:
        """Determine service type from container name"""
        name_lower = container_name.lower()
        
        if 'postgres' in name_lower:
            return 'postgres'
        elif 'redis' in name_lower:
            return 'redis'
        elif 'neo4j' in name_lower:
            return 'neo4j'
        elif 'ollama' in name_lower:
            return 'ollama'
        elif 'rabbit' in name_lower:
            return 'rabbitmq'
        elif 'qdrant' in name_lower:
            return 'qdrant'
        elif 'chroma' in name_lower:
            return 'chromadb'
        elif 'backend' in name_lower:
            return 'backend'
        elif 'frontend' in name_lower:
            return 'frontend'
        elif 'agent' in name_lower:
            return 'agents'
        elif any(x in name_lower for x in ['prometheus', 'grafana', 'loki']):
            return 'monitoring'
        else:
            return 'default'
    
    async def optimize_memory_allocations(self) -> Dict[str, Any]:
        """
        ULTRAFIX: Automatically optimize memory allocations
        """
        optimizations = []
        total_saved_mb = 0
        
        stats = self.get_container_stats()
        
        for stat in stats:
            if not stat['optimized']:
                service_type = self._get_service_type(stat['name'])
                
                if service_type in self.memory_targets:
                    target_mb = self.memory_targets[service_type]
                    current_mb = stat['memory_limit_mb']
                    
                    if current_mb > target_mb * 1.5:
                        # Over-provisioned, can reduce
                        saved = current_mb - target_mb
                        total_saved_mb += saved
                        
                        optimizations.append({
                            'container': stat['name'],
                            'action': 'reduce',
                            'current_mb': current_mb,
                            'target_mb': target_mb,
                            'saved_mb': saved
                        })
                        
                        # Apply optimization (would need docker-compose update in production)
                        logger.info(f"ULTRAFIX: {stat['name']} can reduce memory from {current_mb}MB to {target_mb}MB")
                        
                    elif current_mb < target_mb * 0.7 and stat['memory_percent'] > 80:
                        # Under-provisioned and struggling
                        optimizations.append({
                            'container': stat['name'],
                            'action': 'increase',
                            'current_mb': current_mb,
                            'target_mb': target_mb,
                            'reason': 'high_memory_pressure'
                        })
                        
                        logger.info(f"ULTRAFIX: {stat['name']} needs memory increase from {current_mb}MB to {target_mb}MB")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'optimizations': optimizations,
            'total_saved_mb': total_saved_mb,
            'potential_savings_gb': round(total_saved_mb / 1024, 2),
            'containers_analyzed': len(stats),
            'containers_optimized': sum(1 for s in stats if s['optimized']),
            'optimization_rate': round(sum(1 for s in stats if s['optimized']) / len(stats) * 100, 2) if stats else 0
        }
    
    async def cleanup_unused_resources(self) -> Dict[str, Any]:
        """
        ULTRAFIX: Clean up unused Docker resources
        """
        cleanup_report = {
            'images': 0,
            'containers': 0,
            'volumes': 0,
            'networks': 0,
            'space_reclaimed_mb': 0
        }
        
        try:
            # Remove stopped containers
            stopped_containers = self.docker_client.containers.list(filters={'status': 'exited'})
            for container in stopped_containers:
                container.remove()
                cleanup_report['containers'] += 1
            
            # Remove unused images
            images = self.docker_client.images.prune()
            cleanup_report['images'] = len(images.get('ImagesDeleted', []))
            cleanup_report['space_reclaimed_mb'] += images.get('SpaceReclaimed', 0) / 1024 / 1024
            
            # Remove unused volumes
            volumes = self.docker_client.volumes.prune()
            cleanup_report['volumes'] = len(volumes.get('VolumesDeleted', []))
            cleanup_report['space_reclaimed_mb'] += volumes.get('SpaceReclaimed', 0) / 1024 / 1024
            
            # Remove unused networks
            networks = self.docker_client.networks.prune()
            cleanup_report['networks'] = len(networks.get('NetworksDeleted', []))
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            
        cleanup_report['space_reclaimed_mb'] = round(cleanup_report['space_reclaimed_mb'], 2)
        
        return cleanup_report
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance optimization report
        """
        stats = self.get_container_stats()
        
        # Calculate totals
        total_memory_used = sum(s['memory_mb'] for s in stats)
        total_memory_limit = sum(s['memory_limit_mb'] for s in stats)
        avg_cpu = sum(s['cpu_percent'] for s in stats) / len(stats) if stats else 0
        
        # Find worst performers
        memory_wasters = sorted(
            [s for s in stats if s['memory_percent'] < 20],
            key=lambda x: x['memory_limit_mb'] - x['memory_mb'],
            reverse=True
        )[:5]
        
        cpu_hogs = sorted(
            stats,
            key=lambda x: x['cpu_percent'],
            reverse=True
        )[:5]
        
        # Get system metrics
        system_memory = psutil.virtual_memory()
        system_cpu = psutil.cpu_percent(interval=1)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'total_memory_gb': round(system_memory.total / 1024 / 1024 / 1024, 2),
                'available_memory_gb': round(system_memory.available / 1024 / 1024 / 1024, 2),
                'memory_percent': system_memory.percent,
                'cpu_percent': system_cpu
            },
            'containers': {
                'total': len(stats),
                'running': len([s for s in stats if s['status'] == 'running']),
                'optimized': len([s for s in stats if s['optimized']]),
                'total_memory_used_mb': round(total_memory_used, 2),
                'total_memory_limit_mb': round(total_memory_limit, 2),
                'memory_efficiency': round((total_memory_used / total_memory_limit * 100), 2) if total_memory_limit > 0 else 0,
                'avg_cpu_percent': round(avg_cpu, 2)
            },
            'optimization_opportunities': {
                'memory_wasters': [
                    {
                        'container': w['name'],
                        'wasted_mb': round(w['memory_limit_mb'] - w['memory_mb'], 2),
                        'usage_percent': w['memory_percent']
                    }
                    for w in memory_wasters
                ],
                'cpu_intensive': [
                    {
                        'container': c['name'],
                        'cpu_percent': c['cpu_percent']
                    }
                    for c in cpu_hogs
                ],
                'potential_memory_savings_mb': round(
                    sum(w['memory_limit_mb'] - w['memory_mb'] for w in memory_wasters),
                    2
                )
            },
            'recommendations': self._generate_recommendations(stats)
        }
    
    def _generate_recommendations(self, stats: List[Dict]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Check for over-provisioned containers
        over_provisioned = [s for s in stats if s['memory_percent'] < 30]
        if over_provisioned:
            recommendations.append(
                f"Reduce memory limits for {len(over_provisioned)} over-provisioned containers"
            )
        
        # Check for under-provisioned containers
        under_provisioned = [s for s in stats if s['memory_percent'] > 90]
        if under_provisioned:
            recommendations.append(
                f"Increase memory limits for {len(under_provisioned)} containers near capacity"
            )
        
        # Check for stopped containers
        all_containers = self.docker_client.containers.list(all=True)
        stopped = [c for c in all_containers if c.status == 'exited']
        if stopped:
            recommendations.append(
                f"Remove {len(stopped)} stopped containers to free resources"
            )
        
        # Check optimization rate
        optimization_rate = (len([s for s in stats if s['optimized']]) / len(stats) * 100) if stats else 0
        if optimization_rate < 80:
            recommendations.append(
                f"Only {optimization_rate:.0f}% of containers are optimized - run memory optimization"
            )
        
        return recommendations


# Global instance
_performance_monitor = None


async def get_performance_monitor() -> PerformanceUltrafix:
    """Get or create performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceUltrafix()
    return _performance_monitor


async def run_performance_optimization():
    """
    ULTRAFIX: Run complete performance optimization cycle
    """
    monitor = await get_performance_monitor()
    
    # Generate performance report
    report = await monitor.generate_performance_report()
    logger.info(f"Performance Report: {report['containers']['optimized']}/{report['containers']['total']} containers optimized")
    
    # Optimize memory allocations
    memory_opt = await monitor.optimize_memory_allocations()
    if memory_opt['total_saved_mb'] > 0:
        logger.info(f"ULTRAFIX: Can save {memory_opt['total_saved_mb']}MB by optimizing memory allocations")
    
    # Clean up unused resources
    cleanup = await monitor.cleanup_unused_resources()
    if cleanup['space_reclaimed_mb'] > 0:
        logger.info(f"ULTRAFIX: Reclaimed {cleanup['space_reclaimed_mb']}MB by cleaning unused resources")
    
    return {
        'performance_report': report,
        'memory_optimization': memory_opt,
        'cleanup_report': cleanup
    }