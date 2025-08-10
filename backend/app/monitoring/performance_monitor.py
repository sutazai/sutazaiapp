#!/usr/bin/env python3
"""
Real-time Performance Monitoring Dashboard for SutazAI
Tracks system metrics, response times, and resource usage
"""

import asyncio
import time
import os
import psutil
import httpx
from collections import deque
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Real-time performance monitoring system"""
    
    def __init__(self, backend_url: str = "http://localhost:10010"):
        self.backend_url = backend_url
        self.metrics_history = {
            'timestamps': deque(maxlen=100),
            'cpu_usage': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'response_times': deque(maxlen=100),
            'cache_hit_rate': deque(maxlen=100),
            'active_connections': deque(maxlen=100),
            'requests_per_second': deque(maxlen=100),
            'error_rate': deque(maxlen=100)
        }
        self.alerts = []
        self.thresholds = {
            'cpu_critical': 90,
            'cpu_warning': 70,
            'memory_critical': 90,
            'memory_warning': 80,
            'response_time_critical': 500,  # ms
            'response_time_warning': 200,   # ms
            'error_rate_critical': 0.05,    # 5%
            'error_rate_warning': 0.01      # 1%
        }
        self._monitoring = False
        self._request_counter = 0
        self._error_counter = 0
        self._last_request_count = 0
        self._last_check_time = time.time()
        
    async def start_monitoring(self, interval: int = 5):
        """Start continuous monitoring"""
        self._monitoring = True
        logger.info(f"Starting performance monitoring (interval: {interval}s)")
        
        while self._monitoring:
            try:
                # Collect metrics
                metrics = await self.collect_metrics()
                
                # Store in history
                self.update_history(metrics)
                
                # Check for alerts
                self.check_alerts(metrics)
                
                # Display dashboard
                self.display_dashboard(metrics)
                
                # Wait for next interval
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(interval)
                
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect all system and application metrics"""
        
        metrics = {
            'timestamp': datetime.now(),
            'system': {},
            'application': {},
            'performance': {}
        }
        
        # System metrics
        metrics['system'] = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'network_connections': len(psutil.net_connections()),
            'process_count': len(psutil.pids())
        }
        
        # Application metrics from backend
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Get performance metrics
                response = await client.get(f"{self.backend_url}/api/v1/metrics")
                if response.status_code == 200:
                    app_metrics = response.json()
                    
                    metrics['application'] = {
                        'cache_hit_rate': app_metrics.get('performance', {}).get('cache', {}).get('hit_rate', 0),
                        'cache_size': app_metrics.get('performance', {}).get('cache', {}).get('local_cache_size', 0),
                        'task_queue_pending': (
                            app_metrics.get('performance', {}).get('task_queue', {}).get('pending_high', 0) +
                            app_metrics.get('performance', {}).get('task_queue', {}).get('pending_normal', 0) +
                            app_metrics.get('performance', {}).get('task_queue', {}).get('pending_low', 0)
                        ),
                        'ollama_avg_response_time': app_metrics.get('performance', {}).get('ollama', {}).get('avg_response_time', 0),
                        'connection_pool_active': app_metrics.get('performance', {}).get('connection_pools', {}).get('http_requests', 0)
                    }
                    
                # Test response time
                start_time = time.perf_counter()
                health_response = await client.get(f"{self.backend_url}/health")
                response_time = (time.perf_counter() - start_time) * 1000  # ms
                
                metrics['performance']['response_time_ms'] = response_time
                metrics['performance']['health_status'] = health_response.status_code == 200
                
                self._request_counter += 1
                
        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")
            self._error_counter += 1
            metrics['performance']['response_time_ms'] = -1
            metrics['performance']['health_status'] = False
            
        # Calculate requests per second
        current_time = time.time()
        time_diff = current_time - self._last_check_time
        if time_diff > 0:
            requests_diff = self._request_counter - self._last_request_count
            metrics['performance']['requests_per_second'] = requests_diff / time_diff
            self._last_request_count = self._request_counter
            self._last_check_time = current_time
        else:
            metrics['performance']['requests_per_second'] = 0
            
        # Calculate error rate
        if self._request_counter > 0:
            metrics['performance']['error_rate'] = self._error_counter / self._request_counter
        else:
            metrics['performance']['error_rate'] = 0
            
        return metrics
        
    def update_history(self, metrics: Dict[str, Any]):
        """Update metrics history"""
        
        self.metrics_history['timestamps'].append(metrics['timestamp'])
        self.metrics_history['cpu_usage'].append(metrics['system']['cpu_percent'])
        self.metrics_history['memory_usage'].append(metrics['system']['memory_percent'])
        self.metrics_history['response_times'].append(metrics['performance'].get('response_time_ms', 0))
        self.metrics_history['cache_hit_rate'].append(metrics['application'].get('cache_hit_rate', 0) * 100)
        self.metrics_history['active_connections'].append(metrics['application'].get('connection_pool_active', 0))
        self.metrics_history['requests_per_second'].append(metrics['performance'].get('requests_per_second', 0))
        self.metrics_history['error_rate'].append(metrics['performance'].get('error_rate', 0) * 100)
        
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check for performance alerts"""
        
        alerts = []
        
        # CPU alerts
        cpu = metrics['system']['cpu_percent']
        if cpu > self.thresholds['cpu_critical']:
            alerts.append(f"🔴 CRITICAL: CPU usage at {cpu:.1f}%")
        elif cpu > self.thresholds['cpu_warning']:
            alerts.append(f"🟡 WARNING: CPU usage at {cpu:.1f}%")
            
        # Memory alerts
        memory = metrics['system']['memory_percent']
        if memory > self.thresholds['memory_critical']:
            alerts.append(f"🔴 CRITICAL: Memory usage at {memory:.1f}%")
        elif memory > self.thresholds['memory_warning']:
            alerts.append(f"🟡 WARNING: Memory usage at {memory:.1f}%")
            
        # Response time alerts
        response_time = metrics['performance'].get('response_time_ms', 0)
        if response_time > self.thresholds['response_time_critical']:
            alerts.append(f"🔴 CRITICAL: Response time {response_time:.0f}ms")
        elif response_time > self.thresholds['response_time_warning']:
            alerts.append(f"🟡 WARNING: Response time {response_time:.0f}ms")
            
        # Error rate alerts
        error_rate = metrics['performance'].get('error_rate', 0)
        if error_rate > self.thresholds['error_rate_critical']:
            alerts.append(f"🔴 CRITICAL: Error rate at {error_rate*100:.2f}%")
        elif error_rate > self.thresholds['error_rate_warning']:
            alerts.append(f"🟡 WARNING: Error rate at {error_rate*100:.2f}%")
            
        # Health check
        if not metrics['performance'].get('health_status', True):
            alerts.append(f"🔴 CRITICAL: Backend health check failed")
            
        # Store alerts with timestamp
        for alert in alerts:
            self.alerts.append({
                'timestamp': metrics['timestamp'],
                'message': alert
            })
            
        # Keep only recent alerts (last 100)
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
            
    def display_dashboard(self, metrics: Dict[str, Any]):
        """Display real-time dashboard in console"""
        
        # Clear screen (works on Unix/Linux/Mac)
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("="*80)
        print("🎯 SUTAZAI PERFORMANCE MONITOR".center(80))
        print(f"{metrics['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}".center(80))
        print("="*80)
        
        # System Metrics
        print("\n📊 SYSTEM METRICS")
        print("-"*40)
        print(f"  CPU Usage:         {self._format_bar(metrics['system']['cpu_percent'], 100)} {metrics['system']['cpu_percent']:.1f}%")
        print(f"  Memory Usage:      {self._format_bar(metrics['system']['memory_percent'], 100)} {metrics['system']['memory_percent']:.1f}%")
        print(f"  Available Memory:  {metrics['system']['memory_available_mb']:.0f} MB")
        print(f"  Disk Usage:        {self._format_bar(metrics['system']['disk_usage_percent'], 100)} {metrics['system']['disk_usage_percent']:.1f}%")
        print(f"  Network Conns:     {metrics['system']['network_connections']}")
        print(f"  Process Count:     {metrics['system']['process_count']}")
        
        # Performance Metrics
        print("\n⚡ PERFORMANCE METRICS")
        print("-"*40)
        response_time = metrics['performance'].get('response_time_ms', 0)
        print(f"  Response Time:     {self._format_response_time(response_time)} {response_time:.0f}ms")
        print(f"  Requests/sec:      {metrics['performance'].get('requests_per_second', 0):.2f}")
        print(f"  Error Rate:        {metrics['performance'].get('error_rate', 0)*100:.2f}%")
        print(f"  Health Status:     {'✅ Healthy' if metrics['performance'].get('health_status') else '❌ Unhealthy'}")
        
        # Application Metrics
        print("\n🔧 APPLICATION METRICS")
        print("-"*40)
        cache_hit_rate = metrics['application'].get('cache_hit_rate', 0) * 100
        print(f"  Cache Hit Rate:    {self._format_bar(cache_hit_rate, 100)} {cache_hit_rate:.1f}%")
        print(f"  Cache Size:        {metrics['application'].get('cache_size', 0)} items")
        print(f"  Queue Pending:     {metrics['application'].get('task_queue_pending', 0)} tasks")
        print(f"  Ollama Avg Time:   {metrics['application'].get('ollama_avg_response_time', 0)*1000:.0f}ms")
        print(f"  Active Conns:      {metrics['application'].get('connection_pool_active', 0)}")
        
        # Recent Alerts
        if self.alerts:
            print("\n⚠️  RECENT ALERTS")
            print("-"*40)
            for alert in self.alerts[-5:]:  # Show last 5 alerts
                print(f"  {alert['timestamp'].strftime('%H:%M:%S')} - {alert['message']}")
                
        # Statistics
        if len(self.metrics_history['response_times']) > 10:
            print("\n📈 STATISTICS (Last 100 samples)")
            print("-"*40)
            
            # Response time stats
            recent_response_times = list(self.metrics_history['response_times'])[-20:]
            if recent_response_times:
                avg_response = sum(recent_response_times) / len(recent_response_times)
                max_response = max(recent_response_times)
                min_response = min(recent_response_times)
                print(f"  Response Time:     Avg: {avg_response:.0f}ms | Max: {max_response:.0f}ms | Min: {min_response:.0f}ms")
                
            # CPU stats
            recent_cpu = list(self.metrics_history['cpu_usage'])[-20:]
            if recent_cpu:
                avg_cpu = sum(recent_cpu) / len(recent_cpu)
                max_cpu = max(recent_cpu)
                print(f"  CPU Usage:         Avg: {avg_cpu:.1f}% | Max: {max_cpu:.1f}%")
                
            # Memory stats
            recent_memory = list(self.metrics_history['memory_usage'])[-20:]
            if recent_memory:
                avg_memory = sum(recent_memory) / len(recent_memory)
                max_memory = max(recent_memory)
                print(f"  Memory Usage:      Avg: {avg_memory:.1f}% | Max: {max_memory:.1f}%")
                
        print("\n" + "="*80)
        print("Press Ctrl+C to stop monitoring")
        
    def _format_bar(self, value: float, max_value: float, width: int = 20) -> str:
        """Format a progress bar"""
        filled = int((value / max_value) * width)
        bar = "█" * filled + "░" * (width - filled)
        
        # Color based on value
        if value > 80:
            return f"\033[91m{bar}\033[0m"  # Red
        elif value > 60:
            return f"\033[93m{bar}\033[0m"  # Yellow
        else:
            return f"\033[92m{bar}\033[0m"  # Green
            
    def _format_response_time(self, time_ms: float) -> str:
        """Format response time with color"""
        if time_ms < 0:
            return "\033[91m[ERROR]\033[0m"
        elif time_ms > 500:
            return "\033[91m[SLOW]\033[0m"  # Red
        elif time_ms > 200:
            return "\033[93m[WARN]\033[0m"  # Yellow
        else:
            return "\033[92m[FAST]\033[0m"  # Green
            
    def export_metrics(self, filename: str = "performance_metrics.json"):
        """Export metrics history to file"""
        
        export_data = {
            'export_time': datetime.now().isoformat(),
            'metrics': {
                'timestamps': [t.isoformat() for t in self.metrics_history['timestamps']],
                'cpu_usage': list(self.metrics_history['cpu_usage']),
                'memory_usage': list(self.metrics_history['memory_usage']),
                'response_times': list(self.metrics_history['response_times']),
                'cache_hit_rate': list(self.metrics_history['cache_hit_rate']),
                'active_connections': list(self.metrics_history['active_connections']),
                'requests_per_second': list(self.metrics_history['requests_per_second']),
                'error_rate': list(self.metrics_history['error_rate'])
            },
            'alerts': [
                {
                    'timestamp': a['timestamp'].isoformat(),
                    'message': a['message']
                } for a in self.alerts
            ],
            'statistics': self.calculate_statistics()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        logger.info(f"Metrics exported to {filename}")
        
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate overall statistics"""
        
        stats = {}
        
        # Response time statistics
        if self.metrics_history['response_times']:
            response_times = [t for t in self.metrics_history['response_times'] if t > 0]
            if response_times:
                stats['response_time'] = {
                    'avg': sum(response_times) / len(response_times),
                    'max': max(response_times),
                    'min': min(response_times),
                    'samples': len(response_times)
                }
                
        # CPU statistics
        if self.metrics_history['cpu_usage']:
            cpu_usage = list(self.metrics_history['cpu_usage'])
            stats['cpu'] = {
                'avg': sum(cpu_usage) / len(cpu_usage),
                'max': max(cpu_usage),
                'min': min(cpu_usage),
                'samples': len(cpu_usage)
            }
            
        # Memory statistics
        if self.metrics_history['memory_usage']:
            memory_usage = list(self.metrics_history['memory_usage'])
            stats['memory'] = {
                'avg': sum(memory_usage) / len(memory_usage),
                'max': max(memory_usage),
                'min': min(memory_usage),
                'samples': len(memory_usage)
            }
            
        # Cache statistics
        if self.metrics_history['cache_hit_rate']:
            cache_rates = list(self.metrics_history['cache_hit_rate'])
            stats['cache'] = {
                'avg_hit_rate': sum(cache_rates) / len(cache_rates) if cache_rates else 0,
                'max_hit_rate': max(cache_rates) if cache_rates else 0,
                'min_hit_rate': min(cache_rates) if cache_rates else 0
            }
            
        return stats
        
    async def stop_monitoring(self):
        """Stop monitoring"""
        self._monitoring = False
        logger.info("Monitoring stopped")


async def main():
    """Main monitoring function"""
    
    monitor = PerformanceMonitor()
    
    try:
        # Start monitoring
        await monitor.start_monitoring(interval=5)
        
    except KeyboardInterrupt:
        print("\n\nStopping monitor...")
        await monitor.stop_monitoring()
        
        # Export metrics
        monitor.export_metrics()
        
        # Show final statistics
        stats = monitor.calculate_statistics()
        print("\n📊 FINAL STATISTICS")
        print("-"*40)
        
        if 'response_time' in stats:
            print(f"Response Time:")
            print(f"  Average: {stats['response_time']['avg']:.0f}ms")
            print(f"  Max: {stats['response_time']['max']:.0f}ms")
            print(f"  Min: {stats['response_time']['min']:.0f}ms")
            
        if 'cpu' in stats:
            print(f"\nCPU Usage:")
            print(f"  Average: {stats['cpu']['avg']:.1f}%")
            print(f"  Max: {stats['cpu']['max']:.1f}%")
            
        if 'memory' in stats:
            print(f"\nMemory Usage:")
            print(f"  Average: {stats['memory']['avg']:.1f}%")
            print(f"  Max: {stats['memory']['max']:.1f}%")
            
        print(f"\nTotal Alerts: {len(monitor.alerts)}")
        print(f"Metrics exported to: performance_metrics.json")


if __name__ == "__main__":
    asyncio.run(main())