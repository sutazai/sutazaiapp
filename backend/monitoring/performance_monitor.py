"""
Comprehensive Performance Monitoring for SutazAI
Real-time performance tracking and analytics
"""

import asyncio
import logging
import time
import psutil
import gc
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PerformanceSnapshot:
    """Performance metrics snapshot"""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read: int
    disk_io_write: int
    network_sent: int
    network_recv: int
    active_threads: int
    open_files: int
    gc_collections: int
    response_time_ms: float = 0.0
    
class PerformanceMonitor:
    """Comprehensive performance monitoring system"""
    
    def __init__(self, history_size: int = 10000):
        self.history_size = history_size
        self.snapshots = deque(maxlen=history_size)
        self.alerts = deque(maxlen=1000)
        
        # Performance thresholds
        self.thresholds = {
            "cpu_warning": 70.0,
            "cpu_critical": 90.0,
            "memory_warning": 80.0,
            "memory_critical": 95.0,
            "response_time_warning": 1000.0,  # ms
            "response_time_critical": 5000.0,  # ms
            "gc_frequency_warning": 10  # collections per minute
        }
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_task = None
        
        # Performance counters
        self.counters = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_error": 0,
            "avg_response_time": 0.0,
            "peak_memory": 0.0,
            "peak_cpu": 0.0
        }
        
        # Real-time metrics
        self.current_metrics = {}
        self.metrics_lock = threading.Lock()
    
    async def initialize(self):
        """Initialize performance monitoring"""
        logger.info("🔄 Initializing Performance Monitor")
        
        # Start monitoring loop
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        
        # Start alert processing
        asyncio.create_task(self._alert_processor())
        
        # Start metrics aggregation
        asyncio.create_task(self._metrics_aggregator())
        
        logger.info("✅ Performance Monitor initialized")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        last_io = None
        last_network = None
        last_gc_count = gc.get_count()[0]
        
        while self.monitoring_active:
            try:
                # Get current process
                process = psutil.Process()
                
                # CPU and Memory
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                memory_percent = process.memory_percent()
                
                # I/O statistics
                io_counters = process.io_counters()
                disk_read = io_counters.read_bytes if io_counters else 0
                disk_write = io_counters.write_bytes if io_counters else 0
                
                # Network statistics
                try:
                    net_io = psutil.net_io_counters()
                    net_sent = net_io.bytes_sent if net_io else 0
                    net_recv = net_io.bytes_recv if net_io else 0
                except Exception:
                    net_sent = net_recv = 0
                
                # System information
                active_threads = process.num_threads()
                try:
                    open_files = process.num_fds()
                except Exception:
                    open_files = 0
                
                # Garbage collection
                current_gc_count = gc.get_count()[0]
                gc_collections = current_gc_count - last_gc_count
                last_gc_count = current_gc_count
                
                # Create snapshot
                snapshot = PerformanceSnapshot(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_mb=memory_mb,
                    memory_percent=memory_percent,
                    disk_io_read=disk_read,
                    disk_io_write=disk_write,
                    network_sent=net_sent,
                    network_recv=net_recv,
                    active_threads=active_threads,
                    open_files=open_files,
                    gc_collections=gc_collections
                )
                
                # Store snapshot
                self.snapshots.append(snapshot)
                
                # Update current metrics
                with self.metrics_lock:
                    self.current_metrics = asdict(snapshot)
                
                # Update peak values
                self.counters["peak_memory"] = max(self.counters["peak_memory"], memory_mb)
                self.counters["peak_cpu"] = max(self.counters["peak_cpu"], cpu_percent)
                
                # Check thresholds
                await self._check_thresholds(snapshot)
                
                # Sleep until next collection
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(1.0)
    
    async def _check_thresholds(self, snapshot: PerformanceSnapshot):
        """Check performance thresholds and create alerts"""
        alerts = []
        
        # CPU alerts
        if snapshot.cpu_percent > self.thresholds["cpu_critical"]:
            alerts.append({
                "type": "cpu",
                "severity": "critical",
                "message": f"Critical CPU usage: {snapshot.cpu_percent:.1f}%",
                "value": snapshot.cpu_percent,
                "threshold": self.thresholds["cpu_critical"]
            })
        elif snapshot.cpu_percent > self.thresholds["cpu_warning"]:
            alerts.append({
                "type": "cpu",
                "severity": "warning",
                "message": f"High CPU usage: {snapshot.cpu_percent:.1f}%",
                "value": snapshot.cpu_percent,
                "threshold": self.thresholds["cpu_warning"]
            })
        
        # Memory alerts
        if snapshot.memory_percent > self.thresholds["memory_critical"]:
            alerts.append({
                "type": "memory",
                "severity": "critical",
                "message": f"Critical memory usage: {snapshot.memory_percent:.1f}%",
                "value": snapshot.memory_percent,
                "threshold": self.thresholds["memory_critical"]
            })
        elif snapshot.memory_percent > self.thresholds["memory_warning"]:
            alerts.append({
                "type": "memory",
                "severity": "warning",
                "message": f"High memory usage: {snapshot.memory_percent:.1f}%",
                "value": snapshot.memory_percent,
                "threshold": self.thresholds["memory_warning"]
            })
        
        # Response time alerts
        if hasattr(snapshot, 'response_time_ms') and snapshot.response_time_ms > 0:
            if snapshot.response_time_ms > self.thresholds["response_time_critical"]:
                alerts.append({
                    "type": "response_time",
                    "severity": "critical",
                    "message": f"Critical response time: {snapshot.response_time_ms:.1f}ms",
                    "value": snapshot.response_time_ms,
                    "threshold": self.thresholds["response_time_critical"]
                })
            elif snapshot.response_time_ms > self.thresholds["response_time_warning"]:
                alerts.append({
                    "type": "response_time",
                    "severity": "warning",
                    "message": f"High response time: {snapshot.response_time_ms:.1f}ms",
                    "value": snapshot.response_time_ms,
                    "threshold": self.thresholds["response_time_warning"]
                })
        
        # Add alerts with timestamp
        for alert in alerts:
            alert["timestamp"] = snapshot.timestamp
            self.alerts.append(alert)
    
    async def _alert_processor(self):
        """Process and log alerts"""
        while self.monitoring_active:
            try:
                # Remove old alerts (older than 1 hour)
                cutoff_time = time.time() - 3600
                
                # Filter out old alerts
                recent_alerts = []
                for alert in self.alerts:
                    if alert["timestamp"] > cutoff_time:
                        recent_alerts.append(alert)
                
                self.alerts.clear()
                self.alerts.extend(recent_alerts)
                
                await asyncio.sleep(60)  # Process every minute
                
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_aggregator(self):
        """Aggregate metrics for reporting"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(300)  # Aggregate every 5 minutes
                
                if len(self.snapshots) < 10:
                    continue
                
                # Calculate aggregated metrics
                recent_snapshots = list(self.snapshots)[-300:]  # Last 5 minutes
                
                avg_cpu = sum(s.cpu_percent for s in recent_snapshots) / len(recent_snapshots)
                avg_memory = sum(s.memory_mb for s in recent_snapshots) / len(recent_snapshots)
                max_cpu = max(s.cpu_percent for s in recent_snapshots)
                max_memory = max(s.memory_mb for s in recent_snapshots)
                
                # Log aggregated metrics
                logger.info(f"Performance Summary - CPU: {avg_cpu:.1f}% (max: {max_cpu:.1f}%), "
                           f"Memory: {avg_memory:.1f}MB (max: {max_memory:.1f}MB)")
                
            except Exception as e:
                logger.error(f"Metrics aggregation error: {e}")
                await asyncio.sleep(300)
    
    def record_request(self, response_time_ms: float, success: bool = True):
        """Record a request for performance tracking"""
        with self.metrics_lock:
            self.counters["requests_total"] += 1
            
            if success:
                self.counters["requests_success"] += 1
            else:
                self.counters["requests_error"] += 1
            
            # Update average response time
            total_requests = self.counters["requests_total"]
            current_avg = self.counters["avg_response_time"]
            self.counters["avg_response_time"] = (
                (current_avg * (total_requests - 1) + response_time_ms) / total_requests
            )
            
            # Update current snapshot with response time
            if self.current_metrics:
                self.current_metrics["response_time_ms"] = response_time_ms
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        with self.metrics_lock:
            return {
                **self.current_metrics,
                **self.counters,
                "alerts_active": len([a for a in self.alerts if a["severity"] == "critical"]),
                "monitoring_active": self.monitoring_active
            }
    
    def get_historical_data(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical performance data"""
        cutoff_time = time.time() - (hours * 3600)
        
        historical_data = []
        for snapshot in self.snapshots:
            if snapshot.timestamp > cutoff_time:
                historical_data.append(asdict(snapshot))
        
        return historical_data
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.snapshots:
            return {"message": "No performance data available"}
        
        recent_snapshots = list(self.snapshots)[-100:]  # Last 100 data points
        
        # Calculate statistics
        cpu_values = [s.cpu_percent for s in recent_snapshots]
        memory_values = [s.memory_mb for s in recent_snapshots]
        
        active_alerts = [a for a in self.alerts if time.time() - a["timestamp"] < 300]
        
        return {
            "current": self.get_current_metrics(),
            "statistics": {
                "cpu": {
                    "current": cpu_values[-1] if cpu_values else 0,
                    "average": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                    "max": max(cpu_values) if cpu_values else 0,
                    "min": min(cpu_values) if cpu_values else 0
                },
                "memory": {
                    "current": memory_values[-1] if memory_values else 0,
                    "average": sum(memory_values) / len(memory_values) if memory_values else 0,
                    "max": max(memory_values) if memory_values else 0,
                    "min": min(memory_values) if memory_values else 0
                }
            },
            "alerts": {
                "active": len(active_alerts),
                "critical": len([a for a in active_alerts if a["severity"] == "critical"]),
                "warning": len([a for a in active_alerts if a["severity"] == "warning"]),
                "recent": active_alerts[-10:]
            },
            "counters": self.counters.copy(),
            "thresholds": self.thresholds.copy(),
            "data_points": len(self.snapshots)
        }
    
    async def export_metrics(self, filepath: str):
        """Export performance metrics to file"""
        try:
            data = {
                "export_timestamp": time.time(),
                "summary": self.get_performance_summary(),
                "historical_data": self.get_historical_data(24),
                "alerts": list(self.alerts)
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Performance metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    async def shutdown(self):
        """Shutdown performance monitoring"""
        logger.info("🛑 Shutting down Performance Monitor")
        
        self.monitoring_active = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
        
        # Export final metrics
        try:
            export_path = Path("/opt/sutazaiapp/logs/final_performance_metrics.json")
            await self.export_metrics(str(export_path))
        except Exception as e:
            logger.warning(f"Failed to export final metrics: {e}")
        
        logger.info("✅ Performance Monitor shutdown complete")

# Performance monitoring decorators
def monitor_performance(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        success = True
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            success = False
            raise
        finally:
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            performance_monitor.record_request(response_time, success)
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        success = True
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            success = False
            raise
        finally:
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            performance_monitor.record_request(response_time, success)
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

# Global performance monitor instance
performance_monitor = PerformanceMonitor()
