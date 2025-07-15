"""
Storage Monitoring and Analytics for SutazAI
Comprehensive monitoring of storage systems
"""

import asyncio
import psutil
import logging
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from collections import deque
import threading

logger = logging.getLogger(__name__)

@dataclass
class StorageMetrics:
    """Storage metrics data structure"""
    timestamp: float
    total_space: int
    used_space: int
    free_space: int
    usage_percent: float
    inode_usage: float
    read_operations: int
    write_operations: int
    read_bytes: int
    write_bytes: int

class StorageMonitor:
    """Comprehensive storage monitoring system"""
    
    def __init__(self, monitored_paths: List[str] = None):
        if monitored_paths is None:
            monitored_paths = ["/opt/sutazaiapp"]
        
        self.monitored_paths = [Path(path) for path in monitored_paths]
        self.metrics_history = deque(maxlen=10000)
        self.alerts = []
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Thresholds for alerts
        self.thresholds = {
            "disk_usage_warning": 80.0,  # %
            "disk_usage_critical": 90.0,  # %
            "inode_usage_warning": 80.0,  # %
            "inode_usage_critical": 90.0,  # %
            "io_latency_warning": 100.0,  # ms
            "io_latency_critical": 500.0  # ms
        }
    
    async def initialize(self):
        """Initialize storage monitor"""
        logger.info("🔄 Initializing Storage Monitor")
        
        # Start monitoring loop
        self.monitoring_active = True
        asyncio.create_task(self._monitoring_loop())
        
        # Start alert cleanup task
        asyncio.create_task(self._alert_cleanup_task())
        
        logger.info("✅ Storage monitor initialized")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect metrics for each monitored path
                for path in self.monitored_paths:
                    if path.exists():
                        metrics = await self._collect_path_metrics(path)
                        self.metrics_history.append(metrics)
                        
                        # Check for alerts
                        await self._check_alerts(metrics)
                
                await asyncio.sleep(30)  # Collect metrics every 30 seconds
                
            except Exception as e:
                logger.error(f"Storage monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _collect_path_metrics(self, path: Path) -> StorageMetrics:
        """Collect storage metrics for a path"""
        try:
            # Get disk usage
            disk_usage = psutil.disk_usage(str(path))
            usage_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Get I/O statistics
            io_stats = psutil.disk_io_counters(perdisk=False)
            
            # Try to get inode usage (Linux only)
            inode_usage = 0.0
            try:
                statvfs = os.statvfs(str(path))
                inode_usage = ((statvfs.f_files - statvfs.f_favail) / statvfs.f_files) * 100
            except (AttributeError, OSError):
                pass
            
            return StorageMetrics(
                timestamp=time.time(),
                total_space=disk_usage.total,
                used_space=disk_usage.used,
                free_space=disk_usage.free,
                usage_percent=usage_percent,
                inode_usage=inode_usage,
                read_operations=io_stats.read_count if io_stats else 0,
                write_operations=io_stats.write_count if io_stats else 0,
                read_bytes=io_stats.read_bytes if io_stats else 0,
                write_bytes=io_stats.write_bytes if io_stats else 0
            )
            
        except Exception as e:
            logger.error(f"Failed to collect metrics for {path}: {e}")
            # Return empty metrics
            return StorageMetrics(
                timestamp=time.time(),
                total_space=0, used_space=0, free_space=0,
                usage_percent=0.0, inode_usage=0.0,
                read_operations=0, write_operations=0,
                read_bytes=0, write_bytes=0
            )
    
    async def _check_alerts(self, metrics: StorageMetrics):
        """Check for storage alerts"""
        alerts_triggered = []
        
        # Disk usage alerts
        if metrics.usage_percent > self.thresholds["disk_usage_critical"]:
            alerts_triggered.append({
                "type": "disk_usage",
                "severity": "critical",
                "message": f"Critical disk usage: {metrics.usage_percent:.1f}%",
                "value": metrics.usage_percent,
                "threshold": self.thresholds["disk_usage_critical"]
            })
        elif metrics.usage_percent > self.thresholds["disk_usage_warning"]:
            alerts_triggered.append({
                "type": "disk_usage",
                "severity": "warning",
                "message": f"High disk usage: {metrics.usage_percent:.1f}%",
                "value": metrics.usage_percent,
                "threshold": self.thresholds["disk_usage_warning"]
            })
        
        # Inode usage alerts
        if metrics.inode_usage > self.thresholds["inode_usage_critical"]:
            alerts_triggered.append({
                "type": "inode_usage",
                "severity": "critical",
                "message": f"Critical inode usage: {metrics.inode_usage:.1f}%",
                "value": metrics.inode_usage,
                "threshold": self.thresholds["inode_usage_critical"]
            })
        elif metrics.inode_usage > self.thresholds["inode_usage_warning"]:
            alerts_triggered.append({
                "type": "inode_usage",
                "severity": "warning",
                "message": f"High inode usage: {metrics.inode_usage:.1f}%",
                "value": metrics.inode_usage,
                "threshold": self.thresholds["inode_usage_warning"]
            })
        
        # Add alerts with timestamp
        for alert in alerts_triggered:
            alert["timestamp"] = metrics.timestamp
            self.alerts.append(alert)
            logger.warning(f"Storage alert: {alert['message']}")
    
    async def _alert_cleanup_task(self):
        """Clean up old alerts"""
        while self.monitoring_active:
            try:
                cutoff_time = time.time() - 3600  # Remove alerts older than 1 hour
                self.alerts = [
                    alert for alert in self.alerts
                    if alert["timestamp"] > cutoff_time
                ]
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Alert cleanup error: {e}")
                await asyncio.sleep(300)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current storage metrics"""
        if not self.metrics_history:
            return {}
        
        latest_metrics = self.metrics_history[-1]
        
        return {
            "timestamp": latest_metrics.timestamp,
            "total_space_gb": latest_metrics.total_space / (1024 ** 3),
            "used_space_gb": latest_metrics.used_space / (1024 ** 3),
            "free_space_gb": latest_metrics.free_space / (1024 ** 3),
            "usage_percent": latest_metrics.usage_percent,
            "inode_usage_percent": latest_metrics.inode_usage,
            "io_operations": {
                "reads": latest_metrics.read_operations,
                "writes": latest_metrics.write_operations,
                "read_bytes": latest_metrics.read_bytes,
                "write_bytes": latest_metrics.write_bytes
            }
        }
    
    def get_historical_metrics(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical metrics"""
        cutoff_time = time.time() - (hours * 3600)
        
        historical_data = []
        for metrics in self.metrics_history:
            if metrics.timestamp > cutoff_time:
                historical_data.append({
                    "timestamp": metrics.timestamp,
                    "usage_percent": metrics.usage_percent,
                    "inode_usage": metrics.inode_usage,
                    "read_operations": metrics.read_operations,
                    "write_operations": metrics.write_operations
                })
        
        return historical_data
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active storage alerts"""
        return self.alerts[-50:]  # Return last 50 alerts
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """Get comprehensive storage summary"""
        current_metrics = self.get_current_metrics()
        active_alerts = self.get_active_alerts()
        
        # Calculate trends
        trend_data = self.get_historical_metrics(24)
        
        usage_trend = "stable"
        if len(trend_data) >= 2:
            first_usage = trend_data[0]["usage_percent"]
            last_usage = trend_data[-1]["usage_percent"]
            
            if last_usage > first_usage + 5:
                usage_trend = "increasing"
            elif last_usage < first_usage - 5:
                usage_trend = "decreasing"
        
        return {
            "current_metrics": current_metrics,
            "active_alerts": len([a for a in active_alerts if a["severity"] == "critical"]),
            "warning_alerts": len([a for a in active_alerts if a["severity"] == "warning"]),
            "usage_trend": usage_trend,
            "monitoring_status": "active" if self.monitoring_active else "inactive",
            "metrics_collected": len(self.metrics_history),
            "monitored_paths": [str(path) for path in self.monitored_paths]
        }
    
    def stop_monitoring(self):
        """Stop storage monitoring"""
        self.monitoring_active = False
        logger.info("🛑 Storage monitoring stopped")

# Global storage monitor instance
storage_monitor = StorageMonitor()
