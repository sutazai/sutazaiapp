"""
Edge Inference Telemetry - Comprehensive performance monitoring and telemetry system
"""

import asyncio
import time
import threading
import logging
import json
import sqlite3
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from app.schemas.message_types import AlertSeverity
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import psutil
from pathlib import Path
import weakref
import aiofiles

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"        # Monotonically increasing
    GAUGE = "gauge"           # Point-in-time value
    HISTOGRAM = "histogram"   # Distribution of values
    TIMER = "timer"          # Timing measurements
    RATE = "rate"            # Rate of change


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: Union[float, int]
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None

@dataclass
class Alert:
    """System alert"""
    alert_id: str
    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    node_id: Optional[str] = None
    metric_name: Optional[str] = None
    threshold_value: Optional[float] = None
    current_value: Optional[float] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class PerformanceSummary:
    """Performance summary statistics"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    error_rate: float
    uptime_percentage: float
    time_period: str

class MetricsCollector:
    """Collects and aggregates metrics"""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.RLock()
    
    def record_metric(self, metric: Metric) -> None:
        """Record a metric"""
        with self._lock:
            metric_key = self._get_metric_key(metric.name, metric.labels)
            
            if metric.metric_type == MetricType.COUNTER:
                self.counters[metric_key] += metric.value
            elif metric.metric_type == MetricType.GAUGE:
                self.gauges[metric_key] = metric.value
            elif metric.metric_type == MetricType.HISTOGRAM:
                self.histograms[metric_key].append(metric.value)
                # Limit histogram size
                if len(self.histograms[metric_key]) > 10000:
                    self.histograms[metric_key] = self.histograms[metric_key][-5000:]
            elif metric.metric_type == MetricType.TIMER:
                self.timers[metric_key].append(metric.value)
            
            # Always store in time series
            self.metrics[metric_key].append((metric.timestamp, metric.value))
    
    def _get_metric_key(self, name: str, labels: Dict[str, str]) -> str:
        """Generate unique key for metric"""
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name
    
    def get_counter(self, name: str, labels: Dict[str, str] = None) -> float:
        """Get counter value"""
        key = self._get_metric_key(name, labels or {})
        return self.counters.get(key, 0.0)
    
    def get_gauge(self, name: str, labels: Dict[str, str] = None) -> float:
        """Get gauge value"""
        key = self._get_metric_key(name, labels or {})
        return self.gauges.get(key, 0.0)
    
    def get_histogram_stats(self, name: str, labels: Dict[str, str] = None) -> Dict[str, float]:
        """Get histogram statistics"""
        key = self._get_metric_key(name, labels or {})
        values = self.histograms.get(key, [])
        
        if not values:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
        
        values_array = np.array(values)
        return {
            "count": len(values),
            "mean": np.mean(values_array),
            "p50": np.percentile(values_array, 50),
            "p95": np.percentile(values_array, 95),
            "p99": np.percentile(values_array, 99),
            "min": np.min(values_array),
            "max": np.max(values_array)
        }
    
    def get_timer_stats(self, name: str, labels: Dict[str, str] = None) -> Dict[str, float]:
        """Get timer statistics"""
        key = self._get_metric_key(name, labels or {})
        values = list(self.timers.get(key, []))
        
        if not values:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
        
        values_array = np.array(values)
        return {
            "count": len(values),
            "mean": np.mean(values_array),
            "p50": np.percentile(values_array, 50),
            "p95": np.percentile(values_array, 95),
            "p99": np.percentile(values_array, 99)
        }
    
    def get_time_series(self, 
                       name: str, 
                       labels: Dict[str, str] = None,
                       start_time: datetime = None,
                       end_time: datetime = None) -> List[Tuple[datetime, float]]:
        """Get time series data"""
        key = self._get_metric_key(name, labels or {})
        data = list(self.metrics.get(key, []))
        
        if start_time or end_time:
            filtered_data = []
            for timestamp, value in data:
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue
                filtered_data.append((timestamp, value))
            return filtered_data
        
        return data
    
    def cleanup_old_metrics(self) -> None:
        """Clean up old metrics beyond retention"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        with self._lock:
            for metric_key in list(self.metrics.keys()):
                # Filter out old entries
                filtered_data = deque([
                    (timestamp, value) for timestamp, value in self.metrics[metric_key]
                    if timestamp > cutoff_time
                ], maxlen=10000)
                self.metrics[metric_key] = filtered_data
            
            # Clean up histograms (keep last N entries regardless of time)
            for key in self.histograms:
                if len(self.histograms[key]) > 5000:
                    self.histograms[key] = self.histograms[key][-2500:]

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_rules: List[Callable] = []
        self.subscribers: List[Callable] = []
        self._lock = threading.RLock()
    
    def add_alert_rule(self, rule: Callable[[MetricsCollector], List[Alert]]) -> None:
        """Add an alert rule"""
        self.alert_rules.append(rule)
    
    def subscribe_to_alerts(self, callback: Callable[[Alert], None]) -> None:
        """Subscribe to alert notifications"""
        self.subscribers.append(callback)
    
    def evaluate_alerts(self, metrics: MetricsCollector) -> List[Alert]:
        """Evaluate all alert rules"""
        new_alerts = []
        
        for rule in self.alert_rules:
            try:
                alerts = rule(metrics)
                for alert in alerts:
                    if self._should_fire_alert(alert):
                        new_alerts.append(alert)
                        self._fire_alert(alert)
            except Exception as e:
                logger.error(f"Alert rule evaluation failed: {e}")
        
        return new_alerts
    
    def _should_fire_alert(self, alert: Alert) -> bool:
        """Check if alert should be fired"""
        with self._lock:
            # Check if similar alert is already active
            for active_alert in self.active_alerts.values():
                if (active_alert.name == alert.name and 
                    active_alert.node_id == alert.node_id and
                    not active_alert.resolved):
                    return False
            return True
    
    def _fire_alert(self, alert: Alert) -> None:
        """Fire an alert"""
        with self._lock:
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            
            # Notify subscribers
            for subscriber in self.subscribers:
                try:
                    subscriber(alert)
                except Exception as e:
                    logger.error(f"Alert notification failed: {e}")
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                del self.active_alerts[alert_id]
                return True
            return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        with self._lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get recent alert history"""
        with self._lock:
            return list(self.alert_history)[-limit:]

class SystemMonitor:
    """Monitors system-level metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self, interval: float = 5.0) -> None:
        """Start system monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop(interval))
        logger.info("System monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop system monitoring"""
        self._monitoring = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("System monitoring stopped")
    
    async def _monitoring_loop(self, interval: float) -> None:
        """System monitoring loop"""
        while self._monitoring:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(interval * 2)
    
    async def _collect_system_metrics(self) -> None:
        """Collect system metrics"""
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        self.metrics_collector.record_metric(Metric(
            name="system_cpu_usage",
            value=cpu_percent,
            metric_type=MetricType.GAUGE,
            timestamp=timestamp,
            unit="percent"
        ))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.metrics_collector.record_metric(Metric(
            name="system_memory_usage",
            value=memory.percent,
            metric_type=MetricType.GAUGE,
            timestamp=timestamp,
            unit="percent"
        ))
        
        self.metrics_collector.record_metric(Metric(
            name="system_memory_available",
            value=memory.available / (1024 * 1024 * 1024),  # GB
            metric_type=MetricType.GAUGE,
            timestamp=timestamp,
            unit="GB"
        ))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.metrics_collector.record_metric(Metric(
            name="system_disk_usage",
            value=(disk.used / disk.total) * 100,
            metric_type=MetricType.GAUGE,
            timestamp=timestamp,
            unit="percent"
        ))
        
        # Network metrics
        network = psutil.net_io_counters()
        self.metrics_collector.record_metric(Metric(
            name="system_network_bytes_sent",
            value=network.bytes_sent,
            metric_type=MetricType.COUNTER,
            timestamp=timestamp,
            unit="bytes"
        ))
        
        self.metrics_collector.record_metric(Metric(
            name="system_network_bytes_recv",
            value=network.bytes_recv,
            metric_type=MetricType.COUNTER,
            timestamp=timestamp,
            unit="bytes"
        ))
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        
        self.metrics_collector.record_metric(Metric(
            name="process_memory_rss",
            value=process_memory.rss / (1024 * 1024),  # MB
            metric_type=MetricType.GAUGE,
            timestamp=timestamp,
            unit="MB"
        ))
        
        self.metrics_collector.record_metric(Metric(
            name="process_cpu_percent",
            value=process.cpu_percent(),
            metric_type=MetricType.GAUGE,
            timestamp=timestamp,
            unit="percent"
        ))

class InferenceMetricsTracker:
    """Tracks inference-specific metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_requests: Dict[str, datetime] = {}
        self._lock = threading.RLock()
    
    def start_request(self, request_id: str, model_name: str, node_id: str) -> None:
        """Record start of an inference request"""
        with self._lock:
            self.active_requests[request_id] = datetime.now()
            
            # Increment request counter
            self.metrics_collector.record_metric(Metric(
                name="inference_requests_total",
                value=1,
                metric_type=MetricType.COUNTER,
                timestamp=datetime.now(),
                labels={"model": model_name, "node": node_id}
            ))
            
            # Update active requests gauge
            self.metrics_collector.record_metric(Metric(
                name="inference_requests_active",
                value=len(self.active_requests),
                metric_type=MetricType.GAUGE,
                timestamp=datetime.now()
            ))
    
    def complete_request(self, 
                        request_id: str, 
                        model_name: str, 
                        node_id: str,
                        success: bool,
                        tokens_generated: int = 0) -> None:
        """Record completion of an inference request"""
        with self._lock:
            start_time = self.active_requests.pop(request_id, None)
            if not start_time:
                logger.warning(f"Request {request_id} not found in active requests")
                return
            
            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            # Record latency
            self.metrics_collector.record_metric(Metric(
                name="inference_latency_ms",
                value=latency_ms,
                metric_type=MetricType.HISTOGRAM,
                timestamp=end_time,
                labels={"model": model_name, "node": node_id},
                unit="ms"
            ))
            
            # Record success/failure
            status_label = "success" if success else "error"
            self.metrics_collector.record_metric(Metric(
                name="inference_requests_completed",
                value=1,
                metric_type=MetricType.COUNTER,
                timestamp=end_time,
                labels={"model": model_name, "node": node_id, "status": status_label}
            ))
            
            if tokens_generated > 0:
                # Record tokens generated
                self.metrics_collector.record_metric(Metric(
                    name="inference_tokens_generated",
                    value=tokens_generated,
                    metric_type=MetricType.COUNTER,
                    timestamp=end_time,
                    labels={"model": model_name, "node": node_id}
                ))
                
                # Calculate tokens per second
                tokens_per_second = tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0
                self.metrics_collector.record_metric(Metric(
                    name="inference_tokens_per_second",
                    value=tokens_per_second,
                    metric_type=MetricType.HISTOGRAM,
                    timestamp=end_time,
                    labels={"model": model_name, "node": node_id},
                    unit="tokens/s"
                ))
            
            # Update active requests gauge
            self.metrics_collector.record_metric(Metric(
                name="inference_requests_active",
                value=len(self.active_requests),
                metric_type=MetricType.GAUGE,
                timestamp=end_time
            ))

class TelemetryDatabase:
    """SQLite database for persistent telemetry storage"""
    
    def __init__(self, db_path: str = "/tmp/edge_inference_telemetry.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    labels TEXT,
                    unit TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    node_id TEXT,
                    metric_name TEXT,
                    threshold_value REAL,
                    current_value REAL,
                    resolved INTEGER DEFAULT 0,
                    resolved_at TEXT
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON metrics(name, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
    
    async def store_metrics(self, metrics: List[Metric]) -> None:
        """Store metrics in database"""
        if not metrics:
            return
        
        def _store():
            with sqlite3.connect(self.db_path) as conn:
                for metric in metrics:
                    conn.execute(
                        "INSERT INTO metrics (name, value, metric_type, timestamp, labels, unit) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            metric.name,
                            metric.value,
                            metric.metric_type.value,
                            metric.timestamp.isoformat(),
                            json.dumps(metric.labels) if metric.labels else None,
                            metric.unit
                        )
                    )
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _store)
    
    async def store_alert(self, alert: Alert) -> None:
        """Store alert in database"""
        def _store():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO alerts 
                    (alert_id, name, severity, message, timestamp, node_id, metric_name, threshold_value, current_value, resolved, resolved_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.alert_id,
                    alert.name,
                    alert.severity.value,
                    alert.message,
                    alert.timestamp.isoformat(),
                    alert.node_id,
                    alert.metric_name,
                    alert.threshold_value,
                    alert.current_value,
                    1 if alert.resolved else 0,
                    alert.resolved_at.isoformat() if alert.resolved_at else None
                ))
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _store)
    
    async def cleanup_old_data(self, days_to_keep: int = 7) -> None:
        """Clean up old data from database"""
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        
        def _cleanup():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff_date,))
                conn.execute("DELETE FROM alerts WHERE timestamp < ? AND resolved = 1", (cutoff_date,))
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _cleanup)

class EdgeTelemetrySystem:
    """Main telemetry system for edge inference"""
    
    def __init__(self,
                 enable_database: bool = True,
                 enable_system_monitoring: bool = True,
                 metrics_retention_hours: int = 24):
        
        # Core components
        self.metrics_collector = MetricsCollector(retention_hours=metrics_retention_hours)
        self.alert_manager = AlertManager()
        self.inference_tracker = InferenceMetricsTracker(self.metrics_collector)
        
        # Optional components
        self.system_monitor = SystemMonitor(self.metrics_collector) if enable_system_monitoring else None
        self.database = TelemetryDatabase() if enable_database else None
        
        # Background tasks
        self._running = False
        self._telemetry_task: Optional[asyncio.Task] = None
        self._alert_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
        logger.info("EdgeTelemetrySystem initialized")
    
    def _setup_default_alert_rules(self) -> None:
        """Setup default alert rules"""
        
        def high_cpu_alert(metrics: MetricsCollector) -> List[Alert]:
            cpu_usage = metrics.get_gauge("system_cpu_usage")
            if cpu_usage > 90:
                return [Alert(
                    alert_id=f"high_cpu_{int(time.time())}",
                    name="High CPU Usage",
                    severity=AlertSeverity.WARNING,
                    message=f"CPU usage is {cpu_usage:.1f}%",
                    timestamp=datetime.now(),
                    metric_name="system_cpu_usage",
                    threshold_value=90.0,
                    current_value=cpu_usage
                )]
            # Return empty list when CPU is within normal limits
            return []  # Valid empty list: CPU usage normal, no alerts
        
        def high_memory_alert(metrics: MetricsCollector) -> List[Alert]:
            memory_usage = metrics.get_gauge("system_memory_usage")
            if memory_usage > 95:
                return [Alert(
                    alert_id=f"high_memory_{int(time.time())}",
                    name="High Memory Usage",
                    severity=AlertSeverity.CRITICAL,
                    message=f"Memory usage is {memory_usage:.1f}%",
                    timestamp=datetime.now(),
                    metric_name="system_memory_usage",
                    threshold_value=95.0,
                    current_value=memory_usage
                )]
            # Return empty list when memory is within normal limits
            return []  # Valid empty list: Memory usage normal, no alerts
        
        def high_error_rate_alert(metrics: MetricsCollector) -> List[Alert]:
            # Calculate error rate from inference metrics
            total_requests = metrics.get_counter("inference_requests_total")
            failed_requests = metrics.get_counter("inference_requests_completed", {"status": "error"})
            
            if total_requests > 0:
                error_rate = failed_requests / total_requests
                if error_rate > 0.1:  # 10% error rate
                    return [Alert(
                        alert_id=f"high_error_rate_{int(time.time())}",
                        name="High Error Rate",
                        severity=AlertSeverity.ERROR,
                        message=f"Error rate is {error_rate*100:.1f}%",
                        timestamp=datetime.now(),
                        metric_name="inference_error_rate",
                        threshold_value=0.1,
                        current_value=error_rate
                    )]
            # Return empty list when error rate is within normal limits
            return []  # Valid empty list: Error rate normal, no alerts
        
        self.alert_manager.add_alert_rule(high_cpu_alert)
        self.alert_manager.add_alert_rule(high_memory_alert)
        self.alert_manager.add_alert_rule(high_error_rate_alert)
    
    async def start(self) -> None:
        """Start telemetry system"""
        if self._running:
            return
        
        self._running = True
        
        # Start system monitoring
        if self.system_monitor:
            await self.system_monitor.start_monitoring()
        
        # Start background tasks
        self._telemetry_task = asyncio.create_task(self._telemetry_loop())
        self._alert_task = asyncio.create_task(self._alert_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("EdgeTelemetrySystem started")
    
    async def stop(self) -> None:
        """Stop telemetry system"""
        self._running = False
        
        # Stop system monitoring
        if self.system_monitor:
            await self.system_monitor.stop_monitoring()
        
        # Cancel background tasks
        for task in [self._telemetry_task, self._alert_task, self._cleanup_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("EdgeTelemetrySystem stopped")
    
    async def _telemetry_loop(self) -> None:
        """Main telemetry processing loop"""
        while self._running:
            try:
                # Collect and store metrics if database is enabled
                if self.database:
                    # For now, we don't batch store metrics to avoid complexity
                    # In production, you'd batch collect recent metrics and store them
                    pass
                
                await asyncio.sleep(60)  # Every minute
                
            except Exception as e:
                logger.error(f"Telemetry loop error: {e}")
                await asyncio.sleep(60)
    
    async def _alert_loop(self) -> None:
        """Alert evaluation loop"""
        while self._running:
            try:
                # Evaluate alert rules
                alerts = self.alert_manager.evaluate_alerts(self.metrics_collector)
                
                # Store alerts in database
                if self.database:
                    for alert in alerts:
                        await self.database.store_alert(alert)
                
                await asyncio.sleep(30)  # Every 30 seconds
                
            except Exception as e:
                logger.error(f"Alert loop error: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_loop(self) -> None:
        """Cleanup old data loop"""
        while self._running:
            try:
                # Clean up in-memory metrics
                self.metrics_collector.cleanup_old_metrics()
                
                # Clean up database
                if self.database:
                    await self.database.cleanup_old_data()
                
                await asyncio.sleep(3600)  # Every hour
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(3600)
    
    def record_inference_start(self, request_id: str, model_name: str, node_id: str) -> None:
        """Record start of inference request"""
        self.inference_tracker.start_request(request_id, model_name, node_id)
    
    def record_inference_completion(self,
                                  request_id: str,
                                  model_name: str,
                                  node_id: str,
                                  success: bool,
                                  tokens_generated: int = 0) -> None:
        """Record completion of inference request"""
        self.inference_tracker.complete_request(request_id, model_name, node_id, success, tokens_generated)
    
    def record_custom_metric(self, metric: Metric) -> None:
        """Record a custom metric"""
        self.metrics_collector.record_metric(metric)
    
    def get_performance_summary(self, time_window_minutes: int = 60) -> PerformanceSummary:
        """Get performance summary for time window"""
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=time_window_minutes)
        
        # Get metrics for time window
        total_requests = self.metrics_collector.get_counter("inference_requests_total")
        success_requests = self.metrics_collector.get_counter("inference_requests_completed", {"status": "success"})
        failed_requests = self.metrics_collector.get_counter("inference_requests_completed", {"status": "error"})
        
        # Get latency statistics
        latency_stats = self.metrics_collector.get_histogram_stats("inference_latency_ms")
        
        # Calculate throughput
        throughput_rps = total_requests / (time_window_minutes * 60) if time_window_minutes > 0 else 0
        
        # Calculate error rate
        error_rate = failed_requests / max(total_requests, 1)
        
        return PerformanceSummary(
            total_requests=int(total_requests),
            successful_requests=int(success_requests),
            failed_requests=int(failed_requests),
            avg_latency_ms=latency_stats.get("mean", 0.0),
            p50_latency_ms=latency_stats.get("p50", 0.0),
            p95_latency_ms=latency_stats.get("p95", 0.0),
            p99_latency_ms=latency_stats.get("p99", 0.0),
            throughput_rps=throughput_rps,
            error_rate=error_rate,
            uptime_percentage=99.9,  # Would calculate from actual uptime data
            time_period=f"{time_window_minutes} minutes"
        )
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return {
            "cpu_usage": self.metrics_collector.get_gauge("system_cpu_usage"),
            "memory_usage": self.metrics_collector.get_gauge("system_memory_usage"),
            "memory_available_gb": self.metrics_collector.get_gauge("system_memory_available"),
            "disk_usage": self.metrics_collector.get_gauge("system_disk_usage"),
            "process_memory_mb": self.metrics_collector.get_gauge("process_memory_rss"),
            "process_cpu_percent": self.metrics_collector.get_gauge("process_cpu_percent")
        }
    
    def get_inference_metrics(self) -> Dict[str, Any]:
        """Get inference-specific metrics"""
        latency_stats = self.metrics_collector.get_histogram_stats("inference_latency_ms")
        tokens_stats = self.metrics_collector.get_histogram_stats("inference_tokens_per_second")
        
        return {
            "total_requests": self.metrics_collector.get_counter("inference_requests_total"),
            "active_requests": self.metrics_collector.get_gauge("inference_requests_active"),
            "successful_requests": self.metrics_collector.get_counter("inference_requests_completed", {"status": "success"}),
            "failed_requests": self.metrics_collector.get_counter("inference_requests_completed", {"status": "error"}),
            "tokens_generated": self.metrics_collector.get_counter("inference_tokens_generated"),
            "latency_stats": latency_stats,
            "tokens_per_second_stats": tokens_stats
        }
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts"""
        return self.alert_manager.get_active_alerts()
    
    def subscribe_to_alerts(self, callback: Callable[[Alert], None]) -> None:
        """Subscribe to alert notifications"""
        self.alert_manager.subscribe_to_alerts(callback)

# Global telemetry system instance
_global_telemetry: Optional[EdgeTelemetrySystem] = None

def get_global_telemetry(**kwargs) -> EdgeTelemetrySystem:
    """Get or create global telemetry system instance"""
    global _global_telemetry
    if _global_telemetry is None:
        _global_telemetry = EdgeTelemetrySystem(**kwargs)
    return _global_telemetry
