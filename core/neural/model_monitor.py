#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_monitor.py - Enterprise-grade model monitoring system for SutazAI

This module continuously monitors model performance, detects drift, and
triggers automatic retraining when needed for Dell PowerEdge R720 with E5-2640 CPUs.
"""

import os
import json
import time
import logging
import sqlite3
import threading
import datetime
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass
from threading import Thread

# Internal imports
try:
    from core.neural.model_downloader import ModelRegistry, ModelVersion

    MODEL_DOWNLOADER_AVAILABLE = True
except ImportError:
    MODEL_DOWNLOADER_AVAILABLE = False
    logging.warning(
        "model_downloader module not found, some features will be unavailable"
    )

try:
    from core.neural.llama_utils import get_optimized_model

    LLAMA_UTILS_AVAILABLE = True
except ImportError:
    LLAMA_UTILS_AVAILABLE = False
    logging.warning(
        "llama_utils module not found, Llama model monitoring will be unavailable"
    )

# Setup logging
logger = logging.getLogger("sutazai.model_monitor")

# Constants
DEFAULT_MONITOR_DB_PATH = os.path.join(
    os.path.expanduser("~"), ".cache", "sutazai", "model_monitoring.db"
)
MONITORING_INTERVAL_SEC = 3600  # Default to hourly monitoring
PERFORMANCE_THRESHOLD = 0.15  # 15% degradation threshold for automatic retraining
MAX_INFERENCE_TIME_INCREASE = 0.3  # 30% increase in inference time threshold


@dataclass
class ModelPerformanceMetrics:
    """Model performance metrics for monitoring"""

    model_id: str
    version: str
    timestamp: datetime.datetime
    accuracy: Optional[float] = None
    perplexity: Optional[float] = None
    inference_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    tokens_per_second: Optional[float] = None
    failed_requests: int = 0
    total_requests: int = 0
    avg_request_time_ms: Optional[float] = None
    drift_score: Optional[float] = None
    metadata: Dict[str, Any] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate of the model"""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests

    @property
    def is_healthy(self) -> bool:
        """Determine if the model is healthy based on metrics"""
        if self.success_rate < 0.95:  # Below 95% success rate
            return False
        if self.drift_score and self.drift_score > 0.5:  # Significant drift
            return False
        return True


class ModelMonitorDB:
    """Database for storing and retrieving model monitoring data"""

    def __init__(self, db_path: str = DEFAULT_MONITOR_DB_PATH):
        """Initialize the database connection"""
        self.db_path = db_path
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        """Create the database and tables if they don't exist"""
        db_dir = os.path.dirname(self.db_path)
        os.makedirs(db_dir, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY,
                model_id TEXT NOT NULL,
                version TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                accuracy REAL,
                perplexity REAL,
                inference_time_ms REAL,
                memory_usage_mb REAL,
                tokens_per_second REAL,
                failed_requests INTEGER,
                total_requests INTEGER,
                avg_request_time_ms REAL,
                drift_score REAL,
                metadata TEXT,
                UNIQUE(model_id, version, timestamp)
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS retraining_events (
                id INTEGER PRIMARY KEY,
                model_id TEXT NOT NULL,
                old_version TEXT NOT NULL,
                new_version TEXT,
                timestamp TEXT NOT NULL,
                reason TEXT NOT NULL,
                metrics_before TEXT,
                metrics_after TEXT,
                status TEXT NOT NULL,
                completed_at TEXT
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_alerts (
                id INTEGER PRIMARY KEY,
                model_id TEXT NOT NULL,
                version TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                severity TEXT NOT NULL,
                resolved INTEGER DEFAULT 0,
                resolved_at TEXT
            )
            """)

            conn.commit()

    def insert_metrics(self, metrics: ModelPerformanceMetrics) -> int:
        """Insert new performance metrics into the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
            INSERT OR REPLACE INTO performance_metrics
            (model_id, version, timestamp, accuracy, perplexity, inference_time_ms,
             memory_usage_mb, tokens_per_second, failed_requests, total_requests,
             avg_request_time_ms, drift_score, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metrics.model_id,
                    metrics.version,
                    metrics.timestamp.isoformat(),
                    metrics.accuracy,
                    metrics.perplexity,
                    metrics.inference_time_ms,
                    metrics.memory_usage_mb,
                    metrics.tokens_per_second,
                    metrics.failed_requests,
                    metrics.total_requests,
                    metrics.avg_request_time_ms,
                    metrics.drift_score,
                    json.dumps(metrics.metadata) if metrics.metadata else None,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_latest_metrics(
        self, model_id: str, version: Optional[str] = None
    ) -> Optional[ModelPerformanceMetrics]:
        """Get the latest metrics for a model"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if version:
                cursor.execute(
                    """
                SELECT * FROM performance_metrics
                WHERE model_id = ? AND version = ?
                ORDER BY timestamp DESC LIMIT 1
                """,
                    (model_id, version),
                )
            else:
                cursor.execute(
                    """
                SELECT * FROM performance_metrics
                WHERE model_id = ?
                ORDER BY timestamp DESC LIMIT 1
                """,
                    (model_id,),
                )

            row = cursor.fetchone()
            if row:
                return ModelPerformanceMetrics(
                    model_id=row["model_id"],
                    version=row["version"],
                    timestamp=datetime.datetime.fromisoformat(row["timestamp"]),
                    accuracy=row["accuracy"],
                    perplexity=row["perplexity"],
                    inference_time_ms=row["inference_time_ms"],
                    memory_usage_mb=row["memory_usage_mb"],
                    tokens_per_second=row["tokens_per_second"],
                    failed_requests=row["failed_requests"],
                    total_requests=row["total_requests"],
                    avg_request_time_ms=row["avg_request_time_ms"],
                    drift_score=row["drift_score"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else None,
                )
            return None

    def get_metrics_history(
        self, model_id: str, days: int = 7
    ) -> List[ModelPerformanceMetrics]:
        """Get metrics history for a model over the last N days"""
        cutoff_date = (
            datetime.datetime.now() - datetime.timedelta(days=days)
        ).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
            SELECT * FROM performance_metrics
            WHERE model_id = ? AND timestamp > ?
            ORDER BY timestamp ASC
            """,
                (model_id, cutoff_date),
            )

            return [
                ModelPerformanceMetrics(
                    model_id=row["model_id"],
                    version=row["version"],
                    timestamp=datetime.datetime.fromisoformat(row["timestamp"]),
                    accuracy=row["accuracy"],
                    perplexity=row["perplexity"],
                    inference_time_ms=row["inference_time_ms"],
                    memory_usage_mb=row["memory_usage_mb"],
                    tokens_per_second=row["tokens_per_second"],
                    failed_requests=row["failed_requests"],
                    total_requests=row["total_requests"],
                    avg_request_time_ms=row["avg_request_time_ms"],
                    drift_score=row["drift_score"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else None,
                )
                for row in cursor.fetchall()
            ]

    def record_retraining_event(
        self,
        model_id: str,
        old_version: str,
        reason: str,
        metrics_before: Optional[Dict] = None,
    ) -> int:
        """Record the start of a retraining event"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
            INSERT INTO retraining_events
            (model_id, old_version, timestamp, reason, metrics_before, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    model_id,
                    old_version,
                    datetime.datetime.now().isoformat(),
                    reason,
                    json.dumps(metrics_before) if metrics_before else None,
                    "STARTED",
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def update_retraining_event(
        self,
        event_id: int,
        new_version: Optional[str] = None,
        metrics_after: Optional[Dict] = None,
        status: str = "COMPLETED",
    ) -> bool:
        """Update a retraining event after completion"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            update_fields = []
            params = []

            if new_version:
                update_fields.append("new_version = ?")
                params.append(new_version)

            if metrics_after:
                update_fields.append("metrics_after = ?")
                params.append(json.dumps(metrics_after))

            update_fields.append("status = ?")
            params.append(status)

            update_fields.append("completed_at = ?")
            params.append(datetime.datetime.now().isoformat())

            params.append(event_id)

            cursor.execute(
                f"""
            UPDATE retraining_events
            SET {", ".join(update_fields)}
            WHERE id = ?
            """,
                params,
            )

            conn.commit()
            return cursor.rowcount > 0

    def add_alert(
        self,
        model_id: str,
        version: str,
        alert_type: str,
        message: str,
        severity: str = "WARNING",
    ) -> int:
        """Add an alert for a model"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
            INSERT INTO model_alerts
            (model_id, version, timestamp, alert_type, message, severity)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    model_id,
                    version,
                    datetime.datetime.now().isoformat(),
                    alert_type,
                    message,
                    severity,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def resolve_alert(self, alert_id: int) -> bool:
        """Mark an alert as resolved"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
            UPDATE model_alerts
            SET resolved = 1, resolved_at = ?
            WHERE id = ?
            """,
                (datetime.datetime.now().isoformat(), alert_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_active_alerts(self, model_id: Optional[str] = None) -> List[Dict]:
        """Get all active (unresolved) alerts"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if model_id:
                cursor.execute(
                    """
                SELECT * FROM model_alerts
                WHERE model_id = ? AND resolved = 0
                ORDER BY timestamp DESC
                """,
                    (model_id,),
                )
            else:
                cursor.execute("""
                SELECT * FROM model_alerts
                WHERE resolved = 0
                ORDER BY timestamp DESC
                """)

            return [dict(row) for row in cursor.fetchall()]


class ModelMonitor:
    """
    Enterprise-grade model performance monitor that tracks metrics,
    detects performance degradation, and triggers automatic retraining.

    This class provides continuous monitoring capabilities for ML models,
    specifically optimized for the E5-2640 CPU environment.
    """

    def __init__(
        self,
        model_registry: Optional[ModelRegistry] = None,
        monitor_db: Optional[ModelMonitorDB] = None,
        monitoring_interval: int = MONITORING_INTERVAL_SEC,
        performance_threshold: float = PERFORMANCE_THRESHOLD,
        auto_retrain: bool = False,
        alert_handlers: Optional[List[Callable]] = None,
    ):
        """Initialize the model monitor"""
        if not MODEL_DOWNLOADER_AVAILABLE:
            raise ImportError("model_downloader module is required for ModelMonitor")

        self.model_registry = model_registry or ModelRegistry()
        self.monitor_db = monitor_db or ModelMonitorDB()
        self.monitoring_interval = monitoring_interval
        self.performance_threshold = performance_threshold
        self.auto_retrain = auto_retrain
        self.alert_handlers = alert_handlers or []

        self._models_being_monitored: Set[str] = set()
        self._last_check_time: Dict[str, float] = {}
        self._monitoring_thread: Optional[Thread] = None
        self._stop_event = threading.Event()
        self._monitor_db = monitor_db

    def start_monitoring(self, model_ids: Optional[List[str]] = None):
        """Start the background monitoring thread for specified models."""
        if model_ids:
            for model_id in model_ids:
                self._models_being_monitored.add(model_id)
                self._last_check_time[model_id] = time.time()
            logger.info(f"Starting monitoring for models: {', '.join(model_ids)}")

        # Start monitoring thread if not already running
        if not self._monitoring_thread or not self._monitoring_thread.is_alive():
            self._stop_event.clear()
            self._monitoring_thread = Thread(target=self._monitor_loop, name="ModelMonitorThread")
            # Check if thread is not None before accessing attributes
            if self._monitoring_thread:
                self._monitoring_thread.daemon = True
                self._monitoring_thread.start()
                logger.info("Started background monitoring thread")
            else:
                logger.error("Failed to create monitoring thread.")
                return False
        return True

    def stop_monitoring(self):
        """Stop the background monitoring thread."""
        self._stop_event.set()
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)
            if self._monitoring_thread.is_alive():
                logger.warning("Monitoring thread did not stop gracefully.")
        self._monitoring_thread = None
        logger.info("Monitoring stopped.")
        return True

    def add_model_to_monitor(self, model_id: str):
        """Add a model to the monitoring list."""
        if model_id not in self._models_being_monitored:
            self._models_being_monitored.add(model_id)
            self._last_check_time[model_id] = time.time()
            logger.info(f"Added model {model_id} to monitoring.")

    def remove_model_from_monitor(self, model_id: str):
        """Remove a model from the monitoring list."""
        if model_id in self._models_being_monitored:
            self._models_being_monitored.discard(model_id)
            if model_id in self._last_check_time:
                del self._last_check_time[model_id]
            logger.info(f"Removed model {model_id} from monitoring.")

    def _monitor_loop(self):
        """Background loop to check model performance periodically."""
        logger.info(f"Monitoring loop started. Interval: {self.monitoring_interval}s")
        while not self._stop_event.is_set():
            current_time = time.time()
            models_to_check = list(self._models_being_monitored) # Create a copy

            for model_id in models_to_check:
                last_check = self._last_check_time.get(model_id, 0)
                if current_time - last_check >= self.monitoring_interval:
                    try:
                        logger.debug(f"Checking model {model_id} performance")
                        self._check_model_performance(model_id)
                        self._last_check_time[model_id] = current_time
                    except Exception as e:
                        logger.error(f"Error checking performance for {model_id}: {e}")

            # Sleep efficiently, checking stop event periodically
            self._stop_event.wait(self.monitoring_interval)
        logger.info("Monitoring loop stopped.")

    def _check_model_performance(self, model_id: str):
        """Check the performance of a specific model."""
        # Get latest metrics from the database
        metrics = self.monitor_db.get_latest_metrics(model_id)
        if not metrics:
            logger.warning(f"No metrics found for model {model_id}. Skipping check.")
            return

        # Simple check: compare latency/throughput to baseline or threshold
        # In a real system, this would be more sophisticated
        latency_threshold = 1000 # ms (example)
        throughput_threshold = 10 # tokens/sec (example)

        current_latency = metrics.get("latency_ms", float('inf'))
        current_throughput = metrics.get("throughput_tokens_sec", 0.0)

        alert_triggered = False
        alert_message = ""

        if current_latency > latency_threshold:
            alert_triggered = True
            alert_message += f"Latency high ({current_latency:.2f}ms > {latency_threshold}ms). "

        if current_throughput < throughput_threshold:
            alert_triggered = True
            alert_message += f"Throughput low ({current_throughput:.2f} t/s < {throughput_threshold} t/s)."

        if alert_triggered:
            logger.warning(f"Performance alert for {model_id}: {alert_message}")
            self._trigger_alert(model_id, "Performance Degradation", alert_message, metrics)

    def manually_evaluate_model(
        self, model_id: str, metrics_to_run: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Manually evaluate a model's performance. Placeholder implementation.
        """
        logger.info(f"Manually evaluating model {model_id}")
        # This should call actual benchmark functions
        evaluation_result = {
            "latency_ms": random.uniform(50, 150),
            "throughput": random.uniform(20, 60),
            "memory_mb": random.uniform(500, 2000),
            "cpu_percent": random.uniform(10, 90),
            "metadata": {"evaluation_type": "manual"},
        }

        # Log results
        logger.info(f"Manual evaluation for {model_id}: {evaluation_result}")

        # Add to monitoring database
        self.monitor_db.add_metric(
            ModelPerformanceMetrics(
                model_id=model_id,
                version="manual", # Use manual as version
                timestamp=datetime.now(),
                latency_ms=evaluation_result.get("latency_ms", 0.0),
                throughput_tokens_sec=evaluation_result.get("throughput", 0.0),
                memory_usage_mb=evaluation_result.get("memory_mb", 0.0),
                cpu_usage_percent=evaluation_result.get("cpu_percent", 0.0),
                metadata=evaluation_result.get("metadata") or {}, # Ensure metadata is dict
            )
        )

        return evaluation_result

    def _trigger_alert(self, model_id: str, alert_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Trigger an alert and call registered handlers."""
        alert_id = self.monitor_db.add_alert(
            model_id=model_id,
            alert_type=alert_type,
            message=message,
            details=details,
        )
        if alert_id:
            logger.warning(f"ALERT triggered for {model_id}: {message}")
            for handler in self.alert_handlers:
                try:
                    handler(model_id, alert_type, message, details)
                except Exception as e:
                    logger.error(f"Error executing alert handler: {e}")
        else:
            logger.error(f"Failed to record alert for {model_id}")


# CLI functionality for running the model monitor
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser(description="Enterprise Model Monitor")
    parser.add_argument("--models", type=str, nargs="+", help="Models to monitor")
    parser.add_argument(
        "--interval",
        type=int,
        default=MONITORING_INTERVAL_SEC,
        help="Monitoring interval in seconds",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=PERFORMANCE_THRESHOLD,
        help="Performance degradation threshold",
    )
    parser.add_argument(
        "--auto-retrain", action="store_true", help="Enable automatic retraining"
    )
    parser.add_argument(
        "--evaluate", type=str, help="Manually evaluate a specific model"
    )
    parser.add_argument("--status", action="store_true", help="Show monitoring status")
    args = parser.parse_args()

    # Create model monitor
    monitor = ModelMonitor(
        monitoring_interval=args.interval,
        performance_threshold=args.threshold,
        auto_retrain=args.auto_retrain,
    )

    # Handle CLI commands
    if args.evaluate:
        print(f"Evaluating model: {args.evaluate}")
        metrics = monitor.manually_evaluate_model(args.evaluate)
        if metrics:
            print("Evaluation complete - Results:")
            print(f"  Inference time: {metrics.inference_time_ms:.2f} ms")
            print(f"  Tokens per second: {metrics.tokens_per_second:.2f}")
            print(f"  Memory usage: {metrics.memory_usage_mb:.2f} MB")
            print(f"  Success rate: {metrics.success_rate:.2%}")
        else:
            print("Evaluation failed")

    elif args.status:
        status = monitor.get_monitoring_status()
        print("Monitoring status:")
        print(f"  Active: {status['is_monitoring']}")
        print(
            f"  Models monitored: {', '.join(status['models_monitored']) if status['models_monitored'] else 'None'}"
        )
        print(f"  Interval: {status['monitoring_interval_sec']} seconds")
        print(f"  Auto-retrain: {status['auto_retrain_enabled']}")

        # Show alerts
        alerts = monitor.monitor_db.get_active_alerts()
        print(f"\nActive alerts: {len(alerts)}")
        for alert in alerts[:5]:  # Show top 5 alerts
            print(f"  [{alert['severity']}] {alert['model_id']}: {alert['message']}")

    elif args.models:
        print(f"Starting monitoring for models: {', '.join(args.models)}")
        monitor.start_monitoring(args.models)

        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping monitoring...")
            monitor.stop_monitoring()

    else:
        parser.print_help()
