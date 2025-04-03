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
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

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

        self._monitoring_thread = None
        self._stop_event = threading.Event()
        self._models_being_monitored = set()
        self._last_check_time = {}

    def start_monitoring(self, model_ids: Optional[List[str]] = None):
        """Start continuous monitoring of specified models"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Monitoring thread is already running")
            return

        # If no models specified, monitor all models in registry
        if model_ids is None:
            model_ids = self.model_registry.list_models()

        if not model_ids:
            logger.warning("No models found to monitor")
            return

        # Add models to monitoring set
        for model_id in model_ids:
            self._models_being_monitored.add(model_id)
            self._last_check_time[model_id] = (
                datetime.datetime.now() - datetime.timedelta(days=1)
            )

        logger.info(f"Starting monitoring for models: {', '.join(model_ids)}")

        # Start monitoring thread
        self._stop_event.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, name="ModelMonitorThread"
        )
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()

        return True

    def stop_monitoring(self):
        """Stop the monitoring thread"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.info("Stopping model monitoring")
            self._stop_event.set()
            self._monitoring_thread.join(timeout=5.0)
            return True
        return False

    def _monitoring_loop(self):
        """Main monitoring loop that runs in a separate thread"""
        while not self._stop_event.is_set():
            try:
                current_time = datetime.datetime.now()

                # Check each model in the monitoring set
                for model_id in list(self._models_being_monitored):
                    last_check = self._last_check_time.get(
                        model_id, datetime.datetime.min
                    )
                    seconds_since_last_check = (
                        current_time - last_check
                    ).total_seconds()

                    # Only check if enough time has passed
                    if seconds_since_last_check >= self.monitoring_interval:
                        logger.debug(f"Checking model {model_id} performance")
                        self._check_model_performance(model_id)
                        self._last_check_time[model_id] = current_time

                # Sleep with interrupt checks
                for _ in range(60):  # Check for stop every second for up to 60 seconds
                    if self._stop_event.is_set():
                        break
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Sleep for a minute if there's an error

    def _check_model_performance(self, model_id: str):
        """Check performance of a specific model"""
        try:
            # Get the current active version
            current_version = self.model_registry.get_model(model_id)
            if not current_version:
                logger.warning(f"Model {model_id} not found in registry")
                return

            # Get baseline metrics
            baseline_metrics = self._get_baseline_metrics(
                model_id, current_version.version
            )
            if not baseline_metrics:
                # If no baseline, collect and store initial metrics
                logger.info(
                    f"No baseline for {model_id} ({current_version.version}), collecting initial metrics"
                )
                current_metrics = self._collect_model_metrics(model_id, current_version)
                self.monitor_db.insert_metrics(current_metrics)
                return

            # Collect current metrics
            current_metrics = self._collect_model_metrics(model_id, current_version)
            self.monitor_db.insert_metrics(current_metrics)

            # Check for performance degradation
            degradation = self._detect_performance_degradation(
                baseline_metrics, current_metrics
            )
            if degradation:
                logger.warning(
                    f"Performance degradation detected for {model_id}: {degradation}"
                )

                # Record alert
                message = f"Performance degradation detected: {degradation}"
                self.monitor_db.add_alert(
                    model_id=model_id,
                    version=current_version.version,
                    alert_type="PERFORMANCE_DEGRADATION",
                    message=message,
                    severity="WARNING",
                )

                # Trigger alert handlers
                for handler in self.alert_handlers:
                    try:
                        handler(
                            {
                                "model_id": model_id,
                                "version": current_version.version,
                                "alert_type": "PERFORMANCE_DEGRADATION",
                                "message": message,
                                "severity": "WARNING",
                                "details": degradation,
                            }
                        )
                    except Exception as e:
                        logger.error(f"Error in alert handler: {e}")

                # Trigger retraining if auto-retrain is enabled
                if self.auto_retrain:
                    self._trigger_retraining(
                        model_id, current_version.version, degradation
                    )

        except Exception as e:
            logger.error(f"Error checking model {model_id} performance: {e}")

    def _get_baseline_metrics(
        self, model_id: str, version: str
    ) -> Optional[ModelPerformanceMetrics]:
        """Get baseline metrics for a model"""
        # Try to get version-specific baseline first
        baseline = self.monitor_db.get_latest_metrics(model_id, version)
        if baseline:
            return baseline

        # If no version-specific baseline, try to get any baseline for this model
        return self.monitor_db.get_latest_metrics(model_id)

    def _collect_model_metrics(
        self, model_id: str, model_version: ModelVersion
    ) -> ModelPerformanceMetrics:
        """Collect performance metrics for a model"""
        # Start with basic metrics
        metrics = ModelPerformanceMetrics(
            model_id=model_id,
            version=model_version.version,
            timestamp=datetime.datetime.now(),
            metadata={},
        )

        # Try to load and test the model
        if LLAMA_UTILS_AVAILABLE and model_id in [
            "llama3-70b",
            "llama3-8b",
            "mistral-7b",
        ]:
            # For Llama models, use llama_utils
            metrics = self._evaluate_llama_model(model_id, model_version.path, metrics)
        else:
            # For other models, use generic evaluation
            metrics = self._evaluate_generic_model(
                model_id, model_version.path, metrics
            )

        return metrics

    def _evaluate_llama_model(
        self, model_id: str, model_path: str, metrics: ModelPerformanceMetrics
    ) -> ModelPerformanceMetrics:
        """Evaluate a Llama model for performance metrics"""
        if not LLAMA_UTILS_AVAILABLE:
            logger.warning(
                f"llama_utils not available, skipping detailed evaluation for {model_id}"
            )
            return metrics

        # Standard test prompts for evaluating LLM performance
        test_prompts = [
            "Explain quantum computing in simple terms",
            "Write a short poem about artificial intelligence",
            "List five tips for improving productivity",
        ]

        try:
            # Load model
            start_time = time.time()
            model = get_optimized_model(model_path)
            load_time = time.time() - start_time

            # Get memory usage after loading
            memory_usage = self._get_process_memory_mb()
            metrics.memory_usage_mb = memory_usage
            metrics.metadata["load_time_sec"] = load_time

            # Run inference tests
            total_tokens = 0
            total_time = 0
            for prompt in test_prompts:
                start_time = time.time()
                model.generate(prompt, max_tokens=100)
                elapsed = time.time() - start_time

                tokens_generated = 100  # Assuming we requested 100 tokens
                total_tokens += tokens_generated
                total_time += elapsed

            # Calculate performance metrics
            avg_inference_time = (
                total_time / len(test_prompts)
            ) * 1000  # Convert to ms
            tokens_per_second = total_tokens / total_time if total_time > 0 else 0

            metrics.inference_time_ms = avg_inference_time
            metrics.tokens_per_second = tokens_per_second
            metrics.total_requests = len(test_prompts)
            metrics.avg_request_time_ms = avg_inference_time

            # Clean up
            del model

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating Llama model {model_id}: {e}")
            metrics.failed_requests = len(test_prompts)
            metrics.total_requests = len(test_prompts)
            metrics.metadata["error"] = str(e)
            return metrics

    def _evaluate_generic_model(
        self, model_id: str, model_path: str, metrics: ModelPerformanceMetrics
    ) -> ModelPerformanceMetrics:
        """Evaluate a generic model for performance metrics"""
        # Basic evaluation using file stats and system info
        try:
            # Get file stats
            if os.path.exists(model_path):
                file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
                metrics.metadata["file_size_mb"] = file_size_mb

            # Get system memory info
            metrics.memory_usage_mb = self._get_process_memory_mb()

            # Placeholder values until we implement proper testing
            metrics.total_requests = 1
            metrics.avg_request_time_ms = 0

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating generic model {model_id}: {e}")
            metrics.failed_requests = 1
            metrics.total_requests = 1
            metrics.metadata["error"] = str(e)
            return metrics

    def _detect_performance_degradation(
        self, baseline: ModelPerformanceMetrics, current: ModelPerformanceMetrics
    ) -> Optional[Dict[str, Any]]:
        """
        Detect performance degradation between baseline and current metrics
        Returns a dict with degradation details if detected, None otherwise
        """
        degradation = {}

        # Check inference time (if available)
        if (
            baseline.inference_time_ms
            and current.inference_time_ms
            and baseline.inference_time_ms > 0
        ):
            time_increase = (
                current.inference_time_ms - baseline.inference_time_ms
            ) / baseline.inference_time_ms
            if time_increase > MAX_INFERENCE_TIME_INCREASE:
                degradation["inference_time_increase"] = f"{time_increase:.2%}"

        # Check tokens per second (if available)
        if (
            baseline.tokens_per_second
            and current.tokens_per_second
            and baseline.tokens_per_second > 0
        ):
            tps_decrease = (
                baseline.tokens_per_second - current.tokens_per_second
            ) / baseline.tokens_per_second
            if tps_decrease > self.performance_threshold:
                degradation["tokens_per_second_decrease"] = f"{tps_decrease:.2%}"

        # Check success rate
        if baseline.success_rate > 0 and current.success_rate < baseline.success_rate:
            rate_decrease = baseline.success_rate - current.success_rate
            if rate_decrease > 0.05:  # 5% decrease in success rate
                degradation["success_rate_decrease"] = f"{rate_decrease:.2%}"

        # If we have drift score, check it
        if current.drift_score and current.drift_score > 0.3:  # 30% drift
            degradation["drift_score"] = current.drift_score

        # If any degradation detected, return the details
        return degradation if degradation else None

    def _trigger_retraining(
        self, model_id: str, version: str, degradation_details: Dict[str, Any]
    ):
        """Trigger model retraining process"""
        logger.info(
            f"Triggering retraining for {model_id} due to performance degradation"
        )

        # Get current metrics for before/after comparison
        current_metrics = self.monitor_db.get_latest_metrics(model_id, version)
        metrics_dict = None
        if current_metrics:
            metrics_dict = {
                "inference_time_ms": current_metrics.inference_time_ms,
                "tokens_per_second": current_metrics.tokens_per_second,
                "success_rate": current_metrics.success_rate,
                "memory_usage_mb": current_metrics.memory_usage_mb,
            }

        # Record retraining event
        event_id = self.monitor_db.record_retraining_event(
            model_id=model_id,
            old_version=version,
            reason=f"Performance degradation: {json.dumps(degradation_details)}",
            metrics_before=metrics_dict,
        )

        # TODO: Implement actual retraining logic here
        # This would typically involve:
        # 1. Launching a retraining job (possibly on a separate machine)
        # 2. Monitoring the retraining progress
        # 3. Validating the new model
        # 4. Registering the new model version
        # 5. Updating the retraining event record

        # For now, just log that we would retrain
        logger.info(f"Retraining event {event_id} recorded for {model_id}")

        # Update the event status to indicate it's pending implementation
        self.monitor_db.update_retraining_event(
            event_id=event_id, status="PENDING_IMPLEMENTATION"
        )

    def _get_process_memory_mb(self) -> float:
        """Get current process memory usage in MB"""
        try:
            import psutil

            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def manually_evaluate_model(
        self, model_id: str
    ) -> Optional[ModelPerformanceMetrics]:
        """
        Manually trigger evaluation of a model's performance

        Args:
            model_id: ID of the model to evaluate

        Returns:
            ModelPerformanceMetrics or None if evaluation failed
        """
        model_version = self.model_registry.get_model(model_id)
        if not model_version:
            logger.error(f"Model {model_id} not found in registry")
            return None

        metrics = self._collect_model_metrics(model_id, model_version)
        self.monitor_db.insert_metrics(metrics)

        return metrics

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get the current monitoring status"""
        return {
            "is_monitoring": self._monitoring_thread is not None
            and self._monitoring_thread.is_alive(),
            "models_monitored": list(self._models_being_monitored),
            "last_check_times": {
                model_id: timestamp.isoformat()
                for model_id, timestamp in self._last_check_time.items()
            },
            "monitoring_interval_sec": self.monitoring_interval,
            "auto_retrain_enabled": self.auto_retrain,
        }


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
