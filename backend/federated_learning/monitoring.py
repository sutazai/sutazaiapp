"""
Federated Learning Performance Monitoring
=========================================

Comprehensive monitoring and analytics for federated learning systems.
Tracks training progress, client performance, privacy consumption, and system health.

Features:
- Real-time training metrics collection
- Client performance analysis
- Privacy budget tracking
- Convergence monitoring
- Anomaly detection
- Performance forecasting
- Resource utilization tracking
- Comprehensive reporting
"""

import asyncio
import json
import logging
import time
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from app.schemas.message_types import AlertSeverity
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import statistics
from concurrent.futures import ThreadPoolExecutor

import aioredis
from pydantic import BaseModel


class MetricType(Enum):
    """Types of metrics to monitor"""
    ACCURACY = "accuracy"
    LOSS = "loss"
    CONVERGENCE = "convergence"
    COMMUNICATION_COST = "communication_cost"
    COMPUTATION_TIME = "computation_time"
    CLIENT_PARTICIPATION = "client_participation"
    PRIVACY_BUDGET = "privacy_budget"
    RESOURCE_UTILIZATION = "resource_utilization"
    ANOMALY_SCORE = "anomaly_score"




@dataclass
class TrainingMetric:
    """Individual training metric"""
    training_id: str
    round_number: int
    metric_type: MetricType
    value: float
    timestamp: datetime
    client_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'metric_type': self.metric_type.value,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ClientPerformance:
    """Client performance metrics"""
    client_id: str
    training_id: str
    total_rounds: int
    successful_rounds: int
    failed_rounds: int
    average_training_time: float
    average_communication_time: float
    total_samples_contributed: int
    reliability_score: float
    contribution_score: float
    last_activity: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'last_activity': self.last_activity.isoformat()
        }


@dataclass
class TrainingProgress:
    """Overall training progress tracking"""
    training_id: str
    current_round: int
    total_rounds: int
    start_time: datetime
    estimated_completion: Optional[datetime]
    convergence_status: str
    best_accuracy: float
    current_accuracy: float
    participating_clients: int
    active_clients: int
    privacy_budget_consumed: float
    total_privacy_budget: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'start_time': self.start_time.isoformat(),
            'estimated_completion': self.estimated_completion.isoformat() if self.estimated_completion else None
        }


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    training_id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat()
        }


class AnomalyDetector:
    """Detect anomalies in federated learning metrics"""
    
    def __init__(self, window_size: int = 50, threshold: float = 2.0):
        self.window_size = window_size
        self.threshold = threshold
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        
    def add_metric(self, metric_key: str, value: float):
        """Add a metric value for anomaly detection"""
        self.metric_history[metric_key].append(value)
    
    def detect_anomaly(self, metric_key: str, value: float) -> Tuple[bool, float]:
        """Detect if a value is anomalous"""
        history = self.metric_history[metric_key]
        
        if len(history) < 10:  # Need sufficient history
            return False, 0.0
        
        # Calculate z-score
        mean = statistics.mean(history)
        try:
            stdev = statistics.stdev(history)
            if stdev == 0:
                return False, 0.0
            
            z_score = abs((value - mean) / stdev)
            is_anomaly = z_score > self.threshold
            
            return is_anomaly, z_score
            
        except statistics.StatisticsError:
            return False, 0.0
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of anomaly detection status"""
        summary = {}
        
        for metric_key, history in self.metric_history.items():
            if len(history) >= 10:
                mean = statistics.mean(history)
                stdev = statistics.stdev(history) if len(history) > 1 else 0
                
                summary[metric_key] = {
                    "samples": len(history),
                    "mean": mean,
                    "std": stdev,
                    "latest": history[-1] if history else None
                }
        
        return summary


class PerformancePredictor:
    """Predict federated learning performance trends"""
    
    def __init__(self):
        self.convergence_patterns: Dict[str, List[float]] = {}
        
    def add_training_data(self, training_id: str, accuracy_history: List[float]):
        """Add training data for prediction modeling"""
        self.convergence_patterns[training_id] = accuracy_history.copy()
    
    def predict_convergence(self, training_id: str, current_history: List[float]) -> Dict[str, Any]:
        """Predict convergence based on current history"""
        if len(current_history) < 5:
            return {"prediction": "insufficient_data"}
        
        # Simple trend analysis
        recent_trend = self._calculate_trend(current_history[-10:])
        overall_trend = self._calculate_trend(current_history)
        
        # Estimate rounds to convergence
        current_accuracy = current_history[-1]
        improvement_rate = recent_trend
        
        target_accuracy = 0.95  # Assume 95% target
        
        if improvement_rate > 0:
            rounds_to_target = max(0, (target_accuracy - current_accuracy) / improvement_rate)
        else:
            rounds_to_target = float('inf')
        
        # Convergence status
        if abs(recent_trend) < 0.001:  # Very small improvement
            convergence_status = "converged"
        elif recent_trend > 0.01:
            convergence_status = "improving"
        elif recent_trend < -0.01:
            convergence_status = "degrading"
        else:
            convergence_status = "stable"
        
        return {
            "convergence_status": convergence_status,
            "recent_trend": recent_trend,
            "overall_trend": overall_trend,
            "estimated_rounds_to_target": rounds_to_target,
            "current_accuracy": current_accuracy,
            "prediction_confidence": self._calculate_confidence(current_history)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend using linear regression slope"""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = list(range(n))
        
        # Simple linear regression
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return slope
    
    def _calculate_confidence(self, values: List[float]) -> float:
        """Calculate prediction confidence based on data quality"""
        if len(values) < 5:
            return 0.3
        
        # Confidence based on data consistency and length
        variance = statistics.variance(values) if len(values) > 1 else 0
        length_factor = min(1.0, len(values) / 50)  # Higher confidence with more data
        consistency_factor = max(0.1, 1.0 - variance)  # Lower variance = higher confidence
        
        return min(0.95, length_factor * consistency_factor)


class ResourceMonitor:
    """Monitor system resource utilization"""
    
    def __init__(self):
        self.cpu_history: deque = deque(maxlen=100)
        self.memory_history: deque = deque(maxlen=100)
        self.network_history: deque = deque(maxlen=100)
        
    def record_usage(self, cpu_percent: float, memory_percent: float, 
                    network_bytes: float):
        """Record resource usage"""
        self.cpu_history.append(cpu_percent)
        self.memory_history.append(memory_percent)
        self.network_history.append(network_bytes)
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource utilization summary"""
        summary = {}
        
        if self.cpu_history:
            summary["cpu"] = {
                "current": self.cpu_history[-1],
                "average": statistics.mean(self.cpu_history),
                "peak": max(self.cpu_history),
                "samples": len(self.cpu_history)
            }
        
        if self.memory_history:
            summary["memory"] = {
                "current": self.memory_history[-1],
                "average": statistics.mean(self.memory_history),
                "peak": max(self.memory_history),
                "samples": len(self.memory_history)
            }
        
        if self.network_history:
            summary["network"] = {
                "current": self.network_history[-1],
                "total": sum(self.network_history),
                "average": statistics.mean(self.network_history),
                "peak": max(self.network_history),
                "samples": len(self.network_history)
            }
        
        return summary


class FederatedMonitor:
    """
    Comprehensive Federated Learning Monitor
    
    Monitors training progress, client performance, system health, and provides
    analytics and alerting for federated learning in SutazAI.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379",
                 namespace: str = "sutazai:federated:monitoring"):
        
        self.redis_url = redis_url
        self.namespace = namespace
        self.redis: Optional[aioredis.Redis] = None
        
        # Monitoring components
        self.anomaly_detector = AnomalyDetector()
        self.performance_predictor = PerformancePredictor()
        self.resource_monitor = ResourceMonitor()
        
        # Data storage
        self.training_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.client_performances: Dict[str, ClientPerformance] = {}
        self.training_progress: Dict[str, TrainingProgress] = {}
        self.alerts: Dict[str, Alert] = {}
        
        # Alert thresholds
        self.alert_thresholds = {
            "accuracy_drop": 0.1,  # 10% accuracy drop
            "client_failure_rate": 0.3,  # 30% client failure rate
            "communication_timeout": 300,  # 5 minutes
            "privacy_budget_exhaustion": 0.9,  # 90% budget consumed
            "resource_utilization": 0.9  # 90% resource usage
        }
        
        # Statistics
        self.monitor_stats = {
            "metrics_collected": 0,
            "alerts_generated": 0,
            "anomalies_detected": 0,
            "trainings_monitored": 0,
            "clients_tracked": 0
        }
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        self.logger = logging.getLogger("federated_monitor")
    
    async def initialize(self):
        """Initialize the federated monitor"""
        try:
            self.logger.info("Initializing Federated Learning Monitor")
            
            # Connect to Redis
            self.redis = aioredis.from_url(self.redis_url, decode_responses=True)
            await self.redis.ping()
            
            # Load existing data
            await self._load_monitoring_data()
            
            # Start background tasks
            self._start_background_tasks()
            
            self.logger.info("Federated Learning Monitor initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitor: {e}")
            raise
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        tasks = [
            self._metrics_aggregator(),
            self._alert_processor(),
            self._performance_analyzer(),
            self._resource_collector(),
            self._data_persistence()
        ]
        
        for coro in tasks:
            task = asyncio.create_task(coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
    
    async def _load_monitoring_data(self):
        """Load existing monitoring data from Redis"""
        try:
            # Load training progress
            progress_keys = await self.redis.keys(f"{self.namespace}:progress:*")
            for key in progress_keys:
                data = await self.redis.get(key)
                if data:
                    progress_dict = json.loads(data)
                    training_id = progress_dict["training_id"]
                    progress_dict["start_time"] = datetime.fromisoformat(progress_dict["start_time"])
                    if progress_dict.get("estimated_completion"):
                        progress_dict["estimated_completion"] = datetime.fromisoformat(progress_dict["estimated_completion"])
                    
                    self.training_progress[training_id] = TrainingProgress(**progress_dict)
            
            # Load client performances
            client_keys = await self.redis.keys(f"{self.namespace}:client:*")
            for key in client_keys:
                data = await self.redis.get(key)
                if data:
                    client_dict = json.loads(data)
                    client_dict["last_activity"] = datetime.fromisoformat(client_dict["last_activity"])
                    
                    client_id = client_dict["client_id"]
                    self.client_performances[client_id] = ClientPerformance(**client_dict)
            
            self.logger.info(f"Loaded monitoring data: {len(self.training_progress)} trainings, "
                           f"{len(self.client_performances)} clients")
            
        except Exception as e:
            self.logger.error(f"Failed to load monitoring data: {e}")
    
    async def record_training_metric(self, metric: TrainingMetric):
        """Record a training metric"""
        try:
            # Store metric
            metric_key = f"{metric.training_id}:{metric.metric_type.value}"
            self.training_metrics[metric_key].append(metric)
            
            # Update statistics
            self.monitor_stats["metrics_collected"] += 1
            
            # Anomaly detection
            is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly(metric_key, metric.value)
            
            if is_anomaly:
                await self._generate_anomaly_alert(metric, anomaly_score)
                self.monitor_stats["anomalies_detected"] += 1
            
            # Add to anomaly detector
            self.anomaly_detector.add_metric(metric_key, metric.value)
            
            # Update training progress
            await self._update_training_progress(metric)
            
            self.logger.debug(f"Recorded metric: {metric.training_id} {metric.metric_type.value} = {metric.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to record metric: {e}")
    
    async def record_client_performance(self, client_id: str, training_id: str,
                                      training_time: float, communication_time: float,
                                      samples_contributed: int, success: bool):
        """Record client performance data"""
        try:
            # Get or create client performance record
            if client_id not in self.client_performances:
                self.client_performances[client_id] = ClientPerformance(
                    client_id=client_id,
                    training_id=training_id,
                    total_rounds=0,
                    successful_rounds=0,
                    failed_rounds=0,
                    average_training_time=0.0,
                    average_communication_time=0.0,
                    total_samples_contributed=0,
                    reliability_score=1.0,
                    contribution_score=0.0,
                    last_activity=datetime.utcnow()
                )
            
            performance = self.client_performances[client_id]
            
            # Update performance metrics
            performance.total_rounds += 1
            if success:
                performance.successful_rounds += 1
            else:
                performance.failed_rounds += 1
            
            # Update averages
            n = performance.total_rounds
            performance.average_training_time = (
                (performance.average_training_time * (n - 1) + training_time) / n
            )
            performance.average_communication_time = (
                (performance.average_communication_time * (n - 1) + communication_time) / n
            )
            
            performance.total_samples_contributed += samples_contributed
            performance.last_activity = datetime.utcnow()
            
            # Calculate reliability score
            performance.reliability_score = performance.successful_rounds / performance.total_rounds
            
            # Calculate contribution score (based on samples and reliability)
            performance.contribution_score = (
                performance.total_samples_contributed * performance.reliability_score / 1000.0
            )
            
            # Check for performance alerts
            await self._check_client_performance_alerts(performance)
            
            self.logger.debug(f"Updated client performance: {client_id} (reliability: {performance.reliability_score:.2f})")
            
        except Exception as e:
            self.logger.error(f"Failed to record client performance: {e}")
    
    async def record_training_completion(self, training):
        """Record training completion and generate summary"""
        try:
            training_id = training.training_id
            
            # Generate training summary
            summary = await self._generate_training_summary(training)
            
            # Store summary
            summary_key = f"{self.namespace}:summary:{training_id}"
            await self.redis.set(summary_key, json.dumps(summary))
            
            # Update predictor with training data
            if training.performance_history:
                accuracy_history = [p.get("accuracy", 0.0) for p in training.performance_history]
                self.performance_predictor.add_training_data(training_id, accuracy_history)
            
            # Generate completion alert
            await self._generate_completion_alert(training, summary)
            
            self.logger.info(f"Recorded training completion: {training_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to record training completion: {e}")
    
    async def _generate_training_summary(self, training) -> Dict[str, Any]:
        """Generate comprehensive training summary"""
        training_id = training.training_id
        
        # Calculate training duration
        duration = (training.end_time - training.start_time).total_seconds() if training.end_time else 0
        
        # Get final performance
        final_performance = training.performance_history[-1] if training.performance_history else {}
        
        # Calculate client statistics
        participating_clients = len(training.participating_clients)
        
        client_stats = {
            "total_clients": participating_clients,
            "avg_reliability": 0.0,
            "avg_contribution": 0.0
        }
        
        if participating_clients > 0:
            client_reliabilities = []
            client_contributions = []
            
            for client_id in training.participating_clients:
                if client_id in self.client_performances:
                    perf = self.client_performances[client_id]
                    client_reliabilities.append(perf.reliability_score)
                    client_contributions.append(perf.contribution_score)
            
            if client_reliabilities:
                client_stats["avg_reliability"] = statistics.mean(client_reliabilities)
                client_stats["avg_contribution"] = statistics.mean(client_contributions)
        
        # Get metrics summary
        metrics_summary = {}
        for metric_type in MetricType:
            metric_key = f"{training_id}:{metric_type.value}"
            if metric_key in self.training_metrics:
                values = [m.value for m in self.training_metrics[metric_key]]
                if values:
                    metrics_summary[metric_type.value] = {
                        "final": values[-1],
                        "best": max(values) if metric_type in [MetricType.ACCURACY] else min(values),
                        "average": statistics.mean(values),
                        "samples": len(values)
                    }
        
        summary = {
            "training_id": training_id,
            "algorithm": training.config.algorithm.value,
            "status": training.status.value,
            "duration_seconds": duration,
            "total_rounds": training.current_round,
            "final_performance": final_performance,
            "client_statistics": client_stats,
            "metrics_summary": metrics_summary,
            "privacy_budget_used": getattr(training.config.privacy_budget, 'consumed_epsilon', 0.0) if training.config.privacy_budget else 0.0,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return summary
    
    async def _update_training_progress(self, metric: TrainingMetric):
        """Update training progress based on new metric"""
        try:
            training_id = metric.training_id
            
            if training_id not in self.training_progress:
                # Create new progress tracking
                self.training_progress[training_id] = TrainingProgress(
                    training_id=training_id,
                    current_round=metric.round_number,
                    total_rounds=100,  # Default, will be updated
                    start_time=metric.timestamp,
                    estimated_completion=None,
                    convergence_status="starting",
                    best_accuracy=0.0,
                    current_accuracy=0.0,
                    participating_clients=0,
                    active_clients=0,
                    privacy_budget_consumed=0.0,
                    total_privacy_budget=1.0
                )
                self.monitor_stats["trainings_monitored"] += 1
            
            progress = self.training_progress[training_id]
            
            # Update current round
            progress.current_round = max(progress.current_round, metric.round_number)
            
            # Update accuracy tracking
            if metric.metric_type == MetricType.ACCURACY:
                progress.current_accuracy = metric.value
                progress.best_accuracy = max(progress.best_accuracy, metric.value)
                
                # Update convergence prediction
                accuracy_history = [m.value for m in self.training_metrics[f"{training_id}:accuracy"]]
                if len(accuracy_history) >= 5:
                    prediction = self.performance_predictor.predict_convergence(training_id, accuracy_history)
                    progress.convergence_status = prediction.get("convergence_status", "unknown")
                    
                    # Estimate completion time
                    if prediction.get("estimated_rounds_to_target", float('inf')) != float('inf'):
                        remaining_rounds = prediction["estimated_rounds_to_target"]
                        if remaining_rounds > 0:
                            # Estimate time per round based on history
                            elapsed_time = (metric.timestamp - progress.start_time).total_seconds()
                            time_per_round = elapsed_time / progress.current_round if progress.current_round > 0 else 60
                            estimated_seconds = remaining_rounds * time_per_round
                            progress.estimated_completion = metric.timestamp + timedelta(seconds=estimated_seconds)
            
        except Exception as e:
            self.logger.error(f"Failed to update training progress: {e}")
    
    async def _generate_anomaly_alert(self, metric: TrainingMetric, anomaly_score: float):
        """Generate alert for detected anomaly"""
        try:
            alert_id = f"anomaly_{metric.training_id}_{int(time.time())}"
            
            alert = Alert(
                alert_id=alert_id,
                training_id=metric.training_id,
                severity=AlertSeverity.WARNING,
                title=f"Anomaly Detected: {metric.metric_type.value}",
                description=f"Unusual {metric.metric_type.value} value detected: {metric.value:.4f} (z-score: {anomaly_score:.2f})",
                timestamp=datetime.utcnow(),
                metadata={
                    "metric_type": metric.metric_type.value,
                    "metric_value": metric.value,
                    "anomaly_score": anomaly_score,
                    "round_number": metric.round_number
                }
            )
            
            self.alerts[alert_id] = alert
            await self._store_alert(alert)
            
            self.logger.warning(f"Anomaly alert generated: {alert.title}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate anomaly alert: {e}")
    
    async def _check_client_performance_alerts(self, performance: ClientPerformance):
        """Check for client performance-related alerts"""
        try:
            # Low reliability alert
            if performance.reliability_score < (1.0 - self.alert_thresholds["client_failure_rate"]):
                alert_id = f"client_reliability_{performance.client_id}_{int(time.time())}"
                
                alert = Alert(
                    alert_id=alert_id,
                    training_id=performance.training_id,
                    severity=AlertSeverity.WARNING,
                    title=f"Low Client Reliability: {performance.client_id}",
                    description=f"Client reliability dropped to {performance.reliability_score:.2f}",
                    timestamp=datetime.utcnow(),
                    metadata={
                        "client_id": performance.client_id,
                        "reliability_score": performance.reliability_score,
                        "total_rounds": performance.total_rounds,
                        "failed_rounds": performance.failed_rounds
                    }
                )
                
                self.alerts[alert_id] = alert
                await self._store_alert(alert)
            
            # High communication time alert
            if performance.average_communication_time > self.alert_thresholds["communication_timeout"]:
                alert_id = f"client_communication_{performance.client_id}_{int(time.time())}"
                
                alert = Alert(
                    alert_id=alert_id,
                    training_id=performance.training_id,
                    severity=AlertSeverity.INFO,
                    title=f"High Communication Time: {performance.client_id}",
                    description=f"Average communication time: {performance.average_communication_time:.2f}s",
                    timestamp=datetime.utcnow(),
                    metadata={
                        "client_id": performance.client_id,
                        "average_communication_time": performance.average_communication_time
                    }
                )
                
                self.alerts[alert_id] = alert
                await self._store_alert(alert)
                
        except Exception as e:
            self.logger.error(f"Failed to check client performance alerts: {e}")
    
    async def _generate_completion_alert(self, training, summary: Dict[str, Any]):
        """Generate alert for training completion"""
        try:
            final_accuracy = summary.get("final_performance", {}).get("accuracy", 0.0)
            target_accuracy = training.config.target_accuracy
            
            # Determine severity based on performance
            if final_accuracy >= target_accuracy:
                severity = AlertSeverity.INFO
                title = f"Training Completed Successfully: {training.training_id}"
            elif final_accuracy >= target_accuracy * 0.9:
                severity = AlertSeverity.WARNING
                title = f"Training Completed with Suboptimal Performance: {training.training_id}"
            else:
                severity = AlertSeverity.ERROR
                title = f"Training Completed with Poor Performance: {training.training_id}"
            
            alert_id = f"completion_{training.training_id}_{int(time.time())}"
            
            alert = Alert(
                alert_id=alert_id,
                training_id=training.training_id,
                severity=severity,
                title=title,
                description=f"Final accuracy: {final_accuracy:.4f}, Target: {target_accuracy:.4f}",
                timestamp=datetime.utcnow(),
                metadata=summary
            )
            
            self.alerts[alert_id] = alert
            await self._store_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Failed to generate completion alert: {e}")
    
    async def _store_alert(self, alert: Alert):
        """Store alert in Redis"""
        try:
            alert_key = f"{self.namespace}:alert:{alert.alert_id}"
            await self.redis.set(alert_key, json.dumps(alert.to_dict()))
            
            # Add to alerts list
            alerts_key = f"{self.namespace}:alerts:{alert.training_id}"
            await self.redis.lpush(alerts_key, alert.alert_id)
            
            self.monitor_stats["alerts_generated"] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to store alert: {e}")
    
    # Background task implementations
    async def _metrics_aggregator(self):
        """Aggregate and analyze metrics periodically"""
        while not self._shutdown_event.is_set():
            try:
                # Perform metric aggregation and analysis
                for training_id, progress in self.training_progress.items():
                    await self._analyze_training_metrics(training_id)
                
                await asyncio.sleep(30)  # Aggregate every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Metrics aggregator error: {e}")
                await asyncio.sleep(30)
    
    async def _analyze_training_metrics(self, training_id: str):
        """Analyze metrics for a specific training"""
        try:
            # Check convergence
            accuracy_key = f"{training_id}:accuracy"
            if accuracy_key in self.training_metrics:
                accuracy_metrics = list(self.training_metrics[accuracy_key])
                if len(accuracy_metrics) >= 10:
                    recent_accuracies = [m.value for m in accuracy_metrics[-10:]]
                    
                    # Check for accuracy drop
                    if len(recent_accuracies) >= 2:
                        accuracy_drop = recent_accuracies[0] - recent_accuracies[-1]
                        if accuracy_drop > self.alert_thresholds["accuracy_drop"]:
                            await self._generate_accuracy_drop_alert(training_id, accuracy_drop)
            
        except Exception as e:
            self.logger.error(f"Failed to analyze training metrics: {e}")
    
    async def _generate_accuracy_drop_alert(self, training_id: str, accuracy_drop: float):
        """Generate alert for accuracy drop"""
        try:
            alert_id = f"accuracy_drop_{training_id}_{int(time.time())}"
            
            alert = Alert(
                alert_id=alert_id,
                training_id=training_id,
                severity=AlertSeverity.ERROR,
                title=f"Significant Accuracy Drop: {training_id}",
                description=f"Accuracy dropped by {accuracy_drop:.4f} in recent rounds",
                timestamp=datetime.utcnow(),
                metadata={
                    "accuracy_drop": accuracy_drop,
                    "threshold": self.alert_thresholds["accuracy_drop"]
                }
            )
            
            self.alerts[alert_id] = alert
            await self._store_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Failed to generate accuracy drop alert: {e}")
    
    async def _alert_processor(self):
        """Process and manage alerts"""
        while not self._shutdown_event.is_set():
            try:
                # Clean up old resolved alerts
                current_time = datetime.utcnow()
                expired_alerts = []
                
                for alert_id, alert in self.alerts.items():
                    # Auto-resolve old alerts
                    age = (current_time - alert.timestamp).total_seconds()
                    if age > 3600 and not alert.resolved:  # 1 hour
                        alert.resolved = True
                        expired_alerts.append(alert_id)
                
                # Remove expired alerts
                for alert_id in expired_alerts:
                    if alert_id in self.alerts:
                        del self.alerts[alert_id]
                
                await asyncio.sleep(300)  # Process every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Alert processor error: {e}")
                await asyncio.sleep(300)
    
    async def _performance_analyzer(self):
        """Analyze overall system performance"""
        while not self._shutdown_event.is_set():
            try:
                # Analyze client performance trends
                for client_id, performance in self.client_performances.items():
                    self.monitor_stats["clients_tracked"] = len(self.client_performances)
                
                # Update anomaly detection summaries
                anomaly_summary = self.anomaly_detector.get_anomaly_summary()
                
                # Store analysis results
                analysis_key = f"{self.namespace}:analysis"
                analysis_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "anomaly_summary": anomaly_summary,
                    "monitor_stats": self.monitor_stats
                }
                await self.redis.set(analysis_key, json.dumps(analysis_data))
                
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Performance analyzer error: {e}")
                await asyncio.sleep(300)
    
    async def _resource_collector(self):
        """Collect system resource metrics"""
        while not self._shutdown_event.is_set():
            try:
                # Simulate resource collection (in practice, would use psutil or similar)
                import random
                cpu_percent = random.uniform(20, 80)
                memory_percent = random.uniform(30, 70)
                network_bytes = random.uniform(1000, 10000)
                
                self.resource_monitor.record_usage(cpu_percent, memory_percent, network_bytes)
                
                # Check for resource alerts
                resource_summary = self.resource_monitor.get_resource_summary()
                
                for resource_type, metrics in resource_summary.items():
                    if metrics["current"] > self.alert_thresholds["resource_utilization"] * 100:
                        await self._generate_resource_alert(resource_type, metrics["current"])
                
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                self.logger.error(f"Resource collector error: {e}")
                await asyncio.sleep(60)
    
    async def _generate_resource_alert(self, resource_type: str, usage_percent: float):
        """Generate alert for high resource usage"""
        try:
            alert_id = f"resource_{resource_type}_{int(time.time())}"
            
            alert = Alert(
                alert_id=alert_id,
                training_id="system",
                severity=AlertSeverity.WARNING,
                title=f"High {resource_type.title()} Usage",
                description=f"{resource_type.title()} usage at {usage_percent:.1f}%",
                timestamp=datetime.utcnow(),
                metadata={
                    "resource_type": resource_type,
                    "usage_percent": usage_percent,
                    "threshold": self.alert_thresholds["resource_utilization"] * 100
                }
            )
            
            self.alerts[alert_id] = alert
            await self._store_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Failed to generate resource alert: {e}")
    
    async def _data_persistence(self):
        """Persist monitoring data to Redis"""
        while not self._shutdown_event.is_set():
            try:
                # Store training progress
                for training_id, progress in self.training_progress.items():
                    progress_key = f"{self.namespace}:progress:{training_id}"
                    await self.redis.set(progress_key, json.dumps(progress.to_dict()))
                
                # Store client performances
                for client_id, performance in self.client_performances.items():
                    client_key = f"{self.namespace}:client:{client_id}"
                    await self.redis.set(client_key, json.dumps(performance.to_dict()))
                
                # Store monitor statistics
                stats_key = f"{self.namespace}:stats"
                await self.redis.set(stats_key, json.dumps(self.monitor_stats))
                
                await asyncio.sleep(300)  # Persist every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Data persistence error: {e}")
                await asyncio.sleep(300)
    
    # Public API methods
    def get_training_progress(self, training_id: str) -> Optional[Dict[str, Any]]:
        """Get training progress"""
        progress = self.training_progress.get(training_id)
        return progress.to_dict() if progress else None
    
    def get_client_performance(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get client performance"""
        performance = self.client_performances.get(client_id)
        return performance.to_dict() if performance else None
    
    def get_training_metrics(self, training_id: str, metric_type: MetricType = None) -> List[Dict[str, Any]]:
        """Get training metrics"""
        metrics = []
        
        if metric_type:
            metric_key = f"{training_id}:{metric_type.value}"
            if metric_key in self.training_metrics:
                metrics = [m.to_dict() for m in self.training_metrics[metric_key]]
        else:
            # Get all metrics for training
            for mt in MetricType:
                metric_key = f"{training_id}:{mt.value}"
                if metric_key in self.training_metrics:
                    metrics.extend([m.to_dict() for m in self.training_metrics[metric_key]])
        
        return sorted(metrics, key=lambda x: x['timestamp'])
    
    def get_alerts(self, training_id: str = None, severity: AlertSeverity = None) -> List[Dict[str, Any]]:
        """Get alerts"""
        alerts = []
        
        for alert in self.alerts.values():
            if training_id and alert.training_id != training_id:
                continue
            
            if severity and alert.severity != severity:
                continue
            
            alerts.append(alert.to_dict())
        
        return sorted(alerts, key=lambda x: x['timestamp'], reverse=True)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        resource_summary = self.resource_monitor.get_resource_summary()
        active_alerts = len([a for a in self.alerts.values() if not a.resolved])
        
        # Calculate health score
        health_factors = []
        
        # Resource health
        if resource_summary:
            avg_cpu = resource_summary.get("cpu", {}).get("average", 0)
            avg_memory = resource_summary.get("memory", {}).get("average", 0)
            resource_health = 1.0 - max(avg_cpu, avg_memory) / 100.0
            health_factors.append(resource_health)
        
        # Alert health (fewer alerts = better health)
        alert_health = max(0.0, 1.0 - active_alerts / 10.0)  # Normalize to 0-1
        health_factors.append(alert_health)
        
        # Training health (based on convergence status)
        converged_trainings = sum(1 for p in self.training_progress.values() 
                                if p.convergence_status in ["converged", "improving"])
        total_trainings = len(self.training_progress)
        training_health = converged_trainings / total_trainings if total_trainings > 0 else 1.0
        health_factors.append(training_health)
        
        overall_health = statistics.mean(health_factors) if health_factors else 1.0
        
        return {
            "overall_health_score": overall_health,
            "health_status": "healthy" if overall_health > 0.8 else "warning" if overall_health > 0.6 else "critical",
            "active_alerts": active_alerts,
            "resource_summary": resource_summary,
            "training_summary": {
                "total_trainings": len(self.training_progress),
                "converged_trainings": converged_trainings,
                "active_clients": len(self.client_performances)
            },
            "monitor_stats": self.monitor_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_performance_prediction(self, training_id: str) -> Optional[Dict[str, Any]]:
        """Get performance prediction for a training"""
        accuracy_key = f"{training_id}:accuracy"
        if accuracy_key not in self.training_metrics:
            return None
        
        accuracy_history = [m.value for m in self.training_metrics[accuracy_key]]
        return self.performance_predictor.predict_convergence(training_id, accuracy_history)
    
    async def shutdown(self):
        """Shutdown the federated monitor"""
        self.logger.info("Shutting down Federated Learning Monitor")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Close Redis connection
        if self.redis:
            await self.redis.close()
        
        self.logger.info("Federated Learning Monitor shutdown complete")
