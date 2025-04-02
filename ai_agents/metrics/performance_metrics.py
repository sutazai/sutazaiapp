"""
Agent Performance Metrics Module

This module provides functionality for tracking, analyzing, and visualizing
agent performance metrics, enabling performance optimization and anomaly detection.
"""

import time
import logging
import threading
import json
import statistics
from enum import Enum
from typing import Dict, List, Any, Optional, Deque
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque, defaultdict
import os
import math

# Configure logging
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Type of performance metric."""

    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    ERROR_RATE = "error_rate"
    REQUEST_COUNT = "request_count"
    RESPONSE_TIME = "response_time"
    TOKEN_USAGE = "token_usage"
    TASK_SUCCESS_RATE = "task_success_rate"


@dataclass
class MetricPoint:
    """
    A single data point for a metric.
    """

    value: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMetricSummary:
    """
    Summary of agent metrics.
    """

    agent_id: str
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_active: datetime = field(default_factory=datetime.utcnow)
    execution_count: int = 0
    error_count: int = 0
    avg_execution_time: float = 0.0
    avg_response_time: float = 0.0
    task_success_rate: float = 0.0
    token_usage: int = 0
    status: str = "idle"
    anomalies: List[Dict[str, Any]] = field(default_factory=list)


class PerformanceMetrics:
    """
    Tracks and analyzes performance metrics for agents.

    This class collects various performance metrics for agents, calculates
    aggregate statistics, detects anomalies, and provides optimization suggestions.
    """

    def __init__(
        self,
        window_size: int = 1000,
        anomaly_threshold: float = 3.0,
        metrics_dir: str = "/opt/sutazaiapp/logs/metrics",
    ):
        """
        Initialize the performance metrics tracker.

        Args:
            window_size: Maximum number of data points to keep per metric
            anomaly_threshold: Number of standard deviations for anomaly detection
            metrics_dir: Directory to store metrics data
        """
        self.metrics: Dict[str, Dict[MetricType, Deque[MetricPoint]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=window_size))
        )
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self.metrics_dir = metrics_dir
        self.lock = threading.RLock()

        # Create metrics directory if it doesn't exist
        os.makedirs(metrics_dir, exist_ok=True)

        # Periodic saving
        self.save_interval = 300  # 5 minutes
        self.last_save_time = time.time()

        # Track aggregates for system-wide metrics
        self.system_metrics: Dict[MetricType, Deque[MetricPoint]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

    def add_metric(
        self,
        agent_id: str,
        metric_type: MetricType,
        value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a metric data point for an agent.

        Args:
            agent_id: ID of the agent
            metric_type: Type of metric
            value: Metric value
            metadata: Additional metadata for the metric
        """
        with self.lock:
            metric_point = MetricPoint(
                value=value, timestamp=time.time(), metadata=metadata or {}
            )

            # Add to agent-specific metrics
            self.metrics[agent_id][metric_type].append(metric_point)

            # Add to system-wide aggregates
            self.system_metrics[metric_type].append(metric_point)

            # Check if we should save metrics to disk
            current_time = time.time()
            if current_time - self.last_save_time > self.save_interval:
                self.save_metrics()
                self.last_save_time = current_time

    def record_execution_time(
        self, agent_id: str, execution_time: float, task_type: Optional[str] = None
    ) -> None:
        """
        Record the execution time for an agent task.

        Args:
            agent_id: ID of the agent
            execution_time: Execution time in milliseconds
            task_type: Type of task executed
        """
        metadata = {"task_type": task_type} if task_type else {}
        self.add_metric(
            agent_id=agent_id,
            metric_type=MetricType.EXECUTION_TIME,
            value=execution_time,
            metadata=metadata,
        )

    def record_error(
        self, agent_id: str, error_type: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an error for an agent.

        Args:
            agent_id: ID of the agent
            error_type: Type of error
            details: Error details
        """
        metadata = {"error_type": error_type, "details": details or {}}
        self.add_metric(
            agent_id=agent_id,
            metric_type=MetricType.ERROR_RATE,
            value=1.0,  # Increment error count
            metadata=metadata,
        )

    def record_resource_usage(
        self, agent_id: str, cpu_usage: float, memory_usage: float
    ) -> None:
        """
        Record resource usage for an agent.

        Args:
            agent_id: ID of the agent
            cpu_usage: CPU usage percentage
            memory_usage: Memory usage in MB
        """
        # Record CPU usage
        self.add_metric(
            agent_id=agent_id, metric_type=MetricType.CPU_USAGE, value=cpu_usage
        )

        # Record memory usage
        self.add_metric(
            agent_id=agent_id, metric_type=MetricType.MEMORY_USAGE, value=memory_usage
        )

    def record_task_completion(
        self,
        agent_id: str,
        success: bool,
        task_type: str,
        execution_time: float,
        token_usage: Optional[int] = None,
    ) -> None:
        """
        Record task completion for an agent.

        Args:
            agent_id: ID of the agent
            success: Whether the task was completed successfully
            task_type: Type of task
            execution_time: Execution time in milliseconds
            token_usage: Number of tokens used (for LLM-based agents)
        """
        # Record success/failure
        self.add_metric(
            agent_id=agent_id,
            metric_type=MetricType.TASK_SUCCESS_RATE,
            value=1.0 if success else 0.0,
            metadata={"task_type": task_type},
        )

        # Record execution time
        self.record_execution_time(
            agent_id=agent_id, execution_time=execution_time, task_type=task_type
        )

        # Record token usage if provided
        if token_usage is not None:
            self.add_metric(
                agent_id=agent_id,
                metric_type=MetricType.TOKEN_USAGE,
                value=token_usage,
                metadata={"task_type": task_type},
            )

    def get_agent_metrics(self, agent_id: str) -> AgentMetricSummary:
        """
        Get a summary of metrics for an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            AgentMetricSummary: Summary of agent metrics
        """
        with self.lock:
            if agent_id not in self.metrics:
                return AgentMetricSummary(agent_id=agent_id)

            # Calculate CPU usage (average of last 10 points)
            cpu_points = list(self.metrics[agent_id][MetricType.CPU_USAGE])
            cpu_usage = (
                statistics.mean([p.value for p in cpu_points[-10:]])
                if cpu_points
                else 0.0
            )

            # Calculate memory usage (average of last 10 points)
            memory_points = list(self.metrics[agent_id][MetricType.MEMORY_USAGE])
            memory_usage = (
                statistics.mean([p.value for p in memory_points[-10:]])
                if memory_points
                else 0.0
            )

            # Calculate execution count and time
            execution_points = list(self.metrics[agent_id][MetricType.EXECUTION_TIME])
            execution_count = len(execution_points)
            avg_execution_time = (
                statistics.mean([p.value for p in execution_points])
                if execution_points
                else 0.0
            )

            # Calculate error count
            error_points = list(self.metrics[agent_id][MetricType.ERROR_RATE])
            error_count = len(error_points)

            # Calculate task success rate
            success_points = list(self.metrics[agent_id][MetricType.TASK_SUCCESS_RATE])
            success_rate = (
                statistics.mean([p.value for p in success_points])
                if success_points
                else 0.0
            )

            # Calculate last active time
            all_points = []
            for metric_type in MetricType:
                all_points.extend(self.metrics[agent_id][metric_type])

            last_active = datetime.fromtimestamp(
                max([p.timestamp for p in all_points]) if all_points else time.time()
            )

            # Calculate token usage
            token_points = list(self.metrics[agent_id][MetricType.TOKEN_USAGE])
            token_usage = sum([p.value for p in token_points]) if token_points else 0

            # Check for anomalies
            anomalies = self.detect_anomalies(agent_id)

            return AgentMetricSummary(
                agent_id=agent_id,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                last_active=last_active,
                execution_count=execution_count,
                error_count=error_count,
                avg_execution_time=avg_execution_time,
                task_success_rate=success_rate,
                token_usage=token_usage,
                anomalies=anomalies,
            )

    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get system-wide metrics.

        Returns:
            Dict[str, Any]: System-wide metrics
        """
        with self.lock:
            # Total agents
            total_agents = len(self.metrics)

            # Active agents (agents with activity in the last hour)
            current_time = time.time()
            active_agents = 0

            for agent_id in self.metrics:
                all_points = []
                for metric_type in MetricType:
                    all_points.extend(self.metrics[agent_id][metric_type])

                if (
                    all_points
                    and max([p.timestamp for p in all_points]) > current_time - 3600
                ):
                    active_agents += 1

            # Average metrics
            cpu_points = list(self.system_metrics[MetricType.CPU_USAGE])
            avg_cpu = (
                statistics.mean([p.value for p in cpu_points[-100:]])
                if cpu_points
                else 0.0
            )

            memory_points = list(self.system_metrics[MetricType.MEMORY_USAGE])
            avg_memory = (
                statistics.mean([p.value for p in memory_points[-100:]])
                if memory_points
                else 0.0
            )

            execution_points = list(self.system_metrics[MetricType.EXECUTION_TIME])
            avg_execution_time = (
                statistics.mean([p.value for p in execution_points[-100:]])
                if execution_points
                else 0.0
            )

            # Error rate
            error_points = list(self.system_metrics[MetricType.ERROR_RATE])
            success_points = list(self.system_metrics[MetricType.TASK_SUCCESS_RATE])

            error_rate = len(error_points) / max(len(success_points), 1)

            # Token usage
            token_points = list(self.system_metrics[MetricType.TOKEN_USAGE])
            total_tokens = sum([p.value for p in token_points]) if token_points else 0

            return {
                "total_agents": total_agents,
                "active_agents": active_agents,
                "avg_cpu_usage": avg_cpu,
                "avg_memory_usage": avg_memory,
                "avg_execution_time": avg_execution_time,
                "error_rate": error_rate,
                "total_token_usage": total_tokens,
                "timestamp": datetime.utcnow().isoformat(),
            }

    def detect_anomalies(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Detect anomalies in agent metrics.

        Args:
            agent_id: ID of the agent

        Returns:
            List[Dict[str, Any]]: List of detected anomalies
        """
        anomalies = []

        if agent_id not in self.metrics:
            return anomalies

        for metric_type in MetricType:
            points = list(self.metrics[agent_id][metric_type])

            if len(points) < 10:  # Need at least 10 points for anomaly detection
                continue

            values = [p.value for p in points]

            try:
                mean = statistics.mean(values)
                stdev = statistics.stdev(values)

                if stdev == 0:
                    continue  # Skip if standard deviation is 0

                # Check last 5 points for anomalies
                for point in points[-5:]:
                    z_score = abs(point.value - mean) / stdev

                    if z_score > self.anomaly_threshold:
                        anomalies.append(
                            {
                                "metric_type": metric_type.value,
                                "value": point.value,
                                "timestamp": datetime.fromtimestamp(
                                    point.timestamp
                                ).isoformat(),
                                "z_score": z_score,
                                "mean": mean,
                                "stdev": stdev,
                            }
                        )
            except (statistics.StatisticsError, ZeroDivisionError):
                continue

        return anomalies

    def get_performance_suggestions(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Get performance optimization suggestions for an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            List[Dict[str, Any]]: List of performance suggestions
        """
        suggestions = []

        if agent_id not in self.metrics:
            return suggestions

        # Check execution time
        execution_points = list(self.metrics[agent_id][MetricType.EXECUTION_TIME])
        if execution_points:
            avg_time = statistics.mean([p.value for p in execution_points])

            if avg_time > 1000:  # More than 1 second
                suggestions.append(
                    {
                        "type": "execution_time",
                        "severity": "high" if avg_time > 5000 else "medium",
                        "description": f"High average execution time: {avg_time:.2f}ms",
                        "recommendation": "Consider optimizing task processing or splitting into smaller tasks",
                    }
                )

        # Check error rate
        error_points = list(self.metrics[agent_id][MetricType.ERROR_RATE])
        success_points = list(self.metrics[agent_id][MetricType.TASK_SUCCESS_RATE])

        if success_points:
            error_rate = len(error_points) / len(success_points)

            if error_rate > 0.1:  # More than 10% errors
                suggestions.append(
                    {
                        "type": "error_rate",
                        "severity": "high" if error_rate > 0.3 else "medium",
                        "description": f"High error rate: {error_rate:.2%}",
                        "recommendation": "Review error patterns and implement error handling improvements",
                    }
                )

        # Check memory usage
        memory_points = list(self.metrics[agent_id][MetricType.MEMORY_USAGE])
        if memory_points and len(memory_points) > 10:
            recent_memory = [p.value for p in memory_points[-10:]]
            avg_memory = statistics.mean(recent_memory)

            if avg_memory > 500:  # More than 500MB
                suggestions.append(
                    {
                        "type": "memory_usage",
                        "severity": "medium",
                        "description": f"High memory usage: {avg_memory:.2f}MB",
                        "recommendation": "Check for memory leaks or optimize memory-intensive operations",
                    }
                )

        # Check token usage efficiency
        token_points = list(self.metrics[agent_id][MetricType.TOKEN_USAGE])
        if token_points and execution_points:
            tokens_per_task = sum([p.value for p in token_points]) / len(
                execution_points
            )

            if tokens_per_task > 1000:  # More than 1000 tokens per task
                suggestions.append(
                    {
                        "type": "token_usage",
                        "severity": "low",
                        "description": f"High token usage per task: {tokens_per_task:.2f}",
                        "recommendation": "Optimize prompts or implement caching for repetitive queries",
                    }
                )

        return suggestions

    def save_metrics(self) -> None:
        """Save metrics to disk."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.metrics_dir, f"metrics_{timestamp}.json")

            # Prepare data for serialization
            data = {
                "timestamp": datetime.utcnow().isoformat(),
                "system_metrics": self.get_system_metrics(),
                "agent_metrics": {},
            }

            # Add agent metrics
            for agent_id in self.metrics:
                data["agent_metrics"][agent_id] = self.get_agent_metrics(
                    agent_id
                ).__dict__

                # Convert datetime to string for serialization
                if isinstance(data["agent_metrics"][agent_id]["last_active"], datetime):
                    data["agent_metrics"][agent_id]["last_active"] = data[
                        "agent_metrics"
                    ][agent_id]["last_active"].isoformat()

            # Save to file
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Metrics saved to {filename}")

            # Clean up old files (keep last 24 files)
            self._cleanup_old_files()

        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    def _cleanup_old_files(self) -> None:
        """Clean up old metric files."""
        try:
            files = [
                os.path.join(self.metrics_dir, f)
                for f in os.listdir(self.metrics_dir)
                if f.startswith("metrics_") and f.endswith(".json")
            ]

            # Sort by modification time (oldest first)
            files.sort(key=lambda x: os.path.getmtime(x))

            # Delete all but the last 24 files
            if len(files) > 24:
                for file in files[:-24]:
                    os.remove(file)
                    logger.debug(f"Deleted old metrics file: {file}")

        except Exception as e:
            logger.error(f"Error cleaning up old metric files: {e}")

    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.

        Returns:
            Dict[str, Any]: Performance report
        """
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_metrics": self.get_system_metrics(),
            "agents": {},
            "anomalies": [],
            "suggestions": [],
        }

        # Generate agent-specific metrics and suggestions
        for agent_id in self.metrics:
            metrics = self.get_agent_metrics(agent_id)
            report["agents"][agent_id] = {
                "metrics": metrics.__dict__,
                "suggestions": self.get_performance_suggestions(agent_id),
            }

            # Convert datetime to string for serialization
            if isinstance(
                report["agents"][agent_id]["metrics"]["last_active"], datetime
            ):
                report["agents"][agent_id]["metrics"]["last_active"] = report["agents"][
                    agent_id
                ]["metrics"]["last_active"].isoformat()

            # Add agent anomalies to system-wide list
            for anomaly in metrics.anomalies:
                anomaly["agent_id"] = agent_id
                report["anomalies"].append(anomaly)

            # Add agent suggestions to system-wide list
            for suggestion in report["agents"][agent_id]["suggestions"]:
                suggestion["agent_id"] = agent_id
                report["suggestions"].append(suggestion)

        # Sort anomalies and suggestions by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        report["suggestions"].sort(
            key=lambda x: severity_order.get(x.get("severity", "low"), 3)
        )

        return report


class MetricsAnalyzer:
    """
    Analyzes metrics data to identify patterns, trends, and anomalies.

    This class provides advanced analytics on the collected metrics data,
    including trend analysis, correlation detection, and predictive analytics.
    """

    def __init__(self, metrics: PerformanceMetrics):
        """
        Initialize the metrics analyzer.

        Args:
            metrics: PerformanceMetrics instance
        """
        self.metrics = metrics

    def analyze_trends(
        self, agent_id: str, metric_type: MetricType, window: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze trends for a specific metric.

        Args:
            agent_id: ID of the agent
            metric_type: Type of metric to analyze
            window: Number of data points to analyze

        Returns:
            Dict[str, Any]: Trend analysis results
        """
        with self.metrics.lock:
            if agent_id not in self.metrics.metrics:
                return {"trend": "no_data"}

            points = list(self.metrics.metrics[agent_id][metric_type])

            if len(points) < 10:
                return {"trend": "insufficient_data"}

            # Get last N points
            points = points[-window:]
            values = [p.value for p in points]
            timestamps = [p.timestamp for p in points]

            # Calculate trend (simple linear regression)
            n = len(values)
            if n < 2:
                return {"trend": "insufficient_data"}

            # Normalize timestamps to make calculations more stable
            min_ts = min(timestamps)
            norm_ts = [ts - min_ts for ts in timestamps]

            # Calculate means
            mean_x = sum(norm_ts) / n
            mean_y = sum(values) / n

            # Calculate slope
            numerator = sum(
                (norm_ts[i] - mean_x) * (values[i] - mean_y) for i in range(n)
            )
            denominator = sum((norm_ts[i] - mean_x) ** 2 for i in range(n))

            if denominator == 0:
                slope = 0
            else:
                slope = numerator / denominator

            # Calculate y-intercept
            y_intercept = mean_y - slope * mean_x

            # Calculate R-squared
            y_pred = [slope * x + y_intercept for x in norm_ts]
            ss_total = sum((y - mean_y) ** 2 for y in values)
            ss_residual = sum((values[i] - y_pred[i]) ** 2 for i in range(n))

            if ss_total == 0:
                r_squared = 0
            else:
                r_squared = 1 - (ss_residual / ss_total)

            # Determine trend direction and strength
            if abs(slope) < 0.0001:
                trend = "stable"
            elif slope > 0:
                trend = "increasing"
            else:
                trend = "decreasing"

            if abs(r_squared) < 0.3:
                strength = "weak"
            elif abs(r_squared) < 0.7:
                strength = "moderate"
            else:
                strength = "strong"

            return {
                "trend": trend,
                "strength": strength,
                "slope": slope,
                "r_squared": r_squared,
                "data_points": n,
            }

    def detect_correlations(
        self, agent_id: str, metric_types: Optional[List[MetricType]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect correlations between different metrics.

        Args:
            agent_id: ID of the agent
            metric_types: List of metric types to analyze (None for all)

        Returns:
            List[Dict[str, Any]]: Detected correlations
        """
        correlations = []

        with self.metrics.lock:
            if agent_id not in self.metrics.metrics:
                return correlations

            if metric_types is None:
                metric_types = list(MetricType)

            # Check all pairs of metrics
            for i, metric1 in enumerate(metric_types):
                for j, metric2 in enumerate(metric_types[i + 1 :], i + 1):
                    points1 = list(self.metrics.metrics[agent_id][metric1])
                    points2 = list(self.metrics.metrics[agent_id][metric2])

                    if len(points1) < 10 or len(points2) < 10:
                        continue

                    # Match timestamps (approximately)
                    matched_points = []

                    for p1 in points1:
                        # Find closest point in points2
                        closest_p2 = min(
                            points2, key=lambda p: abs(p.timestamp - p1.timestamp)
                        )

                        # Only match if within 5 seconds
                        if abs(closest_p2.timestamp - p1.timestamp) < 5:
                            matched_points.append((p1.value, closest_p2.value))

                    if len(matched_points) < 10:
                        continue

                    # Calculate correlation coefficient
                    x_values = [p[0] for p in matched_points]
                    y_values = [p[1] for p in matched_points]

                    try:
                        correlation = self._calculate_correlation(x_values, y_values)

                        # Only report significant correlations
                        if abs(correlation) > 0.5:
                            correlations.append(
                                {
                                    "metric1": metric1.value,
                                    "metric2": metric2.value,
                                    "correlation": correlation,
                                    "strength": (
                                        "strong"
                                        if abs(correlation) > 0.7
                                        else "moderate"
                                    ),
                                    "direction": "positive"
                                    if correlation > 0
                                    else "negative",
                                    "data_points": len(matched_points),
                                }
                            )
                    except Exception as e:
                        # F821: Replaced undefined 'key' with metric names
                        logger.error(
                            f"Error calculating correlation between {metric1.value} and {metric2.value}: {e}"
                        )
                        continue

        return correlations

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """
        Calculate Pearson correlation coefficient.

        Args:
            x: First list of values
            y: Second list of values

        Returns:
            float: Correlation coefficient
        """
        n = len(x)

        if n != len(y) or n == 0:
            raise ValueError("Input lists must have the same non-zero length")

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        variance_x = sum((xi - mean_x) ** 2 for xi in x) / n
        variance_y = sum((yi - mean_y) ** 2 for yi in y) / n

        if variance_x == 0 or variance_y == 0:
            return 0

        covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n

        return covariance / (math.sqrt(variance_x) * math.sqrt(variance_y))

    def predict_future_performance(
        self, agent_id: str, metric_type: MetricType, time_horizon: int = 3600
    ) -> Dict[str, Any]:
        """
        Predict future performance based on historical data.

        Args:
            agent_id: ID of the agent
            metric_type: Type of metric to predict
            time_horizon: Time horizon for prediction in seconds

        Returns:
            Dict[str, Any]: Prediction results
        """
        with self.metrics.lock:
            if agent_id not in self.metrics.metrics:
                return {"prediction": "no_data"}

            points = list(self.metrics.metrics[agent_id][metric_type])

            if len(points) < 20:
                return {"prediction": "insufficient_data"}

            # Use last 100 points for prediction
            points = points[-100:]
            values = [p.value for p in points]
            timestamps = [p.timestamp for p in points]

            # Normalize timestamps
            min_ts = min(timestamps)
            norm_ts = [ts - min_ts for ts in timestamps]

            # Simple linear regression
            n = len(values)
            mean_x = sum(norm_ts) / n
            mean_y = sum(values) / n

            numerator = sum(
                (norm_ts[i] - mean_x) * (values[i] - mean_y) for i in range(n)
            )
            denominator = sum((norm_ts[i] - mean_x) ** 2 for i in range(n))

            if denominator == 0:
                slope = 0
            else:
                slope = numerator / denominator

            y_intercept = mean_y - slope * mean_x

            # Current values
            current_time = time.time()
            norm_current = current_time - min_ts
            current_value = slope * norm_current + y_intercept

            # Predicted values
            future_time = current_time + time_horizon
            norm_future = future_time - min_ts
            predicted_value = slope * norm_future + y_intercept

            # Calculate confidence interval (simple approach)
            y_pred = [slope * x + y_intercept for x in norm_ts]
            residuals = [values[i] - y_pred[i] for i in range(n)]
            residual_std = statistics.stdev(residuals) if len(residuals) > 1 else 0

            # 95% confidence interval is approximately 2 standard deviations
            confidence_interval = 2 * residual_std

            return {
                "current_value": current_value,
                "predicted_value": predicted_value,
                "confidence_interval": confidence_interval,
                "time_horizon": time_horizon,
                "prediction_time": datetime.fromtimestamp(future_time).isoformat(),
            }
