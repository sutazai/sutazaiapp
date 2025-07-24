"""
Agent Analytics Router

This module provides REST API endpoints for tracking, analyzing, and visualizing
agent performance metrics, enabling performance optimization and anomaly detection.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Path
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

from ai_agents.metrics.performance_metrics import (
    PerformanceMetrics,
    MetricsAnalyzer,
    MetricType,
)
from ai_agents.dependencies import get_performance_metrics


router = APIRouter()


class MetricTypeEnum(str, Enum):
    """Enum for metric types."""

    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    ERROR_RATE = "error_rate"
    REQUEST_COUNT = "request_count"
    RESPONSE_TIME = "response_time"
    TOKEN_USAGE = "token_usage"
    TASK_SUCCESS_RATE = "task_success_rate"


class AgentMetricsResponse(BaseModel):
    """Response model for agent metrics."""

    agent_id: str = Field(..., description="Agent ID")
    cpu_usage: float = Field(..., description="CPU usage (percentage)")
    memory_usage: float = Field(..., description="Memory usage (MB)")
    last_active: str = Field(..., description="Last active timestamp")
    execution_count: int = Field(..., description="Number of executions")
    error_count: int = Field(..., description="Number of errors")
    avg_execution_time: float = Field(..., description="Average execution time (ms)")
    avg_response_time: float = Field(..., description="Average response time (ms)")
    task_success_rate: float = Field(..., description="Task success rate (percentage)")
    token_usage: int = Field(..., description="Token usage count")
    status: str = Field(..., description="Current status")
    anomalies: List[Dict[str, Any]] = Field(..., description="Detected anomalies")


class SystemMetricsResponse(BaseModel):
    """Response model for system metrics."""

    total_agents: int = Field(..., description="Total number of agents")
    active_agents: int = Field(..., description="Number of active agents")
    total_tasks: int = Field(..., description="Total number of tasks")
    completed_tasks: int = Field(..., description="Number of completed tasks")
    failed_tasks: int = Field(..., description="Number of failed tasks")
    avg_system_cpu: float = Field(
        ..., description="Average system CPU usage (percentage)"
    )
    avg_system_memory: float = Field(
        ..., description="Average system memory usage (MB)"
    )
    total_token_usage: int = Field(..., description="Total token usage")
    error_rate: float = Field(..., description="System error rate (percentage)")
    start_time: str = Field(..., description="System start time")
    uptime_seconds: int = Field(..., description="System uptime in seconds")


class PerformanceSuggestion(BaseModel):
    """Model for performance suggestion."""

    suggestion: str = Field(..., description="Suggestion text")
    metric_type: str = Field(..., description="Related metric type")
    priority: int = Field(..., description="Suggestion priority (1-5, 1 is highest)")
    expected_improvement: float = Field(
        ..., description="Expected improvement percentage"
    )
    implementation_complexity: int = Field(
        ..., description="Implementation complexity (1-5, 1 is easiest)"
    )


class AnomalyModel(BaseModel):
    """Model for anomaly detection result."""

    agent_id: str = Field(..., description="Agent ID")
    metric_type: str = Field(..., description="Metric type with anomaly")
    value: float = Field(..., description="Anomalous value")
    expected_value: float = Field(..., description="Expected value")
    deviation: float = Field(..., description="Deviation from expected value")
    timestamp: str = Field(..., description="Timestamp of anomaly")
    severity: int = Field(..., description="Severity level (1-5, 5 is most severe)")
    description: str = Field(..., description="Anomaly description")


class TrendAnalysisResponse(BaseModel):
    """Response model for trend analysis."""

    agent_id: str = Field(..., description="Agent ID")
    metric_type: str = Field(..., description="Metric type analyzed")
    trend: str = Field(
        ..., description="Trend direction (increasing, decreasing, stable)"
    )
    data_points: List[Dict[str, Any]] = Field(
        ..., description="Data points for visualization"
    )
    window_size: int = Field(..., description="Analysis window size")
    min_value: float = Field(..., description="Minimum value in window")
    max_value: float = Field(..., description="Maximum value in window")
    avg_value: float = Field(..., description="Average value in window")
    regression: Dict[str, Any] = Field(..., description="Regression analysis results")


@router.get("/agents/{agent_id}/metrics", response_model=AgentMetricsResponse)
async def get_agent_metrics(
    agent_id: str = Path(..., description="Agent ID"),
    performance_metrics: PerformanceMetrics = Depends(get_performance_metrics),
):
    """
    Get performance metrics for a specific agent.
    """
    try:
        metrics = performance_metrics.get_agent_metrics(agent_id)
        if not metrics:
            raise HTTPException(
                status_code=404, detail=f"Metrics not found for agent: {agent_id}"
            )

        # Convert datetime to string
        metrics_dict = metrics.__dict__
        metrics_dict["last_active"] = metrics.last_active.isoformat()

        return metrics_dict
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/system/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics(
    performance_metrics: PerformanceMetrics = Depends(get_performance_metrics),
):
    """
    Get system-wide performance metrics.
    """
    try:
        metrics = performance_metrics.get_system_metrics()

        # Convert datetime to string
        if "start_time" in metrics and isinstance(metrics["start_time"], datetime):
            metrics["start_time"] = metrics["start_time"].isoformat()

        return metrics
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/agents/{agent_id}/anomalies", response_model=List[AnomalyModel])
async def detect_anomalies(
    agent_id: str = Path(..., description="Agent ID"),
    performance_metrics: PerformanceMetrics = Depends(get_performance_metrics),
):
    """
    Detect anomalies in agent performance metrics.
    """
    try:
        anomalies = performance_metrics.detect_anomalies(agent_id)

        # Convert datetime to string in each anomaly
        for anomaly in anomalies:
            if "timestamp" in anomaly and isinstance(anomaly["timestamp"], datetime):
                anomaly["timestamp"] = anomaly["timestamp"].isoformat()

        return anomalies
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/agents/{agent_id}/suggestions", response_model=List[PerformanceSuggestion]
)
async def get_performance_suggestions(
    agent_id: str = Path(..., description="Agent ID"),
    performance_metrics: PerformanceMetrics = Depends(get_performance_metrics),
):
    """
    Get performance optimization suggestions for an agent.
    """
    try:
        suggestions = performance_metrics.get_performance_suggestions(agent_id)
        return suggestions
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/report", response_model=Dict[str, Any])
async def generate_performance_report(
    performance_metrics: PerformanceMetrics = Depends(get_performance_metrics),
):
    """
    Generate a comprehensive performance report for all agents.
    """
    try:
        report = performance_metrics.generate_performance_report()

        # Convert datetime objects to strings throughout the report
        def convert_datetimes(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, datetime):
                        obj[key] = value.isoformat()
                    elif isinstance(value, (dict, list)):
                        convert_datetimes(value)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, datetime):
                        obj[i] = item.isoformat()
                    elif isinstance(item, (dict, list)):
                        convert_datetimes(item)

        convert_datetimes(report)
        return report
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/agents/{agent_id}/trends/{metric_type}", response_model=TrendAnalysisResponse
)
async def analyze_metric_trends(
    agent_id: str = Path(..., description="Agent ID"),
    metric_type: MetricTypeEnum = Path(..., description="Metric type to analyze"),
    window: int = Query(100, description="Analysis window size"),
    performance_metrics: PerformanceMetrics = Depends(get_performance_metrics),
):
    """
    Analyze trends in a specific metric for an agent.
    """
    try:
        # Create metrics analyzer
        analyzer = MetricsAnalyzer(performance_metrics)

        # Convert enum to MetricType
        metric_type_enum = MetricType(metric_type.value)

        # Analyze trends
        trend_analysis = analyzer.analyze_trends(
            agent_id=agent_id, metric_type=metric_type_enum, window=window
        )

        # Add agent_id and metric_type to response
        trend_analysis["agent_id"] = agent_id
        trend_analysis["metric_type"] = metric_type.value

        # Convert datetime objects to strings
        for point in trend_analysis.get("data_points", []):
            if "timestamp" in point and isinstance(point["timestamp"], datetime):
                point["timestamp"] = point["timestamp"].isoformat()

        return trend_analysis
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/agents/{agent_id}/correlations", response_model=List[Dict[str, Any]])
async def detect_metric_correlations(
    agent_id: str = Path(..., description="Agent ID"),
    metric_types: Optional[List[MetricTypeEnum]] = Query(
        None, description="Metric types to analyze"
    ),
    performance_metrics: PerformanceMetrics = Depends(get_performance_metrics),
):
    """
    Detect correlations between different metrics for an agent.
    """
    try:
        # Create metrics analyzer
        analyzer = MetricsAnalyzer(performance_metrics)

        # Convert enums to MetricType if provided
        metric_type_enums = None
        if metric_types:
            metric_type_enums = [MetricType(mt.value) for mt in metric_types]

        # Detect correlations
        correlations = analyzer.detect_correlations(
            agent_id=agent_id, metric_types=metric_type_enums
        )

        return correlations
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/agents/{agent_id}/predictions/{metric_type}", response_model=Dict[str, Any]
)
async def predict_future_performance(
    agent_id: str = Path(..., description="Agent ID"),
    metric_type: MetricTypeEnum = Path(..., description="Metric type to predict"),
    time_horizon: int = Query(3600, description="Time horizon in seconds"),
    performance_metrics: PerformanceMetrics = Depends(get_performance_metrics),
):
    """
    Predict future performance for a specific metric.
    """
    try:
        # Create metrics analyzer
        analyzer = MetricsAnalyzer(performance_metrics)

        # Convert enum to MetricType
        metric_type_enum = MetricType(metric_type.value)

        # Predict future performance
        prediction = analyzer.predict_future_performance(
            agent_id=agent_id, metric_type=metric_type_enum, time_horizon=time_horizon
        )

        # Convert datetime objects to strings
        for point in prediction.get("prediction_points", []):
            if "timestamp" in point and isinstance(point["timestamp"], datetime):
                point["timestamp"] = point["timestamp"].isoformat()

        return prediction
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
