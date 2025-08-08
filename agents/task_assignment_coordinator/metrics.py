"""
Compatibility shim: reuse canonical metrics from agents.core.metrics.
"""

from agents.core.metrics import AgentMetrics, setup_metrics_endpoint, MetricsTimer

__all__ = [
    'AgentMetrics',
    'setup_metrics_endpoint',
    'MetricsTimer',
]

