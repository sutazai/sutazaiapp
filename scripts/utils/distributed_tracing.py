#!/usr/bin/env python3
"""
Advanced Distributed Tracing System for SutazAI Agent Interactions
Implements OpenTelemetry-based tracing with correlation across microservices
"""

import json
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import uuid
import threading
from contextlib import contextmanager

import requests
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncIOInstrumentor

logger = logging.getLogger(__name__)

@dataclass
class AgentInteraction:
    """Data class for agent interaction tracking"""
    trace_id: str
    span_id: str
    agent_name: str
    interaction_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "active"
    input_data: Optional[Dict] = None
    output_data: Optional[Dict] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict] = None

class DistributedTracingManager:
    """
    Advanced distributed tracing manager for SutazAI agent ecosystem
    """
    
    def __init__(self, service_name: str = "sutazai-agents", jaeger_endpoint: str = "http://localhost:14268/api/traces"):
        self.service_name = service_name
        self.jaeger_endpoint = jaeger_endpoint
        self.active_traces: Dict[str, AgentInteraction] = {}
        self.trace_correlations: Dict[str, List[str]] = {}
        self.setup_tracing()
        
    def setup_tracing(self):
        """Initialize OpenTelemetry tracing"""
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": "1.0.0",
            "deployment.environment": "production"
        })
        
        trace.set_tracer_provider(TracerProvider(resource=resource))
        tracer = trace.get_tracer(__name__)
        
        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name="sutazai-jaeger",
            agent_port=14268,
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        # Auto-instrument common libraries
        RequestsInstrumentor().instrument()
        AsyncIOInstrumentor().instrument()
        
        self.tracer = tracer
        logger.info("Distributed tracing initialized successfully")
    
    @contextmanager
    def trace_agent_interaction(self, agent_name: str, interaction_type: str, **kwargs):
        """
        Context manager for tracing agent interactions
        """
        trace_id = str(uuid.uuid4())
        span_name = f"{agent_name}.{interaction_type}"
        
        with self.tracer.start_as_current_span(span_name) as span:
            span.set_attribute("agent.name", agent_name)
            span.set_attribute("interaction.type", interaction_type)
            span.set_attribute("trace.id", trace_id)
            
            interaction = AgentInteraction(
                trace_id=trace_id,
                span_id=span.get_span_context().span_id,
                agent_name=agent_name,
                interaction_type=interaction_type,
                start_time=datetime.utcnow(),
                metadata=kwargs
            )
            
            self.active_traces[trace_id] = interaction
            
            try:
                yield interaction
                interaction.status = "completed"
                interaction.end_time = datetime.utcnow()
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                interaction.status = "error"
                interaction.error_message = str(e)
                interaction.end_time = datetime.utcnow()
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            
            finally:
                self._record_interaction_metrics(interaction)
    
    def correlate_traces(self, parent_trace_id: str, child_trace_id: str):
        """
        Establish correlation between parent and child traces
        """
        if parent_trace_id not in self.trace_correlations:
            self.trace_correlations[parent_trace_id] = []
        
        self.trace_correlations[parent_trace_id].append(child_trace_id)
        
        # Add correlation attributes to spans
        if child_trace_id in self.active_traces:
            child_interaction = self.active_traces[child_trace_id]
            child_interaction.metadata = child_interaction.metadata or {}
            child_interaction.metadata["parent_trace_id"] = parent_trace_id
    
    def get_trace_tree(self, trace_id: str) -> Dict[str, Any]:
        """
        Get complete trace tree for analysis
        """
        trace_tree = {
            "root": self.active_traces.get(trace_id),
            "children": []
        }
        
        if trace_id in self.trace_correlations:
            for child_id in self.trace_correlations[trace_id]:
                child_tree = self.get_trace_tree(child_id)
                trace_tree["children"].append(child_tree)
        
        return trace_tree
    
    def _record_interaction_metrics(self, interaction: AgentInteraction):
        """
        Record interaction metrics for monitoring
        """
        duration = (interaction.end_time - interaction.start_time).total_seconds()
        
        # Send metrics to Prometheus
        self._send_prometheus_metrics({
            "agent_interaction_duration_seconds": duration,
            "agent_interaction_total": 1,
            "agent_interaction_errors_total": 1 if interaction.status == "error" else 0
        }, {
            "agent_name": interaction.agent_name,
            "interaction_type": interaction.interaction_type,
            "status": interaction.status
        })
    
    def _send_prometheus_metrics(self, metrics: Dict[str, float], labels: Dict[str, str]):
        """
        Send custom metrics to Prometheus pushgateway
        """
        try:
            from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
            
            registry = CollectorRegistry()
            
            for metric_name, value in metrics.items():
                gauge = Gauge(metric_name, f"SutazAI {metric_name}", labelnames=list(labels.keys()), registry=registry)
                gauge.labels(**labels).set(value)
            
            push_to_gateway('sutazai-prometheus-pushgateway:9091', job='sutazai-tracing', registry=registry)
            
        except Exception as e:
            logger.warning(f"Failed to send metrics to Prometheus: {e}")

class AgentInteractionAnalyzer:
    """
    Analyzer for agent interaction patterns and performance
    """
    
    def __init__(self, tracing_manager: DistributedTracingManager):
        self.tracing_manager = tracing_manager
        self.interaction_patterns: Dict[str, List[Dict]] = {}
        
    def analyze_interaction_patterns(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Analyze agent interaction patterns over time window
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=time_window_hours)
        
        patterns = {
            "frequent_interactions": self._get_frequent_interactions(start_time, end_time),
            "bottleneck_agents": self._identify_bottleneck_agents(start_time, end_time),
            "interaction_chains": self._analyze_interaction_chains(start_time, end_time),
            "error_patterns": self._analyze_error_patterns(start_time, end_time)
        }
        
        return patterns
    
    def _get_frequent_interactions(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get most frequent agent interactions"""
        interaction_counts = {}
        
        for interaction in self.tracing_manager.active_traces.values():
            if start_time <= interaction.start_time <= end_time:
                key = f"{interaction.agent_name}.{interaction.interaction_type}"
                interaction_counts[key] = interaction_counts.get(key, 0) + 1
        
        return sorted([
            {"interaction": k, "count": v}
            for k, v in interaction_counts.items()
        ], key=lambda x: x["count"], reverse=True)[:10]
    
    def _identify_bottleneck_agents(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Identify agents with high latency or error rates"""
        agent_metrics = {}
        
        for interaction in self.tracing_manager.active_traces.values():
            if start_time <= interaction.start_time <= end_time and interaction.end_time:
                agent = interaction.agent_name
                duration = (interaction.end_time - interaction.start_time).total_seconds()
                
                if agent not in agent_metrics:
                    agent_metrics[agent] = {
                        "total_duration": 0,
                        "count": 0,
                        "errors": 0
                    }
                
                agent_metrics[agent]["total_duration"] += duration
                agent_metrics[agent]["count"] += 1
                if interaction.status == "error":
                    agent_metrics[agent]["errors"] += 1
        
        bottlenecks = []
        for agent, metrics in agent_metrics.items():
            avg_duration = metrics["total_duration"] / metrics["count"]
            error_rate = metrics["errors"] / metrics["count"]
            
            if avg_duration > 5.0 or error_rate > 0.1:  # 5s avg or 10% error rate
                bottlenecks.append({
                    "agent": agent,
                    "avg_duration": avg_duration,
                    "error_rate": error_rate,
                    "total_interactions": metrics["count"]
                })
        
        return sorted(bottlenecks, key=lambda x: x["avg_duration"], reverse=True)
    
    def _analyze_interaction_chains(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Analyze chains of agent interactions"""
        chains = []
        
        for parent_trace_id, child_trace_ids in self.tracing_manager.trace_correlations.items():
            if parent_trace_id in self.tracing_manager.active_traces:
                parent = self.tracing_manager.active_traces[parent_trace_id]
                if start_time <= parent.start_time <= end_time:
                    chain = {
                        "parent": parent.agent_name,
                        "children": [],
                        "total_duration": 0,
                        "chain_length": len(child_trace_ids) + 1
                    }
                    
                    for child_id in child_trace_ids:
                        if child_id in self.tracing_manager.active_traces:
                            child = self.tracing_manager.active_traces[child_id]
                            chain["children"].append(child.agent_name)
                    
                    chains.append(chain)
        
        return sorted(chains, key=lambda x: x["chain_length"], reverse=True)[:10]
    
    def _analyze_error_patterns(self, start_time: datetime, end_time: datetime) -> Dict[str, int]:
        """Analyze error patterns"""
        error_patterns = {}
        
        for interaction in self.tracing_manager.active_traces.values():
            if (start_time <= interaction.start_time <= end_time and 
                interaction.status == "error" and interaction.error_message):
                
                # Categorize errors
                error_type = self._categorize_error(interaction.error_message)
                error_patterns[error_type] = error_patterns.get(error_type, 0) + 1
        
        return error_patterns
    
    def _categorize_error(self, error_message: str) -> str:
        """Categorize error messages into types"""
        error_message_lower = error_message.lower()
        
        if "timeout" in error_message_lower:
            return "timeout_error"
        elif "connection" in error_message_lower:
            return "connection_error"
        elif "memory" in error_message_lower:
            return "memory_error"
        elif "permission" in error_message_lower:
            return "permission_error"
        else:
            return "unknown_error"

# Example usage and integration
def setup_agent_tracing():
    """
    Setup distributed tracing for SutazAI agents
    """
    tracing_manager = DistributedTracingManager()
    analyzer = AgentInteractionAnalyzer(tracing_manager)
    
    return tracing_manager, analyzer

# FastAPI middleware for automatic tracing
from fastapi import FastAPI, Request
from fastapi.middleware.base import BaseHTTPMiddleware

class TracingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, tracing_manager: DistributedTracingManager):
        super().__init__(app)
        self.tracing_manager = tracing_manager
    
    async def dispatch(self, request: Request, call_next):
        agent_name = request.headers.get("X-Agent-Name", "unknown")
        interaction_type = f"{request.method}_{request.url.path}"
        
        with self.tracing_manager.trace_agent_interaction(
            agent_name=agent_name,
            interaction_type=interaction_type,
            request_path=str(request.url.path),
            request_method=request.method
        ) as interaction:
            response = await call_next(request)
            interaction.output_data = {"status_code": response.status_code}
            return response

if __name__ == "__main__":
    # Initialize tracing system
    tracing_manager, analyzer = setup_agent_tracing()
    
    # Example usage
    with tracing_manager.trace_agent_interaction("ai-backend-developer", "code_generation") as interaction:
        # Simulate agent work
        time.sleep(2)
        interaction.output_data = {"lines_generated": 150}
    
    # Analyze patterns
    patterns = analyzer.analyze_interaction_patterns()
    print(json.dumps(patterns, indent=2, default=str))