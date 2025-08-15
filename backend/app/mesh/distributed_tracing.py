"""
Distributed Tracing for Service Mesh
Provides request correlation, latency tracking, and service dependency mapping
"""
from __future__ import annotations

import time
import json
import uuid
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import httpx
from prometheus_client import Histogram, Counter

logger = logging.getLogger(__name__)

# Metrics for tracing
trace_duration = Histogram('mesh_trace_duration_seconds', 'Trace duration', ['service', 'operation'])
span_counter = Counter('mesh_spans_total', 'Total spans created', ['service', 'type'])
trace_errors = Counter('mesh_trace_errors_total', 'Trace errors', ['service', 'error_type'])

class SpanType(Enum):
    """Types of spans in distributed tracing"""
    CLIENT = "client"
    SERVER = "server"
    PRODUCER = "producer"
    CONSUMER = "consumer"
    INTERNAL = "internal"

class SpanStatus(Enum):
    """Status of a span"""
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

@dataclass
class SpanContext:
    """Context for a distributed trace span"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    flags: int = 0
    baggage: Dict[str, str] = field(default_factory=dict)
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers for propagation"""
        headers = {
            "X-Trace-Id": self.trace_id,
            "X-Span-Id": self.span_id,
            "X-Trace-Flags": str(self.flags)
        }
        
        if self.parent_span_id:
            headers["X-Parent-Span-Id"] = self.parent_span_id
        
        if self.baggage:
            headers["X-Trace-Baggage"] = json.dumps(self.baggage)
        
        return headers
    
    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> Optional['SpanContext']:
        """Create context from HTTP headers"""
        trace_id = headers.get("X-Trace-Id")
        span_id = headers.get("X-Span-Id")
        
        if not trace_id:
            return None
        
        return cls(
            trace_id=trace_id,
            span_id=span_id or str(uuid.uuid4()),
            parent_span_id=headers.get("X-Parent-Span-Id"),
            flags=int(headers.get("X-Trace-Flags", "0")),
            baggage=json.loads(headers.get("X-Trace-Baggage", "{}"))
        )

@dataclass
class Span:
    """Represents a span in a distributed trace"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    span_type: SpanType
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: SpanStatus = SpanStatus.OK
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    
    def finish(self, status: SpanStatus = SpanStatus.OK):
        """Finish the span"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = status
        
        # Record metrics
        trace_duration.labels(service=self.service_name, operation=self.operation_name).observe(self.duration)
        span_counter.labels(service=self.service_name, type=self.span_type.value).inc()
        
        if status == SpanStatus.ERROR:
            trace_errors.labels(service=self.service_name, error_type="span_error").inc()
    
    def add_tag(self, key: str, value: Any):
        """Add a tag to the span"""
        self.tags[key] = value
    
    def add_log(self, message: str, level: str = "info", **kwargs):
        """Add a log entry to the span"""
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.logs.append(log_entry)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for storage/transmission"""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "service_name": self.service_name,
            "span_type": self.span_type.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "status": self.status.value if self.status else None,
            "tags": self.tags,
            "logs": self.logs,
            "references": self.references
        }

class TraceCollector:
    """Collects and stores distributed traces"""
    
    def __init__(self, max_traces: int = 10000):
        self.max_traces = max_traces
        self.traces: Dict[str, List[Span]] = {}
        self.span_index: Dict[str, Span] = {}
        
    async def add_span(self, span: Span):
        """Add a span to the trace"""
        trace_id = span.trace_id
        
        if trace_id not in self.traces:
            self.traces[trace_id] = []
        
        self.traces[trace_id].append(span)
        self.span_index[span.span_id] = span
        
        # Limit number of stored traces
        if len(self.traces) > self.max_traces:
            oldest_trace = min(self.traces.keys())
            self._remove_trace(oldest_trace)
    
    def _remove_trace(self, trace_id: str):
        """Remove a trace and its spans"""
        if trace_id in self.traces:
            for span in self.traces[trace_id]:
                if span.span_id in self.span_index:
                    del self.span_index[span.span_id]
            del self.traces[trace_id]
    
    def get_trace(self, trace_id: str) -> Optional[List[Span]]:
        """Get all spans for a trace"""
        return self.traces.get(trace_id)
    
    def get_span(self, span_id: str) -> Optional[Span]:
        """Get a specific span"""
        return self.span_index.get(span_id)
    
    def get_service_dependencies(self) -> Dict[str, List[str]]:
        """Analyze traces to determine service dependencies"""
        dependencies = {}
        
        for trace_spans in self.traces.values():
            for span in trace_spans:
                if span.parent_span_id:
                    parent = self.span_index.get(span.parent_span_id)
                    if parent and parent.service_name != span.service_name:
                        if parent.service_name not in dependencies:
                            dependencies[parent.service_name] = []
                        if span.service_name not in dependencies[parent.service_name]:
                            dependencies[parent.service_name].append(span.service_name)
        
        return dependencies
    
    def get_trace_summary(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a trace"""
        spans = self.get_trace(trace_id)
        if not spans:
            return None
        
        # Find root span
        root_span = None
        for span in spans:
            if not span.parent_span_id:
                root_span = span
                break
        
        if not root_span:
            root_span = spans[0]
        
        # Calculate total duration
        min_start = min(s.start_time for s in spans)
        max_end = max(s.end_time for s in spans if s.end_time)
        total_duration = max_end - min_start if max_end else None
        
        # Count errors
        error_count = sum(1 for s in spans if s.status == SpanStatus.ERROR)
        
        # Get service breakdown
        service_times = {}
        for span in spans:
            if span.duration:
                if span.service_name not in service_times:
                    service_times[span.service_name] = 0
                service_times[span.service_name] += span.duration
        
        return {
            "trace_id": trace_id,
            "root_operation": root_span.operation_name,
            "start_time": min_start,
            "total_duration": total_duration,
            "span_count": len(spans),
            "error_count": error_count,
            "services_involved": list(set(s.service_name for s in spans)),
            "service_times": service_times,
            "status": "error" if error_count > 0 else "success"
        }
    
    def search_traces(
        self,
        service_name: Optional[str] = None,
        operation_name: Optional[str] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        error_only: bool = False,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search for traces matching criteria"""
        results = []
        
        for trace_id, spans in self.traces.items():
            # Check if trace matches criteria
            matches = True
            
            if service_name:
                if not any(s.service_name == service_name for s in spans):
                    matches = False
            
            if operation_name:
                if not any(s.operation_name == operation_name for s in spans):
                    matches = False
            
            if error_only:
                if not any(s.status == SpanStatus.ERROR for s in spans):
                    matches = False
            
            if matches:
                summary = self.get_trace_summary(trace_id)
                if summary:
                    if min_duration and summary.get("total_duration", 0) < min_duration:
                        continue
                    if max_duration and summary.get("total_duration", float('inf')) > max_duration:
                        continue
                    
                    results.append(summary)
                    
                    if len(results) >= limit:
                        break
        
        return results

class Tracer:
    """Main tracer for creating and managing spans"""
    
    def __init__(self, service_name: str, collector: Optional[TraceCollector] = None):
        self.service_name = service_name
        self.collector = collector or TraceCollector()
        self.active_spans: Dict[str, Span] = {}
        
    def start_span(
        self,
        operation_name: str,
        context: Optional[SpanContext] = None,
        span_type: SpanType = SpanType.INTERNAL,
        tags: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Start a new span"""
        if context:
            trace_id = context.trace_id
            parent_span_id = context.span_id
        else:
            trace_id = str(uuid.uuid4())
            parent_span_id = None
        
        span = Span(
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            service_name=self.service_name,
            span_type=span_type,
            start_time=time.time(),
            tags=tags or {}
        )
        
        self.active_spans[span.span_id] = span
        return span
    
    async def finish_span(self, span: Span, status: SpanStatus = SpanStatus.OK):
        """Finish a span and send to collector"""
        span.finish(status)
        
        if span.span_id in self.active_spans:
            del self.active_spans[span.span_id]
        
        await self.collector.add_span(span)
    
    def inject_context(self, span: Span, headers: Dict[str, str]):
        """Inject span context into headers for propagation"""
        context = SpanContext(
            trace_id=span.trace_id,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id
        )
        headers.update(context.to_headers())
    
    def extract_context(self, headers: Dict[str, str]) -> Optional[SpanContext]:
        """Extract span context from headers"""
        return SpanContext.from_headers(headers)

class TracingInterceptor:
    """HTTP interceptor for automatic tracing"""
    
    def __init__(self, tracer: Tracer):
        self.tracer = tracer
    
    async def request_interceptor(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Intercept outgoing requests to add tracing"""
        # Extract or create context
        context = self.tracer.extract_context(request.get("headers", {}))
        
        # Start client span
        span = self.tracer.start_span(
            operation_name=f"{request.get('method', 'GET')} {request.get('path', '/')}",
            context=context,
            span_type=SpanType.CLIENT,
            tags={
                "http.method": request.get("method", "GET"),
                "http.url": request.get("path", "/"),
                "service.target": request.get("service_name", "unknown")
            }
        )
        
        # Inject context into headers
        self.tracer.inject_context(span, request["headers"])
        
        # Store span for response handling
        request["_trace_span"] = span
        
        return request
    
    async def response_interceptor(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Intercept responses to complete tracing"""
        # Get span from request
        span = response.get("_trace_span")
        
        if span:
            # Add response tags
            span.add_tag("http.status_code", response.get("status_code"))
            
            # Determine status
            status = SpanStatus.OK
            if response.get("status_code", 200) >= 400:
                status = SpanStatus.ERROR
            
            # Finish span
            await self.tracer.finish_span(span, status)
        
        return response

# Global tracer instance
_tracer: Optional[Tracer] = None

def get_tracer(service_name: str = "backend-api") -> Tracer:
    """Get or create tracer instance"""
    global _tracer
    if _tracer is None:
        _tracer = Tracer(service_name)
    return _tracer