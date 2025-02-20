#!/usr/bin/env python3
"""
SutazAI Advanced Logging and Monitoring System

Provides comprehensive logging, tracing, and monitoring capabilities
with advanced features like distributed tracing and performance tracking.
"""

import json
import logging
import os
import sys
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import opentelemetry
import structlog
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure base logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            "/opt/sutazai_project/SutazAI/logs/system_monitoring.log"
        ),
        logging.StreamHandler(sys.stdout),
    ],
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Configure OpenTelemetry tracing
trace.set_tracer_provider(TracerProvider())
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)
tracer = trace.get_tracer(__name__)


@dataclass
class LogEntry:
    """Comprehensive log entry tracking"""

    timestamp: str
    log_level: str
    message: str
    context: Dict[str, Any]
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    exception_info: Optional[str] = None


class AdvancedLogger:
    """
    Advanced logging system with distributed tracing and performance monitoring
    """

    def __init__(
        self,
        log_dir: str = "/opt/sutazai_project/SutazAI/logs",
        service_name: str = "SutazAI",
    ):
        """
        Initialize advanced logging system

        Args:
            log_dir (str): Base directory for log storage
            service_name (str): Name of the service being logged
        """
        self.log_dir = log_dir
        self.service_name = service_name

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Create structured logger
        self.logger = structlog.get_logger()

        # Performance tracking
        self._performance_logs: List[Dict[str, Any]] = []
        self._performance_lock = threading.Lock()

    def log(
        self,
        message: str,
        level: str = "info",
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
    ) -> LogEntry:
        """
        Create a comprehensive log entry

        Args:
            message (str): Log message
            level (str): Log level
            context (Dict, optional): Additional context
            exception (Exception, optional): Exception details

        Returns:
            Comprehensive log entry
        """
        # Get current trace and span
        current_span = trace.get_current_span()

        log_entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            log_level=level.upper(),
            message=message,
            context=context or {},
            trace_id=current_span.get_span_context().trace_id,
            span_id=current_span.get_span_context().span_id,
            exception_info=traceback.format_exc() if exception else None,
        )

        # Log using structured logging
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(
            message,
            **{
                "context": context or {},
                "trace_id": log_entry.trace_id,
                "span_id": log_entry.span_id,
            },
        )

        # Persist log entry
        self._persist_log_entry(log_entry)

        return log_entry

    def _persist_log_entry(self, log_entry: LogEntry):
        """
        Persist log entry to file

        Args:
            log_entry (LogEntry): Log entry to persist
        """
        log_file = os.path.join(
            self.log_dir,
            f'{self.service_name}_logs_{datetime.now().strftime("%Y%m%d")}.jsonl',
        )

        with open(log_file, "a") as f:
            f.write(json.dumps(asdict(log_entry)) + "\n")

    def trace(
        self, operation_name: str, context: Optional[Dict[str, Any]] = None
    ):
        """
        Create a distributed tracing span

        Args:
            operation_name (str): Name of the operation being traced
            context (Dict, optional): Additional context

        Returns:
            OpenTelemetry Span context
        """
        return tracer.start_as_current_span(
            operation_name, attributes=context or {}
        )

    def track_performance(
        self,
        operation_name: str,
        start_time: float,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Track performance of an operation

        Args:
            operation_name (str): Name of the operation
            start_time (float): Operation start time
            context (Dict, optional): Additional performance context
        """
        end_time = time.time()
        duration = end_time - start_time

        performance_log = {
            "operation": operation_name,
            "start_time": start_time,
            "end_time": end_time,
            "duration_ms": duration * 1000,
            "context": context or {},
        }

        with self._performance_lock:
            self._performance_logs.append(performance_log)

        # Log performance if exceeds threshold
        if duration > 1.0:  # Log operations taking more than 1 second
            self.log(
                f"Performance Warning: {operation_name} took {duration:.2f} seconds",
                level="warning",
                context=performance_log,
            )

    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report

        Returns:
            Performance analysis report
        """
        with self._performance_lock:
            performance_logs = self._performance_logs.copy()
            self._performance_logs.clear()

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_operations": len(performance_logs),
            "average_duration_ms": (
                sum(log["duration_ms"] for log in performance_logs)
                / len(performance_logs)
                if performance_logs
                else 0
            ),
            "slowest_operations": sorted(
                performance_logs, key=lambda x: x["duration_ms"], reverse=True
            )[:5],
        }

        # Persist performance report
        report_path = os.path.join(
            self.log_dir,
            f'{self.service_name}_performance_{datetime.now().strftime("%Y%m%d")}.json',
        )

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report


def main():
    """Demonstration of advanced logging capabilities"""
    logger = AdvancedLogger()

    # Example logging and tracing
    with logger.trace("example_operation", {"user_id": "test_user"}):
        start_time = time.time()
        try:
            # Simulated operation
            time.sleep(0.5)
            logger.log(
                "Operation completed successfully",
                context={"status": "success"},
            )
        except Exception as e:
            logger.log("Operation failed", level="error", exception=e)
        finally:
            logger.track_performance("example_operation", start_time)

    # Generate performance report
    performance_report = logger.generate_performance_report()
    print(json.dumps(performance_report, indent=2))


if __name__ == "__main__":
    main()
