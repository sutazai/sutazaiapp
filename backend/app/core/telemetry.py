"""
OpenTelemetry Configuration for Sutazai Backend
Distributed tracing with Jaeger integration
"""

import logging
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GRPCExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION, SERVICE_NAMESPACE
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from fastapi import FastAPI

logger = logging.getLogger(__name__)


class OpenTelemetryConfig:
    """OpenTelemetry configuration and setup"""
    
    def __init__(
        self,
        service_name: str = "sutazai-backend",
        service_version: str = "4.0.0",
        jaeger_endpoint: str = "http://sutazai-jaeger:4317",
        enable_console: bool = False,
        sample_rate: float = 1.0
    ):
        """
        Initialize OpenTelemetry configuration
        
        Args:
            service_name: Name of the service for tracing
            service_version: Version of the service
            jaeger_endpoint: Jaeger OTLP endpoint URL
            enable_console: Enable console span exporter for debugging
            sample_rate: Sampling rate (0.0-1.0, 1.0 = 100%)
        """
        self.service_name = service_name
        self.service_version = service_version
        self.jaeger_endpoint = jaeger_endpoint
        self.enable_console = enable_console
        self.sample_rate = sample_rate
        self.tracer_provider: Optional[TracerProvider] = None
        
    def setup_tracing(self) -> TracerProvider:
        """
        Configure OpenTelemetry tracing with Jaeger exporter
        
        Returns:
            TracerProvider: Configured tracer provider
        """
        try:
            # Define resource attributes
            resource = Resource.create({
                SERVICE_NAME: self.service_name,
                SERVICE_VERSION: self.service_version,
                SERVICE_NAMESPACE: "sutazai",
                "deployment.environment": "production",
                "service.instance.id": "backend-1"
            })
            
            # Create tracer provider
            self.tracer_provider = TracerProvider(resource=resource)
            
            # Configure OTLP gRPC exporter to Jaeger
            otlp_exporter = GRPCExporter(
                endpoint=self.jaeger_endpoint,
                insecure=True  # No TLS for internal communication
            )
            
            # Add batch span processor for efficient export
            span_processor = BatchSpanProcessor(otlp_exporter)
            self.tracer_provider.add_span_processor(span_processor)
            
            # Console exporter for debugging
            if self.enable_console:
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter
                console_processor = BatchSpanProcessor(ConsoleSpanExporter())
                self.tracer_provider.add_span_processor(console_processor)
            
            # Set global tracer provider
            trace.set_tracer_provider(self.tracer_provider)
            
            logger.info(
                f"OpenTelemetry tracing configured: "
                f"service={self.service_name}, "
                f"endpoint={self.jaeger_endpoint}"
            )
            
            return self.tracer_provider
            
        except Exception as e:
            logger.error(f"Failed to setup OpenTelemetry tracing: {e}")
            # Return a no-op tracer provider to avoid breaking the app
            return TracerProvider()
    
    def instrument_fastapi(self, app: FastAPI) -> None:
        """
        Instrument FastAPI application with OpenTelemetry
        
        Args:
            app: FastAPI application instance
        """
        try:
            FastAPIInstrumentor.instrument_app(
                app,
                tracer_provider=self.tracer_provider,
                excluded_urls="/health,/metrics,/docs,/redoc,/openapi.json"
            )
            logger.info(f"FastAPI instrumented with OpenTelemetry for {self.service_name}")
        except Exception as e:
            logger.error(f"Failed to instrument FastAPI: {e}")
    
    def get_tracer(self, name: str = "sutazai") -> trace.Tracer:
        """
        Get a tracer instance for custom spans
        
        Args:
            name: Tracer name
            
        Returns:
            Tracer: OpenTelemetry tracer
        """
        return trace.get_tracer(name, self.service_version)
    
    def shutdown(self) -> None:
        """Shutdown tracing and flush remaining spans"""
        if self.tracer_provider:
            try:
                self.tracer_provider.shutdown()
                logger.info("OpenTelemetry tracer provider shut down")
            except Exception as e:
                logger.error(f"Error shutting down tracer provider: {e}")


# Global instance
_otel_config: Optional[OpenTelemetryConfig] = None


def get_otel_config() -> Optional[OpenTelemetryConfig]:
    """Get the global OpenTelemetry configuration"""
    return _otel_config


def init_tracing(
    app: FastAPI,
    service_name: str = "sutazai-backend",
    service_version: str = "4.0.0",
    jaeger_endpoint: str = "http://sutazai-jaeger:4317",
    enable_console: bool = False
) -> OpenTelemetryConfig:
    """
    Initialize OpenTelemetry tracing for the application
    
    Args:
        app: FastAPI application
        service_name: Name of the service
        service_version: Version of the service
        jaeger_endpoint: Jaeger collector endpoint
        enable_console: Enable console output for debugging
        
    Returns:
        OpenTelemetryConfig: Configured OpenTelemetry instance
    """
    global _otel_config
    
    _otel_config = OpenTelemetryConfig(
        service_name=service_name,
        service_version=service_version,
        jaeger_endpoint=jaeger_endpoint,
        enable_console=enable_console
    )
    
    # Setup tracing
    _otel_config.setup_tracing()
    
    # Instrument FastAPI
    _otel_config.instrument_fastapi(app)
    
    return _otel_config


# Utility functions for custom spans
def create_span(name: str, attributes: dict = None):
    """
    Create a custom span for manual instrumentation
    
    Args:
        name: Span name
        attributes: Span attributes
        
    Returns:
        Span context manager
    """
    tracer = trace.get_tracer("sutazai")
    span = tracer.start_as_current_span(name)
    
    if attributes:
        for key, value in attributes.items():
            span.set_attribute(key, value)
    
    return span


def add_span_event(name: str, attributes: dict = None):
    """
    Add an event to the current span
    
    Args:
        name: Event name
        attributes: Event attributes
    """
    span = trace.get_current_span()
    if span:
        span.add_event(name, attributes or {})


def set_span_attribute(key: str, value):
    """
    Set an attribute on the current span
    
    Args:
        key: Attribute key
        value: Attribute value
    """
    span = trace.get_current_span()
    if span:
        span.set_attribute(key, value)


def record_exception(exception: Exception):
    """
    Record an exception on the current span
    
    Args:
        exception: Exception to record
    """
    span = trace.get_current_span()
    if span:
        span.record_exception(exception)
