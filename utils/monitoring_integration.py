"""
Monitoring Integration Module for SutazAI

This module integrates all monitoring components into FastAPI applications,
providing middleware, endpoints, and utilities for comprehensive system monitoring.
"""

import os
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from functools import wraps

from fastapi import FastAPI, Request
from pydantic import BaseModel

# Try to import optional dependencies
try:
    from prometheus_client import Counter, Histogram, make_asgi_app
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

# Import our monitoring modules
from utils.logging_setup import get_app_logger
from utils.logging_setup import log_request
from utils.neural_monitoring import (
    create_spiking_network_monitor,
    create_attention_monitor,
    create_plasticity_monitor,
)
from utils.ethics_verification import (
    create_ethics_monitor,
    get_common_ethical_boundaries,
    get_common_ethical_properties,
)
from utils.self_mod_monitoring import create_self_mod_monitor
from utils.hardware_monitoring import (
    create_hardware_monitor,
    get_current_hardware_profile,
)
from utils.security_monitoring import create_security_monitor
from utils.settings import Settings
from utils.metrics_collector import MetricsCollector

logger = get_app_logger()


class MonitoringSettings(BaseModel):
    """Settings for monitoring configuration."""

    system_id: str
    base_dir: str
    log_dir: Optional[str] = None
    prometheus_metrics_path: str = "/metrics"
    enable_neural_monitoring: bool = True
    enable_ethics_monitoring: bool = True
    enable_self_mod_monitoring: bool = True
    enable_hardware_monitoring: bool = True
    enable_security_monitoring: bool = True
    expose_monitoring_ui: bool = False
    monitoring_ui_path: str = "/monitoring"
    security_config: Dict[str, Any] = {}
    collection_interval: float = 30.0


class MonitoringSystem:
    """Integrated monitoring system for SutazAI."""

    def __init__(self, settings: MonitoringSettings):
        """
        Initialize the monitoring system with the provided settings.

        Args:
            settings: Monitoring configuration settings
        """
        self.settings = settings
        self.logger = logger

        # Set up logging directory
        self.log_dir = settings.log_dir or os.path.join(
            os.environ.get("SUTAZAI_LOG_DIR", "/opt/sutazaiapp/logs"), "monitoring"
        )
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize monitoring components
        self.components: Dict[str, Any] = {}
        self._initialize_components()

        # Track model inferences
        self.model_inferences = (
            Counter(
                "sutazai_model_inferences_total",
                "Total number of model inferences",
                ["model_id", "endpoint"],
            )
            if MONITORING_AVAILABLE
            else None
        )

        # Track API metrics
        self.api_requests = (
            Counter(
                "sutazai_api_requests_total",
                "Total number of API requests",
                ["method", "endpoint", "status"],
            )
            if MONITORING_AVAILABLE
            else None
        )

        self.api_request_duration = (
            Histogram(
                "sutazai_api_request_duration_seconds",
                "API request duration in seconds",
                ["method", "endpoint"],
            )
            if MONITORING_AVAILABLE
            else None
        )

        self.logger.info(f"Initialized monitoring system for {settings.system_id}")

    def _initialize_components(self) -> None:
        """Initialize all enabled monitoring components."""
        if self.settings.enable_neural_monitoring:
            self.components["neural"] = {
                "spiking": create_spiking_network_monitor(
                    f"{self.settings.system_id}_snn"
                ),
                "attention": create_attention_monitor(
                    f"{self.settings.system_id}_attention"
                ),
                "plasticity": create_plasticity_monitor(
                    f"{self.settings.system_id}_plasticity"
                ),
            }
            self.logger.info("Initialized neural monitoring components")

        if self.settings.enable_ethics_monitoring:
            ethics_monitor = create_ethics_monitor(
                self.settings.system_id, os.path.join(self.log_dir, "ethics")
            )

            # Register common ethical boundaries and properties
            for boundary in get_common_ethical_boundaries():
                ethics_monitor.register_boundary(boundary)

            for property_ in get_common_ethical_properties():
                ethics_monitor.register_property(property_)

            self.components["ethics"] = ethics_monitor
            self.logger.info("Initialized ethics monitoring components")

        if self.settings.enable_self_mod_monitoring:
            self.components["self_mod"] = create_self_mod_monitor(
                self.settings.system_id,
                self.settings.base_dir,
                components=["app", "models", "utils", "config"],
                log_dir=os.path.join(self.log_dir, "self_mod"),
            )
            self.logger.info("Initialized self-modification monitoring components")

        if self.settings.enable_hardware_monitoring:
            self.components["hardware"] = create_hardware_monitor(
                f"{self.settings.system_id}_node",
                os.path.join(self.log_dir, "hardware"),
                collection_interval=self.settings.collection_interval,
            )
            self.logger.info("Initialized hardware monitoring components")

        if self.settings.enable_security_monitoring:
            self.components["security"] = create_security_monitor(
                self.settings.system_id,
                self.settings.base_dir,
                config=self.settings.security_config,
                log_dir=os.path.join(self.log_dir, "security"),
            )
            self.logger.info("Initialized security monitoring components")

    def start_background_monitoring(self) -> None:
        """Start background monitoring threads for all components."""
        if "hardware" in self.components:
            self.components["hardware"].start_collection()
            self.logger.info("Started hardware metrics collection")

        if "security" in self.components:
            self.components["security"].start_monitoring(
                check_interval=self.settings.collection_interval
                * 10  # Less frequent for full checks
            )
            self.logger.info("Started security monitoring")

        if "neural" in self.components and "spiking" in self.components["neural"]:
            self.components["neural"]["spiking"].start()
            self.logger.info("Started neural network monitoring")

        self.logger.info("All background monitoring components started")

    def stop_background_monitoring(self) -> None:
        """Stop all background monitoring threads."""
        if "hardware" in self.components:
            self.components["hardware"].stop_collection()

        if "security" in self.components:
            self.components["security"].stop_monitoring_thread()

        if "neural" in self.components and "spiking" in self.components["neural"]:
            self.components["neural"]["spiking"].stop()

        self.logger.info("All background monitoring components stopped")

    def setup_fastapi(self, app: FastAPI) -> None:
        """
        Set up monitoring for a FastAPI application.

        Args:
            app: FastAPI application instance
        """
        # Add request monitoring middleware
        app.add_middleware(RequestMonitoringMiddleware, monitoring_system=self)

        # Add Prometheus metrics endpoint if available
        if MONITORING_AVAILABLE:
            metrics_app = make_asgi_app()
            app.mount(self.settings.prometheus_metrics_path, metrics_app)
            self.logger.info(
                f"Mounted Prometheus metrics at {self.settings.prometheus_metrics_path}"
            )

        # Add monitoring API endpoints if UI is enabled
        if self.settings.expose_monitoring_ui:
            self._add_monitoring_endpoints(app)
            self.logger.info(
                f"Monitoring UI endpoints added at {self.settings.monitoring_ui_path}"
            )

        # Start background monitoring
        @app.on_event("startup")
        async def startup_monitoring():
            self.start_background_monitoring()

        # Stop background monitoring
        @app.on_event("shutdown")
        async def shutdown_monitoring():
            self.stop_background_monitoring()

        self.logger.info("Monitoring setup completed for FastAPI application")

    def _add_monitoring_endpoints(self, app: FastAPI) -> None:
        """Add monitoring API endpoints to the FastAPI app."""
        from fastapi import APIRouter

        router = APIRouter(prefix=self.settings.monitoring_ui_path)

        @router.get("/status")
        async def get_monitoring_status():
            """Get the status of all monitoring components."""
            status = {
                "system_id": self.settings.system_id,
                "timestamp": datetime.now().isoformat(),
                "components": {},
            }

            for component_name, component in self.components.items():
                if component_name == "neural":
                    status["components"]["neural"] = {
                        "enabled": True,
                        "monitors": list(component.keys()),
                    }
                elif component_name == "ethics":
                    status["components"]["ethics"] = {
                        "enabled": True,
                        "boundaries": len([b for b in component.boundaries.values()]),
                        "properties": len([p for p in component.properties.values()]),
                    }
                elif component_name == "self_mod":
                    status["components"]["self_mod"] = {
                        "enabled": True,
                        "monitored_components": component.components,
                        "modifications": len(component.modifications),
                    }
                elif component_name == "hardware":
                    status["components"]["hardware"] = {
                        "enabled": True,
                        "hardware_detected": component.hardware_detected,
                    }
                elif component_name == "security":
                    status["components"]["security"] = {
                        "enabled": True,
                        "airgap_status": component.airgap_status,
                        "last_airgap_check": datetime.fromtimestamp(
                            component.last_airgap_check
                        ).isoformat()
                        if component.last_airgap_check
                        else None,
                        "events": len(component.events),
                    }

            return status

        @router.get("/hardware")
        async def get_hardware_metrics():
            """Get the latest hardware metrics."""
            if "hardware" not in self.components:
                return {"error": "Hardware monitoring not enabled"}

            metrics = self.components["hardware"].collect_metrics()
            return metrics

        @router.get("/security/events")
        async def get_security_events(limit: int = 100):
            """Get recent security events."""
            if "security" not in self.components:
                return {"error": "Security monitoring not enabled"}

            events = []
            for event in self.components["security"].events[-limit:]:
                events.append(
                    {
                        "id": event.id,
                        "timestamp": datetime.fromtimestamp(
                            event.timestamp
                        ).isoformat(),
                        "event_type": event.event_type.value,
                        "severity": event.severity.value,
                        "summary": event.summary,
                        "source": event.source,
                    }
                )

            return {"events": events}

        @router.get("/ethics/boundaries")
        async def get_ethical_boundaries():
            """Get registered ethical boundaries."""
            if "ethics" not in self.components:
                return {"error": "Ethics monitoring not enabled"}

            boundaries = []
            for boundary in self.components["ethics"].boundaries.values():
                boundaries.append(
                    {
                        "id": boundary.id,
                        "name": boundary.name,
                        "description": boundary.description,
                        "threshold": boundary.threshold,
                        "lower_is_better": boundary.lower_is_better,
                    }
                )

            return {"boundaries": boundaries}

        @router.get("/self-modifications")
        async def get_self_modifications(limit: int = 100):
            """Get recent self-modifications."""
            if "self_mod" not in self.components:
                return {"error": "Self-modification monitoring not enabled"}

            mods = self.components["self_mod"].get_modifications(limit=limit)
            modifications = []

            for mod in mods:
                modifications.append(
                    {
                        "id": mod.id,
                        "timestamp": datetime.fromtimestamp(mod.timestamp).isoformat(),
                        "component": mod.component,
                        "type": mod.modification_type.value,
                        "description": mod.description,
                        "size_bytes": mod.size_bytes,
                        "is_applied": mod.is_applied,
                        "verification_result": mod.verification_result.value,
                    }
                )

            return {"modifications": modifications}

        app.include_router(router)

    # Neural monitoring utilities
    def record_attention_weights(
        self, model_id: str, layer: str, head: int, weights: Any
    ) -> None:
        """Record attention weights for monitoring."""
        if (
            "neural" not in self.components
            or "attention" not in self.components["neural"]
        ):
            return

        attention_monitor = self.components["neural"]["attention"]
        attention_monitor.record_attention_weights(layer, head, weights)

    def record_synaptic_changes(
        self,
        network_id: str,
        connection_type: str,
        weights: Any,
        module: str = "default",
    ) -> None:
        """Record synaptic weight changes for monitoring."""
        if (
            "neural" not in self.components
            or "plasticity" not in self.components["neural"]
        ):
            return

        plasticity_monitor = self.components["neural"]["plasticity"]
        plasticity_monitor.record_weight_changes(connection_type, weights, module)

    # Ethics monitoring utilities
    def check_ethical_boundary(
        self, boundary_id: str, value: float, context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if a value falls within an ethical boundary."""
        if "ethics" not in self.components:
            return True

        return self.components["ethics"].check_decision(boundary_id, value, context)

    def verify_ethical_property(
        self, property_id: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Verify a formal ethical property."""
        if "ethics" not in self.components:
            return {"status": "unknown", "error": "Ethics monitoring not enabled"}

        result = self.components["ethics"].verify_property(property_id, context)
        return {
            "status": result.status,
            "confidence": result.confidence,
            "verification_time": result.verification_time,
            "details": result.details,
        }

    # Security monitoring utilities
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        summary: str,
        details: Optional[Dict[str, Any]] = None,
        source: str = "",
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        result: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Log a security event."""
        if "security" not in self.components:
            logger.warning(
                f"Security event not logged (monitoring disabled): {summary}"
            )
            return {"error": "Security monitoring not enabled"}

        from utils.security_monitoring import SecurityEventType, SecuritySeverity

        # Convert string types to enums
        try:
            event_type_enum = SecurityEventType[event_type.upper()]
        except (KeyError, AttributeError):
            event_type_enum = SecurityEventType.SYSTEM_CHANGE

        try:
            severity_enum = SecuritySeverity[severity.upper()]
        except (KeyError, AttributeError):
            severity_enum = SecuritySeverity.INFO

        event = self.components["security"].log_event(
            event_type=event_type_enum,
            severity=severity_enum,
            summary=summary,
            details=details,
            source=source,
            user_id=user_id,
            resource_id=resource_id,
            result=result,
        )

        return {
            "id": event.id,
            "timestamp": datetime.fromtimestamp(event.timestamp).isoformat(),
            "status": "logged",
        }

    # Self-modification monitoring utilities
    def record_modification(
        self,
        component: str,
        modified_files: List[str],
        description: str,
        mod_type: Optional[str] = None,
        verification_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a system self-modification."""
        if "self_mod" not in self.components:
            logger.warning(
                f"Modification not recorded (monitoring disabled): {description}"
            )
            return {"error": "Self-modification monitoring not enabled"}

        from utils.self_mod_monitoring import ModificationType

        # Convert string type to enum if provided
        mod_type_enum = None
        if mod_type:
            try:
                mod_type_enum = ModificationType[mod_type.upper()]
            except (KeyError, AttributeError):
                pass

        event = self.components["self_mod"].record_modification(
            component=component,
            modified_files=modified_files,
            description=description,
            mod_type=mod_type_enum,
            verification_result=verification_result,
        )

        return {
            "id": event.id,
            "timestamp": datetime.fromtimestamp(event.timestamp).isoformat(),
            "status": "recorded",
        }

    # Hardware monitoring utilities
    def get_hardware_profile(self) -> Dict[str, Any]:
        """Get the current hardware profile."""
        profile = get_current_hardware_profile()
        return {
            "device_id": profile.device_id,
            "device_type": profile.device_type,
            "compute_units": profile.compute_units,
            "memory_bytes": profile.memory_bytes,
            "peak_flops": profile.peak_flops,
            "peak_memory_bandwidth": profile.peak_memory_bandwidth,
            "description": profile.description,
            "vendor": profile.vendor,
            "model": profile.model,
        }


# ASGI Middleware for Request Monitoring
class RequestMonitoringMiddleware:
    def __init__(self, app: FastAPI, monitoring_system: MonitoringSystem):
        self.app = app
        self.monitoring_system = monitoring_system
        self.logger = get_app_logger()

    async def __call__(self, scope: dict, receive: Callable, send: Callable):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        start_time = time.time()
        status_code = 500  # Default to 500 in case of exception before response

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            # Log exception if it occurs before response is sent
            duration_ms = (time.time() - start_time) * 1000
            endpoint = request.url.path
            log_request(
                component="api_gateway",
                endpoint=endpoint,
                duration=duration_ms,
                status_code=status_code,
                error=str(e),
            )
            # Reraise the exception so Starlette/FastAPI can handle it
            raise e
        finally:
            # Log request after response is sent (or exception handled)
            if status_code != 500: # Avoid double logging if exception occurred before response
                duration_ms = (time.time() - start_time) * 1000
                endpoint = request.url.path
                log_request(
                    component="api_gateway",
                    endpoint=endpoint,
                    duration=duration_ms,
                    status_code=status_code,
                )


# Decorator for monitoring model inferences
def monitor_inference(model_id: str, endpoint: str = "unknown"):
    """
    Decorator for monitoring model inferences.

    Args:
        model_id: Model identifier
        endpoint: API endpoint identifier
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get the monitoring system if available
            monitoring_system = None
            for arg in args:
                if isinstance(arg, MonitoringSystem):
                    monitoring_system = arg
                    break

            for _, arg in kwargs.items():
                if isinstance(arg, MonitoringSystem):
                    monitoring_system = arg
                    break

            # Record start time
            start_time = time.time()

            # Call the original function
            result = await func(*args, **kwargs)

            # Record metrics
            if monitoring_system and monitoring_system.model_inferences is not None:
                monitoring_system.model_inferences.labels(
                    model_id=model_id, endpoint=endpoint
                ).inc()

            # Get duration
            duration = time.time() - start_time

            # Log the inference
            if monitoring_system:
                monitoring_system.logger.debug(
                    f"Model inference: {model_id} on {endpoint}, duration={duration:.3f}s"
                )

            return result

        return wrapper

    return decorator


# Factory function
def create_monitoring_system(
    system_id: str,
    base_dir: str,
    log_dir: Optional[str] = None,
    prometheus_metrics_path: str = "/metrics",
    enable_neural_monitoring: bool = True,
    enable_ethics_monitoring: bool = True,
    enable_self_mod_monitoring: bool = True,
    enable_hardware_monitoring: bool = True,
    enable_security_monitoring: bool = True,
    expose_monitoring_ui: bool = False,
    monitoring_ui_path: str = "/monitoring",
    security_config: Dict[str, Any] = {},
    collection_interval: float = 30.0,
) -> MonitoringSystem:
    """Create and return a new monitoring system."""
    settings = MonitoringSettings(
        system_id=system_id,
        base_dir=base_dir,
        log_dir=log_dir,
        prometheus_metrics_path=prometheus_metrics_path,
        enable_neural_monitoring=enable_neural_monitoring,
        enable_ethics_monitoring=enable_ethics_monitoring,
        enable_self_mod_monitoring=enable_self_mod_monitoring,
        enable_hardware_monitoring=enable_hardware_monitoring,
        enable_security_monitoring=enable_security_monitoring,
        expose_monitoring_ui=expose_monitoring_ui,
        monitoring_ui_path=monitoring_ui_path,
        security_config=security_config,
        collection_interval=collection_interval,
    )

    return MonitoringSystem(settings)
