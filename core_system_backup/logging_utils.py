#!/usr/bin/env python3
"""
SutazAI Advanced Logging and Observability System

Provides intelligent, context-aware logging with
advanced tracing, performance tracking, and
comprehensive system observability.
"""

import functools
import json
import logging
import multiprocessing
import os
import socket
import sys
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional, Union

import opentelemetry
import yaml
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pythonjsonlogger import jsonlogger


class AdvancedLogger:
    """
    Ultra-Comprehensive Logging and Observability Framework

    Key Capabilities:
    - Context-aware logging
    - Performance tracing
    - Distributed system support
    - Intelligent log routing
    - Security and compliance logging
    """

    def __init__(
        self,
        base_dir: str = "/opt/sutazai_project/SutazAI",
        config_path: Optional[str] = None,
        log_level: int = logging.INFO,
    ):
        """
        Initialize Advanced Logger

        Args:
            base_dir (str): Base project directory
            config_path (Optional[str]): Path to logging configuration
            log_level (int): Logging level
        """
        self.base_dir = base_dir
        self.config_path = config_path or os.path.join(
            base_dir, "config", "logging_config.yaml"
        )

        # Ensure log directories exist
        self._create_log_directories()

        # Load logging configuration
        self.config = self._load_logging_config()

        # Configure logging
        self.logger = self._configure_logger(log_level)

        # Configure distributed tracing
        self._configure_distributed_tracing()

    def _create_log_directories(self):
        """
        Create necessary log directories with appropriate permissions
        """
        log_base = os.path.join(self.base_dir, "logs")
        log_subdirs = [
            "system",
            "workers",
            "error_correction",
            "performance",
            "security",
            "distributed_tracing",
        ]

        for subdir in log_subdirs:
            dir_path = os.path.join(log_base, subdir)
            os.makedirs(dir_path, exist_ok=True)
            # Set restrictive permissions
            os.chmod(dir_path, 0o750)

    def _load_logging_config(self) -> Dict[str, Any]:
        """
        Load logging configuration from YAML file

        Returns:
            Logging configuration dictionary
        """
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Logging configuration load failed: {e}")
            return {
                "log_level": "INFO",
                "log_format": "%(asctime)s - %(name)s - %(levelname)s: %(message)s",
            }

    def _configure_logger(self, log_level: int) -> logging.Logger:
        """
        Configure logger with advanced formatting and handlers

        Args:
            log_level (int): Logging level

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger("SutazAI")
        logger.setLevel(log_level)

        # Clear existing handlers
        logger.handlers.clear()

        # JSON Formatter for structured logging
        json_formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d"
        )

        # File Handler (System Log)
        system_log_path = os.path.join(
            self.base_dir, "logs", "system", "system.log"
        )
        file_handler = logging.FileHandler(system_log_path)
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(json_formatter)
        logger.addHandler(console_handler)

        return logger

    def _configure_distributed_tracing(self):
        """
        Configure distributed tracing with OpenTelemetry
        """
        try:
            # Set up Jaeger tracer
            trace.set_tracer_provider(TracerProvider())

            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )

            trace.get_tracer_provider().add_span_processor(
                BatchSpanProcessor(jaeger_exporter)
            )
        except Exception as e:
            self.logger.error(f"Distributed tracing configuration failed: {e}")

    def log(
        self,
        message: str,
        level: str = "info",
        extra: Optional[Dict[str, Any]] = None,
    ):
        """
        Advanced logging method with context and performance tracking

        Args:
            message (str): Log message
            level (str): Logging level
            extra (Optional[Dict]): Additional context information
        """
        try:
            # Prepare log context
            log_context = {
                "hostname": socket.gethostname(),
                "process_id": os.getpid(),
                "thread_id": threading.get_ident(),
                "timestamp": datetime.now().isoformat(),
            }

            # Add extra context if provided
            if extra:
                log_context.update(extra)

            # Select logging method based on level
            log_method = getattr(self.logger, level.lower(), self.logger.info)
            log_method(message, extra=log_context)

        except Exception as e:
            print(f"Logging failed: {e}")

    def trace_performance(
        self, func: Optional[callable] = None, *, log_level: str = "info"
    ):
        """
        Performance tracing decorator

        Args:
            func (callable): Function to trace
            log_level (str): Logging level for performance metrics

        Returns:
            Wrapped function with performance tracking
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()

                try:
                    result = func(*args, **kwargs)

                    # Log performance metrics
                    execution_time = time.time() - start_time
                    self.log(
                        f"Performance: {func.__name__} executed in {execution_time:.4f} seconds",
                        level=log_level,
                        extra={
                            "function_name": func.__name__,
                            "execution_time": execution_time,
                            "args": str(args),
                            "kwargs": str(kwargs),
                        },
                    )

                    return result

                except Exception as e:
                    # Log error with full traceback
                    self.log(
                        f"Error in {func.__name__}: {str(e)}",
                        level="error",
                        extra={
                            "function_name": func.__name__,
                            "traceback": traceback.format_exc(),
                        },
                    )
                    raise

            return wrapper

        # Support using decorator with or without arguments
        return decorator(func) if func else decorator

    def generate_system_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive system logging report

        Returns:
            System logging report dictionary
        """
        try:
            system_report = {
                "timestamp": datetime.now().isoformat(),
                "system_info": {
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "python_version": sys.version,
                    "platform": sys.platform,
                },
                "log_configuration": self.config,
                "log_directories": [
                    os.path.join(self.base_dir, "logs", subdir)
                    for subdir in ["system", "workers", "error_correction"]
                ],
            }

            # Persist report
            report_path = os.path.join(
                self.base_dir,
                "logs",
                f'system_logging_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            )

            with open(report_path, "w") as f:
                json.dump(system_report, f, indent=2)

            self.logger.info(f"System logging report generated: {report_path}")

            return system_report

        except Exception as e:
            self.logger.error(f"System report generation failed: {e}")
            return {}


def main():
    """
    Demonstration of advanced logging capabilities
    """
    logger = AdvancedLogger()

    # Demonstrate logging methods
    logger.log("System initialization started", level="info")

    # Demonstrate performance tracing
    @logger.trace_performance
    def example_function(x, y):
        time.sleep(0.1)  # Simulate work
        return x + y

    result = example_function(10, 20)

    # Generate system report
    report = logger.generate_system_report()
    print("System Logging Report Generated")


if __name__ == "__main__":
    main()
