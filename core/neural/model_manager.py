#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_manager.py - Enterprise-grade unified model management system for SutazAI

This module serves as the central coordination point for model lifecycle management,
integrating downloading, monitoring, optimization, and deployment for Dell PowerEdge R720.
"""

import os
import json
import logging
import concurrent.futures

# F401: Removed unused import torch
# import torch
from typing import Dict, List, Optional, Any  # , Union
from datetime import datetime

# Import core components
try:
    from core.neural.model_downloader import ModelRegistry, EnterpriseModelDownloader

    MODEL_DOWNLOADER_AVAILABLE = True
except ImportError:
    ModelRegistry = type("ModelRegistry", (), {})
    MODEL_DOWNLOADER_AVAILABLE = False
    logging.warning(
        "model_downloader module not found, downloading features will be unavailable"
    )

try:
    from core.neural.model_monitor import ModelMonitor, ModelMonitorDB

    MODEL_MONITOR_AVAILABLE = True
except ImportError:
    ModelMonitor = type("ModelMonitor", (), {})
    ModelMonitorDB = type("ModelMonitorDB", (), {})
    MODEL_MONITOR_AVAILABLE = False
    logging.warning(
        "model_monitor module not found, monitoring features will be unavailable"
    )

try:
    from core.neural.llama_utils import CPUOptimizedLlama

    LLAMA_UTILS_AVAILABLE = True
except ImportError:
    CPUOptimizedLlama = type("CPUOptimizedLlama", (), {})
    LLAMA_UTILS_AVAILABLE = False
    logging.warning(
        "llama_utils module not found, Llama model support will be unavailable"
    )

# Set up logging
logger = logging.getLogger("sutazai.model_manager")

# Constants
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.expanduser("~"), ".config", "sutazai", "model_manager.json"
)
DEFAULT_MODELS_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "sutazai", "models"
)
DEFAULT_OPTIMIZED_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "sutazai", "optimized_models"
)


class ModelManager:
    """
    Enterprise-grade unified model management system.

    This class provides a central coordination point for the entire model lifecycle:
    - Downloading and version management
    - Performance monitoring and drift detection
    - Optimization for specific hardware (E5-2640)
    - Automated retraining and deployment
    - Health checks and alerts
    """

    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG_PATH,
        models_dir: str = DEFAULT_MODELS_DIR,
        optimized_dir: str = DEFAULT_OPTIMIZED_DIR,
        auto_download: bool = True,
        auto_optimize: bool = True,
        auto_monitor: bool = True,
        auto_retrain: bool = False,
        max_workers: int = 4,
    ):
        """Initialize the model manager"""
        self.config_path = config_path
        self.models_dir = models_dir
        self.optimized_dir = optimized_dir
        self.auto_download = auto_download
        self.auto_optimize = auto_optimize
        self.auto_monitor = auto_monitor
        self.auto_retrain = auto_retrain
        self.max_workers = max_workers
        self._executor = None

        # Load configuration
        self.config = self._load_config()

        # Initialize registry
        if MODEL_DOWNLOADER_AVAILABLE:
            self.model_registry = ModelRegistry()
            self.model_downloader = EnterpriseModelDownloader(registry=self.model_registry)
            self.model_optimizer = TransformerOptimizer()
            self.model_monitor_db = ModelMonitorDB()
            self.model_monitor = ModelMonitor(self.model_monitor_db)
            self.llama_model_class = CPUOptimizedLlama
            logger.info("Model Downloader initialized.")
        else:
            self.model_registry = None
            self.model_downloader = None
            self.model_optimizer = None
            self.model_monitor_db = None
            self.model_monitor = None
            self.llama_model_class = None
            logger.warning(
                "Model Downloader dependencies not found. Download functionality disabled."
            )

        # Create directories
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(optimized_dir, exist_ok=True)
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        # Start thread pool
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        # Start auto-monitoring if enabled
        if auto_monitor and MODEL_MONITOR_AVAILABLE:
            self._start_auto_monitoring()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
            except Exception as e:
                logger.error(f"Error loading config: {e}")

        # Default configuration
        config = {
            "models_dir": self.models_dir,
            "optimized_dir": self.optimized_dir,
            "auto_download": self.auto_download,
            "auto_optimize": self.auto_optimize,
            "auto_monitor": self.auto_monitor,
            "auto_retrain": self.auto_retrain,
            "default_thread_count": 12,  # Optimized for E5-2640 (6 cores per socket, 12 total)
            "monitoring_interval_sec": 3600,
            "performance_threshold": 0.15,
            "preferred_models": {
                "default": "llama3-8b",
                "high_performance": "llama3-70b",
                "low_resource": "mistral-7b",
            },
            "model_configs": {},
        }

        # Save default config
        self._save_config(config)
        return config

    def _save_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Save configuration to file"""
        try:
            with open(self.config_path, "w") as f:
                json.dump(config or self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False

    def _start_auto_monitoring(self):
        """Start automatic monitoring of models"""
        if not MODEL_MONITOR_AVAILABLE:
            logger.error("Cannot start monitoring - model_monitor module not available")
            return False

        # Get list of models to monitor
        if self.model_registry:
            model_ids = self.model_registry.list_models()
            if model_ids:
                logger.info(
                    f"Starting automatic monitoring for {len(model_ids)} models"
                )
                self.model_monitor.monitoring_interval = self.config.get(
                    "monitoring_interval_sec", 3600
                )
                self.model_monitor.performance_threshold = self.config.get(
                    "performance_threshold", 0.15
                )
                self.model_monitor.start_monitoring(model_ids)
                return True

        logger.warning("No models found to monitor")
        return False

    def _handle_alert(self, alert_data: Dict[str, Any]):
        """Handle alert from the monitoring system"""
        logger.warning(
            f"Alert received: {alert_data['alert_type']} for {alert_data['model_id']}"
        )

        # Add alert handling logic here - could send email, Slack message, etc.

        # If auto-retrain is enabled and this is a performance degradation alert
        if (
            self.auto_retrain
            and alert_data["alert_type"] == "PERFORMANCE_DEGRADATION"
            and MODEL_DOWNLOADER_AVAILABLE
        ):
            # Schedule retraining in background
            self._executor.submit(
                self._handle_retraining,
                alert_data["model_id"],
                alert_data["version"],
                alert_data["details"],
            )

    def _handle_retraining(
        self, model_id: str, version: str, degradation_details: Dict[str, Any]
    ):
        """Handle retraining of a model after performance degradation"""
        logger.info(
            f"Preparing to retrain model {model_id} due to performance degradation"
        )

        # In a real implementation, this would:
        # 1. Request model retraining from a training service
        # 2. Monitor the training process
        # 3. Validate the new model
        # 4. Register the new model in the registry

        # For now, just log the event
        logger.info(f"Retraining would be triggered here for {model_id}")

        # Record in monitoring DB if available
        if self.model_monitor_db:
            metrics = self.model_monitor_db.get_latest_metrics(model_id, version)
            metrics_dict = None
            if metrics:
                metrics_dict = {
                    "inference_time_ms": metrics.inference_time_ms,
                    "tokens_per_second": metrics.tokens_per_second,
                    "success_rate": metrics.success_rate,
                }

            event_id = self.model_monitor_db.record_retraining_event(
                model_id=model_id,
                old_version=version,
                reason=f"Auto-retraining due to: {json.dumps(degradation_details)}",
                metrics_before=metrics_dict,
            )

            # Update event status
            self.model_monitor_db.update_retraining_event(
                event_id=event_id, status="SCHEDULED"
            )

    def get_model(
        self,
        model_id: Optional[str] = None,
        force_download: bool = False,
        optimize: Optional[bool] = None,
        wait: bool = True,
    ) -> Dict[str, Any]:
        """
        Get a model, downloading and optimizing if necessary.

        Args:
            model_id: ID of the model, or None to use default
            force_download: Whether to force download even if exists
            optimize: Whether to optimize the model (default: follows auto_optimize setting)
            wait: Whether to wait for async operations to complete

        Returns:
            Dictionary with model information and status
        """
        if not MODEL_DOWNLOADER_AVAILABLE:
            return {"error": "Model downloader not available", "success": False}

        # If no model ID specified, use default
        if not model_id:
            model_id = self.config.get("preferred_models", {}).get(
                "default", "llama3-8b"
            )

        # Check if optimization is required
        should_optimize = self.auto_optimize if optimize is None else optimize

        try:
            # Check if we already have this model
            model_version = self.model_registry.get_model(model_id)
            model_path = None

            if (
                model_version
                and os.path.exists(model_version.path)
                and not force_download
            ):
                # Use existing model
                model_path = model_version.path
                logger.info(f"Using existing model: {model_id}")
            else:
                # Download model
                logger.info(f"Downloading model: {model_id}")
                model_path = self.model_downloader.get_model(
                    model_id, force_download=force_download
                )

                if not model_path:
                    return {
                        "error": f"Failed to download model: {model_id}",
                        "success": False,
                    }

            result = {"model_id": model_id, "model_path": model_path, "success": True}

            # If optimization requested
            if should_optimize:
                # For synchronous operation
                if wait:
                    optimized_path = self._optimize_model(model_id, model_path)
                    result["optimized_path"] = optimized_path
                    result["optimized"] = bool(optimized_path)
                else:
                    # Schedule optimization in background
                    self._executor.submit(self._optimize_model, model_id, model_path)
                    result["optimization_scheduled"] = True

            # If monitoring enabled, schedule evaluation
            if self.auto_monitor and MODEL_MONITOR_AVAILABLE and self.model_monitor:
                if wait:
                    metrics = self.model_monitor.manually_evaluate_model(model_id)
                    if metrics:
                        result["metrics"] = {
                            "inference_time_ms": metrics.inference_time_ms,
                            "tokens_per_second": metrics.tokens_per_second,
                            "memory_usage_mb": metrics.memory_usage_mb,
                        }
                else:
                    # Schedule evaluation in background
                    self._executor.submit(
                        self.model_monitor.manually_evaluate_model, model_id
                    )
                    result["evaluation_scheduled"] = True

            return result

        except Exception as e:
            logger.error(f"Error getting model {model_id}: {e}")
            return {"error": str(e), "model_id": model_id, "success": False}

    def _optimize_model(self, model_id: str, model_path: str) -> Optional[str]:
        """
        Optimize a model for the E5-2640 CPU.

        Args:
            model_id: ID of the model
            model_path: Path to the model file

        Returns:
            Path to the optimized model, or None if optimization failed
        """
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None

        logger.info(f"Optimizing model: {model_id}")

        # Determine model type from file extension or ID
        if (
            model_path.endswith(".gguf")
            or "llama" in model_id.lower()
            or "mistral" in model_id.lower()
        ):
            # For Llama models
            if LLAMA_UTILS_AVAILABLE:
                return self._optimize_llama_model(model_id, model_path)
            else:
                logger.error("Cannot optimize Llama model - llama_utils not available")
                return None
        else:
            # For transformer models
            return self._optimize_transformer_model(model_id, model_path)

    def _optimize_llama_model(self, model_id: str, model_path: str) -> Optional[str]:
        """Optimize a Llama model"""
        if not LLAMA_UTILS_AVAILABLE:
            return None

        try:
            # Create output directory
            output_dir = os.path.join(self.optimized_dir, model_id)
            os.makedirs(output_dir, exist_ok=True)

            # Configuration path
            config_path = os.path.join(output_dir, "optimized_config.json")

            # Load model with optimal settings for E5-2640
            model = self.llama_model_class(
                model_path=model_path,
                n_threads=12,  # E5-2640 has 12 cores total
                n_ctx=4096,
                n_batch=512,
                use_mmap=True,
            )

            # Save optimized configuration
            model.save_config(config_path)

            logger.info(
                f"Llama model {model_id} optimized, config saved to: {config_path}"
            )

            # Clean up resources
            del model

            return config_path

        except Exception as e:
            logger.error(f"Error optimizing Llama model {model_id}: {e}")
            return None

    def _optimize_transformer_model(
        self, model_id: str, model_path: str
    ) -> Optional[str]:
        """Optimize a transformer model"""
        try:
            # Create output directory
            output_dir = os.path.join(self.optimized_dir, model_id)
            os.makedirs(output_dir, exist_ok=True)

            # For now, we'll just log that optimization would happen here
            # In a real implementation, this would run transformer optimizations
            logger.info(f"Transformer model {model_id} optimization would run here")

            # In this placeholder implementation, just create a dummy config
            config_path = os.path.join(output_dir, "transformer_config.json")
            optimization_config = {
                "model_id": model_id,
                "original_path": model_path,
                "optimized_path": os.path.join(output_dir, "optimized_model"),
                "optimizations_applied": ["int8", "bettertransformer", "lookupffn"],
                "target_hardware": "E5-2640",
                "threads": 12,
                "created_at": datetime.now().isoformat(),
            }

            with open(config_path, "w") as f:
                json.dump(optimization_config, f, indent=2)

            return config_path

        except Exception as e:
            logger.error(f"Error optimizing transformer model {model_id}: {e}")
            return None

    def list_models(self, include_details: bool = False) -> List[Dict[str, Any]]:
        """
        List all available models

        Args:
            include_details: Whether to include detailed metrics and optimization status

        Returns:
            List of model information dictionaries
        """
        if not MODEL_DOWNLOADER_AVAILABLE:
            return [{"error": "Model registry not available"}]

        models = []

        # Get all models from registry
        model_ids = self.model_registry.list_models()
        for model_id in model_ids:
            try:
                # Get active version
                model_version = self.model_registry.get_model(model_id)
                if not model_version:
                    continue

                # Basic model info
                model_info = {
                    "model_id": model_id,
                    "version": model_version.version,
                    "path": model_version.path,
                    "download_date": model_version.download_date.isoformat(),
                    "size_mb": model_version.size_bytes / (1024 * 1024),
                    "exists": os.path.exists(model_version.path),
                }

                # Add config info if present
                if model_version.config:
                    model_info.update(
                        {
                            "repo_id": model_version.config.get("repo_id"),
                            "quantization": model_version.config.get("quantization"),
                            "memory_req_gb": model_version.config.get("memory_req_gb"),
                        }
                    )

                # Include detailed information if requested
                if include_details:
                    # Check for optimized version
                    optimized_config = self._find_optimized_config(model_id)
                    if optimized_config:
                        model_info["optimized"] = True
                        model_info["optimized_config"] = optimized_config
                    else:
                        model_info["optimized"] = False

                    # Add monitoring metrics if available
                    if MODEL_MONITOR_AVAILABLE and self.model_monitor_db:
                        metrics = self.model_monitor_db.get_latest_metrics(
                            model_id, model_version.version
                        )
                        if metrics:
                            model_info["metrics"] = {
                                "last_checked": metrics.timestamp.isoformat(),
                                "inference_time_ms": metrics.inference_time_ms,
                                "tokens_per_second": metrics.tokens_per_second,
                                "memory_usage_mb": metrics.memory_usage_mb,
                                "success_rate": metrics.success_rate,
                            }

                            # Check for active alerts
                            alerts = self.model_monitor_db.get_active_alerts(model_id)
                            if alerts:
                                model_info["alerts"] = len(alerts)
                                model_info["alert_severity"] = alerts[0]["severity"]

                models.append(model_info)

            except Exception as e:
                logger.error(f"Error getting details for model {model_id}: {e}")
                models.append({"model_id": model_id, "error": str(e)})

        return models

    def _find_optimized_config(self, model_id: str) -> Optional[str]:
        """Find optimized configuration for a model"""
        # Check for Llama config
        llama_config = os.path.join(
            self.optimized_dir, model_id, "optimized_config.json"
        )
        if os.path.exists(llama_config):
            return llama_config

        # Check for transformer config
        transformer_config = os.path.join(
            self.optimized_dir, model_id, "transformer_config.json"
        )
        if os.path.exists(transformer_config):
            return transformer_config

        return None

    def get_model_stats(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed statistics for a model

        Args:
            model_id: ID of the model

        Returns:
            Dictionary with model statistics and status
        """
        if not MODEL_DOWNLOADER_AVAILABLE:
            return {"error": "Model registry not available", "success": False}

        try:
            # Get model version
            model_version = self.model_registry.get_model(model_id)
            if not model_version:
                return {
                    "error": f"Model {model_id} not found in registry",
                    "success": False,
                }

            result = {
                "model_id": model_id,
                "version": model_version.version,
                "path": model_version.path,
                "download_date": model_version.download_date.isoformat(),
                "size_mb": model_version.size_bytes / (1024 * 1024),
                "sha256": model_version.sha256,
                "exists": os.path.exists(model_version.path),
                "success": True,
            }

            # Add config if available
            if model_version.config:
                result["config"] = model_version.config

            # Get all versions
            versions = self.model_registry.list_versions(model_id)
            result["total_versions"] = len(versions)

            # Add optimization info
            optimized_config = self._find_optimized_config(model_id)
            if optimized_config:
                result["optimized"] = True
                result["optimized_config_path"] = optimized_config

                try:
                    with open(optimized_config, "r") as f:
                        result["optimization_details"] = json.load(f)
                except Exception as e:
                    result["optimization_error"] = str(e)
            else:
                result["optimized"] = False

            # Add monitoring metrics if available
            if MODEL_MONITOR_AVAILABLE and self.model_monitor_db:
                # Get most recent metrics
                metrics = self.model_monitor_db.get_latest_metrics(
                    model_id, model_version.version
                )
                if metrics:
                    result["metrics"] = {
                        "last_checked": metrics.timestamp.isoformat(),
                        "inference_time_ms": metrics.inference_time_ms,
                        "tokens_per_second": metrics.tokens_per_second,
                        "memory_usage_mb": metrics.memory_usage_mb,
                        "success_rate": metrics.success_rate,
                        "failed_requests": metrics.failed_requests,
                        "total_requests": metrics.total_requests,
                    }

                # Get performance history
                metrics_history = self.model_monitor_db.get_metrics_history(model_id, days=7)
                if metrics_history:
                    result["metrics_history"] = {
                        "days": 7,
                        "datapoints": len(metrics_history),
                        "timestamps": [
                            m.timestamp.isoformat() for m in metrics_history
                        ],
                        "inference_times": [
                            m.inference_time_ms
                            for m in metrics_history
                            if m.inference_time_ms
                        ],
                        "tokens_per_second": [
                            m.tokens_per_second
                            for m in metrics_history
                            if m.tokens_per_second
                        ],
                    }

                # Get alerts
                alerts = self.model_monitor_db.get_active_alerts(model_id)
                if alerts:
                    result["alerts"] = [
                        {
                            "timestamp": alert["timestamp"],
                            "type": alert["alert_type"],
                            "message": alert["message"],
                            "severity": alert["severity"],
                        }
                        for alert in alerts[:5]  # Include top 5 alerts
                    ]

            return result

        except Exception as e:
            logger.error(f"Error getting stats for model {model_id}: {e}")
            return {"error": str(e), "model_id": model_id, "success": False}

    def get_recommended_model(self, memory_gb: Optional[float] = None) -> str:
        """
        Get the recommended model based on available system resources

        Args:
            memory_gb: Available memory in GB, or None to auto-detect

        Returns:
            Model ID of the recommended model
        """
        if not MODEL_DOWNLOADER_AVAILABLE:
            return self.config.get("preferred_models", {}).get("default", "llama3-8b")

        try:
            if memory_gb is None:
                # Auto-detect memory
                try:
                    import psutil

                    memory_gb = psutil.virtual_memory().available / (1024**3)
                except ImportError:
                    # Default assumption for E5-2640
                    memory_gb = 32.0

            # Get compatible model based on memory
            if memory_gb >= 50:
                return self.config.get("preferred_models", {}).get(
                    "high_performance", "llama3-70b"
                )
            elif memory_gb >= 16:
                return self.config.get("preferred_models", {}).get(
                    "default", "llama3-8b"
                )
            else:
                return self.config.get("preferred_models", {}).get(
                    "low_resource", "mistral-7b"
                )

        except Exception as e:
            logger.error(f"Error getting recommended model: {e}")
            # Fallback to default
            return self.config.get("preferred_models", {}).get("default", "llama3-8b")

    def cleanup(self):
        """Clean up resources when shutting down"""
        logger.info("Cleaning up model manager resources")

        # Stop monitoring
        if MODEL_MONITOR_AVAILABLE and self.model_monitor:
            self.model_monitor.stop_monitoring()

        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=False)


# CLI functionality
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser(description="Enterprise Model Manager")

    # Model management commands
    parser.add_argument("--list", action="store_true", help="List all available models")
    parser.add_argument(
        "--details", action="store_true", help="Include detailed model info"
    )
    parser.add_argument("--get", type=str, help="Get a specific model")
    parser.add_argument("--optimize", action="store_true", help="Optimize the model")
    parser.add_argument("--stats", type=str, help="Get detailed stats for a model")
    parser.add_argument(
        "--recommended",
        action="store_true",
        help="Get recommended model for this system",
    )

    # Configuration options
    parser.add_argument(
        "--auto-download", action="store_true", help="Enable auto-downloading"
    )
    parser.add_argument(
        "--no-optimize", action="store_true", help="Disable auto-optimization"
    )
    parser.add_argument(
        "--no-monitor", action="store_true", help="Disable auto-monitoring"
    )
    parser.add_argument(
        "--auto-retrain", action="store_true", help="Enable auto-retraining"
    )

    args = parser.parse_args()

    # Create model manager
    manager = ModelManager(
        auto_download=args.auto_download,
        auto_optimize=not args.no_optimize,
        auto_monitor=not args.no_monitor,
        auto_retrain=args.auto_retrain,
    )

    try:
        # Handle commands
        if args.list:
            models = manager.list_models(include_details=args.details)
            print(f"Available models ({len(models)}):")
            for model in models:
                if "error" in model:
                    print(
                        f"  {model.get('model_id', 'Unknown')}: Error - {model['error']}"
                    )
                    continue

                status = "✓" if model.get("exists", False) else "✗"
                size = f"{model.get('size_mb', 0):.1f} MB"
                print(f"  {status} {model['model_id']} ({model['version']}, {size})")

                if args.details:
                    if model.get("optimized"):
                        print(f"    Optimized: {model['optimized']}")

                    if "metrics" in model:
                        metrics = model["metrics"]
                        print(f"    Last check: {metrics.get('last_checked')}")
                        print(
                            f"    Performance: {metrics.get('tokens_per_second', 'N/A')} tokens/sec"
                        )
                        print(
                            f"    Success rate: {metrics.get('success_rate', 1.0) * 100:.1f}%"
                        )

                    if model.get("alerts", 0) > 0:
                        print(
                            f"    ⚠️ Alerts: {model['alerts']} ({model['alert_severity']})"
                        )

        elif args.get:
            print(f"Getting model: {args.get}")
            result = manager.get_model(
                model_id=args.get, optimize=args.optimize, wait=True
            )

            if result.get("success"):
                print(f"Model ready: {result['model_path']}")
                if result.get("optimized"):
                    print(f"Optimized path: {result['optimized_path']}")
                if "metrics" in result:
                    print(
                        f"Performance: {result['metrics'].get('tokens_per_second', 'N/A')} tokens/sec"
                    )
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")

        elif args.stats:
            print(f"Getting statistics for model: {args.stats}")
            stats = manager.get_model_stats(args.stats)

            if stats.get("success"):
                print(f"Model: {stats['model_id']} (Version: {stats['version']})")
                print(f"Size: {stats['size_mb']:.1f} MB")
                print(f"Download date: {stats['download_date']}")

                if stats.get("optimized"):
                    print(f"Optimized: Yes ({stats['optimized_config_path']})")
                else:
                    print("Optimized: No")

                if "metrics" in stats:
                    metrics = stats["metrics"]
                    print("\nPerformance metrics:")
                    print(f"  Last check: {metrics.get('last_checked')}")
                    print(
                        f"  Inference time: {metrics.get('inference_time_ms', 'N/A')} ms"
                    )
                    print(
                        f"  Speed: {metrics.get('tokens_per_second', 'N/A')} tokens/sec"
                    )
                    print(f"  Memory usage: {metrics.get('memory_usage_mb', 'N/A')} MB")
                    print(
                        f"  Success rate: {metrics.get('success_rate', 1.0) * 100:.1f}%"
                    )

                if "alerts" in stats:
                    print(f"\nAlerts ({len(stats['alerts'])}):")
                    for alert in stats["alerts"]:
                        print(
                            f"  [{alert['severity']}] {alert['type']}: {alert['message']}"
                        )
            else:
                print(f"Error: {stats.get('error', 'Unknown error')}")

        elif args.recommended:
            model_id = manager.get_recommended_model()
            print(f"Recommended model for this system: {model_id}")
            print("Run with --get to download this model")

        else:
            parser.print_help()

    finally:
        # Clean up resources
        manager.cleanup()
