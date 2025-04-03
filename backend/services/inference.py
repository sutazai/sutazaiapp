# /opt/sutazaiapp/backend/services/inference.py

import os
import time
import logging
import psutil
import numpy as np
import gc
import json
import threading
from typing import Dict, List, Any, Optional
import torch
import hashlib
import subprocess
from sentence_transformers import SentenceTransformer
from multiprocessing.pool import ThreadPool

logger = logging.getLogger("sutazai.inference")


class MemoryMonitor:
    """Monitors system memory usage and provides recommendations."""

    def __init__(self, update_interval: float = 5.0):
        """
        Initialize memory monitor.

        Args:
            update_interval: How often to update memory stats (seconds)
        """
        self.update_interval = update_interval
        self.last_update = 0
        self.memory_stats: Dict[str, Any] = {}
        self.update_memory_stats()

        # Memory usage history for trend analysis
        self.history: List[Dict[str, Any]] = []
        self.history_max_size = 100

        # Start background monitoring thread
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def update_memory_stats(self):
        """Update memory statistics."""
        current_time = time.time()
        if current_time - self.last_update > self.update_interval:
            vm = psutil.virtual_memory()
            self.memory_stats = {
                "total": vm.total / (1024 * 1024),  # MB
                "available": vm.available / (1024 * 1024),  # MB
                "used": vm.used / (1024 * 1024),  # MB
                "free": vm.free / (1024 * 1024),  # MB
                "percent": vm.percent,
                "time": current_time,
            }

            # Add to history
            self.history.append(self.memory_stats.copy())
            if len(self.history) > self.history_max_size:
                self.history.pop(0)

            self.last_update = current_time

            # Log if memory usage is high
            if vm.percent > 90:
                logger.warning(f"High memory usage: {vm.percent}%")

    def _monitor_loop(self):
        """Background thread for continuous memory monitoring."""
        while self.running:
            self.update_memory_stats()
            time.sleep(self.update_interval)

    def get_available_memory(self) -> float:
        """Get available memory in MB."""
        self.update_memory_stats()
        return self.memory_stats["available"]

    def get_memory_usage_percent(self) -> float:
        """Get memory usage percentage."""
        self.update_memory_stats()
        return self.memory_stats["percent"]

    def get_memory_trend(self, minutes: int = 5) -> Dict[str, Any]:
        """
        Analyze memory usage trend over time.

        Args:
            minutes: Time period to analyze in minutes

        Returns:
            Dict with trend information
        """
        self.update_memory_stats()

        # Filter history by time
        cutoff_time = time.time() - (minutes * 60)
        recent_history = [
            entry for entry in self.history if entry["time"] >= cutoff_time
        ]

        if len(recent_history) < 2:
            return {
                "trend": "stable",
                "change_rate": 0.0,
                "data_points": len(recent_history),
            }

        # Calculate trend
        first = recent_history[0]["percent"]
        last = recent_history[-1]["percent"]
        time_diff = recent_history[-1]["time"] - recent_history[0]["time"]

        if time_diff <= 0:
            return {
                "trend": "unknown",
                "change_rate": 0.0,
                "data_points": len(recent_history),
            }

        change_rate = (last - first) / time_diff * 60  # Percent per minute

        trend = "stable"
        if change_rate > 1.0:
            trend = "increasing_fast"
        elif change_rate > 0.2:
            trend = "increasing"
        elif change_rate < -1.0:
            trend = "decreasing_fast"
        elif change_rate < -0.2:
            trend = "decreasing"

        return {
            "trend": trend,
            "change_rate": change_rate,
            "data_points": len(recent_history),
        }

    def is_memory_pressure_high(self) -> bool:
        """Check if system is under memory pressure."""
        self.update_memory_stats()

        # Check current usage
        if self.memory_stats["percent"] > 85:
            return True

        # Check trend
        trend = self.get_memory_trend(minutes=1)
        if trend["trend"] == "increasing_fast" and self.memory_stats["percent"] > 70:
            return True

        return False

    def estimate_safe_memory_allocation(self) -> int:
        """Estimate safe amount of memory to allocate in MB."""
        available = self.get_available_memory()

        # Leave buffer depending on current usage and trend
        buffer_percent = 20  # Default buffer

        if self.get_memory_usage_percent() > 70:
            buffer_percent = 30

        trend = self.get_memory_trend(minutes=1)
        if trend["trend"] in ("increasing", "increasing_fast"):
            buffer_percent += 10

        safe_allocation = int(available * (1 - buffer_percent / 100))

        # Ensure minimum allocation is reasonable
        return max(128, safe_allocation)

    def stop(self):
        """Stop the monitoring thread."""
        self.running = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=3)


class ModelManager:
    """Manages AI models with efficient resource usage."""

    def __init__(self, models_dir: str):
        """
        Initialize model manager.

        Args:
            models_dir: Directory containing models
        """
        self.models_dir = models_dir
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.model_locks: Dict[str, threading.RLock] = {}
        self.memory_monitor = MemoryMonitor()

        # Load model metadata
        self._load_model_metadata()

        logger.info(f"Initialized model manager with {len(self.model_metadata)} models")

    def _load_model_metadata(self):
        """Load metadata for available models."""
        if not os.path.exists(self.models_dir):
            logger.warning(f"Models directory not found: {self.models_dir}")
            return

        # Look for model metadata files
        for model_name in os.listdir(self.models_dir):
            model_dir = os.path.join(self.models_dir, model_name)
            metadata_path = os.path.join(model_dir, "metadata.json")

            if os.path.isdir(model_dir) and os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)

                    # Add the model to metadata
                    self.model_metadata[model_name] = metadata

                    # Initialize lock for this model
                    self.model_locks[model_name] = threading.RLock()

                    logger.info(f"Loaded metadata for model: {model_name}")
                except Exception as e:
                    logger.error(
                        f"Error loading metadata for model {model_name}: {str(e)}"
                    )

    def get_model_memory_requirement(self, model_name: str) -> int:
        """
        Get estimated memory requirement for a model in MB.

        Args:
            model_name: Name of the model

        Returns:
            Estimated memory requirement in MB
        """
        if model_name not in self.model_metadata:
            return 0

        # Get memory requirement from metadata if available
        memory_req = self.model_metadata[model_name].get("memory_requirement_mb", 0)

        if memory_req > 0:
            return memory_req

        # Estimate based on parameters if available
        parameters = self.model_metadata[model_name].get("parameters", 0)
        if parameters > 0:
            # Rough estimate: 4 bytes per parameter for FP32
            # or 2 bytes per parameter for FP16 or 1 byte for int8
            precision = self.model_metadata[model_name].get("precision", "fp16")

            bytes_per_param = 4  # fp32 default
            if precision == "fp16":
                bytes_per_param = 2
            elif precision == "int8":
                bytes_per_param = 1

            # Add 20% overhead for additional tensors and buffers
            memory_req = int(parameters * bytes_per_param * 1.2 / (1024 * 1024))
            return max(memory_req, 100)  # Minimum 100MB

        # Default estimate based on model type
        model_type = self.model_metadata[model_name].get("type", "unknown")

        if model_type == "embedding":
            return 500  # 500MB for embedding models
        elif model_type == "onnx":
            return 2000  # 2GB for ONNX models
        else:
            return 4000  # 4GB default for unknown models

    def can_load_model(self, model_name: str) -> bool:
        """
        Check if a model can be loaded given current memory constraints.

        Args:
            model_name: Name of the model

        Returns:
            True if the model can be loaded, False otherwise
        """
        # If model is already loaded, it can be used
        if model_name in self.loaded_models:
            return True

        # Get memory requirement for the model
        memory_req = self.get_model_memory_requirement(model_name)

        # Check if there's enough memory available
        safe_allocation = self.memory_monitor.estimate_safe_memory_allocation()

        if memory_req > safe_allocation:
            logger.warning(
                f"Insufficient memory to load model {model_name}. "
                f"Required: {memory_req}MB, Available: {safe_allocation}MB"
            )
            return False

        return True

    def load_model(
        self, model_name: str, optimization_level: str = "medium"
    ) -> Optional[Any]:
        """
        Load a model with specified optimization level.

        Args:
            model_name: Name of the model to load
            optimization_level: One of "none", "low", "medium", "high"

        Returns:
            Loaded model or None if loading failed
        """
        if model_name not in self.model_metadata:
            logger.error(f"Model not found in metadata: {model_name}")
            return None

        # Check if model is already loaded
        if model_name in self.loaded_models:
            logger.info(f"Model already loaded: {model_name}")
            return self.loaded_models[model_name]["model"]

        # Check if model can be loaded
        if not self.can_load_model(model_name):
            logger.error(f"Cannot load model due to memory constraints: {model_name}")
            return None

        # Acquire lock for this model
        with self.model_locks[model_name]:
            # Double-check after acquiring lock
            if model_name in self.loaded_models:
                return self.loaded_models[model_name]["model"]

            logger.info(
                f"Loading model: {model_name} with optimization level: {optimization_level}"
            )

            try:
                # Record memory before loading
                mem_before = self.memory_monitor.get_memory_usage_percent()

                # Get model path and type
                model_path = os.path.join(self.models_dir, model_name)
                model_type = self.model_metadata[model_name].get("type", "unknown")

                # Different loading strategies based on model type and optimization level
                model = None

                if model_type == "onnx":
                    model = self._load_onnx_model(
                        model_name, model_path, optimization_level
                    )
                elif model_type in ["causal_lm", "seq2seq", "encoder"]:
                    model = self._load_transformer_model(
                        model_name, model_path, optimization_level
                    )
                elif model_type == "embedding":
                    model = self._load_embedding_model(
                        model_name, model_path, optimization_level
                    )
                else:
                    logger.error(f"Unsupported model type: {model_type}")
                    return None

                if model is None:
                    logger.error(f"Failed to load model: {model_name}")
                    return None

                # Record memory after loading
                mem_after = self.memory_monitor.get_memory_usage_percent()

                # Store the loaded model
                self.loaded_models[model_name] = {
                    "model": model,
                    "load_time": time.time(),
                    "last_used": time.time(),
                    "usage_count": 0,
                    "memory_impact": mem_after - mem_before,
                }

                logger.info(
                    f"Successfully loaded model: {model_name}. Memory impact: {mem_after - mem_before}%"
                )

                # Perform garbage collection to clean up any temporary objects
                gc.collect()

                return model

            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
                # Force garbage collection to clean up partial load
                gc.collect()
                return None

    def _load_onnx_model(
        self, model_name: str, model_path: str, optimization_level: str
    ) -> Optional[Any]:
        """Load an ONNX model with optimizations."""
        try:
            import onnxruntime as ort

            # Configure session options based on optimization level
            sess_options = ort.SessionOptions()

            # Set optimization level
            if optimization_level == "none":
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                )
            elif optimization_level == "low":
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                )
            elif optimization_level == "medium":
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
                )
            else:  # high
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )

            # Enable parallel execution on E5-2640 (12 cores, 24 threads)
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            sess_options.intra_op_num_threads = 12  # One per physical core
            sess_options.inter_op_num_threads = 12  # One per physical core

            # Enable memory pattern optimization
            sess_options.enable_mem_pattern = True

            # Reduce memory usage
            sess_options.enable_cpu_mem_arena = False

            # Find model file
            model_file = os.path.join(model_path, "model.onnx")
            if not os.path.exists(model_file):
                # Try to find any .onnx file
                onnx_files = [f for f in os.listdir(model_path) if f.endswith(".onnx")]
                if not onnx_files:
                    logger.error(f"No ONNX model file found in {model_path}")
                    return None
                model_file = os.path.join(model_path, onnx_files[0])

            # Load the model
            logger.info(f"Loading ONNX model from {model_file}")
            model = ort.InferenceSession(model_file, sess_options)

            # Warmup the model with a dummy input
            input_names = [input.name for input in model.get_inputs()]
            dummy_inputs = {}
            for input_name in input_names:
                # Get input shape and type
                input_info = next(
                    input for input in model.get_inputs() if input.name == input_name
                )
                shape = input_info.shape
                dtype = input_info.type

                # Create dummy input
                if -1 in shape:
                    # Replace dynamic dimensions with small values
                    shape = [1 if dim == -1 else dim for dim in shape]

                if "float" in dtype:
                    dummy_inputs[input_name] = np.zeros(shape, dtype=np.float32)
                elif "int" in dtype:
                    dummy_inputs[input_name] = np.zeros(shape, dtype=np.int64)
                else:
                    dummy_inputs[input_name] = np.zeros(shape, dtype=np.float32)

            # Run warmup inference
            if dummy_inputs:
                logger.info(f"Running warmup inference for ONNX model {model_name}")
                _ = model.run(None, dummy_inputs)

            return model

        except ImportError:
            logger.error(
                "ONNX Runtime not installed. Please install onnxruntime package."
            )
            return None
        except Exception as e:
            logger.error(f"Error loading ONNX model: {str(e)}")
            return None

    def _load_transformer_model(
        self, model_name: str, model_path: str, optimization_level: str
    ) -> Optional[Any]:
        """Load a transformer model with optimizations."""
        try:
            from transformers import (
                AutoModel,
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
            )

            # Get model type
            model_type = self.model_metadata[model_name].get("type", "causal_lm")

            # Set loading parameters based on optimization level
            loading_params = {
                "device_map": "cpu",
                "torch_dtype": torch.float16,  # Default to FP16 for memory efficiency
            }

            # Apply quantization for higher optimization levels
            if optimization_level in ["medium", "high"]:
                try:
                    # Try to use bitsandbytes for quantization
                    # F401: Removed unused import bitsandbytes
                    # import bitsandbytes as bnb
                    if optimization_level == "medium":
                        # 8-bit quantization
                        loading_params["load_in_8bit"] = True
                    else:
                        # 4-bit quantization
                        loading_params["load_in_4bit"] = True
                        loading_params["bnb_4bit_compute_dtype"] = torch.float16
                        loading_params["bnb_4bit_quant_type"] = "nf4"
                except ImportError:
                    logger.warning("bitsandbytes not available, falling back to fp16")

            # Low CPU memory usage
            loading_params["low_cpu_mem_usage"] = True

            # Try to use optimum library for CPU optimization if available
            try:
                if optimization_level in ["medium", "high"]:
                    from optimum.onnxruntime import (
                        ORTModelForCausalLM,
                        ORTModelForSeq2SeqLM,
                    )

                    if model_type == "causal_lm":
                        # Load with ONNX runtime
                        model = ORTModelForCausalLM.from_pretrained(
                            model_path, export=True, provider="CPUExecutionProvider"
                        )
                        logger.info(
                            f"Loaded model {model_name} with optimum ORTModelForCausalLM"
                        )
                        return model
                    elif model_type == "seq2seq":
                        # Load with ONNX runtime
                        model = ORTModelForSeq2SeqLM.from_pretrained(
                            model_path, export=True, provider="CPUExecutionProvider"
                        )
                        logger.info(
                            f"Loaded model {model_name} with optimum ORTModelForSeq2SeqLM"
                        )
                        return model
            except ImportError:
                logger.info(
                    "optimum.onnxruntime not available, falling back to standard loading"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load with optimum: {str(e)}, falling back to standard loading"
                )

            # Standard loading based on model type
            if model_type == "causal_lm":
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, **loading_params
                )
            elif model_type == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path, **loading_params
                )
            else:  # encoder or other
                model = AutoModel.from_pretrained(model_path, **loading_params)

            # Apply BetterTransformer optimization for high optimization level
            if optimization_level == "high":
                try:
                    model = model.to_bettertransformer()
                    logger.info("Applied BetterTransformer optimization")
                except Exception as e:
                    logger.warning(
                        f"BetterTransformer optimization not available or failed for this model: {e}"
                    )

            # Set model to evaluation mode
            model.eval()

            # Optimize torch inference
            if hasattr(torch, "set_num_threads"):
                torch.set_num_threads(12)  # Number of physical cores

            # Run warmup with small input
            logger.info(f"Running warmup inference for model {model_name}")
            model_input = torch.ones((1, 8), dtype=torch.long)
            with torch.no_grad():
                _ = model(model_input)

            return model

        except ImportError:
            logger.error(
                "Transformers library not installed. Please install transformers package."
            )
            return None
        except Exception as e:
            logger.error(f"Error loading transformer model: {str(e)}")
            return None

    def _load_embedding_model(
        self, model_name: str, model_path: str, optimization_level: str
    ) -> Optional[Any]:
        """Load an embedding model with optimizations."""
        try:
            from sentence_transformers import SentenceTransformer

            # Optimize for CPU
            model = SentenceTransformer(model_path, device="cpu")

            # Set threading options
            if hasattr(torch, "set_num_threads"):
                torch.set_num_threads(12)  # Number of physical cores

            # Warmup
            logger.info(f"Running warmup inference for embedding model {model_name}")
            _ = model.encode("Warmup text", convert_to_numpy=True)

            return model

        except ImportError:
            logger.error(
                "SentenceTransformer not installed. Please install sentence-transformers package."
            )
            return None
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            return None

    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model to free up memory.

        Args:
            model_name: Name of the model to unload

        Returns:
            True if successful, False otherwise
        """
        if model_name not in self.loaded_models:
            logger.warning(f"Model not loaded: {model_name}")
            return False

        # Acquire lock for this model
        with self.model_locks[model_name]:
            # Double-check after acquiring lock
            if model_name not in self.loaded_models:
                return False

            try:
                logger.info(f"Unloading model: {model_name}")

                # Save usage statistics
                model_info = self.loaded_models[model_name]
                usage_time = time.time() - model_info["load_time"]

                # Delete the model
                del self.loaded_models[model_name]["model"]
                del self.loaded_models[model_name]

                # Force GPU memory cleanup if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Force garbage collection
                gc.collect()

                logger.info(
                    f"Successfully unloaded model: {model_name}. Usage time: {usage_time:.2f} seconds"
                )

                return True

            except Exception as e:
                logger.error(f"Error unloading model {model_name}: {str(e)}")
                return False

    def get_model_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for loaded models."""
        stats = {}

        for model_name, model_info in self.loaded_models.items():
            stats[model_name] = {
                "load_time": model_info["load_time"],
                "usage_count": model_info["usage_count"],
                "last_used": model_info["last_used"],
                "memory_impact": model_info["memory_impact"],
                "age_seconds": time.time() - model_info["load_time"],
            }

        return stats

    def manage_memory(self, required_memory_mb: int = 0) -> bool:
        """
        Manage memory by unloading models if necessary.

        Args:
            required_memory_mb: Additional memory required in MB

        Returns:
            True if sufficient memory is available or was freed, False otherwise
        """
        # Check current memory situation
        safe_allocation = self.memory_monitor.estimate_safe_memory_allocation()

        # If we already have enough memory, return True
        if required_memory_mb <= safe_allocation:
            return True

        logger.info(
            f"Memory management triggered. Required: {required_memory_mb}MB, Available: {safe_allocation}MB"
        )

        # Get model usage stats
        model_stats = []
        for model_name, model_info in self.loaded_models.items():
            model_stats.append(
                {
                    "name": model_name,
                    "last_used": model_info["last_used"],
                    "usage_count": model_info["usage_count"],
                    "memory_impact": model_info["memory_impact"],
                }
            )

        # Sort by last used time (oldest first)
        model_stats.sort(key=lambda x: x["last_used"])

        # Try to free up memory by unloading models
        freed_memory = 0
        for model_stat in model_stats:
            model_name = model_stat["name"]
            memory_req = self.get_model_memory_requirement(model_name)

            # Unload the model
            if self.unload_model(model_name):
                freed_memory += memory_req
                logger.info(f"Unloaded model {model_name} to free {memory_req}MB")

                # Check if we have enough memory now
                safe_allocation = self.memory_monitor.estimate_safe_memory_allocation()
                if required_memory_mb <= safe_allocation:
                    return True

        # If we still don't have enough memory
        safe_allocation = self.memory_monitor.estimate_safe_memory_allocation()
        if required_memory_mb <= safe_allocation:
            return True

        logger.warning(
            f"Could not free enough memory. Still need {required_memory_mb - safe_allocation}MB more"
        )
        return False

    def mark_model_used(self, model_name: str):
        """Mark a model as recently used to prevent early unloading."""
        if model_name in self.loaded_models:
            self.loaded_models[model_name]["last_used"] = time.time()
            self.loaded_models[model_name]["usage_count"] += 1

    def cleanup(self):
        """Clean up resources."""
        # Unload all models
        model_names = list(self.loaded_models.keys())
        for model_name in model_names:
            self.unload_model(model_name)

        # Stop memory monitor
        self.memory_monitor.stop()

        logger.info("Model manager cleanup complete")


class MemoryEfficientInference:
    """Manages efficient inference for transformer models on CPU."""

    def __init__(
        self,
        models_dir: str,
        cache_dir: Optional[str] = None,
        max_concurrent_models: int = 3,
    ):
        """
        Initialize memory-efficient inference manager.

        Args:
            models_dir: Directory containing models
            cache_dir: Directory for caching results (optional)
            max_concurrent_models: Maximum number of concurrently loaded models
        """
        self.model_manager = ModelManager(models_dir)
        self.cache_dir = cache_dir
        self.max_concurrent_models = max_concurrent_models

        # Create cache directory if specified
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        # Inference cache
        self.result_cache = ResultCache(cache_dir) if cache_dir else None

        # Inference parallelism
        self.inference_pool = threading.ThreadPool(
            max_workers=min(12, os.cpu_count() or 4)
        )

        # Locks for concurrent model usage
        self.model_usage_locks: Dict[str, threading.RLock] = {}

        logger.info(
            f"Initialized memory-efficient inference with model dir: {models_dir}"
        )

    def generate_text(
        self,
        model_name: str,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        optimization_level: str = "medium",
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate text using a language model.

        Args:
            model_name: Name of the model to use
            prompt: Text prompt for generation
            max_length: Maximum number of tokens to generate
            temperature: Temperature for sampling (higher = more random)
            optimization_level: Optimization level (none, low, medium, high)
            use_cache: Whether to use result caching

        Returns:
            Dictionary with generation results
        """
        # Check cache first if enabled
        if use_cache and self.result_cache:
            cache_key = self.result_cache.make_key(
                model_name=model_name,
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
            )

            cached_result = self.result_cache.get(cache_key)
            if cached_result:
                cached_result["cached"] = True
                cached_result["generation_time"] = 0.0
                logger.info(f"Cache hit for generation with model {model_name}")
                return cached_result

        # Initialize result
        result = {
            "model": model_name,
            "prompt": prompt,
            "generated_text": "",
            "error": None,
            "generation_time": 0.0,
            "cached": False,
        }

        # Check memory availability
        memory_req = self.model_manager.get_model_memory_requirement(model_name)
        if not self.model_manager.manage_memory(memory_req):
            result["error"] = "Insufficient memory to load model"
            return result

        # Get or initialize model lock
        if model_name not in self.model_usage_locks:
            self.model_usage_locks[model_name] = threading.RLock()

        # Load the model
        with self.model_usage_locks[model_name]:
            model = self.model_manager.load_model(model_name, optimization_level)

            if model is None:
                result["error"] = "Failed to load model"
                return result

            # Mark model as used
            self.model_manager.mark_model_used(model_name)

            try:
                # Get model type
                model_type = self.model_manager.model_metadata[model_name].get(
                    "type", "unknown"
                )

                generation_start = time.time()

                if model_type == "onnx":
                    generated_text = self._generate_with_onnx(
                        model, prompt, max_length, temperature
                    )
                else:  # transformer models
                    generated_text = self._generate_with_transformer(
                        model, prompt, max_length, temperature
                    )

                generation_time = time.time() - generation_start

                # Update result
                result["generated_text"] = generated_text
                result["generation_time"] = generation_time

                # Cache result if enabled
                if use_cache and self.result_cache:
                    self.result_cache.put(cache_key, result.copy())

                logger.info(
                    f"Generated text with model {model_name} in {generation_time:.2f}s"
                )

                return result

            except Exception as e:
                logger.error(f"Error generating text with model {model_name}: {str(e)}")
                result["error"] = str(e)
                return result

    def _generate_with_transformer(
        self, model, prompt: str, max_length: int, temperature: float
    ) -> str:
        """Generate text with a transformer model."""
        try:
            from transformers import AutoTokenizer
            import torch

            # Get or create tokenizer
            if not hasattr(model, "tokenizer"):
                try:
                    # Try to get from model's config
                    model_id = getattr(model.config, "name_or_path", None)
                    if model_id:
                        tokenizer = AutoTokenizer.from_pretrained(model_id)
                    else:
                        # Assume tokenizer is in same directory
                        model_dir = os.path.dirname(model.config._name_or_path)
                        tokenizer = AutoTokenizer.from_pretrained(model_dir)
                except Exception as e:
                    logger.warning(
                        f"Could not load model-specific tokenizer, using default. Error: {e}"
                    )
                    # Fallback to default tokenizer
                    tokenizer = AutoTokenizer.from_pretrained("gpt2")
                model.tokenizer = tokenizer
            else:
                tokenizer = model.tokenizer

            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            input_length = inputs.input_ids.shape[1]

            # Ensure max_length is reasonable
            total_length = input_length + max_length

            # Configure generation parameters
            generation_config = {
                "max_length": total_length,
                "temperature": max(0.1, temperature),  # Minimum temperature
                "do_sample": temperature
                > 0.1,  # Use sampling if temperature is significant
                "pad_token_id": tokenizer.eos_token_id,  # Use EOS as pad token if needed
                "num_return_sequences": 1,
                "use_cache": True,
            }

            # If temperature is very low, use greedy decoding
            if temperature <= 0.1:
                generation_config["do_sample"] = False
                generation_config["num_beams"] = 1

            # Generate with memory optimization
            with torch.no_grad():
                # Process in chunks if the output is very long
                if max_length > 256 and hasattr(model, "generate"):
                    # Generate in chunks of 256 tokens
                    chunk_size = 256
                    output_ids = inputs.input_ids.clone()

                    remaining_tokens = max_length
                    while remaining_tokens > 0:
                        chunk = min(chunk_size, remaining_tokens)

                        # Configure chunked generation
                        chunk_config = generation_config.copy()
                        chunk_config["max_length"] = input_length + chunk

                        # Generate chunk
                        outputs = model.generate(inputs=output_ids, **chunk_config)

                        # Update output and input length
                        output_ids = outputs
                        input_length = output_ids.shape[1]

                        # Update remaining tokens
                        remaining_tokens -= chunk

                        # Break if model seems to have completed generation
                        if outputs[0][-1].item() == tokenizer.eos_token_id:
                            break
                else:
                    # Single generation for shorter outputs
                    outputs = model.generate(**inputs, **generation_config)

            # Decode generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the prompt from the beginning
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt) :]

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Error in transformer generation: {str(e)}")
            raise

    def _generate_with_onnx(
        self, onnx_model, prompt: str, max_length: int, temperature: float
    ) -> str:
        """Generate text with an ONNX model."""
        try:
            import numpy as np
            from transformers import AutoTokenizer

            # We need a tokenizer for the ONNX model
            if not hasattr(onnx_model, "tokenizer"):
                # Try to find a tokenizer in the model's directory
                model_dir = os.path.dirname(
                    onnx_model.get_session().get_inputs()[0].name
                )
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_dir)
                except Exception as e:
                    logger.warning(
                        f"Could not load model-specific tokenizer for ONNX, using default. Error: {e}"
                    )
                    # Fallback to default tokenizer
                    tokenizer = AutoTokenizer.from_pretrained("gpt2")
                onnx_model.tokenizer = tokenizer
            else:
                tokenizer = onnx_model.tokenizer

            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.numpy()

            # Get input names
            input_names = [input.name for input in onnx_model.get_inputs()]

            # Create input dict
            onnx_inputs = {"input_ids": input_ids}

            # Add attention mask if needed
            if "attention_mask" in input_names:
                attention_mask = np.ones((1, input_ids.shape[1]), dtype=np.int64)
                onnx_inputs["attention_mask"] = attention_mask

            # Run inference
            outputs = onnx_model.run(None, onnx_inputs)

            # Get output logits
            logits = outputs[0]

            # Simple greedy decoding (no temperature support in this basic implementation)
            next_token_id = np.argmax(logits[0, -1, :])
            generated_ids = [next_token_id]

            # Continue generating tokens
            for _ in range(max_length - 1):
                # Add new token to input
                next_input_ids = np.concatenate(
                    [input_ids, np.array([[next_token_id]])], axis=1
                )

                # Update attention mask if needed
                if "attention_mask" in input_names:
                    attention_mask = np.ones(
                        (1, next_input_ids.shape[1]), dtype=np.int64
                    )
                    onnx_inputs["attention_mask"] = attention_mask

                # Run inference
                onnx_inputs["input_ids"] = next_input_ids
                outputs = onnx_model.run(None, onnx_inputs)

                # Get next token
                logits = outputs[0]
                next_token_id = np.argmax(logits[0, -1, :])
                generated_ids.append(next_token_id)

                # Check for EOS
                if next_token_id == tokenizer.eos_token_id:
                    break

                # Update input for next iteration
                input_ids = next_input_ids

            # Decode generated text
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Error in ONNX generation: {str(e)}")
            raise

    def create_embeddings(
        self,
        model_name: str,
        texts: List[str],
        batch_size: int = 8,
        optimization_level: str = "medium",
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Create embeddings for texts.

        Args:
            model_name: Name of embedding model
            texts: List of texts to embed
            batch_size: Batch size for processing
            optimization_level: Optimization level
            use_cache: Whether to use result caching

        Returns:
            Dictionary with embedding results
        """
        # Check cache first if enabled
        if use_cache and self.result_cache:
            cache_key = self.result_cache.make_key(
                model_name=model_name, texts=texts, operation="embeddings"
            )

            cached_result = self.result_cache.get(cache_key)
            if cached_result:
                cached_result["cached"] = True
                cached_result["embedding_time"] = 0.0
                logger.info(f"Cache hit for embeddings with model {model_name}")
                return cached_result

        # Initialize result
        result = {
            "model": model_name,
            "text_count": len(texts),
            "embeddings": [],
            "embedding_time": 0.0,
            "error": None,
            "cached": False,
        }

        # Check memory availability
        memory_req = self.model_manager.get_model_memory_requirement(model_name)
        if not self.model_manager.manage_memory(memory_req):
            result["error"] = "Insufficient memory to load model"
            return result

        # Get or initialize model lock
        if model_name not in self.model_usage_locks:
            self.model_usage_locks[model_name] = threading.RLock()

        # Load the model
        with self.model_usage_locks[model_name]:
            model = self.model_manager.load_model(model_name, optimization_level)

            if model is None:
                result["error"] = "Failed to load model"
                return result

            # Mark model as used
            self.model_manager.mark_model_used(model_name)

            try:
                embedding_start = time.time()

                # Process in batches for memory efficiency
                all_embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i : i + batch_size]

                    # Create embeddings
                    if hasattr(model, "encode"):
                        # For sentence-transformers models
                        batch_embeddings = model.encode(
                            batch_texts, convert_to_numpy=True, show_progress_bar=False
                        )
                    else:
                        # For transformers models
                        from transformers import AutoTokenizer
                        import torch

                        # Get tokenizer
                        if not hasattr(model, "tokenizer"):
                            model_id = getattr(model.config, "name_or_path", None)
                            tokenizer = AutoTokenizer.from_pretrained(
                                model_id or "bert-base-uncased"
                            )
                            model.tokenizer = tokenizer
                        else:
                            tokenizer = model.tokenizer

                        # Tokenize
                        batch_encodings = tokenizer(
                            batch_texts,
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                        )

                        # Generate embeddings
                        with torch.no_grad():
                            outputs = model(
                                **batch_encodings, output_hidden_states=True
                            )

                        # Use the last hidden state
                        hidden_states = outputs.hidden_states[-1]

                        # Create sentence embeddings (mean pooling)
                        attention_mask = batch_encodings.attention_mask.unsqueeze(-1)
                        batch_embeddings = torch.sum(
                            hidden_states * attention_mask, 1
                        ) / torch.sum(attention_mask, 1)
                        batch_embeddings = batch_embeddings.cpu().numpy()

                    all_embeddings.extend(batch_embeddings.tolist())

                embedding_time = time.time() - embedding_start

                # Update result
                result["embeddings"] = all_embeddings
                result["embedding_time"] = embedding_time

                # Cache result if enabled
                if use_cache and self.result_cache:
                    self.result_cache.put(cache_key, result.copy())

                logger.info(
                    f"Created embeddings for {len(texts)} texts with model {model_name} in {embedding_time:.2f}s"
                )

                return result

            except Exception as e:
                logger.error(
                    f"Error creating embeddings with model {model_name}: {str(e)}"
                )
                result["error"] = str(e)
                return result

    def process_parallel(
        self,
        model_name: str,
        texts: List[str],
        task: str,
        batch_size: int = 8,
        optimization_level: str = "medium",
    ) -> List[Dict[str, Any]]:
        """
        Process multiple texts in parallel.

        Args:
            model_name: Name of the model
            texts: List of input texts
            task: Task type ('generate' or 'embed')
            batch_size: Batch size for processing
            optimization_level: Optimization level

        Returns:
            List of results
        """
        # Prepare batches
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        if task == "generate":
            # Use thread pool to process in parallel
            results = self.inference_pool.map(
                lambda batch: [
                    self.generate_text(
                        model_name=model_name,
                        prompt=text,
                        optimization_level=optimization_level,
                    )
                    for text in batch
                ],
                batches,
            )
            # Flatten results
            return [item for sublist in results for item in sublist]

        elif task == "embed":
            # Embedding is already batched, so we process each batch
            results = []
            for batch in batches:
                embed_result = self.create_embeddings(
                    model_name=model_name,
                    texts=batch,
                    optimization_level=optimization_level,
                )

                # Transform to individual results
                if embed_result["error"] is None:
                    for i, text in enumerate(batch):
                        results.append(
                            {
                                "text": text,
                                "embedding": embed_result["embeddings"][i],
                                "error": None,
                            }
                        )
                else:
                    # Add error for each text in batch
                    for text in batch:
                        results.append(
                            {
                                "text": text,
                                "embedding": [],
                                "error": embed_result["error"],
                            }
                        )

            return results

        else:
            logger.error(f"Unsupported task: {task}")
            return [{"error": f"Unsupported task: {task}"} for _ in texts]

    def cleanup(self):
        """Clean up resources."""
        # Clean up model manager
        self.model_manager.cleanup()

        # Clean up thread pool
        self.inference_pool.shutdown()

        # Clean up cache
        if self.result_cache:
            self.result_cache.cleanup()

        logger.info("Memory-efficient inference cleanup complete")


class ResultCache:
    """Cache for inference results."""

    def __init__(
        self, cache_dir: str, max_size_mb: int = 1000, ttl_seconds: int = 3600
    ):
        """
        Initialize result cache.

        Args:
            cache_dir: Directory for cache files
            max_size_mb: Maximum cache size in MB
            ttl_seconds: Time-to-live for cache entries
        """
        self.cache_dir = cache_dir
        self.max_size_mb = max_size_mb
        self.ttl_seconds = ttl_seconds

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Cache metadata
        self.cache_size = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # Cache index
        self.cache_index = self._load_index()

        # Start cleanup thread
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

        logger.info(f"Initialized result cache at {cache_dir}")

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from disk."""
        index_path = os.path.join(self.cache_dir, "cache_index.json")

        if os.path.exists(index_path):
            try:
                with open(index_path, "r") as f:
                    index = json.load(f)

                # Update cache size
                self.cache_size = sum(entry.get("size", 0) for entry in index.values())

                return index
            except Exception as e:
                logger.error(f"Error loading cache index: {str(e)}")

        return {}

    def _save_index(self):
        """Save cache index to disk."""
        index_path = os.path.join(self.cache_dir, "cache_index.json")

        try:
            with open(index_path, "w") as f:
                json.dump(self.cache_index, f)
        except Exception as e:
            logger.error(f"Error saving cache index: {str(e)}")

    def make_key(self, **kwargs) -> str:
        """Create a cache key from input parameters."""
        # Sort kwargs by key for consistent keys
        sorted_items = sorted(kwargs.items())

        # Create a string representation
        key_str = json.dumps(sorted_items, sort_keys=True)

        # Create a hash
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a cached result by key."""
        if key not in self.cache_index:
            self.cache_misses += 1
            return None

        entry = self.cache_index[key]

        # Check TTL
        if time.time() - entry["timestamp"] > self.ttl_seconds:
            # Expired
            self._remove_entry(key)
            self.cache_misses += 1
            return None

        # Load cached result
        try:
            cache_path = os.path.join(self.cache_dir, f"{key}.json")
            with open(cache_path, "r") as f:
                result = json.load(f)

            # Update access time
            entry["last_accessed"] = time.time()
            self.cache_index[key] = entry

            self.cache_hits += 1
            return result
        except Exception as e:
            logger.error(f"Error loading cached result: {str(e)}")
            self._remove_entry(key)
            self.cache_misses += 1
            return None

    def put(self, key: str, result: Dict[str, Any]):
        """Put a result in the cache."""
        # Serialize result
        result_str = json.dumps(result)
        size = len(result_str)

        # Check if it's too large
        if size > self.max_size_mb * 1024 * 1024 / 10:
            # Too large for caching (more than 10% of max size)
            logger.warning(f"Result too large for caching: {size} bytes")
            return

        # Ensure we have enough space
        self._ensure_cache_space(size)

        # Save result
        try:
            cache_path = os.path.join(self.cache_dir, f"{key}.json")
            with open(cache_path, "w") as f:
                f.write(result_str)

            # Update index
            self.cache_index[key] = {
                "timestamp": time.time(),
                "last_accessed": time.time(),
                "size": size,
            }

            # Update cache size
            self.cache_size += size

            # Save index
            self._save_index()
        except Exception as e:
            logger.error(f"Error caching result: {str(e)}")

    def _ensure_cache_space(self, size: int):
        """Ensure there's enough space for a new entry by removing old entries if needed."""
        max_size_bytes = self.max_size_mb * 1024 * 1024

        # Check if we need to make space
        if self.cache_size + size <= max_size_bytes:
            return

        # Get entries sorted by last accessed time (oldest first)
        entries = [(k, v) for k, v in self.cache_index.items()]
        entries.sort(key=lambda x: x[1]["last_accessed"])

        # Remove entries until we have enough space
        for key, entry in entries:
            self._remove_entry(key)

            if self.cache_size + size <= max_size_bytes:
                break

    def _remove_entry(self, key: str):
        """Remove a cache entry."""
        if key not in self.cache_index:
            return

        try:
            # Remove file
            cache_path = os.path.join(self.cache_dir, f"{key}.json")
            if os.path.exists(cache_path):
                os.remove(cache_path)

            # Update cache size
            self.cache_size -= self.cache_index[key].get("size", 0)

            # Remove from index
            del self.cache_index[key]
        except Exception as e:
            logger.error(f"Error removing cache entry: {str(e)}")

    def _cleanup_loop(self):
        """Background thread for cache cleanup."""
        while self.running:
            # Sleep for 10 minutes
            for _ in range(600):
                if not self.running:
                    break
                time.sleep(1)

            if not self.running:
                break

            # Clean up expired entries
            self._cleanup_expired()

    def _cleanup_expired(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = []

        # Find expired entries
        for key, entry in self.cache_index.items():
            if current_time - entry["timestamp"] > self.ttl_seconds:
                expired_keys.append(key)

        # Remove expired entries
        for key in expired_keys:
            self._remove_entry(key)

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
            self._save_index()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size_mb": self.cache_size / (1024 * 1024),
            "entry_count": len(self.cache_index),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_ratio": self.cache_hits / max(1, self.cache_hits + self.cache_misses),
        }

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=3)
        self._save_index()


def optimize_model(
    model_dir: str,
    output_dir: str,
    model_type: str = "causal_lm",
    sequence_lengths: str = "128,512,1024",
) -> Dict[str, Any]:
    """
    Optimize a transformer model for Dell PowerEdge R720 with E5-2640 processors.

    Args:
        model_dir: Directory containing the model
        output_dir: Directory to save optimized models
        model_type: Type of model ("causal_lm" or "seq2seq")
        sequence_lengths: Comma-separated list of sequence lengths to benchmark

    Returns:
        Optimization results with recommendations
    """
    logger.info(f"Optimizing model in {model_dir}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Path to optimization script
    script_path = os.path.join("/opt/sutazaiapp/scripts/optimize_transformer_models.sh")

    # Run optimization script
    try:
        subprocess.run(
            [
                script_path,
                "--model-dir",
                model_dir,
                "--output-dir",
                output_dir,
                "--model-type",
                model_type,
                "--seq-lengths",
                sequence_lengths,
            ],
            check=True,
        )

        # Load optimization results
        summary_path = os.path.join(output_dir, "optimization_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                results = json.load(f)

            # Get recommended model path
            recommendations = results.get("recommendations", {})
            e5_specific = recommendations.get("e5_2640_specific", {})

            # Use E5-2640 specific recommendation if available, otherwise use best inference
            primary_rec = e5_specific.get(
                "primary_recommendation",
                recommendations.get("best_inference_overall", "original"),
            )

            # Construct optimized model path
            if primary_rec != "original":
                optimized_model_path = os.path.join(output_dir, primary_rec)
                results["optimized_model_path"] = optimized_model_path
            else:
                results["optimized_model_path"] = model_dir

            logger.info(
                f"Model optimization complete. Recommended method: {primary_rec}"
            )
            return results
        else:
            logger.error(f"Optimization summary not found at {summary_path}")
            return {"error": "Optimization summary not found"}

    except subprocess.CalledProcessError as e:
        logger.error(f"Error optimizing model: {str(e)}")
        return {"error": f"Optimization failed: {str(e)}"}
    except Exception as e:
        logger.error(f"Error during model optimization: {str(e)}")
        return {"error": str(e)}
