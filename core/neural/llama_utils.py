#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llama_utils.py - Utilities for working with Llama3 70B models on CPU

This module provides functions and classes to efficiently run Llama3 models
on CPU hardware, specifically optimized for the Dell PowerEdge R720 with E5-2640 CPUs.
It includes utilities for model loading, inference optimization, and memory management.
"""

import os
import logging
import time
import json
from typing import Dict, Any, Optional

import torch

try:
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logging.warning("llama-cpp-python not installed, some features will be unavailable")

try:
    from core.neural.model_downloader import (
        # F401: Removed unused import EnterpriseModelDownloader
        # EnterpriseModelDownloader,
        ensure_model_downloaded,
        get_optimal_model_for_e5_2640,
    )

    MODEL_DOWNLOADER_AVAILABLE = True
except ImportError:
    MODEL_DOWNLOADER_AVAILABLE = False
    logging.warning(
        "model_downloader module not found, auto-download features will be unavailable"
    )

logger = logging.getLogger("sutazai.llama")


class CPUOptimizedLlama:
    """
    A wrapper around llama-cpp-python that optimizes Llama3 70B model inference
    for CPU-only environments, specifically targeting the E5-2640 CPUs.

    This class implements:
    1. Efficient quantization loading
    2. Memory-managed inference
    3. Adaptive batch sizing based on available CPU cores
    4. Context management to avoid memory leaks
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_id: Optional[str] = None,
        config_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_batch: int = 512,
        n_gpu_layers: int = 0,
        n_threads: int = 12,  # Optimized for E5-2640 (6 cores per socket, 12 total)
        use_mlock: bool = False,
        use_mmap: bool = True,
        seed: int = 42,
        verbose: bool = False,
        auto_download: bool = True,
    ):
        """
        Initialize a CPU-optimized Llama model.

        Args:
            model_path: Path to the model file (.gguf format)
            model_id: ID of the model in the catalog (e.g., "llama3-70b")
            config_path: Path to a JSON configuration file (alternative to model_path)
            n_ctx: Context size
            n_batch: Batch size for prompt evaluation
            n_gpu_layers: Number of layers to offload to GPU (0 for CPU-only)
            n_threads: Number of threads to use (default: 12 for E5-2640)
            use_mlock: Lock model in memory
            use_mmap: Use memory mapping for model loading
            seed: Random seed
            verbose: Enable verbose logging
            auto_download: Whether to auto-download model if not found
        """
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.use_mlock = use_mlock
        self.use_mmap = use_mmap
        self.seed = seed
        self.verbose = verbose
        self._model = None
        self.model_path = None
        self.auto_download = auto_download
        self._model_initialized = False
        self._init_time = None

        # Set environment variables for Intel optimization
        self._set_intel_optimizations()

        # Load from config file if provided
        if config_path:
            self._load_from_config(config_path)
        # Use model_id to auto-download if provided
        elif model_id and auto_download and MODEL_DOWNLOADER_AVAILABLE:
            self._load_with_auto_download(model_id)
        # Otherwise use direct model path
        elif model_path:
            self.model_path = model_path
        # If nothing provided, try to use optimal model
        elif auto_download and MODEL_DOWNLOADER_AVAILABLE:
            model_id = get_optimal_model_for_e5_2640()
            logger.info(
                f"No model specified, using optimal model for E5-2640: {model_id}"
            )
            self._load_with_auto_download(model_id)
        else:
            raise ValueError(
                "Either model_path, model_id, or config_path must be provided"
            )

    def _set_intel_optimizations(self):
        """Set environment variables for optimal performance on Intel CPUs"""
        os.environ["OMP_NUM_THREADS"] = str(self.n_threads)
        os.environ["MKL_NUM_THREADS"] = str(self.n_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(self.n_threads)
        os.environ["MKL_ENABLE_INSTRUCTIONS"] = "AVX"  # E5-2640 has AVX but not AVX2
        os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
        os.environ["KMP_BLOCKTIME"] = "0"  # Reduce thread pool overhead

    def _load_from_config(self, config_path: str):
        """Load model configuration from a JSON file"""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)

            self.model_path = config.get("model_path")
            if not self.model_path:
                raise ValueError("Config file must contain 'model_path'")

            # Override init parameters with config values if provided
            self.n_ctx = config.get("n_ctx", self.n_ctx)
            self.n_batch = config.get("n_batch", self.n_batch)
            self.n_gpu_layers = config.get("n_gpu_layers", self.n_gpu_layers)
            self.n_threads = config.get("n_threads", self.n_threads)
            self.use_mlock = config.get("use_mlock", self.use_mlock)
            self.use_mmap = config.get("use_mmap", self.use_mmap)

            logger.info(f"Loaded configuration from {config_path}")

            # Re-apply Intel optimizations with potentially new thread count
            self._set_intel_optimizations()

        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            raise

    def _load_with_auto_download(self, model_id: str):
        """Auto-download the model if it doesn't exist"""
        if not MODEL_DOWNLOADER_AVAILABLE:
            raise ImportError("model_downloader module not available for auto-download")

        logger.info(f"Auto-downloading model: {model_id}")
        model_path = ensure_model_downloaded(model_id)

        if not model_path:
            raise FileNotFoundError(f"Failed to download model: {model_id}")

        self.model_path = model_path
        logger.info(f"Model downloaded to: {model_path}")

    def _initialize(self):
        """Initialize the model with optimized settings."""
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python is not installed")

        if self._model_initialized:
            return

        logger.info(f"Loading Llama model from {self.model_path}")
        start_time = time.time()

        # Check if model path exists
        if not os.path.exists(self.model_path):
            if self.auto_download and MODEL_DOWNLOADER_AVAILABLE:
                # Try to extract model ID from filename
                filename = os.path.basename(self.model_path)
                if "llama-3-70b" in filename.lower():
                    model_id = "llama3-70b"
                elif "llama-3-8b" in filename.lower():
                    model_id = "llama3-8b"
                elif "mistral-7b" in filename.lower():
                    model_id = "mistral-7b"
                else:
                    model_id = get_optimal_model_for_e5_2640()

                logger.info(f"Model file not found, attempting to download {model_id}")
                self._load_with_auto_download(model_id)
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            self._model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                use_mlock=self.use_mlock,
                use_mmap=self.use_mmap,
                seed=self.seed,
                verbose=self.verbose,
            )

            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds")
            self._model_initialized = True
            self._init_time = time.time()

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text from a prompt with optimized settings for CPU inference.

        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            repeat_penalty: Repetition penalty

        Returns:
            Dictionary containing generation results
        """
        if not self._model_initialized:
            self._initialize()
        # Ensure model is initialized after the call
        assert self._model is not None

        # Log memory usage before inference
        memory_before = self._get_memory_usage()

        # Load model with optimized settings
        try:
            start_time = time.time()

            result = self._model.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                **kwargs,
            )

            generation_time = time.time() - start_time
            tokens_per_second = (
                max_tokens / generation_time if generation_time > 0 else 0
            )

            # Log memory usage after inference
            memory_after = self._get_memory_usage()
            memory_used = memory_after - memory_before

            logger.info(
                f"Generated {max_tokens} tokens in {generation_time:.2f}s ({tokens_per_second:.2f} tokens/s)"
            )
            logger.info(f"Memory used: {memory_used:.2f} MB")

            return result
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

    def _get_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        try:
            import psutil

            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def save_config(self, config_path: str):
        """Save the current configuration to a JSON file"""
        config = {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_batch": self.n_batch,
            "n_gpu_layers": self.n_gpu_layers,
            "n_threads": self.n_threads,
            "use_mlock": self.use_mlock,
            "use_mmap": self.use_mmap,
            "seed": self.seed,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        try:
            os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False

    def __del__(self):
        """Cleanup resources when the object is deleted."""
        if hasattr(self, "_model") and self._model is not None:
            # Free model resources explicitly
            del self._model
            self._model = None


def get_optimized_model(
    model_id_or_path: Optional[str] = None,
    force_download: bool = False,
    config_path: Optional[str] = None,
    thread_count: int = 12,
) -> CPUOptimizedLlama:
    """
    Enterprise-level utility to get an optimized Llama model.

    This function handles all the complexities of model loading, including:
    1. Automatic downloading if needed
    2. Configuration for E5-2640 CPUs
    3. Optimal thread and memory settings

    Args:
        model_id_or_path: Model ID (e.g., "llama3-70b") or path to model file
        force_download: Whether to force a new download even if model exists
        config_path: Path to a JSON configuration file
        thread_count: Number of threads to use (default: 12 for E5-2640)

    Returns:
        Initialized CPUOptimizedLlama instance
    """
    if not LLAMA_CPP_AVAILABLE:
        raise ImportError("llama-cpp-python is not installed")

    model_path: Optional[str] = None

    # If config file provided, use it directly
    if config_path:
        return CPUOptimizedLlama(config_path=config_path, n_threads=thread_count)

    # If it looks like a path, use it directly
    if model_id_or_path and (
        os.path.exists(model_id_or_path)
        or model_id_or_path.endswith(".gguf")
        or "/" in model_id_or_path
        or "\\" in model_id_or_path
    ):
        model_path = model_id_or_path
        logger.info(f"Using model path: {model_path}")
        return CPUOptimizedLlama(model_path=model_path, n_threads=thread_count)

    # Otherwise, treat as a model ID and auto-download
    model_id = model_id_or_path

    # If no model ID provided, use the optimal model for E5-2640
    if not model_id and MODEL_DOWNLOADER_AVAILABLE:
        model_id = get_optimal_model_for_e5_2640()
        logger.info(f"No model specified, using optimal model for E5-2640: {model_id}")

    # Ensure model_id is set before attempting download
    if model_id is None:
        raise ValueError(
            "Could not determine a model ID to download or load. Please specify a model ID or path."
        )

    if MODEL_DOWNLOADER_AVAILABLE:
        if force_download:
            logger.info(f"Forcing download of model: {model_id}")
            model_path = ensure_model_downloaded(model_id, force=True)
        else:
            model_path = ensure_model_downloaded(model_id)

        if model_path:
            logger.info(f"Using downloaded model: {model_path}")
            return CPUOptimizedLlama(model_path=model_path, n_threads=thread_count)

    # If we get here, we need to directly initialize with model_id
    return CPUOptimizedLlama(
        model_id=model_id, n_threads=thread_count, auto_download=True
    )


def create_llama_prompt(system_prompt: str, user_message: str) -> str:
    """
    Create a properly formatted prompt for Llama models.

    Args:
        system_prompt: System instructions
        user_message: User query or message

    Returns:
        Formatted prompt string
    """
    return f"""<|system|>
{system_prompt}
<|user|>
{user_message}
<|assistant|>
"""


def optimize_for_e5_2640():
    """Configure environment for optimal performance on E5-2640 CPUs."""
    # Set thread count to match physical cores (E5-2640 has 6 cores per socket, 12 total)
    os.environ["OMP_NUM_THREADS"] = "12"
    os.environ["MKL_NUM_THREADS"] = "12"

    # E5-2640 supports AVX but not AVX2
    os.environ["MKL_ENABLE_INSTRUCTIONS"] = "AVX"

    # Intel-specific optimizations
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    os.environ["KMP_BLOCKTIME"] = "0"  # Reduce overhead of thread pool

    # PyTorch specific settings
    if torch.cuda.is_available():
        # Not expected on E5-2640 systems, but handle gracefully
        logger.warning("CUDA detected but E5-2640 optimization is CPU-focused")
    else:
        # Set PyTorch thread count
        torch.set_num_threads(12)

    logger.info("Environment configured for Dell PowerEdge R720 with E5-2640 CPUs")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    import argparse

    parser = argparse.ArgumentParser(description="Llama3 Model Utilities")
    parser.add_argument("--model", type=str, help="Model ID or path")
    parser.add_argument("--prompt", type=str, help="Prompt to generate from")
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Generation temperature"
    )
    parser.add_argument(
        "--download", action="store_true", help="Force download the model"
    )
    args = parser.parse_args()

    if args.model and args.prompt:
        model = get_optimized_model(args.model, force_download=args.download)
        result = model.generate(
            args.prompt, max_tokens=args.max_tokens, temperature=args.temperature
        )
        print(result["choices"][0]["text"])
    elif args.model:
        print(f"Loading model: {args.model}")
        model = get_optimized_model(args.model, force_download=args.download)
        print("Model loaded successfully. Use --prompt to generate text.")
    else:
        print("Please specify a model with --model")
        if MODEL_DOWNLOADER_AVAILABLE:
            print("\nAvailable models (Ollama):")
            from core.neural.model_downloader import OLLAMA_MODELS

            for model_id, info in OLLAMA_MODELS.items():
                print(f"  {model_id}: {info['model_name']} ({info['size_gb']:.1f} GB)")
