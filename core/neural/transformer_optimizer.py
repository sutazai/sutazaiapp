#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transform_optimizer.py - Transformer optimization for Dell PowerEdge R720 with E5-2640 CPUs
"""

import os
import sys
import json
import logging
import platform
import tempfile
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Mapping

import torch
import psutil

try:
    from transformers import AutoConfig
    import transformers
except ImportError:
    logging.warning("transformers package not found. Some features will be disabled.")
import gc

# Import local modules
try:
    from core.neural.benchmark_utils import (
        benchmark_original,
        benchmark_fp16,
        benchmark_int8,
        benchmark_onnx,
        benchmark_bettertransformer,
        benchmark_lookupffn,
    )
    from core.neural.llama_utils import CPUOptimizedLlama
except ImportError:
    # Relative import for direct testing
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    try:
        from core.neural.benchmark_utils import (
            benchmark_original,
            benchmark_fp16,
            benchmark_int8,
            benchmark_onnx,
            benchmark_bettertransformer,
            benchmark_lookupffn,
        )
        from core.neural.llama_utils import CPUOptimizedLlama
    except ImportError:
        logging.warning("Local modules not found. Some features will be disabled.")

logger = logging.getLogger("sutazai.optimizer")

# Define a type alias for benchmark results, allowing for error strings
BenchmarkResultType = Optional[Dict[str, Union[float, int, str]]]


class TransformerOptimizer:
    """
    Optimizes transformer models for inference on Dell PowerEdge R720 with E5-2640 CPUs.

    This class applies a series of optimizations to transformer models to improve
    inference performance on CPU, specifically targeting the E5-2640 processors.
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the transformer optimizer.

        Args:
            output_dir: Directory to save benchmarks and optimized models
        """
        self.output_dir = (
            output_dir or Path(tempfile.gettempdir()) / "sutazai_optimized"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Configure logging
        self.log_file = os.path.join(self.output_dir, "optimization.log")
        self._setup_logging()

        # Configure CPU settings for E5-2640
        self._configure_cpu_settings()
        self._log_system_info()

    def _setup_logging(self):
        """Set up logging for the optimizer."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )

    def _set_power_profile(self):
        """
        Set power management recommendations for E5-2640 CPUs.

        Note: This doesn't directly modify system settings but provides recommendations.
        """
        logger.info(
            "Power management recommendations for Dell PowerEdge R720 with E5-2640:"
        )
        logger.info("1. Set BIOS power profile to 'Performance Per Watt (DAPC)'")
        logger.info("2. Disable C-states beyond C1E to reduce latency")
        logger.info("3. Set CPU power management to 'Maximum Performance'")
        logger.info(
            "4. Expected power consumption: ~95W per processor (vs ~120W at full load)"
        )
        logger.info(
            "5. Consider setting 'Fan Speed Offset' to 'Low Power' for quieter operation"
        )

    def _configure_cpu_settings(self):
        """Configure environment variables and libraries for optimal E5-2640 performance."""
        # Set number of threads to match physical cores
        # E5-2640 has 6 cores per socket, 12 total with hyper-threading
        n_threads = 12

        # Set thread count for various libraries
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)

        # Set Intel MKL settings
        # E5-2640 supports AVX but not AVX2
        os.environ["MKL_ENABLE_INSTRUCTIONS"] = "AVX"
        os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
        os.environ["KMP_BLOCKTIME"] = "0"  # Reduce thread pool overhead

        # Set PyTorch threads
        if torch.cuda.is_available():
            logger.warning("CUDA detected but this optimizer is designed for CPU only")
        torch.set_num_threads(n_threads)

        logger.info(f"CPU settings configured for E5-2640: {n_threads} threads")

    def _log_system_info(self):
        """Log system information for diagnostics."""
        # System information
        uname = platform.uname()
        logger.info(f"System: {uname.system} {uname.release}")
        logger.info(f"Machine: {uname.machine}")
        logger.info(f"Processor: {uname.processor}")

        # CPU information
        cpu_info = {}
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_info["model"] = line.strip().split(": ")[1]
                        break
        except Exception:
            cpu_info["model"] = "Unknown"

        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)

        logger.info(f"CPU Model: {cpu_info.get('model', 'Unknown')}")
        logger.info(f"Physical cores: {physical_cores}")
        logger.info(f"Logical cores: {logical_cores}")

        # Memory information
        mem = psutil.virtual_memory()
        logger.info(f"Total memory: {mem.total / (1024**3):.2f} GB")
        logger.info(f"Available memory: {mem.available / (1024**3):.2f} GB")

        # Python and library versions
        logger.info(f"Python version: {platform.python_version()}")
        logger.info(f"PyTorch version: {torch.__version__}")
        try:
            logger.info(f"Transformers version: {transformers.__version__}")
        except NameError:
            logger.info("Transformers package not available")

    def optimize_model(
        self,
        model_path: str,
        optimizations: Optional[List[str]] = None,
        benchmark: bool = True,
    ) -> Dict[str, Any]:
        """
        Apply selected optimizations to a transformer model and benchmark the results.

        Args:
            model_path: Path to the model (local path or Hugging Face model ID)
            optimizations: List of optimizations to apply.
                          Options: ["int8", "fp16", "bettertransformer", "lookupffn", "onnx"]
            benchmark: Whether to benchmark the model before and after optimization

        Returns:
            Dictionary with optimization results and benchmarks
        """
        if optimizations is None:
            optimizations = ["int8", "bettertransformer"]

        logger.info(f"Optimizing model: {model_path}")
        logger.info(f"Selected optimizations: {optimizations}")

        results = {"model": model_path, "optimizations": {}}
        benchmark_results = {"original": None}

        # Check if model is a Llama model (based on name or architecture)
        is_llama = self._check_if_llama_model(model_path)

        if is_llama:
            logger.info(f"Detected Llama model: {model_path}")
            return self.optimize_llama_model(model_path, optimizations, benchmark)

        # Benchmark original model
        if benchmark:
            logger.info("Benchmarking original model...")
            benchmark_results["original"] = benchmark_original(model_path)
            results["benchmarks"]: Dict[str, BenchmarkResultType] = benchmark_results

        # Apply optimizations
        optimized_model_paths = {}

        try:
            # INT8 Quantization
            if "int8" in optimizations:
                logger.info("Applying INT8 quantization...")
                int8_path = self._apply_int8_quantization(model_path)
                optimized_model_paths["int8"] = int8_path

                if benchmark:
                    benchmark_results["int8"] = benchmark_int8(model_path, output_path=int8_path)[0]

            # FP16 Quantization
            if "fp16" in optimizations:
                logger.info("Applying FP16 precision...")
                fp16_path = self._apply_fp16_quantization(model_path)
                optimized_model_paths["fp16"] = fp16_path

                if benchmark:
                    benchmark_results["fp16"] = benchmark_fp16(model_path, output_path=fp16_path)[0]

            # BetterTransformer
            if "bettertransformer" in optimizations:
                logger.info("Applying BetterTransformer optimization...")
                bt_path = self._apply_bettertransformer(model_path)
                optimized_model_paths["bettertransformer"] = bt_path

                if benchmark:
                    benchmark_results["bettertransformer"] = (
                        benchmark_bettertransformer(model_path, output_path=bt_path)[0]
                    )

            # LookupFFN
            if "lookupffn" in optimizations:
                logger.info("Applying LookupFFN optimization...")
                lookupffn_path = self._apply_lookupffn(model_path)
                optimized_model_paths["lookupffn"] = lookupffn_path

                if benchmark:
                    benchmark_results["lookupffn"] = benchmark_lookupffn(model_path, output_path=lookupffn_path)[0]

            # ONNX conversion
            if "onnx" in optimizations:
                logger.info("Converting to ONNX format...")
                onnx_path = self._convert_to_onnx(model_path)
                optimized_model_paths["onnx"] = onnx_path

                if benchmark:
                    benchmark_results["onnx"] = benchmark_onnx(model_path, output_path=onnx_path)[0]

        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            results["error"] = str(e)

        # Save benchmark results
        if benchmark:
            results["benchmarks"] = benchmark_results
            benchmark_path = os.path.join(self.output_dir, "benchmark_results.json")
            with open(benchmark_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Benchmark results saved to {benchmark_path}")

        # Find best optimization based on benchmarks
        if benchmark and all(x is not None for x in benchmark_results.values()):
            best_opt = self._find_best_optimization(benchmark_results)
            results["recommended"] = best_opt
            logger.info(f"Recommended optimization: {best_opt}")

        results["optimized_models"] = optimized_model_paths
        return results

    def optimize_llama_model(
        self,
        model_path: str,
        optimizations: Optional[List[str]] = None,
        benchmark: bool = True,
    ) -> Dict[str, Any]:
        """
        Optimize a Llama3 model for inference on E5-2640 CPUs.

        This method applies optimizations specifically designed for Llama3 models,
        which require different approaches compared to standard transformer models.

        Args:
            model_path: Path to the Llama model (GGUF format preferred)
            optimizations: List of optimizations to apply
            benchmark: Whether to benchmark the model before and after optimization

        Returns:
            Dictionary with optimization results and benchmarks
        """
        if optimizations is None:
            optimizations = ["q4_0", "cpu_threads"]

        logger.info(f"Optimizing Llama model: {model_path}")
        logger.info(f"Selected optimizations for Llama: {optimizations}")

        results = {"model": model_path, "type": "llama", "optimizations": {}}
        benchmark_results = {}

        # Benchmark original model if requested
        if benchmark:
            logger.info("Benchmarking original Llama model...")
            start_time = time.time()

            try:
                # Initialize with default settings
                model = CPUOptimizedLlama(
                    model_path=model_path,
                    n_threads=1,  # Use single thread for baseline
                    n_ctx=1024,  # Smaller context for benchmarking
                )

                # Run a simple generation test
                test_prompt = "Once upon a time"
                model.generate(prompt=test_prompt, max_tokens=50, temperature=0.7)

                latency = time.time() - start_time
                memory_used = model._get_memory_usage()

                benchmark_results["original"] = {
                    "time_seconds": latency,
                    "memory_mb": memory_used,
                    "tokens_per_second": 50 / latency if latency > 0 else 0.0,
                }

                del model
                gc.collect()

            except Exception as e:
                logger.error(f"Error benchmarking original Llama model: {str(e)}")
                benchmark_results["original"] = {"error": str(e)}

        # Apply Llama-specific optimizations
        optimized_configs = {}

        try:
            # Thread count optimization
            if "cpu_threads" in optimizations:
                thread_configs = [1, 4, 6, 8, 12]  # Test different thread counts
                best_threads = 12  # Default to physical core count
                best_tokens_per_second = 0.0

                for n_threads in thread_configs:
                    logger.info(f"Testing with {n_threads} threads...")

                    model = CPUOptimizedLlama(
                        model_path=model_path,
                        n_threads=n_threads,
                        n_ctx=1024,
                    )

                    start_time = time.time()
                    test_prompt = "Once upon a time"
                    model.generate(prompt=test_prompt, max_tokens=50, temperature=0.7)
                    latency = time.time() - start_time

                    tokens_per_second = 50 / latency if latency > 0 else 0

                    logger.info(
                        f"Threads: {n_threads}, Tokens/sec: {tokens_per_second:.2f}"
                    )

                    if tokens_per_second > best_tokens_per_second:
                        best_tokens_per_second = tokens_per_second
                        best_threads = n_threads

                    del model
                    gc.collect()

                logger.info(
                    f"Best thread count: {best_threads} with {best_tokens_per_second:.2f} tokens/sec"
                )
                optimized_configs["cpu_threads"] = best_threads
                benchmark_results["cpu_threads"] = {
                    "tokens_per_second": best_tokens_per_second,
                    "thread_count": best_threads,
                }

            # Context size optimization
            if "context_size" in optimizations:
                # Find optimal context size based on available memory
                mem = psutil.virtual_memory()
                available_gb = mem.available / (1024**3)

                # Heuristic for Llama3 models based on available memory
                if "70b" in model_path.lower():
                    if available_gb > 50:
                        optimal_ctx = 4096
                    elif available_gb > 30:
                        optimal_ctx = 2048
                    else:
                        optimal_ctx = 1024
                else:  # Smaller models
                    if available_gb > 16:
                        optimal_ctx = 8192
                    elif available_gb > 8:
                        optimal_ctx = 4096
                    else:
                        optimal_ctx = 2048

                logger.info(
                    f"Optimal context size for available memory ({available_gb:.2f} GB): {optimal_ctx}"
                )
                optimized_configs["context_size"] = optimal_ctx
                benchmark_results["context_size"] = {
                    "context_length": optimal_ctx,
                    "available_memory_gb": available_gb,
                }

            # Batch size optimization
            if "batch_size" in optimizations:
                # Find optimal batch size
                batch_sizes = [32, 64, 128, 256, 512]
                best_batch_size = 512  # Default
                best_tokens_per_second = 0.0

                # Only test a couple of batch sizes to save time
                for batch_size in batch_sizes[:3]:  # Test the first 3 sizes
                    logger.info(f"Testing batch size: {batch_size}...")

                    model = CPUOptimizedLlama(
                        model_path=model_path,
                        n_threads=optimized_configs.get("cpu_threads", 12),
                        n_batch=batch_size,
                        n_ctx=1024,
                    )

                    start_time = time.time()
                    test_prompt = "Once upon a time"
                    model.generate(prompt=test_prompt, max_tokens=50, temperature=0.7)
                    latency = time.time() - start_time

                    tokens_per_second = 50.0 / latency if latency > 0 else 0.0

                    logger.info(
                        f"Batch size: {batch_size}, Tokens/sec: {tokens_per_second:.2f}"
                    )

                    if tokens_per_second > best_tokens_per_second:
                        best_tokens_per_second = tokens_per_second
                        best_batch_size = batch_size

                    del model
                    gc.collect()

                logger.info(
                    f"Best batch size: {best_batch_size} with {best_tokens_per_second:.2f} tokens/sec"
                )
                optimized_configs["batch_size"] = best_batch_size
                benchmark_results["batch_size"] = {
                    "tokens_per_second": best_tokens_per_second,
                    "batch_size": best_batch_size,
                    "time_seconds": latency,
                    "memory_mb": memory_used,
                }

        except Exception as e:
            logger.error(f"Error optimizing Llama model: {str(e)}")
            results["error"] = str(e)

        # Create final optimized model with best settings
        try:
            # Create configuration file for the optimized model
            optimized_config = {
                "model_path": model_path,
                "n_threads": optimized_configs.get("cpu_threads", 12),
                "n_batch": optimized_configs.get("batch_size", 512),
                "n_ctx": optimized_configs.get("context_size", 4096),
                "use_mlock": False,
                "use_mmap": True,
            }

            # Save configuration
            config_path = os.path.join(self.output_dir, "llama_optimized_config.json")
            with open(config_path, "w") as f:
                json.dump(optimized_config, f, indent=2)

            logger.info(f"Optimized Llama configuration saved to {config_path}")
            results["optimized_config_path"] = config_path
            results["optimized_config"] = optimized_config

        except Exception as e:
            logger.error(f"Error saving optimized configuration: {str(e)}")

        # Save benchmark results
        if benchmark and benchmark_results:
            results["benchmarks"] = benchmark_results
            benchmark_path = os.path.join(
                self.output_dir, "llama_benchmark_results.json"
            )
            with open(benchmark_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Benchmark results saved to {benchmark_path}")

        return results

    def _check_if_llama_model(self, model_path: str) -> bool:
        """Check if the model is a Llama model based on name or metadata."""
        # Check by name
        model_name_lower = model_path.lower()
        llama_indicators = ["llama", "meta-llama", ".gguf"]

        for indicator in llama_indicators:
            if indicator in model_name_lower:
                return True

        # If local path exists, try to check metadata
        if os.path.exists(model_path):
            if model_path.endswith(".gguf"):
                return True

            # For Hugging Face models, check config
            try:
                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model_type = getattr(config, "model_type", "").lower()

                if "llama" in model_type:
                    return True
            except Exception:
                pass

        return False

    def _apply_int8_quantization(self, model_path: str) -> str:
        """Apply INT8 quantization to the model."""
        output_path = os.path.join(self.output_dir, "model_int8")

        try:
            from transformers import AutoModelForCausalLM

            # Load the model with INT8 quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_path, load_in_8bit=True, device_map="auto", trust_remote_code=True
            )

            # Save the model
            model.save_pretrained(output_path)

            # Clean up
            del model
            gc.collect()
            torch.cuda.empty_cache()

            logger.info(f"INT8 quantized model saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error applying INT8 quantization: {str(e)}")
            raise

    def _apply_fp16_quantization(self, model_path: str) -> str:
        """Apply FP16 precision to the model."""
        output_path = os.path.join(self.output_dir, "model_fp16")

        try:
            from transformers import AutoModelForCausalLM

            # Load the model with FP16 precision
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

            # Save the model
            model.save_pretrained(output_path)

            # Clean up
            del model
            gc.collect()
            torch.cuda.empty_cache()

            logger.info(f"FP16 model saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error applying FP16 precision: {str(e)}")
            raise

    def _apply_bettertransformer(self, model_path: str) -> str:
        """Apply BetterTransformer optimization to the model."""
        output_path = os.path.join(self.output_dir, "model_bettertransformer")

        try:
            from transformers import AutoModelForCausalLM
            from optimum.bettertransformer import BetterTransformer

            # Load the model
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto", trust_remote_code=True
            )

            # Apply BetterTransformer
            model = BetterTransformer.transform(model)

            # Save the model
            model.save_pretrained(output_path)

            # Clean up
            del model
            gc.collect()
            torch.cuda.empty_cache()

            logger.info(f"BetterTransformer model saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error applying BetterTransformer: {str(e)}")
            raise

    def _apply_lookupffn(self, model_path: str) -> str:
        """Apply LookupFFN optimization to the model."""
        output_path = os.path.join(self.output_dir, "model_lookupffn")

        try:
            from transformers import AutoModelForCausalLM

            # Load the model
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto", trust_remote_code=True
            )

            # Apply LookupFFN optimization
            # This is a placeholder - actual implementation would modify
            # the feed-forward layers to use the LookupFFN class
            logger.info("LookupFFN optimization not fully implemented")

            # Save the model
            model.save_pretrained(output_path)

            # Clean up
            del model
            gc.collect()
            torch.cuda.empty_cache()

            logger.info(f"LookupFFN model saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error applying LookupFFN: {str(e)}")
            raise

    def _convert_to_onnx(self, model_path: str) -> str:
        """Convert the model to ONNX format."""
        output_path = os.path.join(self.output_dir, "model_onnx")

        try:
            from optimum.onnxruntime import ORTModelForCausalLM

            # Export the model to ONNX
            ort_model = ORTModelForCausalLM.from_pretrained(
                model_path, from_transformers=True, provider="CPUExecutionProvider"
            )

            # Save the model
            ort_model.save_pretrained(output_path)

            # Clean up
            del ort_model
            gc.collect()

            logger.info(f"ONNX model saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error converting to ONNX: {str(e)}")
            raise

    def _find_best_optimization(
        self, benchmark_results: Mapping[str, BenchmarkResultType]
    ) -> str:
        """Find the best optimization based on benchmark results."""
        best_opt = "original"
        best_throughput = 0.0

        for opt, results in benchmark_results.items():
            if results is None:
                continue

            # Skip if there was an error
            if "error" in results:
                continue

            throughput = results.get("tokens_per_second", 0)
            # Check if throughput is numeric before comparison
            if isinstance(throughput, (int, float)):
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_opt = opt

        return best_opt


# Command-line interface
def main():
    parser = argparse.ArgumentParser(
        description="Optimize transformer models for Dell PowerEdge R720 with E5-2640 CPUs"
    )
    parser.add_argument("--model", required=True, help="Path to the model or model ID")
    parser.add_argument(
        "--output", default=None, help="Output directory for optimized models"
    )
    parser.add_argument(
        "--model_type",
        choices=["transformer", "llama"],
        default="transformer",
        help="Type of model to optimize",
    )
    parser.add_argument(
        "--threads", type=int, default=12, help="Number of threads to use"
    )
    parser.add_argument(
        "--optimizations",
        nargs="+",
        default=["int8", "bettertransformer"],
        help="Optimizations to apply",
    )
    parser.add_argument(
        "--seq_length", type=int, default=512, help="Sequence length for benchmarking"
    )
    parser.add_argument("--log_file", default=None, help="Path to log file")
    parser.add_argument("--no_benchmark", action="store_true", help="Skip benchmarking")

    args = parser.parse_args()

    # Set thread count from arguments
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    # Create optimizer
    optimizer = TransformerOptimizer(output_dir=args.output)

    # Run optimization
    if args.model_type == "llama":
        result = optimizer.optimize_llama_model(
            model_path=args.model,
            optimizations=args.optimizations,
            benchmark=not args.no_benchmark,
        )
    else:
        result = optimizer.optimize_model(
            model_path=args.model,
            optimizations=args.optimizations,
            benchmark=not args.no_benchmark,
        )

    # Print summary
    print("\n===== Optimization Results =====")
    print(f"Model: {args.model}")
    print(f"Type: {args.model_type}")

    if "recommended" in result:
        print(f"Recommended optimization: {result['recommended']}")

    if "optimized_config_path" in result:
        print(f"Optimized configuration: {result['optimized_config_path']}")

    if "optimized_models" in result:
        print("\nOptimized model paths:")
        for opt, path in result["optimized_models"].items():
            print(f"  - {opt}: {path}")


if __name__ == "__main__":
    main()
