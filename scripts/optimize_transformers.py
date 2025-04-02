#!/usr/bin/env python
# /opt/sutazaiapp/scripts/optimize_transformers.py

import os
import sys
import time
import json
import argparse
import logging
import numpy as np
from typing import Dict, List, Any, Tuple
import torch
import psutil
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("transformer_optimization.log"),
    ],
)

logger = logging.getLogger("transformer_optimization")

try:
    # Removed unused import: onnxruntime
    # import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    logger.warning("ONNX Runtime not found. ONNX optimization disabled.")
    ONNX_AVAILABLE = False

try:
    from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForSeq2SeqLM

    OPTIMUM_AVAILABLE = True
except ImportError:
    logger.warning(
        "Optimum library not available. Optimum optimizations will be skipped."
    )
    OPTIMUM_AVAILABLE = False

try:
    import transformers
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.error("Transformers library not available. Please install it.")
    TRANSFORMERS_AVAILABLE = False
    sys.exit(1)


class SystemInfo:
    """Get information about the system hardware."""

    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        """Get CPU information."""
        import platform

        cpu_info = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "architecture": platform.machine(),
            "processor": platform.processor(),
        }

        # Try to get AVX2 support information
        try:
            import cpuinfo

            cpu_features = cpuinfo.get_cpu_info().get("flags", [])
            cpu_info["avx2_support"] = "avx2" in cpu_features
            cpu_info["avx512_support"] = any(
                flag.startswith("avx512") for flag in cpu_features
            )
            cpu_info["vendor"] = cpuinfo.get_cpu_info().get("vendor_id", "unknown")
        except ImportError:
            logger.warning("py-cpuinfo not available. AVX support detection disabled.")
            cpu_info["avx2_support"] = None
            cpu_info["avx512_support"] = None

        return cpu_info

    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """Get memory information."""
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "used_percent": mem.percent,
        }

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get complete system information."""
        return {
            "cpu": SystemInfo.get_cpu_info(),
            "memory": SystemInfo.get_memory_info(),
            "platform": sys.platform,
        }


class TransformerOptimizer:
    """Optimize transformer models for CPU inference."""

    def __init__(self, model_dir: str, output_dir: str, device: str = "cpu"):
        """
        Initialize the optimizer.

        Args:
            model_dir: Directory containing the model
            output_dir: Directory to save optimized models
            device: Device to use ("cpu" or "cuda")
        """
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.device = device

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # System information
        self.system_info = SystemInfo.get_system_info()
        logger.info(f"System info: {json.dumps(self.system_info, indent=2)}")

        # Dell PowerEdge R720 with E5-2640 optimization flags
        self.avx2_available = self.system_info["cpu"].get("avx2_support", False)
        self.core_count = (
            self.system_info["cpu"]["physical_cores"] or 12
        )  # Default to 12 cores for R720

        # Flag for E5-2640-specific optimizations
        self.is_e5_2640 = "E5-2640" in self.system_info["cpu"].get("processor", "")

        # Set thread count for optimal performance
        if torch.get_num_threads() != self.core_count:
            logger.info(f"Setting PyTorch thread count to {self.core_count}")
            torch.set_num_threads(self.core_count)

    def load_model(self, model_type: str = "causal_lm") -> Tuple[Any, Any]:
        """
        Load a model for optimization.

        Args:
            model_type: Model type to load ("causal_lm" or "seq2seq")

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model from {self.model_dir}")

        # Load tokenizer first (lightweight)
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        # Load model based on type
        if model_type == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(
                self.model_dir, device_map=self.device, torch_dtype=torch.float16
            )
        elif model_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_dir, device_map=self.device, torch_dtype=torch.float16
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Set to evaluation mode
        model.eval()

        return model, tokenizer

    def benchmark_original(
        self, model: Any, tokenizer: Any, sequence_length: int = 128
    ) -> Dict[str, Any]:
        """
        Benchmark the original model.

        Args:
            model: Model to benchmark
            tokenizer: Tokenizer for the model
            sequence_length: Input sequence length

        Returns:
            Benchmark results
        """
        logger.info(
            f"Benchmarking original model with sequence length {sequence_length}"
        )

        # Prepare benchmark data
        input_text = "This is a benchmark input " * (sequence_length // 5 + 1)
        input_text = input_text[: sequence_length * 4]  # Rough character count

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids

        # Ensure input is the right length
        if input_ids.shape[1] > sequence_length:
            input_ids = input_ids[:, :sequence_length]

        # Warm up
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids)

        # Memory before
        gc.collect()
        memory_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

        # Benchmark inference time
        times = []
        with torch.no_grad():
            for _ in range(10):
                start_time = time.time()
                _ = model(input_ids)
                end_time = time.time()
                times.append(end_time - start_time)

        # Memory after
        memory_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

        # Generation benchmark (faster for comparison)
        gen_times = []
        with torch.no_grad():
            for _ in range(3):
                start_time = time.time()
                _ = model.generate(input_ids, max_new_tokens=20, do_sample=False)
                end_time = time.time()
                gen_times.append(end_time - start_time)

        # Calculate results
        avg_time = np.mean(times)
        avg_gen_time = np.mean(gen_times)

        results = {
            "model": "original",
            "sequence_length": sequence_length,
            "avg_inference_time": avg_time,
            "avg_generation_time": avg_gen_time,
            "memory_usage_mb": memory_after - memory_before,
            "tokens_per_second": sequence_length / avg_time,
            "generation_tokens_per_second": 20 / avg_gen_time,
        }

        logger.info(
            f"Original model benchmark results: {json.dumps(results, indent=2)}"
        )

        return results

    def optimize_with_onnx(
        self, model: Any, tokenizer: Any, sequence_length: int = 128
    ) -> Dict[str, Any]:
        """
        Optimize model using ONNX Runtime.

        Args:
            model: Model to optimize
            tokenizer: Tokenizer for the model
            sequence_length: Input sequence length

        Returns:
            Benchmark results for optimized model
        """
        if not ONNX_AVAILABLE:
            return {"error": "ONNX Runtime not available"}

        if not OPTIMUM_AVAILABLE:
            return {"error": "Optimum library not available"}

        logger.info("Optimizing model with ONNX Runtime")

        # Determine if model is causal LM or seq2seq
        is_causal_lm = isinstance(model, transformers.PreTrainedModel) and hasattr(
            model, "generate"
        )

        # Create ONNX output path
        onnx_path = os.path.join(self.output_dir, "onnx")
        os.makedirs(onnx_path, exist_ok=True)

        try:
            # Export/load ONNX model
            if is_causal_lm:
                onnx_model = ORTModelForCausalLM.from_pretrained(
                    self.model_dir, export=True, provider="CPUExecutionProvider"
                )
            else:
                onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
                    self.model_dir, export=True, provider="CPUExecutionProvider"
                )

            # Save optimized model
            onnx_model.save_pretrained(onnx_path)
            tokenizer.save_pretrained(onnx_path)

            # Benchmark data
            input_text = "This is a benchmark input " * (sequence_length // 5 + 1)
            input_text = input_text[: sequence_length * 4]

            input_ids = tokenizer(input_text, return_tensors="pt").input_ids

            # Ensure input is the right length
            if input_ids.shape[1] > sequence_length:
                input_ids = input_ids[:, :sequence_length]

            # Warm up
            for _ in range(3):
                _ = onnx_model(input_ids)

            # Memory before
            gc.collect()
            memory_before = psutil.Process(os.getpid()).memory_info().rss / (
                1024 * 1024
            )

            # Benchmark inference time
            times = []
            for _ in range(10):
                start_time = time.time()
                _ = onnx_model(input_ids)
                end_time = time.time()
                times.append(end_time - start_time)

            # Memory after
            memory_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

            # Generation benchmark (faster for comparison)
            gen_times = []
            for _ in range(3):
                start_time = time.time()
                _ = onnx_model.generate(input_ids, max_new_tokens=20, do_sample=False)
                end_time = time.time()
                gen_times.append(end_time - start_time)

            # Calculate results
            avg_time = np.mean(times)
            avg_gen_time = np.mean(gen_times)

            results = {
                "model": "onnx",
                "sequence_length": sequence_length,
                "avg_inference_time": avg_time,
                "avg_generation_time": avg_gen_time,
                "memory_usage_mb": memory_after - memory_before,
                "tokens_per_second": sequence_length / avg_time,
                "generation_tokens_per_second": 20 / avg_gen_time,
            }

            logger.info(
                f"ONNX model benchmark results: {json.dumps(results, indent=2)}"
            )

            return results

        except Exception as e:
            logger.error(f"Error optimizing with ONNX: {str(e)}")
            return {"error": str(e)}

    def optimize_with_bettertransformer(
        self, model: Any, tokenizer: Any, sequence_length: int = 128
    ) -> Dict[str, Any]:
        """
        Optimize model using BetterTransformer.

        Args:
            model: Model to optimize
            tokenizer: Tokenizer for the model
            sequence_length: Input sequence length

        Returns:
            Benchmark results for optimized model
        """
        logger.info("Optimizing model with BetterTransformer")

        try:
            # Convert to BetterTransformer
            bt_model = model.to_bettertransformer()

            # Create BetterTransformer output path
            bt_path = os.path.join(self.output_dir, "bettertransformer")
            os.makedirs(bt_path, exist_ok=True)

            # Save optimized model
            bt_model.save_pretrained(bt_path)
            tokenizer.save_pretrained(bt_path)

            # Benchmark data
            input_text = "This is a benchmark input " * (sequence_length // 5 + 1)
            input_text = input_text[: sequence_length * 4]

            input_ids = tokenizer(input_text, return_tensors="pt").input_ids

            # Ensure input is the right length
            if input_ids.shape[1] > sequence_length:
                input_ids = input_ids[:, :sequence_length]

            # Warm up
            with torch.no_grad():
                for _ in range(3):
                    _ = bt_model(input_ids)

            # Memory before
            gc.collect()
            memory_before = psutil.Process(os.getpid()).memory_info().rss / (
                1024 * 1024
            )

            # Benchmark inference time
            times = []
            with torch.no_grad():
                for _ in range(10):
                    start_time = time.time()
                    _ = bt_model(input_ids)
                    end_time = time.time()
                    times.append(end_time - start_time)

            # Memory after
            memory_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

            # Generation benchmark (faster for comparison)
            gen_times = []
            with torch.no_grad():
                for _ in range(3):
                    start_time = time.time()
                    _ = bt_model.generate(input_ids, max_new_tokens=20, do_sample=False)
                    end_time = time.time()
                    gen_times.append(end_time - start_time)

            # Calculate results
            avg_time = np.mean(times)
            avg_gen_time = np.mean(gen_times)

            results = {
                "model": "bettertransformer",
                "sequence_length": sequence_length,
                "avg_inference_time": avg_time,
                "avg_generation_time": avg_gen_time,
                "memory_usage_mb": memory_after - memory_before,
                "tokens_per_second": sequence_length / avg_time,
                "generation_tokens_per_second": 20 / avg_gen_time,
            }

            logger.info(
                f"BetterTransformer model benchmark results: {json.dumps(results, indent=2)}"
            )

            return results

        except Exception as e:
            logger.error(f"Error optimizing with BetterTransformer: {str(e)}")
            return {"error": str(e)}

    def optimize_with_quantization(
        self, model: Any, tokenizer: Any, sequence_length: int = 128, bits: int = 8
    ) -> Dict[str, Any]:
        """
        Optimize model with quantization.

        Args:
            model: Model to optimize
            tokenizer: Tokenizer for the model
            sequence_length: Input sequence length
            bits: Quantization bits (8 or 4)

        Returns:
            Benchmark results for optimized model
        """
        logger.info(f"Optimizing model with {bits}-bit quantization")

        try:
            # Create quantized output path
            quant_path = os.path.join(self.output_dir, f"quantized_{bits}bit")
            os.makedirs(quant_path, exist_ok=True)

            # Load model with quantization
            if bits == 8:
                # Check if the model architecture is supported
                model_config = model.config
                model_type = model_config.model_type

                # Load 8-bit quantized model
                if isinstance(model, transformers.PreTrainedModel) and hasattr(
                    model, "generate"
                ):
                    quant_model = AutoModelForCausalLM.from_pretrained(
                        self.model_dir, load_in_8bit=True, device_map="cpu"
                    )
                else:
                    # Not supported
                    return {
                        "error": f"8-bit quantization not supported for this model type: {model_type}"
                    }

            elif bits == 4:
                # Load 4-bit quantized model
                if isinstance(model, transformers.PreTrainedModel) and hasattr(
                    model, "generate"
                ):
                    quant_model = AutoModelForCausalLM.from_pretrained(
                        self.model_dir,
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        device_map="cpu",
                    )
                else:
                    # Not supported
                    return {
                        "error": "4-bit quantization not supported for this model type"
                    }

            else:
                return {"error": f"Unsupported quantization bits: {bits}. Use 4 or 8."}

            # Save model config (we can't save the quantized model directly)
            import json

            model_config_dict = model.config.to_dict()
            model_config_dict["_quantization_config"] = {
                "bits": bits,
                "load_in_4bit": bits == 4,
                "load_in_8bit": bits == 8,
            }

            with open(os.path.join(quant_path, "config.json"), "w") as f:
                json.dump(model_config_dict, f, indent=2)

            # Save quantization method
            with open(os.path.join(quant_path, "quantization_config.json"), "w") as f:
                json.dump(
                    {
                        "bits": bits,
                        "load_in_4bit": bits == 4,
                        "load_in_8bit": bits == 8,
                        "bnb_4bit_compute_dtype": "float16" if bits == 4 else None,
                        "bnb_4bit_quant_type": "nf4" if bits == 4 else None,
                    },
                    f,
                    indent=2,
                )

            # Save tokenizer
            tokenizer.save_pretrained(quant_path)

            # Save readme with instructions
            with open(os.path.join(quant_path, "README.md"), "w") as f:
                f.write(f"# {bits}-bit Quantized Model\n\n")
                f.write(
                    "This model has been quantized and should be loaded with the following parameters:\n\n"
                )
                if bits == 8:
                    f.write("```python\n")
                    f.write(
                        "from transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
                    )
                    f.write("model = AutoModelForCausalLM.from_pretrained(\n")
                    f.write("    'path/to/model',\n")
                    f.write("    load_in_8bit=True,\n")
                    f.write("    device_map='cpu'\n")
                    f.write(")\n")
                    f.write("```\n")
                else:
                    f.write("```python\n")
                    f.write(
                        "from transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
                    )
                    f.write("model = AutoModelForCausalLM.from_pretrained(\n")
                    f.write("    'path/to/model',\n")
                    f.write("    load_in_4bit=True,\n")
                    f.write("    bnb_4bit_compute_dtype=torch.float16,\n")
                    f.write("    bnb_4bit_quant_type='nf4',\n")
                    f.write("    device_map='cpu'\n")
                    f.write(")\n")
                    f.write("```\n")

            # Benchmark data
            input_text = "This is a benchmark input " * (sequence_length // 5 + 1)
            input_text = input_text[: sequence_length * 4]

            input_ids = tokenizer(input_text, return_tensors="pt").input_ids

            # Ensure input is the right length
            if input_ids.shape[1] > sequence_length:
                input_ids = input_ids[:, :sequence_length]

            # Warm up
            with torch.no_grad():
                for _ in range(3):
                    _ = quant_model(input_ids)

            # Memory before
            gc.collect()
            memory_before = psutil.Process(os.getpid()).memory_info().rss / (
                1024 * 1024
            )

            # Benchmark inference time
            times = []
            with torch.no_grad():
                for _ in range(10):
                    start_time = time.time()
                    _ = quant_model(input_ids)
                    end_time = time.time()
                    times.append(end_time - start_time)

            # Memory after
            memory_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

            # Generation benchmark (faster for comparison)
            gen_times = []
            with torch.no_grad():
                for _ in range(3):
                    start_time = time.time()
                    _ = quant_model.generate(
                        input_ids, max_new_tokens=20, do_sample=False
                    )
                    end_time = time.time()
                    gen_times.append(end_time - start_time)

            # Calculate results
            avg_time = np.mean(times)
            avg_gen_time = np.mean(gen_times)

            results = {
                "model": f"quantized_{bits}bit",
                "sequence_length": sequence_length,
                "avg_inference_time": avg_time,
                "avg_generation_time": avg_gen_time,
                "memory_usage_mb": memory_after - memory_before,
                "tokens_per_second": sequence_length / avg_time,
                "generation_tokens_per_second": 20 / avg_gen_time,
            }

            logger.info(
                f"{bits}-bit quantization model benchmark results: {json.dumps(results, indent=2)}"
            )

            return results

        except Exception as e:
            logger.error(f"Error optimizing with {bits}-bit quantization: {str(e)}")
            return {"error": str(e)}

    def optimize_all(
        self,
        model_type: str = "causal_lm",
        sequence_lengths: List[int] = [128, 512, 1024],
    ) -> Dict[str, Any]:
        """
        Run all optimizations and compare results.

        Args:
            model_type: Model type to optimize
            sequence_lengths: List of sequence lengths to benchmark

        Returns:
            Comparison of all optimization methods
        """
        logger.info(f"Running all optimizations for model type: {model_type}")

        # Load model for optimization
        model, tokenizer = self.load_model(model_type)

        # Store all results
        all_results = []
        optimizations = []

        # Run benchmarks for each sequence length
        for seq_length in sequence_lengths:
            # Benchmark original model
            original_results = self.benchmark_original(model, tokenizer, seq_length)
            all_results.append(original_results)

            # Optimize with BetterTransformer
            try:
                bt_results = self.optimize_with_bettertransformer(
                    model, tokenizer, seq_length
                )
                if "error" not in bt_results:
                    all_results.append(bt_results)
                    if "bettertransformer" not in optimizations:
                        optimizations.append("bettertransformer")
            except Exception as e:
                logger.error(f"Error in BetterTransformer optimization: {str(e)}")

            # Optimize with ONNX
            if ONNX_AVAILABLE and OPTIMUM_AVAILABLE:
                try:
                    onnx_results = self.optimize_with_onnx(model, tokenizer, seq_length)
                    if "error" not in onnx_results:
                        all_results.append(onnx_results)
                        if "onnx" not in optimizations:
                            optimizations.append("onnx")
                except Exception as e:
                    logger.error(f"Error in ONNX optimization: {str(e)}")

            # Optimize with 8-bit quantization
            try:
                quant8_results = self.optimize_with_quantization(
                    model, tokenizer, seq_length, 8
                )
                if "error" not in quant8_results:
                    all_results.append(quant8_results)
                    if "quantized_8bit" not in optimizations:
                        optimizations.append("quantized_8bit")
            except Exception as e:
                logger.error(f"Error in 8-bit quantization: {str(e)}")

            # Optimize with 4-bit quantization
            try:
                quant4_results = self.optimize_with_quantization(
                    model, tokenizer, seq_length, 4
                )
                if "error" not in quant4_results:
                    all_results.append(quant4_results)
                    if "quantized_4bit" not in optimizations:
                        optimizations.append("quantized_4bit")
            except Exception as e:
                logger.error(f"Error in 4-bit quantization: {str(e)}")

        # Prepare summary
        summary = {
            "model_type": model_type,
            "system_info": self.system_info,
            "optimizations_available": optimizations,
            "results": all_results,
            "recommendations": self._generate_recommendations(
                all_results, self.system_info
            ),
        }

        # Save summary
        summary_path = os.path.join(self.output_dir, "optimization_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Optimization summary saved to {summary_path}")

        return summary

    def _generate_recommendations(
        self, results: List[Dict[str, Any]], system_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate optimization recommendations based on benchmark results."""
        # Group results by sequence length
        by_seq_length = {}
        for result in results:
            if "error" in result:
                continue

            seq_length = result["sequence_length"]
            if seq_length not in by_seq_length:
                by_seq_length[seq_length] = []
            by_seq_length[seq_length].append(result)

        recommendations = {
            "best_inference_overall": None,
            "best_generation_overall": None,
            "best_memory_overall": None,
            "by_sequence_length": {},
        }

        # Find best model overall
        all_valid_results = [r for r in results if "error" not in r]
        if all_valid_results:
            inference_sorted = sorted(
                all_valid_results, key=lambda x: x["avg_inference_time"]
            )
            generation_sorted = sorted(
                all_valid_results, key=lambda x: x["avg_generation_time"]
            )
            memory_sorted = sorted(
                all_valid_results, key=lambda x: x["memory_usage_mb"]
            )

            recommendations["best_inference_overall"] = inference_sorted[0]["model"]
            recommendations["best_generation_overall"] = generation_sorted[0]["model"]
            recommendations["best_memory_overall"] = memory_sorted[0]["model"]

        # Find best model for each sequence length
        for seq_length, seq_results in by_seq_length.items():
            inference_sorted = sorted(
                seq_results, key=lambda x: x["avg_inference_time"]
            )
            generation_sorted = sorted(
                seq_results, key=lambda x: x["avg_generation_time"]
            )
            memory_sorted = sorted(seq_results, key=lambda x: x["memory_usage_mb"])

            recommendations["by_sequence_length"][seq_length] = {
                "best_inference": inference_sorted[0]["model"],
                "best_generation": generation_sorted[0]["model"],
                "best_memory": memory_sorted[0]["model"],
                "speedup_vs_original": {
                    "inference": inference_sorted[0]["avg_inference_time"]
                    / next(
                        r["avg_inference_time"]
                        for r in seq_results
                        if r["model"] == "original"
                    ),
                    "generation": generation_sorted[0]["avg_generation_time"]
                    / next(
                        r["avg_generation_time"]
                        for r in seq_results
                        if r["model"] == "original"
                    ),
                },
            }

        # Make specific recommendations for Dell PowerEdge R720 with E5-2640
        if self.is_e5_2640 or "E5-2640" in system_info["cpu"].get("processor", ""):
            recommendations["e5_2640_specific"] = {
                "notes": [
                    "Optimized for Dell PowerEdge R720 with Intel Xeon E5-2640 processors",
                    "E5-2640 has AVX instructions but not AVX2 or AVX-512",
                    "Models should be optimized for 12 physical cores",
                ]
            }

            # Check if ONNX is available and performed well
            if "onnx" in [r["model"] for r in all_valid_results]:
                onnx_results = [r for r in all_valid_results if r["model"] == "onnx"]
                original_results = [
                    r for r in all_valid_results if r["model"] == "original"
                ]

                if onnx_results and original_results:
                    avg_onnx_speedup = np.mean(
                        [
                            o["avg_inference_time"]
                            / next(
                                r["avg_inference_time"]
                                for r in original_results
                                if r["sequence_length"] == o["sequence_length"]
                            )
                            for o in onnx_results
                        ]
                    )

                    if avg_onnx_speedup < 0.7:  # At least 30% faster
                        recommendations["e5_2640_specific"][
                            "primary_recommendation"
                        ] = "onnx"
                    else:
                        # Check quantization results
                        quant_results = [
                            r for r in all_valid_results if "quantized" in r["model"]
                        ]
                        if quant_results:
                            quant8_results = [
                                r
                                for r in quant_results
                                if r["model"] == "quantized_8bit"
                            ]
                            if quant8_results:
                                recommendations["e5_2640_specific"][
                                    "primary_recommendation"
                                ] = "quantized_8bit"
                            else:
                                recommendations["e5_2640_specific"][
                                    "primary_recommendation"
                                ] = "quantized_4bit"
                        else:
                            recommendations["e5_2640_specific"][
                                "primary_recommendation"
                            ] = "bettertransformer"
            else:
                # No ONNX, check quantization
                quant_results = [
                    r for r in all_valid_results if "quantized" in r["model"]
                ]
                if quant_results:
                    quant8_results = [
                        r for r in quant_results if r["model"] == "quantized_8bit"
                    ]
                    if quant8_results:
                        recommendations["e5_2640_specific"][
                            "primary_recommendation"
                        ] = "quantized_8bit"
                    else:
                        recommendations["e5_2640_specific"][
                            "primary_recommendation"
                        ] = "quantized_4bit"
                else:
                    recommendations["e5_2640_specific"]["primary_recommendation"] = (
                        "bettertransformer"
                    )

        # Consider available memory
        available_memory_gb = system_info["memory"]["available_gb"]
        if available_memory_gb < 16:
            recommendations["memory_constrained"] = True
            recommendations["memory_notes"] = [
                f"System has limited available memory ({available_memory_gb:.2f} GB)",
                "Quantization or ONNX runtime recommended for memory efficiency",
            ]

        return recommendations


def main():
    """Main function to run optimization."""
    parser = argparse.ArgumentParser(
        description="Optimize transformer models for CPU inference"
    )
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Directory containing the model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save optimized models",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="causal_lm",
        choices=["causal_lm", "seq2seq"],
        help="Type of model to optimize",
    )
    parser.add_argument(
        "--sequence_lengths",
        type=str,
        default="128,512,1024",
        help="Comma-separated list of sequence lengths to benchmark",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for optimization",
    )

    args = parser.parse_args()

    # Parse sequence lengths
    sequence_lengths = [int(s.strip()) for s in args.sequence_lengths.split(",")]

    # Initialize optimizer
    optimizer = TransformerOptimizer(args.model_dir, args.output_dir, args.device)

    # Run optimization
    results = optimizer.optimize_all(args.model_type, sequence_lengths)

    # Print summary
    recommendations = results.get("recommendations", {})

    print("\n===== OPTIMIZATION SUMMARY =====")
    print(f"Model type: {args.model_type}")
    print(f"Sequence lengths tested: {sequence_lengths}")
    print(
        f"Best overall inference method: {recommendations.get('best_inference_overall', 'N/A')}"
    )
    print(
        f"Best overall generation method: {recommendations.get('best_generation_overall', 'N/A')}"
    )
    print(
        f"Best memory-efficient method: {recommendations.get('best_memory_overall', 'N/A')}"
    )

    # Print Dell PowerEdge R720 specific recommendations if available
    if "e5_2640_specific" in recommendations:
        print("\nDell PowerEdge R720 with E5-2640 specific recommendations:")
        for note in recommendations["e5_2640_specific"].get("notes", []):
            print(f"- {note}")
        print(
            f"Primary recommendation: {recommendations['e5_2640_specific'].get('primary_recommendation', 'N/A')}"
        )

    # Print memory notes if available
    if "memory_notes" in recommendations:
        print("\nMemory considerations:")
        for note in recommendations["memory_notes"]:
            print(f"- {note}")

    print(
        "\nDetailed results saved to:",
        os.path.join(args.output_dir, "optimization_summary.json"),
    )


if __name__ == "__main__":
    main()
