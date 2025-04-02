#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_utils.py - Benchmark utilities for transformer optimization

This module provides functions to benchmark different transformer model optimizations
based on Intel's research techniques for CPU-optimized inference.
"""

import os
import time
import json
import logging
import sys
from typing import Dict, Any, Tuple

import torch

logger = logging.getLogger("sutazai.benchmark")


def get_process_memory() -> float:
    """Get the current memory usage of the process in MB."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        return memory_mb
    except ImportError:
        logger.warning("psutil not available, returning estimated memory")
        # Return a rough estimate if psutil is not available
        return (
            torch.cuda.memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available()
            else 0.0
        )


def benchmark_original(
    model_path: str, sequence_length: int = 512, model_type: str = "causal_lm"
) -> Dict[str, Any]:
    """
    Benchmark the original unoptimized transformer model.

    Args:
        model_path: Path to the model
        sequence_length: Sequence length to benchmark
        model_type: Type of model (causal_lm or seq2seq)

    Returns:
        Benchmark results
    """
    logger.info(
        f"Benchmarking original model at {model_path} with sequence length {sequence_length}"
    )

    # Record memory before loading model
    start_memory = get_process_memory()
    load_start_time = time.time()

    try:
        # Import transformers dynamically to avoid import errors if not available
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load appropriate model based on type
        if model_type == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(model_path)
        elif model_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model.eval()  # Set to evaluation mode

        # Record model load time
        load_time = time.time() - load_start_time

        # Record memory after loading model
        model_memory = get_process_memory() - start_memory

        # Prepare input
        if model_type == "causal_lm":
            input_text = "This is a test input to benchmark the model performance."
            inputs = tokenizer(input_text, return_tensors="pt")

            # Generate a short text to benchmark
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)

            # Warm-up run
            with torch.no_grad():
                _ = model.generate(
                    input_ids,
                    max_new_tokens=sequence_length,
                    attention_mask=attention_mask,
                    do_sample=False,
                    num_beams=1,
                )

            # Benchmarking run
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=sequence_length,
                    attention_mask=attention_mask,
                    do_sample=False,
                    num_beams=1,
                )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

        elif model_type == "seq2seq":
            input_text = "translate English to French: This is a test input to benchmark the model performance."
            inputs = tokenizer(input_text, return_tensors="pt")

            # Warm-up run
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=sequence_length,
                    do_sample=False,
                    num_beams=1,
                )

            # Benchmarking run
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=sequence_length,
                    do_sample=False,
                    num_beams=1,
                )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

        # Calculate results
        inference_time = end_time - start_time
        output_length = output.shape[1]
        tokens_per_second = output_length / inference_time

        logger.info(
            f"Original model benchmark completed in {inference_time:.4f} seconds"
        )
        logger.info(f"Tokens per second: {tokens_per_second:.2f}")

        # Return results
        return {
            "model_type": model_type,
            "sequence_length": sequence_length,
            "load_time_seconds": load_time,
            "inference_time_seconds": inference_time,
            "tokens_per_second": tokens_per_second,
            "memory_mb": model_memory,
            "model_size_parameters": sum(p.numel() for p in model.parameters()),
        }

    except Exception as e:
        logger.error(f"Error benchmarking original model: {e}")
        return {"error": str(e)}


def benchmark_fp16(
    model_path: str,
    output_path: str,
    sequence_length: int = 512,
    model_type: str = "causal_lm",
) -> Tuple[Dict[str, Any], str]:
    """
    Benchmark the model with FP16 optimization.

    Args:
        model_path: Path to the model
        output_path: Path to save the optimized model
        sequence_length: Sequence length to benchmark
        model_type: Type of model (causal_lm or seq2seq)

    Returns:
        Tuple of (benchmark results, path to optimized model)
    """
    logger.info(f"Benchmarking FP16 model with sequence length {sequence_length}")

    # Record memory before loading model
    start_memory = get_process_memory()
    load_start_time = time.time()

    try:
        # Import transformers dynamically to avoid import errors if not available
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
        )

        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load appropriate model based on type with FP16
        if model_type == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16
            )
        elif model_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path, torch_dtype=torch.float16
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model.eval()  # Set to evaluation mode

        # Record model load time
        load_time = time.time() - load_start_time

        # Record memory after loading model
        model_memory = get_process_memory() - start_memory

        # Prepare input
        if model_type == "causal_lm":
            input_text = "This is a test input to benchmark the model performance."
            inputs = tokenizer(input_text, return_tensors="pt")

            # Generate a short text to benchmark
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)

            # Warm-up run
            with torch.no_grad():
                _ = model.generate(
                    input_ids,
                    max_new_tokens=sequence_length,
                    attention_mask=attention_mask,
                    do_sample=False,
                    num_beams=1,
                )

            # Benchmarking run
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=sequence_length,
                    attention_mask=attention_mask,
                    do_sample=False,
                    num_beams=1,
                )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

        elif model_type == "seq2seq":
            input_text = "translate English to French: This is a test input to benchmark the model performance."
            inputs = tokenizer(input_text, return_tensors="pt")

            # Warm-up run
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=sequence_length,
                    do_sample=False,
                    num_beams=1,
                )

            # Benchmarking run
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=sequence_length,
                    do_sample=False,
                    num_beams=1,
                )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

        # Calculate results
        inference_time = end_time - start_time
        output_length = output.shape[1]
        tokens_per_second = output_length / inference_time

        logger.info(f"FP16 model benchmark completed in {inference_time:.4f} seconds")
        logger.info(f"Tokens per second: {tokens_per_second:.2f}")

        # Save the model
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        # Return results
        return {
            "model_type": model_type,
            "sequence_length": sequence_length,
            "optimization": "fp16",
            "load_time_seconds": load_time,
            "inference_time_seconds": inference_time,
            "tokens_per_second": tokens_per_second,
            "memory_mb": model_memory,
            "model_size_parameters": sum(p.numel() for p in model.parameters()),
        }, output_path

    except Exception as e:
        logger.error(f"Error benchmarking FP16 model: {e}")
        return {"error": str(e)}, output_path


def benchmark_int8(
    model_path: str,
    output_path: str,
    sequence_length: int = 512,
    model_type: str = "causal_lm",
) -> Tuple[Dict[str, Any], str]:
    """
    Benchmark the model with Int8 quantization.

    Args:
        model_path: Path to the model
        output_path: Path to save the optimized model
        sequence_length: Sequence length to benchmark
        model_type: Type of model (causal_lm or seq2seq)

    Returns:
        Tuple of (benchmark results, path to optimized model)
    """
    logger.info(f"Benchmarking Int8 model with sequence length {sequence_length}")

    # Record memory before loading model
    start_memory = get_process_memory()
    load_start_time = time.time()

    try:
        # Import transformers dynamically to avoid import errors if not available
        from transformers import AutoTokenizer

        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load appropriate model based on type with Int8 quantization
        try:
            # Newer transformers versions with BitsAndBytesConfig
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                BitsAndBytesConfig,
            )

            # Configure quantization
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_threshold=6.0
            )

            if model_type == "causal_lm":
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
            elif model_type == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        except (ImportError, AttributeError):
            # Fallback for older transformers versions
            logger.info("Using legacy 8-bit quantization approach")

            if model_type == "causal_lm":
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(
                    model_path, load_in_8bit=True, device_map="auto"
                )
            elif model_type == "seq2seq":
                from transformers import AutoModelForSeq2SeqLM

                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path, load_in_8bit=True, device_map="auto"
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        model.eval()  # Set to evaluation mode

        # Record model load time
        load_time = time.time() - load_start_time

        # Record memory after loading model
        model_memory = get_process_memory() - start_memory

        # Prepare input
        if model_type == "causal_lm":
            input_text = "This is a test input to benchmark the model performance."
            inputs = tokenizer(input_text, return_tensors="pt")

            # Generate a short text to benchmark
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)

            # Warm-up run
            with torch.no_grad():
                _ = model.generate(
                    input_ids,
                    max_new_tokens=sequence_length,
                    attention_mask=attention_mask,
                    do_sample=False,
                    num_beams=1,
                )

            # Benchmarking run
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=sequence_length,
                    attention_mask=attention_mask,
                    do_sample=False,
                    num_beams=1,
                )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

        elif model_type == "seq2seq":
            input_text = "translate English to French: This is a test input to benchmark the model performance."
            inputs = tokenizer(input_text, return_tensors="pt")

            # Warm-up run
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=sequence_length,
                    do_sample=False,
                    num_beams=1,
                )

            # Benchmarking run
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=sequence_length,
                    do_sample=False,
                    num_beams=1,
                )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

        # Calculate results
        inference_time = end_time - start_time
        output_length = output.shape[1]
        tokens_per_second = output_length / inference_time

        logger.info(f"Int8 model benchmark completed in {inference_time:.4f} seconds")
        logger.info(f"Tokens per second: {tokens_per_second:.2f}")

        # Save the model configuration and metadata (can't save quantized models directly)
        with open(os.path.join(output_path, "optimization_info.json"), "w") as f:
            json.dump(
                {
                    "optimization": "int8",
                    "sequence_length": sequence_length,
                    "inference_time_seconds": inference_time,
                    "tokens_per_second": tokens_per_second,
                    "memory_mb": model_memory,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
            )

        # Return results
        return {
            "model_type": model_type,
            "sequence_length": sequence_length,
            "optimization": "int8",
            "load_time_seconds": load_time,
            "inference_time_seconds": inference_time,
            "tokens_per_second": tokens_per_second,
            "memory_mb": model_memory,
        }, output_path

    except Exception as e:
        logger.error(f"Error benchmarking Int8 model: {e}")
        return {"error": str(e)}, output_path


def benchmark_onnx(
    model_path: str,
    output_path: str,
    sequence_length: int = 512,
    model_type: str = "causal_lm",
) -> Tuple[Dict[str, Any], str]:
    """
    Benchmark the model with ONNX conversion.

    Args:
        model_path: Path to the model
        output_path: Path to save the optimized model
        sequence_length: Sequence length to benchmark
        model_type: Type of model (causal_lm or seq2seq)

    Returns:
        Tuple of (benchmark results, path to optimized model)
    """
    logger.info(f"Benchmarking ONNX model with sequence length {sequence_length}")

    # Record memory before loading model
    start_memory = get_process_memory()
    load_start_time = time.time()

    try:
        # Try to import optimum for ONNX conversion
        try:
            from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForSeq2SeqLM
            from transformers import AutoTokenizer
        except ImportError:
            logger.error("optimum library not available for ONNX conversion")
            return {
                "error": "optimum library not available for ONNX conversion"
            }, output_path

        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load appropriate model based on type and convert to ONNX
        if model_type == "causal_lm":
            # Convert to ONNX
            onnx_model = ORTModelForCausalLM.from_pretrained(
                model_path, from_transformers=True, provider="CPUExecutionProvider"
            )
        elif model_type == "seq2seq":
            # Convert to ONNX
            onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
                model_path, from_transformers=True, provider="CPUExecutionProvider"
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Record model load time
        load_time = time.time() - load_start_time

        # Record memory after loading model
        model_memory = get_process_memory() - start_memory

        # Prepare input
        if model_type == "causal_lm":
            input_text = "This is a test input to benchmark the model performance."
            inputs = tokenizer(input_text, return_tensors="pt")

            # Warm-up run
            _ = onnx_model.generate(
                **inputs, max_new_tokens=sequence_length, do_sample=False, num_beams=1
            )

            # Benchmarking run
            start_time = time.time()

            output = onnx_model.generate(
                **inputs, max_new_tokens=sequence_length, do_sample=False, num_beams=1
            )

            end_time = time.time()

        elif model_type == "seq2seq":
            input_text = "translate English to French: This is a test input to benchmark the model performance."
            inputs = tokenizer(input_text, return_tensors="pt")

            # Warm-up run
            _ = onnx_model.generate(
                **inputs, max_new_tokens=sequence_length, do_sample=False, num_beams=1
            )

            # Benchmarking run
            start_time = time.time()

            output = onnx_model.generate(
                **inputs, max_new_tokens=sequence_length, do_sample=False, num_beams=1
            )

            end_time = time.time()

        # Calculate results
        inference_time = end_time - start_time
        output_length = output.shape[1]
        tokens_per_second = output_length / inference_time

        logger.info(f"ONNX model benchmark completed in {inference_time:.4f} seconds")
        logger.info(f"Tokens per second: {tokens_per_second:.2f}")

        # Save the ONNX model
        onnx_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        # Return results
        return {
            "model_type": model_type,
            "sequence_length": sequence_length,
            "optimization": "onnx",
            "load_time_seconds": load_time,
            "inference_time_seconds": inference_time,
            "tokens_per_second": tokens_per_second,
            "memory_mb": model_memory,
        }, output_path

    except Exception as e:
        logger.error(f"Error benchmarking ONNX model: {e}")
        return {"error": str(e)}, output_path


def benchmark_bettertransformer(
    model_path: str,
    output_path: str,
    sequence_length: int = 512,
    model_type: str = "causal_lm",
) -> Tuple[Dict[str, Any], str]:
    """
    Benchmark the model with BetterTransformer optimization.

    Args:
        model_path: Path to the model
        output_path: Path to save the optimized model
        sequence_length: Sequence length to benchmark
        model_type: Type of model (causal_lm or seq2seq)

    Returns:
        Tuple of (benchmark results, path to optimized model)
    """
    logger.info(
        f"Benchmarking BetterTransformer model with sequence length {sequence_length}"
    )

    # Record memory before loading model
    start_memory = get_process_memory()
    load_start_time = time.time()

    try:
        # Import transformers dynamically to avoid import errors if not available
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
        )

        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load appropriate model based on type
        if model_type == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(model_path)
        elif model_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Apply BetterTransformer
        try:
            # Try with newer transformers versions
            try:
                from transformers.utils import BetterTransformer

                model = BetterTransformer.transform(model)
                logger.info(
                    "BetterTransformer applied successfully using transformers.utils.BetterTransformer"
                )
            except (ImportError, AttributeError):
                # Fallback for older versions
                model = model.to_bettertransformer()
                logger.info(
                    "BetterTransformer applied successfully using to_bettertransformer()"
                )
        except Exception as e:
            logger.error(f"Failed to apply BetterTransformer: {e}")
            return {"error": f"Failed to apply BetterTransformer: {e}"}, output_path

        model.eval()  # Set to evaluation mode

        # Record model load time
        load_time = time.time() - load_start_time

        # Record memory after loading model
        model_memory = get_process_memory() - start_memory

        # Prepare input
        if model_type == "causal_lm":
            input_text = "This is a test input to benchmark the model performance."
            inputs = tokenizer(input_text, return_tensors="pt")

            # Generate a short text to benchmark
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)

            # Warm-up run
            with torch.no_grad():
                _ = model.generate(
                    input_ids,
                    max_new_tokens=sequence_length,
                    attention_mask=attention_mask,
                    do_sample=False,
                    num_beams=1,
                )

            # Benchmarking run
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=sequence_length,
                    attention_mask=attention_mask,
                    do_sample=False,
                    num_beams=1,
                )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

        elif model_type == "seq2seq":
            input_text = "translate English to French: This is a test input to benchmark the model performance."
            inputs = tokenizer(input_text, return_tensors="pt")

            # Warm-up run
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=sequence_length,
                    do_sample=False,
                    num_beams=1,
                )

            # Benchmarking run
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=sequence_length,
                    do_sample=False,
                    num_beams=1,
                )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

        # Calculate results
        inference_time = end_time - start_time
        output_length = output.shape[1]
        tokens_per_second = output_length / inference_time

        logger.info(
            f"BetterTransformer model benchmark completed in {inference_time:.4f} seconds"
        )
        logger.info(f"Tokens per second: {tokens_per_second:.2f}")

        # Save configuration and metadata
        with open(os.path.join(output_path, "optimization_info.json"), "w") as f:
            json.dump(
                {
                    "optimization": "bettertransformer",
                    "sequence_length": sequence_length,
                    "inference_time_seconds": inference_time,
                    "tokens_per_second": tokens_per_second,
                    "memory_mb": model_memory,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
            )

        # Return results
        return {
            "model_type": model_type,
            "sequence_length": sequence_length,
            "optimization": "bettertransformer",
            "load_time_seconds": load_time,
            "inference_time_seconds": inference_time,
            "tokens_per_second": tokens_per_second,
            "memory_mb": model_memory,
            "model_size_parameters": sum(p.numel() for p in model.parameters()),
        }, output_path

    except Exception as e:
        logger.error(f"Error benchmarking BetterTransformer model: {e}")
        return {"error": str(e)}, output_path


def benchmark_lookupffn(
    model_path: str,
    output_path: str,
    sequence_length: int = 512,
    model_type: str = "causal_lm",
) -> Tuple[Dict[str, Any], str]:
    """
    Benchmark the model with LookupFFN optimization.

    Args:
        model_path: Path to the model
        output_path: Path to save the optimized model
        sequence_length: Sequence length to benchmark
        model_type: Type of model (causal_lm or seq2seq)

    Returns:
        Tuple of (benchmark results, path to optimized model)
    """
    logger.info(f"Benchmarking LookupFFN model with sequence length {sequence_length}")

    # Record memory before loading model
    start_memory = get_process_memory()
    load_start_time = time.time()

    try:
        # Try to import LookupFFN
        try:
            sys.path.append("/opt/sutazaiapp")
            from core.neural.lookup_ffn import LookupFFN
        except ImportError:
            logger.error(
                "LookupFFN module not found. Make sure it's properly installed."
            )
            return {"error": "LookupFFN module not found"}, output_path

        # Import transformers dynamically to avoid import errors if not available
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
        )

        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load appropriate model based on type
        if model_type == "causal_lm":
            model = AutoModelForCausalLM.from_pretrained(model_path)
        elif model_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Apply LookupFFN optimization
        ffn_modules_replaced = 0

        # Loop through the model's modules and replace FFNs with LookupFFN
        for name, module in model.named_modules():
            # Look for modules that might be FFNs based on naming patterns
            if any(
                pattern in name.lower() for pattern in ["ffn", "mlp", "feed_forward"]
            ):
                if hasattr(module, "fc1") and hasattr(module, "fc2"):
                    # This looks like an FFN - replace it with LookupFFN
                    try:
                        # Convert to LookupFFN
                        lookup_ffn = LookupFFN.from_standard_ffn(module)

                        # Replace module
                        module_path = name.split(".")
                        parent_module = model
                        for part in module_path[:-1]:
                            parent_module = getattr(parent_module, part)

                        setattr(parent_module, module_path[-1], lookup_ffn)
                        ffn_modules_replaced += 1

                        if ffn_modules_replaced % 5 == 0:
                            logger.info(
                                f"Replaced {ffn_modules_replaced} FFN modules with LookupFFN"
                            )

                    except Exception as e:
                        logger.warning(f"Failed to replace FFN module {name}: {e}")

        if ffn_modules_replaced == 0:
            logger.warning("No FFN modules replaced with LookupFFN")
            return {"error": "No FFN modules replaced with LookupFFN"}, output_path
        else:
            logger.info(
                f"LookupFFN optimization: replaced {ffn_modules_replaced} FFN modules"
            )

        model.eval()  # Set to evaluation mode

        # Record model load time
        load_time = time.time() - load_start_time

        # Record memory after loading model
        model_memory = get_process_memory() - start_memory

        # Prepare input
        if model_type == "causal_lm":
            input_text = "This is a test input to benchmark the model performance."
            inputs = tokenizer(input_text, return_tensors="pt")

            # Generate a short text to benchmark
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)

            # Warm-up run
            with torch.no_grad():
                _ = model.generate(
                    input_ids,
                    max_new_tokens=sequence_length,
                    attention_mask=attention_mask,
                    do_sample=False,
                    num_beams=1,
                )

            # Benchmarking run
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=sequence_length,
                    attention_mask=attention_mask,
                    do_sample=False,
                    num_beams=1,
                )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

        elif model_type == "seq2seq":
            input_text = "translate English to French: This is a test input to benchmark the model performance."
            inputs = tokenizer(input_text, return_tensors="pt")

            # Warm-up run
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=sequence_length,
                    do_sample=False,
                    num_beams=1,
                )

            # Benchmarking run
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=sequence_length,
                    do_sample=False,
                    num_beams=1,
                )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

        # Calculate results
        inference_time = end_time - start_time
        output_length = output.shape[1]
        tokens_per_second = output_length / inference_time

        logger.info(
            f"LookupFFN model benchmark completed in {inference_time:.4f} seconds"
        )
        logger.info(f"Tokens per second: {tokens_per_second:.2f}")

        # Save configuration and metadata
        with open(os.path.join(output_path, "optimization_info.json"), "w") as f:
            json.dump(
                {
                    "optimization": "lookupffn",
                    "sequence_length": sequence_length,
                    "inference_time_seconds": inference_time,
                    "tokens_per_second": tokens_per_second,
                    "memory_mb": model_memory,
                    "ffn_modules_replaced": ffn_modules_replaced,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
            )

        # Return results
        return {
            "model_type": model_type,
            "sequence_length": sequence_length,
            "optimization": "lookupffn",
            "load_time_seconds": load_time,
            "inference_time_seconds": inference_time,
            "tokens_per_second": tokens_per_second,
            "memory_mb": model_memory,
            "ffn_modules_replaced": ffn_modules_replaced,
            "model_size_parameters": sum(p.numel() for p in model.parameters()),
        }, output_path

    except Exception as e:
        logger.error(f"Error benchmarking LookupFFN model: {e}")
        return {"error": str(e)}, output_path
