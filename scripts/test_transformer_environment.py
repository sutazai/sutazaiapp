#!/usr/bin/env python3
# Test script to verify transformer optimization environment

import os
import platform
import torch


def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 50)
    print(f" {title}")
    print("=" * 50)


def print_env_var(name):
    """Print an environment variable value."""
    value = os.environ.get(name, "Not set")
    print(f"{name}: {value}")


print_section("System Information")
print(f"Python version: {platform.python_version()}")
print(f"CPU: {platform.processor()}")
print(f"Platform: {platform.platform()}")

# Get CPU details
try:
    import psutil

    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    print(f"Physical cores: {physical_cores}")
    print(f"Logical cores: {logical_cores}")

    memory = psutil.virtual_memory()
    print(f"Total memory: {memory.total / (1024**3):.2f} GB")
    print(f"Available memory: {memory.available / (1024**3):.2f} GB")
except ImportError:
    print("psutil not installed, skipping detailed CPU info")

print_section("PyTorch Information")
print(f"PyTorch version: {torch.__version__}")
print(f"Available backend: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
print(f"PyTorch thread count: {torch.get_num_threads()}")

# Test Intel optimizations
try:
    import intel_extension_for_pytorch as ipex

    print(f"Intel Extension for PyTorch version: {ipex.__version__}")
    print("Intel Extension for PyTorch is available")
except ImportError:
    print("Intel Extension for PyTorch not installed")

print_section("Environment Variables")
print_env_var("MKL_NUM_THREADS")
print_env_var("OMP_NUM_THREADS")
print_env_var("MKL_ENABLE_INSTRUCTIONS")
print_env_var("KMP_AFFINITY")
print_env_var("KMP_BLOCKTIME")

print_section("Tensor Performance Test")
# Simple matrix multiplication benchmark
try:
    sizes = [1000, 2000, 4000]
    results = []

    print("Running simple matrix multiplication benchmark...")
    for size in sizes:
        # Create random matrices
        a = torch.randn(size, size)
        b = torch.randn(size, size)

        # Warm-up
        for _ in range(3):
            c = torch.matmul(a, b)

        # Benchmark
        import time

        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()

        results.append((size, end - start))

    print("\nMatrix multiplication results:")
    print("Size\tTime (s)")
    for size, duration in results:
        print(f"{size}\t{duration:.4f}")
except Exception as e:
    print(f"Error running benchmark: {e}")

print("\nEnvironment check completed.")
