# SutazAI Enterprise Model Management System

The SutazAI Model Management System is an enterprise-grade solution for automating the entire lifecycle of large language models (LLMs), with special optimization for Dell PowerEdge R720 servers equipped with E5-2640 CPUs.

## Features

- **Automatic Model Downloading**: Efficiently downloads models from HuggingFace Hub using parallel processing for faster acquisition.
- **Hardware-Specific Optimization**: Automatically optimizes models for Dell PowerEdge R720 with E5-2640 CPUs.
- **Performance Monitoring**: Continuously monitors model performance metrics and detects degradation.
- **Automated Retraining**: Triggers retraining when performance drops below thresholds.
- **Enterprise Integration**: Centralized management with monitoring, alerts, and detailed statistics.
- **Simplified API**: Easy-to-use controller interface for applications to leverage optimized models.

## System Architecture

The system consists of several integrated components:

- **Model Downloader**: Enterprise-grade parallel downloader with checksum verification.
- **Model Monitor**: Tracks performance metrics and provides alerts when issues are detected.
- **Model Manager**: Orchestrates the entire model lifecycle, from downloading to optimization to retraining.
- **Model Controller**: Provides a simple API for applications to work with managed models.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sutazaiapp.git
cd sutazaiapp

# Install dependencies
pip install -r requirements.txt

# Set up the environment
mkdir -p models optimized_models logs
```

## Basic Usage

### Command Line Interface

The system includes a comprehensive CLI for working with models:

```bash
# List available models
python main.py models list

# Download and optimize a model
python main.py models get llama3-70b --auto-optimize

# Generate text with an optimized model
python main.py generate --model llama3-70b --prompt "Tell me about quantum computing" --output result.txt
```

### Python API

```python
from core.neural.model_controller import ModelController

# Initialize the controller
controller = ModelController()

# Generate text with automatic model management
result = controller.generate(
    prompt="Explain the theory of relativity",
    model_id="llama3-8b",
    max_tokens=500
)

print(result["response"])
```

## Hardware Optimization

The system is specifically optimized for Dell PowerEdge R720 servers with E5-2640 CPUs, including:

- Thread allocation optimized for E5-2640's dual-socket, 6-core configuration
- Memory utilization tuned for typical R720 memory configurations
- Power efficiency settings for extended inference operations

## Monitoring and Management

The system provides comprehensive monitoring capabilities:

- Real-time performance metrics
- Automatic detection of performance degradation
- Alerts for critical issues
- Usage statistics and resource utilization tracking

## Advanced Configuration

Configuration files are stored in JSON format and can be customized to adjust:

- Performance thresholds for alerts
- Optimization parameters
- Monitoring intervals
- Resource limits

## Testing

Comprehensive test suite to ensure system reliability:

```bash
# Run all tests
python -m unittest discover -s core/neural/tests

# Run specific test case
python -m unittest core.neural.tests.test_models.TestModelDownloader
```

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for GPU acceleration)
- 16+ GB RAM (more recommended for larger models)
- 100+ GB storage space for models

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [HuggingFace](https://huggingface.co/) for model access
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for Llama model support

## Support

For enterprise support, contact support@sutazai.com

## Cache Management

SutazAI includes a powerful cache management system to prevent excessive cache buildup and maintain system performance.

### Automatic Cache Cleanup

The system has built-in automatic cache cleanup that runs daily (at midnight) to enforce cache limits and once weekly (Sunday at 4 AM) to perform a full cleanup.

These are managed through cron jobs defined in `/etc/cron.d/sutazai-cache`.

### Manual Cache Cleanup

You can manually clean cache at any time using the provided scripts:

1. Basic cleanup:
   ```bash
   ./cleanup_cache.sh
   ```

2. Advanced cleanup with options:
   ```bash
   # Clean all caches with verbose output
   ./cache_cleanup.py --verbose
   
   # Clean only Python bytecode cache
   ./cache_cleanup.py --pycache-only
   
   # Clean only model cache
   ./cache_cleanup.py --model-cache-only
   
   # Just enforce limits without full cleanup
   ./cache_cleanup.py --enforce-limits
   ```

### Cache Prevention

The system is configured to prevent Python from creating bytecode cache files (`__pycache__` directories and `.pyc` files) by setting the `PYTHONDONTWRITEBYTECODE=1` environment variable in:

- `/etc/profile` (system-wide)
- Virtual environment activation scripts

### Cache Management API

For programmatic cache management, you can use the provided Python API:

```python
from utils.cache_manager import CacheManager

# Create a cache manager with custom settings
manager = CacheManager(
    max_cache_size_mb=5000,  # 5GB max cache
    cache_ttl_days=30        # 30 days time-to-live
)

# Clean all caches
manager.clean_all_caches()

# Or just clean specific caches
manager.clean_pycache()
manager.clean_model_cache()

# Check current cache size
size_mb = manager.get_cache_size()
print(f"Current cache size: {size_mb:.2f} MB")

# Enforce cache limits (cleans if over max size or TTL)
manager.enforce_cache_limits()
```

This cache management system helps keep the application running smoothly by preventing excessive cache buildup. 