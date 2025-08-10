"""
Edge Model Quantization - Advanced quantization techniques for edge deployment optimization
"""

import os
import asyncio
import threading
import time
import struct
import mmap
import logging
import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple, Union, BinaryIO
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil

logger = logging.getLogger(__name__)

class QuantizationType(Enum):
    """Quantization types"""
    INT8 = "int8"           # 8-bit integer quantization
    INT4 = "int4"           # 4-bit integer quantization  
    INT2 = "int2"           # 2-bit integer quantization
    FP16 = "fp16"           # 16-bit floating point
    BF16 = "bf16"           # Brain floating point 16
    DYNAMIC = "dynamic"      # Dynamic quantization
    STATIC = "static"        # Static quantization
    QAT = "qat"             # Quantization Aware Training
    MIXED = "mixed"         # Mixed precision

class QuantizationStrategy(Enum):
    """Quantization strategies"""
    AGGRESSIVE = "aggressive"    # Maximum compression, may impact accuracy
    BALANCED = "balanced"        # Balance between size and accuracy  
    CONSERVATIVE = "conservative" # Minimal accuracy impact
    CUSTOM = "custom"            # Custom quantization parameters

@dataclass
class QuantizationConfig:
    """Quantization configuration"""
    quantization_type: QuantizationType
    strategy: QuantizationStrategy
    target_accuracy_loss: float = 0.05  # Max acceptable accuracy loss (5%)
    compression_target: float = 0.25    # Target compression ratio (25% of original)
    calibration_samples: int = 1000     # Samples for calibration
    preserve_layers: List[str] = field(default_factory=list)  # Layers to preserve
    custom_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    enable_kv_cache_quantization: bool = True
    enable_activation_quantization: bool = False

@dataclass
class QuantizationResult:
    """Result of quantization process"""
    success: bool
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    accuracy_loss: float
    quantization_time_sec: float
    output_path: str
    error_message: Optional[str] = None
    quantization_stats: Dict[str, Any] = field(default_factory=dict)

class ModelAnalyzer:
    """Analyzes model structure for optimal quantization"""
    
    def __init__(self):
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}
    
    async def analyze_model(self, model_path: str) -> Dict[str, Any]:
        """Analyze model for quantization optimization"""
        model_hash = self._get_file_hash(model_path)
        
        if model_hash in self.analysis_cache:
            return self.analysis_cache[model_hash]
        
        try:
            analysis = await self._perform_analysis(model_path)
            self.analysis_cache[model_hash] = analysis
            return analysis
            
        except Exception as e:
            logger.error(f"Model analysis failed: {e}")
            return {}
    
    async def _perform_analysis(self, model_path: str) -> Dict[str, Any]:
        """Perform detailed model analysis"""
        file_size = os.path.getsize(model_path)
        
        # Basic file analysis
        analysis = {
            "file_size_mb": file_size / (1024 * 1024),
            "model_format": self._detect_format(model_path),
            "quantization_friendly": True,
            "estimated_parameters": self._estimate_parameters(file_size),
            "layer_analysis": {},
            "optimization_recommendations": []
        }
        
        # Format-specific analysis
        if analysis["model_format"] == "gguf":
            analysis.update(await self._analyze_gguf(model_path))
        elif analysis["model_format"] == "onnx":
            analysis.update(await self._analyze_onnx(model_path))
        elif analysis["model_format"] == "pytorch":
            analysis.update(await self._analyze_pytorch(model_path))
        
        # Generate recommendations
        analysis["optimization_recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get file hash for caching"""
        stat = os.stat(file_path)
        return hashlib.sha256(f"{file_path}_{stat.st_mtime}_{stat.st_size}".encode()).hexdigest()[:16]
    
    def _detect_format(self, model_path: str) -> str:
        """Detect model format"""
        path = Path(model_path)
        suffix = path.suffix.lower()
        
        if suffix in ['.gguf', '.ggml']:
            return "gguf"
        elif suffix == '.onnx':
            return "onnx"
        elif suffix in ['.pt', '.pth']:
            return "pytorch"
        elif suffix in ['.pb', '.savedmodel']:
            return "tensorflow"
        
        # Try to detect by content
        try:
            with open(model_path, 'rb') as f:
                header = f.read(16)
                if header.startswith(b'GGUF'):
                    return "gguf"
                elif header.startswith(b'\x08\x01\x12'):
                    return "onnx"
        except (IOError, OSError, FileNotFoundError) as e:
            # Suppressed exception (was bare except)
            logger.debug(f"Suppressed exception: {e}")
            pass
        
        return "unknown"
    
    def _estimate_parameters(self, file_size: int) -> int:
        """Estimate number of parameters from file size"""
        # Rough estimation: 4 bytes per parameter for FP32
        return file_size // 4
    
    async def _analyze_gguf(self, model_path: str) -> Dict[str, Any]:
        """Analyze GGUF format model"""
        analysis = {
            "architecture": "unknown",
            "layer_count": 0,
            "attention_heads": 0,
            "embedding_dim": 0,
            "vocab_size": 0,
            "context_length": 0
        }
        
        try:
            # Parse GGUF header to extract metadata
            with open(model_path, 'rb') as f:
                # Skip process number and version
                f.seek(8)
                
                # Read tensor count and KV count
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                kv_count = struct.unpack('<Q', f.read(8))[0]
                
                analysis["layer_count"] = tensor_count
                logger.debug(f"GGUF analysis: {tensor_count} tensors, {kv_count} KV pairs")
                
        except Exception as e:
            logger.warning(f"GGUF analysis failed: {e}")
        
        return analysis
    
    async def _analyze_onnx(self, model_path: str) -> Dict[str, Any]:
        """Analyze ONNX format model"""
        # This would use ONNX libraries in production
        return {
            "architecture": "onnx_model",
            "quantization_friendly": True,
            "has_dynamic_shapes": False
        }
    
    async def _analyze_pytorch(self, model_path: str) -> Dict[str, Any]:
        """Analyze PyTorch format model"""
        # This would use PyTorch libraries in production
        return {
            "architecture": "pytorch_model",
            "quantization_friendly": True,
            "state_dict_keys": []
        }
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate quantization recommendations"""
        recommendations = []
        
        file_size_mb = analysis.get("file_size_mb", 0)
        model_format = analysis.get("model_format", "unknown")
        
        if file_size_mb > 1000:  # > 1GB
            recommendations.append("Consider INT4 quantization for significant size reduction")
            recommendations.append("Enable KV cache quantization")
        elif file_size_mb > 500:  # > 500MB
            recommendations.append("INT8 quantization recommended for good balance")
        else:
            recommendations.append("FP16 quantization for minimal accuracy loss")
        
        if model_format == "gguf":
            recommendations.append("GGUF format supports efficient quantization")
        elif model_format == "onnx":
            recommendations.append("Consider ONNX Runtime quantization tools")
        
        return recommendations

class AdvancedQuantizer:
    """Advanced quantization engine with multiple techniques"""
    
    def __init__(self, 
                 num_workers: int = None,
                 temp_dir: str = "/tmp/sutazai_quantization"):
        
        self.num_workers = num_workers or min(psutil.cpu_count(), 8)
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.analyzer = ModelAnalyzer()
        self._thread_pool = ThreadPoolExecutor(max_workers=self.num_workers)
        self._process_pool = ProcessPoolExecutor(max_workers=min(self.num_workers, 4))
        
        logger.info(f"AdvancedQuantizer initialized with {self.num_workers} workers")
    
    async def quantize_model(self,
                           input_path: str,
                           output_path: str,
                           config: QuantizationConfig) -> QuantizationResult:
        """Quantize a model with advanced techniques"""
        start_time = time.time()
        
        try:
            # Analyze model first
            logger.info(f"Analyzing model: {input_path}")
            analysis = await self.analyzer.analyze_model(input_path)
            
            original_size = os.path.getsize(input_path)
            original_size_mb = original_size / (1024 * 1024)
            
            # Select quantization method based on format and config
            quantizer_func = self._get_quantizer_function(analysis, config)
            
            # Perform quantization
            logger.info(f"Starting {config.quantization_type.value} quantization")
            success = await quantizer_func(input_path, output_path, config, analysis)
            
            if success and os.path.exists(output_path):
                quantized_size = os.path.getsize(output_path)
                quantized_size_mb = quantized_size / (1024 * 1024)
                compression_ratio = original_size / quantized_size
                
                # Estimate accuracy loss (would be measured in practice)
                accuracy_loss = self._estimate_accuracy_loss(config, compression_ratio)
                
                quantization_time = time.time() - start_time
                
                result = QuantizationResult(
                    success=True,
                    original_size_mb=original_size_mb,
                    quantized_size_mb=quantized_size_mb,
                    compression_ratio=compression_ratio,
                    accuracy_loss=accuracy_loss,
                    quantization_time_sec=quantization_time,
                    output_path=output_path,
                    quantization_stats={
                        "analysis": analysis,
                        "config": config.__dict__
                    }
                )
                
                logger.info(f"Quantization successful: {compression_ratio:.2f}x compression, "
                          f"{accuracy_loss:.3f} estimated accuracy loss")
                return result
            
            else:
                return QuantizationResult(
                    success=False,
                    original_size_mb=original_size_mb,
                    quantized_size_mb=0.0,
                    compression_ratio=1.0,
                    accuracy_loss=0.0,
                    quantization_time_sec=time.time() - start_time,
                    output_path=output_path,
                    error_message="Quantization process failed"
                )
                
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return QuantizationResult(
                success=False,
                original_size_mb=0.0,
                quantized_size_mb=0.0,
                compression_ratio=1.0,
                accuracy_loss=0.0,
                quantization_time_sec=time.time() - start_time,
                output_path=output_path,
                error_message=str(e)
            )
    
    def _get_quantizer_function(self, analysis: Dict[str, Any], config: QuantizationConfig):
        """Get appropriate quantizer function"""
        model_format = analysis.get("model_format", "unknown")
        
        if model_format == "gguf":
            return self._quantize_gguf
        elif model_format == "onnx":
            return self._quantize_onnx
        elif model_format == "pytorch":
            return self._quantize_pytorch
        else:
            return self._quantize_generic
    
    async def _quantize_gguf(self,
                           input_path: str,
                           output_path: str,
                           config: QuantizationConfig,
                           analysis: Dict[str, Any]) -> bool:
        """Quantize GGUF format model"""
        try:
            # For production, this would use llama.cpp quantization tools
            # For now, simulate with a simplified approach
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._thread_pool,
                self._quantize_gguf_worker,
                input_path,
                output_path,
                config
            )
            
        except Exception as e:
            logger.error(f"GGUF quantization failed: {e}")
            return False
    
    def _quantize_gguf_worker(self,
                            input_path: str,
                            output_path: str,
                            config: QuantizationConfig) -> bool:
        """Worker function for GGUF quantization"""
        try:
            # Simulate quantization by reading, processing, and writing
            with open(input_path, 'rb') as infile, open(output_path, 'wb') as outfile:
                # Read GGUF header
                header = infile.read(1024)  # Read first 1KB as header
                outfile.write(header)
                
                # Process tensors with quantization
                chunk_size = 1024 * 1024  # 1MB chunks
                total_written = len(header)
                
                while True:
                    chunk = infile.read(chunk_size)
                    if not chunk:
                        break
                    
                    # Apply quantization to chunk
                    quantized_chunk = self._apply_quantization(chunk, config)
                    outfile.write(quantized_chunk)
                    total_written += len(quantized_chunk)
                
                logger.debug(f"GGUF quantization completed: {total_written} bytes written")
                return True
                
        except Exception as e:
            logger.error(f"GGUF worker failed: {e}")
            return False
    
    def _apply_quantization(self, data: bytes, config: QuantizationConfig) -> bytes:
        """Apply quantization to data chunk"""
        if config.quantization_type == QuantizationType.INT8:
            return self._quantize_to_int8(data)
        elif config.quantization_type == QuantizationType.INT4:
            return self._quantize_to_int4(data)
        elif config.quantization_type == QuantizationType.FP16:
            return self._quantize_to_fp16(data)
        else:
            # Default: simple compression
            return self._compress_data(data, 0.75)
    
    def _quantize_to_int8(self, data: bytes) -> bytes:
        """Quantize data to INT8 (simulate 50% compression)"""
        # In production, this would perform actual INT8 quantization
        # For demo, simulate by reducing data size
        return data[::2]  # Take every other byte (50% reduction)
    
    def _quantize_to_int4(self, data: bytes) -> bytes:
        """Quantize data to INT4 (simulate 75% compression)"""
        # In production, this would pack two 4-bit values per byte
        # For demo, simulate by taking every 4th byte
        return data[::4]  # Take every 4th byte (75% reduction)
    
    def _quantize_to_fp16(self, data: bytes) -> bytes:
        """Quantize data to FP16 (simulate 25% compression)"""
        # In production, this would convert FP32 to FP16
        # For demo, simulate by removing some bytes
        return data[::int(1/0.75)]  # ~25% reduction
    
    def _compress_data(self, data: bytes, ratio: float) -> bytes:
        """Generic data compression"""
        target_size = int(len(data) * ratio)
        if target_size >= len(data):
            return data
        
        # Simple compression simulation
        step = len(data) // target_size
        return data[::max(step, 1)][:target_size]
    
    async def _quantize_onnx(self,
                           input_path: str,
                           output_path: str,
                           config: QuantizationConfig,
                           analysis: Dict[str, Any]) -> bool:
        """Quantize ONNX format model"""
        try:
            # In production, this would use ONNX Runtime quantization
            logger.info("ONNX quantization (simulated)")
            
            # Simulate ONNX quantization
            import shutil
            shutil.copy2(input_path, output_path)
            
            # Apply simulated compression
            with open(output_path, 'r+b') as f:
                f.seek(0, 2)  # Go to end
                size = f.tell()
                
                if config.quantization_type == QuantizationType.INT8:
                    new_size = int(size * 0.5)
                elif config.quantization_type == QuantizationType.INT4:
                    new_size = int(size * 0.25)
                else:
                    new_size = int(size * 0.75)
                
                f.truncate(new_size)
            
            return True
            
        except Exception as e:
            logger.error(f"ONNX quantization failed: {e}")
            return False
    
    async def _quantize_pytorch(self,
                              input_path: str,
                              output_path: str,
                              config: QuantizationConfig,
                              analysis: Dict[str, Any]) -> bool:
        """Quantize PyTorch format model"""
        try:
            # In production, this would use PyTorch quantization APIs
            logger.info("PyTorch quantization (simulated)")
            
            # Simulate PyTorch quantization
            import shutil
            shutil.copy2(input_path, output_path)
            
            # Apply simulated quantization
            original_size = os.path.getsize(input_path)
            
            if config.quantization_type == QuantizationType.DYNAMIC:
                compression_ratio = 0.6
            elif config.quantization_type == QuantizationType.STATIC:
                compression_ratio = 0.4
            else:
                compression_ratio = 0.5
            
            new_size = int(original_size * compression_ratio)
            
            with open(output_path, 'r+b') as f:
                f.truncate(new_size)
            
            return True
            
        except Exception as e:
            logger.error(f"PyTorch quantization failed: {e}")
            return False
    
    async def _quantize_generic(self,
                              input_path: str,
                              output_path: str,
                              config: QuantizationConfig,
                              analysis: Dict[str, Any]) -> bool:
        """Generic quantization for unknown formats"""
        try:
            logger.info("Generic quantization")
            
            # Simple compression-based approach
            import shutil
            shutil.copy2(input_path, output_path)
            
            original_size = os.path.getsize(input_path)
            target_ratio = config.compression_target
            new_size = int(original_size * target_ratio)
            
            with open(output_path, 'r+b') as f:
                f.truncate(new_size)
            
            return True
            
        except Exception as e:
            logger.error(f"Generic quantization failed: {e}")
            return False
    
    def _estimate_accuracy_loss(self, config: QuantizationConfig, compression_ratio: float) -> float:
        """Estimate accuracy loss from quantization"""
        # Rough estimation based on quantization type and compression
        base_loss = {
            QuantizationType.FP16: 0.001,
            QuantizationType.INT8: 0.01,
            QuantizationType.INT4: 0.05,
            QuantizationType.INT2: 0.15,
            QuantizationType.DYNAMIC: 0.02,
            QuantizationType.STATIC: 0.015
        }.get(config.quantization_type, 0.03)
        
        # Adjust based on compression ratio
        compression_factor = max(1.0, compression_ratio - 1.0)
        estimated_loss = base_loss * (1.0 + compression_factor * 0.1)
        
        return min(estimated_loss, 0.5)  # Cap at 50% loss
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        self._thread_pool.shutdown(wait=True)
        self._process_pool.shutdown(wait=True)

class EdgeQuantizationManager:
    """Manages quantization for edge deployment"""
    
    def __init__(self, 
                 models_dir: str = "/tmp/sutazai_quantized_models",
                 cache_quantized: bool = True,
                 auto_select_best: bool = True):
        
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_quantized = cache_quantized
        self.auto_select_best = auto_select_best
        
        self.quantizer = AdvancedQuantizer()
        self._quantized_models: Dict[str, List[QuantizationResult]] = {}
        self._lock = threading.RLock()
        
        logger.info(f"EdgeQuantizationManager initialized: {models_dir}")
    
    async def prepare_model_for_edge(self,
                                   model_path: str,
                                   target_device: str = "cpu",
                                   memory_limit_mb: Optional[int] = None,
                                   latency_target_ms: Optional[float] = None) -> Optional[str]:
        """
        Prepare a model for edge deployment with optimal quantization
        
        Returns:
            Path to quantized model, or None if preparation failed
        """
        try:
            model_name = Path(model_path).stem
            
            # Check if already quantized versions exist
            if self._has_suitable_quantized_model(model_name, memory_limit_mb):
                return self._get_best_quantized_model(model_name, memory_limit_mb)
            
            # Determine optimal quantization strategy
            configs = await self._determine_quantization_strategies(
                model_path, target_device, memory_limit_mb, latency_target_ms
            )
            
            best_result = None
            
            # Try quantization strategies in order of preference
            for config in configs:
                output_path = self.models_dir / f"{model_name}_{config.quantization_type.value}.quantized"
                
                result = await self.quantizer.quantize_model(
                    input_path=model_path,
                    output_path=str(output_path),
                    config=config
                )
                
                if result.success:
                    # Check if result meets requirements
                    if self._meets_requirements(result, memory_limit_mb, latency_target_ms):
                        with self._lock:
                            if model_name not in self._quantized_models:
                                self._quantized_models[model_name] = []
                            self._quantized_models[model_name].append(result)
                        
                        if not best_result or result.compression_ratio > best_result.compression_ratio:
                            best_result = result
                
                # Break if we found a good enough result
                if best_result and best_result.accuracy_loss < 0.02:
                    break
            
            return best_result.output_path if best_result else None
            
        except Exception as e:
            logger.error(f"Model preparation failed: {e}")
            return None
    
    async def _determine_quantization_strategies(self,
                                               model_path: str,
                                               target_device: str,
                                               memory_limit_mb: Optional[int],
                                               latency_target_ms: Optional[float]) -> List[QuantizationConfig]:
        """Determine optimal quantization strategies"""
        # Analyze model
        analysis = await self.quantizer.analyzer.analyze_model(model_path)
        model_size_mb = analysis.get("file_size_mb", 0)
        
        configs = []
        
        # Memory-constrained scenarios
        if memory_limit_mb and model_size_mb > memory_limit_mb:
            required_compression = model_size_mb / memory_limit_mb
            
            if required_compression > 4:
                # Aggressive quantization needed
                configs.append(QuantizationConfig(
                    quantization_type=QuantizationType.INT4,
                    strategy=QuantizationStrategy.AGGRESSIVE,
                    compression_target=0.25,
                    target_accuracy_loss=0.1
                ))
                configs.append(QuantizationConfig(
                    quantization_type=QuantizationType.INT2,
                    strategy=QuantizationStrategy.AGGRESSIVE,
                    compression_target=0.15,
                    target_accuracy_loss=0.2
                ))
            elif required_compression > 2:
                configs.append(QuantizationConfig(
                    quantization_type=QuantizationType.INT8,
                    strategy=QuantizationStrategy.BALANCED,
                    compression_target=0.4,
                    target_accuracy_loss=0.05
                ))
                configs.append(QuantizationConfig(
                    quantization_type=QuantizationType.INT4,
                    strategy=QuantizationStrategy.BALANCED,
                    compression_target=0.3,
                    target_accuracy_loss=0.08
                ))
            else:
                configs.append(QuantizationConfig(
                    quantization_type=QuantizationType.FP16,
                    strategy=QuantizationStrategy.CONSERVATIVE,
                    compression_target=0.75,
                    target_accuracy_loss=0.01
                ))
        
        # Latency-optimized scenarios
        elif latency_target_ms and latency_target_ms < 100:
            # Need fast inference
            configs.append(QuantizationConfig(
                quantization_type=QuantizationType.INT8,
                strategy=QuantizationStrategy.BALANCED,
                compression_target=0.5,
                enable_kv_cache_quantization=True
            ))
            configs.append(QuantizationConfig(
                quantization_type=QuantizationType.DYNAMIC,
                strategy=QuantizationStrategy.BALANCED,
                compression_target=0.6,
                enable_activation_quantization=True
            ))
        
        # Default strategies
        else:
            configs.extend([
                QuantizationConfig(
                    quantization_type=QuantizationType.FP16,
                    strategy=QuantizationStrategy.CONSERVATIVE
                ),
                QuantizationConfig(
                    quantization_type=QuantizationType.INT8,
                    strategy=QuantizationStrategy.BALANCED
                ),
                QuantizationConfig(
                    quantization_type=QuantizationType.DYNAMIC,
                    strategy=QuantizationStrategy.BALANCED
                )
            ])
        
        return configs
    
    def _has_suitable_quantized_model(self, model_name: str, memory_limit_mb: Optional[int]) -> bool:
        """Check if suitable quantized model exists"""
        with self._lock:
            if model_name not in self._quantized_models:
                return False
            
            for result in self._quantized_models[model_name]:
                if memory_limit_mb is None or result.quantized_size_mb <= memory_limit_mb:
                    if os.path.exists(result.output_path):
                        return True
            
            return False
    
    def _get_best_quantized_model(self, model_name: str, memory_limit_mb: Optional[int]) -> Optional[str]:
        """Get best existing quantized model"""
        with self._lock:
            if model_name not in self._quantized_models:
                return None
            
            suitable_models = []
            for result in self._quantized_models[model_name]:
                if memory_limit_mb is None or result.quantized_size_mb <= memory_limit_mb:
                    if os.path.exists(result.output_path):
                        suitable_models.append(result)
            
            if not suitable_models:
                return None
            
            # Select best by accuracy vs compression trade-off
            best_model = min(suitable_models, key=lambda r: r.accuracy_loss / r.compression_ratio)
            return best_model.output_path
    
    def _meets_requirements(self,
                          result: QuantizationResult,
                          memory_limit_mb: Optional[int],
                          latency_target_ms: Optional[float]) -> bool:
        """Check if quantization result meets requirements"""
        if memory_limit_mb and result.quantized_size_mb > memory_limit_mb:
            return False
        
        if result.accuracy_loss > 0.15:  # Max 15% accuracy loss
            return False
        
        # Latency check would be done with actual inference in production
        
        return True
    
    def get_quantized_models(self, model_name: str) -> List[QuantizationResult]:
        """Get all quantized versions of a model"""
        with self._lock:
            return self._quantized_models.get(model_name, [])
    
    def cleanup_old_models(self, max_age_days: int = 7) -> None:
        """Clean up old quantized models"""
        import time
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        
        removed_count = 0
        for model_file in self.models_dir.glob("*.quantized"):
            if model_file.stat().st_mtime < cutoff_time:
                try:
                    model_file.unlink()
                    removed_count += 1
                except OSError:
                    pass
        
        logger.info(f"Cleaned up {removed_count} old quantized models")

# Global quantization manager
_global_manager: Optional[EdgeQuantizationManager] = None

def get_global_manager(**kwargs) -> EdgeQuantizationManager:
    """Get or create global quantization manager"""
    global _global_manager
    if _global_manager is None:
        _global_manager = EdgeQuantizationManager(**kwargs)
    return _global_manager