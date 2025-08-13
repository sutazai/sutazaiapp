"""
Advanced Quantization Strategies for SutazAI
Implements various quantization techniques for efficient CPU deployment
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
import struct
import asyncio
import aiohttp
from enum import Enum
import time

logger = logging.getLogger(__name__)

class QuantizationType(Enum):
    """Types of quantization available"""
    INT8 = "int8"
    INT4 = "int4"
    INT2 = "int2"
    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "quantization_aware_training"  # Quantization Aware Training
    PTQ = "post_training_quantization"   # Post Training Quantization

@dataclass
class QuantizationConfig:
    """Configuration for quantization strategies"""
    quantization_type: QuantizationType = QuantizationType.INT8
    
    # Quantization parameters
    bits: int = 8
    symmetric: bool = True
    per_channel: bool = True
    reduce_range: bool = False
    
    # Calibration settings
    calibration_method: str = "minmax"  # minmax, entropy, percentile
    calibration_data_size: int = 100
    percentile_value: float = 99.9
    
    # QAT specific settings
    qat_epochs: int = 5
    qat_learning_rate: float = 1e-5
    fake_quantize: bool = True
    
    # Advanced options
    weight_only: bool = False  # Quantize only weights, not activations
    mixed_precision: bool = False  # Use different precisions for different layers
    block_wise: bool = True  # Apply quantization block-wise
    group_size: int = 128  # Group size for block-wise quantization
    
    # Optimization settings
    optimize_for_cpu: bool = True
    use_vnni: bool = True  # Vector Neural Network Instructions (Intel)
    use_neon: bool = True  # ARM NEON optimizations
    
    # Quality preservation
    preserve_accuracy: bool = True
    accuracy_threshold: float = 0.05  # Maximum accuracy drop allowed
    sensitive_layers: List[str] = None  # Layers to skip quantization
    
    def __post_init__(self):
        if self.sensitive_layers is None:
            self.sensitive_layers = ['embedding', 'lm_head', 'output']

class QuantizationCalibrator:
    """Handles calibration for quantization"""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.calibration_data = []
        self.statistics = {}
    
    def collect_statistics(self, model: nn.Module, data_loader) -> Dict[str, Any]:
        """Collect statistics for quantization calibration"""
        logger.info("Collecting calibration statistics...")
        
        model.eval()
        self.statistics = {}
        
        # Hook to collect activations
        hooks = []
        
        def create_hook(name):
            def hook(module, input, output):
                if name not in self.statistics:
                    self.statistics[name] = {
                        'activations': [],
                        'weights': []
                    }
                
                # Collect activation statistics
                if isinstance(output, torch.Tensor):
                    self.statistics[name]['activations'].append(output.detach().cpu())
                
                # Collect weight statistics
                if hasattr(module, 'weight') and module.weight is not None:
                    self.statistics[name]['weights'].append(module.weight.detach().cpu())
            
            return hook
        
        # Register hooks
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hook = module.register_forward_hook(create_hook(name))
                hooks.append(hook)
        
        # Run calibration
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= self.config.calibration_data_size:
                    break
                
                # Forward pass to collect statistics
                if isinstance(batch, dict):
                    # Handle different input formats
                    inputs = batch.get('input_ids', batch.get('inputs', None))
                else:
                    inputs = batch
                
                if inputs is not None:
                    try:
                        _ = model(inputs)
                    except Exception as e:
                        logger.warning(f"Calibration forward pass failed: {e}")
                        continue
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Process statistics
        self._process_statistics()
        
        logger.info(f"Calibration completed with {len(self.statistics)} layers")
        return self.statistics
    
    def _process_statistics(self):
        """Process collected statistics to compute quantization parameters"""
        for layer_name, stats in self.statistics.items():
            # Process activations
            if stats['activations']:
                activations = torch.cat(stats['activations'], dim=0)
                stats['activation_min'] = activations.min().item()
                stats['activation_max'] = activations.max().item()
                stats['activation_mean'] = activations.mean().item()
                stats['activation_std'] = activations.std().item()
                
                # Calculate quantization parameters
                if self.config.calibration_method == "minmax":
                    stats['activation_scale'], stats['activation_zero_point'] = self._compute_minmax_params(
                        activations, self.config.bits
                    )
                elif self.config.calibration_method == "percentile":
                    stats['activation_scale'], stats['activation_zero_point'] = self._compute_percentile_params(
                        activations, self.config.bits, self.config.percentile_value
                    )
            
            # Process weights
            if stats['weights']:
                weights = torch.cat([w.flatten() for w in stats['weights']], dim=0)
                stats['weight_min'] = weights.min().item()
                stats['weight_max'] = weights.max().item()
                stats['weight_mean'] = weights.mean().item()
                stats['weight_std'] = weights.std().item()
                
                # Calculate weight quantization parameters
                stats['weight_scale'], stats['weight_zero_point'] = self._compute_minmax_params(
                    weights, self.config.bits
                )
    
    def _compute_minmax_params(self, tensor: torch.Tensor, bits: int) -> Tuple[float, int]:
        """Compute quantization scale and zero point using min-max method"""
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        if self.config.symmetric:
            # Symmetric quantization
            abs_max = max(abs(min_val), abs(max_val))
            qmin = -(2 ** (bits - 1))
            qmax = 2 ** (bits - 1) - 1
            scale = abs_max / max(abs(qmin), abs(qmax))
            zero_point = 0
        else:
            # Asymmetric quantization
            qmin = 0 if bits == 8 else -(2 ** (bits - 1))
            qmax = 2 ** bits - 1 if bits == 8 else 2 ** (bits - 1) - 1
            
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = qmin - min_val / scale
            zero_point = int(round(zero_point))
            zero_point = max(qmin, min(qmax, zero_point))
        
        return scale, zero_point
    
    def _compute_percentile_params(self, tensor: torch.Tensor, bits: int, 
                                 percentile: float) -> Tuple[float, int]:
        """Compute quantization parameters using percentile method"""
        # Use percentile to clip outliers
        lower_percentile = (100 - percentile) / 2
        upper_percentile = 100 - lower_percentile
        
        min_val = torch.quantile(tensor, lower_percentile / 100).item()
        max_val = torch.quantile(tensor, upper_percentile / 100).item()
        
        # Apply min-max quantization to clipped range
        tensor_clipped = torch.clamp(tensor, min_val, max_val)
        return self._compute_minmax_params(tensor_clipped, bits)

class QuantizationStrategy(ABC):
    """Abstract base class for quantization strategies"""
    
    @abstractmethod
    def quantize(self, model: nn.Module, config: QuantizationConfig) -> nn.Module:
        """Apply quantization to the model"""
        pass
    
    @abstractmethod
    def estimate_speedup(self, model: nn.Module) -> float:
        """Estimate inference speedup from quantization"""
        pass

class PostTrainingQuantization(QuantizationStrategy):
    """Post-training quantization implementation"""
    
    def __init__(self):
        self.calibrator = None
        self.quantized_layers = {}
    
    def quantize(self, model: nn.Module, config: QuantizationConfig) -> nn.Module:
        """Apply post-training quantization"""
        logger.info(f"Applying {config.bits}-bit post-training quantization...")
        
        self.calibrator = QuantizationCalibrator(config)
        
        # Apply quantization to each layer
        for name, module in model.named_modules():
            if self._should_quantize_layer(name, module, config):
                quantized_module = self._quantize_layer(module, name, config)
                self._replace_module(model, name, quantized_module)
                self.quantized_layers[name] = quantized_module
        
        logger.info(f"Quantized {len(self.quantized_layers)} layers")
        return model
    
    def _should_quantize_layer(self, name: str, module: nn.Module, 
                             config: QuantizationConfig) -> bool:
        """Determine if a layer should be quantized"""
        # Skip sensitive layers
        for sensitive in config.sensitive_layers:
            if sensitive in name.lower():
                return False
        
        # Only quantize linear and conv layers
        return isinstance(module, (nn.Linear, nn.Conv2d))
    
    def _quantize_layer(self, module: nn.Module, name: str, 
                       config: QuantizationConfig) -> nn.Module:
        """Quantize a single layer"""
        if isinstance(module, nn.Linear):
            return self._quantize_linear(module, name, config)
        elif isinstance(module, nn.Conv2d):
            return self._quantize_conv2d(module, name, config)
        else:
            return module
    
    def _quantize_linear(self, layer: nn.Linear, name: str, 
                        config: QuantizationConfig) -> nn.Module:
        """Quantize a linear layer"""
        # Create quantized linear layer
        if config.bits == 8:
            quantized_layer = QuantizedLinear8bit(
                layer.in_features,
                layer.out_features,
                bias=layer.bias is not None,
                config=config
            )
        elif config.bits == 4:
            quantized_layer = QuantizedLinear4bit(
                layer.in_features,
                layer.out_features,
                bias=layer.bias is not None,
                config=config
            )
        else:
            # Fallback to 8-bit for unsupported bit widths
            quantized_layer = QuantizedLinear8bit(
                layer.in_features,
                layer.out_features,
                bias=layer.bias is not None,
                config=config
            )
        
        # Quantize weights
        quantized_layer.quantize_weights(layer.weight.data, layer.bias.data if layer.bias is not None else None)
        
        return quantized_layer
    
    def _quantize_conv2d(self, layer: nn.Conv2d, name: str, 
                        config: QuantizationConfig) -> nn.Module:
        """Quantize a convolutional layer"""
        # Create quantized conv layer
        quantized_layer = QuantizedConv2d8bit(
            layer.in_channels,
            layer.out_channels,
            layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None,
            config=config
        )
        
        # Quantize weights
        quantized_layer.quantize_weights(layer.weight.data, layer.bias.data if layer.bias is not None else None)
        
        return quantized_layer
    
    def _replace_module(self, model: nn.Module, name: str, new_module: nn.Module):
        """Replace a module in the model"""
        # Split the name to get parent and child
        parts = name.split('.')
        parent = model
        
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, parts[-1], new_module)
    
    def estimate_speedup(self, model: nn.Module) -> float:
        """Estimate speedup from quantization"""
        # Rough estimates based on bit width
        total_params = sum(p.numel() for p in model.parameters())
        quantized_params = sum(p.numel() for name, p in model.named_parameters() 
                             if any(q_name in name for q_name in self.quantized_layers.keys()))
        
        if total_params == 0:
            return 1.0
        
        quantization_ratio = quantized_params / total_params
        
        # Estimated speedup factors
        speedup_factor = 1.0 + quantization_ratio * 1.5  # Conservative estimate
        
        return speedup_factor

class QuantizedLinear8bit(nn.Module):
    """8-bit quantized linear layer"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 config: QuantizationConfig = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantizationConfig()
        
        # Quantized weight storage
        self.register_buffer('weight_quantized', torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer('weight_scale', torch.ones(out_features if config.per_channel else 1))
        self.register_buffer('weight_zero_point', torch.zeros(out_features if config.per_channel else 1, dtype=torch.int8))
        
        if bias:
            self.register_buffer('bias_quantized', torch.zeros(out_features, dtype=torch.int32))
            self.register_buffer('bias_scale', torch.ones(1))
        else:
            self.bias_quantized = None
            self.bias_scale = None
    
    def quantize_weights(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Quantize and store weights"""
        if self.config.per_channel:
            # Per-channel quantization
            scales = []
            zero_points = []
            quantized_weights = []
            
            for i in range(weight.size(0)):
                channel_weight = weight[i]
                scale, zero_point = self._compute_quantization_params(channel_weight)
                quantized_weight = self._quantize_tensor(channel_weight, scale, zero_point)
                
                scales.append(scale)
                zero_points.append(zero_point)
                quantized_weights.append(quantized_weight)
            
            self.weight_scale = torch.tensor(scales)
            self.weight_zero_point = torch.tensor(zero_points, dtype=torch.int8)
            self.weight_quantized = torch.stack(quantized_weights)
        else:
            # Per-tensor quantization
            scale, zero_point = self._compute_quantization_params(weight)
            self.weight_scale = torch.tensor([scale])
            self.weight_zero_point = torch.tensor([zero_point], dtype=torch.int8)
            self.weight_quantized = self._quantize_tensor(weight, scale, zero_point)
        
        # Quantize bias if present
        if bias is not None and self.bias_quantized is not None:
            # Bias uses higher precision (int32)
            bias_scale = self.weight_scale.mean() * 0.1  # Scale bias appropriately
            self.bias_scale = torch.tensor([bias_scale])
            self.bias_quantized = (bias / bias_scale).round().clamp(-2**31, 2**31-1).to(torch.int32)
    
    def _compute_quantization_params(self, tensor: torch.Tensor) -> Tuple[float, int]:
        """Compute quantization scale and zero point"""
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        if self.config.symmetric:
            abs_max = max(abs(min_val), abs(max_val))
            scale = abs_max / 127.0  # int8 range: -128 to 127
            zero_point = 0
        else:
            scale = (max_val - min_val) / 255.0  # uint8 range: 0 to 255
            zero_point = int(round(-min_val / scale))
            zero_point = max(0, min(255, zero_point))
        
        return scale, zero_point
    
    def _quantize_tensor(self, tensor: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
        """Quantize a tensor"""
        quantized = (tensor / scale + zero_point).round().clamp(-128, 127)
        return quantized.to(torch.int8)
    
    def _dequantize_tensor(self, quantized: torch.Tensor, scale: torch.Tensor, 
                          zero_point: torch.Tensor) -> torch.Tensor:
        """Dequantize a tensor"""
        return scale * (quantized.float() - zero_point.float())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized computation"""
        # Dequantize weights for computation
        if self.config.per_channel:
            weight_dequantized = self._dequantize_tensor(
                self.weight_quantized, 
                self.weight_scale.unsqueeze(1), 
                self.weight_zero_point.unsqueeze(1)
            )
        else:
            weight_dequantized = self._dequantize_tensor(
                self.weight_quantized, 
                self.weight_scale, 
                self.weight_zero_point
            )
        
        # Compute linear transformation
        output = F.linear(x, weight_dequantized)
        
        # Add bias if present
        if self.bias_quantized is not None:
            bias_dequantized = self.bias_scale * self.bias_quantized.float()
            output = output + bias_dequantized
        
        return output

class QuantizedLinear4bit(nn.Module):
    """4-bit quantized linear layer with block-wise quantization"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 config: QuantizationConfig = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantizationConfig()
        self.group_size = min(config.group_size, in_features)
        
        # Calculate number of groups
        self.num_groups = (in_features + self.group_size - 1) // self.group_size
        
        # 4-bit weights packed into uint8 (2 weights per byte)
        weight_elements = (out_features * in_features + 1) // 2
        self.register_buffer('weight_quantized', torch.zeros(weight_elements, dtype=torch.uint8))
        
        # Scales and zero points per group
        self.register_buffer('weight_scales', torch.ones((out_features, self.num_groups)))
        self.register_buffer('weight_zero_points', torch.zeros((out_features, self.num_groups), dtype=torch.uint8))
        
        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.bias = None
    
    def quantize_weights(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Quantize weights using 4-bit block-wise quantization"""
        quantized_data = []
        scales = []
        zero_points = []
        
        for out_idx in range(self.out_features):
            channel_scales = []
            channel_zero_points = []
            channel_quantized = []
            
            for group_idx in range(self.num_groups):
                start_idx = group_idx * self.group_size
                end_idx = min(start_idx + self.group_size, self.in_features)
                
                group_weight = weight[out_idx, start_idx:end_idx]
                
                # Compute quantization parameters for this group
                min_val = group_weight.min().item()
                max_val = group_weight.max().item()
                
                scale = (max_val - min_val) / 15.0  # 4-bit range: 0 to 15
                zero_point = int(round(-min_val / scale)) if scale > 0 else 0
                zero_point = max(0, min(15, zero_point))
                
                # Quantize group
                if scale > 0:
                    group_quantized = ((group_weight - min_val) / scale).round().clamp(0, 15)
                else:
                    group_quantized = torch.zeros_like(group_weight)
                
                channel_scales.append(scale)
                channel_zero_points.append(zero_point)
                channel_quantized.append(group_quantized)
            
            scales.append(channel_scales)
            zero_points.append(channel_zero_points)
            
            # Concatenate quantized weights for this channel
            channel_data = torch.cat(channel_quantized)
            quantized_data.append(channel_data)
        
        # Pack 4-bit values into uint8
        all_quantized = torch.cat(quantized_data)
        packed_weights = self._pack_4bit_weights(all_quantized)
        
        self.weight_quantized = packed_weights
        self.weight_scales = torch.tensor(scales)
        self.weight_zero_points = torch.tensor(zero_points, dtype=torch.uint8)
        
        if bias is not None:
            self.bias = bias.clone()
    
    def _pack_4bit_weights(self, weights_4bit: torch.Tensor) -> torch.Tensor:
        """Pack 4-bit weights into uint8 array"""
        # Pad to even length
        if weights_4bit.numel() % 2 == 1:
            weights_4bit = torch.cat([weights_4bit, torch.zeros(1)])
        
        # Reshape and pack
        weights_reshaped = weights_4bit.view(-1, 2)
        packed = (weights_reshaped[:, 0] + weights_reshaped[:, 1] * 16).to(torch.uint8)
        
        return packed
    
    def _unpack_4bit_weights(self, packed_weights: torch.Tensor) -> torch.Tensor:
        """Unpack uint8 array to 4-bit weights"""
        low_bits = packed_weights % 16
        high_bits = packed_weights // 16
        
        unpacked = torch.stack([low_bits, high_bits], dim=1).view(-1)
        
        # Trim to original size
        expected_size = self.out_features * self.in_features
        return unpacked[:expected_size].float()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with 4-bit quantized computation"""
        # Unpack and dequantize weights
        unpacked_weights = self._unpack_4bit_weights(self.weight_quantized)
        weight_dequantized = torch.zeros((self.out_features, self.in_features), device=x.device)
        
        for out_idx in range(self.out_features):
            for group_idx in range(self.num_groups):
                start_idx = group_idx * self.group_size
                end_idx = min(start_idx + self.group_size, self.in_features)
                
                # Get quantized values for this group
                weight_start = out_idx * self.in_features + start_idx
                weight_end = out_idx * self.in_features + end_idx
                group_quantized = unpacked_weights[weight_start:weight_end]
                
                # Dequantize
                scale = self.weight_scales[out_idx, group_idx]
                zero_point = self.weight_zero_points[out_idx, group_idx]
                
                group_dequantized = scale * (group_quantized - zero_point)
                weight_dequantized[out_idx, start_idx:end_idx] = group_dequantized
        
        # Compute linear transformation
        output = F.linear(x, weight_dequantized, self.bias)
        
        return output

class QuantizedConv2d8bit(nn.Module):
    """8-bit quantized 2D convolution layer"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias: bool = True,
                 config: QuantizationConfig = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.config = config or QuantizationConfig()
        
        # Quantized weight storage
        weight_shape = (out_channels, in_channels // groups, *self.kernel_size)
        self.register_buffer('weight_quantized', torch.zeros(weight_shape, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.ones(out_channels if config.per_channel else 1))
        self.register_buffer('weight_zero_point', torch.zeros(out_channels if config.per_channel else 1, dtype=torch.int8))
        
        if bias:
            self.register_buffer('bias_quantized', torch.zeros(out_channels, dtype=torch.int32))
            self.register_buffer('bias_scale', torch.ones(1))
        else:
            self.bias_quantized = None
            self.bias_scale = None
    
    def quantize_weights(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Quantize convolutional weights"""
        if self.config.per_channel:
            # Per-channel quantization (per output channel)
            scales = []
            zero_points = []
            quantized_weights = []
            
            for i in range(weight.size(0)):
                channel_weight = weight[i]
                min_val = channel_weight.min().item()
                max_val = channel_weight.max().item()
                
                if self.config.symmetric:
                    abs_max = max(abs(min_val), abs(max_val))
                    scale = abs_max / 127.0
                    zero_point = 0
                else:
                    scale = (max_val - min_val) / 255.0
                    zero_point = int(round(-min_val / scale))
                    zero_point = max(0, min(255, zero_point))
                
                quantized_weight = ((channel_weight - min_val) / scale).round().clamp(-128, 127).to(torch.int8)
                
                scales.append(scale)
                zero_points.append(zero_point)
                quantized_weights.append(quantized_weight)
            
            self.weight_scale = torch.tensor(scales)
            self.weight_zero_point = torch.tensor(zero_points, dtype=torch.int8)
            self.weight_quantized = torch.stack(quantized_weights)
        else:
            # Per-tensor quantization
            min_val = weight.min().item()
            max_val = weight.max().item()
            
            scale = (max_val - min_val) / 255.0
            zero_point = int(round(-min_val / scale))
            
            self.weight_scale = torch.tensor([scale])
            self.weight_zero_point = torch.tensor([zero_point], dtype=torch.int8)
            self.weight_quantized = ((weight - min_val) / scale).round().clamp(-128, 127).to(torch.int8)
        
        # Quantize bias
        if bias is not None and self.bias_quantized is not None:
            bias_scale = self.weight_scale.mean() * 0.1
            self.bias_scale = torch.tensor([bias_scale])
            self.bias_quantized = (bias / bias_scale).round().clamp(-2**31, 2**31-1).to(torch.int32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized convolution"""
        # Dequantize weights
        if self.config.per_channel:
            weight_scales = self.weight_scale.view(-1, 1, 1, 1)
            weight_zero_points = self.weight_zero_point.view(-1, 1, 1, 1)
        else:
            weight_scales = self.weight_scale
            weight_zero_points = self.weight_zero_point
        
        weight_dequantized = weight_scales * (self.weight_quantized.float() - weight_zero_points.float())
        
        # Compute convolution
        output = F.conv2d(
            x, weight_dequantized, None, 
            self.stride, self.padding, self.dilation, self.groups
        )
        
        # Add bias if present
        if self.bias_quantized is not None:
            bias_dequantized = self.bias_scale * self.bias_quantized.float()
            output = output + bias_dequantized.view(1, -1, 1, 1)
        
        return output

class DynamicQuantization(QuantizationStrategy):
    """Dynamic quantization that adapts to input statistics"""
    
    def __init__(self):
        self.quantized_layers = {}
        self.running_stats = {}
    
    def quantize(self, model: nn.Module, config: QuantizationConfig) -> nn.Module:
        """Apply dynamic quantization"""
        logger.info("Applying dynamic quantization...")
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Replace with dynamic quantized version
                dynamic_module = self._create_dynamic_layer(module, config)
                self._replace_module(model, name, dynamic_module)
                self.quantized_layers[name] = dynamic_module
        
        logger.info(f"Applied dynamic quantization to {len(self.quantized_layers)} layers")
        return model
    
    def _create_dynamic_layer(self, module: nn.Module, config: QuantizationConfig) -> nn.Module:
        """Create a dynamic quantized version of the layer"""
        if isinstance(module, nn.Linear):
            return DynamicQuantizedLinear(module, config)
        elif isinstance(module, nn.Conv2d):
            return DynamicQuantizedConv2d(module, config)
        else:
            return module
    
    def _replace_module(self, model: nn.Module, name: str, new_module: nn.Module):
        """Replace a module in the model"""
        parts = name.split('.')
        parent = model
        
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, parts[-1], new_module)
    
    def estimate_speedup(self, model: nn.Module) -> float:
        """Estimate speedup from dynamic quantization"""
        return 1.3  # Conservative estimate for dynamic quantization

class DynamicQuantizedLinear(nn.Module):
    """Dynamically quantized linear layer"""
    
    def __init__(self, original_layer: nn.Linear, config: QuantizationConfig):
        super().__init__()
        self.original_layer = original_layer
        self.config = config
        self.running_min = None
        self.running_max = None
        self.momentum = 0.1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic quantization"""
        # Update running statistics
        current_min = x.min().item()
        current_max = x.max().item()
        
        if self.running_min is None:
            self.running_min = current_min
            self.running_max = current_max
        else:
            self.running_min = (1 - self.momentum) * self.running_min + self.momentum * current_min
            self.running_max = (1 - self.momentum) * self.running_max + self.momentum * current_max
        
        # Quantize input dynamically
        if self.training:
            # During training, use current statistics
            scale = (current_max - current_min) / 255.0 if current_max > current_min else 1.0
            zero_point = int(round(-current_min / scale)) if scale > 0 else 0
        else:
            # During inference, use running statistics
            scale = (self.running_max - self.running_min) / 255.0 if self.running_max > self.running_min else 1.0
            zero_point = int(round(-self.running_min / scale)) if scale > 0 else 0
        
        # Quantize and dequantize input
        if scale > 0:
            x_quantized = ((x - self.running_min) / scale).round().clamp(0, 255)
            x_dequantized = self.running_min + scale * x_quantized
        else:
            x_dequantized = x
        
        # Apply original layer
        return self.original_layer(x_dequantized)

class DynamicQuantizedConv2d(nn.Module):
    """Dynamically quantized convolutional layer"""
    
    def __init__(self, original_layer: nn.Conv2d, config: QuantizationConfig):
        super().__init__()
        self.original_layer = original_layer
        self.config = config
        self.running_min = None
        self.running_max = None
        self.momentum = 0.1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic quantization"""
        # Update running statistics
        current_min = x.min().item()
        current_max = x.max().item()
        
        if self.running_min is None:
            self.running_min = current_min
            self.running_max = current_max
        else:
            self.running_min = (1 - self.momentum) * self.running_min + self.momentum * current_min
            self.running_max = (1 - self.momentum) * self.running_max + self.momentum * current_max
        
        # Apply original layer (could add quantization here too)
        return self.original_layer(x)

class QuantizationOptimizer:
    """Optimizes quantization strategies for specific hardware"""
    
    def __init__(self, target_platform: str = "cpu"):
        self.target_platform = target_platform
        self.optimization_cache = {}
    
    def optimize_for_platform(self, model: nn.Module, config: QuantizationConfig) -> QuantizationConfig:
        """Optimize quantization configuration for target platform"""
        optimized_config = QuantizationConfig(**config.__dict__)
        
        if self.target_platform.lower() == "cpu":
            # CPU optimizations
            optimized_config.use_vnni = True  # Intel VNNI instructions
            optimized_config.use_neon = True  # ARM NEON
            optimized_config.per_channel = True  # Better accuracy with   overhead
            optimized_config.block_wise = True  # Memory locality
            
        elif self.target_platform.lower() == "arm":
            # ARM-specific optimizations
            optimized_config.use_neon = True
            optimized_config.bits = 8  # Better ARM support for int8
            optimized_config.symmetric = True  # Simpler ARM operations
            
        elif self.target_platform.lower() == "intel":
            # Intel-specific optimizations
            optimized_config.use_vnni = True
            optimized_config.bits = 8  # VNNI supports int8
            optimized_config.per_channel = True
        
        return optimized_config
    
    def auto_tune_quantization(self, model: nn.Module, 
                              test_data: torch.Tensor,
                              target_accuracy_drop: float = 0.05) -> QuantizationConfig:
        """Automatically tune quantization parameters"""
        logger.info("Auto-tuning quantization parameters...")
        
        best_config = None
        best_score = float('inf')
        
        # Test different configurations
        configs_to_test = [
            QuantizationConfig(bits=8, per_channel=True, symmetric=True),
            QuantizationConfig(bits=8, per_channel=True, symmetric=False),
            QuantizationConfig(bits=8, per_channel=False, symmetric=True),
            QuantizationConfig(bits=4, per_channel=True, block_wise=True),
        ]
        
        original_outputs = self._get_model_outputs(model, test_data)
        
        for config in configs_to_test:
            try:
                # Create a copy of the model for testing
                model_copy = self._create_model_copy(model)
                
                # Apply quantization
                quantizer = PostTrainingQuantization()
                quantized_model = quantizer.quantize(model_copy, config)
                
                # Evaluate
                quantized_outputs = self._get_model_outputs(quantized_model, test_data)
                accuracy_drop = self._calculate_accuracy_drop(original_outputs, quantized_outputs)
                speedup = quantizer.estimate_speedup(quantized_model)
                
                # Score based on accuracy preservation and speedup
                if accuracy_drop <= target_accuracy_drop:
                    score = -speedup  # Negative because we want to maximize speedup
                    
                    if score < best_score:
                        best_score = score
                        best_config = config
                        
            except Exception as e:
                logger.warning(f"Failed to test config {config}: {e}")
        
        if best_config is None:
            logger.warning("No suitable quantization config found, using default")
            best_config = QuantizationConfig()
        
        logger.info(f"Best quantization config: {best_config.bits}-bit, "
                   f"per_channel={best_config.per_channel}")
        
        return best_config
    
    def _get_model_outputs(self, model: nn.Module, test_data: torch.Tensor) -> torch.Tensor:
        """Get model outputs for test data"""
        model.eval()
        with torch.no_grad():
            outputs = model(test_data)
        return outputs
    
    def _create_model_copy(self, model: nn.Module) -> nn.Module:
        """Create a copy of the model for testing"""
        # Simple state dict copy
        state_dict = model.state_dict()
        model_copy = type(model)()  # Assumes model can be instantiated without args
        model_copy.load_state_dict(state_dict)
        return model_copy
    
    def _calculate_accuracy_drop(self, original: torch.Tensor, quantized: torch.Tensor) -> float:
        """Calculate accuracy drop between original and quantized outputs"""
        mse = F.mse_loss(original, quantized).item()
        relative_error = mse / (original.var().item() + 1e-8)
        return relative_error

# Factory functions for easy integration
def create_quantization_pipeline(config: QuantizationConfig = None) -> PostTrainingQuantization:
    """Create a quantization pipeline"""
    if config is None:
        config = QuantizationConfig()
    
    return PostTrainingQuantization()

def create_dynamic_quantization_pipeline(config: QuantizationConfig = None) -> DynamicQuantization:
    """Create a dynamic quantization pipeline"""
    if config is None:
        config = QuantizationConfig(quantization_type=QuantizationType.DYNAMIC)
    
    return DynamicQuantization()

if __name__ == "__main__":
    # Example usage
    config = QuantizationConfig(
        bits=8,
        per_channel=True,
        symmetric=True,
        optimize_for_cpu=True
    )
    
    quantizer = create_quantization_pipeline(config)
    logger.info("Quantization framework initialized successfully")