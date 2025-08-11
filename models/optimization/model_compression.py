"""
Advanced Model Compression Framework for SutazAI
Implements pruning, structured sparsity, and advanced compression techniques
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class CompressionConfig:
    """Configuration for model compression"""
    sparsity_ratio: float = 0.3  # Target sparsity ratio
    structured_pruning: bool = True  # Use structured pruning
    magnitude_based: bool = True  # Magnitude-based pruning
    gradual_pruning: bool = True  # Gradual pruning schedule
    knowledge_distillation: bool = True  # Use knowledge distillation
    quantization_bits: int = 8  # Quantization bit width
    compression_schedule: Dict[str, float] = None  # Layer-wise compression ratios
    
    def __post_init__(self):
        if self.compression_schedule is None:
            self.compression_schedule = {
                'attention': 0.2,
                'feedforward': 0.4,
                'embedding': 0.1,
                'output': 0.15
            }

class CompressionStrategy(ABC):
    """Abstract base class for compression strategies"""
    
    @abstractmethod
    def compress(self, model: nn.Module, config: CompressionConfig) -> nn.Module:
        """Apply compression strategy to model"""
        pass
    
    @abstractmethod
    def estimate_compression_ratio(self, model: nn.Module) -> float:
        """Estimate compression ratio for the model"""
        pass

class MagnitudePruning(CompressionStrategy):
    """Magnitude-based pruning strategy"""
    
    def __init__(self):
        self.pruned_weights = {}
    
    def compress(self, model: nn.Module, config: CompressionConfig) -> nn.Module:
        """Apply magnitude-based pruning"""
        logger.info("Applying magnitude-based pruning...")
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self._prune_layer(module, name, config)
        
        return model
    
    def _prune_layer(self, layer: nn.Module, name: str, config: CompressionConfig):
        """Prune individual layer based on magnitude"""
        if hasattr(layer, 'weight') and layer.weight is not None:
            weight = layer.weight.data
            
            # Calculate magnitude threshold
            weight_magnitude = torch.abs(weight).flatten()
            threshold_idx = int(len(weight_magnitude) * config.sparsity_ratio)
            threshold = torch.topk(weight_magnitude, threshold_idx, largest=False)[0][-1]
            
            # Create mask
            mask = torch.abs(weight) > threshold
            
            # Apply pruning
            layer.weight.data *= mask.float()
            
            # Store mask for fine-tuning
            self.pruned_weights[name] = mask
            
            pruning_ratio = (mask == 0).float().mean().item()
            logger.info(f"Pruned {name}: {pruning_ratio:.2%} of weights removed")
    
    def estimate_compression_ratio(self, model: nn.Module) -> float:
        """Estimate compression ratio"""
        total_params = sum(p.numel() for p in model.parameters())
        pruned_params = sum(
            (mask == 0).sum().item() 
            for mask in self.pruned_weights.values()
        )
        return pruned_params / total_params if total_params > 0 else 0.0

class StructuredPruning(CompressionStrategy):
    """Structured pruning strategy (removes entire neurons/channels)"""
    
    def __init__(self):
        self.pruned_structure = {}
    
    def compress(self, model: nn.Module, config: CompressionConfig) -> nn.Module:
        """Apply structured pruning"""
        logger.info("Applying structured pruning...")
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._prune_linear_layer(module, name, config)
            elif isinstance(module, nn.Conv2d):
                self._prune_conv_layer(module, name, config)
        
        return model
    
    def _prune_linear_layer(self, layer: nn.Linear, name: str, config: CompressionConfig):
        """Prune linear layer by removing entire neurons"""
        if layer.weight.data.numel() == 0:
            return
            
        weight = layer.weight.data
        
        # Calculate importance scores (L2 norm of each neuron)
        importance_scores = torch.norm(weight, dim=1)
        
        # Determine neurons to keep
        num_neurons = weight.size(0)
        num_keep = int(num_neurons * (1 - config.sparsity_ratio))
        
        if num_keep < 1:
            num_keep = 1  # Keep at least one neuron
        
        _, indices_to_keep = torch.topk(importance_scores, num_keep)
        
        # Create new layer with reduced size
        new_layer = nn.Linear(
            layer.in_features, 
            num_keep, 
            bias=layer.bias is not None
        )
        
        # Copy weights and biases
        new_layer.weight.data = weight[indices_to_keep]
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data[indices_to_keep]
        
        # Store pruning information
        self.pruned_structure[name] = {
            'kept_indices': indices_to_keep,
            'original_size': num_neurons,
            'new_size': num_keep
        }
        
        # Replace layer in model (this would need model-specific implementation)
        logger.info(f"Structured pruning {name}: {num_neurons} -> {num_keep} neurons")
    
    def _prune_conv_layer(self, layer: nn.Conv2d, name: str, config: CompressionConfig):
        """Prune convolutional layer by removing entire channels"""
        weight = layer.weight.data  # [out_channels, in_channels, kernel_h, kernel_w]
        
        # Calculate importance scores (L2 norm of each output channel)
        importance_scores = torch.norm(weight.view(weight.size(0), -1), dim=1)
        
        # Determine channels to keep
        num_channels = weight.size(0)
        num_keep = int(num_channels * (1 - config.sparsity_ratio))
        
        if num_keep < 1:
            num_keep = 1
        
        _, indices_to_keep = torch.topk(importance_scores, num_keep)
        
        # Store pruning information
        self.pruned_structure[name] = {
            'kept_indices': indices_to_keep,
            'original_size': num_channels,
            'new_size': num_keep
        }
        
        logger.info(f"Structured pruning {name}: {num_channels} -> {num_keep} channels")
    
    def estimate_compression_ratio(self, model: nn.Module) -> float:
        """Estimate compression ratio for structured pruning"""
        total_reduction = 0
        total_params = 0
        
        for info in self.pruned_structure.values():
            reduction = info['original_size'] - info['new_size']
            total_reduction += reduction
            total_params += info['original_size']
        
        return total_reduction / total_params if total_params > 0 else 0.0

class GradualPruning:
    """Implements gradual pruning over training iterations"""
    
    def __init__(self, initial_sparsity: float = 0.0, final_sparsity: float = 0.5, 
                 pruning_frequency: int = 100):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.pruning_frequency = pruning_frequency
        self.current_step = 0
    
    def get_current_sparsity(self, step: int) -> float:
        """Calculate current sparsity based on schedule"""
        if step <= 0:
            return self.initial_sparsity
        
        # Polynomial decay schedule
        progress = min(step / 1000, 1.0)  # Assume 1000 steps for full schedule
        sparsity = self.initial_sparsity + (
            self.final_sparsity - self.initial_sparsity
        ) * (1 - (1 - progress) ** 3)
        
        return sparsity
    
    def should_prune(self, step: int) -> bool:
        """Check if pruning should be applied at current step"""
        return step % self.pruning_frequency == 0

class ModelCompressor:
    """Main model compression orchestrator"""
    
    def __init__(self, config: CompressionConfig = None):
        self.config = config or CompressionConfig()
        self.strategies = {
            'magnitude': MagnitudePruning(),
            'structured': StructuredPruning()
        }
        self.compression_history = []
    
    def compress_model(self, model: nn.Module, strategy: str = 'magnitude') -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress model using specified strategy"""
        if strategy not in self.strategies:
            raise ValueError(f"Unknown compression strategy: {strategy}")
        
        logger.info(f"Starting model compression with {strategy} strategy...")
        
        # Record original model stats
        original_params = self._count_parameters(model)
        original_size = self._estimate_model_size(model)
        
        # Apply compression
        strategy_impl = self.strategies[strategy]
        compressed_model = strategy_impl.compress(model, self.config)
        
        # Record compressed model stats
        compressed_params = self._count_parameters(compressed_model)
        compressed_size = self._estimate_model_size(compressed_model)
        compression_ratio = strategy_impl.estimate_compression_ratio(model)
        
        # Create compression report
        report = {
            'strategy': strategy,
            'original_parameters': original_params,
            'compressed_parameters': compressed_params,
            'parameter_reduction': (original_params - compressed_params) / original_params,
            'original_size_mb': original_size,
            'compressed_size_mb': compressed_size,
            'size_reduction': (original_size - compressed_size) / original_size,
            'estimated_compression_ratio': compression_ratio,
            'config': self.config.__dict__
        }
        
        self.compression_history.append(report)
        
        logger.info(f"Compression complete. Parameter reduction: {report['parameter_reduction']:.2%}")
        logger.info(f"Size reduction: {report['size_reduction']:.2%}")
        
        return compressed_model, report
    
    def _count_parameters(self, model: nn.Module) -> int:
        """Count total number of parameters in model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def _estimate_model_size(self, model: nn.Module) -> float:
        """Estimate model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)  # Convert to MB
    
    def save_compressed_model(self, model: nn.Module, path: str, metadata: Dict[str, Any] = None):
        """Save compressed model with metadata"""
        save_dict = {
            'model_state_dict': model.state_dict(),
            'compression_config': self.config.__dict__,
            'compression_history': self.compression_history,
            'metadata': metadata or {}
        }
        
        torch.save(save_dict, path)
        logger.info(f"Compressed model saved to {path}")
    
    def load_compressed_model(self, model: nn.Module, path: str) -> Tuple[nn.Module, Dict[str, Any]]:
        """Load compressed model with metadata"""
        checkpoint = torch.load(path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint.get('metadata', {})
    
    def benchmark_compression_strategies(self, model: nn.Module) -> Dict[str, Dict[str, Any]]:
        """Benchmark different compression strategies"""
        results = {}
        
        for strategy_name in self.strategies.keys():
            # Create a copy of the model for testing
            model_copy = self._deep_copy_model(model)
            
            try:
                _, report = self.compress_model(model_copy, strategy_name)
                results[strategy_name] = report
            except Exception as e:
                logger.error(f"Failed to benchmark {strategy_name}: {e}")
                results[strategy_name] = {'error': str(e)}
        
        return results
    
    def _deep_copy_model(self, model: nn.Module) -> nn.Module:
        """Create a deep copy of the model"""
        # Simple approach - save and reload state dict
        state_dict = model.state_dict()
        model_copy = type(model)()  # Assumes model can be instantiated without args
        model_copy.load_state_dict(state_dict)
        return model_copy
    
    def optimize_compression_config(self, model: nn.Module, target_compression: float = 0.5) -> CompressionConfig:
        """Automatically optimize compression configuration"""
        logger.info(f"Optimizing compression config for target ratio: {target_compression}")
        
        best_config = None
        best_score = float('inf')
        
        # Grid search over compression parameters
        sparsity_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        
        for sparsity in sparsity_ratios:
            config = CompressionConfig(sparsity_ratio=sparsity)
            
            try:
                model_copy = self._deep_copy_model(model)
                _, report = self.compress_model(model_copy, 'magnitude')
                
                # Score based on how close we are to target and quality preservation
                compression_achieved = report['parameter_reduction']
                score = abs(compression_achieved - target_compression)
                
                if score < best_score:
                    best_score = score
                    best_config = config
                    
            except Exception as e:
                logger.warning(f"Failed to test sparsity {sparsity}: {e}")
        
        logger.info(f"Optimal compression config found with sparsity: {best_config.sparsity_ratio}")
        return best_config or CompressionConfig()

class CompressionAnalyzer:
    """Analyzes and reports on model compression results"""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_compression_impact(self, original_model: nn.Module, 
                                 compressed_model: nn.Module,
                                 test_data: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Analyze the impact of compression on model performance"""
        analysis = {
            'size_analysis': self._analyze_size_reduction(original_model, compressed_model),
            'parameter_analysis': self._analyze_parameter_reduction(original_model, compressed_model),
            'structure_analysis': self._analyze_structure_changes(original_model, compressed_model)
        }
        
        if test_data is not None:
            analysis['performance_analysis'] = self._analyze_performance_impact(
                original_model, compressed_model, test_data
            )
        
        return analysis
    
    def _analyze_size_reduction(self, original: nn.Module, compressed: nn.Module) -> Dict[str, float]:
        """Analyze model size reduction"""
        orig_size = sum(p.numel() * p.element_size() for p in original.parameters())
        comp_size = sum(p.numel() * p.element_size() for p in compressed.parameters())
        
        return {
            'original_size_mb': orig_size / (1024 * 1024),
            'compressed_size_mb': comp_size / (1024 * 1024),
            'size_reduction_ratio': (orig_size - comp_size) / orig_size,
            'compression_factor': orig_size / comp_size if comp_size > 0 else 0
        }
    
    def _analyze_parameter_reduction(self, original: nn.Module, compressed: nn.Module) -> Dict[str, Any]:
        """Analyze parameter count reduction"""
        orig_params = sum(p.numel() for p in original.parameters())
        comp_params = sum(p.numel() for p in compressed.parameters())
        
        # Layer-wise analysis
        layer_analysis = {}
        for (orig_name, orig_param), (comp_name, comp_param) in zip(
            original.named_parameters(), compressed.named_parameters()
        ):
            if orig_name == comp_name:
                layer_analysis[orig_name] = {
                    'original_params': orig_param.numel(),
                    'compressed_params': comp_param.numel(),
                    'reduction_ratio': (orig_param.numel() - comp_param.numel()) / orig_param.numel()
                }
        
        return {
            'original_parameters': orig_params,
            'compressed_parameters': comp_params,
            'total_reduction_ratio': (orig_params - comp_params) / orig_params,
            'layer_wise_analysis': layer_analysis
        }
    
    def _analyze_structure_changes(self, original: nn.Module, compressed: nn.Module) -> Dict[str, Any]:
        """Analyze structural changes in the model"""
        orig_modules = dict(original.named_modules())
        comp_modules = dict(compressed.named_modules())
        
        structure_changes = {
            'removed_modules': [],
            'modified_modules': [],
            'unchanged_modules': []
        }
        
        for name in orig_modules:
            if name not in comp_modules:
                structure_changes['removed_modules'].append(name)
            elif type(orig_modules[name]) != type(comp_modules[name]):
                structure_changes['modified_modules'].append({
                    'name': name,
                    'original_type': type(orig_modules[name]).__name__,
                    'compressed_type': type(comp_modules[name]).__name__
                })
            else:
                structure_changes['unchanged_modules'].append(name)
        
        return structure_changes
    
    def _analyze_performance_impact(self, original: nn.Module, compressed: nn.Module, 
                                  test_data: torch.Tensor) -> Dict[str, Any]:
        """Analyze performance impact using test data"""
        original.eval()
        compressed.eval()
        
        with torch.no_grad():
            orig_output = original(test_data)
            comp_output = compressed(test_data)
            
            # Calculate metrics
            mse_loss = F.mse_loss(orig_output, comp_output).item()
            cosine_sim = F.cosine_similarity(
                orig_output.flatten(), comp_output.flatten(), dim=0
            ).item()
            
            # Output statistics
            orig_stats = {
                'mean': orig_output.mean().item(),
                'std': orig_output.std().item(),
                'min': orig_output.min().item(),
                'max': orig_output.max().item()
            }
            
            comp_stats = {
                'mean': comp_output.mean().item(),
                'std': comp_output.std().item(),
                'min': comp_output.min().item(),
                'max': comp_output.max().item()
            }
        
        return {
            'mse_loss': mse_loss,
            'cosine_similarity': cosine_sim,
            'output_correlation': float(np.corrcoef(
                orig_output.flatten().numpy(), 
                comp_output.flatten().numpy()
            )[0, 1]),
            'original_stats': orig_stats,
            'compressed_stats': comp_stats
        }
    
    def generate_compression_report(self, analysis: Dict[str, Any], 
                                  output_path: str = None) -> str:
        """Generate a comprehensive compression report"""
        report = []
        report.append("# Model Compression Analysis Report")
        report.append("=" * 50)
        
        # Size Analysis
        if 'size_analysis' in analysis:
            size_data = analysis['size_analysis']
            report.append("\n## Size Reduction Analysis")
            report.append(f"Original Size: {size_data['original_size_mb']:.2f} MB")
            report.append(f"Compressed Size: {size_data['compressed_size_mb']:.2f} MB")
            report.append(f"Size Reduction: {size_data['size_reduction_ratio']:.2%}")
            report.append(f"Compression Factor: {size_data['compression_factor']:.2f}x")
        
        # Parameter Analysis
        if 'parameter_analysis' in analysis:
            param_data = analysis['parameter_analysis']
            report.append("\n## Parameter Reduction Analysis")
            report.append(f"Original Parameters: {param_data['original_parameters']:,}")
            report.append(f"Compressed Parameters: {param_data['compressed_parameters']:,}")
            report.append(f"Parameter Reduction: {param_data['total_reduction_ratio']:.2%}")
        
        # Performance Analysis
        if 'performance_analysis' in analysis:
            perf_data = analysis['performance_analysis']
            report.append("\n## Performance Impact Analysis")
            report.append(f"MSE Loss: {perf_data['mse_loss']:.6f}")
            report.append(f"Cosine Similarity: {perf_data['cosine_similarity']:.4f}")
            report.append(f"Output Correlation: {perf_data['output_correlation']:.4f}")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Compression report saved to {output_path}")
        
        return report_text

# Example usage and integration
def create_compression_pipeline() -> ModelCompressor:
    """Create a pre-configured compression pipeline for SutazAI agents"""
    config = CompressionConfig(
        sparsity_ratio=0.3,
        structured_pruning=True,
        magnitude_based=True,
        gradual_pruning=True,
        knowledge_distillation=True,
        quantization_bits=8
    )
    
    return ModelCompressor(config)

if __name__ == "__main__":
    # Example usage
    compressor = create_compression_pipeline()
    analyzer = CompressionAnalyzer()
    
