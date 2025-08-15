#!/usr/bin/env python3
"""
Purpose: Context window and model quantization optimizer for Ollama
Usage: Optimizes context usage and implements model quantization strategies
Requirements: ollama, transformers, torch, numpy
"""

import json
import time
import logging
import asyncio
from dataclasses import dataclass
from datetime import datetime
import httpx
import numpy as np
from collections import defaultdict, deque

# Configure logging
logger = logging.getLogger('ollama-context-optimizer')

@dataclass
class ContextMetrics:
    """Metrics for context window usage"""
    model: str
    average_input_length: float
    average_output_length: float
    max_context_used: int
    context_utilization: float
    truncation_rate: float
    optimal_window_size: int

@dataclass
class QuantizationConfig:
    """Configuration for model quantization"""
    model: str
    original_size_mb: float
    quantized_size_mb: float
    quantization_method: str
    performance_impact: float
    memory_savings: float

class OllamaContextOptimizer:
    """Advanced context window and quantization optimizer"""
    
    def __init__(self, ollama_url: str = "http://localhost:10104"):
        self.ollama_url = ollama_url
        self.client = httpx.AsyncClient(timeout=120.0)
        
        # Context tracking
        self.context_metrics: Dict[str, ContextMetrics] = {}
        self.context_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Quantization tracking
        self.quantization_configs: Dict[str, QuantizationConfig] = {}
        
        # Optimization settings
        self.min_context_window = 512
        self.max_context_window = 8192
        self.optimal_utilization = 0.85
        
    async def analyze_context_usage(self, model: str, num_samples: int = 100) -> ContextMetrics:
        """Analyze context window usage patterns for a model"""
        logger.info(f"Analyzing context usage for {model} with {num_samples} samples")
        
        input_lengths = []
        output_lengths = []
        context_used = []
        truncations = 0
        
        # Test prompts of varying lengths
        test_prompts = self._generate_test_prompts(num_samples)
        
        for i, prompt in enumerate(test_prompts):
            try:
                start_time = time.time()
                
                response = await self.client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_predict": 256,  # Consistent output length
                            "temperature": 0.1   # Consistent responses
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get('response', '')
                    
                    # Track lengths
                    input_len = len(prompt.split())
                    output_len = len(response_text.split())
                    total_context = input_len + output_len
                    
                    input_lengths.append(input_len)
                    output_lengths.append(output_len)
                    context_used.append(total_context)
                    
                    # Check for truncation indicators
                    if self._detect_truncation(response_text):
                        truncations += 1
                    
                    # Log progress
                    if (i + 1) % 10 == 0:
                        logger.debug(f"Processed {i + 1}/{num_samples} samples")
                
                else:
                    logger.warning(f"Request failed with status {response.status_code}")
            
            except Exception as e:
                logger.error(f"Error in sample {i}: {e}")
                continue
        
        # Calculate metrics
        if input_lengths:
            avg_input = np.mean(input_lengths)
            avg_output = np.mean(output_lengths)
            max_context = max(context_used)
            utilization = max_context / self.max_context_window if context_used else 0
            truncation_rate = truncations / len(input_lengths)
            
            # Calculate optimal window size
            percentile_95 = np.percentile(context_used, 95)
            optimal_window = min(int(percentile_95 * 1.2), self.max_context_window)
            optimal_window = max(optimal_window, self.min_context_window)
            
            metrics = ContextMetrics(
                model=model,
                average_input_length=avg_input,
                average_output_length=avg_output,
                max_context_used=max_context,
                context_utilization=utilization,
                truncation_rate=truncation_rate,
                optimal_window_size=optimal_window
            )
            
            self.context_metrics[model] = metrics
            
            logger.info(f"Context analysis completed for {model}:")
            logger.info(f"  Average input: {avg_input:.1f} tokens")
            logger.info(f"  Average output: {avg_output:.1f} tokens")
            logger.info(f"  Max context used: {max_context} tokens")
            logger.info(f"  Optimal window size: {optimal_window} tokens")
            logger.info(f"  Truncation rate: {truncation_rate*100:.1f}%")
            
            return metrics
        
        else:
            logger.error(f"No valid samples collected for {model}")
            return ContextMetrics(
                model=model,
                average_input_length=0,
                average_output_length=0,
                max_context_used=0,
                context_utilization=0,
                truncation_rate=1.0,
                optimal_window_size=self.min_context_window
            )
    
    def _generate_test_prompts(self, num_samples: int) -> List[str]:
        """Generate test prompts of varying lengths"""
        base_prompts = [
            "Explain artificial intelligence.",
            "Write a short story about a robot.",
            "Describe the process of machine learning.",
            "What are the benefits and risks of AI?",
            "Compare different programming languages.",
            "Explain advanced computing concepts.",
            "Describe the history of computers.",
            "What is the future of technology?"
        ]
        
        prompts = []
        
        for i in range(num_samples):
            base = base_prompts[i % len(base_prompts)]
            
            # Vary prompt length
            if i % 4 == 0:
                # Short prompt
                prompts.append(base)
            elif i % 4 == 1:
                # Medium prompt with context
                prompts.append(f"Given the context of modern technology development, {base.lower()} Please provide detailed examples and explanations.")
            elif i % 4 == 2:
                # Long prompt with multiple questions
                prompts.append(f"{base} Additionally, discuss the implications for society, potential applications in various industries, and how this technology might evolve in the next decade. What are the main challenges and opportunities?")
            else:
                # Very long prompt with background
                background = "In the rapidly evolving landscape of technology, artificial intelligence and machine learning have become pivotal forces shaping our future. From autonomous vehicles to medical diagnosis, AI applications are transforming industries worldwide. "
                prompts.append(f"{background}{base} Please provide a comprehensive analysis covering technical aspects, real-world applications, ethical considerations, and future prospects. Include specific examples and case studies where relevant.")
        
        return prompts
    
    def _detect_truncation(self, response: str) -> bool:
        """Detect if response was truncated due to context limits"""
        truncation_indicators = [
            response.endswith("..."),
            response.endswith("[truncated]"),
            len(response) < 10,  # Very short responses might indicate truncation
            not response.strip()  # Empty responses
        ]
        
        return any(truncation_indicators)
    
    async def optimize_context_windows(self) -> Dict[str, int]:
        """Optimize context window sizes for all models"""
        logger.info("Optimizing context windows for all models")
        
        models = ['tinyllama', 'tinyllama.2:3b', 'tinyllama']
        optimized_windows = {}
        
        for model in models:
            try:
                # Analyze if not already done
                if model not in self.context_metrics:
                    await self.analyze_context_usage(model)
                
                metrics = self.context_metrics[model]
                optimized_windows[model] = metrics.optimal_window_size
                
                logger.info(f"Optimized context window for {model}: {metrics.optimal_window_size} tokens")
            
            except Exception as e:
                logger.error(f"Error optimizing context for {model}: {e}")
                optimized_windows[model] = 2048  # Default fallback
        
        return optimized_windows
    
    async def analyze_quantization_opportunities(self) -> Dict[str, QuantizationConfig]:
        """Analyze opportunities for model quantization"""
        logger.info("Analyzing quantization opportunities")
        
        # Get model information
        response = await self.client.get(f"{self.ollama_url}/api/tags")
        if response.status_code != 200:
            logger.error("Failed to get model information")
            return {}
        
        models_info = response.json().get('models', [])
        quantization_configs = {}
        
        for model_info in models_info:
            model_name = model_info.get('name', '')
            size_bytes = model_info.get('size', 0)
            size_mb = size_bytes / (1024 * 1024)
            
            details = model_info.get('details', {})
            current_quantization = details.get('quantization_level', 'unknown')
            
            # Analyze quantization opportunities
            config = self._analyze_model_quantization(model_name, size_mb, current_quantization)
            if config:
                quantization_configs[model_name] = config
        
        self.quantization_configs = quantization_configs
        return quantization_configs
    
    def _analyze_model_quantization(self, model_name: str, size_mb: float, current_quant: str) -> Optional[QuantizationConfig]:
        """Analyze quantization options for a specific model"""
        
        # Quantization mappings and impact estimates
        quantization_options = {
            'Q8_0': {'size_factor': 0.6, 'performance_impact': 0.02},   # 8-bit,   impact
            'Q4_K_M': {'size_factor': 0.4, 'performance_impact': 0.05}, # 4-bit, small impact  
            'Q4_0': {'size_factor': 0.35, 'performance_impact': 0.08},  # 4-bit, moderate impact
            'Q3_K_M': {'size_factor': 0.3, 'performance_impact': 0.12}, # 3-bit, higher impact
            'Q2_K': {'size_factor': 0.25, 'performance_impact': 0.2}    # 2-bit, significant impact
        }
        
        # Current quantization analysis
        if current_quant in quantization_options:
            current_factor = quantization_options[current_quant]['size_factor']
        else:
            current_factor = 1.0  # Assume no quantization
        
        # Find optimal quantization
        best_option = None
        best_savings = 0
        
        for method, info in quantization_options.items():
            if method != current_quant and info['size_factor'] < current_factor:
                savings = (current_factor - info['size_factor']) / current_factor
                if savings > best_savings and info['performance_impact'] < 0.1:  # Max 10% performance impact
                    best_option = method
                    best_savings = savings
        
        if best_option:
            new_size = size_mb * quantization_options[best_option]['size_factor']
            memory_savings = size_mb - new_size
            
            return QuantizationConfig(
                model=model_name,
                original_size_mb=size_mb,
                quantized_size_mb=new_size,
                quantization_method=best_option,
                performance_impact=quantization_options[best_option]['performance_impact'],
                memory_savings=memory_savings
            )
        
        return None
    
    async def implement_context_optimization(self, model: str, optimal_window: int) -> bool:
        """Implement context window optimization for a model"""
        logger.info(f"Implementing context optimization for {model}: {optimal_window} tokens")
        
        # For now, we'll save the optimization to a config file
        # In a full implementation, this would modify the model configuration
        
        config_path = f"/opt/sutazaiapp/config/context_optimization_{model.replace(':', '_')}.json"
        
        config = {
            'model': model,
            'optimal_context_window': optimal_window,
            'optimization_timestamp': datetime.now().isoformat(),
            'metrics': self.context_metrics.get(model).__dict__ if model in self.context_metrics else None
        }
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Context optimization config saved to {config_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save context optimization config: {e}")
            return False
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'context_metrics': {},
            'quantization_configs': {},
            'recommendations': [],
            'potential_savings': {
                'memory_mb': 0,
                'context_efficiency': 0
            }
        }
        
        # Add context metrics
        for model, metrics in self.context_metrics.items():
            report['context_metrics'][model] = {
                'average_input_length': metrics.average_input_length,
                'average_output_length': metrics.average_output_length,
                'max_context_used': metrics.max_context_used,
                'context_utilization': metrics.context_utilization,
                'truncation_rate': metrics.truncation_rate,
                'optimal_window_size': metrics.optimal_window_size,
                'current_window_efficiency': metrics.context_utilization
            }
            
            # Calculate efficiency improvement
            if metrics.context_utilization < self.optimal_utilization:
                efficiency_gain = self.optimal_utilization - metrics.context_utilization
                report['potential_savings']['context_efficiency'] += efficiency_gain
        
        # Add quantization configs
        for model, config in self.quantization_configs.items():
            report['quantization_configs'][model] = {
                'original_size_mb': config.original_size_mb,
                'quantized_size_mb': config.quantized_size_mb,
                'quantization_method': config.quantization_method,
                'performance_impact': config.performance_impact,
                'memory_savings_mb': config.memory_savings
            }
            
            report['potential_savings']['memory_mb'] += config.memory_savings
        
        # Generate recommendations
        recommendations = []
        
        for model, metrics in self.context_metrics.items():
            if metrics.truncation_rate > 0.1:
                recommendations.append(f"Consider increasing context window for {model} - high truncation rate ({metrics.truncation_rate*100:.1f}%)")
            
            if metrics.context_utilization < 0.5:
                recommendations.append(f"Consider reducing context window for {model} - low utilization ({metrics.context_utilization*100:.1f}%)")
        
        for model, config in self.quantization_configs.items():
            if config.memory_savings > 100:  # More than 100MB savings
                recommendations.append(f"Apply {config.quantization_method} quantization to {model} - save {config.memory_savings:.0f}MB with {config.performance_impact*100:.1f}% impact")
        
        report['recommendations'] = recommendations
        
        # Save report
        report_path = f"/opt/sutazaiapp/logs/context_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Optimization report saved to: {report_path}")
        logger.info(f"Potential memory savings: {report['potential_savings']['memory_mb']:.0f}MB")
        logger.info(f"Context efficiency improvement: {report['potential_savings']['context_efficiency']*100:.1f}%")
        
        return report
    
    async def run_full_optimization(self) -> Dict[str, Any]:
        """Run complete context and quantization optimization"""
        logger.info("Starting full optimization process")
        
        models = ['tinyllama', 'tinyllama.2:3b', 'tinyllama']
        
        # 1. Analyze context usage for all models
        for model in models:
            try:
                await self.analyze_context_usage(model, num_samples=50)
            except Exception as e:
                logger.error(f"Context analysis failed for {model}: {e}")
        
        # 2. Optimize context windows
        optimized_windows = await self.optimize_context_windows()
        
        # 3. Analyze quantization opportunities
        quantization_configs = await self.analyze_quantization_opportunities()
        
        # 4. Implement optimizations
        for model, window_size in optimized_windows.items():
            await self.implement_context_optimization(model, window_size)
        
        # 5. Generate comprehensive report
        report = self.generate_optimization_report()
        
        logger.info("Full optimization process completed")
        return report

# CLI interface
async def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ollama Context and Quantization Optimizer')
    parser.add_argument('--analyze', choices=['all', 'tinyllama', 'tinyllama.2:3b', 'tinyllama'], help='Analyze context usage')
    parser.add_argument('--optimize', action='store_true', help='Run full optimization')
    parser.add_argument('--quantization', action='store_true', help='Analyze quantization opportunities')
    parser.add_argument('--report', action='store_true', help='Generate optimization report')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples for analysis')
    
    args = parser.parse_args()
    
    optimizer = OllamaContextOptimizer()
    
    try:
        if args.analyze:
            if args.analyze == 'all':
                models = ['tinyllama', 'tinyllama.2:3b', 'tinyllama']
                for model in models:
                    await optimizer.analyze_context_usage(model, args.samples)
            else:
                await optimizer.analyze_context_usage(args.analyze, args.samples)
        
        elif args.optimize:
            await optimizer.run_full_optimization()
        
        elif args.quantization:
            configs = await optimizer.analyze_quantization_opportunities()
            logger.info("Quantization Analysis Results:")
            for model, config in configs.items():
                logger.info(f"  {model}: {config.memory_savings:.0f}MB savings with {config.quantization_method}")
        
        elif args.report:
            report = optimizer.generate_optimization_report()
            logger.info(json.dumps(report, indent=2))
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import sys
        sys.exit(1)
    
    finally:
        await optimizer.client.aclose()

if __name__ == '__main__':
    asyncio.run(main())