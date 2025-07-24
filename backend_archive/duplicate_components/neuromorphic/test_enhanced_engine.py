#!/usr/bin/env python3
"""
Test suite for Enhanced Neuromorphic Engine

This module contains comprehensive tests for the biological neural modeling
and enhanced neuromorphic processing capabilities.
"""

import asyncio
import numpy as np
import torch
import logging
from typing import Dict, Any

from .enhanced_engine import (
    EnhancedNeuromorphicEngine,
    ProcessingMode,
    NetworkState,
    ProcessingStats
)
from .biological_modeling import (
    NeuralLinkNetwork,
    BiologicalNeuron,
    BiologicalParameters,
    NeuronType
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestEnhancedEngine")

def run_integration_test():
    """Run a comprehensive integration test"""
    logger.info("Starting Enhanced Neuromorphic Engine Integration Test")
    
    # Configuration for integration test
    config = {
        'processing_mode': 'adaptive',
        'network': {
            'population_sizes': {
                'input': 64,
                'excitatory': 256,
                'inhibitory': 64,
                'output': 32
            }
        },
        'encoding': {
            'method': 'poisson',
            'max_frequency': 200.0
        },
        'decoding': {
            'method': 'population'
        },
        'plasticity': {
            'rules': ['STDP', 'homeostatic'],
            'learning_rate': 5e-4
        },
        'learning_enabled': True,
        'max_workers': 4
    }
    
    async def run_async_test():
        try:
            # Initialize engine
            engine = EnhancedNeuromorphicEngine(config)
            logger.info("Engine initialized successfully")
            
            # Create complex input pattern
            batch_size = 2
            input_size = 64
            time_steps = 200
            
            # Generate input with multiple patterns
            input_data = torch.zeros(batch_size, input_size, time_steps)
            
            # Pattern 1: Moving wave
            for t in range(time_steps):
                wave_pos = int((t / time_steps) * input_size)
                if wave_pos < input_size:
                    input_data[0, wave_pos, t] = 1.0
            
            # Pattern 2: Oscillatory pattern
            t_array = np.linspace(0, 4*np.pi, time_steps)
            for i in range(input_size):
                freq = 0.1 + i * 0.05
                input_data[1, i, :] = torch.from_numpy(0.5 * (1 + np.sin(freq * t_array)))
            
            # Process both patterns
            logger.info("Processing complex input patterns...")
            results = await engine.process_input(input_data)
            
            # Analyze results
            output = results['output']
            stats = results['statistics']
            state = results['network_state']
            
            logger.info("Integration test results:")
            logger.info(f"  Output shape: {output.shape}")
            logger.info(f"  Processing time: {stats.processing_time:.4f}s")
            logger.info(f"  Spike rate: {stats.spike_rate:.6f}")
            logger.info(f"  Energy efficiency: {stats.energy_efficiency:.2f} spikes/J")
            logger.info(f"  Total spikes: {state.total_spikes}")
            logger.info(f"  Network activity: {state.network_activity:.6f}")
            logger.info(f"  Memory usage: {stats.memory_usage:.1f}MB")
            
            # Test biological neuron model
            logger.info("Testing biological neuron model...")
            params = BiologicalParameters()
            neuron = BiologicalNeuron(NeuronType.PYRAMIDAL, params)
            
            # Test single spike generation
            input_current = torch.tensor([[50.0]])  # Strong input
            spike_output = neuron(input_current, dt=0.1)
            
            logger.info(f"  Neuron output shape: {spike_output.shape}")
            logger.info(f"  Spike generated: {spike_output.item() > 0}")
            
            # Test Neural Link Network
            logger.info("Testing Neural Link Network...")
            network_config = {
                'population_sizes': {
                    'input': 16,
                    'excitatory': 32,
                    'inhibitory': 8,
                    'output': 4
                }
            }
            
            network = NeuralLinkNetwork(network_config)
            test_input = torch.randn(1, 16, 50)
            network_results = await network.process_input(test_input)
            
            stats = network.get_network_statistics()
            logger.info(f"  Network total neurons: {stats['total_neurons']}")
            logger.info(f"  Network output shape: {network_results['spikes']['output'].shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Integration test failed with error: {e}")
            return False
    
    # Run the async test
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    success = loop.run_until_complete(run_async_test())
    loop.close()
    
    if success:
        logger.info(" Integration test completed successfully!")
    else:
        logger.error(" Integration test failed!")
    
    return success

if __name__ == "__main__":
    # Run integration test if script is executed directly
    run_integration_test()