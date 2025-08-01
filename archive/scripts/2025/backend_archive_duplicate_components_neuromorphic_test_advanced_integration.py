#!/usr/bin/env python3
"""
Test script for Advanced Biological Modeling Integration
Tests the integration of advanced biological neural networks with the neuromorphic engine
"""

import os
import sys
import asyncio
import torch
import numpy as np
import logging
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuromorphic.enhanced_engine import EnhancedNeuromorphicEngine
from neuromorphic.advanced_biological_modeling import (
    AdvancedNeuralLinkNetwork,
    MultiCompartmentNeuron,
    AdvancedBiologicalParameters,
    CellType,
    create_advanced_neural_network
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_advanced_biological_integration():
    """Test the integration of advanced biological modeling"""
    
    print("=" * 60)
    print("Testing Advanced Biological Modeling Integration")
    print("=" * 60)
    
    # Configuration for the neuromorphic engine
    config = {
        'processing_mode': 'adaptive',
        'use_advanced_biological_modeling': True,
        'network': {
            'population_sizes': {
                'sensory': 64,
                'l2_3_pyramidal': 128,
                'l5_pyramidal': 64,
                'fast_spiking': 32,
                'dopaminergic': 16,
                'output': 32
            }
        },
        'encoding': {
            'method': 'poisson',
            'max_frequency': 100.0
        },
        'decoding': {
            'method': 'rate',
            'time_window': 50.0
        },
        'plasticity': {
            'rules': ['STDP', 'homeostatic', 'metaplasticity'],
            'learning_rate': 1e-4
        },
        'learning_enabled': True,
        'max_workers': 4
    }
    
    try:
        # Initialize the enhanced neuromorphic engine
        print("\n1. Initializing Enhanced Neuromorphic Engine...")
        engine = EnhancedNeuromorphicEngine(config)
        print("✓ Engine initialized successfully")
        
        # Test with various input patterns
        test_cases = [
            {
                'name': 'Simple Pattern',
                'data': torch.randn(1, 32, 10),
                'context': {'attention_bias': [0.1] * 32}
            },
            {
                'name': 'Complex Temporal Pattern',
                'data': torch.sin(torch.linspace(0, 4*np.pi, 50)).unsqueeze(0).unsqueeze(0).expand(1, 32, 50),
                'context': {'attention_bias': [0.2] * 32}
            },
            {
                'name': 'Noisy Input',
                'data': torch.randn(1, 32, 25) * 0.5,
                'context': None
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i+1}. Testing {test_case['name']}...")
            
            # Process input through the enhanced engine
            results = await engine.process_input(
                test_case['data'], 
                context=test_case['context']
            )
            
            # Validate results
            print(f"   Processing time: {results['processing_time']:.3f}s")
            print(f"   Energy consumed: {results['energy_consumed']:.6f} units")
            print(f"   Network activity: {results['network_state'].network_activity:.3f}")
            print(f"   Output shape: {results['output'].shape}")
            
            # Check for biological network statistics
            if 'network_activity' in results and hasattr(results['network_activity'], 'get'):
                network_stats = results['network_activity'].get('network_statistics', {})
                if network_stats:
                    print(f"   Total spikes: {network_stats.get('total_spikes', 'N/A')}")
                    print(f"   Memory usage: {network_stats.get('memory_usage_mb', 'N/A'):.1f} MB")
            
            print(f"   ✓ {test_case['name']} processed successfully")
        
        # Test network state persistence
        print(f"\n{len(test_cases)+2}. Testing Network State...")
        network_state = engine.get_network_state()
        print(f"   Total spikes: {network_state.total_spikes}")
        print(f"   Energy consumed: {network_state.energy_consumed:.6f}")
        print(f"   Network activity: {network_state.network_activity:.3f}")
        print("   ✓ Network state retrieved successfully")
        
        # Test processing statistics
        print(f"\n{len(test_cases)+3}. Testing Processing Statistics...")
        stats = engine.get_processing_stats()
        print(f"   Processing time: {stats.processing_time:.3f}s")
        print(f"   Energy efficiency: {stats.energy_efficiency:.3f}")
        print(f"   Memory usage: {stats.memory_usage:.1f} MB")
        print("   ✓ Processing statistics retrieved successfully")
        
        print("\n" + "=" * 60)
        print("Advanced Biological Modeling Integration Test PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

async def test_standalone_advanced_network():
    """Test the standalone advanced neural network"""
    
    print("\n" + "=" * 60)
    print("Testing Standalone Advanced Neural Network")
    print("=" * 60)
    
    try:
        # Create advanced neural network configuration
        config = {
            'population_sizes': {
                'sensory': 32,
                'l2_3_pyramidal': 64,
                'l5_pyramidal': 32,
                'fast_spiking': 16,
                'dopaminergic': 8,
                'output': 16
            },
            'learning_enabled': True,
            'plasticity_rules': ['STDP', 'homeostatic', 'metaplasticity'],
            'deep_integration': True
        }
        
        print("\n1. Creating Advanced Neural Network...")
        network = create_advanced_neural_network(config)
        print("✓ Advanced neural network created successfully")
        
        # Test input processing
        print("\n2. Testing Input Processing...")
        test_input = torch.randn(1, 32, 20)  # batch_size=1, features=32, time_steps=20
        
        results = await network.process_input(test_input, simulation_duration=50.0)
        
        # Validate results
        print(f"   Input shape: {test_input.shape}")
        print(f"   Processing completed in {results['network_statistics']['processing_time']:.3f}s")
        print(f"   Total spikes: {results['network_statistics']['total_spikes']}")
        print(f"   Memory usage: {results['network_statistics']['memory_usage_mb']:.1f} MB")
        
        # Check layer-wise activity
        for layer_name, spikes in results['spikes'].items():
            if isinstance(spikes, torch.Tensor) and spikes.numel() > 0:
                avg_activity = spikes.mean().item()
                print(f"   {layer_name} average activity: {avg_activity:.4f}")
        
        print("   ✓ Input processing completed successfully")
        
        # Test network state
        print("\n3. Testing Network State...")
        network_state = network.get_network_state()
        print(f"   Simulation time: {network_state['simulation_time']:.2f} ms")
        print(f"   Total neurons: {network_state['neuron_count']}")
        print(f"   Total synapses: {network_state['synapse_count']}")
        print("   ✓ Network state retrieved successfully")
        
        print("\n" + "=" * 60)
        print("Standalone Advanced Neural Network Test PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

async def test_biological_neuron_dynamics():
    """Test individual biological neuron dynamics"""
    
    print("\n" + "=" * 60)
    print("Testing Biological Neuron Dynamics")
    print("=" * 60)
    
    try:
        # Test different cell types
        cell_types = [
            CellType.PYRAMIDAL_L5,
            CellType.FAST_SPIKING_INTERNEURON,
            CellType.DOPAMINERGIC_VTA
        ]
        
        params = AdvancedBiologicalParameters()
        
        for i, cell_type in enumerate(cell_types, 1):
            print(f"\n{i}. Testing {cell_type.value} neuron...")
            
            # Create neuron
            neuron = MultiCompartmentNeuron(cell_type, params)
            
            # Test with synaptic input
            synaptic_input = torch.randn(1, 10) * 0.1  # Small synaptic currents
            
            # Simulate neuron dynamics
            outputs = []
            for t in range(100):  # 100 time steps
                output = neuron(synaptic_input, dt=0.1)
                outputs.append(output)
            
            # Analyze results
            spikes = torch.cat([out['spike'] for out in outputs])
            potentials = torch.cat([out['membrane_potential'] for out in outputs])
            calcium = torch.cat([out['calcium'] for out in outputs])
            
            spike_count = spikes.sum().item()
            avg_potential = potentials.mean().item()
            avg_calcium = calcium.mean().item()
            
            print(f"   Spikes generated: {spike_count}")
            print(f"   Average membrane potential: {avg_potential:.2f} mV")
            print(f"   Average calcium: {avg_calcium:.3f} μM")
            print(f"   ✓ {cell_type.value} neuron tested successfully")
        
        print("\n" + "=" * 60)
        print("Biological Neuron Dynamics Test PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

async def main():
    """Main test runner"""
    
    print("Starting Advanced Biological Modeling Integration Tests")
    print("=" * 80)
    
    # Run all tests
    tests = [
        test_biological_neuron_dynamics,
        test_standalone_advanced_network,
        test_advanced_biological_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"Test failed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests PASSED! Advanced biological modeling integration is working correctly.")
    else:
        print("❌ Some tests FAILED. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())