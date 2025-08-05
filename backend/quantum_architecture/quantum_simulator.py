#!/usr/bin/env python3
"""
Purpose: Quantum Circuit Simulator for Testing Quantum Algorithms
Usage: Simulates quantum circuits for algorithm development and testing
Requirements: numpy, scipy, matplotlib (optional for visualization)
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
import multiprocessing as mp
from functools import lru_cache
import scipy.linalg as la

# Configure logging
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger('quantum-simulator')


class QuantumGate(Enum):
    """Standard quantum gates"""
    # Single-qubit gates
    I = "identity"
    X = "pauli_x"
    Y = "pauli_y"
    Z = "pauli_z"
    H = "hadamard"
    S = "phase"
    T = "t_gate"
    RX = "rotation_x"
    RY = "rotation_y"
    RZ = "rotation_z"
    
    # Two-qubit gates
    CNOT = "controlled_not"
    CZ = "controlled_z"
    SWAP = "swap"
    
    # Three-qubit gates
    TOFFOLI = "toffoli"
    FREDKIN = "fredkin"


@dataclass
class QuantumCircuit:
    """Represents a quantum circuit"""
    n_qubits: int
    gates: List[Tuple[QuantumGate, List[int], Optional[float]]] = field(default_factory=list)
    measurements: List[int] = field(default_factory=list)
    
    def add_gate(self, gate: QuantumGate, qubits: List[int], parameter: Optional[float] = None):
        """Add a gate to the circuit"""
        self.gates.append((gate, qubits, parameter))
    
    def add_measurement(self, qubit: int):
        """Add measurement to specific qubit"""
        if qubit not in self.measurements:
            self.measurements.append(qubit)
    
    def depth(self) -> int:
        """Calculate circuit depth"""
        if not self.gates:
            return 0
        
        # Track the last layer each qubit was used
        qubit_layers = [-1] * self.n_qubits
        max_depth = 0
        
        for gate, qubits, _ in self.gates:
            # Find the latest layer among involved qubits
            current_layer = max(qubit_layers[q] for q in qubits) + 1
            
            # Update layers for involved qubits
            for q in qubits:
                qubit_layers[q] = current_layer
            
            max_depth = max(max_depth, current_layer)
        
        return max_depth + 1


class QuantumSimulator:
    """
    Full state vector quantum circuit simulator
    Supports up to ~20 qubits on typical hardware
    """
    
    def __init__(self, backend: str = "numpy"):
        self.backend = backend
        self.gate_matrices = self._initialize_gate_matrices()
        self.measurement_cache = {}
        
    def _initialize_gate_matrices(self) -> Dict[QuantumGate, np.ndarray]:
        """Initialize matrices for standard gates"""
        sqrt2 = np.sqrt(2)
        
        matrices = {
            QuantumGate.I: np.array([[1, 0], [0, 1]], dtype=complex),
            QuantumGate.X: np.array([[0, 1], [1, 0]], dtype=complex),
            QuantumGate.Y: np.array([[0, -1j], [1j, 0]], dtype=complex),
            QuantumGate.Z: np.array([[1, 0], [0, -1]], dtype=complex),
            QuantumGate.H: np.array([[1, 1], [1, -1]], dtype=complex) / sqrt2,
            QuantumGate.S: np.array([[1, 0], [0, 1j]], dtype=complex),
            QuantumGate.T: np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
            QuantumGate.CNOT: np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 0, 1],
                                        [0, 0, 1, 0]], dtype=complex),
            QuantumGate.CZ: np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, -1]], dtype=complex),
            QuantumGate.SWAP: np.array([[1, 0, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 0, 1]], dtype=complex),
            QuantumGate.TOFFOLI: self._create_toffoli_matrix(),
            QuantumGate.FREDKIN: self._create_fredkin_matrix()
        }
        
        return matrices
    
    def _create_toffoli_matrix(self) -> np.ndarray:
        """Create Toffoli (CCNOT) gate matrix"""
        matrix = np.eye(8, dtype=complex)
        matrix[6, 6] = 0
        matrix[6, 7] = 1
        matrix[7, 6] = 1
        matrix[7, 7] = 0
        return matrix
    
    def _create_fredkin_matrix(self) -> np.ndarray:
        """Create Fredkin (CSWAP) gate matrix"""
        matrix = np.eye(8, dtype=complex)
        matrix[5, 5] = 0
        matrix[5, 6] = 1
        matrix[6, 5] = 1
        matrix[6, 6] = 0
        return matrix
    
    def get_parameterized_gate(self, gate: QuantumGate, parameter: float) -> np.ndarray:
        """Get matrix for parameterized gates"""
        if gate == QuantumGate.RX:
            cos = np.cos(parameter / 2)
            sin = np.sin(parameter / 2)
            return np.array([[cos, -1j * sin],
                           [-1j * sin, cos]], dtype=complex)
        
        elif gate == QuantumGate.RY:
            cos = np.cos(parameter / 2)
            sin = np.sin(parameter / 2)
            return np.array([[cos, -sin],
                           [sin, cos]], dtype=complex)
        
        elif gate == QuantumGate.RZ:
            return np.array([[np.exp(-1j * parameter / 2), 0],
                           [0, np.exp(1j * parameter / 2)]], dtype=complex)
        
        else:
            raise ValueError(f"Gate {gate} is not parameterized")
    
    def initialize_state(self, n_qubits: int, initial_state: Optional[str] = None) -> np.ndarray:
        """
        Initialize quantum state
        
        Args:
            n_qubits: Number of qubits
            initial_state: Initial state as binary string (e.g., "0101")
            
        Returns:
            State vector
        """
        state_size = 2 ** n_qubits
        
        if initial_state is None:
            # Default to |00...0>
            state = np.zeros(state_size, dtype=complex)
            state[0] = 1.0
        else:
            # Initialize to specified computational basis state
            if len(initial_state) != n_qubits:
                raise ValueError(f"Initial state must have {n_qubits} bits")
            
            state_index = int(initial_state, 2)
            state = np.zeros(state_size, dtype=complex)
            state[state_index] = 1.0
        
        return state
    
    def apply_gate(self, state: np.ndarray, gate: QuantumGate, 
                   qubits: List[int], parameter: Optional[float] = None) -> np.ndarray:
        """
        Apply quantum gate to state
        
        Args:
            state: Current quantum state
            gate: Gate to apply
            qubits: Qubits to apply gate to
            parameter: Parameter for parameterized gates
            
        Returns:
            New state after gate application
        """
        n_qubits = int(np.log2(len(state)))
        
        # Get gate matrix
        if parameter is not None:
            gate_matrix = self.get_parameterized_gate(gate, parameter)
        else:
            gate_matrix = self.gate_matrices[gate]
        
        # Apply gate based on number of qubits
        if len(qubits) == 1:
            return self._apply_single_qubit_gate(state, gate_matrix, qubits[0], n_qubits)
        elif len(qubits) == 2:
            return self._apply_two_qubit_gate(state, gate_matrix, qubits[0], qubits[1], n_qubits)
        elif len(qubits) == 3:
            return self._apply_three_qubit_gate(state, gate_matrix, qubits, n_qubits)
        else:
            raise ValueError(f"Gates on {len(qubits)} qubits not supported")
    
    def _apply_single_qubit_gate(self, state: np.ndarray, gate_matrix: np.ndarray, 
                                 qubit: int, n_qubits: int) -> np.ndarray:
        """Apply single-qubit gate"""
        new_state = np.zeros_like(state)
        
        # Iterate through all basis states
        for i in range(len(state)):
            # Get bit value at qubit position
            bit_mask = 1 << (n_qubits - qubit - 1)
            bit_value = (i & bit_mask) >> (n_qubits - qubit - 1)
            
            # Calculate new indices after gate application
            for new_bit in range(2):
                new_index = i ^ (bit_value << (n_qubits - qubit - 1)) ^ (new_bit << (n_qubits - qubit - 1))
                new_state[new_index] += gate_matrix[new_bit, bit_value] * state[i]
        
        return new_state
    
    def _apply_two_qubit_gate(self, state: np.ndarray, gate_matrix: np.ndarray,
                             qubit1: int, qubit2: int, n_qubits: int) -> np.ndarray:
        """Apply two-qubit gate"""
        new_state = np.zeros_like(state)
        
        # Create bit masks
        mask1 = 1 << (n_qubits - qubit1 - 1)
        mask2 = 1 << (n_qubits - qubit2 - 1)
        
        for i in range(len(state)):
            # Extract bit values
            bit1 = (i & mask1) >> (n_qubits - qubit1 - 1)
            bit2 = (i & mask2) >> (n_qubits - qubit2 - 1)
            
            # Combined bit value for indexing gate matrix
            combined_bits = (bit1 << 1) | bit2
            
            # Apply gate
            for new_combined in range(4):
                new_bit1 = (new_combined >> 1) & 1
                new_bit2 = new_combined & 1
                
                # Calculate new index
                new_index = i
                new_index = (new_index & ~mask1) | (new_bit1 << (n_qubits - qubit1 - 1))
                new_index = (new_index & ~mask2) | (new_bit2 << (n_qubits - qubit2 - 1))
                
                new_state[new_index] += gate_matrix[new_combined, combined_bits] * state[i]
        
        return new_state
    
    def _apply_three_qubit_gate(self, state: np.ndarray, gate_matrix: np.ndarray,
                               qubits: List[int], n_qubits: int) -> np.ndarray:
        """Apply three-qubit gate"""
        new_state = np.zeros_like(state)
        
        # Create bit masks
        masks = [1 << (n_qubits - q - 1) for q in qubits]
        
        for i in range(len(state)):
            # Extract bit values
            bits = [(i & mask) >> (n_qubits - qubits[j] - 1) 
                   for j, mask in enumerate(masks)]
            
            # Combined bit value
            combined_bits = (bits[0] << 2) | (bits[1] << 1) | bits[2]
            
            # Apply gate
            for new_combined in range(8):
                new_bits = [(new_combined >> (2-j)) & 1 for j in range(3)]
                
                # Calculate new index
                new_index = i
                for j, (qubit, new_bit) in enumerate(zip(qubits, new_bits)):
                    new_index = (new_index & ~masks[j]) | (new_bit << (n_qubits - qubit - 1))
                
                new_state[new_index] += gate_matrix[new_combined, combined_bits] * state[i]
        
        return new_state
    
    def simulate(self, circuit: QuantumCircuit, 
                initial_state: Optional[str] = None) -> Dict[str, Any]:
        """
        Simulate quantum circuit
        
        Args:
            circuit: Quantum circuit to simulate
            initial_state: Initial state as binary string
            
        Returns:
            Simulation results including final state and measurements
        """
        start_time = time.time()
        
        # Initialize state
        state = self.initialize_state(circuit.n_qubits, initial_state)
        
        # Apply gates
        for gate, qubits, parameter in circuit.gates:
            state = self.apply_gate(state, gate, qubits, parameter)
        
        # Perform measurements if specified
        measurement_results = {}
        if circuit.measurements:
            measurement_results = self.measure(state, circuit.measurements)
        
        simulation_time = time.time() - start_time
        
        return {
            'final_state': state,
            'measurements': measurement_results,
            'simulation_time': simulation_time,
            'circuit_depth': circuit.depth(),
            'n_gates': len(circuit.gates)
        }
    
    def measure(self, state: np.ndarray, qubits: List[int], 
               n_shots: int = 1000) -> Dict[str, Any]:
        """
        Perform measurement on specified qubits
        
        Args:
            state: Quantum state
            qubits: Qubits to measure
            n_shots: Number of measurement shots
            
        Returns:
            Measurement results
        """
        n_qubits = int(np.log2(len(state)))
        probabilities = np.abs(state) ** 2
        
        # Sample from probability distribution
        outcomes = np.random.choice(len(state), size=n_shots, p=probabilities)
        
        # Extract measured qubit values
        measured_values = {}
        for outcome in outcomes:
            # Extract bits for measured qubits
            measured_bits = ""
            for qubit in sorted(qubits):
                bit = (outcome >> (n_qubits - qubit - 1)) & 1
                measured_bits += str(bit)
            
            measured_values[measured_bits] = measured_values.get(measured_bits, 0) + 1
        
        # Calculate probabilities from counts
        measured_probabilities = {
            bits: count / n_shots 
            for bits, count in measured_values.items()
        }
        
        return {
            'counts': measured_values,
            'probabilities': measured_probabilities,
            'n_shots': n_shots
        }
    
    def get_expectation_value(self, state: np.ndarray, 
                            observable: np.ndarray) -> float:
        """
        Calculate expectation value of observable
        
        Args:
            state: Quantum state
            observable: Hermitian observable matrix
            
        Returns:
            Expectation value
        """
        return np.real(np.vdot(state, observable @ state))
    
    def get_entanglement_entropy(self, state: np.ndarray, 
                               partition: List[int]) -> float:
        """
        Calculate entanglement entropy across partition
        
        Args:
            state: Quantum state
            partition: List of qubit indices in first partition
            
        Returns:
            Entanglement entropy
        """
        n_qubits = int(np.log2(len(state)))
        n_partition = len(partition)
        n_rest = n_qubits - n_partition
        
        # Reshape state to matrix
        state_matrix = state.reshape(2**n_partition, 2**n_rest)
        
        # Compute reduced density matrix
        reduced_density = state_matrix @ state_matrix.conj().T
        
        # Calculate von Neumann entropy
        eigenvalues = np.linalg.eigvalsh(reduced_density)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]  # Remove zeros
        
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        
        return entropy


class NoiseModel:
    """
    Noise model for realistic quantum simulation
    """
    
    def __init__(self, error_rates: Dict[str, float]):
        self.error_rates = error_rates
        self.kraus_operators = self._initialize_kraus_operators()
        
    def _initialize_kraus_operators(self) -> Dict[str, List[np.ndarray]]:
        """Initialize Kraus operators for various noise channels"""
        operators = {}
        
        # Depolarizing channel
        p_depol = self.error_rates.get('depolarizing', 0.001)
        operators['depolarizing'] = self._depolarizing_kraus(p_depol)
        
        # Amplitude damping
        gamma = self.error_rates.get('amplitude_damping', 0.01)
        operators['amplitude_damping'] = self._amplitude_damping_kraus(gamma)
        
        # Phase damping
        lambda_phase = self.error_rates.get('phase_damping', 0.01)
        operators['phase_damping'] = self._phase_damping_kraus(lambda_phase)
        
        return operators
    
    def _depolarizing_kraus(self, p: float) -> List[np.ndarray]:
        """Kraus operators for depolarizing channel"""
        # Pauli matrices
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Kraus operators
        K0 = np.sqrt(1 - 3*p/4) * I
        K1 = np.sqrt(p/4) * X
        K2 = np.sqrt(p/4) * Y
        K3 = np.sqrt(p/4) * Z
        
        return [K0, K1, K2, K3]
    
    def _amplitude_damping_kraus(self, gamma: float) -> List[np.ndarray]:
        """Kraus operators for amplitude damping"""
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
        
        return [K0, K1]
    
    def _phase_damping_kraus(self, lambda_: float) -> List[np.ndarray]:
        """Kraus operators for phase damping"""
        K0 = np.array([[1, 0], [0, np.sqrt(1 - lambda_)]], dtype=complex)
        K1 = np.array([[0, 0], [0, np.sqrt(lambda_)]], dtype=complex)
        
        return [K0, K1]
    
    def apply_noise(self, state: np.ndarray, gate_type: QuantumGate, 
                   qubits: List[int]) -> np.ndarray:
        """Apply noise after gate operation"""
        # Apply depolarizing noise to each qubit
        for qubit in qubits:
            state = self._apply_single_qubit_noise(
                state, self.kraus_operators['depolarizing'], qubit
            )
        
        # Additional noise for two-qubit gates
        if len(qubits) == 2:
            # Higher error rate for two-qubit gates
            extra_error = self.error_rates.get('two_qubit_extra', 0.01)
            if extra_error > 0:
                for qubit in qubits:
                    state = self._apply_single_qubit_noise(
                        state, self._depolarizing_kraus(extra_error), qubit
                    )
        
        return state
    
    def _apply_single_qubit_noise(self, state: np.ndarray, 
                                 kraus_ops: List[np.ndarray], 
                                 qubit: int) -> np.ndarray:
        """Apply single-qubit noise channel"""
        n_qubits = int(np.log2(len(state)))
        new_state = np.zeros_like(state)
        
        # Apply each Kraus operator
        for K in kraus_ops:
            # Create full operator
            full_K = self._expand_operator(K, qubit, n_qubits)
            new_state += full_K @ state
        
        return new_state
    
    def _expand_operator(self, op: np.ndarray, qubit: int, 
                        n_qubits: int) -> np.ndarray:
        """Expand single-qubit operator to full Hilbert space"""
        # Use tensor product to expand operator
        full_op = 1
        
        for i in range(n_qubits):
            if i == qubit:
                full_op = np.kron(full_op, op)
            else:
                full_op = np.kron(full_op, np.eye(2))
        
        return full_op


class QuantumAlgorithmTester:
    """
    Test harness for quantum algorithms
    """
    
    def __init__(self, simulator: QuantumSimulator):
        self.simulator = simulator
        self.test_results = []
        
    def test_bell_state_preparation(self) -> Dict[str, Any]:
        """Test Bell state preparation"""
        circuit = QuantumCircuit(n_qubits=2)
        
        # Create Bell state |Φ+> = (|00> + |11>)/√2
        circuit.add_gate(QuantumGate.H, [0])
        circuit.add_gate(QuantumGate.CNOT, [0, 1])
        
        result = self.simulator.simulate(circuit)
        state = result['final_state']
        
        # Check if it's a Bell state
        expected_state = np.zeros(4, dtype=complex)
        expected_state[0] = 1/np.sqrt(2)  # |00>
        expected_state[3] = 1/np.sqrt(2)  # |11>
        
        fidelity = np.abs(np.vdot(state, expected_state)) ** 2
        
        return {
            'test': 'bell_state_preparation',
            'passed': fidelity > 0.99,
            'fidelity': fidelity,
            'state': state
        }
    
    def test_grover_search(self, n_qubits: int = 3, 
                          marked_item: int = 5) -> Dict[str, Any]:
        """Test Grover's search algorithm"""
        N = 2 ** n_qubits
        n_iterations = int(np.pi / 4 * np.sqrt(N))
        
        circuit = QuantumCircuit(n_qubits=n_qubits)
        
        # Initialize in equal superposition
        for i in range(n_qubits):
            circuit.add_gate(QuantumGate.H, [i])
        
        # Grover iterations
        for _ in range(n_iterations):
            # Oracle (mark the item)
            self._add_oracle(circuit, marked_item, n_qubits)
            
            # Diffusion operator
            self._add_diffusion(circuit, n_qubits)
        
        # Measure all qubits
        for i in range(n_qubits):
            circuit.add_measurement(i)
        
        result = self.simulator.simulate(circuit)
        measurements = result['measurements']
        
        # Check if marked item has highest probability
        marked_binary = format(marked_item, f'0{n_qubits}b')
        success_probability = measurements['probabilities'].get(marked_binary, 0)
        
        return {
            'test': 'grover_search',
            'passed': success_probability > 0.5,
            'success_probability': success_probability,
            'marked_item': marked_item,
            'measurements': measurements
        }
    
    def _add_oracle(self, circuit: QuantumCircuit, marked_item: int, 
                   n_qubits: int):
        """Add oracle for Grover's algorithm"""
        # Simple phase oracle - flips phase of marked item
        marked_binary = format(marked_item, f'0{n_qubits}b')
        
        # Add X gates to flip qubits that should be 0
        for i, bit in enumerate(marked_binary):
            if bit == '0':
                circuit.add_gate(QuantumGate.X, [i])
        
        # Multi-controlled Z gate (simplified)
        if n_qubits == 2:
            circuit.add_gate(QuantumGate.CZ, [0, 1])
        elif n_qubits == 3:
            # CCZ using Toffoli and phase gates
            circuit.add_gate(QuantumGate.H, [2])
            circuit.add_gate(QuantumGate.TOFFOLI, [0, 1, 2])
            circuit.add_gate(QuantumGate.H, [2])
        
        # Undo X gates
        for i, bit in enumerate(marked_binary):
            if bit == '0':
                circuit.add_gate(QuantumGate.X, [i])
    
    def _add_diffusion(self, circuit: QuantumCircuit, n_qubits: int):
        """Add diffusion operator for Grover's algorithm"""
        # Apply Hadamard gates
        for i in range(n_qubits):
            circuit.add_gate(QuantumGate.H, [i])
        
        # Apply X gates
        for i in range(n_qubits):
            circuit.add_gate(QuantumGate.X, [i])
        
        # Multi-controlled Z gate
        if n_qubits == 2:
            circuit.add_gate(QuantumGate.CZ, [0, 1])
        elif n_qubits == 3:
            circuit.add_gate(QuantumGate.H, [2])
            circuit.add_gate(QuantumGate.TOFFOLI, [0, 1, 2])
            circuit.add_gate(QuantumGate.H, [2])
        
        # Undo X gates
        for i in range(n_qubits):
            circuit.add_gate(QuantumGate.X, [i])
        
        # Apply Hadamard gates
        for i in range(n_qubits):
            circuit.add_gate(QuantumGate.H, [i])
    
    def test_quantum_fourier_transform(self, n_qubits: int = 3) -> Dict[str, Any]:
        """Test Quantum Fourier Transform"""
        circuit = QuantumCircuit(n_qubits=n_qubits)
        
        # Initialize in computational basis state |5>
        initial_state = format(5, f'0{n_qubits}b')
        
        # Apply QFT
        self._add_qft(circuit, n_qubits)
        
        result = self.simulator.simulate(circuit, initial_state=initial_state)
        state = result['final_state']
        
        # Verify QFT properties
        # The QFT of |5> should have specific phase relationships
        success = self._verify_qft_output(state, 5, n_qubits)
        
        return {
            'test': 'quantum_fourier_transform',
            'passed': success,
            'initial_state': initial_state,
            'final_state': state
        }
    
    def _add_qft(self, circuit: QuantumCircuit, n_qubits: int):
        """Add Quantum Fourier Transform to circuit"""
        for i in range(n_qubits):
            # Hadamard on qubit i
            circuit.add_gate(QuantumGate.H, [i])
            
            # Controlled phase rotations
            for j in range(i + 1, n_qubits):
                angle = np.pi / (2 ** (j - i))
                # Using CZ as approximation (should use controlled phase)
                circuit.add_gate(QuantumGate.CZ, [j, i])
        
        # Swap qubits to get correct order
        for i in range(n_qubits // 2):
            circuit.add_gate(QuantumGate.SWAP, [i, n_qubits - i - 1])
    
    def _verify_qft_output(self, state: np.ndarray, input_value: int, 
                          n_qubits: int) -> bool:
        """Verify QFT output has correct properties"""
        N = 2 ** n_qubits
        
        # Expected QFT output for computational basis input
        expected = np.zeros(N, dtype=complex)
        for k in range(N):
            phase = 2 * np.pi * input_value * k / N
            expected[k] = np.exp(1j * phase) / np.sqrt(N)
        
        # Calculate fidelity
        fidelity = np.abs(np.vdot(state, expected)) ** 2
        
        return fidelity > 0.95
    
    def test_variational_circuit(self, n_qubits: int = 4, 
                               n_layers: int = 2) -> Dict[str, Any]:
        """Test variational quantum circuit"""
        # Random parameters
        n_params = n_qubits * n_layers * 3  # 3 rotation gates per qubit per layer
        params = np.random.randn(n_params) * 0.5
        
        circuit = QuantumCircuit(n_qubits=n_qubits)
        
        param_idx = 0
        for layer in range(n_layers):
            # Rotation layer
            for qubit in range(n_qubits):
                circuit.add_gate(QuantumGate.RX, [qubit], params[param_idx])
                param_idx += 1
                circuit.add_gate(QuantumGate.RY, [qubit], params[param_idx])
                param_idx += 1
                circuit.add_gate(QuantumGate.RZ, [qubit], params[param_idx])
                param_idx += 1
            
            # Entangling layer
            for qubit in range(n_qubits - 1):
                circuit.add_gate(QuantumGate.CNOT, [qubit, qubit + 1])
        
        # Measure all qubits
        for i in range(n_qubits):
            circuit.add_measurement(i)
        
        result = self.simulator.simulate(circuit)
        
        # Check circuit executed successfully
        success = result['simulation_time'] < 1.0  # Should be fast
        
        return {
            'test': 'variational_circuit',
            'passed': success,
            'n_parameters': n_params,
            'circuit_depth': result['circuit_depth'],
            'measurements': result['measurements']
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all quantum algorithm tests"""
        tests = [
            self.test_bell_state_preparation(),
            self.test_grover_search(n_qubits=3, marked_item=5),
            self.test_quantum_fourier_transform(n_qubits=3),
            self.test_variational_circuit(n_qubits=4, n_layers=2)
        ]
        
        passed = sum(1 for test in tests if test['passed'])
        
        return {
            'total_tests': len(tests),
            'passed': passed,
            'failed': len(tests) - passed,
            'test_results': tests
        }


# Example circuits for quantum algorithms
class QuantumCircuitLibrary:
    """Library of common quantum circuits"""
    
    @staticmethod
    def create_ghz_circuit(n_qubits: int) -> QuantumCircuit:
        """Create GHZ state circuit"""
        circuit = QuantumCircuit(n_qubits=n_qubits)
        
        # Create GHZ state
        circuit.add_gate(QuantumGate.H, [0])
        for i in range(1, n_qubits):
            circuit.add_gate(QuantumGate.CNOT, [0, i])
        
        return circuit
    
    @staticmethod
    def create_qpe_circuit(n_precision_qubits: int, 
                          unitary_gate: QuantumGate) -> QuantumCircuit:
        """Create Quantum Phase Estimation circuit"""
        n_qubits = n_precision_qubits + 1
        circuit = QuantumCircuit(n_qubits=n_qubits)
        
        # Initialize precision qubits in superposition
        for i in range(n_precision_qubits):
            circuit.add_gate(QuantumGate.H, [i])
        
        # Controlled unitary operations
        for i in range(n_precision_qubits):
            for _ in range(2**i):
                # Simplified - should apply controlled version of unitary
                circuit.add_gate(QuantumGate.CZ, [i, n_precision_qubits])
        
        # Inverse QFT on precision qubits
        # (simplified implementation)
        for i in range(n_precision_qubits):
            circuit.add_gate(QuantumGate.H, [i])
        
        return circuit
    
    @staticmethod
    def create_vqe_ansatz(n_qubits: int, n_layers: int, 
                         parameters: List[float]) -> QuantumCircuit:
        """Create VQE ansatz circuit"""
        circuit = QuantumCircuit(n_qubits=n_qubits)
        
        param_idx = 0
        for layer in range(n_layers):
            # Single-qubit rotations
            for qubit in range(n_qubits):
                if param_idx < len(parameters):
                    circuit.add_gate(QuantumGate.RY, [qubit], parameters[param_idx])
                    param_idx += 1
            
            # Entangling gates
            for qubit in range(0, n_qubits - 1, 2):
                circuit.add_gate(QuantumGate.CNOT, [qubit, qubit + 1])
            
            for qubit in range(1, n_qubits - 1, 2):
                circuit.add_gate(QuantumGate.CNOT, [qubit, qubit + 1])
        
        return circuit


def demonstrate_quantum_simulator():
    """Demonstrate quantum simulator capabilities"""
    
    logger.info("=== Quantum Circuit Simulator Demonstration ===")
    
    # Initialize simulator
    simulator = QuantumSimulator(backend="numpy")
    
    # 1. Basic circuit simulation
    logger.info("\n1. Basic Circuit Simulation")
    circuit = QuantumCircuit(n_qubits=3)
    circuit.add_gate(QuantumGate.H, [0])
    circuit.add_gate(QuantumGate.CNOT, [0, 1])
    circuit.add_gate(QuantumGate.CNOT, [1, 2])
    
    result = simulator.simulate(circuit)
    logger.info(f"Circuit depth: {result['circuit_depth']}")
    logger.info(f"Simulation time: {result['simulation_time']:.4f}s")
    
    # 2. Test quantum algorithms
    logger.info("\n2. Quantum Algorithm Tests")
    tester = QuantumAlgorithmTester(simulator)
    test_results = tester.run_all_tests()
    
    logger.info(f"Tests passed: {test_results['passed']}/{test_results['total_tests']}")
    for test in test_results['test_results']:
        logger.info(f"  {test['test']}: {'PASSED' if test['passed'] else 'FAILED'}")
    
    # 3. Noise simulation
    logger.info("\n3. Noise Simulation")
    noise_model = NoiseModel({
        'depolarizing': 0.001,
        'amplitude_damping': 0.01,
        'two_qubit_extra': 0.01
    })
    
    # Create noisy simulator
    noisy_sim = QuantumSimulator(backend="numpy")
    
    # Compare clean vs noisy Bell state
    bell_circuit = QuantumCircuitLibrary.create_ghz_circuit(2)
    
    clean_result = simulator.simulate(bell_circuit)
    clean_state = clean_result['final_state']
    
    # Apply noise manually (simplified)
    noisy_state = clean_state.copy()
    noisy_state = noise_model.apply_noise(
        noisy_state, QuantumGate.CNOT, [0, 1]
    )
    
    logger.info(f"Clean state fidelity: 1.000")
    logger.info(f"Noisy state fidelity: {np.abs(np.vdot(clean_state, noisy_state))**2:.3f}")
    
    # 4. Quantum circuit library
    logger.info("\n4. Quantum Circuit Library")
    
    # GHZ state
    ghz_circuit = QuantumCircuitLibrary.create_ghz_circuit(4)
    ghz_result = simulator.simulate(ghz_circuit)
    
    # Check GHZ properties
    ghz_state = ghz_result['final_state']
    ghz_prob_00 = np.abs(ghz_state[0])**2
    ghz_prob_11 = np.abs(ghz_state[-1])**2
    
    logger.info(f"4-qubit GHZ state:")
    logger.info(f"  P(0000) = {ghz_prob_00:.3f}")
    logger.info(f"  P(1111) = {ghz_prob_11:.3f}")
    logger.info(f"  Total = {ghz_prob_00 + ghz_prob_11:.3f}")
    
    # VQE ansatz
    vqe_params = np.random.randn(12) * 0.5
    vqe_circuit = QuantumCircuitLibrary.create_vqe_ansatz(4, 2, vqe_params)
    vqe_result = simulator.simulate(vqe_circuit)
    
    logger.info(f"\nVQE ansatz circuit:")
    logger.info(f"  Parameters: {len(vqe_params)}")
    logger.info(f"  Circuit depth: {vqe_result['circuit_depth']}")
    logger.info(f"  Gates: {vqe_result['n_gates']}")
    
    logger.info("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    demonstrate_quantum_simulator()