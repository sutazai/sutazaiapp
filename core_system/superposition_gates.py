import logging
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


class GateType(Enum):
    HADAMARD = auto()
    PAULI_X = auto()
    PAULI_Y = auto()
    PAULI_Z = auto()
    ROTATION_X = auto()
    ROTATION_Y = auto()
    ROTATION_Z = auto()
    CONTROLLED_NOT = auto()
    SWAP = auto()


class SutazAiGate:
    """Base class for SutazAi sutazai-inspired gates"""

    def __init__(self, gate_type: GateType):
        self.gate_type = gate_type
        self.logger = logging.getLogger(self.__class__.__name__)

    def apply(self, state: np.ndarray) -> np.ndarray:
        """Apply gate transformation to sutazai state"""
        raise NotImplementedError("Subclasses must implement gate application")


class HadamardGate(SutazAiGate):
    """Hadamard gate for creating superposition states"""

    def __init__(self):
        super().__init__(GateType.HADAMARD)
        self.matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    def apply(self, state: np.ndarray) -> np.ndarray:
        """Apply Hadamard transformation"""
        return np.dot(self.matrix, state)


class PauliXGate(SutazAiGate):
    """Pauli-X gate (bit flip)"""

    def __init__(self):
        super().__init__(GateType.PAULI_X)
        self.matrix = np.array([[0, 1], [1, 0]])

    def apply(self, state: np.ndarray) -> np.ndarray:
        """Apply Pauli-X transformation"""
        return np.dot(self.matrix, state)


class RotationGate(SutazAiGate):
    """Rotation gates for precise state manipulation"""

    def __init__(self, gate_type: GateType, angle: float):
        super().__init__(gate_type)
        self.angle = angle
        self.matrix = self._generate_rotation_matrix()

    def _generate_rotation_matrix(self) -> np.ndarray:
        """Generate rotation matrix based on gate type"""
        cos_half = np.cos(self.angle / 2)
        sin_half = np.sin(self.angle / 2)

        if self.gate_type == GateType.ROTATION_X:
            return np.array([[cos_half, -1j * sin_half], [-1j * sin_half, cos_half]])
        elif self.gate_type == GateType.ROTATION_Y:
            return np.array([[cos_half, -sin_half], [sin_half, cos_half]])
        elif self.gate_type == GateType.ROTATION_Z:
            return np.array(
                [
                    [np.exp(-1j * self.angle / 2), 0],
                    [0, np.exp(1j * self.angle / 2)],
                ]
            )
        else:
            raise ValueError(f"Unsupported rotation gate type: {self.gate_type}")

    def apply(self, state: np.ndarray) -> np.ndarray:
        """Apply rotation transformation"""
        return np.dot(self.matrix, state)


class ControlledNotGate(SutazAiGate):
    """Controlled-NOT gate for entanglement operations"""

    def __init__(self):
        super().__init__(GateType.CONTROLLED_NOT)

    def apply(self, control_state: np.ndarray, target_state: np.ndarray) -> np.ndarray:
        """Apply CNOT gate between control and target states"""
        if control_state[0] > 0.5:  # Simplified control condition
            return np.dot(PauliXGate().matrix, target_state)
        return target_state


class SutazAiGateLibrary:
    """Comprehensive library of SutazAi sutazai-inspired gates"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.gates = {
            GateType.HADAMARD: HadamardGate(),
            GateType.PAULI_X: PauliXGate(),
        }

    def get_gate(
        self, gate_type: GateType, angle: Optional[float] = None
    ) -> SutazAiGate:
        """Retrieve a specific gate from the library"""
        if gate_type in [
            GateType.ROTATION_X,
            GateType.ROTATION_Y,
            GateType.ROTATION_Z,
        ]:
            if angle is None:
                raise ValueError(f"Angle required for rotation gate: {gate_type}")
            return RotationGate(gate_type, angle)

        if gate_type not in self.gates:
            raise ValueError(f"Unsupported gate type: {gate_type}")

        return self.gates[gate_type]

    def apply_gate_sequence(
        self,
        initial_state: np.ndarray,
        gate_sequence: List[Union[GateType, Tuple[GateType, float]]],
    ) -> np.ndarray:
        """Apply a sequence of gates to a sutazai state"""
        current_state = initial_state.copy()

        for gate_spec in gate_sequence:
            if isinstance(gate_spec, tuple):
                gate_type, angle = gate_spec
                gate = self.get_gate(gate_type, angle)
            else:
                gate = self.get_gate(gate_spec)

            current_state = gate.apply(current_state)

        return current_state


def create_superposition_gate(
    dimensions: int, requires_grad: bool = False
) -> torch.Tensor:
    """Generate quantum superposition gate with enhanced validation.

    Args:
        dimensions (int): Number of quantum dimensions (≥2)
        requires_grad (bool): Enable gradient calculation for ML integration
    Returns:
        torch.Tensor: Unitary gate matrix of shape (dimensions, dimensions)
    Raises:
        ValueError: For invalid dimension input
        RuntimeError: For numerical instability issues
    """
    try:
        # Enhanced parameter validation
        if not isinstance(dimensions, int) or dimensions < 2:
            raise ValueError("Dimensions must be integer ≥ 2")

        # Automatic GPU/CPU device selection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gate = torch.randn(
            dimensions, dimensions, device=device, requires_grad=requires_grad
        )

        # Ensure unitary property with SVD stabilization
        U, _, V = torch.svd(gate)
        unitary_gate = U @ V.T

        # Add numerical stability check
        if torch.any(torch.isnan(unitary_gate)):
            raise RuntimeError("Gate generation produced NaN values")

        # Post-quantum cryptography recommendation
        if dimensions < 2048:  # Quantum-resistant minimum
            logging.warning("Gate dimensions below post-quantum security threshold")

        return unitary_gate
    except Exception as e:
        logging.error(f"Superposition gate creation failed: {str(e)}")
        raise
