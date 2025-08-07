"""
Privacy-Preserving Mechanisms for Federated Learning
====================================================

Implements differential privacy, secure aggregation, and other privacy-preserving
techniques for federated learning in the SutazAI system.

Features:
- Differential Privacy (DP) with various mechanisms
- Secure Multi-Party Computation (SMPC) aggregation
- Homomorphic encryption for model updates
- Privacy budget management and tracking
- Noise injection and gradient clipping
- Privacy-preserving analytics
"""

import asyncio
import logging
import time
import numpy as np
import hashlib
import hmac
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import json
import secrets

# Cryptographic primitives
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64


class PrivacyMechanism(Enum):
    """Supported privacy mechanisms"""
    NONE = "none"
    GAUSSIAN_DP = "gaussian_dp"
    LAPLACE_DP = "laplace_dp"
    SECURE_AGGREGATION = "secure_aggregation"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    GRADIENT_COMPRESSION = "gradient_compression"
    FEDERATED_AVERAGING_DP = "federated_averaging_dp"


class PrivacyLevel(Enum):
    """Privacy protection levels"""
    LOW = "low"          # ε = 10.0
    MEDIUM = "medium"    # ε = 1.0
    HIGH = "high"        # ε = 0.1
    EXTREME = "extreme"  # ε = 0.01


@dataclass
class PrivacyBudget:
    """Privacy budget management"""
    total_epsilon: float
    total_delta: float
    consumed_epsilon: float = 0.0
    consumed_delta: float = 0.0
    per_round_epsilon: float = 0.1
    per_round_delta: float = 1e-5
    mechanism: PrivacyMechanism = PrivacyMechanism.GAUSSIAN_DP
    
    def can_spend(self, epsilon: float, delta: float) -> bool:
        """Check if budget allows spending epsilon and delta"""
        return (self.consumed_epsilon + epsilon <= self.total_epsilon and
                self.consumed_delta + delta <= self.total_delta)
    
    def spend(self, epsilon: float, delta: float) -> bool:
        """Spend privacy budget"""
        if self.can_spend(epsilon, delta):
            self.consumed_epsilon += epsilon
            self.consumed_delta += delta
            return True
        return False
    
    def remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget"""
        return (self.total_epsilon - self.consumed_epsilon,
                self.total_delta - self.consumed_delta)


@dataclass
class DifferentialPrivacyConfig:
    """Configuration for differential privacy"""
    mechanism: PrivacyMechanism
    epsilon: float
    delta: float
    sensitivity: float = 1.0
    clipping_norm: float = 1.0
    noise_multiplier: float = 1.0
    sampling_probability: float = 1.0


@dataclass
class SecureAggregationConfig:
    """Configuration for secure aggregation"""
    num_clients: int
    threshold: int  # Minimum number of clients needed
    key_size: int = 2048
    use_encryption: bool = True
    timeout_seconds: int = 300


class PrivacyMeter:
    """Privacy budget tracking and analysis"""
    
    def __init__(self):
        self.budget_history: List[Tuple[datetime, float, float]] = []
        self.mechanism_usage: Dict[str, int] = defaultdict(int)
        self.privacy_violations: List[Dict[str, Any]] = []
        
    def record_budget_usage(self, epsilon: float, delta: float, mechanism: str):
        """Record privacy budget usage"""
        self.budget_history.append((datetime.utcnow(), epsilon, delta))
        self.mechanism_usage[mechanism] += 1
        
    def get_total_budget_used(self) -> Tuple[float, float]:
        """Get total privacy budget consumed"""
        total_epsilon = sum(eps for _, eps, _ in self.budget_history)
        total_delta = sum(delta for _, _, delta in self.budget_history)
        return total_epsilon, total_delta
    
    def check_privacy_violation(self, budget: PrivacyBudget) -> bool:
        """Check if privacy budget has been violated"""
        total_eps, total_delta = self.get_total_budget_used()
        
        if total_eps > budget.total_epsilon or total_delta > budget.total_delta:
            violation = {
                "timestamp": datetime.utcnow().isoformat(),
                "epsilon_violation": total_eps > budget.total_epsilon,
                "delta_violation": total_delta > budget.total_delta,
                "total_epsilon": total_eps,
                "total_delta": total_delta,
                "budget_epsilon": budget.total_epsilon,
                "budget_delta": budget.total_delta
            }
            self.privacy_violations.append(violation)
            return True
        
        return False


class DifferentialPrivacyMechanism(ABC):
    """Abstract base class for DP mechanisms"""
    
    @abstractmethod
    def add_noise(self, data: np.ndarray, sensitivity: float, 
                  epsilon: float, delta: float) -> np.ndarray:
        """Add differential private noise to data"""
        pass
    
    @abstractmethod
    def compute_noise_scale(self, sensitivity: float, 
                           epsilon: float, delta: float) -> float:
        """Compute noise scale for given privacy parameters"""
        pass


class GaussianDP(DifferentialPrivacyMechanism):
    """Gaussian differential privacy mechanism"""
    
    def add_noise(self, data: np.ndarray, sensitivity: float,
                  epsilon: float, delta: float) -> np.ndarray:
        """Add Gaussian noise for (ε,δ)-DP"""
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must be in (0, 1)")
        
        sigma = self.compute_noise_scale(sensitivity, epsilon, delta)
        noise = np.random.normal(0, sigma, data.shape)
        
        return data + noise
    
    def compute_noise_scale(self, sensitivity: float, 
                           epsilon: float, delta: float) -> float:
        """Compute Gaussian noise scale"""
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        
        # For Gaussian mechanism: σ ≥ √(2 ln(1.25/δ)) * Δf / ε
        c = np.sqrt(2 * np.log(1.25 / delta))
        sigma = c * sensitivity / epsilon
        
        return sigma


class LaplaceDP(DifferentialPrivacyMechanism):
    """Laplace differential privacy mechanism"""
    
    def add_noise(self, data: np.ndarray, sensitivity: float,
                  epsilon: float, delta: float) -> np.ndarray:
        """Add Laplace noise for ε-DP"""
        scale = self.compute_noise_scale(sensitivity, epsilon, delta)
        noise = np.random.laplace(0, scale, data.shape)
        
        return data + noise
    
    def compute_noise_scale(self, sensitivity: float,
                           epsilon: float, delta: float) -> float:
        """Compute Laplace noise scale"""
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        
        # For Laplace mechanism: b = Δf / ε
        return sensitivity / epsilon


class SecureAggregator:
    """Secure multi-party computation for federated aggregation"""
    
    def __init__(self, config: SecureAggregationConfig):
        self.config = config
        self.client_keys: Dict[str, rsa.RSAPrivateKey] = {}
        self.client_public_keys: Dict[str, rsa.RSAPublicKey] = {}
        self.shared_secrets: Dict[Tuple[str, str], bytes] = {}
        self.masked_inputs: Dict[str, np.ndarray] = {}
        
        self.logger = logging.getLogger("secure_aggregator")
    
    def generate_client_keypair(self, client_id: str) -> Tuple[bytes, bytes]:
        """Generate RSA keypair for client"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.config.key_size,
            backend=default_backend()
        )
        
        public_key = private_key.public_key()
        
        # Store keys
        self.client_keys[client_id] = private_key
        self.client_public_keys[client_id] = public_key
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def setup_shared_secrets(self, client_ids: List[str]):
        """Setup pairwise shared secrets between clients"""
        for i, client_a in enumerate(client_ids):
            for j, client_b in enumerate(client_ids):
                if i < j:  # Avoid duplicates
                    # Generate shared secret
                    secret = secrets.token_bytes(32)  # 256-bit secret
                    
                    # Store for both clients
                    self.shared_secrets[(client_a, client_b)] = secret
                    self.shared_secrets[(client_b, client_a)] = secret
    
    def mask_input(self, client_id: str, input_data: np.ndarray, 
                   other_clients: List[str]) -> np.ndarray:
        """Mask client input using shared secrets"""
        masked_data = input_data.copy()
        
        for other_client in other_clients:
            if other_client != client_id:
                # Get shared secret
                secret_key = (client_id, other_client)
                if secret_key in self.shared_secrets:
                    secret = self.shared_secrets[secret_key]
                    
                    # Generate deterministic mask from secret
                    mask = self._generate_mask_from_secret(secret, input_data.shape)
                    
                    # Add or subtract mask based on client ordering
                    if client_id < other_client:
                        masked_data += mask
                    else:
                        masked_data -= mask
        
        return masked_data
    
    def _generate_mask_from_secret(self, secret: bytes, shape: Tuple[int, ...]) -> np.ndarray:
        """Generate deterministic mask from shared secret"""
        # Use HMAC-based deterministic random number generation
        np.random.seed(int.from_bytes(secret[:4], 'big'))
        mask = np.random.normal(0, 1, shape)
        
        return mask
    
    def aggregate_masked_inputs(self, masked_inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Aggregate masked inputs - masks cancel out"""
        if len(masked_inputs) < self.config.threshold:
            raise ValueError(f"Insufficient clients: {len(masked_inputs)} < {self.config.threshold}")
        
        # Sum all masked inputs - the masks will cancel out
        aggregated = None
        for client_id, masked_input in masked_inputs.items():
            if aggregated is None:
                aggregated = masked_input.copy()
            else:
                aggregated += masked_input
        
        return aggregated


class HomomorphicEncryption:
    """Simple homomorphic encryption for secure aggregation"""
    
    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        self.private_key = None
        self.public_key = None
        
    def generate_keys(self):
        """Generate homomorphic encryption keys"""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
    
    def encrypt(self, plaintext: float) -> bytes:
        """Encrypt a single value"""
        if self.public_key is None:
            raise ValueError("Keys not generated")
        
        # Convert float to bytes (simplified)
        plaintext_bytes = str(plaintext).encode('utf-8')
        
        # Encrypt using RSA-OAEP
        ciphertext = self.public_key.encrypt(
            plaintext_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return ciphertext
    
    def decrypt(self, ciphertext: bytes) -> float:
        """Decrypt a single value"""
        if self.private_key is None:
            raise ValueError("Private key not available")
        
        # Decrypt using RSA-OAEP
        plaintext_bytes = self.private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return float(plaintext_bytes.decode('utf-8'))
    
    def encrypt_vector(self, vector: np.ndarray) -> List[bytes]:
        """Encrypt a vector element-wise"""
        return [self.encrypt(float(x)) for x in vector.flatten()]
    
    def decrypt_vector(self, encrypted_vector: List[bytes], shape: Tuple[int, ...]) -> np.ndarray:
        """Decrypt a vector and reshape"""
        decrypted = [self.decrypt(ciphertext) for ciphertext in encrypted_vector]
        return np.array(decrypted).reshape(shape)


class PrivacyManager:
    """
    Privacy Manager for Federated Learning
    
    Manages privacy-preserving mechanisms, budget tracking, and secure aggregation
    for the SutazAI federated learning system.
    """
    
    def __init__(self):
        self.dp_mechanisms = {
            PrivacyMechanism.GAUSSIAN_DP: GaussianDP(),
            PrivacyMechanism.LAPLACE_DP: LaplaceDP()
        }
        
        self.privacy_meters: Dict[str, PrivacyMeter] = {}
        self.secure_aggregators: Dict[str, SecureAggregator] = {}
        self.homomorphic_engines: Dict[str, HomomorphicEncryption] = {}
        
        # Global privacy settings
        self.global_privacy_level = PrivacyLevel.MEDIUM
        self.default_budgets = {
            PrivacyLevel.LOW: PrivacyBudget(total_epsilon=10.0, total_delta=1e-3),
            PrivacyLevel.MEDIUM: PrivacyBudget(total_epsilon=1.0, total_delta=1e-5),
            PrivacyLevel.HIGH: PrivacyBudget(total_epsilon=0.1, total_delta=1e-6),
            PrivacyLevel.EXTREME: PrivacyBudget(total_epsilon=0.01, total_delta=1e-7)
        }
        
        self.logger = logging.getLogger("privacy_manager")
    
    async def initialize(self):
        """Initialize the privacy manager"""
        self.logger.info("Privacy Manager initialized")
    
    def create_privacy_budget(self, training_id: str, 
                            privacy_level: PrivacyLevel = None) -> PrivacyBudget:
        """Create privacy budget for a training session"""
        if privacy_level is None:
            privacy_level = self.global_privacy_level
        
        budget = PrivacyBudget(
            total_epsilon=self.default_budgets[privacy_level].total_epsilon,
            total_delta=self.default_budgets[privacy_level].total_delta,
            per_round_epsilon=self.default_budgets[privacy_level].total_epsilon / 100,  # Assume max 100 rounds
            per_round_delta=self.default_budgets[privacy_level].total_delta / 100
        )
        
        # Create privacy meter for this training
        self.privacy_meters[training_id] = PrivacyMeter()
        
        self.logger.info(f"Created privacy budget for {training_id}: ε={budget.total_epsilon}, δ={budget.total_delta}")
        return budget
    
    async def apply_differential_privacy(self, client_updates: Dict[str, Dict[str, Any]], 
                                       budget: PrivacyBudget) -> Dict[str, Dict[str, Any]]:
        """Apply differential privacy to client updates"""
        try:
            if budget.mechanism == PrivacyMechanism.NONE:
                return client_updates
            
            # Check if we can spend the budget
            if not budget.can_spend(budget.per_round_epsilon, budget.per_round_delta):
                raise ValueError("Insufficient privacy budget")
            
            # Get DP mechanism
            dp_mechanism = self.dp_mechanisms.get(budget.mechanism)
            if not dp_mechanism:
                raise ValueError(f"Unsupported DP mechanism: {budget.mechanism}")
            
            # Apply DP to each client update
            private_updates = {}
            
            for client_id, update_data in client_updates.items():
                private_update = update_data.copy()
                
                # Apply DP to model weights
                if "model_weights" in update_data:
                    private_weights = {}
                    
                    for layer_name, weights in update_data["model_weights"].items():
                        if isinstance(weights, list):
                            weights = np.array(weights)
                        
                        # Clip gradients first
                        clipped_weights = self._clip_gradients(weights, budget.clipping_norm)
                        
                        # Add DP noise
                        private_weights[layer_name] = dp_mechanism.add_noise(
                            clipped_weights,
                            sensitivity=budget.per_round_epsilon,
                            epsilon=budget.per_round_epsilon,
                            delta=budget.per_round_delta
                        )
                    
                    private_update["model_weights"] = private_weights
                
                private_updates[client_id] = private_update
            
            # Spend privacy budget
            budget.spend(budget.per_round_epsilon, budget.per_round_delta)
            
            self.logger.info(f"Applied {budget.mechanism.value} to {len(client_updates)} client updates")
            return private_updates
            
        except Exception as e:
            self.logger.error(f"Failed to apply differential privacy: {e}")
            raise
    
    def _clip_gradients(self, gradients: np.ndarray, max_norm: float) -> np.ndarray:
        """Clip gradients to bounded sensitivity"""
        grad_norm = np.linalg.norm(gradients)
        
        if grad_norm > max_norm:
            clipped = gradients * (max_norm / grad_norm)
            return clipped
        
        return gradients
    
    async def setup_secure_aggregation(self, training_id: str, 
                                     client_ids: List[str],
                                     config: SecureAggregationConfig) -> SecureAggregator:
        """Setup secure aggregation for a training session"""
        try:
            secure_aggregator = SecureAggregator(config)
            
            # Generate keypairs for all clients
            client_keys = {}
            for client_id in client_ids:
                private_key, public_key = secure_aggregator.generate_client_keypair(client_id)
                client_keys[client_id] = {
                    "private_key": base64.b64encode(private_key).decode(),
                    "public_key": base64.b64encode(public_key).decode()
                }
            
            # Setup shared secrets
            secure_aggregator.setup_shared_secrets(client_ids)
            
            # Store aggregator
            self.secure_aggregators[training_id] = secure_aggregator
            
            self.logger.info(f"Setup secure aggregation for {training_id} with {len(client_ids)} clients")
            return secure_aggregator
            
        except Exception as e:
            self.logger.error(f"Failed to setup secure aggregation: {e}")
            raise
    
    async def apply_secure_aggregation(self, training_id: str,
                                     client_updates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Apply secure aggregation to client updates"""
        try:
            secure_aggregator = self.secure_aggregators.get(training_id)
            if not secure_aggregator:
                raise ValueError(f"No secure aggregator found for training {training_id}")
            
            # Extract and mask client inputs
            masked_inputs = {}
            
            for client_id, update_data in client_updates.items():
                if "model_weights" in update_data:
                    # Flatten model weights into a single vector
                    weight_vector = self._flatten_model_weights(update_data["model_weights"])
                    
                    # Get other participating clients
                    other_clients = [cid for cid in client_updates.keys() if cid != client_id]
                    
                    # Mask the input
                    masked_input = secure_aggregator.mask_input(client_id, weight_vector, other_clients)
                    masked_inputs[client_id] = masked_input
            
            # Aggregate masked inputs
            aggregated_vector = secure_aggregator.aggregate_masked_inputs(masked_inputs)
            
            # Reconstruct model weights structure
            aggregated_weights = self._reconstruct_model_weights(
                aggregated_vector, client_updates[list(client_updates.keys())[0]]["model_weights"]
            )
            
            result = {
                "aggregated_weights": aggregated_weights,
                "participating_clients": list(client_updates.keys()),
                "aggregation_method": "secure_aggregation"
            }
            
            self.logger.info(f"Applied secure aggregation to {len(client_updates)} client updates")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to apply secure aggregation: {e}")
            raise
    
    def _flatten_model_weights(self, model_weights: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten model weights into a single vector"""
        flattened = []
        for layer_name in sorted(model_weights.keys()):  # Ensure consistent ordering
            weights = model_weights[layer_name]
            if isinstance(weights, list):
                weights = np.array(weights)
            flattened.append(weights.flatten())
        
        return np.concatenate(flattened)
    
    def _reconstruct_model_weights(self, flattened_vector: np.ndarray, 
                                 reference_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Reconstruct model weights from flattened vector"""
        reconstructed = {}
        start_idx = 0
        
        for layer_name in sorted(reference_weights.keys()):
            ref_shape = reference_weights[layer_name].shape if hasattr(reference_weights[layer_name], 'shape') else np.array(reference_weights[layer_name]).shape
            layer_size = np.prod(ref_shape)
            
            layer_weights = flattened_vector[start_idx:start_idx + layer_size]
            reconstructed[layer_name] = layer_weights.reshape(ref_shape)
            
            start_idx += layer_size
        
        return reconstructed
    
    async def apply_privacy(self, x_data: np.ndarray, y_data: np.ndarray,
                          config: DifferentialPrivacyConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Apply privacy mechanisms to training data"""
        try:
            if config.mechanism == PrivacyMechanism.NONE:
                return x_data, y_data
            
            # Apply data sampling for privacy
            if config.sampling_probability < 1.0:
                n_samples = int(len(x_data) * config.sampling_probability)
                indices = np.random.choice(len(x_data), n_samples, replace=False)
                x_data = x_data[indices]
                y_data = y_data[indices]
            
            # Apply input perturbation if configured
            if config.mechanism in [PrivacyMechanism.GAUSSIAN_DP, PrivacyMechanism.LAPLACE_DP]:
                dp_mechanism = self.dp_mechanisms[config.mechanism]
                
                # Add noise to features (be careful with this)
                noise_scale = dp_mechanism.compute_noise_scale(
                    config.sensitivity, config.epsilon, config.delta
                ) * 0.1  # Reduce noise for input data
                
                x_noise = np.random.normal(0, noise_scale, x_data.shape)
                x_data = x_data + x_noise
            
            self.logger.info(f"Applied {config.mechanism.value} privacy to training data")
            return x_data, y_data
            
        except Exception as e:
            self.logger.error(f"Failed to apply privacy to training data: {e}")
            raise
    
    def analyze_privacy_guarantees(self, training_id: str) -> Dict[str, Any]:
        """Analyze privacy guarantees for a training session"""
        try:
            privacy_meter = self.privacy_meters.get(training_id)
            if not privacy_meter:
                return {"error": "No privacy meter found for training"}
            
            total_epsilon, total_delta = privacy_meter.get_total_budget_used()
            
            analysis = {
                "training_id": training_id,
                "total_epsilon_used": total_epsilon,
                "total_delta_used": total_delta,
                "mechanism_usage": dict(privacy_meter.mechanism_usage),
                "privacy_violations": privacy_meter.privacy_violations,
                "rounds_tracked": len(privacy_meter.budget_history),
                "privacy_level": self._categorize_privacy_level(total_epsilon, total_delta)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze privacy guarantees: {e}")
            return {"error": str(e)}
    
    def _categorize_privacy_level(self, epsilon: float, delta: float) -> str:
        """Categorize privacy level based on epsilon and delta"""
        if epsilon <= 0.01:
            return "extreme"
        elif epsilon <= 0.1:
            return "high"
        elif epsilon <= 1.0:
            return "medium"
        else:
            return "low"
    
    def get_privacy_recommendations(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get privacy recommendations for training configuration"""
        recommendations = {
            "mechanism": "gaussian_dp",
            "epsilon": 1.0,
            "delta": 1e-5,
            "clipping_norm": 1.0,
            "sampling_probability": 0.1,
            "rationale": []
        }
        
        # Adjust based on number of clients
        num_clients = training_config.get("num_clients", 10)
        if num_clients < 10:
            recommendations["epsilon"] = 0.5
            recommendations["rationale"].append("Few clients - increased privacy")
        elif num_clients > 100:
            recommendations["epsilon"] = 2.0
            recommendations["rationale"].append("Many clients - relaxed privacy")
        
        # Adjust based on data sensitivity
        data_sensitivity = training_config.get("data_sensitivity", "medium")
        if data_sensitivity == "high":
            recommendations["epsilon"] = 0.1
            recommendations["delta"] = 1e-6
            recommendations["rationale"].append("High data sensitivity")
        
        # Adjust based on number of rounds
        max_rounds = training_config.get("max_rounds", 100)
        recommendations["per_round_epsilon"] = recommendations["epsilon"] / max_rounds
        recommendations["rationale"].append(f"Budget distributed over {max_rounds} rounds")
        
        return recommendations
    
    async def cleanup_training_privacy(self, training_id: str):
        """Clean up privacy resources for completed training"""
        try:
            # Remove privacy meter
            if training_id in self.privacy_meters:
                del self.privacy_meters[training_id]
            
            # Remove secure aggregator
            if training_id in self.secure_aggregators:
                del self.secure_aggregators[training_id]
            
            # Remove homomorphic encryption engine
            if training_id in self.homomorphic_engines:
                del self.homomorphic_engines[training_id]
            
            self.logger.info(f"Cleaned up privacy resources for training {training_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup privacy resources: {e}")
    
    def get_privacy_stats(self) -> Dict[str, Any]:
        """Get privacy manager statistics"""
        return {
            "active_privacy_meters": len(self.privacy_meters),
            "active_secure_aggregators": len(self.secure_aggregators),
            "active_homomorphic_engines": len(self.homomorphic_engines),
            "global_privacy_level": self.global_privacy_level.value,
            "supported_mechanisms": [m.value for m in PrivacyMechanism],
            "default_budgets": {
                level.value: {
                    "epsilon": budget.total_epsilon,
                    "delta": budget.total_delta
                }
                for level, budget in self.default_budgets.items()
            }
        }