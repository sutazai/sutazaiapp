#!/usr/bin/env python3
"""
Cross-Modal Learning System for SutazAI Multi-Modal Fusion

This module implements advanced cross-modal learning capabilities including:
- Contrastive learning across modalities
- Cross-modal transfer learning
- Dynamic attention mechanisms
- Modality-specific adaptation
- Knowledge distillation between modalities

Integrates with SutazAI's agent ecosystem and knowledge graph for enhanced
understanding and learning from multi-modal interactions.

Author: SutazAI Multi-Modal Fusion System
Version: 1.0.0
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import pickle
from pathlib import Path
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid

from .multi_modal_fusion_coordinator import ModalityType, ModalityData, FusionResult
from .unified_representation import UnifiedRepresentation, RepresentationLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningStrategy(Enum):
    """Cross-modal learning strategies"""
    CONTRASTIVE = "contrastive"
    ALIGNMENT = "alignment"  
    TRANSFER = "transfer"
    DISTILLATION = "distillation"
    ADVERSARIAL = "adversarial"

@dataclass
class CrossModalSample:
    """Sample for cross-modal learning"""
    sample_id: str
    modality_data: Dict[ModalityType, ModalityData]
    unified_representation: Optional[UnifiedRepresentation] = None
    labels: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class LearningMetrics:
    """Metrics for cross-modal learning"""
    alignment_loss: float = 0.0
    contrastive_loss: float = 0.0
    reconstruction_loss: float = 0.0
    transfer_accuracy: float = 0.0
    modality_confusion: Dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.001
    epoch: int = 0
    batch_size: int = 32

class ContrastiveLearningModule(nn.Module):
    """Contrastive learning for cross-modal alignment"""
    
    def __init__(self, feature_dim: int = 768, temperature: float = 0.07):
        super().__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature
        
        # Projection heads for each modality
        self.projection_heads = nn.ModuleDict()
        
    def add_modality(self, modality: ModalityType, input_dim: int):
        """Add projection head for a modality"""
        self.projection_heads[modality.value] = nn.Sequential(
            nn.Linear(input_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim)
        )
        
    def forward(self, modality_features: Dict[ModalityType, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for contrastive learning
        
        Args:
            modality_features: Dictionary of modality feature tensors
            
        Returns:
            Tuple of (contrastive_loss, projected_features)
        """
        projected_features = {}
        
        # Project features for each modality
        for modality, features in modality_features.items():
            if modality.value in self.projection_heads:
                projected = self.projection_heads[modality.value](features)
                projected = F.normalize(projected, dim=1)
                projected_features[modality.value] = projected
        
        # Calculate contrastive loss between modality pairs
        total_loss = 0.0
        num_pairs = 0
        
        modality_list = list(projected_features.keys())
        for i in range(len(modality_list)):
            for j in range(i + 1, len(modality_list)):
                mod1, mod2 = modality_list[i], modality_list[j]
                loss = self._contrastive_loss(
                    projected_features[mod1], 
                    projected_features[mod2]
                )
                total_loss += loss
                num_pairs += 1
        
        avg_loss = total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)
        
        return avg_loss, projected_features
    
    def _contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Calculate InfoNCE contrastive loss between two modalities"""
        batch_size = z1.size(0)
        
        # Cosine similarity
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature
        
        # Labels for positive pairs (diagonal elements)
        labels = torch.arange(batch_size, device=z1.device)
        
        # InfoNCE loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss

class CrossModalAttentionLearner(nn.Module):
    """Dynamic cross-modal attention learning mechanism"""
    
    def __init__(self, feature_dim: int = 768, num_heads: int = 8, 
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Multi-layer cross-modal attention
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Layer normalization for each layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(feature_dim) for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.feed_forwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim * 4, feature_dim)
            ) for _ in range(num_layers)
        ])
        
        # Modality type embeddings
        self.modality_embeddings = nn.Embedding(len(ModalityType), feature_dim)
        
    def forward(self, modality_features: Dict[ModalityType, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through cross-modal attention layers
        
        Args:
            modality_features: Dictionary of modality features
            
        Returns:
            Tuple of (fused_representation, attention_weights)
        """
        # Prepare input sequences
        sequences = []
        modality_types = []
        
        for modality_type, features in modality_features.items():
            # Add modality type embedding
            mod_embedding = self.modality_embeddings(torch.tensor(list(ModalityType).index(modality_type)))
            enhanced_features = features + mod_embedding.unsqueeze(0).expand_as(features)
            sequences.append(enhanced_features)
            modality_types.append(modality_type)
        
        # Concatenate all modality sequences
        if len(sequences) == 1:
            x = sequences[0].unsqueeze(1)  # Add sequence dimension
        else:
            x = torch.stack(sequences, dim=1)  # [batch, num_modalities, feature_dim]
        
        attention_weights = {}
        
        # Apply multi-layer cross-modal attention
        for layer_idx in range(self.num_layers):
            # Self-attention across modalities
            attended, weights = self.attention_layers[layer_idx](x, x, x)
            
            # Residual connection and layer norm
            x = self.layer_norms[layer_idx](x + attended)
            
            # Feed-forward network
            ff_output = self.feed_forwards[layer_idx](x)
            x = x + ff_output
            
            # Store attention weights for analysis
            attention_weights[f'layer_{layer_idx}'] = weights
        
        # Aggregate across modalities (weighted sum based on attention)
        if x.size(1) > 1:
            # Use average of last layer attention weights for aggregation
            last_weights = attention_weights[f'layer_{self.num_layers-1}']
            avg_weights = torch.mean(last_weights, dim=1)  # Average across heads
            
            # Weighted sum of modalities
            modality_weights = F.softmax(torch.sum(avg_weights, dim=-1), dim=-1)
            fused = torch.sum(x * modality_weights.unsqueeze(-1), dim=1)
        else:
            fused = x.squeeze(1)
        
        return fused, attention_weights

class ModalityTransferLearner:
    """Transfer learning between modalities"""
    
    def __init__(self, source_modality: ModalityType, target_modality: ModalityType,
                 feature_dim: int = 768):
        self.source_modality = source_modality
        self.target_modality = target_modality
        self.feature_dim = feature_dim
        
        # Transfer mapping network
        self.transfer_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Tanh()
        )
        
        # Discriminator for adversarial training
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.transfer_optimizer = optim.Adam(self.transfer_network.parameters(), lr=0.001)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0005)
        
    def train_transfer(self, source_features: torch.Tensor, 
                      target_features: torch.Tensor, 
                      num_epochs: int = 100) -> Dict[str, float]:
        """Train transfer mapping between modalities"""
        
        transfer_losses = []
        discriminator_losses = []
        
        for epoch in range(num_epochs):
            # Train discriminator
            self.discriminator_optimizer.zero_grad()
            
            # Real target features
            real_pred = self.discriminator(target_features)
            real_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred))
            
            # Fake target features (transferred from source)
            with torch.no_grad():
                fake_target = self.transfer_network(source_features)
            fake_pred = self.discriminator(fake_target.detach())
            fake_loss = F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))
            
            discriminator_loss = (real_loss + fake_loss) / 2
            discriminator_loss.backward()
            self.discriminator_optimizer.step()
            
            # Train transfer network
            self.transfer_optimizer.zero_grad()
            
            # Transfer source to target domain
            transferred = self.transfer_network(source_features)
            
            # Adversarial loss (fool discriminator)
            adversarial_pred = self.discriminator(transferred)
            adversarial_loss = F.binary_cross_entropy(adversarial_pred, torch.ones_like(adversarial_pred))
            
            # Reconstruction loss (if paired data available)
            reconstruction_loss = F.mse_loss(transferred, target_features)
            
            # Combined transfer loss
            transfer_loss = adversarial_loss + 10.0 * reconstruction_loss
            transfer_loss.backward()
            self.transfer_optimizer.step()
            
            transfer_losses.append(transfer_loss.item())
            discriminator_losses.append(discriminator_loss.item())
            
            if epoch % 10 == 0:
                logger.info(f"Transfer Learning Epoch {epoch}: "
                           f"Transfer Loss: {transfer_loss.item():.4f}, "
                           f"Discriminator Loss: {discriminator_loss.item():.4f}")
        
        return {
            'final_transfer_loss': transfer_losses[-1],
            'final_discriminator_loss': discriminator_losses[-1],
            'transfer_loss_history': transfer_losses,
            'discriminator_loss_history': discriminator_losses
        }
    
    def transfer_features(self, source_features: torch.Tensor) -> torch.Tensor:
        """Transfer features from source to target modality"""
        with torch.no_grad():
            transferred = self.transfer_network(source_features)
        return transferred

class CrossModalDataset(Dataset):
    """Dataset for cross-modal learning"""
    
    def __init__(self, samples: List[CrossModalSample]):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert modality data to tensors
        modality_tensors = {}
        for modality, data in sample.modality_data.items():
            if data.embedding is not None:
                modality_tensors[modality] = torch.tensor(data.embedding, dtype=torch.float32)
            else:
                # Generate placeholder embeddings
                modality_tensors[modality] = torch.randn(768)
        
        return {
            'sample_id': sample.sample_id,
            'modality_tensors': modality_tensors,
            'labels': sample.labels,
            'metadata': sample.metadata
        }

class CrossModalLearningSystem:
    """
    Main system for cross-modal learning in SutazAI platform
    
    Integrates with:
    - Agent orchestration system for distributed learning
    - Knowledge graph for semantic understanding
    - Ollama models for enhanced text understanding
    - Vector databases for efficient storage/retrieval
    """
    
    def __init__(self, config_path: str = "/opt/sutazaiapp/config/fusion_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Core learning components
        self.contrastive_learner = None
        self.attention_learner = None
        self.transfer_learners = {}
        
        # Training data and metrics
        self.training_data = []
        self.validation_data = []
        self.learning_metrics = LearningMetrics()
        
        # Learning state
        self.learning_history = defaultdict(list)
        self.model_checkpoints = {}
        self.active_learning_tasks = {}
        
        # Processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 8))
        self.learning_lock = threading.Lock()
        
        # Integration clients
        self.agent_orchestrator = None
        self.knowledge_graph = None
        
        logger.info("Cross-Modal Learning System initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load learning configuration"""
        default_config = {
            'feature_dim': 768,
            'num_attention_heads': 8,
            'num_attention_layers': 3,
            'contrastive_temperature': 0.07,
            'learning_rate': 0.001,
            'batch_size': 32,
            'max_epochs': 100,
            'validation_split': 0.2,
            'save_checkpoints': True,
            'checkpoint_interval': 10,
            'max_workers': 8,
            'enable_transfer_learning': True,
            'enable_contrastive_learning': True,
            'enable_attention_learning': True
        }
        
        if self.config_path.exists():
            try:
                import yaml
                with open(self.config_path) as f:
                    loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Failed to load learning config: {e}")
        
        return default_config
    
    def initialize_learners(self, modality_dims: Dict[ModalityType, int]):
        """Initialize learning components with modality dimensions"""
        feature_dim = self.config['feature_dim']
        
        # Initialize contrastive learner
        if self.config.get('enable_contrastive_learning', True):
            self.contrastive_learner = ContrastiveLearningModule(
                feature_dim=feature_dim,
                temperature=self.config['contrastive_temperature']
            )
            
            # Add modalities to contrastive learner
            for modality, dim in modality_dims.items():
                self.contrastive_learner.add_modality(modality, dim)
        
        # Initialize attention learner
        if self.config.get('enable_attention_learning', True):
            self.attention_learner = CrossModalAttentionLearner(
                feature_dim=feature_dim,
                num_heads=self.config['num_attention_heads'],
                num_layers=self.config['num_attention_layers']
            )
        
        # Initialize transfer learners for all modality pairs
        if self.config.get('enable_transfer_learning', True):
            modalities = list(modality_dims.keys())
            for i, source in enumerate(modalities):
                for j, target in enumerate(modalities):
                    if i != j:
                        key = f"{source.value}_to_{target.value}"
                        self.transfer_learners[key] = ModalityTransferLearner(
                            source, target, feature_dim
                        )
        
        logger.info(f"Initialized learners for modalities: {list(modality_dims.keys())}")
    
    def add_training_sample(self, sample: CrossModalSample):
        """Add sample to training data"""
        with self.learning_lock:
            self.training_data.append(sample)
            
            # Maintain maximum training data size
            max_samples = self.config.get('max_training_samples', 10000)
            if len(self.training_data) > max_samples:
                self.training_data = self.training_data[-max_samples:]
        
        logger.debug(f"Added training sample {sample.sample_id}, "
                    f"total samples: {len(self.training_data)}")
    
    def add_training_samples(self, samples: List[CrossModalSample]):
        """Add multiple samples to training data"""
        for sample in samples:
            self.add_training_sample(sample)
    
    async def train_cross_modal_alignment(self, num_epochs: int = None) -> Dict[str, Any]:
        """Train cross-modal alignment using available learning strategies"""
        if num_epochs is None:
            num_epochs = self.config['max_epochs']
        
        if not self.training_data:
            logger.warning("No training data available for cross-modal learning")
            return {}
        
        logger.info(f"Starting cross-modal alignment training with {len(self.training_data)} samples")
        
        # Split data into training and validation
        split_idx = int(len(self.training_data) * (1 - self.config['validation_split']))
        train_samples = self.training_data[:split_idx]
        val_samples = self.training_data[split_idx:]
        
        # Create data loaders
        train_dataset = CrossModalDataset(train_samples)
        val_dataset = CrossModalDataset(val_samples) if val_samples else None
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False
            )
        
        # Training metrics
        training_results = {
            'contrastive_losses': [],
            'attention_losses': [],
            'validation_metrics': [],
            'epoch_times': []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train contrastive learning
            if self.contrastive_learner:
                contrastive_loss = await self._train_contrastive_epoch(train_loader)
                training_results['contrastive_losses'].append(contrastive_loss)
            
            # Train attention learning
            if self.attention_learner:
                attention_loss = await self._train_attention_epoch(train_loader)
                training_results['attention_losses'].append(attention_loss)
            
            # Validation
            if val_dataset and epoch % 5 == 0:
                val_metrics = await self._validate_models(val_loader)
                training_results['validation_metrics'].append(val_metrics)
            
            epoch_time = time.time() - epoch_start
            training_results['epoch_times'].append(epoch_time)
            
            # Logging
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{num_epochs} completed in {epoch_time:.2f}s")
                if self.contrastive_learner:
                    logger.info(f"  Contrastive Loss: {training_results['contrastive_losses'][-1]:.4f}")
                if self.attention_learner:
                    logger.info(f"  Attention Loss: {training_results['attention_losses'][-1]:.4f}")
            
            # Save checkpoints
            if self.config.get('save_checkpoints', True) and epoch % self.config.get('checkpoint_interval', 10) == 0:
                await self._save_checkpoint(epoch)
        
        # Update learning metrics
        self.learning_metrics.epoch = num_epochs
        if training_results['contrastive_losses']:
            self.learning_metrics.contrastive_loss = training_results['contrastive_losses'][-1]
        if training_results['attention_losses']:
            self.learning_metrics.alignment_loss = training_results['attention_losses'][-1]
        
        logger.info(f"Cross-modal alignment training completed after {num_epochs} epochs")
        
        return training_results
    
    async def _train_contrastive_epoch(self, train_loader: DataLoader) -> float:
        """Train contrastive learning for one epoch"""
        if not self.contrastive_learner:
            return 0.0
        
        total_loss = 0.0
        num_batches = 0
        
        optimizer = optim.Adam(self.contrastive_learner.parameters(), lr=self.config['learning_rate'])
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Extract modality features from batch
            modality_features = {}
            for modality_type in ModalityType:
                if modality_type in batch['modality_tensors'][0]:
                    # Stack features across batch
                    features = torch.stack([
                        sample_data[modality_type] 
                        for sample_data in batch['modality_tensors']
                    ])
                    modality_features[modality_type] = features
            
            if len(modality_features) < 2:
                continue  # Need at least 2 modalities for contrastive learning
            
            # Forward pass
            loss, projected_features = self.contrastive_learner(modality_features)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    async def _train_attention_epoch(self, train_loader: DataLoader) -> float:
        """Train attention learning for one epoch"""
        if not self.attention_learner:
            return 0.0
        
        total_loss = 0.0
        num_batches = 0
        
        optimizer = optim.Adam(self.attention_learner.parameters(), lr=self.config['learning_rate'])
        criterion = nn.MSELoss()
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Extract modality features from batch
            modality_features = {}
            for modality_type in ModalityType:
                if modality_type in batch['modality_tensors'][0]:
                    features = torch.stack([
                        sample_data[modality_type] 
                        for sample_data in batch['modality_tensors']
                    ])
                    modality_features[modality_type] = features
            
            if len(modality_features) < 1:
                continue
            
            # Forward pass
            fused_representation, attention_weights = self.attention_learner(modality_features)
            
            # Self-supervised loss: reconstruct original features
            reconstruction_loss = 0.0
            for modality, original_features in modality_features.items():
                # Simple reconstruction target (would be more sophisticated in practice)
                target = torch.mean(original_features, dim=1)  # Average pooling
                loss = criterion(fused_representation, target)
                reconstruction_loss += loss
            
            reconstruction_loss /= len(modality_features)
            
            # Backward pass
            reconstruction_loss.backward()
            optimizer.step()
            
            total_loss += reconstruction_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    async def _validate_models(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate trained models"""
        validation_metrics = {}
        
        # Contrastive validation
        if self.contrastive_learner:
            self.contrastive_learner.eval()
            contrastive_val_loss = 0.0
            num_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    modality_features = {}
                    for modality_type in ModalityType:
                        if modality_type in batch['modality_tensors'][0]:
                            features = torch.stack([
                                sample_data[modality_type] 
                                for sample_data in batch['modality_tensors']
                            ])
                            modality_features[modality_type] = features
                    
                    if len(modality_features) >= 2:
                        loss, _ = self.contrastive_learner(modality_features)
                        contrastive_val_loss += loss.item()
                        num_batches += 1
            
            validation_metrics['contrastive_val_loss'] = (
                contrastive_val_loss / num_batches if num_batches > 0 else 0.0
            )
            self.contrastive_learner.train()
        
        # Attention validation
        if self.attention_learner:
            self.attention_learner.eval()
            attention_val_loss = 0.0
            num_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    modality_features = {}
                    for modality_type in ModalityType:
                        if modality_type in batch['modality_tensors'][0]:
                            features = torch.stack([
                                sample_data[modality_type] 
                                for sample_data in batch['modality_tensors']
                            ])
                            modality_features[modality_type] = features
                    
                    if len(modality_features) >= 1:
                        fused_repr, _ = self.attention_learner(modality_features)
                        
                        # Reconstruction validation loss
                        reconstruction_loss = 0.0
                        for modality, original_features in modality_features.items():
                            target = torch.mean(original_features, dim=1)
                            loss = F.mse_loss(fused_repr, target)
                            reconstruction_loss += loss.item()
                        
                        attention_val_loss += reconstruction_loss / len(modality_features)
                        num_batches += 1
            
            validation_metrics['attention_val_loss'] = (
                attention_val_loss / num_batches if num_batches > 0 else 0.0
            )
            self.attention_learner.train()
        
        return validation_metrics
    
    async def _save_checkpoint(self, epoch: int):
        """Save model checkpoints"""
        checkpoint_dir = Path("/opt/sutazaiapp/fusion/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            'epoch': epoch,
            'learning_metrics': self.learning_metrics.__dict__,
            'config': self.config
        }
        
        if self.contrastive_learner:
            checkpoint_data['contrastive_learner_state'] = self.contrastive_learner.state_dict()
        
        if self.attention_learner:
            checkpoint_data['attention_learner_state'] = self.attention_learner.state_dict()
        
        checkpoint_path = checkpoint_dir / f"fusion_checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint_data, checkpoint_path)
        
        logger.info(f"Saved checkpoint at epoch {epoch} to {checkpoint_path}")
    
    def get_cross_modal_similarity(self, 
                                  representation1: UnifiedRepresentation,
                                  representation2: UnifiedRepresentation) -> Dict[str, float]:
        """Calculate cross-modal similarity between representations"""
        similarities = {}
        
        # Overall unified representation similarity
        unified_sim = np.dot(representation1.unified_embedding, representation2.unified_embedding) / (
            np.linalg.norm(representation1.unified_embedding) * 
            np.linalg.norm(representation2.unified_embedding)
        )
        similarities['unified'] = float(unified_sim)
        
        # Modality-specific similarities
        for modality in representation1.modality_embeddings:
            if modality in representation2.modality_embeddings:
                emb1 = representation1.modality_embeddings[modality]
                emb2 = representation2.modality_embeddings[modality]
                
                sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                similarities[f'{modality.value}_similarity'] = float(sim)
        
        # Semantic feature similarities
        common_features = set(representation1.semantic_features.keys()) & set(representation2.semantic_features.keys())
        if common_features:
            semantic_similarities = []
            for feature in common_features:
                val1 = representation1.semantic_features[feature]
                val2 = representation2.semantic_features[feature]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Normalized difference for numeric features
                    sim = 1.0 - abs(val1 - val2) / 2.0  # Assuming values in [-1, 1]
                    semantic_similarities.append(sim)
            
            if semantic_similarities:
                similarities['semantic'] = float(np.mean(semantic_similarities))
        
        return similarities
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning system statistics"""
        stats = {
            'training_samples': len(self.training_data),
            'validation_samples': len(self.validation_data),
            'learning_metrics': self.learning_metrics.__dict__,
            'active_learning_tasks': len(self.active_learning_tasks),
            'transfer_learners': len(self.transfer_learners),
            'models_initialized': {
                'contrastive_learner': self.contrastive_learner is not None,
                'attention_learner': self.attention_learner is not None
            }
        }
        
        # Add model parameter counts
        if self.contrastive_learner:
            stats['contrastive_parameters'] = sum(
                p.numel() for p in self.contrastive_learner.parameters()
            )
        
        if self.attention_learner:
            stats['attention_parameters'] = sum(
                p.numel() for p in self.attention_learner.parameters()
            )
        
        return stats
    
    async def shutdown(self):
        """Shutdown learning system"""
        logger.info("Shutting down Cross-Modal Learning System")
        
        # Save final checkpoint
        if self.config.get('save_checkpoints', True):
            await self._save_checkpoint(self.learning_metrics.epoch)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Cross-Modal Learning System shutdown complete")

# Example usage
async def main():
    """Example usage of Cross-Modal Learning System"""
    
    # Initialize learning system
    learning_system = CrossModalLearningSystem()
    
    # Initialize with sample modality dimensions
    modality_dims = {
        ModalityType.TEXT: 768,
        ModalityType.VOICE: 768,
        ModalityType.VISUAL: 768
    }
    
    learning_system.initialize_learners(modality_dims)
    
    # Create sample training data
    samples = []
    for i in range(100):
        text_data = ModalityData(
            modality_type=ModalityType.TEXT,
            data=f"Sample text {i}",
            timestamp=time.time(),
            confidence=0.9,
            embedding=np.random.rand(768)
        )
        
        voice_data = ModalityData(
            modality_type=ModalityType.VOICE,
            data=f"[audio_{i}]",
            timestamp=time.time(),
            confidence=0.8,
            embedding=np.random.rand(768)
        )
        
        sample = CrossModalSample(
            sample_id=f"sample_{i}",
            modality_data={
                ModalityType.TEXT: text_data,
                ModalityType.VOICE: voice_data
            },
            labels={'category': i % 5}
        )
        
        samples.append(sample)
    
    # Add training samples
    learning_system.add_training_samples(samples)
    
    logger.info(f"Added {len(samples)} training samples")
    
    # Train cross-modal alignment
    logger.info("Starting cross-modal training...")
    training_results = await learning_system.train_cross_modal_alignment(num_epochs=20)
    
    logger.info("Training Results:")
    logger.info(f"  Contrastive Losses: {training_results.get('contrastive_losses', [])[-5:]}")
    logger.info(f"  Attention Losses: {training_results.get('attention_losses', [])[-5:]}")
    
    # Get learning statistics
    stats = learning_system.get_learning_statistics()
    logger.info(f"Learning Statistics: {json.dumps(stats, indent=2)}")
    
    # Shutdown
    await learning_system.shutdown()

if __name__ == "__main__":
    asyncio.run(main())