#!/usr/bin/env python3
"""
Multi-Modal Fusion Coordinator for SutazAI Platform

This system enables coordinated processing and understanding of multiple data modalities
simultaneously, including text, voice, visual data, and sensor inputs.

Architecture:
- Early Fusion: Low-level feature integration
- Late Fusion: High-level decision combination  
- Hybrid Fusion: Multi-stage integration strategies
- Cross-Modal Attention: Dynamic weighting mechanisms
- Temporal Synchronization: Alignment across time-varying modalities

Author: SutazAI Multi-Modal Fusion System
Version: 1.0.0
"""

import asyncio
import logging
import json
import numpy as np
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModalityType(Enum):
    """Supported modality types"""
    TEXT = "text"
    VOICE = "voice" 
    VISUAL = "visual"
    SENSOR = "sensor"
    STRUCTURED = "structured"
    TEMPORAL = "temporal"

class FusionStrategy(Enum):
    """Fusion strategy types"""
    EARLY = "early"
    LATE = "late"
    HYBRID = "hybrid"
    ATTENTION = "attention"

@dataclass
class ModalityData:
    """Container for modality-specific data"""
    modality_type: ModalityType
    data: Any
    timestamp: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    preprocessing_applied: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()

@dataclass
class FusionResult:
    """Result of fusion operation"""
    fused_representation: np.ndarray
    fusion_strategy: FusionStrategy
    contributing_modalities: List[ModalityType]
    confidence_scores: Dict[ModalityType, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_latency: float = 0.0
    fusion_weights: Optional[Dict[ModalityType, float]] = None

class TemporalSynchronizer:
    """Handles temporal alignment of multi-modal data"""
    
    def __init__(self, window_size: float = 1.0, tolerance: float = 0.1):
        self.window_size = window_size  # seconds
        self.tolerance = tolerance      # alignment tolerance
        self.data_buffers = defaultdict(deque)
        self.sync_lock = threading.Lock()
    
    def add_data(self, data: ModalityData) -> None:
        """Add data to temporal buffer"""
        with self.sync_lock:
            self.data_buffers[data.modality_type].append(data)
            self._clean_old_data()
    
    def _clean_old_data(self) -> None:
        """Remove data outside temporal window"""
        current_time = time.time()
        for modality, buffer in self.data_buffers.items():
            while buffer and (current_time - buffer[0].timestamp) > self.window_size:
                buffer.popleft()
    
    def get_synchronized_batch(self, target_timestamp: float) -> Dict[ModalityType, ModalityData]:
        """Get temporally synchronized data batch"""
        with self.sync_lock:
            synchronized = {}
            
            for modality, buffer in self.data_buffers.items():
                # Find closest data within tolerance
                closest_data = None
                min_diff = float('inf')
                
                for data in buffer:
                    time_diff = abs(data.timestamp - target_timestamp)
                    if time_diff < min_diff and time_diff <= self.tolerance:
                        min_diff = time_diff
                        closest_data = data
                
                if closest_data:
                    synchronized[modality] = closest_data
            
            return synchronized

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for dynamic fusion weighting"""
    
    def __init__(self, feature_dims: Dict[ModalityType, int], hidden_dim: int = 256):
        super().__init__()
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        
        # Projection layers for each modality
        self.projections = nn.ModuleDict({
            modality.value: nn.Linear(dim, hidden_dim)
            for modality, dim in feature_dims.items()
        })
        
        # Attention computation
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, modality_features: Dict[ModalityType, torch.Tensor]) -> Tuple[torch.Tensor, Dict[ModalityType, float]]:
        """
        Compute cross-modal attention weights and fused representation
        
        Args:
            modality_features: Dict mapping modalities to feature tensors
            
        Returns:
            Tuple of (fused_features, attention_weights)
        """
        # Project all modalities to common space
        projected = {}
        for modality, features in modality_features.items():
            projected[modality] = self.projections[modality.value](features)
        
        # Stack for attention computation
        modality_stack = torch.stack(list(projected.values()), dim=1)  # [batch, num_modalities, hidden_dim]
        
        # Compute self-attention across modalities
        attended, attention_weights = self.attention(modality_stack, modality_stack, modality_stack)
        
        # Apply layer norm and residual connection
        attended = self.layer_norm(attended + modality_stack)
        
        # Aggregate across modalities (mean pooling)
        fused = torch.mean(attended, dim=1)
        
        # Extract attention weights for interpretation
        modality_weights = {}
        for i, modality in enumerate(projected.keys()):
            modality_weights[modality] = attention_weights[0, 0, i].item()  # Use first head, first query
        
        return fused, modality_weights

class EarlyFusionProcessor:
    """Early fusion strategy - combines raw features before processing"""
    
    def __init__(self, target_dim: int = 512):
        self.target_dim = target_dim
        
    def fuse(self, modality_data: Dict[ModalityType, ModalityData]) -> np.ndarray:
        """
        Perform early fusion by concatenating and projecting features
        
        Args:
            modality_data: Dict of modality data to fuse
            
        Returns:
            Fused feature representation
        """
        # Extract features from each modality
        features = []
        for modality, data in modality_data.items():
            if data.embedding is not None:
                features.append(data.embedding.flatten())
            else:
                # Convert raw data to features based on modality type
                if modality == ModalityType.TEXT:
                    # Simple text features (would use proper embeddings in production)
                    text_features = self._text_to_features(data.data)
                    features.append(text_features)
                elif modality == ModalityType.VOICE:
                    # Voice spectral features
                    voice_features = self._voice_to_features(data.data)
                    features.append(voice_features)
                elif modality == ModalityType.VISUAL:
                    # Visual features
                    visual_features = self._visual_to_features(data.data)
                    features.append(visual_features)
                elif modality == ModalityType.SENSOR:
                    # Sensor data features
                    sensor_features = self._sensor_to_features(data.data)
                    features.append(sensor_features)
        
        if not features:
            return np.zeros(self.target_dim)
        
        # Concatenate all features
        concatenated = np.concatenate(features)
        
        # Project to target dimension
        if len(concatenated) > self.target_dim:
            # Simple dimensionality reduction (would use PCA/learned projection in production)
            step = len(concatenated) // self.target_dim
            projected = concatenated[::step][:self.target_dim]
        else:
            # Pad if needed
            projected = np.pad(concatenated, (0, max(0, self.target_dim - len(concatenated))))
        
        return projected
    
    def _text_to_features(self, text: str) -> np.ndarray:
        """Convert text to feature vector"""
        # Simple bag-of-words representation (placeholder)
        words = text.lower().split()
        features = np.zeros(100)  # 100-dim feature vector
        for i, word in enumerate(words[:100]):
            features[i % 100] += hash(word) % 1000 / 1000.0
        return features
    
    def _voice_to_features(self, audio_data: Any) -> np.ndarray:
        """Convert voice data to feature vector"""
        # Placeholder for MFCC or spectral features
        return np.random.rand(128)  # 128-dim audio features
    
    def _visual_to_features(self, image_data: Any) -> np.ndarray:
        """Convert visual data to feature vector"""
        # Placeholder for CNN features
        return np.random.rand(256)  # 256-dim visual features
    
    def _sensor_to_features(self, sensor_data: Any) -> np.ndarray:
        """Convert sensor data to feature vector"""
        # Statistical features from sensor time series
        if isinstance(sensor_data, (list, np.ndarray)):
            data_array = np.array(sensor_data)
            features = np.array([
                np.mean(data_array),
                np.std(data_array),
                np.min(data_array),
                np.max(data_array),
                np.median(data_array)
            ])
            return np.pad(features, (0, 45))  # Pad to 50 dims
        return np.zeros(50)

class LateFusionProcessor:
    """Late fusion strategy - combines high-level decisions"""
    
    def __init__(self, fusion_method: str = "weighted_average"):
        self.fusion_method = fusion_method
        
    def fuse(self, modality_results: Dict[ModalityType, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform late fusion by combining modality-specific results
        
        Args:
            modality_results: Dict of results from each modality processor
            
        Returns:
            Fused decision/result
        """
        if self.fusion_method == "weighted_average":
            return self._weighted_average_fusion(modality_results)
        elif self.fusion_method == "voting":
            return self._voting_fusion(modality_results)
        elif self.fusion_method == "max_confidence":
            return self._max_confidence_fusion(modality_results)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def _weighted_average_fusion(self, modality_results: Dict[ModalityType, Dict[str, Any]]) -> Dict[str, Any]:
        """Weighted average fusion based on confidence scores"""
        if not modality_results:
            return {}
        
        # Extract numeric results and weights
        weighted_sum = 0.0
        total_weight = 0.0
        
        combined_metadata = {}
        
        for modality, result in modality_results.items():
            confidence = result.get('confidence', 1.0)
            if 'score' in result:
                weighted_sum += result['score'] * confidence
                total_weight += confidence
            
            # Combine metadata
            if 'metadata' in result:
                combined_metadata.update(result['metadata'])
        
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return {
            'score': final_score,
            'fusion_method': 'weighted_average',
            'contributing_modalities': list(modality_results.keys()),
            'metadata': combined_metadata
        }
    
    def _voting_fusion(self, modality_results: Dict[ModalityType, Dict[str, Any]]) -> Dict[str, Any]:
        """Majority voting fusion"""
        decisions = []
        for modality, result in modality_results.items():
            if 'decision' in result:
                decisions.append(result['decision'])
        
        if not decisions:
            return {}
        
        # Simple majority vote
        from collections import Counter
        vote_counts = Counter(decisions)
        winning_decision = vote_counts.most_common(1)[0][0]
        
        return {
            'decision': winning_decision,
            'fusion_method': 'voting',
            'vote_counts': dict(vote_counts),
            'contributing_modalities': list(modality_results.keys())
        }
    
    def _max_confidence_fusion(self, modality_results: Dict[ModalityType, Dict[str, Any]]) -> Dict[str, Any]:
        """Take result from most confident modality"""
        if not modality_results:
            return {}
        
        max_confidence = -1
        best_result = None
        best_modality = None
        
        for modality, result in modality_results.items():
            confidence = result.get('confidence', 0.0)
            if confidence > max_confidence:
                max_confidence = confidence
                best_result = result
                best_modality = modality
        
        if best_result:
            best_result['fusion_method'] = 'max_confidence'
            best_result['selected_modality'] = best_modality
        
        return best_result or {}

class MultiModalFusionCoordinator:
    """
    Main coordinator for multi-modal fusion processing
    
    Integrates with SutazAI's existing infrastructure:
    - Ollama/tinyllama for text processing
    - Jarvis for voice interface
    - Vector databases for embeddings
    - Agent orchestration system
    """
    
    def __init__(self, config_path: str = "/opt/sutazaiapp/config/fusion_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Core components
        self.temporal_sync = TemporalSynchronizer(
            window_size=self.config.get('temporal_window', 1.0),
            tolerance=self.config.get('sync_tolerance', 0.1)
        )
        
        # Fusion processors
        self.early_fusion = EarlyFusionProcessor(self.config.get('feature_dim', 512))
        self.late_fusion = LateFusionProcessor(self.config.get('late_fusion_method', 'weighted_average'))
        
        # Cross-modal attention (initialized when needed)
        self.cross_attention = None
        
        # Processing queues
        self.input_queue = queue.Queue(maxsize=1000)
        self.processing_queue = queue.Queue(maxsize=500)
        self.output_queue = queue.Queue(maxsize=500)
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 8))
        self.process_executor = ProcessPoolExecutor(max_workers=self.config.get('max_processes', 4))
        
        # State tracking
        self.active_sessions = {}
        self.processing_stats = defaultdict(int)
        self.fusion_cache = {}
        
        # Integration endpoints
        self.ollama_url = self.config.get('ollama_url', 'http://ollama:10104')
        self.jarvis_url = self.config.get('jarvis_url', 'http://jarvis:8080')
        self.chromadb_url = self.config.get('chromadb_url', 'http://chromadb:8000')
        
        logger.info("Multi-Modal Fusion Coordinator initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load fusion configuration"""
        default_config = {
            'temporal_window': 2.0,
            'sync_tolerance': 0.2,
            'feature_dim': 768,
            'late_fusion_method': 'weighted_average',
            'max_workers': 12,
            'max_processes': 6,
            'cache_size': 1000,
            'enable_attention': True,
            'ollama_url': 'http://ollama:10104',
            'jarvis_url': 'http://jarvis:8080',
            'chromadb_url': 'http://chromadb:8000'
        }
        
        if self.config_path.exists():
            try:
                import yaml
                with open(self.config_path) as f:
                    loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")
        
        return default_config
    
    async def process_multi_modal_input(self, 
                                      modality_data: Dict[ModalityType, ModalityData],
                                      fusion_strategy: FusionStrategy = FusionStrategy.HYBRID,
                                      session_id: Optional[str] = None) -> FusionResult:
        """
        Process multi-modal input through fusion pipeline
        
        Args:
            modality_data: Dictionary of modality data to fuse
            fusion_strategy: Strategy to use for fusion
            session_id: Optional session identifier for tracking
            
        Returns:
            FusionResult containing the fused representation
        """
        start_time = time.time()
        
        if not session_id:
            session_id = str(uuid.uuid4())
        
        logger.info(f"Processing multi-modal input with {len(modality_data)} modalities "
                   f"using {fusion_strategy.value} fusion (session: {session_id})")
        
        try:
            # Temporal synchronization
            if len(modality_data) > 1:
                # Get reference timestamp (most recent data)
                ref_timestamp = max(data.timestamp for data in modality_data.values())
                
                # Add to temporal synchronizer
                for data in modality_data.values():
                    self.temporal_sync.add_data(data)
                
                # Get synchronized batch
                synchronized_data = self.temporal_sync.get_synchronized_batch(ref_timestamp)
                
                # Use synchronized data if available, otherwise use original
                if synchronized_data:
                    modality_data = synchronized_data
                    logger.info(f"Used temporally synchronized data with {len(synchronized_data)} modalities")
            
            # Apply fusion strategy
            if fusion_strategy == FusionStrategy.EARLY:
                fused_result = await self._early_fusion_pipeline(modality_data, session_id)
            elif fusion_strategy == FusionStrategy.LATE:
                fused_result = await self._late_fusion_pipeline(modality_data, session_id)
            elif fusion_strategy == FusionStrategy.ATTENTION:
                fused_result = await self._attention_fusion_pipeline(modality_data, session_id)
            else:  # HYBRID
                fused_result = await self._hybrid_fusion_pipeline(modality_data, session_id)
            
            # Add processing metadata
            processing_time = time.time() - start_time
            fused_result.processing_latency = processing_time
            fused_result.metadata.update({
                'session_id': session_id,
                'processing_time': processing_time,
                'input_modalities': [m.value for m in modality_data.keys()],
                'timestamp': time.time()
            })
            
            # Update statistics
            self.processing_stats['total_requests'] += 1
            self.processing_stats['total_processing_time'] += processing_time
            self.processing_stats[f'{fusion_strategy.value}_requests'] += 1
            
            logger.info(f"Multi-modal fusion completed in {processing_time:.3f}s "
                       f"(strategy: {fusion_strategy.value}, session: {session_id})")
            
            return fused_result
            
        except Exception as e:
            logger.error(f"Multi-modal fusion failed for session {session_id}: {e}")
            # Return empty result with error information
            return FusionResult(
                fused_representation=np.zeros(self.config['feature_dim']),
                fusion_strategy=fusion_strategy,
                contributing_modalities=list(modality_data.keys()),
                confidence_scores={m: 0.0 for m in modality_data.keys()},
                metadata={'error': str(e), 'session_id': session_id},
                processing_latency=time.time() - start_time
            )
    
    async def _early_fusion_pipeline(self, modality_data: Dict[ModalityType, ModalityData], 
                                   session_id: str) -> FusionResult:
        """Early fusion processing pipeline"""
        logger.debug(f"Executing early fusion pipeline (session: {session_id})")
        
        # Preprocess modalities if needed
        preprocessed_data = {}
        for modality, data in modality_data.items():
            preprocessed_data[modality] = await self._preprocess_modality(data)
        
        # Perform early fusion
        fused_features = self.early_fusion.fuse(preprocessed_data)
        
        # Calculate confidence scores based on data quality
        confidence_scores = {}
        for modality, data in preprocessed_data.items():
            confidence_scores[modality] = data.confidence
        
        return FusionResult(
            fused_representation=fused_features,
            fusion_strategy=FusionStrategy.EARLY,
            contributing_modalities=list(modality_data.keys()),
            confidence_scores=confidence_scores,
            metadata={'preprocessing_applied': True}
        )
    
    async def _late_fusion_pipeline(self, modality_data: Dict[ModalityType, ModalityData],
                                  session_id: str) -> FusionResult:
        """Late fusion processing pipeline"""
        logger.debug(f"Executing late fusion pipeline (session: {session_id})")
        
        # Process each modality independently
        modality_results = {}
        confidence_scores = {}
        
        for modality, data in modality_data.items():
            result = await self._process_modality_independently(data, session_id)
            modality_results[modality] = result
            confidence_scores[modality] = result.get('confidence', data.confidence)
        
        # Perform late fusion
        fused_decision = self.late_fusion.fuse(modality_results)
        
        # Convert to consistent representation format
        if 'score' in fused_decision:
            # Create representation based on decision
            representation = np.array([fused_decision['score']] * self.config['feature_dim'])
        else:
            representation = np.zeros(self.config['feature_dim'])
        
        return FusionResult(
            fused_representation=representation,
            fusion_strategy=FusionStrategy.LATE,
            contributing_modalities=list(modality_data.keys()),
            confidence_scores=confidence_scores,
            metadata=fused_decision
        )
    
    async def _attention_fusion_pipeline(self, modality_data: Dict[ModalityType, ModalityData],
                                       session_id: str) -> FusionResult:
        """Attention-based fusion processing pipeline"""
        logger.debug(f"Executing attention fusion pipeline (session: {session_id})")
        
        # Initialize cross-modal attention if needed
        if self.cross_attention is None:
            feature_dims = {modality: self.config['feature_dim'] for modality in modality_data.keys()}
            self.cross_attention = CrossModalAttention(feature_dims)
        
        # Extract features for attention mechanism
        modality_features = {}
        confidence_scores = {}
        
        for modality, data in modality_data.items():
            # Preprocess and extract features
            preprocessed = await self._preprocess_modality(data)
            if preprocessed.embedding is not None:
                features = torch.tensor(preprocessed.embedding, dtype=torch.float32).unsqueeze(0)
            else:
                # Generate features based on modality type
                features = torch.randn(1, self.config['feature_dim'])  # Placeholder
            
            modality_features[modality] = features
            confidence_scores[modality] = preprocessed.confidence
        
        # Apply cross-modal attention
        fused_features, attention_weights = self.cross_attention(modality_features)
        
        return FusionResult(
            fused_representation=fused_features.detach().numpy().flatten(),
            fusion_strategy=FusionStrategy.ATTENTION,
            contributing_modalities=list(modality_data.keys()),
            confidence_scores=confidence_scores,
            fusion_weights=attention_weights,
            metadata={'attention_mechanism': 'cross_modal'}
        )
    
    async def _hybrid_fusion_pipeline(self, modality_data: Dict[ModalityType, ModalityData],
                                    session_id: str) -> FusionResult:
        """Hybrid fusion combining multiple strategies"""
        logger.debug(f"Executing hybrid fusion pipeline (session: {session_id})")
        
        # First stage: Early fusion for tightly coupled modalities
        early_group = {}
        late_group = {}
        
        # Group modalities based on coupling (simple heuristic)
        for modality, data in modality_data.items():
            if modality in [ModalityType.TEXT, ModalityType.VOICE]:
                # Text and voice are tightly coupled (semantic content)
                early_group[modality] = data
            else:
                # Visual and sensor data processed separately
                late_group[modality] = data
        
        # Process early fusion group
        early_result = None
        if early_group:
            early_result = await self._early_fusion_pipeline(early_group, f"{session_id}_early")
        
        # Process late fusion group
        late_results = {}
        for modality, data in late_group.items():
            result = await self._process_modality_independently(data, f"{session_id}_late")
            late_results[modality] = result
        
        # Combine results
        all_confidence = {}
        
        if early_result:
            all_confidence.update(early_result.confidence_scores)
        
        # Second stage: Combine early fusion result with late fusion results
        final_features = []
        
        if early_result:
            final_features.append(early_result.fused_representation)
        
        for modality, result in late_results.items():
            # Convert result to feature vector
            if 'score' in result:
                feature_vec = np.array([result['score']] * (self.config['feature_dim'] // 4))
                final_features.append(feature_vec)
            all_confidence[modality] = result.get('confidence', 1.0)
        
        # Concatenate and project to target dimension
        if final_features:
            combined_features = np.concatenate(final_features)
            if len(combined_features) > self.config['feature_dim']:
                # Downsample
                step = len(combined_features) // self.config['feature_dim']
                final_representation = combined_features[::step][:self.config['feature_dim']]
            else:
                # Pad
                final_representation = np.pad(combined_features, 
                                            (0, self.config['feature_dim'] - len(combined_features)))
        else:
            final_representation = np.zeros(self.config['feature_dim'])
        
        return FusionResult(
            fused_representation=final_representation,
            fusion_strategy=FusionStrategy.HYBRID,
            contributing_modalities=list(modality_data.keys()),
            confidence_scores=all_confidence,
            metadata={
                'early_fusion_modalities': list(early_group.keys()),
                'late_fusion_modalities': list(late_group.keys()),
                'hybrid_stages': 2
            }
        )
    
    async def _preprocess_modality(self, data: ModalityData) -> ModalityData:
        """Preprocess modality data"""
        # Apply modality-specific preprocessing
        if data.modality_type == ModalityType.TEXT:
            return await self._preprocess_text(data)
        elif data.modality_type == ModalityType.VOICE:
            return await self._preprocess_voice(data)
        elif data.modality_type == ModalityType.VISUAL:
            return await self._preprocess_visual(data)
        elif data.modality_type == ModalityType.SENSOR:
            return await self._preprocess_sensor(data)
        else:
            return data
    
    async def _preprocess_text(self, data: ModalityData) -> ModalityData:
        """Preprocess text data using Ollama integration"""
        try:
            # In a real implementation, would call Ollama API for embeddings
            # For now, create a simple text representation
            text = str(data.data)
            
            # Simple text features (placeholder for Ollama embeddings)
            embedding = np.random.rand(self.config['feature_dim'])  # Would be actual embeddings
            
            processed_data = ModalityData(
                modality_type=data.modality_type,
                data=text.lower().strip(),
                timestamp=data.timestamp,
                confidence=min(data.confidence, len(text) / 100.0),  # Length-based confidence
                metadata=data.metadata,
                embedding=embedding,
                preprocessing_applied=['lowercase', 'strip', 'embedding']
            )
            
            return processed_data
            
        except Exception as e:
            logger.warning(f"Text preprocessing failed: {e}")
            return data
    
    async def _preprocess_voice(self, data: ModalityData) -> ModalityData:
        """Preprocess voice data"""
        try:
            # Placeholder for voice preprocessing (MFCC, spectrograms, etc.)
            embedding = np.random.rand(self.config['feature_dim'])
            
            processed_data = ModalityData(
                modality_type=data.modality_type,
                data=data.data,
                timestamp=data.timestamp,
                confidence=data.confidence * 0.9,  # Slight reduction for processing uncertainty
                metadata=data.metadata,
                embedding=embedding,
                preprocessing_applied=['spectral_analysis', 'mfcc', 'embedding']
            )
            
            return processed_data
            
        except Exception as e:
            logger.warning(f"Voice preprocessing failed: {e}")
            return data
    
    async def _preprocess_visual(self, data: ModalityData) -> ModalityData:
        """Preprocess visual data"""
        try:
            # Placeholder for visual preprocessing (CNN features, etc.)
            embedding = np.random.rand(self.config['feature_dim'])
            
            processed_data = ModalityData(
                modality_type=data.modality_type,
                data=data.data,
                timestamp=data.timestamp,
                confidence=data.confidence,
                metadata=data.metadata,
                embedding=embedding,
                preprocessing_applied=['resize', 'normalize', 'cnn_features']
            )
            
            return processed_data
            
        except Exception as e:
            logger.warning(f"Visual preprocessing failed: {e}")
            return data
    
    async def _preprocess_sensor(self, data: ModalityData) -> ModalityData:
        """Preprocess sensor data"""
        try:
            # Statistical and time-series features
            if isinstance(data.data, (list, np.ndarray)):
                sensor_array = np.array(data.data)
                
                # Extract statistical features
                features = np.array([
                    np.mean(sensor_array),
                    np.std(sensor_array),
                    np.min(sensor_array),
                    np.max(sensor_array),
                    np.median(sensor_array),
                    np.percentile(sensor_array, 25),
                    np.percentile(sensor_array, 75)
                ])
                
                # Pad to target dimension
                embedding = np.pad(features, (0, self.config['feature_dim'] - len(features)))
                
                processed_data = ModalityData(
                    modality_type=data.modality_type,
                    data=data.data,
                    timestamp=data.timestamp,
                    confidence=data.confidence,
                    metadata=data.metadata,
                    embedding=embedding,
                    preprocessing_applied=['statistical_features', 'normalization']
                )
                
                return processed_data
            
        except Exception as e:
            logger.warning(f"Sensor preprocessing failed: {e}")
        
        return data
    
    async def _process_modality_independently(self, data: ModalityData, session_id: str) -> Dict[str, Any]:
        """Process individual modality for late fusion"""
        try:
            # Modality-specific processing
            if data.modality_type == ModalityType.TEXT:
                # Text analysis using Ollama (placeholder)
                sentiment_score = np.random.rand()  # Would use actual NLP analysis
                return {
                    'score': sentiment_score,
                    'confidence': data.confidence,
                    'decision': 'positive' if sentiment_score > 0.5 else 'negative',
                    'metadata': {'analysis_type': 'sentiment', 'model': 'tinyllama'}
                }
            
            elif data.modality_type == ModalityType.VOICE:
                # Voice analysis (emotion, speech-to-text, etc.)
                emotion_score = np.random.rand()
                return {
                    'score': emotion_score,
                    'confidence': data.confidence,
                    'decision': 'calm' if emotion_score < 0.3 else 'excited' if emotion_score > 0.7 else 'neutral',
                    'metadata': {'analysis_type': 'emotion', 'features': 'prosodic'}
                }
            
            elif data.modality_type == ModalityType.VISUAL:
                # Visual analysis (object detection, scene understanding)
                complexity_score = np.random.rand()
                return {
                    'score': complexity_score,
                    'confidence': data.confidence,
                    'decision': 'simple' if complexity_score < 0.4 else 'complex',
                    'metadata': {'analysis_type': 'scene_complexity'}
                }
            
            elif data.modality_type == ModalityType.SENSOR:
                # Sensor data analysis (anomaly detection, pattern recognition)
                if isinstance(data.data, (list, np.ndarray)):
                    sensor_array = np.array(data.data)
                    anomaly_score = np.abs(np.mean(sensor_array) - np.median(sensor_array))
                    return {
                        'score': min(anomaly_score, 1.0),
                        'confidence': data.confidence,
                        'decision': 'anomaly' if anomaly_score > 0.5 else 'normal',
                        'metadata': {'analysis_type': 'anomaly_detection'}
                    }
            
            # Default case
            return {
                'score': 0.5,
                'confidence': data.confidence,
                'decision': 'unknown',
                'metadata': {'analysis_type': 'default'}
            }
            
        except Exception as e:
            logger.error(f"Independent modality processing failed: {e}")
            return {
                'score': 0.0,
                'confidence': 0.0,
                'decision': 'error',
                'metadata': {'error': str(e)}
            }
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = dict(self.processing_stats)
        
        if stats.get('total_requests', 0) > 0:
            stats['average_processing_time'] = stats.get('total_processing_time', 0) / stats['total_requests']
        
        stats['active_sessions'] = len(self.active_sessions)
        stats['queue_sizes'] = {
            'input': self.input_queue.qsize(),
            'processing': self.processing_queue.qsize(),
            'output': self.output_queue.qsize()
        }
        
        return stats
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Multi-Modal Fusion Coordinator")
        
        # Stop executors
        self.executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        
        logger.info("Multi-Modal Fusion Coordinator shutdown complete")

# Example usage and testing
async def main():
    """Example usage of the Multi-Modal Fusion Coordinator"""
    
    # Initialize coordinator
    coordinator = MultiModalFusionCoordinator()
    
    # Create sample multi-modal data
    text_data = ModalityData(
        modality_type=ModalityType.TEXT,
        data="Hello, this is a test message for multi-modal fusion.",
        timestamp=time.time(),
        confidence=0.95
    )
    
    voice_data = ModalityData(
        modality_type=ModalityType.VOICE,
        data="[audio_data_placeholder]",  # Would contain actual audio data
        timestamp=time.time() + 0.1,  # Slightly later
        confidence=0.85
    )
    
    sensor_data = ModalityData(
        modality_type=ModalityType.SENSOR,
        data=[1.0, 1.2, 0.8, 1.1, 0.9, 1.0, 1.3, 0.7, 1.0, 1.1],  # Time series
        timestamp=time.time() + 0.05,
        confidence=0.9
    )
    
    # Test different fusion strategies
    modality_input = {
        ModalityType.TEXT: text_data,
        ModalityType.VOICE: voice_data,
        ModalityType.SENSOR: sensor_data
    }
    
    # Early fusion
    print("Testing Early Fusion:")
    early_result = await coordinator.process_multi_modal_input(
        modality_input, 
        FusionStrategy.EARLY
    )
    print(f"  Result shape: {early_result.fused_representation.shape}")
    print(f"  Processing time: {early_result.processing_latency:.3f}s")
    print(f"  Confidence scores: {early_result.confidence_scores}")
    
    # Late fusion
    print("\nTesting Late Fusion:")
    late_result = await coordinator.process_multi_modal_input(
        modality_input,
        FusionStrategy.LATE
    )
    print(f"  Result shape: {late_result.fused_representation.shape}")
    print(f"  Processing time: {late_result.processing_latency:.3f}s")
    print(f"  Metadata: {late_result.metadata}")
    
    # Hybrid fusion
    print("\nTesting Hybrid Fusion:")
    hybrid_result = await coordinator.process_multi_modal_input(
        modality_input,
        FusionStrategy.HYBRID
    )
    print(f"  Result shape: {hybrid_result.fused_representation.shape}")
    print(f"  Processing time: {hybrid_result.processing_latency:.3f}s")
    print(f"  Fusion weights: {hybrid_result.fusion_weights}")
    
    # Print statistics
    print("\nProcessing Statistics:")
    stats = coordinator.get_processing_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Shutdown
    await coordinator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())