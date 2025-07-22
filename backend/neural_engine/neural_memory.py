#!/usr/bin/env python3
"""
Neural Memory System
Implements various forms of neural memory including working memory, long-term memory, and episodic memory
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone
import numpy as np
from collections import deque
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """Configuration for neural memory system"""
    # Working memory
    working_memory_size: int = 256
    working_memory_decay: float = 0.1
    
    # Long-term memory
    ltm_capacity: int = 10000
    ltm_consolidation_threshold: float = 0.8
    ltm_retrieval_threshold: float = 0.7
    
    # Episodic memory
    episode_max_length: int = 1000
    episode_capacity: int = 1000
    temporal_context_size: int = 64
    
    # Associative memory
    associative_memory_size: int = 512
    association_strength_threshold: float = 0.5
    
    # Memory consolidation
    consolidation_rate: float = 0.01
    interference_threshold: float = 0.3
    
    # Device settings
    device: str = "auto"
    dtype: str = "float32"

class WorkingMemory(nn.Module):
    """Working memory system with attention mechanism"""
    
    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config
        
        # Memory buffer
        self.memory_buffer = nn.Parameter(
            torch.zeros(config.working_memory_size, 512),
            requires_grad=False
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=0.1
        )
        
        # Gating mechanism
        self.gate = nn.Linear(512, 512)
        
        # Current write position
        self.write_position = 0
        
        # Usage tracking
        self.usage_count = torch.zeros(config.working_memory_size)
        self.access_time = torch.zeros(config.working_memory_size)
        
    def forward(self, input_data: torch.Tensor, query: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Process working memory operations"""
        batch_size = input_data.size(0)
        
        # Store in working memory
        self.store(input_data)
        
        # Retrieve from working memory
        if query is not None:
            retrieved = self.retrieve(query)
        else:
            retrieved = self.retrieve(input_data)
        
        # Apply attention
        attended_output, attention_weights = self.attention(
            query=retrieved.unsqueeze(0),
            key=self.memory_buffer.unsqueeze(0),
            value=self.memory_buffer.unsqueeze(0)
        )
        
        # Apply gating
        gate_signal = torch.sigmoid(self.gate(input_data))
        gated_output = gate_signal * attended_output.squeeze(0) + (1 - gate_signal) * input_data
        
        return {
            "output": gated_output,
            "retrieved": retrieved,
            "attention_weights": attention_weights,
            "memory_usage": self.get_memory_usage()
        }
    
    def store(self, data: torch.Tensor):
        """Store data in working memory"""
        batch_size = data.size(0)
        
        for i in range(batch_size):
            # Find least recently used position
            if self.write_position >= self.config.working_memory_size:
                # Find position with oldest access time
                oldest_pos = torch.argmin(self.access_time).item()
                self.write_position = oldest_pos
            
            # Store data
            self.memory_buffer[self.write_position] = data[i]
            self.usage_count[self.write_position] += 1
            self.access_time[self.write_position] = datetime.now(timezone.utc).timestamp()
            
            self.write_position = (self.write_position + 1) % self.config.working_memory_size
    
    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve data from working memory based on query"""
        # Calculate similarity with stored memories
        similarities = torch.matmul(query, self.memory_buffer.T)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(similarities, dim=-1)
        
        # Retrieve weighted combination
        retrieved = torch.matmul(attention_weights, self.memory_buffer)
        
        # Update access times for accessed memories
        accessed_positions = torch.argmax(attention_weights, dim=-1)
        for pos in accessed_positions:
            self.access_time[pos] = datetime.now(timezone.utc).timestamp()
        
        return retrieved
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get working memory usage statistics"""
        return {
            "utilization": (self.usage_count > 0).float().mean().item(),
            "average_usage": self.usage_count.mean().item(),
            "max_usage": self.usage_count.max().item(),
            "write_position": self.write_position
        }
    
    def clear(self):
        """Clear working memory"""
        self.memory_buffer.zero_()
        self.usage_count.zero_()
        self.access_time.zero_()
        self.write_position = 0

class LongTermMemory:
    """Long-term memory with consolidation and retrieval"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Memory storage
        self.memories = {}
        self.memory_strengths = {}
        self.memory_ages = {}
        self.memory_access_counts = {}
        
        # Consolidation parameters
        self.consolidation_threshold = config.ltm_consolidation_threshold
        self.retrieval_threshold = config.ltm_retrieval_threshold
        
        # Memory indexing
        self.next_memory_id = 0
        
    def store(self, memory_data: torch.Tensor, memory_key: str = None, 
             consolidation_strength: float = 1.0) -> str:
        """Store memory in long-term memory"""
        if memory_key is None:
            memory_key = f"memory_{self.next_memory_id}"
            self.next_memory_id += 1
        
        # Store memory
        self.memories[memory_key] = memory_data.detach().clone()
        self.memory_strengths[memory_key] = consolidation_strength
        self.memory_ages[memory_key] = datetime.now(timezone.utc).timestamp()
        self.memory_access_counts[memory_key] = 0
        
        # Consolidate if above threshold
        if consolidation_strength >= self.consolidation_threshold:
            self._consolidate_memory(memory_key)
        
        # Manage memory capacity
        self._manage_capacity()
        
        return memory_key
    
    def retrieve(self, query: torch.Tensor, top_k: int = 5) -> List[Tuple[str, torch.Tensor, float]]:
        """Retrieve memories based on query"""
        if not self.memories:
            return []
        
        # Calculate similarities
        similarities = {}
        for key, memory in self.memories.items():
            if self.memory_strengths[key] >= self.retrieval_threshold:
                similarity = F.cosine_similarity(query.flatten(), memory.flatten(), dim=0)
                similarities[key] = similarity.item()
        
        # Sort by similarity
        sorted_memories = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-k memories
        results = []
        for key, similarity in sorted_memories[:top_k]:
            memory_data = self.memories[key]
            
            # Update access count
            self.memory_access_counts[key] += 1
            
            # Strengthen memory based on retrieval
            self._strengthen_memory(key, similarity)
            
            results.append((key, memory_data, similarity))
        
        return results
    
    def _consolidate_memory(self, memory_key: str):
        """Consolidate memory to make it more permanent"""
        if memory_key in self.memory_strengths:
            # Increase strength
            self.memory_strengths[memory_key] = min(
                self.memory_strengths[memory_key] * 1.2, 
                2.0
            )
            
            # Update age to reflect consolidation
            self.memory_ages[memory_key] = datetime.now(timezone.utc).timestamp()
    
    def _strengthen_memory(self, memory_key: str, retrieval_strength: float):
        """Strengthen memory based on retrieval"""
        if memory_key in self.memory_strengths:
            strength_increase = retrieval_strength * self.config.consolidation_rate
            self.memory_strengths[memory_key] += strength_increase
            self.memory_strengths[memory_key] = min(self.memory_strengths[memory_key], 2.0)
    
    def _manage_capacity(self):
        """Manage memory capacity by removing weak memories"""
        if len(self.memories) > self.config.ltm_capacity:
            # Calculate memory importance scores
            importance_scores = {}
            current_time = datetime.now(timezone.utc).timestamp()
            
            for key in self.memories:
                age = current_time - self.memory_ages[key]
                access_count = self.memory_access_counts[key]
                strength = self.memory_strengths[key]
                
                # Importance based on strength, recency, and access frequency
                importance = strength * (1 + access_count) / (1 + age / 3600)  # Age in hours
                importance_scores[key] = importance
            
            # Remove least important memories
            sorted_memories = sorted(importance_scores.items(), key=lambda x: x[1])
            memories_to_remove = len(self.memories) - self.config.ltm_capacity
            
            for key, _ in sorted_memories[:memories_to_remove]:
                del self.memories[key]
                del self.memory_strengths[key]
                del self.memory_ages[key]
                del self.memory_access_counts[key]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get long-term memory statistics"""
        if not self.memories:
            return {"total_memories": 0}
        
        strengths = list(self.memory_strengths.values())
        ages = list(self.memory_ages.values())
        access_counts = list(self.memory_access_counts.values())
        
        current_time = datetime.now(timezone.utc).timestamp()
        
        return {
            "total_memories": len(self.memories),
            "average_strength": np.mean(strengths),
            "max_strength": max(strengths),
            "average_age_hours": np.mean([(current_time - age) / 3600 for age in ages]),
            "average_access_count": np.mean(access_counts),
            "consolidated_memories": sum(1 for s in strengths if s >= self.consolidation_threshold)
        }
    
    def clear(self):
        """Clear long-term memory"""
        self.memories.clear()
        self.memory_strengths.clear()
        self.memory_ages.clear()
        self.memory_access_counts.clear()
        self.next_memory_id = 0

class EpisodicMemory:
    """Episodic memory for sequential experiences"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Episode storage
        self.episodes = deque(maxlen=config.episode_capacity)
        self.current_episode = []
        self.episode_contexts = {}
        
        # Temporal context
        self.temporal_context = torch.zeros(config.temporal_context_size, device=self.device)
        
    def add_experience(self, experience: torch.Tensor, context: Optional[Dict[str, Any]] = None):
        """Add experience to current episode"""
        experience_data = {
            "data": experience.detach().clone(),
            "timestamp": datetime.now(timezone.utc).timestamp(),
            "context": context or {},
            "temporal_context": self.temporal_context.clone()
        }
        
        self.current_episode.append(experience_data)
        
        # Update temporal context
        self._update_temporal_context(experience)
        
        # Check if episode is complete
        if len(self.current_episode) >= self.config.episode_max_length:
            self.finalize_episode()
    
    def finalize_episode(self, episode_summary: Optional[Dict[str, Any]] = None):
        """Finalize current episode and start new one"""
        if self.current_episode:
            episode_data = {
                "experiences": self.current_episode.copy(),
                "summary": episode_summary or {},
                "start_time": self.current_episode[0]["timestamp"],
                "end_time": self.current_episode[-1]["timestamp"],
                "length": len(self.current_episode)
            }
            
            self.episodes.append(episode_data)
            self.current_episode.clear()
            
            # Reset temporal context
            self.temporal_context.zero_()
    
    def _update_temporal_context(self, experience: torch.Tensor):
        """Update temporal context with new experience"""
        # Simple decay and update
        self.temporal_context *= 0.9
        
        # Add new experience influence
        if experience.numel() <= self.config.temporal_context_size:
            self.temporal_context[:experience.numel()] += experience.flatten()
        else:
            # Compress experience to fit context size
            compressed = F.adaptive_avg_pool1d(
                experience.flatten().unsqueeze(0).unsqueeze(0),
                self.config.temporal_context_size
            ).squeeze()
            self.temporal_context += compressed
    
    def retrieve_episodes(self, query: torch.Tensor, max_episodes: int = 10) -> List[Dict[str, Any]]:
        """Retrieve episodes based on query"""
        if not self.episodes:
            return []
        
        # Calculate episode similarities
        episode_similarities = []
        
        for i, episode in enumerate(self.episodes):
            # Calculate similarity with episode experiences
            similarities = []
            for exp in episode["experiences"]:
                sim = F.cosine_similarity(query.flatten(), exp["data"].flatten(), dim=0)
                similarities.append(sim.item())
            
            # Use maximum similarity as episode similarity
            episode_sim = max(similarities) if similarities else 0
            episode_similarities.append((i, episode_sim))
        
        # Sort by similarity
        episode_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top episodes
        results = []
        for i, sim in episode_similarities[:max_episodes]:
            episode_data = self.episodes[i]
            results.append({
                "episode": episode_data,
                "similarity": sim,
                "episode_index": i
            })
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get episodic memory statistics"""
        if not self.episodes:
            return {
                "total_episodes": 0,
                "current_episode_length": len(self.current_episode)
            }
        
        episode_lengths = [ep["length"] for ep in self.episodes]
        episode_durations = [ep["end_time"] - ep["start_time"] for ep in self.episodes]
        
        return {
            "total_episodes": len(self.episodes),
            "current_episode_length": len(self.current_episode),
            "average_episode_length": np.mean(episode_lengths),
            "max_episode_length": max(episode_lengths),
            "average_episode_duration": np.mean(episode_durations),
            "total_experiences": sum(episode_lengths)
        }
    
    def clear(self):
        """Clear episodic memory"""
        self.episodes.clear()
        self.current_episode.clear()
        self.temporal_context.zero_()

class NeuralMemorySystem:
    """
    Integrated neural memory system
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = MemoryConfig(**config) if config else MemoryConfig()
        
        # Setup device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        # Memory subsystems
        self.working_memory = WorkingMemory(self.config)
        self.long_term_memory = LongTermMemory(self.config)
        self.episodic_memory = EpisodicMemory(self.config)
        
        # Move to device
        self.working_memory.to(self.device)
        
        # State
        self.is_initialized = False
        self.memory_operations = 0
        
        # Statistics
        self.memory_stats = {
            "stores": 0,
            "retrievals": 0,
            "consolidations": 0,
            "episodes": 0
        }
        
        logger.info("Neural memory system created")
    
    async def initialize(self) -> bool:
        """Initialize neural memory system"""
        try:
            if self.is_initialized:
                return True
            
            # Initialize working memory
            self.working_memory.clear()
            
            # Initialize long-term memory
            self.long_term_memory.clear()
            
            # Initialize episodic memory
            self.episodic_memory.clear()
            
            self.is_initialized = True
            logger.info("Neural memory system initialized")
            return True
            
        except Exception as e:
            logger.error(f"Memory system initialization failed: {e}")
            return False
    
    async def store(self, data: torch.Tensor, results: Dict[str, Any]) -> Dict[str, Any]:
        """Store data and results in memory system"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            data = data.to(self.device)
            storage_results = {}
            
            # Store in working memory
            wm_result = self.working_memory(data)
            storage_results["working_memory"] = wm_result
            
            # Store in long-term memory if significant
            if "output" in results:
                output_tensor = results["output"]
                if isinstance(output_tensor, torch.Tensor):
                    # Calculate significance based on activation magnitude
                    significance = torch.abs(output_tensor).mean().item()
                    
                    if significance > 0.1:  # Threshold for significance
                        memory_key = self.long_term_memory.store(
                            output_tensor, 
                            consolidation_strength=significance
                        )
                        storage_results["long_term_memory"] = memory_key
                        self.memory_stats["stores"] += 1
            
            # Add to episodic memory
            context = {
                "operation": "neural_processing",
                "results": results,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.episodic_memory.add_experience(data, context)
            
            self.memory_operations += 1
            
            return storage_results
            
        except Exception as e:
            logger.error(f"Memory storage failed: {e}")
            raise
    
    async def retrieve(self, query: torch.Tensor, memory_type: str = "all") -> Dict[str, Any]:
        """Retrieve memories based on query"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            query = query.to(self.device)
            retrieval_results = {}
            
            # Retrieve from working memory
            if memory_type in ["all", "working"]:
                wm_result = self.working_memory(query, query)
                retrieval_results["working_memory"] = wm_result
            
            # Retrieve from long-term memory
            if memory_type in ["all", "long_term"]:
                ltm_results = self.long_term_memory.retrieve(query)
                retrieval_results["long_term_memory"] = ltm_results
                self.memory_stats["retrievals"] += 1
            
            # Retrieve from episodic memory
            if memory_type in ["all", "episodic"]:
                episodic_results = self.episodic_memory.retrieve_episodes(query)
                retrieval_results["episodic_memory"] = episodic_results
            
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            raise
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        return {
            "system_stats": {
                "memory_operations": self.memory_operations,
                "stores": self.memory_stats["stores"],
                "retrievals": self.memory_stats["retrievals"],
                "is_initialized": self.is_initialized
            },
            "working_memory": self.working_memory.get_memory_usage(),
            "long_term_memory": self.long_term_memory.get_statistics(),
            "episodic_memory": self.episodic_memory.get_statistics()
        }
    
    def consolidate_memories(self):
        """Trigger memory consolidation process"""
        # Transfer important working memories to long-term storage
        # This is a simplified version - in practice, this would be more sophisticated
        self.memory_stats["consolidations"] += 1
        logger.info("Memory consolidation triggered")
    
    def finalize_episode(self, summary: Optional[Dict[str, Any]] = None):
        """Finalize current episode"""
        self.episodic_memory.finalize_episode(summary)
        self.memory_stats["episodes"] += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get memory system status"""
        return {
            "is_initialized": self.is_initialized,
            "device": str(self.device),
            "config": self.config.__dict__,
            "statistics": self.get_memory_statistics()
        }
    
    def health_check(self) -> bool:
        """Check memory system health"""
        try:
            return (
                self.is_initialized and
                self.working_memory is not None and
                self.long_term_memory is not None and
                self.episodic_memory is not None
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown neural memory system"""
        try:
            # Clear all memory systems
            self.working_memory.clear()
            self.long_term_memory.clear()
            self.episodic_memory.clear()
            
            # Reset statistics
            self.memory_stats = {
                "stores": 0,
                "retrievals": 0,
                "consolidations": 0,
                "episodes": 0
            }
            
            self.memory_operations = 0
            self.is_initialized = False
            
            logger.info("Neural memory system shutdown completed")
            return True
            
        except Exception as e:
            logger.error(f"Memory system shutdown failed: {e}")
            return False