#!/usr/bin/env python3
"""
Cognitive Memory Implementation Example
Production-ready implementation of the cognitive architecture memory schema
"""

import json
import uuid
import hashlib
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
import numpy as np
from pathlib import Path

# Memory type enumeration
class MemoryType(Enum):
    IMMEDIATE = "immediate"
    SESSION = "session" 
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    META = "meta"
    COLLECTIVE = "collective"

# Relationship types for code connections
class RelationshipType(Enum):
    IMPORTS = "imports"
    EXPORTS = "exports"
    CALLS = "calls"
    CALLED_BY = "called_by"
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    DEPENDS_ON = "depends_on"
    DEPENDENCY_OF = "dependency_of"
    TESTS = "tests"
    TESTED_BY = "tested_by"
    DOCUMENTS = "documents"
    DOCUMENTED_BY = "documented_by"
    REFERENCES = "references"
    REFERENCED_BY = "referenced_by"
    MODIFIES = "modifies"
    MODIFIED_BY = "modified_by"

@dataclass
class FileReference:
    """Reference to a specific location in code"""
    path: str
    line_start: int
    line_end: Optional[int] = None
    commit_hash: Optional[str] = None
    content_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

@dataclass
class CodeRelationship:
    """Relationship between code elements"""
    type: RelationshipType
    source: FileReference
    target: FileReference
    strength: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "strength": self.strength
        }

@dataclass
class VectorEmbedding:
    """Vector embedding for semantic search"""
    model: str
    dimensions: int
    vector: np.ndarray
    sparse_indices: Optional[List[int]] = None
    sparse_values: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "model": self.model,
            "dimensions": self.dimensions,
            "vector": self.vector.tolist()
        }
        if self.sparse_indices:
            result["sparse_indices"] = self.sparse_indices
        if self.sparse_values:
            result["sparse_values"] = self.sparse_values
        return result

@dataclass
class Tag:
    """Categorized tag for memory organization"""
    name: str
    category: str
    weight: float = 1.0
    
    VALID_CATEGORIES = [
        "language", "framework", "pattern", "domain", "feature",
        "error", "performance", "security", "architecture", "cognitive"
    ]
    
    def __post_init__(self):
        if self.category not in self.VALID_CATEGORIES:
            raise ValueError(f"Invalid category: {self.category}")
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class Memory:
    """Base memory class with common attributes"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    type: MemoryType = MemoryType.IMMEDIATE
    importance: int = 5
    confidence: float = 1.0
    content: Dict[str, Any] = field(default_factory=dict)
    tags: List[Tag] = field(default_factory=list)
    embeddings: List[VectorEmbedding] = field(default_factory=list)
    ttl_seconds: int = -1
    updated_at: Optional[datetime] = None
    accessed_at: Optional[datetime] = None
    access_count: int = 0
    
    def __post_init__(self):
        if not 1 <= self.importance <= 10:
            raise ValueError(f"Importance must be between 1 and 10")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "type": self.type.value,
            "importance": self.importance,
            "confidence": self.confidence,
            "content": self.content,
            "tags": [tag.to_dict() for tag in self.tags],
            "embeddings": [emb.to_dict() for emb in self.embeddings],
            "ttl_seconds": self.ttl_seconds,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "accessed_at": self.accessed_at.isoformat() if self.accessed_at else None,
            "access_count": self.access_count
        }
    
    def is_expired(self) -> bool:
        """Check if memory has expired based on TTL"""
        if self.ttl_seconds <= 0:
            return False
        expiry_time = self.created_at + timedelta(seconds=self.ttl_seconds)
        return datetime.utcnow() > expiry_time

@dataclass
class ImmediateContextMemory(Memory):
    """Current conversation and active working memory"""
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    intent: Optional[str] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    context_window: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        super().__post_init__()
        self.type = MemoryType.IMMEDIATE
        if self.ttl_seconds == -1:
            self.ttl_seconds = 3600  # 1 hour default

@dataclass
class EpisodicMemory(Memory):
    """Event sequences and temporal experiences"""
    event_type: str = "code_change"
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    causal_chain: List[Dict[str, Any]] = field(default_factory=list)
    lessons_learned: List[Dict[str, Any]] = field(default_factory=list)
    
    VALID_EVENT_TYPES = [
        "code_change", "error_resolution", "deployment", "incident",
        "performance_optimization", "refactoring", "feature_implementation",
        "bug_fix", "configuration_change", "user_interaction"
    ]
    
    def __post_init__(self):
        super().__post_init__()
        self.type = MemoryType.EPISODIC
        if self.event_type not in self.VALID_EVENT_TYPES:
            raise ValueError(f"Invalid event type: {self.event_type}")
        if self.ttl_seconds == -1:
            self.ttl_seconds = 7776000  # 90 days default

@dataclass
class SemanticMemory(Memory):
    """Facts, concepts, and relationships"""
    concept: str = ""
    definition: str = ""
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        super().__post_init__()
        self.type = MemoryType.SEMANTIC
        self.ttl_seconds = -1  # Permanent by default

@dataclass
class ProceduralMemory(Memory):
    """How-to knowledge and automated procedures"""
    procedure_name: str = ""
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        super().__post_init__()
        self.type = MemoryType.PROCEDURAL
        self.ttl_seconds = -1  # Permanent by default

class CognitiveMemorySystem:
    """
    Main cognitive memory system implementation
    Manages all memory types with persistence and retrieval
    """
    
    def __init__(self, storage_path: str = "/opt/sutazaiapp/data/cognitive_memory"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory indices for fast access
        self.memories: Dict[str, Memory] = {}
        self.type_index: Dict[MemoryType, List[str]] = {t: [] for t in MemoryType}
        self.tag_index: Dict[str, List[str]] = {}
        self.importance_index: Dict[int, List[str]] = {i: [] for i in range(1, 11)}
        
        # Load existing memories from disk
        self._load_memories()
    
    def store(self, memory: Memory) -> str:
        """Store a memory with indexing and persistence"""
        
        # Update access metadata
        memory.accessed_at = datetime.utcnow()
        
        # Store in memory
        self.memories[memory.id] = memory
        
        # Update indices
        self.type_index[memory.type].append(memory.id)
        self.importance_index[memory.importance].append(memory.id)
        
        for tag in memory.tags:
            if tag.name not in self.tag_index:
                self.tag_index[tag.name] = []
            self.tag_index[tag.name].append(memory.id)
        
        # Persist to disk
        self._persist_memory(memory)
        
        # Trigger consolidation if needed
        if len(self.memories) % 100 == 0:
            asyncio.create_task(self._consolidate_memories())
        
        return memory.id
    
    def retrieve(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID"""
        
        memory = self.memories.get(memory_id)
        if memory:
            # Update access metadata
            memory.accessed_at = datetime.utcnow()
            memory.access_count += 1
            
            # Check expiration
            if memory.is_expired():
                self.delete(memory_id)
                return None
        
        return memory
    
    def search(self,
              query: Optional[str] = None,
              memory_types: Optional[List[MemoryType]] = None,
              importance_min: int = 1,
              tags: Optional[List[str]] = None,
              limit: int = 100) -> List[Memory]:
        """Search memories with multiple criteria"""
        
        results = []
        
        # Start with all memories or filtered by type
        if memory_types:
            candidate_ids = set()
            for mem_type in memory_types:
                candidate_ids.update(self.type_index.get(mem_type, []))
        else:
            candidate_ids = set(self.memories.keys())
        
        # Filter by importance
        if importance_min > 1:
            importance_ids = set()
            for imp in range(importance_min, 11):
                importance_ids.update(self.importance_index.get(imp, []))
            candidate_ids &= importance_ids
        
        # Filter by tags
        if tags:
            tag_ids = set()
            for tag in tags:
                tag_ids.update(self.tag_index.get(tag, []))
            candidate_ids &= tag_ids
        
        # Get memories and filter by query if provided
        for mem_id in candidate_ids:
            memory = self.memories.get(mem_id)
            if memory and not memory.is_expired():
                if query:
                    # Simple text search in content
                    content_str = json.dumps(memory.content).lower()
                    if query.lower() in content_str:
                        results.append(memory)
                else:
                    results.append(memory)
                
                if len(results) >= limit:
                    break
        
        # Sort by importance and recency
        results.sort(key=lambda m: (m.importance, m.created_at), reverse=True)
        
        return results[:limit]
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory and update indices"""
        
        memory = self.memories.get(memory_id)
        if not memory:
            return False
        
        # Remove from indices
        self.type_index[memory.type].remove(memory_id)
        self.importance_index[memory.importance].remove(memory_id)
        
        for tag in memory.tags:
            if tag.name in self.tag_index:
                self.tag_index[tag.name].remove(memory_id)
        
        # Remove from storage
        del self.memories[memory_id]
        
        # Remove from disk
        memory_file = self.storage_path / f"{memory_id}.json"
        if memory_file.exists():
            memory_file.unlink()
        
        return True
    
    def consolidate(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Consolidate short-term memories into long-term patterns"""
        
        # Find session memories
        session_memories = []
        for memory in self.memories.values():
            if memory.type in [MemoryType.IMMEDIATE, MemoryType.SESSION]:
                if isinstance(memory, ImmediateContextMemory):
                    if not session_id or memory.session_id == session_id:
                        session_memories.append(memory)
        
        # Extract patterns (simplified version)
        patterns = self._extract_patterns(session_memories)
        
        # Create long-term memories from patterns
        consolidated = []
        for pattern in patterns:
            if pattern['confidence'] > 0.7:
                long_term = Memory(
                    type=MemoryType.LONG_TERM,
                    importance=pattern['importance'],
                    confidence=pattern['confidence'],
                    content={
                        'pattern': pattern['description'],
                        'evidence': [m.id for m in pattern['memories']],
                        'session_id': session_id
                    },
                    tags=[Tag(name="consolidated", category="cognitive")]
                )
                memory_id = self.store(long_term)
                consolidated.append(memory_id)
        
        # Archive original memories
        for memory in session_memories:
            memory.importance = max(1, memory.importance - 2)  # Reduce importance
            memory.tags.append(Tag(name="archived", category="cognitive"))
        
        return {
            'consolidated_count': len(consolidated),
            'archived_count': len(session_memories),
            'patterns_found': len(patterns),
            'memory_ids': consolidated
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        
        total_memories = len(self.memories)
        by_type = {t.value: len(ids) for t, ids in self.type_index.items()}
        
        # Calculate total size
        total_size = sum(
            len(json.dumps(m.to_dict()).encode())
            for m in self.memories.values()
        )
        
        # Get average access metrics
        accessed_memories = [m for m in self.memories.values() if m.access_count > 0]
        avg_access_count = (
            sum(m.access_count for m in accessed_memories) / len(accessed_memories)
            if accessed_memories else 0
        )
        
        return {
            'total_memories': total_memories,
            'by_type': by_type,
            'total_size_bytes': total_size,
            'average_access_count': avg_access_count,
            'tag_count': len(self.tag_index),
            'expired_count': sum(1 for m in self.memories.values() if m.is_expired())
        }
    
    def cleanup_expired(self) -> int:
        """Remove expired memories"""
        
        expired_ids = [
            m.id for m in self.memories.values()
            if m.is_expired()
        ]
        
        for mem_id in expired_ids:
            self.delete(mem_id)
        
        return len(expired_ids)
    
    def _persist_memory(self, memory: Memory):
        """Save memory to disk"""
        
        memory_file = self.storage_path / f"{memory.id}.json"
        with open(memory_file, 'w') as f:
            json.dump(memory.to_dict(), f, indent=2)
    
    def _load_memories(self):
        """Load memories from disk on startup"""
        
        for memory_file in self.storage_path.glob("*.json"):
            try:
                with open(memory_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct memory object based on type
                mem_type = MemoryType(data['type'])
                
                if mem_type == MemoryType.IMMEDIATE:
                    memory = ImmediateContextMemory(**self._prepare_memory_data(data))
                elif mem_type == MemoryType.EPISODIC:
                    memory = EpisodicMemory(**self._prepare_memory_data(data))
                elif mem_type == MemoryType.SEMANTIC:
                    memory = SemanticMemory(**self._prepare_memory_data(data))
                elif mem_type == MemoryType.PROCEDURAL:
                    memory = ProceduralMemory(**self._prepare_memory_data(data))
                else:
                    memory = Memory(**self._prepare_memory_data(data))
                
                # Store in memory without re-persisting
                self.memories[memory.id] = memory
                
                # Update indices
                self.type_index[memory.type].append(memory.id)
                self.importance_index[memory.importance].append(memory.id)
                
                for tag in memory.tags:
                    if tag.name not in self.tag_index:
                        self.tag_index[tag.name] = []
                    self.tag_index[tag.name].append(memory.id)
                    
            except Exception as e:
                print(f"Error loading memory from {memory_file}: {e}")
    
    def _prepare_memory_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare raw data for memory object creation"""
        
        # Convert ISO strings back to datetime
        if 'created_at' in data:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and data['updated_at']:
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if 'accessed_at' in data and data['accessed_at']:
            data['accessed_at'] = datetime.fromisoformat(data['accessed_at'])
        
        # Convert tags
        if 'tags' in data:
            data['tags'] = [Tag(**tag) for tag in data['tags']]
        
        # Convert embeddings (simplified - would need numpy reconstruction)
        if 'embeddings' in data:
            data['embeddings'] = []  # Skip for now
        
        # Convert type
        if 'type' in data:
            data['type'] = MemoryType(data['type'])
        
        return data
    
    def _extract_patterns(self, memories: List[Memory]) -> List[Dict[str, Any]]:
        """Extract patterns from a collection of memories"""
        
        patterns = []
        
        # Simple pattern extraction based on tag frequency
        tag_counts = {}
        for memory in memories:
            for tag in memory.tags:
                if tag.name not in tag_counts:
                    tag_counts[tag.name] = 0
                tag_counts[tag.name] += 1
        
        # Create patterns for frequently occurring tags
        for tag_name, count in tag_counts.items():
            if count >= 3:  # Minimum threshold
                pattern_memories = [
                    m for m in memories
                    if any(t.name == tag_name for t in m.tags)
                ]
                
                patterns.append({
                    'description': f"Pattern: {tag_name} (occurred {count} times)",
                    'confidence': min(1.0, count / 10),
                    'importance': min(10, 5 + count // 2),
                    'memories': pattern_memories
                })
        
        return patterns
    
    async def _consolidate_memories(self):
        """Async task to consolidate memories periodically"""
        
        await asyncio.sleep(1)  # Prevent blocking
        
        # Consolidate old immediate memories
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        old_immediate = [
            m for m in self.memories.values()
            if m.type == MemoryType.IMMEDIATE
            and m.created_at < cutoff_time
        ]
        
        if len(old_immediate) >= 10:
            self.consolidate()


# Example usage
def example_usage():
    """Demonstrate cognitive memory system usage"""
    
    # Initialize system
    memory_system = CognitiveMemorySystem()
    
    # Store immediate context memory
    immediate_memory = ImmediateContextMemory(
        conversation_id="conv-123",
        session_id="session-456",
        user_id="user-789",
        content={
            "text": "User wants to implement a REST API",
            "intent": "feature_request",
            "context": "backend development"
        },
        importance=7,
        tags=[
            Tag(name="api", category="feature"),
            Tag(name="backend", category="domain")
        ]
    )
    immediate_id = memory_system.store(immediate_memory)
    print(f"Stored immediate memory: {immediate_id}")
    
    # Store episodic memory
    episodic_memory = EpisodicMemory(
        event_type="feature_implementation",
        content={
            "feature": "REST API",
            "duration_hours": 4,
            "files_modified": 12
        },
        timeline=[
            {
                "timestamp": datetime.utcnow().isoformat(),
                "action": "Created API routes",
                "outcome": "success"
            },
            {
                "timestamp": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
                "action": "Added authentication",
                "outcome": "success"
            }
        ],
        importance=8,
        tags=[
            Tag(name="api", category="feature"),
            Tag(name="authentication", category="security")
        ]
    )
    episodic_id = memory_system.store(episodic_memory)
    print(f"Stored episodic memory: {episodic_id}")
    
    # Store semantic memory
    semantic_memory = SemanticMemory(
        concept="REST API",
        definition="Representational State Transfer Application Programming Interface",
        relationships=[
            {
                "target_concept": "HTTP",
                "relationship_type": "uses",
                "strength": 1.0
            },
            {
                "target_concept": "JSON",
                "relationship_type": "uses",
                "strength": 0.9
            }
        ],
        properties={
            "stateless": True,
            "cacheable": True,
            "layered": True
        },
        examples=[
            {
                "code": "GET /api/users",
                "explanation": "Retrieve list of users"
            }
        ],
        importance=9,
        tags=[
            Tag(name="api", category="architecture"),
            Tag(name="rest", category="pattern")
        ]
    )
    semantic_id = memory_system.store(semantic_memory)
    print(f"Stored semantic memory: {semantic_id}")
    
    # Store procedural memory
    procedural_memory = ProceduralMemory(
        procedure_name="Deploy REST API",
        triggers=[
            {"type": "manual", "expression": "deploy command"}
        ],
        steps=[
            {
                "order": 1,
                "action": "Run tests",
                "command": "pytest tests/",
                "estimated_duration_ms": 30000
            },
            {
                "order": 2,
                "action": "Build Docker image",
                "command": "docker build -t api:latest .",
                "estimated_duration_ms": 60000
            },
            {
                "order": 3,
                "action": "Deploy to production",
                "command": "kubectl apply -f k8s/",
                "estimated_duration_ms": 120000
            }
        ],
        success_criteria=[
            "All tests pass",
            "Docker image built successfully",
            "Kubernetes pods running"
        ],
        importance=10,
        tags=[
            Tag(name="deployment", category="devops"),
            Tag(name="api", category="feature")
        ]
    )
    procedural_id = memory_system.store(procedural_memory)
    print(f"Stored procedural memory: {procedural_id}")
    
    # Search memories
    print("\n=== Searching for API-related memories ===")
    api_memories = memory_system.search(
        query="api",
        importance_min=7,
        tags=["api"]
    )
    
    for memory in api_memories:
        print(f"- {memory.type.value}: {memory.id} (importance: {memory.importance})")
    
    # Get statistics
    print("\n=== Memory System Statistics ===")
    stats = memory_system.get_statistics()
    print(json.dumps(stats, indent=2))
    
    # Consolidate memories
    print("\n=== Consolidating memories ===")
    consolidation_result = memory_system.consolidate(session_id="session-456")
    print(json.dumps(consolidation_result, indent=2))
    
    # Clean up expired memories
    expired_count = memory_system.cleanup_expired()
    print(f"\nCleaned up {expired_count} expired memories")


if __name__ == "__main__":
    example_usage()