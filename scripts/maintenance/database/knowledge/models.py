#!/usr/bin/env python3
"""
Knowledge management data models
Extracted from knowledge_manager.py for modularity
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pydantic import BaseModel
import uuid

class EntityType(Enum):
    CONCEPT = "concept"
    PERSON = "person"
    PLACE = "place"
    ORGANIZATION = "organization"
    TECHNOLOGY = "technology"
    PROCESS = "process"
    EVENT = "event"

class RelationType(Enum):
    RELATES_TO = "relates_to"
    PART_OF = "part_of"
    CAUSES = "causes"
    INFLUENCES = "influences"
    IMPLEMENTS = "implements"
    USES = "uses"
    DEPENDS_ON = "depends_on"

@dataclass
class Entity:
    """Knowledge graph entity"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    entity_type: EntityType = EntityType.CONCEPT
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    source: str = "manual"

@dataclass
class Relationship:
    """Knowledge graph relationship"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relation_type: RelationType = RelationType.RELATES_TO
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class KnowledgeQuery:
    """Query structure for knowledge retrieval"""
    query_text: str
    entity_types: Optional[List[EntityType]] = None
    relation_types: Optional[List[RelationType]] = None
    limit: int = 10
    min_confidence: float = 0.5
    include_embeddings: bool = False

class EntityRequest(BaseModel):
    """API request model for entity operations"""
    name: str
    entity_type: str
    description: Optional[str] = ""
    properties: Optional[Dict[str, Any]] = {}

class RelationshipRequest(BaseModel):
    """API request model for relationship operations"""
    source_id: str
    target_id: str
    relation_type: str
    properties: Optional[Dict[str, Any]] = {}
    weight: Optional[float] = 1.0

class SearchRequest(BaseModel):
    """API request model for search operations"""
    query: str
    entity_types: Optional[List[str]] = []
    limit: Optional[int] = 10
    min_confidence: Optional[float] = 0.5

class KnowledgeStats(BaseModel):
    """Knowledge graph statistics"""
    total_entities: int
    total_relationships: int
    entity_type_counts: Dict[str, int]
    relation_type_counts: Dict[str, int]
    last_updated: datetime