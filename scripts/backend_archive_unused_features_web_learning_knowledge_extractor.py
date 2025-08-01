#!/usr/bin/env python3
"""
Knowledge Extractor for SutazAI V7 Self-Supervised Learning
Extracts structured knowledge from processed content using biological neural networks
"""

import os
import sys
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import re
from pathlib import Path

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_learning.content_processor import ProcessedContent, ContentType
from neuromorphic.enhanced_engine import EnhancedNeuromorphicEngine
from neuromorphic.advanced_biological_modeling import create_advanced_neural_network

logger = logging.getLogger(__name__)

class KnowledgeType(Enum):
    """Types of knowledge that can be extracted"""
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    CONCEPTUAL = "conceptual"
    RELATIONAL = "relational"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"

@dataclass
class KnowledgeUnit:
    """Individual unit of extracted knowledge"""
    id: str
    content: str
    knowledge_type: KnowledgeType
    confidence: float
    source_url: str
    context: str = ""
    entities: List[str] = field(default_factory=list)
    relations: List[Dict[str, str]] = field(default_factory=list)
    numerical_values: List[Dict[str, Any]] = field(default_factory=list)
    temporal_markers: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())
    validity_score: float = 0.0
    uniqueness_score: float = 0.0

@dataclass
class KnowledgeGraph:
    """Knowledge graph structure"""
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    concepts: Set[str] = field(default_factory=set)
    relations: Set[str] = field(default_factory=set)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_node(self, node_id: str, node_type: str, properties: Dict[str, Any]):
        """Add a node to the knowledge graph"""
        self.nodes[node_id] = {
            'type': node_type,
            'properties': properties,
            'created_at': datetime.now().isoformat()
        }
        
        if node_type == 'concept':
            self.concepts.add(node_id)
    
    def add_edge(self, source: str, target: str, relation: str, properties: Dict[str, Any] = None):
        """Add an edge to the knowledge graph"""
        self.edges.append({
            'source': source,
            'target': target,
            'relation': relation,
            'properties': properties or {},
            'created_at': datetime.now().isoformat()
        })
        
        self.relations.add(relation)

class FactualExtractor:
    """Extracts factual knowledge from content"""
    
    def __init__(self):
        self.fact_patterns = [
            r'(.+?)\s+is\s+(.+?)(?:\.|$)',
            r'(.+?)\s+was\s+(.+?)(?:\.|$)',
            r'(.+?)\s+has\s+(.+?)(?:\.|$)',
            r'(.+?)\s+contains\s+(.+?)(?:\.|$)',
            r'(.+?)\s+consists\s+of\s+(.+?)(?:\.|$)',
            r'(.+?)\s+measures\s+(.+?)(?:\.|$)',
            r'(.+?)\s+weighs\s+(.+?)(?:\.|$)',
            r'(.+?)\s+costs\s+(.+?)(?:\.|$)',
            r'(.+?)\s+equals\s+(.+?)(?:\.|$)',
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) 
                                 for pattern in self.fact_patterns]
    
    def extract_facts(self, content: ProcessedContent) -> List[KnowledgeUnit]:
        """Extract factual knowledge units"""
        facts = []
        
        if not content.text_content:
            return facts
        
        sentences = self._split_sentences(content.text_content)
        
        for sentence in sentences:
            for pattern in self.compiled_patterns:
                matches = pattern.findall(sentence)
                
                for match in matches:
                    if len(match) == 2:
                        subject, predicate = match
                        subject = subject.strip()
                        predicate = predicate.strip()
                        
                        if len(subject) > 2 and len(predicate) > 2:
                            fact_id = hashlib.md5(f"{subject}:{predicate}".encode()).hexdigest()[:12]
                            
                            facts.append(KnowledgeUnit(
                                id=fact_id,
                                content=f"{subject} {predicate}",
                                knowledge_type=KnowledgeType.FACTUAL,
                                confidence=0.7,
                                source_url=content.url,
                                context=sentence,
                                entities=[subject, predicate],
                                validity_score=self._calculate_fact_validity(sentence)
                            ))
        
        return facts
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _calculate_fact_validity(self, sentence: str) -> float:
        """Calculate validity score for a factual statement"""
        score = 0.5  # Base score
        
        # Increase score for numerical data
        if re.search(r'\d+', sentence):
            score += 0.2
        
        # Increase score for specific entities
        if any(char.isupper() for char in sentence):
            score += 0.1
        
        # Increase score for authoritative language
        authoritative_words = ['research', 'study', 'according to', 'data shows', 'evidence']
        if any(word in sentence.lower() for word in authoritative_words):
            score += 0.2
        
        return min(1.0, score)

class ConceptualExtractor:
    """Extracts conceptual knowledge from content"""
    
    def __init__(self):
        self.concept_indicators = [
            'concept', 'idea', 'theory', 'principle', 'approach', 'method',
            'technique', 'strategy', 'framework', 'model', 'paradigm'
        ]
        
        self.definition_patterns = [
            r'(.+?)\s+is\s+defined\s+as\s+(.+?)(?:\.|$)',
            r'(.+?)\s+refers\s+to\s+(.+?)(?:\.|$)',
            r'(.+?)\s+means\s+(.+?)(?:\.|$)',
            r'(.+?)\s+can\s+be\s+described\s+as\s+(.+?)(?:\.|$)',
            r'(.+?)\s+represents\s+(.+?)(?:\.|$)',
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) 
                                 for pattern in self.definition_patterns]
    
    def extract_concepts(self, content: ProcessedContent) -> List[KnowledgeUnit]:
        """Extract conceptual knowledge units"""
        concepts = []
        
        if not content.text_content:
            return concepts
        
        sentences = self._split_sentences(content.text_content)
        
        for sentence in sentences:
            # Look for definition patterns
            for pattern in self.compiled_patterns:
                matches = pattern.findall(sentence)
                
                for match in matches:
                    if len(match) == 2:
                        concept_name, definition = match
                        concept_name = concept_name.strip()
                        definition = definition.strip()
                        
                        if len(concept_name) > 2 and len(definition) > 10:
                            concept_id = hashlib.md5(f"concept:{concept_name}".encode()).hexdigest()[:12]
                            
                            concepts.append(KnowledgeUnit(
                                id=concept_id,
                                content=f"{concept_name}: {definition}",
                                knowledge_type=KnowledgeType.CONCEPTUAL,
                                confidence=0.8,
                                source_url=content.url,
                                context=sentence,
                                entities=[concept_name],
                                validity_score=0.7
                            ))
        
        return concepts
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

class RelationalExtractor:
    """Extracts relational knowledge from content"""
    
    def __init__(self):
        self.relation_patterns = [
            (r'(.+?)\s+causes\s+(.+?)(?:\.|$)', 'causes'),
            (r'(.+?)\s+leads\s+to\s+(.+?)(?:\.|$)', 'leads_to'),
            (r'(.+?)\s+results\s+in\s+(.+?)(?:\.|$)', 'results_in'),
            (r'(.+?)\s+depends\s+on\s+(.+?)(?:\.|$)', 'depends_on'),
            (r'(.+?)\s+requires\s+(.+?)(?:\.|$)', 'requires'),
            (r'(.+?)\s+includes\s+(.+?)(?:\.|$)', 'includes'),
            (r'(.+?)\s+contains\s+(.+?)(?:\.|$)', 'contains'),
            (r'(.+?)\s+is\s+part\s+of\s+(.+?)(?:\.|$)', 'part_of'),
            (r'(.+?)\s+belongs\s+to\s+(.+?)(?:\.|$)', 'belongs_to'),
            (r'(.+?)\s+is\s+related\s+to\s+(.+?)(?:\.|$)', 'related_to'),
        ]
        
        self.compiled_patterns = [(re.compile(pattern, re.IGNORECASE), relation) 
                                 for pattern, relation in self.relation_patterns]
    
    def extract_relations(self, content: ProcessedContent) -> List[KnowledgeUnit]:
        """Extract relational knowledge units"""
        relations = []
        
        if not content.text_content:
            return relations
        
        sentences = self._split_sentences(content.text_content)
        
        for sentence in sentences:
            for pattern, relation_type in self.compiled_patterns:
                matches = pattern.findall(sentence)
                
                for match in matches:
                    if len(match) == 2:
                        source, target = match
                        source = source.strip()
                        target = target.strip()
                        
                        if len(source) > 2 and len(target) > 2:
                            relation_id = hashlib.md5(f"{source}:{relation_type}:{target}".encode()).hexdigest()[:12]
                            
                            relations.append(KnowledgeUnit(
                                id=relation_id,
                                content=f"{source} {relation_type} {target}",
                                knowledge_type=KnowledgeType.RELATIONAL,
                                confidence=0.6,
                                source_url=content.url,
                                context=sentence,
                                entities=[source, target],
                                relations=[{
                                    'type': relation_type,
                                    'source': source,
                                    'target': target
                                }],
                                validity_score=0.6
                            ))
        
        return relations
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

class NumericalExtractor:
    """Extracts numerical knowledge from content"""
    
    def __init__(self):
        self.numerical_patterns = [
            (r'(\d+(?:\.\d+)?)\s*(percent|%)', 'percentage'),
            (r'(\d+(?:\.\d+)?)\s*(million|billion|trillion)', 'large_number'),
            (r'\$(\d+(?:\.\d+)?)', 'currency'),
            (r'(\d+(?:\.\d+)?)\s*(kg|km|m|cm|mm|g|lb|ft|in|mph|kmh)', 'measurement'),
            (r'(\d{4})', 'year'),
            (r'(\d+(?:\.\d+)?)', 'number')
        ]
        
        self.compiled_patterns = [(re.compile(pattern, re.IGNORECASE), data_type) 
                                 for pattern, data_type in self.numerical_patterns]
    
    def extract_numerical_data(self, content: ProcessedContent) -> List[KnowledgeUnit]:
        """Extract numerical knowledge units"""
        numerical_units = []
        
        if not content.text_content:
            return numerical_units
        
        sentences = self._split_sentences(content.text_content)
        
        for sentence in sentences:
            for pattern, data_type in self.compiled_patterns:
                matches = pattern.finditer(sentence)
                
                for match in matches:
                    value = match.group(1)
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(sentence), match.end() + 50)
                    context = sentence[context_start:context_end]
                    
                    numerical_id = hashlib.md5(f"{value}:{data_type}:{context}".encode()).hexdigest()[:12]
                    
                    numerical_units.append(KnowledgeUnit(
                        id=numerical_id,
                        content=f"{value} ({data_type})",
                        knowledge_type=KnowledgeType.NUMERICAL,
                        confidence=0.9,
                        source_url=content.url,
                        context=context,
                        numerical_values=[{
                            'value': value,
                            'type': data_type,
                            'context': context
                        }],
                        validity_score=0.8
                    ))
        
        return numerical_units
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

class KnowledgeExtractor:
    """
    Main knowledge extractor that integrates with biological neural networks
    """
    
    def __init__(self):
        self.factual_extractor = FactualExtractor()
        self.conceptual_extractor = ConceptualExtractor()
        self.relational_extractor = RelationalExtractor()
        self.numerical_extractor = NumericalExtractor()
        
        # Neural network for knowledge validation
        self.neural_engine = None
        self.neural_config = {
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
            'learning_enabled': True
        }
        
        # Knowledge storage
        self.knowledge_units: List[KnowledgeUnit] = []
        self.knowledge_graph = KnowledgeGraph()
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'facts_extracted': 0,
            'concepts_extracted': 0,
            'relations_extracted': 0,
            'numerical_extracted': 0,
            'start_time': datetime.now()
        }
        
        logger.info("KnowledgeExtractor initialized")
    
    async def initialize_neural_engine(self):
        """Initialize the neural engine for knowledge validation"""
        if TORCH_AVAILABLE:
            try:
                self.neural_engine = EnhancedNeuromorphicEngine(self.neural_config)
                logger.info("Neural engine initialized for knowledge validation")
            except Exception as e:
                logger.warning(f"Could not initialize neural engine: {e}")
                self.neural_engine = None
        else:
            logger.warning("PyTorch not available. Neural validation disabled.")
    
    async def extract_knowledge(self, content: ProcessedContent) -> Dict[str, List[KnowledgeUnit]]:
        """
        Extract all types of knowledge from processed content
        
        Args:
            content: ProcessedContent object to extract knowledge from
            
        Returns:
            Dictionary containing extracted knowledge units by type
        """
        self.stats['total_processed'] += 1
        
        try:
            # Extract different types of knowledge
            facts = self.factual_extractor.extract_facts(content)
            concepts = self.conceptual_extractor.extract_concepts(content)
            relations = self.relational_extractor.extract_relations(content)
            numerical = self.numerical_extractor.extract_numerical_data(content)
            
            # Update statistics
            self.stats['facts_extracted'] += len(facts)
            self.stats['concepts_extracted'] += len(concepts)
            self.stats['relations_extracted'] += len(relations)
            self.stats['numerical_extracted'] += len(numerical)
            
            # Validate knowledge using neural network
            if self.neural_engine:
                await self._validate_knowledge_neurally(facts + concepts + relations + numerical)
            
            # Add to knowledge graph
            self._add_to_knowledge_graph(facts + concepts + relations + numerical)
            
            # Store knowledge units
            all_units = facts + concepts + relations + numerical
            self.knowledge_units.extend(all_units)
            
            result = {
                'facts': facts,
                'concepts': concepts,
                'relations': relations,
                'numerical': numerical,
                'total_units': len(all_units)
            }
            
            logger.info(f"Extracted {len(all_units)} knowledge units from {content.url}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting knowledge from {content.url}: {e}")
            return {
                'facts': [],
                'concepts': [],
                'relations': [],
                'numerical': [],
                'total_units': 0
            }
    
    async def _validate_knowledge_neurally(self, knowledge_units: List[KnowledgeUnit]):
        """Validate knowledge units using biological neural network"""
        if not self.neural_engine:
            return
        
        try:
            for unit in knowledge_units:
                # Convert knowledge unit to neural input
                if TORCH_AVAILABLE:
                    import torch
                    
                    # Simple encoding: text length, confidence, and type
                    input_vector = torch.tensor([
                        len(unit.content) / 100.0,  # Normalized length
                        unit.confidence,
                        len(unit.entities) / 10.0,  # Normalized entity count
                        unit.validity_score
                    ]).unsqueeze(0).unsqueeze(-1)
                    
                    # Process through neural network
                    results = await self.neural_engine.process_input(input_vector)
                    
                    # Extract validation score from neural output
                    if 'output' in results:
                        neural_score = float(results['output'].mean())
                        
                        # Update confidence based on neural validation
                        unit.confidence = (unit.confidence + neural_score) / 2.0
                        unit.validity_score = neural_score
                        
        except Exception as e:
            logger.error(f"Error in neural validation: {e}")
    
    def _add_to_knowledge_graph(self, knowledge_units: List[KnowledgeUnit]):
        """Add knowledge units to the knowledge graph"""
        for unit in knowledge_units:
            # Add main node
            self.knowledge_graph.add_node(
                unit.id,
                unit.knowledge_type.value,
                {
                    'content': unit.content,
                    'confidence': unit.confidence,
                    'source': unit.source_url,
                    'entities': unit.entities
                }
            )
            
            # Add entity nodes and relationships
            for entity in unit.entities:
                entity_id = hashlib.md5(f"entity:{entity}".encode()).hexdigest()[:12]
                self.knowledge_graph.add_node(
                    entity_id,
                    'entity',
                    {'name': entity}
                )
                
                # Connect knowledge unit to entity
                self.knowledge_graph.add_edge(
                    unit.id,
                    entity_id,
                    'mentions',
                    {'confidence': unit.confidence}
                )
            
            # Add relation edges
            for relation in unit.relations:
                if 'source' in relation and 'target' in relation:
                    source_id = hashlib.md5(f"entity:{relation['source']}".encode()).hexdigest()[:12]
                    target_id = hashlib.md5(f"entity:{relation['target']}".encode()).hexdigest()[:12]
                    
                    self.knowledge_graph.add_edge(
                        source_id,
                        target_id,
                        relation['type'],
                        {'confidence': unit.confidence}
                    )
    
    def get_knowledge_by_type(self, knowledge_type: KnowledgeType) -> List[KnowledgeUnit]:
        """Get all knowledge units of a specific type"""
        return [unit for unit in self.knowledge_units if unit.knowledge_type == knowledge_type]
    
    def search_knowledge(self, query: str, max_results: int = 10) -> List[KnowledgeUnit]:
        """Search knowledge units by content"""
        query_lower = query.lower()
        results = []
        
        for unit in self.knowledge_units:
            if query_lower in unit.content.lower():
                results.append(unit)
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        return results[:max_results]
    
    def get_related_knowledge(self, entity: str, max_results: int = 10) -> List[KnowledgeUnit]:
        """Get knowledge units related to a specific entity"""
        entity_lower = entity.lower()
        results = []
        
        for unit in self.knowledge_units:
            if any(entity_lower in e.lower() for e in unit.entities):
                results.append(unit)
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        return results[:max_results]
    
    def export_knowledge_graph(self, output_file: str):
        """Export knowledge graph to file"""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            graph_data = {
                'nodes': self.knowledge_graph.nodes,
                'edges': self.knowledge_graph.edges,
                'concepts': list(self.knowledge_graph.concepts),
                'relations': list(self.knowledge_graph.relations),
                'statistics': self.get_statistics(),
                'exported_at': datetime.now().isoformat()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Knowledge graph exported to {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting knowledge graph: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge extraction statistics"""
        runtime = datetime.now() - self.stats['start_time']
        
        return {
            **self.stats,
            'total_knowledge_units': len(self.knowledge_units),
            'knowledge_graph_nodes': len(self.knowledge_graph.nodes),
            'knowledge_graph_edges': len(self.knowledge_graph.edges),
            'unique_concepts': len(self.knowledge_graph.concepts),
            'unique_relations': len(self.knowledge_graph.relations),
            'runtime_seconds': runtime.total_seconds(),
            'extraction_rate': len(self.knowledge_units) / max(1, runtime.total_seconds())
        }

# Factory function
def create_knowledge_extractor() -> KnowledgeExtractor:
    """Create a configured KnowledgeExtractor instance"""
    return KnowledgeExtractor()

# Example usage
if __name__ == "__main__":
    async def main():
        extractor = create_knowledge_extractor()
        await extractor.initialize_neural_engine()
        
        # Example processed content
        from web_learning.content_processor import ProcessedContent, ContentType
        
        content = ProcessedContent(
            url="https://example.com/ml-article",
            content_type=ContentType.HTML,
            title="Machine Learning Advances",
            text_content="Machine learning is a subset of artificial intelligence. "
                        "Neural networks consist of interconnected nodes. "
                        "Deep learning requires large datasets. "
                        "The accuracy improved by 15% over previous methods. "
                        "Research shows that convolutional neural networks excel at image recognition.",
            entities=[{'text': 'Machine Learning', 'type': 'TECHNOLOGY'}],
            keywords=['machine learning', 'neural networks', 'deep learning', 'accuracy']
        )
        
        # Extract knowledge
        results = await extractor.extract_knowledge(content)
        
        print(f"Extracted {results['total_units']} knowledge units:")
        print(f"  Facts: {len(results['facts'])}")
        print(f"  Concepts: {len(results['concepts'])}")
        print(f"  Relations: {len(results['relations'])}")
        print(f"  Numerical: {len(results['numerical'])}")
        
        # Search functionality
        search_results = extractor.search_knowledge("neural networks")
        print(f"\nSearch results for 'neural networks': {len(search_results)} units")
        
        # Statistics
        stats = extractor.get_statistics()
        print(f"\nExtraction rate: {stats['extraction_rate']:.2f} units/second")
        
        # Export knowledge graph
        extractor.export_knowledge_graph("/tmp/knowledge_graph.json")
    
    asyncio.run(main())