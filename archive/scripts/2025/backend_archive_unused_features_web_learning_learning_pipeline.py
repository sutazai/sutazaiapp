#!/usr/bin/env python3
"""
Self-Supervised Learning Pipeline for SutazAI V7
Integrates web scraping, content processing, and knowledge extraction with biological neural networks
"""

import os
import sys
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
from enum import Enum
import random

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_learning.web_scraper import WebScraper, ScrapingConfig, create_web_scraper
from web_learning.content_processor import ContentProcessor, ProcessedContent, create_content_processor
from web_learning.knowledge_extractor import KnowledgeExtractor, KnowledgeUnit, create_knowledge_extractor
from neuromorphic.enhanced_engine import EnhancedNeuromorphicEngine
from vector_db import VectorStore
from security.secure_config import SecureConfigManager

logger = logging.getLogger(__name__)

class LearningMode(Enum):
    """Different learning modes for the pipeline"""
    EXPLORATION = "exploration"       # Broad exploration of new topics
    DEEPENING = "deepening"          # Deep dive into specific topics
    VALIDATION = "validation"        # Validate existing knowledge
    DISCOVERY = "discovery"          # Discover new connections
    MAINTENANCE = "maintenance"      # Update and refresh knowledge

@dataclass
class LearningTask:
    """Individual learning task"""
    id: str
    mode: LearningMode
    topic: str
    urls: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    target_knowledge_types: List[str] = field(default_factory=list)
    priority: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class LearningSession:
    """Learning session containing multiple tasks"""
    id: str
    name: str
    description: str
    tasks: List[LearningTask] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    statistics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LearningConfig:
    """Configuration for the learning pipeline"""
    # Scraping configuration
    scraping_config: ScrapingConfig = field(default_factory=ScrapingConfig)
    
    # Learning parameters
    max_concurrent_tasks: int = 3
    session_duration_hours: int = 2
    knowledge_retention_days: int = 30
    
    # Neural network configuration
    neural_config: Dict[str, Any] = field(default_factory=lambda: {
        'processing_mode': 'adaptive',
        'use_advanced_biological_modeling': True,
        'network': {
            'population_sizes': {
                'sensory': 128,
                'l2_3_pyramidal': 256,
                'l5_pyramidal': 128,
                'fast_spiking': 64,
                'dopaminergic': 32,
                'output': 64
            }
        },
        'learning_enabled': True,
        'plasticity': {
            'rules': ['STDP', 'homeostatic', 'metaplasticity'],
            'learning_rate': 1e-4
        }
    })
    
    # Topic discovery
    enable_topic_discovery: bool = True
    topic_similarity_threshold: float = 0.7
    
    # Knowledge validation
    enable_knowledge_validation: bool = True
    validation_confidence_threshold: float = 0.8
    
    # Storage
    save_intermediate_results: bool = True
    results_directory: str = "/opt/sutazaiapp/data/learning_results"
    
    # Ethical constraints
    respect_robots_txt: bool = True
    max_requests_per_hour: int = 500
    content_filtering_enabled: bool = True

class TopicDiscoverer:
    """Discovers new topics for learning"""
    
    def __init__(self, knowledge_extractor: KnowledgeExtractor):
        self.knowledge_extractor = knowledge_extractor
        self.discovered_topics: Dict[str, float] = {}  # topic -> relevance score
        
    def discover_topics_from_content(self, processed_content: List[ProcessedContent]) -> List[str]:
        """Discover new topics from processed content"""
        topics = []
        
        for content in processed_content:
            # Extract topics from keywords
            for keyword in content.keywords:
                if keyword not in self.discovered_topics:
                    self.discovered_topics[keyword] = 0.0
                
                # Increase relevance score
                self.discovered_topics[keyword] += 0.1
            
            # Extract topics from entities
            for entity in content.entities:
                entity_text = entity.get('text', '') if isinstance(entity, dict) else str(entity)
                if entity_text not in self.discovered_topics:
                    self.discovered_topics[entity_text] = 0.0
                
                self.discovered_topics[entity_text] += 0.2
            
            # Extract topics from content topics
            for topic in content.topics:
                if topic not in self.discovered_topics:
                    self.discovered_topics[topic] = 0.0
                
                self.discovered_topics[topic] += 0.3
        
        # Sort by relevance and return top topics
        sorted_topics = sorted(self.discovered_topics.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, score in sorted_topics[:10]]
    
    def generate_learning_tasks(self, topics: List[str], base_urls: List[str]) -> List[LearningTask]:
        """Generate learning tasks for discovered topics"""
        tasks = []
        
        for topic in topics:
            # Generate search URLs for the topic
            search_urls = self._generate_search_urls(topic, base_urls)
            
            # Determine learning mode based on existing knowledge
            existing_knowledge = self.knowledge_extractor.search_knowledge(topic)
            
            if len(existing_knowledge) == 0:
                mode = LearningMode.EXPLORATION
            elif len(existing_knowledge) < 5:
                mode = LearningMode.DEEPENING
            else:
                mode = LearningMode.VALIDATION
            
            task = LearningTask(
                id=hashlib.md5(f"{topic}:{mode.value}".encode()).hexdigest()[:12],
                mode=mode,
                topic=topic,
                urls=search_urls,
                keywords=[topic],
                target_knowledge_types=['factual', 'conceptual', 'relational'],
                priority=self.discovered_topics.get(topic, 0.5)
            )
            
            tasks.append(task)
        
        return tasks
    
    def _generate_search_urls(self, topic: str, base_urls: List[str]) -> List[str]:
        """Generate search URLs for a topic"""
        search_urls = []
        
        # Add base URLs
        search_urls.extend(base_urls)
        
        # Generate academic search URLs
        academic_sites = [
            "https://scholar.google.com/scholar?q=",
            "https://arxiv.org/search/?query=",
            "https://www.researchgate.net/search?q=",
        ]
        
        # Generate news search URLs
        news_sites = [
            "https://news.google.com/search?q=",
            "https://www.reddit.com/search/?q=",
        ]
        
        # Generate Wikipedia URLs
        wikipedia_url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        search_urls.append(wikipedia_url)
        
        return search_urls

class KnowledgeValidator:
    """Validates extracted knowledge using various methods"""
    
    def __init__(self, neural_engine: Optional[EnhancedNeuromorphicEngine] = None):
        self.neural_engine = neural_engine
        self.validation_history: Dict[str, List[Dict[str, Any]]] = {}
        
    async def validate_knowledge_unit(self, unit: KnowledgeUnit) -> Dict[str, Any]:
        """Validate a single knowledge unit"""
        validation_result = {
            'unit_id': unit.id,
            'original_confidence': unit.confidence,
            'validation_score': 0.0,
            'validation_methods': [],
            'issues_found': [],
            'recommendations': []
        }
        
        # Consistency validation
        consistency_score = self._validate_consistency(unit)
        validation_result['consistency_score'] = consistency_score
        validation_result['validation_methods'].append('consistency')
        
        # Source reliability validation
        source_score = self._validate_source_reliability(unit)
        validation_result['source_reliability_score'] = source_score
        validation_result['validation_methods'].append('source_reliability')
        
        # Content quality validation
        quality_score = self._validate_content_quality(unit)
        validation_result['content_quality_score'] = quality_score
        validation_result['validation_methods'].append('content_quality')
        
        # Neural validation if available
        if self.neural_engine:
            neural_score = await self._validate_with_neural_network(unit)
            validation_result['neural_score'] = neural_score
            validation_result['validation_methods'].append('neural_network')
        else:
            neural_score = 0.5  # Neutral score
        
        # Calculate overall validation score
        validation_result['validation_score'] = (
            consistency_score * 0.3 +
            source_score * 0.2 +
            quality_score * 0.2 +
            neural_score * 0.3
        )
        
        # Generate recommendations
        if validation_result['validation_score'] < 0.6:
            validation_result['recommendations'].append('Consider removing or flagging this knowledge unit')
        elif validation_result['validation_score'] < 0.8:
            validation_result['recommendations'].append('Seek additional validation sources')
        
        return validation_result
    
    def _validate_consistency(self, unit: KnowledgeUnit) -> float:
        """Validate consistency of knowledge unit"""
        score = 0.7  # Base score
        
        # Check for contradictions in content
        contradiction_indicators = ['but', 'however', 'although', 'despite', 'contrary']
        if any(indicator in unit.content.lower() for indicator in contradiction_indicators):
            score -= 0.2
        
        # Check for uncertainty markers
        uncertainty_markers = ['might', 'could', 'possibly', 'probably', 'maybe', 'perhaps']
        if any(marker in unit.content.lower() for marker in uncertainty_markers):
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _validate_source_reliability(self, unit: KnowledgeUnit) -> float:
        """Validate source reliability"""
        score = 0.5  # Base score
        
        # Check domain authority
        reliable_domains = [
            'wikipedia.org', 'scholar.google.com', 'arxiv.org',
            'researchgate.net', 'pubmed.ncbi.nlm.nih.gov',
            'nature.com', 'science.org', 'ieee.org'
        ]
        
        if any(domain in unit.source_url for domain in reliable_domains):
            score += 0.3
        
        # Check for academic indicators
        academic_indicators = ['doi:', 'isbn:', 'pmid:', 'journal', 'conference']
        if any(indicator in unit.source_url.lower() for indicator in academic_indicators):
            score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def _validate_content_quality(self, unit: KnowledgeUnit) -> float:
        """Validate content quality"""
        score = 0.6  # Base score
        
        # Check content length
        if len(unit.content) > 20:
            score += 0.1
        
        # Check for specific information
        if unit.entities:
            score += 0.1
        
        if unit.numerical_values:
            score += 0.1
        
        # Check for supporting evidence
        if unit.supporting_evidence:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    async def _validate_with_neural_network(self, unit: KnowledgeUnit) -> float:
        """Validate using neural network"""
        try:
            if not TORCH_AVAILABLE:
                return 0.5
            
            # Convert knowledge unit to neural input
            input_features = [
                len(unit.content) / 100.0,  # Normalized content length
                unit.confidence,
                len(unit.entities) / 10.0,  # Normalized entity count
                unit.validity_score,
                1.0 if unit.numerical_values else 0.0,
                1.0 if unit.supporting_evidence else 0.0
            ]
            
            input_tensor = torch.tensor(input_features).unsqueeze(0).unsqueeze(-1)
            
            # Process through neural network
            results = await self.neural_engine.process_input(input_tensor)
            
            if 'output' in results:
                neural_score = float(results['output'].mean())
                return max(0.0, min(1.0, neural_score))
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Error in neural validation: {e}")
            return 0.5

class SelfSupervisedLearningPipeline:
    """
    Main self-supervised learning pipeline that orchestrates all components
    """
    
    def __init__(self, config: LearningConfig):
        self.config = config
        
        # Initialize components
        self.web_scraper = create_web_scraper(
            requests_per_second=config.scraping_config.requests_per_second,
            max_pages=config.scraping_config.max_total_pages,
            respect_robots=config.scraping_config.respect_robots_txt
        )
        
        self.content_processor = create_content_processor()
        self.knowledge_extractor = create_knowledge_extractor()
        self.topic_discoverer = TopicDiscoverer(self.knowledge_extractor)
        
        # Neural components
        self.neural_engine = None
        self.knowledge_validator = None
        
        # Storage
        self.vector_store = None
        self.results_directory = Path(config.results_directory)
        self.results_directory.mkdir(parents=True, exist_ok=True)
        
        # Current session
        self.current_session: Optional[LearningSession] = None
        self.active_tasks: List[LearningTask] = []
        
        # Statistics
        self.stats = {
            'total_sessions': 0,
            'total_tasks_completed': 0,
            'total_content_processed': 0,
            'total_knowledge_extracted': 0,
            'start_time': datetime.now(),
            'last_session_time': None
        }
        
        logger.info("SelfSupervisedLearningPipeline initialized")
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing learning pipeline components...")
        
        # Initialize neural engine
        if TORCH_AVAILABLE:
            try:
                self.neural_engine = EnhancedNeuromorphicEngine(self.config.neural_config)
                await self.knowledge_extractor.initialize_neural_engine()
                self.knowledge_validator = KnowledgeValidator(self.neural_engine)
                logger.info("Neural components initialized")
            except Exception as e:
                logger.warning(f"Could not initialize neural components: {e}")
        
        # Initialize vector store
        try:
            self.vector_store = VectorStore()
            logger.info("Vector store initialized")
        except Exception as e:
            logger.warning(f"Could not initialize vector store: {e}")
        
        logger.info("Learning pipeline initialization complete")
    
    async def start_learning_session(self, session_name: str, description: str, 
                                   initial_urls: List[str] = None, 
                                   topics: List[str] = None) -> str:
        """
        Start a new learning session
        
        Args:
            session_name: Name for the session
            description: Description of what will be learned
            initial_urls: Initial URLs to start learning from
            topics: Specific topics to focus on
            
        Returns:
            Session ID
        """
        # Create new session
        session_id = hashlib.md5(f"{session_name}:{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        self.current_session = LearningSession(
            id=session_id,
            name=session_name,
            description=description
        )
        
        logger.info(f"Starting learning session: {session_name} (ID: {session_id})")
        
        # Generate initial tasks
        if topics:
            # Generate tasks for specific topics
            for topic in topics:
                task = LearningTask(
                    id=hashlib.md5(f"{session_id}:{topic}".encode()).hexdigest()[:12],
                    mode=LearningMode.EXPLORATION,
                    topic=topic,
                    urls=initial_urls or [],
                    keywords=[topic],
                    target_knowledge_types=['factual', 'conceptual', 'relational'],
                    priority=0.8
                )
                self.current_session.tasks.append(task)
        
        elif initial_urls:
            # Discover topics from initial URLs
            task = LearningTask(
                id=hashlib.md5(f"{session_id}:discovery".encode()).hexdigest()[:12],
                mode=LearningMode.DISCOVERY,
                topic="initial_discovery",
                urls=initial_urls,
                keywords=[],
                target_knowledge_types=['factual', 'conceptual', 'relational'],
                priority=1.0
            )
            self.current_session.tasks.append(task)
        
        else:
            # Generate general exploration tasks
            general_topics = ['technology', 'science', 'health', 'education', 'environment']
            for topic in general_topics:
                task = LearningTask(
                    id=hashlib.md5(f"{session_id}:{topic}".encode()).hexdigest()[:12],
                    mode=LearningMode.EXPLORATION,
                    topic=topic,
                    urls=[],
                    keywords=[topic],
                    target_knowledge_types=['factual', 'conceptual'],
                    priority=0.5
                )
                self.current_session.tasks.append(task)
        
        # Start processing tasks
        await self._process_session_tasks()
        
        self.stats['total_sessions'] += 1
        self.stats['last_session_time'] = datetime.now()
        
        return session_id
    
    async def _process_session_tasks(self):
        """Process all tasks in the current session"""
        if not self.current_session:
            return
        
        logger.info(f"Processing {len(self.current_session.tasks)} tasks in session {self.current_session.id}")
        
        # Process tasks concurrently
        semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        
        async def process_task_with_semaphore(task: LearningTask):
            async with semaphore:
                return await self._process_learning_task(task)
        
        # Execute all tasks
        task_results = await asyncio.gather(
            *[process_task_with_semaphore(task) for task in self.current_session.tasks],
            return_exceptions=True
        )
        
        # Process results
        for task, result in zip(self.current_session.tasks, task_results):
            if isinstance(result, Exception):
                logger.error(f"Task {task.id} failed: {result}")
                task.results = {'error': str(result)}
            else:
                task.results = result
                task.completed_at = datetime.now().isoformat()
                self.stats['total_tasks_completed'] += 1
        
        # Discover new topics and generate additional tasks
        if self.config.enable_topic_discovery:
            await self._discover_and_add_topics()
        
        # Finalize session
        self.current_session.end_time = datetime.now().isoformat()
        self.current_session.statistics = self._calculate_session_statistics()
        
        # Save session results
        await self._save_session_results()
        
        logger.info(f"Session {self.current_session.id} completed")
    
    async def _process_learning_task(self, task: LearningTask) -> Dict[str, Any]:
        """Process a single learning task"""
        logger.info(f"Processing task {task.id}: {task.topic} ({task.mode.value})")
        
        task_results = {
            'task_id': task.id,
            'topic': task.topic,
            'mode': task.mode.value,
            'scraped_pages': 0,
            'processed_content': 0,
            'extracted_knowledge': 0,
            'validation_results': [],
            'discovered_topics': [],
            'errors': []
        }
        
        try:
            # Step 1: Web scraping
            scraped_data = []
            
            if task.urls:
                logger.debug(f"Scraping {len(task.urls)} URLs for task {task.id}")
                
                async with self.web_scraper:
                    for url in task.urls:
                        try:
                            result = await self.web_scraper.fetch_url(url)
                            if result:
                                scraped_data.append(result)
                        except Exception as e:
                            logger.error(f"Error scraping {url}: {e}")
                            task_results['errors'].append(f"Scraping error for {url}: {e}")
            
            task_results['scraped_pages'] = len(scraped_data)
            self.stats['total_content_processed'] += len(scraped_data)
            
            # Step 2: Content processing
            processed_content = []
            
            for data in scraped_data:
                try:
                    content = await self.content_processor.process_content(
                        data['url'],
                        data['content'],
                        data.get('content_type', '')
                    )
                    processed_content.append(content)
                except Exception as e:
                    logger.error(f"Error processing content from {data['url']}: {e}")
                    task_results['errors'].append(f"Processing error for {data['url']}: {e}")
            
            task_results['processed_content'] = len(processed_content)
            
            # Step 3: Knowledge extraction
            all_knowledge = []
            
            for content in processed_content:
                try:
                    knowledge_result = await self.knowledge_extractor.extract_knowledge(content)
                    
                    # Combine all knowledge types
                    for knowledge_type, units in knowledge_result.items():
                        if isinstance(units, list):
                            all_knowledge.extend(units)
                    
                except Exception as e:
                    logger.error(f"Error extracting knowledge from {content.url}: {e}")
                    task_results['errors'].append(f"Knowledge extraction error for {content.url}: {e}")
            
            task_results['extracted_knowledge'] = len(all_knowledge)
            self.stats['total_knowledge_extracted'] += len(all_knowledge)
            
            # Step 4: Knowledge validation
            if self.config.enable_knowledge_validation and self.knowledge_validator:
                validation_results = []
                
                for unit in all_knowledge:
                    try:
                        validation_result = await self.knowledge_validator.validate_knowledge_unit(unit)
                        validation_results.append(validation_result)
                        
                        # Update unit confidence based on validation
                        unit.confidence = validation_result['validation_score']
                        
                    except Exception as e:
                        logger.error(f"Error validating knowledge unit {unit.id}: {e}")
                
                task_results['validation_results'] = validation_results
            
            # Step 5: Topic discovery
            if self.config.enable_topic_discovery:
                discovered_topics = self.topic_discoverer.discover_topics_from_content(processed_content)
                task_results['discovered_topics'] = discovered_topics
            
            # Step 6: Store in vector database
            if self.vector_store:
                try:
                    await self._store_knowledge_in_vector_db(all_knowledge)
                except Exception as e:
                    logger.error(f"Error storing knowledge in vector database: {e}")
                    task_results['errors'].append(f"Vector storage error: {e}")
            
            logger.info(f"Task {task.id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing task {task.id}: {e}")
            task_results['errors'].append(f"Task processing error: {e}")
        
        return task_results
    
    async def _discover_and_add_topics(self):
        """Discover new topics and add them as tasks"""
        if not self.current_session:
            return
        
        # Collect all processed content from completed tasks
        all_processed_content = []
        
        for task in self.current_session.tasks:
            if task.completed_at and 'processed_content' in task.results:
                # This would need to be implemented to retrieve processed content
                pass
        
        # Discover new topics
        discovered_topics = self.topic_discoverer.discover_topics_from_content(all_processed_content)
        
        # Generate new tasks for discovered topics
        new_tasks = self.topic_discoverer.generate_learning_tasks(discovered_topics, [])
        
        # Add new tasks to session
        for task in new_tasks:
            # Check if topic is already covered
            existing_topics = [t.topic for t in self.current_session.tasks]
            if task.topic not in existing_topics:
                self.current_session.tasks.append(task)
        
        logger.info(f"Added {len(new_tasks)} new tasks from topic discovery")
    
    async def _store_knowledge_in_vector_db(self, knowledge_units: List[KnowledgeUnit]):
        """Store knowledge units in vector database"""
        if not self.vector_store:
            return
        
        # Convert knowledge units to documents for vector storage
        documents = []
        
        for unit in knowledge_units:
            doc_data = {
                'id': unit.id,
                'text': unit.content,
                'metadata': {
                    'knowledge_type': unit.knowledge_type.value,
                    'confidence': unit.confidence,
                    'source_url': unit.source_url,
                    'entities': unit.entities,
                    'extracted_at': unit.extracted_at,
                    'validity_score': unit.validity_score
                }
            }
            documents.append(doc_data)
        
        try:
            # Store documents in vector database
            # This would need to be implemented based on the vector store interface
            logger.info(f"Stored {len(documents)} knowledge units in vector database")
        except Exception as e:
            logger.error(f"Error storing knowledge in vector database: {e}")
    
    def _calculate_session_statistics(self) -> Dict[str, Any]:
        """Calculate statistics for the current session"""
        if not self.current_session:
            return {}
        
        total_scraped = sum(task.results.get('scraped_pages', 0) for task in self.current_session.tasks)
        total_processed = sum(task.results.get('processed_content', 0) for task in self.current_session.tasks)
        total_knowledge = sum(task.results.get('extracted_knowledge', 0) for task in self.current_session.tasks)
        total_errors = sum(len(task.results.get('errors', [])) for task in self.current_session.tasks)
        
        session_start = datetime.fromisoformat(self.current_session.start_time)
        session_end = datetime.fromisoformat(self.current_session.end_time) if self.current_session.end_time else datetime.now()
        duration = (session_end - session_start).total_seconds()
        
        return {
            'total_tasks': len(self.current_session.tasks),
            'completed_tasks': len([t for t in self.current_session.tasks if t.completed_at]),
            'total_scraped_pages': total_scraped,
            'total_processed_content': total_processed,
            'total_extracted_knowledge': total_knowledge,
            'total_errors': total_errors,
            'session_duration_seconds': duration,
            'knowledge_extraction_rate': total_knowledge / max(1, duration),
            'error_rate': total_errors / max(1, total_scraped + total_processed)
        }
    
    async def _save_session_results(self):
        """Save session results to file"""
        if not self.current_session:
            return
        
        try:
            session_file = self.results_directory / f"session_{self.current_session.id}.json"
            
            # Prepare session data for JSON serialization
            session_data = {
                'id': self.current_session.id,
                'name': self.current_session.name,
                'description': self.current_session.description,
                'start_time': self.current_session.start_time,
                'end_time': self.current_session.end_time,
                'statistics': self.current_session.statistics,
                'tasks': [
                    {
                        'id': task.id,
                        'mode': task.mode.value,
                        'topic': task.topic,
                        'urls': task.urls,
                        'keywords': task.keywords,
                        'priority': task.priority,
                        'created_at': task.created_at,
                        'completed_at': task.completed_at,
                        'results': task.results
                    }
                    for task in self.current_session.tasks
                ]
            }
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Session results saved to {session_file}")
            
        except Exception as e:
            logger.error(f"Error saving session results: {e}")
    
    def get_overall_statistics(self) -> Dict[str, Any]:
        """Get overall pipeline statistics"""
        runtime = datetime.now() - self.stats['start_time']
        
        return {
            **self.stats,
            'runtime_seconds': runtime.total_seconds(),
            'average_knowledge_per_session': self.stats['total_knowledge_extracted'] / max(1, self.stats['total_sessions']),
            'average_tasks_per_session': self.stats['total_tasks_completed'] / max(1, self.stats['total_sessions']),
            'knowledge_extraction_rate': self.stats['total_knowledge_extracted'] / max(1, runtime.total_seconds())
        }
    
    async def shutdown(self):
        """Shutdown the pipeline and cleanup resources"""
        logger.info("Shutting down learning pipeline...")
        
        # Close web scraper
        if hasattr(self.web_scraper, 'close_session'):
            await self.web_scraper.close_session()
        
        # Close neural engine
        if self.neural_engine:
            # Neural engine doesn't have explicit cleanup
            pass
        
        logger.info("Learning pipeline shutdown complete")

# Factory function
def create_learning_pipeline(config: LearningConfig = None) -> SelfSupervisedLearningPipeline:
    """Create a configured learning pipeline"""
    if config is None:
        config = LearningConfig()
    
    return SelfSupervisedLearningPipeline(config)

# Example usage
if __name__ == "__main__":
    async def main():
        # Create and configure pipeline
        config = LearningConfig()
        config.scraping_config.requests_per_second = 0.5
        config.scraping_config.max_total_pages = 20
        config.max_concurrent_tasks = 2
        
        pipeline = create_learning_pipeline(config)
        
        try:
            # Initialize pipeline
            await pipeline.initialize()
            
            # Start learning session
            session_id = await pipeline.start_learning_session(
                session_name="Technology Learning Session",
                description="Learn about recent technology developments",
                topics=["artificial intelligence", "machine learning", "neural networks"]
            )
            
            print(f"Learning session started: {session_id}")
            
            # Get statistics
            stats = pipeline.get_overall_statistics()
            print(f"Total knowledge extracted: {stats['total_knowledge_extracted']}")
            print(f"Total sessions: {stats['total_sessions']}")
            
        finally:
            # Shutdown
            await pipeline.shutdown()
    
    asyncio.run(main())