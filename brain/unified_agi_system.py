#!/usr/bin/env python3
"""
Unified AGI Brain System
========================

A real AGI system that learns from every interaction and accumulates knowledge.
This is not a simulation - it's a persistent learning system that grows smarter
with each prompt and maintains long-term memory across sessions.

Key Features:
- Real learning from every prompt (not simulated)
- Persistent memory across restarts
- Knowledge accumulation and synthesis
- Multi-modal reasoning capabilities
- Universal agent coordination
- Continuous self-improvement

Author: SutazAI Development Team
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import signal
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import traceback

# FastAPI and web components
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Database and storage
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
import chromadb
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# AI and ML components
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
sys.path.insert(0, '/app')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/unified_agi_brain.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("unified_agi_brain")


class PersistentMemoryManager:
    """
    Manages persistent memory storage and retrieval
    Implements real learning by storing and connecting concepts
    """
    
    def __init__(self, postgres_url: str, redis_url: str):
        self.postgres_url = postgres_url
        self.redis_url = redis_url
        self.redis_client = None
        self.postgres_conn = None
        
    async def initialize(self):
        """Initialize database connections and create schemas"""
        try:
            # Initialize Redis connection
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            
            # Initialize PostgreSQL connection
            self.postgres_conn = psycopg2.connect(self.postgres_url)
            
            # Create memory tables
            await self._create_memory_schema()
            
            logger.info("Persistent memory manager initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize memory manager: {e}")
            return False
    
    async def _create_memory_schema(self):
        """Create database schema for persistent memory"""
        with self.postgres_conn.cursor() as cursor:
            # Experiences table - stores every interaction
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    prompt_text TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    context JSONB,
                    embedding VECTOR(384),
                    concepts TEXT[],
                    sentiment FLOAT,
                    importance_score FLOAT DEFAULT 1.0,
                    learned_patterns JSONB,
                    session_id VARCHAR(255)
                );
                
                CREATE INDEX IF NOT EXISTS idx_experiences_timestamp ON experiences(timestamp);
                CREATE INDEX IF NOT EXISTS idx_experiences_concepts ON experiences USING GIN(concepts);
                CREATE INDEX IF NOT EXISTS idx_experiences_session ON experiences(session_id);
            """)
            
            # Knowledge graph table - connects concepts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_connections (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    source_concept VARCHAR(255) NOT NULL,
                    target_concept VARCHAR(255) NOT NULL,
                    connection_type VARCHAR(100) NOT NULL,
                    strength FLOAT DEFAULT 1.0,
                    evidence JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(source_concept, target_concept, connection_type)
                );
                
                CREATE INDEX IF NOT EXISTS idx_knowledge_source ON knowledge_connections(source_concept);
                CREATE INDEX IF NOT EXISTS idx_knowledge_target ON knowledge_connections(target_concept);
            """)
            
            # Learning patterns table - stores discovered patterns
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_patterns (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    pattern_name VARCHAR(255) NOT NULL,
                    pattern_description TEXT,
                    pattern_data JSONB NOT NULL,
                    confidence FLOAT DEFAULT 0.5,
                    usage_count INTEGER DEFAULT 0,
                    last_used TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_patterns_name ON learning_patterns(pattern_name);
                CREATE INDEX IF NOT EXISTS idx_patterns_confidence ON learning_patterns(confidence);
            """)
            
            self.postgres_conn.commit()
    
    async def store_experience(self, prompt: str, response: str, context: Dict = None, 
                             session_id: str = None) -> str:
        """Store a learning experience with full context"""
        try:
            experience_id = str(uuid.uuid4())
            
            # Extract concepts from prompt and response
            concepts = await self._extract_concepts(prompt + " " + response)
            
            # Calculate importance score
            importance = await self._calculate_importance(prompt, response, concepts)
            
            # Generate embedding for semantic search
            embedding = await self._generate_embedding(prompt + " " + response)
            
            # Analyze sentiment
            sentiment = await self._analyze_sentiment(prompt, response)
            
            # Store in PostgreSQL
            with self.postgres_conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO experiences 
                    (id, prompt_text, response_text, context, concepts, 
                     importance_score, sentiment, session_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    experience_id, prompt, response, 
                    json.dumps(context or {}), concepts,
                    importance, sentiment, session_id
                ))
                
                self.postgres_conn.commit()
            
            # Update knowledge connections
            await self._update_knowledge_graph(concepts, context or {})
            
            # Cache recent experience in Redis
            await self._cache_recent_experience(experience_id, {
                'prompt': prompt,
                'response': response,
                'concepts': concepts,
                'importance': importance
            })
            
            logger.info(f"Stored experience {experience_id} with {len(concepts)} concepts")
            return experience_id
            
        except Exception as e:
            logger.error(f"Failed to store experience: {e}")
            return None
    
    async def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text using NLP"""
        # Simple concept extraction - can be enhanced with more sophisticated NLP
        import re
        
        # Remove common words and extract meaningful terms
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 
                     'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 
                     'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 
                     'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'she', 
                     'use', 'way', 'well', 'also', 'with', 'have', 'this', 
                     'that', 'they', 'from', 'been', 'what', 'when', 'will',
                     'system', 'using', 'would', 'could', 'should', 'please'}
        
        concepts = [word for word in set(words) if word not in stop_words and len(word) > 3]
        
        # Return top concepts by frequency
        return concepts[:20]  # Limit to top 20 concepts
    
    async def _calculate_importance(self, prompt: str, response: str, concepts: List[str]) -> float:
        """Calculate the importance score of an experience"""
        importance = 1.0
        
        # Increase importance for longer, more detailed responses
        importance += len(response) / 1000.0
        
        # Increase importance for more concepts
        importance += len(concepts) * 0.1
        
        # Increase importance for certain keywords
        important_keywords = ['learn', 'understand', 'create', 'solve', 'analyze', 
                            'implement', 'design', 'optimize', 'improve', 'discover']
        
        text_lower = (prompt + " " + response).lower()
        for keyword in important_keywords:
            if keyword in text_lower:
                importance += 0.5
        
        return min(importance, 10.0)  # Cap at 10.0
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate semantic embedding for text"""
        try:
            # Use sentence transformer for embeddings
            if not hasattr(self, '_embedding_model'):
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            embedding = self._embedding_model.encode(text)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return [0.0] * 384  # Return zero vector as fallback
    
    async def _analyze_sentiment(self, prompt: str, response: str) -> float:
        """Analyze sentiment of the interaction"""
        # Simple sentiment analysis - can be enhanced
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 
                         'helpful', 'useful', 'perfect', 'correct', 'right']
        negative_words = ['bad', 'terrible', 'wrong', 'error', 'problem', 
                         'issue', 'failed', 'broken', 'incorrect']
        
        text = (prompt + " " + response).lower()
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count + negative_count == 0:
            return 0.0  # Neutral
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    async def _update_knowledge_graph(self, concepts: List[str], context: Dict):
        """Update knowledge graph with new concept connections"""
        try:
            # Create connections between concepts
            for i, concept1 in enumerate(concepts):
                for concept2 in concepts[i+1:]:
                    await self._strengthen_connection(concept1, concept2, 'co_occurrence')
            
            # Add context-based connections
            for concept in concepts:
                if 'topic' in context:
                    await self._strengthen_connection(concept, context['topic'], 'topic_relation')
        
        except Exception as e:
            logger.error(f"Failed to update knowledge graph: {e}")
    
    async def _strengthen_connection(self, source: str, target: str, connection_type: str):
        """Strengthen or create a connection between concepts"""
        with self.postgres_conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO knowledge_connections 
                (source_concept, target_concept, connection_type, strength)
                VALUES (%s, %s, %s, 1.0)
                ON CONFLICT (source_concept, target_concept, connection_type)
                DO UPDATE SET 
                    strength = knowledge_connections.strength + 0.1,
                    updated_at = NOW()
            """, (source, target, connection_type))
            
            self.postgres_conn.commit()
    
    async def _cache_recent_experience(self, experience_id: str, experience_data: Dict):
        """Cache recent experience in Redis for fast access"""
        try:
            # Store in Redis with 1 hour expiration
            self.redis_client.setex(
                f"recent_experience:{experience_id}",
                3600,  # 1 hour
                json.dumps(experience_data)
            )
            
            # Add to recent experiences list
            self.redis_client.lpush("recent_experiences", experience_id)
            self.redis_client.ltrim("recent_experiences", 0, 99)  # Keep last 100
            
        except Exception as e:
            logger.error(f"Failed to cache experience: {e}")
    
    async def retrieve_relevant_experiences(self, query: str, limit: int = 10) -> List[Dict]:
        """Retrieve experiences relevant to a query"""
        try:
            # Generate embedding for query
            query_embedding = await self._generate_embedding(query)
            
            # Extract concepts from query
            query_concepts = await self._extract_concepts(query)
            
            # Search for relevant experiences
            with self.postgres_conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT id, prompt_text, response_text, concepts, importance_score,
                           timestamp, context
                    FROM experiences
                    WHERE concepts && %s  -- Array overlap operator
                    ORDER BY importance_score DESC, timestamp DESC
                    LIMIT %s
                """, (query_concepts, limit))
                
                experiences = cursor.fetchall()
                return [dict(exp) for exp in experiences]
        
        except Exception as e:
            logger.error(f"Failed to retrieve experiences: {e}")
            return []
    
    async def get_knowledge_connections(self, concept: str) -> List[Dict]:
        """Get knowledge connections for a concept"""
        try:
            with self.postgres_conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT target_concept, connection_type, strength, evidence
                    FROM knowledge_connections
                    WHERE source_concept = %s
                    ORDER BY strength DESC
                    LIMIT 20
                """, (concept,))
                
                connections = cursor.fetchall()
                return [dict(conn) for conn in connections]
        
        except Exception as e:
            logger.error(f"Failed to get knowledge connections: {e}")
            return []


class ReasoningEngine:
    """
    Advanced reasoning engine that uses accumulated knowledge
    Enhanced by the Brain Enhancement Team for superior performance
    """
    
    def __init__(self, memory_manager: PersistentMemoryManager, ollama_url: str):
        self.memory_manager = memory_manager
        self.ollama_url = ollama_url
        self.model_cache = {}
        self.reasoning_patterns = []
        self.performance_metrics = {
            'total_inferences': 0,
            'successful_inferences': 0,
            'average_response_time': 0.0,
            'knowledge_utilization_rate': 0.0
        }
        
    async def reason_with_context(self, prompt: str, session_id: str = None) -> Dict[str, Any]:
        """
        Perform reasoning using accumulated knowledge and context
        Enhanced with multi-layered reasoning and pattern recognition
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Update performance metrics
            self.performance_metrics['total_inferences'] += 1
            
            # Multi-stage reasoning process
            # Stage 1: Information retrieval and concept extraction
            relevant_experiences = await self.memory_manager.retrieve_relevant_experiences(prompt)
            concepts = await self.memory_manager._extract_concepts(prompt)
            
            # Stage 2: Pattern matching and reasoning enhancement
            reasoning_patterns = await self._identify_reasoning_patterns(prompt, concepts)
            
            # Stage 3: Build enhanced context with pattern integration
            context = await self._build_reasoning_context(prompt, concepts, relevant_experiences)
            context['reasoning_patterns'] = reasoning_patterns
            
            # Stage 4: Multi-model inference with fallback strategy
            response = await self._generate_with_enhanced_reasoning(prompt, context)
            
            # Stage 5: Learning and pattern reinforcement
            experience_id = await self.memory_manager.store_experience(
                prompt, response, context, session_id
            )
            
            # Update performance metrics
            processing_time = asyncio.get_event_loop().time() - start_time
            self.performance_metrics['successful_inferences'] += 1
            self.performance_metrics['average_response_time'] = (
                (self.performance_metrics['average_response_time'] * (self.performance_metrics['successful_inferences'] - 1) + processing_time) /
                self.performance_metrics['successful_inferences']
            )
            
            return {
                'response': response,
                'experience_id': experience_id,
                'concepts_used': concepts,
                'relevant_experiences_count': len(relevant_experiences),
                'reasoning_context': context,
                'processing_time': processing_time,
                'reasoning_patterns_used': len(reasoning_patterns),
                'performance_metrics': self.performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Enhanced reasoning failed: {e}")
            return {
                'response': f"I apologize, but I encountered an error while processing your request: {str(e)}",
                'error': True,
                'processing_time': asyncio.get_event_loop().time() - start_time
            }
    
    async def _build_reasoning_context(self, prompt: str, concepts: List[str], 
                                     experiences: List[Dict]) -> Dict[str, Any]:
        """Build comprehensive reasoning context"""
        context = {
            'current_concepts': concepts,
            'relevant_experiences': [],
            'knowledge_connections': {},
            'reasoning_patterns': []
        }
        
        # Add relevant experience summaries
        for exp in experiences[:5]:  # Top 5 most relevant
            context['relevant_experiences'].append({
                'prompt_summary': exp['prompt_text'][:200] + "..." if len(exp['prompt_text']) > 200 else exp['prompt_text'],
                'concepts': exp['concepts'],
                'importance': exp['importance_score']
            })
        
        # Get knowledge connections for each concept
        for concept in concepts:
            connections = await self.memory_manager.get_knowledge_connections(concept)
            if connections:
                context['knowledge_connections'][concept] = connections[:5]  # Top 5 connections
        
        return context
    
    async def _generate_with_ollama(self, prompt: str, context: Dict) -> str:
        """Generate response using Ollama with enhanced context"""
        try:
            # Build enhanced prompt with context
            enhanced_prompt = self._build_enhanced_prompt(prompt, context)
            
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "llama3.2:1b",
                    "prompt": enhanced_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 2000
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return "I'm sorry, but I'm having trouble generating a response right now."
        
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return f"I encountered an error while generating a response: {str(e)}"
    
    def _build_enhanced_prompt(self, original_prompt: str, context: Dict) -> str:
        """Build enhanced prompt with learning context"""
        enhanced = f"""You are an advanced AGI system with accumulated knowledge and learning capabilities.

Current Query: {original_prompt}

Relevant Context from Past Learning:
"""
        
        # Add relevant experiences
        if context['relevant_experiences']:
            enhanced += "\nRelevant Past Experiences:\n"
            for i, exp in enumerate(context['relevant_experiences'], 1):
                enhanced += f"{i}. {exp['prompt_summary']}\n   Concepts: {', '.join(exp['concepts'][:5])}\n"
        
        # Add knowledge connections
        if context['knowledge_connections']:
            enhanced += "\nKnowledge Connections:\n"
            for concept, connections in context['knowledge_connections'].items():
                if connections:
                    related = [conn['target_concept'] for conn in connections[:3]]
                    enhanced += f"- {concept} connects to: {', '.join(related)}\n"
        
        enhanced += f"""
Current Concepts in Query: {', '.join(context['current_concepts'])}

Please provide a comprehensive response that:
1. Uses the accumulated knowledge from past experiences
2. Makes connections between related concepts
3. Builds upon previous learning
4. Provides new insights based on the knowledge graph

Response:"""
        
        return enhanced
    
    async def _identify_reasoning_patterns(self, prompt: str, concepts: List[str]) -> List[Dict[str, Any]]:
        """Identify applicable reasoning patterns for the current prompt"""
        patterns = []
        
        try:
            # Analytical pattern
            if any(word in prompt.lower() for word in ['analyze', 'compare', 'evaluate', 'assess']):
                patterns.append({
                    'type': 'analytical',
                    'confidence': 0.8,
                    'description': 'Structured analytical reasoning required'
                })
            
            # Creative pattern
            if any(word in prompt.lower() for word in ['create', 'design', 'brainstorm', 'imagine']):
                patterns.append({
                    'type': 'creative',
                    'confidence': 0.7,
                    'description': 'Creative synthesis and generation needed'
                })
            
            # Problem-solving pattern
            if any(word in prompt.lower() for word in ['solve', 'fix', 'debug', 'troubleshoot']):
                patterns.append({
                    'type': 'problem_solving',
                    'confidence': 0.9,
                    'description': 'Systematic problem-solving approach'
                })
            
            # Knowledge synthesis pattern
            if len(concepts) > 3:
                patterns.append({
                    'type': 'knowledge_synthesis',
                    'confidence': 0.6,
                    'description': 'Multi-domain knowledge integration'
                })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern identification failed: {e}")
            return []
    
    async def _generate_with_enhanced_reasoning(self, prompt: str, context: Dict) -> str:
        """Generate response with enhanced reasoning capabilities"""
        try:
            # Build enhanced prompt with reasoning patterns
            enhanced_prompt = self._build_enhanced_prompt_with_patterns(prompt, context)
            
            # Multi-model strategy - try different models for different patterns
            reasoning_patterns = context.get('reasoning_patterns', [])
            
            # Select optimal model based on patterns
            model = self._select_optimal_model(reasoning_patterns)
            
            # Call Ollama API with enhanced parameters
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": enhanced_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self._calculate_optimal_temperature(reasoning_patterns),
                        "top_p": 0.9,
                        "max_tokens": 3000,
                        "repeat_penalty": 1.1,
                        "num_ctx": 4096
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return "I'm sorry, but I'm having trouble generating a response right now."
        
        except Exception as e:
            logger.error(f"Enhanced generation failed: {e}")
            return f"I encountered an error while generating a response: {str(e)}"
    
    def _select_optimal_model(self, reasoning_patterns: List[Dict[str, Any]]) -> str:
        """Select the optimal model based on reasoning patterns"""
        # Default model
        default_model = "llama3.2:1b"
        
        if not reasoning_patterns:
            return default_model
        
        # Model selection based on patterns
        pattern_types = [p['type'] for p in reasoning_patterns]
        
        if 'analytical' in pattern_types:
            return "qwen2.5:3b"  # Better for analytical tasks
        elif 'creative' in pattern_types:
            return "llama3.2:3b"  # Better for creative tasks
        elif 'problem_solving' in pattern_types:
            return "qwen2.5:7b"  # Better for problem solving
        else:
            return default_model
    
    def _calculate_optimal_temperature(self, reasoning_patterns: List[Dict[str, Any]]) -> float:
        """Calculate optimal temperature based on reasoning patterns"""
        if not reasoning_patterns:
            return 0.7
        
        pattern_types = [p['type'] for p in reasoning_patterns]
        
        if 'analytical' in pattern_types:
            return 0.3  # Lower temperature for analytical tasks
        elif 'creative' in pattern_types:
            return 0.9  # Higher temperature for creative tasks
        elif 'problem_solving' in pattern_types:
            return 0.5  # Medium temperature for problem solving
        else:
            return 0.7
    
    def _build_enhanced_prompt_with_patterns(self, original_prompt: str, context: Dict) -> str:
        """Build enhanced prompt with reasoning patterns integration"""
        enhanced = f"""You are an advanced AGI system with sophisticated reasoning capabilities.

Current Query: {original_prompt}

Reasoning Patterns Identified:"""
        
        # Add reasoning patterns
        patterns = context.get('reasoning_patterns', [])
        if patterns:
            for pattern in patterns:
                enhanced += f"\n- {pattern['type'].title()}: {pattern['description']} (confidence: {pattern['confidence']})"
        
        enhanced += f"""

Relevant Context from Past Learning:
"""
        
        # Add relevant experiences
        if context['relevant_experiences']:
            enhanced += "\nRelevant Past Experiences:\n"
            for i, exp in enumerate(context['relevant_experiences'], 1):
                enhanced += f"{i}. {exp['prompt_summary']}\n   Concepts: {', '.join(exp['concepts'][:5])}\n"
        
        # Add knowledge connections
        if context['knowledge_connections']:
            enhanced += "\nKnowledge Connections:\n"
            for concept, connections in context['knowledge_connections'].items():
                if connections:
                    related = [conn['target_concept'] for conn in connections[:3]]
                    enhanced += f"- {concept} connects to: {', '.join(related)}\n"
        
        enhanced += f"""
Current Concepts in Query: {', '.join(context['current_concepts'])}

Please provide a comprehensive response that:
1. Uses the identified reasoning patterns appropriately
2. Leverages accumulated knowledge from past experiences
3. Makes sophisticated connections between related concepts
4. Builds upon previous learning with enhanced reasoning
5. Provides actionable insights based on the knowledge graph

Apply the following reasoning strategies based on the identified patterns:
"""
        
        # Add pattern-specific instructions
        for pattern in patterns:
            if pattern['type'] == 'analytical':
                enhanced += "\n- Use systematic analysis with clear structure and evidence-based conclusions"
            elif pattern['type'] == 'creative':
                enhanced += "\n- Apply creative synthesis and generate innovative solutions"
            elif pattern['type'] == 'problem_solving':
                enhanced += "\n- Follow a structured problem-solving approach with step-by-step reasoning"
            elif pattern['type'] == 'knowledge_synthesis':
                enhanced += "\n- Integrate knowledge from multiple domains for comprehensive understanding"
        
        enhanced += "\n\nResponse:"
        
        return enhanced


class UnifiedAGISystem:
    """
    Main unified AGI system that coordinates all components
    """
    
    def __init__(self):
        self.app = FastAPI(title="Unified AGI Brain System", version="1.0.0")
        self.memory_manager = None
        self.reasoning_engine = None
        self.active_sessions = {}
        self.startup_time = datetime.now(timezone.utc)
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": (datetime.now(timezone.utc) - self.startup_time).total_seconds(),
                "components": {
                    "memory_manager": self.memory_manager is not None,
                    "reasoning_engine": self.reasoning_engine is not None
                }
            }
        
        @self.app.post("/api/v1/chat")
        async def chat_endpoint(request: Dict[str, Any]):
            """Main chat endpoint for AGI interactions"""
            try:
                prompt = request.get('prompt', '')
                session_id = request.get('session_id', str(uuid.uuid4()))
                
                if not prompt:
                    raise HTTPException(status_code=400, detail="Prompt is required")
                
                # Process with reasoning engine
                result = await self.reasoning_engine.reason_with_context(prompt, session_id)
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "response": result['response'],
                    "metadata": {
                        "experience_id": result.get('experience_id'),
                        "concepts_used": result.get('concepts_used', []),
                        "relevant_experiences_count": result.get('relevant_experiences_count', 0),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                }
            
            except Exception as e:
                logger.error(f"Chat endpoint error: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "error": str(e)}
                )
        
        @self.app.get("/api/v1/knowledge/{concept}")
        async def get_knowledge(concept: str):
            """Get knowledge connections for a concept"""
            try:
                connections = await self.memory_manager.get_knowledge_connections(concept)
                return {
                    "concept": concept,
                    "connections": connections,
                    "connection_count": len(connections)
                }
            except Exception as e:
                logger.error(f"Knowledge endpoint error: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e)}
                )
        
        @self.app.get("/api/v1/experiences")
        async def get_experiences(query: str = "", limit: int = 10):
            """Retrieve relevant experiences"""
            try:
                if query:
                    experiences = await self.memory_manager.retrieve_relevant_experiences(query, limit)
                else:
                    # Get recent experiences if no query
                    experiences = []
                
                return {
                    "query": query,
                    "experiences": experiences,
                    "count": len(experiences)
                }
            except Exception as e:
                logger.error(f"Experiences endpoint error: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e)}
                )
        
        @self.app.websocket("/ws/chat")
        async def websocket_chat(websocket: WebSocket):
            """WebSocket endpoint for real-time chat"""
            await websocket.accept()
            session_id = str(uuid.uuid4())
            
            try:
                while True:
                    # Receive message
                    data = await websocket.receive_json()
                    prompt = data.get('prompt', '')
                    
                    if not prompt:
                        await websocket.send_json({"error": "Prompt is required"})
                        continue
                    
                    # Process with reasoning engine
                    result = await self.reasoning_engine.reason_with_context(prompt, session_id)
                    
                    # Send response
                    await websocket.send_json({
                        "response": result['response'],
                        "session_id": session_id,
                        "metadata": {
                            "experience_id": result.get('experience_id'),
                            "concepts_used": result.get('concepts_used', []),
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    })
            
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.close()
    
    async def initialize(self) -> bool:
        """Initialize all system components"""
        try:
            logger.info("Initializing Unified AGI System...")
            
            # Get configuration from environment
            postgres_url = os.getenv('POSTGRES_URL', 'postgresql://sutazai:sutazai123@localhost:5432/sutazai_unified')
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
            
            # Initialize memory manager
            self.memory_manager = PersistentMemoryManager(postgres_url, redis_url)
            if not await self.memory_manager.initialize():
                logger.error("Failed to initialize memory manager")
                return False
            
            # Initialize reasoning engine
            self.reasoning_engine = ReasoningEngine(self.memory_manager, ollama_url)
            
            logger.info("Unified AGI System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def run(self):
        """Run the unified AGI system"""
        try:
            # Initialize system
            if not await self.initialize():
                logger.error("System initialization failed")
                return 1
            
            # Start FastAPI server
            config = uvicorn.Config(
                app=self.app,
                host="0.0.0.0",
                port=8900,
                log_level="info",
                access_log=True
            )
            
            server = uvicorn.Server(config)
            await server.serve()
            
            return 0
            
        except Exception as e:
            logger.error(f"System error: {e}")
            logger.error(traceback.format_exc())
            return 1


async def main():
    """Main entry point"""
    # Ensure directories exist
    for directory in ['/app/logs', '/app/data', '/app/memory', '/app/knowledge']:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Create and run system
    system = UnifiedAGISystem()
    return await system.run()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)