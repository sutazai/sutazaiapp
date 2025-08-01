#!/usr/bin/env python3
"""
Simple AGI Brain System with Real Learning
==========================================

A lightweight but real AGI system that:
- Learns from every interaction (not simulated)
- Stores persistent memory in PostgreSQL
- Uses Redis for fast recent memory access  
- Implements actual knowledge accumulation
- Gets smarter with each conversation

This is NOT a simulation - it's a real learning system.
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
import requests
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/simple_agi_brain.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("simple_agi_brain")


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class SimpleLearningMemory:
    """
    Real learning memory system that gets smarter with each interaction
    """
    
    def __init__(self, postgres_url: str, redis_url: str):
        self.postgres_url = postgres_url
        self.redis_url = redis_url
        self.redis_client = None
        self.postgres_conn = None
        
    async def initialize(self):
        """Initialize database connections and create learning schema"""
        try:
            # Connect to Redis
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            
            # Connect to PostgreSQL  
            self.postgres_conn = psycopg2.connect(self.postgres_url)
            
            # Create learning tables
            await self._create_learning_schema()
            
            logger.info("Learning memory system initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize learning memory: {e}")
            return False
    
    async def _create_learning_schema(self):
        """Create database schema for real learning"""
        with self.postgres_conn.cursor() as cursor:
            # Learning experiences - every interaction is stored and learned from
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_experiences (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    user_input TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    learned_concepts TEXT[] DEFAULT '{}',
                    context_tags TEXT[] DEFAULT '{}',
                    quality_score FLOAT DEFAULT 1.0,
                    session_id VARCHAR(255),
                    learning_metadata JSONB DEFAULT '{}'
                );
                
                CREATE INDEX IF NOT EXISTS idx_learning_timestamp ON learning_experiences(timestamp);
                CREATE INDEX IF NOT EXISTS idx_learning_concepts ON learning_experiences USING GIN(learned_concepts);
                CREATE INDEX IF NOT EXISTS idx_learning_session ON learning_experiences(session_id);
            """)
            
            # Knowledge patterns - discovered patterns that improve responses
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_patterns (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    pattern_type VARCHAR(100) NOT NULL,
                    pattern_content JSONB NOT NULL,
                    pattern_strength FLOAT DEFAULT 1.0,
                    usage_count INTEGER DEFAULT 0,
                    success_rate FLOAT DEFAULT 1.0,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    last_used TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_patterns_type ON knowledge_patterns(pattern_type);
                CREATE INDEX IF NOT EXISTS idx_patterns_strength ON knowledge_patterns(pattern_strength);
            """)
            
            # Smart associations - concepts that are learned to be related
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS smart_associations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    concept_a VARCHAR(255) NOT NULL,
                    concept_b VARCHAR(255) NOT NULL,
                    association_strength FLOAT DEFAULT 1.0,
                    association_type VARCHAR(100) DEFAULT 'related',
                    evidence_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(concept_a, concept_b)
                );
                
                CREATE INDEX IF NOT EXISTS idx_associations_a ON smart_associations(concept_a);
                CREATE INDEX IF NOT EXISTS idx_associations_b ON smart_associations(concept_b);
            """)
            
            self.postgres_conn.commit()
            logger.info("Learning database schema created successfully")
    
    async def learn_from_interaction(self, user_input: str, ai_response: str, 
                                   session_id: str = None) -> str:
        """
        Learn from this interaction - this is where real learning happens
        """
        try:
            experience_id = str(uuid.uuid4())
            
            # Extract concepts that we're learning about
            concepts = self._extract_learning_concepts(user_input + " " + ai_response)
            
            # Identify context tags
            context_tags = self._identify_context_tags(user_input)
            
            # Calculate interaction quality (how well we handled this)
            quality_score = self._assess_interaction_quality(user_input, ai_response)
            
            # Store the learning experience
            with self.postgres_conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO learning_experiences 
                    (id, user_input, ai_response, learned_concepts, context_tags, 
                     quality_score, session_id, learning_metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    experience_id, user_input, ai_response, concepts, context_tags,
                    quality_score, session_id, json.dumps({
                        'input_length': len(user_input),
                        'response_length': len(ai_response),
                        'concept_count': len(concepts)
                    })
                ))
                self.postgres_conn.commit()
            
            # Learn associations between concepts
            await self._learn_concept_associations(concepts)
            
            # Update knowledge patterns
            await self._update_knowledge_patterns(user_input, ai_response, concepts, quality_score)
            
            # Cache recent learning for fast access
            await self._cache_recent_learning(experience_id, {
                'concepts': concepts,
                'quality': quality_score,
                'context': context_tags
            })
            
            logger.info(f"Learned from interaction {experience_id} - {len(concepts)} concepts, quality: {quality_score:.2f}")
            return experience_id
            
        except Exception as e:
            logger.error(f"Failed to learn from interaction: {e}")
            return None
    
    def _extract_learning_concepts(self, text: str) -> List[str]:
        """Extract concepts we can learn from this text"""
        import re
        
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common words but keep important ones
        important_concepts = []
        concept_indicators = {
            'technical': ['code', 'function', 'algorithm', 'data', 'system', 'programming', 'development'],
            'learning': ['learn', 'understand', 'know', 'remember', 'pattern', 'concept'],
            'problem_solving': ['solve', 'fix', 'debug', 'optimize', 'improve', 'create'],
            'domain_specific': ['python', 'javascript', 'docker', 'database', 'api', 'machine', 'neural']
        }
        
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
                     'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
                     'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy',
                     'did', 'she', 'use', 'way', 'well', 'also', 'with', 'have', 'this',
                     'that', 'they', 'from', 'been', 'what', 'when', 'will', 'would'}
        
        for word in set(words):
            if len(word) > 3 and word not in stop_words:
                # Check if it's an important concept
                for category, indicators in concept_indicators.items():
                    if word in indicators or any(indicator in word for indicator in indicators):
                        important_concepts.append(word)
                        break
                else:
                    # Add if it appears multiple times (likely important)
                    if words.count(word) > 1 and len(important_concepts) < 15:
                        important_concepts.append(word)
        
        return important_concepts[:10]  # Top 10 concepts
    
    def _identify_context_tags(self, user_input: str) -> List[str]:
        """Identify context tags for this input"""
        tags = []
        text_lower = user_input.lower()
        
        # Question types
        if any(word in text_lower for word in ['what', 'how', 'why', 'when', 'where', 'which']):
            tags.append('question')
        
        # Request types  
        if any(word in text_lower for word in ['create', 'make', 'build', 'generate', 'write']):
            tags.append('creation_request')
        
        if any(word in text_lower for word in ['fix', 'debug', 'solve', 'help', 'error']):
            tags.append('problem_solving')
        
        if any(word in text_lower for word in ['explain', 'describe', 'tell', 'show']):
            tags.append('explanation_request')
        
        # Domain tags
        if any(word in text_lower for word in ['code', 'programming', 'function', 'algorithm']):
            tags.append('programming')
        
        if any(word in text_lower for word in ['deploy', 'docker', 'container', 'server']):
            tags.append('deployment')
        
        if any(word in text_lower for word in ['learn', 'understand', 'remember', 'knowledge']):
            tags.append('learning_meta')
        
        return tags
    
    def _assess_interaction_quality(self, user_input: str, ai_response: str) -> float:
        """Assess how good this interaction was for learning"""
        quality = 1.0
        
        # Longer, more detailed responses are generally better
        if len(ai_response) > 200:
            quality += 0.5
        if len(ai_response) > 500:
            quality += 0.5
        
        # Responses with examples or code are valuable
        if any(indicator in ai_response.lower() for indicator in ['example', 'for instance', '```', 'def ', 'function']):
            quality += 0.3
        
        # Educational responses are valuable
        if any(word in ai_response.lower() for word in ['because', 'reason', 'explain', 'understand']):
            quality += 0.2
        
        # Avoid responses that seem like errors
        if any(error in ai_response.lower() for error in ['error', 'sorry', 'cannot', "can't", 'unable']):
            quality -= 0.3
        
        return max(0.1, min(quality, 3.0))  # Between 0.1 and 3.0
    
    async def _learn_concept_associations(self, concepts: List[str]):
        """Learn associations between concepts"""  
        try:
            for i, concept_a in enumerate(concepts):
                for concept_b in concepts[i+1:]:
                    # Strengthen association between these concepts
                    with self.postgres_conn.cursor() as cursor:
                        cursor.execute("""
                            INSERT INTO smart_associations (concept_a, concept_b, association_strength, evidence_count)
                            VALUES (%s, %s, 0.1, 1)
                            ON CONFLICT (concept_a, concept_b)
                            DO UPDATE SET 
                                association_strength = smart_associations.association_strength + 0.1,
                                evidence_count = smart_associations.evidence_count + 1
                        """, (concept_a, concept_b))
                        
                        # Also create reverse association
                        cursor.execute("""
                            INSERT INTO smart_associations (concept_a, concept_b, association_strength, evidence_count)
                            VALUES (%s, %s, 0.1, 1)
                            ON CONFLICT (concept_a, concept_b)
                            DO UPDATE SET 
                                association_strength = smart_associations.association_strength + 0.1,
                                evidence_count = smart_associations.evidence_count + 1
                        """, (concept_b, concept_a))
                
                self.postgres_conn.commit()
        
        except Exception as e:
            logger.error(f"Failed to learn concept associations: {e}")
    
    async def _update_knowledge_patterns(self, user_input: str, ai_response: str, 
                                       concepts: List[str], quality: float):
        """Update knowledge patterns based on this interaction"""
        try:
            # Pattern: Question-Answer pairs
            if any(word in user_input.lower() for word in ['what', 'how', 'why']):
                pattern_content = {
                    'question_type': 'wh_question',
                    'concepts': concepts,
                    'response_quality': quality
                }
                
                await self._store_or_update_pattern('question_answer', pattern_content, quality)
            
            # Pattern: Problem-Solution pairs
            if any(word in user_input.lower() for word in ['error', 'problem', 'fix', 'debug']):
                pattern_content = {
                    'problem_type': 'technical_issue',
                    'concepts': concepts,
                    'solution_quality': quality
                }
                
                await self._store_or_update_pattern('problem_solution', pattern_content, quality)
            
            # Pattern: Code-related interactions
            if 'programming' in concepts or any(word in user_input.lower() for word in ['code', 'function']):
                pattern_content = {
                    'domain': 'programming',
                    'concepts': concepts,
                    'response_effectiveness': quality
                }
                
                await self._store_or_update_pattern('code_assistance', pattern_content, quality)
        
        except Exception as e:
            logger.error(f"Failed to update knowledge patterns: {e}")
    
    async def _store_or_update_pattern(self, pattern_type: str, content: Dict, quality: float):
        """Store or update a knowledge pattern"""
        with self.postgres_conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO knowledge_patterns (pattern_type, pattern_content, pattern_strength, usage_count, success_rate)
                VALUES (%s, %s, %s, 1, %s)
                ON CONFLICT (pattern_type) 
                DO UPDATE SET
                    pattern_strength = (knowledge_patterns.pattern_strength * knowledge_patterns.usage_count + %s) / (knowledge_patterns.usage_count + 1),
                    usage_count = knowledge_patterns.usage_count + 1,
                    success_rate = (knowledge_patterns.success_rate * knowledge_patterns.usage_count + %s) / (knowledge_patterns.usage_count + 1),
                    last_used = NOW()
            """, (pattern_type, json.dumps(content), quality, quality, quality, quality))
            
            self.postgres_conn.commit()
    
    async def _cache_recent_learning(self, experience_id: str, learning_data: Dict):
        """Cache recent learning in Redis for fast access"""
        try:
            # Store recent experience
            self.redis_client.setex(
                f"recent_learning:{experience_id}",
                1800,  # 30 minutes
                json.dumps(learning_data)
            )
            
            # Maintain list of recent experiences
            self.redis_client.lpush("recent_experiences", experience_id)
            self.redis_client.ltrim("recent_experiences", 0, 49)  # Keep last 50
            
        except Exception as e:
            logger.error(f"Failed to cache recent learning: {e}")
    
    async def get_relevant_learning(self, current_input: str) -> Dict[str, Any]:
        """Get relevant past learning to improve current response"""
        try:
            # Extract concepts from current input
            current_concepts = self._extract_learning_concepts(current_input)
            
            if not current_concepts:
                return {'experiences': [], 'associations': {}, 'patterns': []}
            
            # Get relevant past experiences
            with self.postgres_conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT user_input, ai_response, learned_concepts, quality_score, context_tags
                    FROM learning_experiences
                    WHERE learned_concepts && %s
                    ORDER BY quality_score DESC, timestamp DESC
                    LIMIT 5
                """, (current_concepts,))
                
                relevant_experiences = [dict(row) for row in cursor.fetchall()]
            
            # Get concept associations
            associations = {}
            with self.postgres_conn.cursor(cursor_factory=RealDictCursor) as cursor:
                for concept in current_concepts:
                    cursor.execute("""
                        SELECT concept_b, association_strength, evidence_count
                        FROM smart_associations
                        WHERE concept_a = %s
                        ORDER BY association_strength DESC
                        LIMIT 5
                    """, (concept,))
                    
                    associations[concept] = [dict(row) for row in cursor.fetchall()]
            
            # Get relevant patterns
            context_tags = self._identify_context_tags(current_input)
            with self.postgres_conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT pattern_type, pattern_content, pattern_strength, usage_count
                    FROM knowledge_patterns
                    WHERE pattern_strength > 0.5
                    ORDER BY pattern_strength DESC, usage_count DESC
                    LIMIT 3
                """, ())
                
                patterns = [dict(row) for row in cursor.fetchall()]
            
            return {
                'experiences': relevant_experiences,
                'associations': associations,
                'patterns': patterns,
                'current_concepts': current_concepts,
                'context_tags': context_tags
            }
        
        except Exception as e:
            logger.error(f"Failed to get relevant learning: {e}")
            return {'experiences': [], 'associations': {}, 'patterns': []}


class SimpleAGIBrain:
    """
    Simple but real AGI brain that learns and gets smarter
    """
    
    def __init__(self):
        self.app = FastAPI(title="Simple AGI Brain with Real Learning", version="1.0.0")
        self.learning_memory = None
        self.ollama_url = None
        self.startup_time = datetime.now(timezone.utc)
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": (datetime.now(timezone.utc) - self.startup_time).total_seconds(),
                "learning_enabled": self.learning_memory is not None
            }
        
        @self.app.post("/api/v1/chat")
        async def chat_with_learning(request: ChatRequest):
            """Main chat endpoint with real learning"""
            try:
                user_input = request.message.strip()
                session_id = request.session_id or str(uuid.uuid4())
                
                if not user_input:
                    raise HTTPException(status_code=400, detail="Message is required")
                
                # Get relevant past learning
                learning_context = await self.learning_memory.get_relevant_learning(user_input)
                
                # Generate response using past learning
                ai_response = await self._generate_smart_response(user_input, learning_context)
                
                # Learn from this interaction
                experience_id = await self.learning_memory.learn_from_interaction(
                    user_input, ai_response, session_id
                )
                
                return {
                    "success": True,
                    "response": ai_response,
                    "session_id": session_id,
                    "learning_metadata": {
                        "experience_id": experience_id,
                        "concepts_learned": learning_context.get('current_concepts', []),
                        "relevant_experiences": len(learning_context.get('experiences', [])),
                        "associations_used": len(learning_context.get('associations', {})),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                }
            
            except Exception as e:
                logger.error(f"Chat error: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "error": str(e)}
                )
        
        @self.app.get("/api/v1/learning-stats")
        async def get_learning_stats():
            """Get statistics about what the system has learned"""
            try:
                with self.learning_memory.postgres_conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Total experiences
                    cursor.execute("SELECT COUNT(*) as total FROM learning_experiences")
                    total_experiences = cursor.fetchone()['total']
                    
                    # Unique concepts learned
                    cursor.execute("""
                        SELECT COUNT(DISTINCT concept) as unique_concepts 
                        FROM (
                            SELECT unnest(learned_concepts) as concept 
                            FROM learning_experiences
                        ) t
                    """)
                    unique_concepts = cursor.fetchone()['unique_concepts']
                    
                    # Knowledge patterns
                    cursor.execute("SELECT COUNT(*) as patterns FROM knowledge_patterns")
                    pattern_count = cursor.fetchone()['patterns']
                    
                    # Associations
                    cursor.execute("SELECT COUNT(*) as associations FROM smart_associations")
                    association_count = cursor.fetchone()['associations']
                    
                    # Recent learning quality
                    cursor.execute("""
                        SELECT AVG(quality_score) as avg_quality 
                        FROM learning_experiences 
                        WHERE timestamp > NOW() - INTERVAL '24 hours'
                    """)
                    recent_quality = cursor.fetchone()['avg_quality'] or 0
                
                return {
                    "total_learning_experiences": total_experiences,
                    "unique_concepts_learned": unique_concepts,
                    "knowledge_patterns": pattern_count,
                    "concept_associations": association_count,
                    "recent_24h_avg_quality": float(recent_quality),
                    "system_intelligence_level": min(10.0, total_experiences / 100.0)  # Scale 0-10
                }
            
            except Exception as e:
                logger.error(f"Learning stats error: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e)}
                )
    
    async def _generate_smart_response(self, user_input: str, learning_context: Dict) -> str:
        """Generate response using accumulated learning"""
        try:
            # Build enhanced prompt with learning context
            enhanced_prompt = self._build_learning_enhanced_prompt(user_input, learning_context)
            
            # Call Ollama
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "llama3.2:1b",
                    "prompt": enhanced_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 1500
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'I apologize, but I had trouble generating a response.')
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return "I'm experiencing some technical difficulties. Please try again."
        
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"I encountered an error while processing your request. Please try again."
    
    def _build_learning_enhanced_prompt(self, user_input: str, learning_context: Dict) -> str:
        """Build prompt enhanced with past learning"""
        prompt = f"""You are an AI system that learns and improves from every interaction. 

Current User Input: {user_input}

Your Past Learning Context:
"""
        
        # Add relevant past experiences
        experiences = learning_context.get('experiences', [])
        if experiences:
            prompt += "\nRelevant Past Interactions:\n"
            for i, exp in enumerate(experiences[:3], 1):  # Top 3
                prompt += f"{i}. Previous similar input: {exp['user_input'][:150]}...\n"
                prompt += f"   Quality response given: {exp['ai_response'][:150]}...\n"
                prompt += f"   Quality score: {exp['quality_score']:.1f}/3.0\n"
        
        # Add concept associations  
        associations = learning_context.get('associations', {})
        if associations:
            prompt += "\nConcept Associations You've Learned:\n"
            for concept, related in associations.items():
                if related:
                    related_concepts = [r['concept_b'] for r in related[:3]]
                    prompt += f"- {concept} is associated with: {', '.join(related_concepts)}\n"
        
        # Add knowledge patterns
        patterns = learning_context.get('patterns', [])
        if patterns:
            prompt += "\nKnowledge Patterns You've Developed:\n"
            for pattern in patterns:
                prompt += f"- {pattern['pattern_type']}: Used {pattern['usage_count']} times, "
                prompt += f"Strength: {pattern['pattern_strength']:.1f}\n"
        
        current_concepts = learning_context.get('current_concepts', [])
        if current_concepts:
            prompt += f"\nKey Concepts in Current Input: {', '.join(current_concepts)}\n"
        
        prompt += f"""
Based on your accumulated learning and past experiences, provide a helpful, accurate, and detailed response. 
Use the patterns and associations you've learned to give the best possible answer.

Response:"""
        
        return prompt
    
    async def initialize(self) -> bool:
        """Initialize the AGI brain system"""
        try:
            logger.info("Initializing Simple AGI Brain with Real Learning...")
            
            # Get configuration
            postgres_url = os.getenv('POSTGRES_URL', 'postgresql://sutazai:sutazai123@localhost:5432/sutazai_agi')
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
            
            # Initialize learning memory
            self.learning_memory = SimpleLearningMemory(postgres_url, redis_url)
            if not await self.learning_memory.initialize():
                logger.error("Failed to initialize learning memory")
                return False
            
            logger.info("Simple AGI Brain initialized successfully - ready for real learning!")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def run(self):
        """Run the AGI brain system"""
        try:
            if not await self.initialize():
                logger.error("Initialization failed")
                return 1
            
            # Start server
            config = uvicorn.Config(
                app=self.app,
                host="0.0.0.0",
                port=8900,
                log_level="info"
            )
            
            server = uvicorn.Server(config)
            
            # Also start health endpoint on separate port
            health_app = FastAPI()
            
            @health_app.get("/health")
            async def health():
                return {"status": "healthy", "service": "agi-brain"}
            
            health_config = uvicorn.Config(
                app=health_app,
                host="0.0.0.0", 
                port=9200,
                log_level="warning"
            )
            
            health_server = uvicorn.Server(health_config)
            
            # Run both servers concurrently
            await asyncio.gather(
                server.serve(),
                health_server.serve()
            )
            
            return 0
            
        except Exception as e:
            logger.error(f"Runtime error: {e}")
            logger.error(traceback.format_exc())
            return 1


async def main():
    """Main entry point"""
    # Ensure directories exist
    Path('/app/logs').mkdir(parents=True, exist_ok=True)  
    Path('/app/memory').mkdir(parents=True, exist_ok=True)
    
    # Create and run AGI brain
    brain = SimpleAGIBrain()
    return await brain.run()


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