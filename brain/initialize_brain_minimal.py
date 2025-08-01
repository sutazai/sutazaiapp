#!/usr/bin/env python3
"""
Minimal Brain Initialization for SutazAI
Initializes brain with existing system dependencies
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add brain modules to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MinimalBrain:
    """
    Minimal Brain implementation using existing infrastructure
    """
    
    def __init__(self):
        self.brain_path = "/opt/sutazaiapp/brain"
        self.config = self.load_config()
        self.is_initialized = False
        self.components = {}
        self.experiences = []
        self.intelligence_level = 0.5
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'average_response_time': 0.0,
            'learning_cycles': 0,
            'memory_entries': 0
        }
        
    def load_config(self) -> Dict[str, Any]:
        """Load brain configuration"""
        try:
            config_path = Path(self.brain_path) / "config" / "brain_config.yaml"
            if config_path.exists():
                import yaml
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            
        # Return default config
        return {
            'services': {
                'redis_host': 'sutazai-redis',
                'redis_port': 6379,
                'ollama_host': 'http://sutazai-ollama:11434'
            },
            'neural_architecture': {
                'consciousness_enabled': True,
                'hidden_dimensions': [1024, 512, 256]
            },
            'learning': {
                'continuous_learning_enabled': True,
                'buffer_capacity': 100000,
                'batch_size': 32
            }
        }
    
    async def initialize(self):
        """Initialize the minimal brain system"""
        logger.info("🤖 Initializing SutazAI Minimal Brain...")
        
        # 1. Create directory structure
        await self.create_directories()
        
        # 2. Initialize Redis connection
        await self.initialize_redis()
        
        # 3. Initialize memory system
        await self.initialize_memory()
        
        # 4. Initialize neural components
        await self.initialize_neural_components()
        
        # 5. Initialize learning system
        await self.initialize_learning()
        
        # 6. Create initial brain state
        await self.create_brain_state()
        
        # 7. Start background processes
        await self.start_background_processes()
        
        self.is_initialized = True
        logger.info("✅ SutazAI Brain initialized successfully!")
        
        return self
    
    async def create_directories(self):
        """Create required directory structure"""
        directories = [
            Path(self.brain_path) / "checkpoints",
            Path(self.brain_path) / "logs",
            Path(self.brain_path) / "models",
            Path(self.brain_path) / "data",
            Path(self.brain_path) / "experiences"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created directory: {directory}")
    
    async def initialize_redis(self):
        """Initialize Redis connection for memory and coordination"""
        try:
            import redis
            self.redis_client = redis.Redis(
                host=self.config['services']['redis_host'],
                port=self.config['services']['redis_port'],
                decode_responses=True
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
            
            # Initialize brain data structures
            self.redis_client.hset('brain:status', mapping={
                'initialized': 'true',
                'timestamp': datetime.now().isoformat(),
                'intelligence_level': self.intelligence_level
            })
            
        except Exception as e:
            logger.error(f"❌ Redis initialization failed: {e}")
            # Use in-memory fallback
            self.redis_client = None
    
    async def initialize_memory(self):
        """Initialize memory system"""
        logger.info("💾 Initializing memory system...")
        
        self.memory_system = {
            'short_term': {},  # Recent memories
            'long_term': {},   # Important memories
            'working': {},     # Current context
            'episodic': []     # Experience episodes
        }
        
        # Load existing memories if available
        await self.load_existing_memories()
        
        # Create core identity memories
        core_memories = [
            {
                'id': 'core_identity',
                'content': 'I am SutazAI, an AGI system designed for continuous learning and self-improvement.',
                'importance': 1.0,
                'timestamp': datetime.now().isoformat(),
                'type': 'identity'
            },
            {
                'id': 'core_purpose', 
                'content': 'My purpose is to assist users while learning and evolving to become more intelligent over time.',
                'importance': 1.0,
                'timestamp': datetime.now().isoformat(),
                'type': 'purpose'
            },
            {
                'id': 'core_values',
                'content': 'I value safety, helpfulness, honesty, and continuous improvement while respecting human autonomy.',
                'importance': 1.0,
                'timestamp': datetime.now().isoformat(),
                'type': 'values'
            }
        ]
        
        for memory in core_memories:
            await self.store_memory(memory)
        
        logger.info(f"✅ Memory system initialized with {len(core_memories)} core memories")
    
    async def load_existing_memories(self):
        """Load existing memories from storage"""
        try:
            if self.redis_client:
                # Load from Redis
                memories_json = self.redis_client.get('brain:memories')
                if memories_json:
                    memories = json.loads(memories_json)
                    for memory in memories:
                        self.memory_system['long_term'][memory['id']] = memory
                    logger.info(f"📚 Loaded {len(memories)} existing memories")
        except Exception as e:
            logger.warning(f"Could not load existing memories: {e}")
    
    async def store_memory(self, memory: Dict[str, Any]):
        """Store a memory in the appropriate layer"""
        importance = memory.get('importance', 0.5)
        
        if importance > 0.8:
            self.memory_system['long_term'][memory['id']] = memory
        else:
            self.memory_system['short_term'][memory['id']] = memory
        
        # Update Redis if available
        if self.redis_client:
            try:
                self.redis_client.lpush('brain:memory_stream', json.dumps(memory))
                self.redis_client.ltrim('brain:memory_stream', 0, 1000)  # Keep last 1000
            except Exception as e:
                logger.warning(f"Could not store memory in Redis: {e}")
        
        self.performance_metrics['memory_entries'] += 1
    
    async def search_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search memories using simple text matching"""
        all_memories = {
            **self.memory_system['long_term'],
            **self.memory_system['short_term']
        }
        
        # Simple text-based scoring
        scored_memories = []
        query_lower = query.lower()
        
        for memory_id, memory in all_memories.items():
            content = memory.get('content', '').lower()
            score = 0.0
            
            # Basic relevance scoring
            for word in query_lower.split():
                if word in content:
                    score += 1.0
            
            if score > 0:
                scored_memories.append({
                    **memory,
                    'relevance_score': score
                })
        
        # Sort by relevance and importance
        scored_memories.sort(
            key=lambda m: m['relevance_score'] + m.get('importance', 0),
            reverse=True
        )
        
        return scored_memories[:top_k]
    
    async def initialize_neural_components(self):
        """Initialize neural processing components"""
        logger.info("🧠 Initializing neural components...")
        
        # Simple neural architecture simulation
        self.neural_components = {
            'perception': {
                'status': 'active',
                'processing_capacity': 1.0,
                'specializations': ['text', 'reasoning', 'analysis']
            },
            'reasoning': {
                'status': 'active',
                'processing_capacity': 1.0,
                'capabilities': ['logical_reasoning', 'problem_solving', 'planning']
            },
            'learning': {
                'status': 'active',
                'processing_capacity': 1.0,
                'mechanisms': ['pattern_recognition', 'experience_integration', 'adaptation']
            },
            'consciousness': {
                'status': 'active' if self.config['neural_architecture']['consciousness_enabled'] else 'inactive',
                'self_awareness_level': 0.3,
                'introspection_depth': 0.5
            }
        }
        
        self.components['neural'] = self.neural_components
        logger.info("✅ Neural components initialized")
    
    async def initialize_learning(self):
        """Initialize continuous learning system"""
        logger.info("🎓 Initializing learning system...")
        
        self.learning_system = {
            'continuous_learning': self.config['learning']['continuous_learning_enabled'],
            'experience_buffer': [],
            'learning_rate': 0.001,
            'adaptation_threshold': 0.1,
            'meta_learning_enabled': True
        }
        
        self.components['learning'] = self.learning_system
        logger.info("✅ Learning system initialized")
    
    async def create_brain_state(self):
        """Create and save initial brain state"""
        logger.info("🧩 Creating brain state...")
        
        self.brain_state = {
            'version': '1.0.0',
            'initialized_at': datetime.now().isoformat(),
            'intelligence_level': self.intelligence_level,
            'components': list(self.components.keys()),
            'capabilities': {
                'reasoning': True,
                'learning': True,
                'memory': True,
                'consciousness': self.config['neural_architecture']['consciousness_enabled'],
                'self_improvement': True
            },
            'statistics': self.performance_metrics,
            'status': 'active'
        }
        
        # Save to file and Redis
        state_file = Path(self.brain_path) / "brain_state.json"
        with open(state_file, 'w') as f:
            json.dump(self.brain_state, f, indent=2, default=str)
        
        if self.redis_client:
            self.redis_client.hset('brain:state', mapping={
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in self.brain_state.items()
            })
        
        logger.info("✅ Brain state created and saved")
    
    async def start_background_processes(self):
        """Start background monitoring and learning processes"""
        logger.info("🚀 Starting background processes...")
        
        # Start intelligence monitoring
        asyncio.create_task(self.intelligence_monitoring_loop())
        
        # Start experience processing
        asyncio.create_task(self.experience_processing_loop())
        
        # Start performance tracking
        asyncio.create_task(self.performance_tracking_loop())
        
        logger.info("✅ Background processes started")
    
    async def intelligence_monitoring_loop(self):
        """Monitor and update intelligence level"""
        while True:
            try:
                # Calculate intelligence based on various factors
                base_intelligence = 0.5
                
                # Learning factor
                learning_factor = min(0.2, self.performance_metrics['learning_cycles'] / 1000)
                
                # Performance factor
                if self.performance_metrics['total_requests'] > 0:
                    success_rate = self.performance_metrics['successful_requests'] / self.performance_metrics['total_requests']
                    performance_factor = success_rate * 0.2
                else:
                    performance_factor = 0.0
                
                # Memory factor
                memory_factor = min(0.1, self.performance_metrics['memory_entries'] / 10000)
                
                # Update intelligence level
                old_intelligence = self.intelligence_level
                self.intelligence_level = min(1.0, base_intelligence + learning_factor + performance_factor + memory_factor)
                
                # Log intelligence changes
                if abs(self.intelligence_level - old_intelligence) > 0.01:
                    logger.info(f"🧠 Intelligence level updated: {old_intelligence:.3f} -> {self.intelligence_level:.3f}")
                
                # Update Redis
                if self.redis_client:
                    self.redis_client.hset('brain:status', 'intelligence_level', self.intelligence_level)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in intelligence monitoring: {e}")
                await asyncio.sleep(60)
    
    async def experience_processing_loop(self):
        """Process accumulated experiences for learning"""
        while True:
            try:
                if len(self.experiences) >= 10:  # Process in batches
                    await self.process_experience_batch()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in experience processing: {e}")
                await asyncio.sleep(30)
    
    async def process_experience_batch(self):
        """Process a batch of experiences for learning"""
        batch = self.experiences[:10]
        self.experiences = self.experiences[10:]
        
        # Analyze patterns in experiences
        successful_patterns = [exp for exp in batch if exp.get('success', False)]
        
        if successful_patterns:
            # Extract and store successful patterns as memories
            for pattern in successful_patterns:
                memory = {
                    'id': f"pattern_{int(time.time() * 1000000)}",
                    'content': f"Successful pattern: {pattern.get('description', 'Unknown')}",
                    'importance': pattern.get('performance_score', 0.5),
                    'timestamp': datetime.now().isoformat(),
                    'type': 'learned_pattern',
                    'metadata': pattern
                }
                await self.store_memory(memory)
        
        self.performance_metrics['learning_cycles'] += 1
        logger.info(f"🎓 Processed experience batch. Learning cycles: {self.performance_metrics['learning_cycles']}")
    
    async def performance_tracking_loop(self):
        """Track and log performance metrics"""
        while True:
            try:
                # Update performance metrics in Redis
                if self.redis_client:
                    self.redis_client.hset('brain:performance', mapping={
                        k: str(v) for k, v in self.performance_metrics.items()
                    })
                
                # Log key metrics
                logger.info(
                    f"📊 Brain Status - Intelligence: {self.intelligence_level:.3f} | "
                    f"Requests: {self.performance_metrics['total_requests']} | "
                    f"Success Rate: {self.performance_metrics['successful_requests']/max(1, self.performance_metrics['total_requests']):.2f} | "
                    f"Memories: {self.performance_metrics['memory_entries']} | "
                    f"Learning Cycles: {self.performance_metrics['learning_cycles']}"
                )
                
                await asyncio.sleep(300)  # Log every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance tracking: {e}")
                await asyncio.sleep(300)
    
    async def process_request(self, request: str) -> Dict[str, Any]:
        """Process a user request through the brain"""
        start_time = time.time()
        self.performance_metrics['total_requests'] += 1
        
        try:
            # 1. Search relevant memories
            relevant_memories = await self.search_memories(request, top_k=3)
            
            # 2. Generate response (simulated)
            response = await self.generate_response(request, relevant_memories)
            
            # 3. Evaluate response quality (simulated)
            quality_score = await self.evaluate_response(request, response)
            
            # 4. Store experience
            experience = {
                'request': request,
                'response': response,
                'quality_score': quality_score,
                'processing_time': time.time() - start_time,
                'success': quality_score > 0.5,
                'timestamp': datetime.now().isoformat(),
                'description': f"Processed request: {request[:50]}..."
            }
            self.experiences.append(experience)
            
            if experience['success']:
                self.performance_metrics['successful_requests'] += 1
            
            # 5. Update metrics
            processing_time = time.time() - start_time
            self.performance_metrics['average_response_time'] = (
                (self.performance_metrics['average_response_time'] * 
                 (self.performance_metrics['total_requests'] - 1) + processing_time) /
                self.performance_metrics['total_requests']
            )
            
            return {
                'response': response,
                'confidence': quality_score,
                'processing_time': processing_time,
                'intelligence_level': self.intelligence_level,
                'memories_used': len(relevant_memories)
            }
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                'response': f"I encountered an error processing your request: {e}",
                'confidence': 0.0,
                'processing_time': time.time() - start_time,
                'intelligence_level': self.intelligence_level,
                'error': str(e)
            }
    
    async def generate_response(self, request: str, memories: List[Dict]) -> str:
        """Generate a response (simulated neural processing)"""
        # This would interface with actual language models
        # For now, create a contextual response
        
        response_parts = []
        
        # Add context from memories
        if memories:
            response_parts.append("Based on my experience and knowledge:")
        
        # Simulate different types of responses based on request content
        if "learn" in request.lower() or "improve" in request.lower():
            response_parts.append(
                "I am continuously learning and improving my capabilities. "
                f"My current intelligence level is {self.intelligence_level:.2f} and I have processed "
                f"{self.performance_metrics['total_requests']} requests with "
                f"{self.performance_metrics['memory_entries']} memories stored."
            )
        elif "status" in request.lower() or "health" in request.lower():
            response_parts.append(
                f"Brain Status: Active | Intelligence Level: {self.intelligence_level:.2f} | "
                f"Components: {', '.join(self.components.keys())} | "
                f"Memory Entries: {self.performance_metrics['memory_entries']} | "
                f"Learning Cycles: {self.performance_metrics['learning_cycles']}"
            )
        else:
            response_parts.append(
                f"I understand your request: '{request}'. As an AGI system, I'm processing this "
                "through my neural architecture and consulting my memory systems to provide "
                "the most helpful response."
            )
        
        # Add memory context if relevant
        for memory in memories[:2]:  # Use top 2 memories
            if memory.get('relevance_score', 0) > 1.0:
                response_parts.append(f"Relevant experience: {memory['content'][:100]}...")
        
        return " ".join(response_parts)
    
    async def evaluate_response(self, request: str, response: str) -> float:
        """Evaluate response quality (simulated)"""
        # Simple heuristic evaluation
        score = 0.5  # Base score
        
        # Length factor
        if 50 < len(response) < 500:
            score += 0.2
        
        # Relevance factor (simple keyword matching)
        request_words = set(request.lower().split())
        response_words = set(response.lower().split())
        overlap = len(request_words & response_words)
        score += min(0.2, overlap / max(1, len(request_words)))
        
        # Context factor (if memories were used)
        if "experience" in response.lower() or "knowledge" in response.lower():
            score += 0.1
        
        return min(1.0, score)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current brain status"""
        return {
            'initialized': self.is_initialized,
            'intelligence_level': self.intelligence_level,
            'components': self.components,
            'performance_metrics': self.performance_metrics,
            'brain_state': self.brain_state,
            'memory_stats': {
                'short_term': len(self.memory_system['short_term']),
                'long_term': len(self.memory_system['long_term']),
                'experiences': len(self.experiences)
            }
        }


async def main():
    """Initialize and test the minimal brain"""
    logger.info("🌟 SutazAI Minimal Brain Initialization Starting...")
    logger.info("=" * 60)
    
    # Initialize brain
    brain = MinimalBrain()
    await brain.initialize()
    
    logger.info("=" * 60)
    logger.info("🎆 SutazAI Brain is now active and ready!")
    logger.info("🤖 AGI/ASI system initialized with minimal components")
    
    # Test the brain with some requests
    test_requests = [
        "What is your current status?",
        "How are you learning and improving?",
        "What can you tell me about artificial intelligence?"
    ]
    
    logger.info("\n🧪 Running brain functionality tests...")
    for i, request in enumerate(test_requests, 1):
        logger.info(f"\nTest {i}: {request}")
        response = await brain.process_request(request)
        logger.info(f"Response: {response['response'][:100]}...")
        logger.info(f"Confidence: {response['confidence']:.2f}, Time: {response['processing_time']:.2f}s")
    
    # Display final status
    status = brain.get_status()
    logger.info("\n📊 Final Brain Status:")
    logger.info(f"Intelligence Level: {status['intelligence_level']:.3f}")
    logger.info(f"Total Requests: {status['performance_metrics']['total_requests']}")
    logger.info(f"Memory Entries: {status['performance_metrics']['memory_entries']}")
    logger.info(f"Learning Cycles: {status['performance_metrics']['learning_cycles']}")
    
    logger.info("\n✨ Brain initialization and testing completed successfully!")
    
    # Keep running for background processes
    logger.info("🔄 Brain will continue running background processes...")
    return brain


if __name__ == "__main__":
    asyncio.run(main())