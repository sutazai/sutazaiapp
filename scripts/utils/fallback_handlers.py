#!/usr/bin/env python3
"""
Fallback Handlers for SutazAI Services

This module implements specific fallback handlers for different SutazAI services
to ensure graceful degradation when primary services are unavailable.

Author: SutazAI Infrastructure Team
Version: 1.0.0
"""

import asyncio
import json
import logging
import time
import sqlite3
import pickle
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import aiofiles
import hashlib

from feature_flags import (
    FeatureFlagManager, FallbackStrategy, DegradationLevel,
    feature_manager
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SutazAIFallbackHandlers:
    """
    Comprehensive fallback handlers for SutazAI services
    """
    
    def __init__(self, cache_dir: str = "/tmp/sutazai_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize local SQLite cache
        self.local_db = sqlite3.connect(
            self.cache_dir / "fallback_cache.db",
            check_same_thread=False
        )
        self._init_local_cache()
        
        # Register all fallback strategies
        self._register_fallback_strategies()
    
    def _init_local_cache(self):
        """Initialize local SQLite cache for fallbacks"""
        cursor = self.local_db.cursor()
        
        # Vector embeddings cache
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vector_cache (
                id TEXT PRIMARY KEY,
                content TEXT,
                embedding BLOB,
                metadata TEXT,
                created_at REAL
            )
        ''')
        
        # Chat completions cache
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_cache (
                query_hash TEXT PRIMARY KEY,
                prompt TEXT,
                response TEXT,
                model TEXT,
                created_at REAL
            )
        ''')
        
        # Memory cache
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                expires_at REAL
            )
        ''')
        
        # Agent state cache
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_state_cache (
                agent_id TEXT PRIMARY KEY,
                state BLOB,
                updated_at REAL
            )
        ''')
        
        self.local_db.commit()
    
    def _register_fallback_strategies(self):
        """Register all fallback strategies with the feature manager"""
        
        # AI Agent fallbacks
        feature_manager.register_fallback(
            "ai_agents_enabled",
            FallbackStrategy(
                name="offline_agent_processing",
                priority=1,
                handler=self.offline_agent_fallback,
                degradation_level=DegradationLevel.REDUCED,
                cache_ttl=300
            )
        )
        
        # Vector search fallbacks
        feature_manager.register_fallback(
            "vector_search",
            FallbackStrategy(
                name="keyword_search",
                priority=1,
                handler=self.keyword_search_fallback,
                degradation_level=DegradationLevel.REDUCED,
                cache_ttl=600
            )
        )
        
        feature_manager.register_fallback(
            "vector_search",
            FallbackStrategy(
                name="cached_vectors",
                priority=2,
                handler=self.cached_vector_fallback,
                degradation_level=DegradationLevel. ,
                cache_ttl=3600
            )
        )
        
        # Memory persistence fallbacks
        feature_manager.register_fallback(
            "memory_persistence",
            FallbackStrategy(
                name="local_file_storage",
                priority=1,
                handler=self.file_storage_fallback,
                degradation_level=DegradationLevel.REDUCED,
                cache_ttl=0  # No caching for storage operations
            )
        )
        
        # LLM processing fallbacks
        feature_manager.register_fallback(
            "letta_agent",
            FallbackStrategy(
                name="cached_responses",
                priority=1,
                handler=self.cached_llm_fallback,
                degradation_level=DegradationLevel. ,
                cache_ttl=1800
            )
        )
        
        # Real-time collaboration fallbacks
        feature_manager.register_fallback(
            "real_time_collaboration",
            FallbackStrategy(
                name="polling_updates",
                priority=1,
                handler=self.polling_collaboration_fallback,
                degradation_level=DegradationLevel.REDUCED,
                cache_ttl=60
            )
        )
        
        logger.info("Registered all fallback strategies")
    
    async def offline_agent_fallback(self, *args, **kwargs) -> Dict[str, Any]:
        """Fallback for AI agent processing when agents are unavailable"""
        agent_request = kwargs.get('request', {})
        prompt = agent_request.get('prompt', '')
        agent_type = agent_request.get('agent_type', 'general')
        
        # Check cache first
        query_hash = hashlib.md5(f"{agent_type}:{prompt}".encode()).hexdigest()
        cached_response = self._get_cached_chat_completion(query_hash)
        
        if cached_response:
            logger.info("Returning cached AI agent response")
            return {
                'response': cached_response,
                'source': 'cache',
                'degraded': True,
                'agent_type': agent_type
            }
        
        # Generate simple rule-based response
        response = self._generate_rule_based_response(prompt, agent_type)
        
        # Cache the response
        self._cache_chat_completion(query_hash, prompt, response, agent_type)
        
        return {
            'response': response,
            'source': 'rule_based',
            'degraded': True,
            'agent_type': agent_type,
            'timestamp': time.time()
        }
    
    def _generate_rule_based_response(self, prompt: str, agent_type: str) -> str:
        """Generate rule-based response when AI is unavailable"""
        prompt_lower = prompt.lower()
        
        # Define response templates by agent type
        templates = {
            'general': {
                'greeting': ["hello", "hi", "hey", "good morning", "good afternoon"],
                'help': ["help", "assist", "support", "how to"],
                'status': ["status", "health", "how are", "working"],
                'goodbye': ["bye", "goodbye", "see you", "farewell"]
            },
            'coding': {
                'debug': ["error", "bug", "debug", "fix", "problem"],
                'explain': ["explain", "what is", "how does", "understand"],
                'generate': ["create", "generate", "write", "build", "make"]
            },
            'data': {
                'analyze': ["analyze", "analysis", "examine", "study"],
                'visualize': ["plot", "chart", "graph", "visualize", "show"],
                'summarize': ["summary", "summarize", "overview", "brief"]
            }
        }
        
        responses = {
            'general': {
                'greeting': "Hello! I'm currently running in offline mode with limited capabilities.",
                'help': "I'm operating in degraded mode. I can provide basic assistance with cached responses.",
                'status': "System is running in degraded mode. Some services are temporarily unavailable.",
                'goodbye': "Goodbye! I hope to be back to full functionality soon.",
                'default': "I'm currently in offline mode with limited capabilities. Please try again later when full services are restored."
            },
            'coding': {
                'debug': "I'm in offline mode and cannot debug code right now. Please check logs and documentation.",
                'explain': "Code explanation is limited in offline mode. Please refer to cached documentation.",
                'generate': "Code generation is unavailable in offline mode. Please try again when services are restored.",
                'default': "Coding assistance is limited in offline mode. Basic cached responses only."
            },
            'data': {
                'analyze': "Data analysis is limited in offline mode. Only cached results are available.",
                'visualize': "Data visualization is unavailable in offline mode.",
                'summarize': "Data summarization is limited in offline mode.",
                'default': "Data services are limited in offline mode."
            }
        }
        
        # Get appropriate template and response sets
        template_set = templates.get(agent_type, templates['general'])
        response_set = responses.get(agent_type, responses['general'])
        
        # Find matching template
        for category, keywords in template_set.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return response_set.get(category, response_set['default'])
        
        return response_set['default']
    
    async def keyword_search_fallback(self, *args, **kwargs) -> Dict[str, Any]:
        """Keyword-based search fallback when vector search is unavailable"""
        query = kwargs.get('query', '')
        documents = kwargs.get('documents', [])
        
        if not query or not documents:
            return {'results': [], 'degraded': True, 'method': 'keyword_search'}
        
        # Simple keyword matching
        query_words = set(query.lower().split())
        results = []
        
        for i, doc in enumerate(documents):
            content = str(doc).lower()
            doc_words = set(content.split())
            
            # Calculate simple keyword overlap score
            overlap = len(query_words.intersection(doc_words))
            if overlap > 0:
                score = overlap / len(query_words)
                results.append({
                    'id': i,
                    'content': doc,
                    'score': score,
                    'method': 'keyword_match'
                })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'results': results[:10],  # Top 10 results
            'degraded': True,
            'method': 'keyword_search',
            'total_matches': len(results)
        }
    
    async def cached_vector_fallback(self, *args, **kwargs) -> Dict[str, Any]:
        """Return cached vector search results"""
        query = kwargs.get('query', '')
        
        # Simple hash-based cache lookup
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        cursor = self.local_db.cursor()
        cursor.execute(
            'SELECT content, metadata FROM vector_cache WHERE id = ?',
            (query_hash,)
        )
        
        cached_results = cursor.fetchall()
        
        if cached_results:
            results = []
            for content, metadata_json in cached_results:
                metadata = json.loads(metadata_json) if metadata_json else {}
                results.append({
                    'content': content,
                    'metadata': metadata,
                    'source': 'cache',
                    'score': 0.5  # Default cache score
                })
            
            return {
                'results': results,
                'degraded': True,
                'method': 'cached_vectors',
                'cache_hit': True
            }
        
        return {
            'results': [],
            'degraded': True,
            'method': 'cached_vectors',
            'cache_hit': False
        }
    
    async def file_storage_fallback(self, *args, **kwargs) -> Dict[str, Any]:
        """File-based storage fallback when database is unavailable"""
        operation = kwargs.get('operation', 'get')
        key = kwargs.get('key', '')
        value = kwargs.get('value', None)
        
        file_path = self.cache_dir / f"memory_{hashlib.md5(key.encode()).hexdigest()}.json"
        
        try:
            if operation == 'set':
                data = {
                    'key': key,
                    'value': value,
                    'timestamp': time.time()
                }
                
                async with aiofiles.open(file_path, 'w') as f:
                    await f.write(json.dumps(data))
                
                return {
                    'success': True,
                    'method': 'file_storage',
                    'degraded': True
                }
            
            elif operation == 'get':
                if file_path.exists():
                    async with aiofiles.open(file_path, 'r') as f:
                        content = await f.read()
                        data = json.loads(content)
                    
                    return {
                        'value': data.get('value'),
                        'found': True,
                        'method': 'file_storage',
                        'degraded': True,
                        'timestamp': data.get('timestamp')
                    }
                else:
                    return {
                        'value': None,
                        'found': False,
                        'method': 'file_storage',
                        'degraded': True
                    }
            
        except Exception as e:
            logger.error(f"File storage fallback error: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': 'file_storage',
                'degraded': True
            }
    
    async def cached_llm_fallback(self, *args, **kwargs) -> Dict[str, Any]:
        """Return cached LLM responses when models are unavailable"""
        prompt = kwargs.get('prompt', '')
        model = kwargs.get('model', 'default')
        
        query_hash = hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()
        cached_response = self._get_cached_chat_completion(query_hash)
        
        if cached_response:
            return {
                'response': cached_response,
                'model': model,
                'source': 'cache',
                'degraded': True,
                'timestamp': time.time()
            }
        
        # If no cache, return a helpful message
        return {
            'response': "I'm currently offline and don't have a cached response for this query. Please try again when the AI models are available.",
            'model': model,
            'source': 'offline_message',
            'degraded': True,
            'timestamp': time.time()
        }
    
    async def polling_collaboration_fallback(self, *args, **kwargs) -> Dict[str, Any]:
        """Polling-based collaboration when real-time features are unavailable"""
        action = kwargs.get('action', 'get_updates')
        user_id = kwargs.get('user_id', '')
        data = kwargs.get('data', {})
        
        updates_file = self.cache_dir / "collaboration_updates.json"
        
        try:
            if action == 'send_update':
                # Append update to file
                update = {
                    'user_id': user_id,
                    'data': data,
                    'timestamp': time.time()
                }
                
                # Read existing updates
                updates = []
                if updates_file.exists():
                    async with aiofiles.open(updates_file, 'r') as f:
                        content = await f.read()
                        if content.strip():
                            updates = json.loads(content)
                
                # Add new update
                updates.append(update)
                
                # Keep only last 100 updates
                updates = updates[-100:]
                
                # Write back
                async with aiofiles.open(updates_file, 'w') as f:
                    await f.write(json.dumps(updates))
                
                return {
                    'success': True,
                    'method': 'polling_collaboration',
                    'degraded': True
                }
            
            elif action == 'get_updates':
                since_timestamp = kwargs.get('since', 0)
                
                if updates_file.exists():
                    async with aiofiles.open(updates_file, 'r') as f:
                        content = await f.read()
                        if content.strip():
                            all_updates = json.loads(content)
                            
                            # Filter updates since timestamp
                            new_updates = [
                                update for update in all_updates
                                if update['timestamp'] > since_timestamp
                            ]
                            
                            return {
                                'updates': new_updates,
                                'method': 'polling_collaboration',
                                'degraded': True,
                                'poll_required': True
                            }
                
                return {
                    'updates': [],
                    'method': 'polling_collaboration',
                    'degraded': True,
                    'poll_required': True
                }
                
        except Exception as e:
            logger.error(f"Polling collaboration fallback error: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': 'polling_collaboration',
                'degraded': True
            }
    
    def _get_cached_chat_completion(self, query_hash: str) -> Optional[str]:
        """Get cached chat completion response"""
        cursor = self.local_db.cursor()
        cursor.execute(
            'SELECT response FROM chat_cache WHERE query_hash = ? AND created_at > ?',
            (query_hash, time.time() - 3600)  # 1 hour cache
        )
        
        result = cursor.fetchone()
        return result[0] if result else None
    
    def _cache_chat_completion(self, query_hash: str, prompt: str, response: str, model: str):
        """Cache chat completion response"""
        cursor = self.local_db.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO chat_cache (query_hash, prompt, response, model, created_at) VALUES (?, ?, ?, ?, ?)',
            (query_hash, prompt, response, model, time.time())
        )
        self.local_db.commit()
    
    async def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cache entries"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        cursor = self.local_db.cursor()
        
        # Clean up old entries
        cursor.execute('DELETE FROM vector_cache WHERE created_at < ?', (cutoff_time,))
        cursor.execute('DELETE FROM chat_cache WHERE created_at < ?', (cutoff_time,))
        cursor.execute('DELETE FROM memory_cache WHERE expires_at < ?', (time.time(),))
        cursor.execute('DELETE FROM agent_state_cache WHERE updated_at < ?', (cutoff_time,))
        
        self.local_db.commit()
        
        logger.info(f"Cleaned up cache entries older than {max_age_hours} hours")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        cursor = self.local_db.cursor()
        
        stats = {}
        
        # Count entries in each table
        tables = ['vector_cache', 'chat_cache', 'memory_cache', 'agent_state_cache']
        for table in tables:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            count = cursor.fetchone()[0]
            stats[table] = count
        
        # Get cache directory size
        total_size = sum(
            f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file()
        )
        stats['total_cache_size_bytes'] = total_size
        stats['total_cache_size_mb'] = total_size / (1024 * 1024)
        
        return stats

# Global fallback handlers instance
fallback_handlers = SutazAIFallbackHandlers()

# Convenience functions for easy integration
async def execute_with_ai_fallback(prompt: str, agent_type: str = 'general') -> Dict[str, Any]:
    """Execute AI request with fallback"""
    return await feature_manager.execute_with_fallback(
        "ai_agents_enabled",
        lambda: None,  # Primary function would be actual AI call
        request={'prompt': prompt, 'agent_type': agent_type}
    )

async def search_with_fallback(query: str, documents: List[str] = None) -> Dict[str, Any]:
    """Execute search with fallback"""
    documents = documents or []
    return await feature_manager.execute_with_fallback(
        "vector_search",
        lambda: None,  # Primary function would be vector search
        query=query,
        documents=documents
    )

async def store_with_fallback(key: str, value: Any) -> Dict[str, Any]:
    """Store data with fallback"""
    return await feature_manager.execute_with_fallback(
        "memory_persistence",
        lambda: None,  # Primary function would be database storage
        operation='set',
        key=key,
        value=value
    )

async def retrieve_with_fallback(key: str) -> Dict[str, Any]:
    """Retrieve data with fallback"""
    return await feature_manager.execute_with_fallback(
        "memory_persistence",
        lambda: None,  # Primary function would be database retrieval
        operation='get',
        key=key
    )

# Example usage
async def example_usage():
    """Example usage of fallback handlers"""
    
    # AI agent fallback
    result = await execute_with_ai_fallback(
        "Hello, can you help me?",
        "general"
    )
    print(f"AI Response: {result}")
    
    # Search fallback
    search_result = await search_with_fallback(
        "machine learning",
        ["AI and machine learning", "deep learning models", "natural language processing"]
    )
    print(f"Search Result: {search_result}")
    
    # Storage fallback
    store_result = await store_with_fallback(
        "user_preference",
        {"theme": "dark", "language": "en"}
    )
    print(f"Store Result: {store_result}")
    
    # Retrieval fallback
    retrieve_result = await retrieve_with_fallback("user_preference")
    print(f"Retrieve Result: {retrieve_result}")
    
    # Get cache statistics
    cache_stats = fallback_handlers.get_cache_stats()
    print(f"Cache Stats: {cache_stats}")

if __name__ == "__main__":
    asyncio.run(example_usage())