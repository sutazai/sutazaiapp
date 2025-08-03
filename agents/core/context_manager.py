#!/usr/bin/env python3
"""
Purpose: Manage conversation context efficiently across different model context windows
Usage: from agents.core.context_manager import ContextManager
Requirements: numpy, asyncio, json
"""

import json
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
import logging
import asyncio
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Single message in conversation"""
    role: str  # system, user, assistant
    content: str
    timestamp: float = field(default_factory=time.time)
    tokens: int = 0
    importance: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextWindow:
    """Configuration for model context windows"""
    model: str
    max_tokens: int
    optimal_tokens: int  # Leave room for response
    compression_threshold: float  # When to start compressing


class ContextManager:
    """Manage conversation context for optimal model performance"""
    
    # Model context configurations
    CONTEXT_CONFIGS = {
        'tinyllama': ContextWindow(
            model='tinyllama',
            max_tokens=2048,
            optimal_tokens=1500,
            compression_threshold=0.75
        ),
        'qwen2.5-coder:7b': ContextWindow(
            model='qwen2.5-coder:7b',
            max_tokens=32768,
            optimal_tokens=8192,
            compression_threshold=0.8
        ),
        'deepseek-r1:8b': ContextWindow(
            model='deepseek-r1:8b',
            max_tokens=32768,
            optimal_tokens=16384,
            compression_threshold=0.85
        )
    }
    
    def __init__(self, model: str = 'tinyllama'):
        self.model = model
        self.config = self.CONTEXT_CONFIGS.get(model, self.CONTEXT_CONFIGS['tinyllama'])
        self.messages: List[Message] = []
        self.summary_cache: Dict[str, str] = {}
        self.importance_scorer = ImportanceScorer()
        self.token_counter = SimpleTokenCounter()
        
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to the context"""
        tokens = self.token_counter.count(content)
        importance = self.importance_scorer.score(role, content, metadata)
        
        message = Message(
            role=role,
            content=content,
            tokens=tokens,
            importance=importance,
            metadata=metadata or {}
        )
        
        self.messages.append(message)
        logger.debug(f"Added {role} message: {tokens} tokens, importance: {importance:.2f}")
        
    def get_optimized_context(self) -> List[Dict[str, str]]:
        """Get optimized context that fits within model limits"""
        if not self.messages:
            return []
        
        # Calculate current token usage
        total_tokens = sum(msg.tokens for msg in self.messages)
        
        # If within limits, return all messages
        if total_tokens <= self.config.optimal_tokens:
            return self._messages_to_dict(self.messages)
        
        # Need to optimize
        logger.info(f"Optimizing context: {total_tokens} tokens -> {self.config.optimal_tokens}")
        
        # Try different strategies
        optimized_messages = self._apply_optimization_strategies(total_tokens)
        
        return self._messages_to_dict(optimized_messages)
    
    def _apply_optimization_strategies(self, current_tokens: int) -> List[Message]:
        """Apply various strategies to fit context within limits"""
        # Strategy 1: Remove low-importance messages
        if current_tokens < self.config.max_tokens * 1.5:
            return self._remove_low_importance_messages()
        
        # Strategy 2: Summarize old messages
        if current_tokens < self.config.max_tokens * 2:
            return self._summarize_old_messages()
        
        # Strategy 3: Aggressive compression
        return self._aggressive_compression()
    
    def _remove_low_importance_messages(self) -> List[Message]:
        """Remove messages with low importance scores"""
        # Always keep system message and recent messages
        kept_messages = []
        
        # Keep system message
        system_messages = [msg for msg in self.messages if msg.role == 'system']
        if system_messages:
            kept_messages.append(system_messages[0])
        
        # Sort by importance and recency
        scored_messages = []
        for i, msg in enumerate(self.messages):
            if msg.role == 'system':
                continue
            
            # Boost recent messages
            recency_boost = 1.0 + (i / len(self.messages)) * 0.5
            score = msg.importance * recency_boost
            scored_messages.append((score, msg))
        
        # Sort by score
        scored_messages.sort(key=lambda x: x[0], reverse=True)
        
        # Add messages until we reach limit
        current_tokens = sum(msg.tokens for msg in kept_messages)
        
        for score, msg in scored_messages:
            if current_tokens + msg.tokens <= self.config.optimal_tokens:
                kept_messages.append(msg)
                current_tokens += msg.tokens
            else:
                # Try to at least keep a summary
                if current_tokens + 50 < self.config.optimal_tokens:
                    summary = self._create_summary([msg], max_tokens=50)
                    if summary:
                        kept_messages.append(Message(
                            role='system',
                            content=f"[Summary: {summary}]",
                            tokens=50,
                            importance=msg.importance * 0.7
                        ))
                        current_tokens += 50
        
        # Sort by timestamp to maintain order
        kept_messages.sort(key=lambda x: x.timestamp)
        
        return kept_messages
    
    def _summarize_old_messages(self) -> List[Message]:
        """Summarize older messages to save space"""
        if len(self.messages) < 10:
            return self._remove_low_importance_messages()
        
        # Keep system and recent messages
        kept_messages = []
        
        # System message
        system_messages = [msg for msg in self.messages if msg.role == 'system']
        if system_messages:
            kept_messages.append(system_messages[0])
        
        # Determine cutoff for "old" messages (first 50%)
        cutoff_index = len(self.messages) // 2
        old_messages = self.messages[1:cutoff_index]  # Skip system message
        recent_messages = self.messages[cutoff_index:]
        
        # Summarize old messages by role
        if old_messages:
            summary = self._create_conversation_summary(old_messages)
            summary_msg = Message(
                role='system',
                content=f"[Previous conversation summary: {summary}]",
                tokens=self.token_counter.count(summary),
                importance=0.8
            )
            kept_messages.append(summary_msg)
        
        # Add recent messages
        kept_messages.extend(recent_messages)
        
        # Check if still too large
        total_tokens = sum(msg.tokens for msg in kept_messages)
        if total_tokens > self.config.optimal_tokens:
            # Apply importance filtering to recent messages
            return self._filter_by_importance(kept_messages)
        
        return kept_messages
    
    def _aggressive_compression(self) -> List[Message]:
        """Aggressively compress context for very long conversations"""
        kept_messages = []
        
        # Always keep system message
        system_messages = [msg for msg in self.messages if msg.role == 'system']
        if system_messages:
            kept_messages.append(system_messages[0])
        
        # Create a comprehensive summary
        summary = self._create_aggressive_summary()
        summary_msg = Message(
            role='system',
            content=summary,
            tokens=self.token_counter.count(summary),
            importance=0.9
        )
        kept_messages.append(summary_msg)
        
        # Only keep very recent and very important messages
        recent_count = min(5, len(self.messages) // 10)
        recent_messages = self.messages[-recent_count:]
        
        # Add most recent messages
        kept_messages.extend(recent_messages)
        
        return kept_messages
    
    def _create_summary(self, messages: List[Message], max_tokens: int = 200) -> str:
        """Create a summary of messages"""
        # Simple extractive summary
        key_points = []
        
        for msg in messages:
            # Extract key sentences
            sentences = msg.content.split('. ')
            if sentences:
                # Take first and most important sentence
                key_points.append(sentences[0][:100])
        
        summary = '; '.join(key_points)
        
        # Trim to token limit
        while self.token_counter.count(summary) > max_tokens and key_points:
            key_points.pop()
            summary = '; '.join(key_points)
        
        return summary
    
    def _create_conversation_summary(self, messages: List[Message]) -> str:
        """Create a summary of a conversation segment"""
        # Group by role
        user_points = []
        assistant_points = []
        
        for msg in messages:
            if msg.role == 'user':
                # Extract main request/question
                first_sentence = msg.content.split('.')[0].split('?')[0][:100]
                if first_sentence and first_sentence not in user_points:
                    user_points.append(first_sentence)
            elif msg.role == 'assistant':
                # Extract main response
                first_sentence = msg.content.split('.')[0][:100]
                if first_sentence and first_sentence not in assistant_points:
                    assistant_points.append(first_sentence)
        
        summary_parts = []
        
        if user_points:
            summary_parts.append(f"User asked about: {', '.join(user_points[:3])}")
        
        if assistant_points:
            summary_parts.append(f"Assistant helped with: {', '.join(assistant_points[:3])}")
        
        return '. '.join(summary_parts)
    
    def _create_aggressive_summary(self) -> str:
        """Create an aggressive summary for very long conversations"""
        # Extract main topics and outcomes
        topics = set()
        outcomes = set()
        
        for msg in self.messages:
            if msg.role == 'user':
                # Extract topics (nouns, questions)
                words = msg.content.lower().split()
                for word in words:
                    if len(word) > 5 and word.isalpha():
                        topics.add(word)
            elif msg.role == 'assistant':
                # Extract outcomes (completed, fixed, created, etc.)
                if any(keyword in msg.content.lower() for keyword in ['completed', 'fixed', 'created', 'implemented']):
                    sentences = msg.content.split('.')
                    if sentences:
                        outcomes.add(sentences[0][:100])
        
        summary = f"Conversation covered: {', '.join(list(topics)[:5])}. "
        if outcomes:
            summary += f"Completed tasks: {'; '.join(list(outcomes)[:3])}"
        
        return summary[:500]  # Hard limit
    
    def _filter_by_importance(self, messages: List[Message]) -> List[Message]:
        """Filter messages by importance to fit within token limit"""
        # Calculate importance threshold dynamically
        total_tokens = sum(msg.tokens for msg in messages)
        compression_ratio = self.config.optimal_tokens / total_tokens
        
        # Sort by importance
        sorted_messages = sorted(messages, key=lambda x: x.importance, reverse=True)
        
        kept_messages = []
        current_tokens = 0
        
        for msg in sorted_messages:
            if current_tokens + msg.tokens <= self.config.optimal_tokens:
                kept_messages.append(msg)
                current_tokens += msg.tokens
        
        # Restore chronological order
        kept_messages.sort(key=lambda x: x.timestamp)
        
        return kept_messages
    
    def _messages_to_dict(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Convert Message objects to dict format for API"""
        return [
            {
                'role': msg.role,
                'content': msg.content
            }
            for msg in messages
        ]
    
    def get_token_usage(self) -> Dict[str, int]:
        """Get current token usage statistics"""
        total_tokens = sum(msg.tokens for msg in self.messages)
        
        return {
            'total_tokens': total_tokens,
            'message_count': len(self.messages),
            'max_tokens': self.config.max_tokens,
            'optimal_tokens': self.config.optimal_tokens,
            'usage_percentage': (total_tokens / self.config.max_tokens) * 100
        }
    
    def clear_old_messages(self, max_age_hours: int = 24) -> int:
        """Clear messages older than specified hours"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        original_count = len(self.messages)
        
        self.messages = [
            msg for msg in self.messages 
            if msg.timestamp > cutoff_time or msg.role == 'system'
        ]
        
        removed = original_count - len(self.messages)
        if removed > 0:
            logger.info(f"Cleared {removed} old messages")
        
        return removed
    
    def export_context(self) -> Dict[str, Any]:
        """Export context for persistence"""
        return {
            'model': self.model,
            'messages': [
                {
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp,
                    'tokens': msg.tokens,
                    'importance': msg.importance,
                    'metadata': msg.metadata
                }
                for msg in self.messages
            ],
            'summary_cache': self.summary_cache
        }
    
    def import_context(self, data: Dict[str, Any]) -> None:
        """Import previously exported context"""
        self.model = data.get('model', self.model)
        self.config = self.CONTEXT_CONFIGS.get(self.model, self.CONTEXT_CONFIGS['tinyllama'])
        
        self.messages = []
        for msg_data in data.get('messages', []):
            self.messages.append(Message(**msg_data))
        
        self.summary_cache = data.get('summary_cache', {})


class ImportanceScorer:
    """Score message importance for context optimization"""
    
    def __init__(self):
        self.keywords = {
            'high': ['error', 'bug', 'critical', 'security', 'fix', 'urgent', 'important'],
            'medium': ['need', 'should', 'must', 'require', 'problem', 'issue'],
            'low': ['maybe', 'might', 'could', 'perhaps', 'info', 'fyi']
        }
        
    def score(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        """Score importance from 0.0 to 1.0"""
        score = 0.5  # Base score
        
        # Role-based scoring
        if role == 'system':
            score = 0.9  # System messages are important
        elif role == 'user':
            score = 0.7  # User messages are generally important
        else:
            score = 0.6  # Assistant messages
        
        # Content-based scoring
        content_lower = content.lower()
        
        # High importance keywords
        for keyword in self.keywords['high']:
            if keyword in content_lower:
                score = min(1.0, score + 0.2)
        
        # Medium importance keywords
        for keyword in self.keywords['medium']:
            if keyword in content_lower:
                score = min(1.0, score + 0.1)
        
        # Low importance keywords
        for keyword in self.keywords['low']:
            if keyword in content_lower:
                score = max(0.2, score - 0.1)
        
        # Length factor (very short or very long messages may be less important)
        word_count = len(content.split())
        if word_count < 5:
            score *= 0.8
        elif word_count > 500:
            score *= 0.9
        
        # Metadata factors
        if metadata:
            if metadata.get('is_error', False):
                score = min(1.0, score + 0.3)
            if metadata.get('is_code', False):
                score = min(1.0, score + 0.1)
            if metadata.get('is_summary', False):
                score = min(1.0, score + 0.2)
        
        return round(score, 2)


class SimpleTokenCounter:
    """Simple token counter for estimation"""
    
    def count(self, text: str) -> int:
        """Estimate token count (simple implementation)"""
        # Average of 4 characters per token
        # More sophisticated: use tiktoken or model-specific tokenizer
        return len(text) // 4


class ContextWindowManager:
    """Manage multiple context windows for different conversations"""
    
    def __init__(self, default_model: str = 'tinyllama'):
        self.default_model = default_model
        self.contexts: Dict[str, ContextManager] = {}
        self.max_contexts = 100
        
    def get_context(self, conversation_id: str, model: Optional[str] = None) -> ContextManager:
        """Get or create context for a conversation"""
        if conversation_id not in self.contexts:
            # Create new context
            model = model or self.default_model
            self.contexts[conversation_id] = ContextManager(model)
            
            # Clean up old contexts if needed
            if len(self.contexts) > self.max_contexts:
                self._cleanup_old_contexts()
        
        return self.contexts[conversation_id]
    
    def _cleanup_old_contexts(self) -> None:
        """Remove oldest contexts to maintain memory limits"""
        # Sort by last message timestamp
        sorted_contexts = sorted(
            self.contexts.items(),
            key=lambda x: x[1].messages[-1].timestamp if x[1].messages else 0
        )
        
        # Remove oldest 20%
        remove_count = len(self.contexts) // 5
        for conv_id, _ in sorted_contexts[:remove_count]:
            del self.contexts[conv_id]
        
        logger.info(f"Cleaned up {remove_count} old contexts")
    
    def clear_context(self, conversation_id: str) -> None:
        """Clear a specific conversation context"""
        if conversation_id in self.contexts:
            del self.contexts[conversation_id]


# Example usage
if __name__ == "__main__":
    # Test context management
    manager = ContextManager('tinyllama')
    
    # Add some messages
    manager.add_message('system', 'You are a helpful assistant.')
    manager.add_message('user', 'I need help with a critical security bug in my authentication system.')
    manager.add_message('assistant', 'I can help you fix the security issue. What specific problem are you encountering?')
    
    # Add many messages to test compression
    for i in range(50):
        manager.add_message('user', f'Question {i}: How do I implement feature {i}?')
        manager.add_message('assistant', f'Answer {i}: Here is how you implement feature {i}...')
    
    # Get optimized context
    optimized = manager.get_optimized_context()
    
    print(f"Original messages: {len(manager.messages)}")
    print(f"Optimized messages: {len(optimized)}")
    print(f"Token usage: {manager.get_token_usage()}")
    
    # Show optimized context
    print("\nOptimized context:")
    for msg in optimized[:5]:  # Show first 5
        print(f"{msg['role']}: {msg['content'][:100]}...")