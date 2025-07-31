#!/usr/bin/env python3
"""
Context Optimization Engineer Agent
Manages token usage and context optimization for AI models
"""

import os
import time
import logging
import requests
import tiktoken
from flask import Flask, jsonify, request
from threading import Thread
import schedule
import json
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class ContextOptimizationEngineer:
    def __init__(self):
        self.openai_api_base = os.getenv('OPENAI_API_BASE', 'http://litellm:4000')
        self.openai_api_key = os.getenv('OPENAI_API_KEY', 'local')
        self.max_context_length = int(os.getenv('MAX_CONTEXT_LENGTH', 4096))
        self.optimization_level = os.getenv('OPTIMIZATION_LEVEL', 'balanced')
        
        # Token usage tracking
        self.token_usage = defaultdict(lambda: {'input': 0, 'output': 0, 'requests': 0})
        self.context_cache = {}
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {e}")
            self.tokenizer = None
    
    def count_tokens(self, text):
        """Count tokens in text"""
        if not self.tokenizer:
            # Fallback estimation: ~4 chars per token
            return len(text) // 4
        
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.error(f"Token counting error: {e}")
            return len(text) // 4
    
    def optimize_context(self, messages, target_length=None):
        """Optimize context to fit within token limits"""
        if target_length is None:
            target_length = self.max_context_length
        
        if not messages:
            return messages
        
        # Calculate current token count
        total_tokens = sum(self.count_tokens(str(msg)) for msg in messages)
        
        if total_tokens <= target_length:
            return messages
        
        logger.info(f"Optimizing context: {total_tokens} -> {target_length} tokens")
        
        # Optimization strategies based on level
        if self.optimization_level == 'aggressive':
            return self._aggressive_optimization(messages, target_length)
        elif self.optimization_level == 'conservative':
            return self._conservative_optimization(messages, target_length)
        else:  # balanced
            return self._balanced_optimization(messages, target_length)
    
    def _aggressive_optimization(self, messages, target_length):
        """Aggressive context optimization - maximum compression"""
        optimized = []
        current_tokens = 0
        
        # Keep only system message and most recent messages
        for msg in reversed(messages):
            msg_tokens = self.count_tokens(str(msg))
            
            if current_tokens + msg_tokens <= target_length:
                optimized.insert(0, msg)
                current_tokens += msg_tokens
            elif msg.get('role') == 'system':
                # Always keep system message, truncate if necessary
                truncated = self._truncate_message(msg, target_length - current_tokens)
                optimized.insert(0, truncated)
                break
        
        return optimized
    
    def _conservative_optimization(self, messages, target_length):
        """Conservative optimization - minimal changes"""
        if not messages:
            return messages
        
        # Simple truncation from the beginning, keeping recent context
        total_tokens = sum(self.count_tokens(str(msg)) for msg in messages)
        
        if total_tokens <= target_length:
            return messages
        
        # Remove oldest messages until we fit
        optimized = messages.copy()
        while optimized and sum(self.count_tokens(str(msg)) for msg in optimized) > target_length:
            # Keep system message if it exists
            if optimized[0].get('role') == 'system' and len(optimized) > 1:
                optimized.pop(1)
            else:
                optimized.pop(0)
        
        return optimized
    
    def _balanced_optimization(self, messages, target_length):
        """Balanced optimization - smart compression"""
        if not messages:
            return messages
        
        # Separate system messages from conversation
        system_msgs = [msg for msg in messages if msg.get('role') == 'system']
        conversation_msgs = [msg for msg in messages if msg.get('role') != 'system']
        
        # Calculate tokens for system messages
        system_tokens = sum(self.count_tokens(str(msg)) for msg in system_msgs)
        available_tokens = target_length - system_tokens
        
        if available_tokens <= 0:
            # Truncate system messages if they're too long
            system_msgs = [self._truncate_message(system_msgs[0], target_length)] if system_msgs else []
            return system_msgs
        
        # Optimize conversation messages
        optimized_conversation = []
        current_tokens = 0
        
        # Keep recent messages first
        for msg in reversed(conversation_msgs):
            msg_tokens = self.count_tokens(str(msg))
            
            if current_tokens + msg_tokens <= available_tokens:
                optimized_conversation.insert(0, msg)
                current_tokens += msg_tokens
            else:
                # Try to fit a truncated version
                remaining_tokens = available_tokens - current_tokens
                if remaining_tokens > 50:  # Only if we have reasonable space
                    truncated = self._truncate_message(msg, remaining_tokens)
                    optimized_conversation.insert(0, truncated)
                break
        
        return system_msgs + optimized_conversation
    
    def _truncate_message(self, message, max_tokens):
        """Truncate a message to fit within token limit"""
        if isinstance(message, dict):
            content = str(message.get('content', ''))
            if self.count_tokens(content) <= max_tokens:
                return message
            
            # Binary search for optimal truncation point
            left, right = 0, len(content)
            best_length = 0
            
            while left <= right:
                mid = (left + right) // 2
                truncated = content[:mid] + "..."
                
                if self.count_tokens(truncated) <= max_tokens:
                    best_length = mid
                    left = mid + 1
                else:
                    right = mid - 1
            
            truncated_message = message.copy()
            truncated_message['content'] = content[:best_length] + "..."
            return truncated_message
        
        return message
    
    def track_token_usage(self, model, input_tokens, output_tokens):
        """Track token usage for analytics"""
        self.token_usage[model]['input'] += input_tokens
        self.token_usage[model]['output'] += output_tokens
        self.token_usage[model]['requests'] += 1
    
    def get_usage_stats(self):
        """Get token usage statistics"""
        total_stats = {
            'total_input_tokens': sum(stats['input'] for stats in self.token_usage.values()),
            'total_output_tokens': sum(stats['output'] for stats in self.token_usage.values()),
            'total_requests': sum(stats['requests'] for stats in self.token_usage.values()),
            'models': dict(self.token_usage)
        }
        return total_stats
    
    def monitor_context_efficiency(self):
        """Monitor and log context optimization efficiency"""
        stats = self.get_usage_stats()
        logger.info(f"Token usage - Input: {stats['total_input_tokens']}, "
                   f"Output: {stats['total_output_tokens']}, "
                   f"Requests: {stats['total_requests']}")

# Global instance
optimizer = ContextOptimizationEngineer()

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'max_context_length': optimizer.max_context_length,
        'optimization_level': optimizer.optimization_level,
        'timestamp': time.time()
    })

@app.route('/optimize', methods=['POST'])
def optimize_context():
    """Optimize context for given messages"""
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        target_length = data.get('target_length', optimizer.max_context_length)
        
        optimized_messages = optimizer.optimize_context(messages, target_length)
        
        original_tokens = sum(optimizer.count_tokens(str(msg)) for msg in messages)
        optimized_tokens = sum(optimizer.count_tokens(str(msg)) for msg in optimized_messages)
        
        return jsonify({
            'status': 'success',
            'original_messages': len(messages),
            'optimized_messages': len(optimized_messages),
            'original_tokens': original_tokens,
            'optimized_tokens': optimized_tokens,
            'compression_ratio': optimized_tokens / original_tokens if original_tokens > 0 else 1,
            'messages': optimized_messages
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/count-tokens', methods=['POST'])
def count_tokens():
    """Count tokens in provided text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        token_count = optimizer.count_tokens(text)
        
        return jsonify({
            'status': 'success',
            'text_length': len(text),
            'token_count': token_count,
            'ratio': token_count / len(text) if len(text) > 0 else 0
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/stats')
def stats():
    """Get usage statistics"""
    return jsonify(optimizer.get_usage_stats())

@app.route('/track-usage', methods=['POST'])
def track_usage():
    """Track token usage"""
    try:
        data = request.get_json()
        model = data.get('model', 'unknown')
        input_tokens = data.get('input_tokens', 0)
        output_tokens = data.get('output_tokens', 0)
        
        optimizer.track_token_usage(model, input_tokens, output_tokens)
        
        return jsonify({'status': 'success', 'message': 'Usage tracked'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def run_scheduler():
    """Run scheduled monitoring tasks"""
    schedule.every(5).minutes.do(optimizer.monitor_context_efficiency)
    
    while True:
        schedule.run_pending()
        time.sleep(30)

if __name__ == '__main__':
    # Start scheduler in background
    scheduler_thread = Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    app.run(host='0.0.0.0', port=8524, debug=False)